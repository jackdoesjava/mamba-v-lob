# What has to be bounded

`h` is the only tensor in a block that accumulates across timesteps. Every other
intermediate is a fixed-depth function of the current position and a few neighbours, so it
inherits a bound from the layer input directly. `h` does not, and a bound on it has to hold
after any number of steps.

## The block

A selective state space model computes, at every position, a timescale `delta` and an input
matrix `B` and output matrix `C`, all from the layer input, then discretises a diagonal,
strictly negative continuous-time `A` with that timescale. One block, as written in
`src/models/mamba.py`:

```
xn        = LayerNorm(x)                       x is (batch, L, d_model)
[u0,gate] = in_proj(xn)                        split into two (batch, L, d_inner) halves
u         = SiLU(DepthwiseCausalConv1d(u0))    position t reads u0[t-d_conv+1 .. t]
[p, B, C] = x_proj(u)                          split dt_rank, d_state, d_state
delta     = dt_min * (dt_max/dt_min)^sigmoid(dt_proj(p))
A         = -exp(A_log)                        strictly negative, (d_inner, d_state)
Abar[t]   = exp(delta[t] * A)
Bbar[t]   = A^-1 (exp(delta[t] * A) - I) B[t] = delta[t] * phi(delta[t] * A) * B[t]
h[t]      = Abar[t] * h[t-1] + Bbar[t] * u[t]        h[-1] = 0
y[t]      = sum_n C[t,n] * h[t,:,n] + D * u[t]
out       = out_proj(y * SiLU(gate)) + x
```

Products are elementwise and broadcast over the state axis, and `phi(v) = expm1(v)/v`
extended by `phi(0) = 1`. `A` is negative for every real `A_log`, which follows from the
parametrisation and not from training.

## Nothing in the recursion is time-invariant

Only `A_log` and `D` are parameters. `delta`, `B` and `C` are activations, recomputed from
the block's own input at every timestep, so the system is linear time-varying and no
argument may lean on time invariance. The common claim that Mamba is an LTI system and is
therefore boundable by a spectral radius is false. A block's state dict holds `A_log`, `D`
and the weights of `norm`, `in_proj`, `conv1d`, `x_proj`, `dt_proj` and `out_proj`. It
contains no `delta`, no `B` and no `C`, at any position. A verifier handed the checkpoint
and asked to reason about `h' = A h + B u` cannot fill in `B`, because the tensor does not
exist until an input arrives. What is fixed across inputs is the chain of weights that
produces those activations, and a bound has to hold uniformly over everything that chain
can emit.

## Why bounded verification struggles here

Existing verification for sequence models takes one of two routes. The first unrolls the
recurrence a fixed number of steps and propagates a bound through the resulting feedforward
network, which is what IBP and CROWN style methods do and what star set reachability does
for recurrent networks (Tran et al., HSCC 2023). The bound is tied to the length it was
computed for and grows with it. On layer 0 here the unrolled state bound
runs from `127.56` at `L = 10` to `1169.03` at `L = 100` and `1.226e4` at `L = 5000`, while
nothing about the network has changed.

The second infers an invariant with a verifier and then checks it, which is what Jacoby,
Barrett and Katz do for recurrent networks (ATVA 2020); Bonassi, Farina and Scattolini reach
a related invariant for gated recurrent units from the gate itself (Systems and Control
Letters, 2021). Length independence comes for free that way, and using invariants on neural
sequence models is prior art rather than anything new here. The cost is that the invariant
comes out of a search that can return unknown, and that the search returns a set of numbers
for one checkpoint rather than a formula in the architecture's own quantities.

The route taken here is to write the invariant down in closed form from the discretisation,
then discharge its premise with the model's own normalisation bounds. That is
[02-invariant.md](02-invariant.md).

## Symbols

| symbol | shape or value | what it is |
| --- | --- | --- |
| `d_model` | 64 | width of the residual stream |
| `d_state` | 16 | states per channel, indexed `n` |
| `expand` | 2 | inner width multiplier |
| `d_inner` | 128 | `expand * d_model`, the number of SSM channels |
| `d_conv` | 4 | width of the causal depthwise kernel |
| `dt_rank` | 4 | rank of the timescale projection, `ceil(d_model/16)` |
| `L` | 100 | lookback window, in ticks |
| `A_log` | `(d_inner, d_state)` | parameter, initialised S4D-Real |
| `A` | `(d_inner, d_state)` | `-exp(A_log)`; on the trained weights it spans `[-16.8437, -0.9651]` |
| `D` | `(d_inner,)` | parameter, the skip path `D * u` |
| `delta` | `(batch, L, d_inner)` | activation; one timescale per channel per step |
| `dt_min`, `dt_max` | `1e-3`, `1e-1` | the closed range `delta` is confined to |
| `B` | `(batch, L, d_state)` | activation, continuous-time input matrix, shared across channels |
| `C` | `(batch, L, d_state)` | activation, output matrix |
| `Abar` | `(batch, L, d_inner, d_state)` | `exp(delta * A)`, in `(0, 1)` entrywise |
| `Bbar` | `(batch, L, d_inner, d_state)` | the exact zero-order hold input matrix |
| `h` | `(batch, d_inner, d_state)` per step | the state, from `h[-1] = 0` |
| `u` | `(batch, L, d_inner)` | SSM branch input, after the convolution and SiLU |
| `gate` | `(batch, L, d_inner)` | second half of `in_proj`, entering as `SiLU(gate)` |
| `y` | `(batch, L, d_inner)` | SSM output before the gate |
| `phi` | elementwise | `expm1(v)/v`, extended by `phi(0) = 1` |
| `w_lo`, `w_hi` | `(43,)` each | winsorisation limits, the 0.1% and 99.9% train-split quantiles, applied at inference |

Winsorisation runs at inference and not only during training, so the standardised input box
`[-6.60, 18.87]` is enforced rather than observed. A bar marks a discretisation and not a
normalisation: `A` and `B` name the continuous-time system `h' = A h + B u`, and `Abar` and
`Bbar` name its exact zero-order hold over one step of length `delta`.
