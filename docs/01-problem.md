# The problem

A selective state space model such as Mamba carries one hidden state `h` along its input. Every
other quantity in a block depends only on the current position and a few neighbours, so it
inherits a bound from the block input directly. The state builds up over time, so a bound on it,
or on how far a change of input can move it, has to hold after any number of steps. On a stream
such as limit order book updates, a deployed model carries its state far beyond the windows it
was trained on.

## The block

At every position the model computes a timescale `delta`, an input matrix `B` and an output
matrix `C` from the block input. It then discretises a diagonal, strictly negative matrix `A`
with that timescale. One block, as written in `src/models/mamba.py`:

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

The listing is the bounded arm. The softplus arm computes `delta = softplus(dt_proj(p))`
instead. Products are elementwise and broadcast over the state axis, and `phi(v) = expm1(v)/v`
with `phi(0) = 1`. `A = -exp(A_log)` is negative for every real `A_log`, so this holds for every
checkpoint, trained or not.

## What is fixed and what is not

Within the scan and the readout, only `A_log` and `D` are parameters. The timescale `delta` and
the matrices `B` and `C` are activations: the block recomputes them from its own input at every
step. The recursion is therefore linear and time-varying, and an argument that treats `delta`,
`B` or `C` as fixed proves something about a different system. Mamba is sometimes described as
linear and time-invariant, with a bound from a spectral radius. That description does not fit the
selective scan.

A block's state dict holds `A_log`, `D` and the weights of `norm`, `in_proj`, `conv1d`, `x_proj`,
`dt_proj` and `out_proj`. It holds no `delta`, `B` or `C`, because those tensors exist only once
an input arrives. What is fixed across inputs is the chain of weights that produces them, so a
bound has to hold for everything that chain can produce.

Time invariance survives only in the accumulated time `tau_t = sum_{s<t} delta_s`. In that time
each cell solves `h' = A h + w` with `w` piecewise constant. Cao et al. (ICML 2026, Lemma 3.1)
analyse the block this way, and Appendix B of the paper uses the same view to derive the cell
theorem, for a common start, from their lemmas.

## The question the paper answers

The paper asks how far a small change of input can move the state after any number of steps. The
closest bound we know of is Theorem 3.3(ii) of Cao et al. It measures the change of the timescale
additively, as `|delta - delta~|`, and it grows linearly with the position. Their Remark 3.4 caps
the growth under a floor, a positive lower bound on the timescale, but the cap grows like
`1/floor` as the floor shrinks. Mamba's softplus timescale has a floor only when its
pre-activation is bounded below, and the floor we certify on one of our trained models is about
`1e-21`.

The paper measures the change of the timescale on a log scale instead, and proves a bound that
holds at every length with no floor. [02-invariant.md](02-invariant.md) goes through it.

## Why existing verification methods struggle

Verifiers for sequence models take one of two routes. The first unrolls the recurrence for a
fixed number of steps and propagates a bound through the resulting feedforward network. IBP and
CROWN-style methods do this, and so does star-set reachability for recurrent networks (Tran et
al., HSCC 2023). The bound is tied to the length it was computed for and grows with it. On layer
0 of our model the unrolled state bound grows from `168.84` at `L = 10` to `1514.25` at
`L = 100` and `1.665e4` at `L = 5000`, although nothing about the network changes.

The second route searches for an invariant with a verifier and then checks it. Jacoby, Barrett
and Katz do this for recurrent networks (ATVA 2020), and Bonassi, Farina and Scattolini reach a
related invariant for gated recurrent units from the gate (Systems and Control Letters, 2021). An
invariant holds at every length, but the search can fail, and it returns numbers for one
checkpoint rather than a formula.

A third route needs no search. Horbatko (arXiv:2604.18580, Lemma 4.4) writes the zero-order hold
update as a convex combination and proves the invariant by induction, and the same per-cell
radius follows from Cao et al.'s Theorem 3.3(i). The paper builds on this argument and applies
it to the difference of two runs.

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
| `A` | `(d_inner, d_state)` | `-exp(A_log)`; on the trained weights it spans `[-17.1351, -0.9121]` |
| `D` | `(d_inner,)` | parameter, the skip path `D * u` |
| `delta` | `(batch, L, d_inner)` | activation; one timescale per channel per step |
| `dt_min`, `dt_max` | `1e-3`, `1e-1` | the closed range `delta` is confined to on the bounded arm |
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

Winsorisation runs at inference as well as in training, so the standardised input box
`[-22.48, 19.25]` is enforced rather than observed. A bar marks a discretisation: `A` and `B`
name the continuous-time system `h' = A h + B u`, and `Abar` and `Bbar` name its exact zero-order
hold over one step of length `delta`.

The released `state-spaces/mamba-130m-hf` differs from this block in three ways: RMSNorm instead
of LayerNorm, the exponential-Euler input term `dt B u` instead of the zero-order hold, and an
unclamped softplus timescale. It has 24 layers with `d_model = 768`, `d_inner = 1536`,
`d_state = 16` and `dt_rank = 48`. The paper does not treat it. The repository computes
data-free and local bounds for its float64 forward pass, and [03-certificate.md](03-certificate.md)
describes what changes in the argument.
