# The model

`src/models/mamba.py` holds the selective state space model. `src/models/lstm.py` and
`src/models/transformer.py` hold the two baselines. The SSM is written so that its
reachable set can be bounded; the baselines exist to show that writing it that way costs
nothing in predictive accuracy.

## One block

`SelectiveSSMBlock` is a Mamba block in the sense of Gu and Dao (2023), with an input `x`
of shape `(batch, L, d_model)` and `d_inner = expand * d_model`:

```
xn         = LayerNorm(x)
[u0, g]    = in_proj(xn)
u          = SiLU(causal_depthwise_conv(u0))
[p, B, C]  = x_proj(u)
delta      = delta_from_pre(dt_proj(p))
A          = -exp(A_log)                            strictly negative, (d_inner, d_state)
A_bar[t]   = exp(delta[t] * A)
B_bar[t]   = delta[t] * phi(delta[t] * A) * B[t],   phi(v) = expm1(v)/v, phi(0) = 1
h[t]       = A_bar[t] * h[t-1] + B_bar[t] * u[t],   h starts at 0
y[t]       = sum_n C[t,n] * h[t,:,n] + D * u[t]
out        = out_proj(y * SiLU(g)) + x
```

Every product is elementwise, broadcast over the state axis. `A` is initialised S4D-Real,
`A[d, n] = -(n + 1)` shared across channels, so the decay rates start spread over `1`
to `d_state`.

Only `A_log` and `D` are parameters of the recursion. `delta`, `B` and `C` are produced
from the layer's own input at every timestep by `x_proj` and `dt_proj`, which makes the
block linear time-varying. A state dict therefore does not determine the dynamics, and
nothing in it can be read off as a transition matrix. What the export ships instead is the
chain of weights that produces those activations, together with intervals they provably
lie in.

## Three departures from the reference

The reference discretises `A` exactly and `B` by forward Euler, a pairing that is
consistent only when `|delta*A|` is small. This implementation uses the closed-form
zero-order hold input matrix, available in closed form because `A` is diagonal. The helper
`zoh_phi` carries a Taylor branch below `|v| = 1e-4` so the `0/0` limit stays finite and
differentiable. Measured `|delta*A|` reaches `1.68` here, which is not small, and the
reasoning is in [03-discretisation.md](03-discretisation.md).

The reference sets `delta = softplus(z)`, whose infimum is zero. Since `A_bar = exp(delta*A)`,
that gives `sup A_bar = 1` and no uniform contraction factor. This implementation
parametrises

```
delta = dt_min * (dt_max/dt_min)^sigmoid(pre),   then clamped to [dt_min, dt_max]
```

so the timescale lies in a closed positive interval for every weight and every input.
Setting `dt_parametrisation: "softplus"` restores reference behaviour for the ablation arm.
Why the lower bound is the whole result is in [04-stability.md](04-stability.md).

The convolution is depthwise, `groups = d_inner`, as in the reference. Causality comes from
padding `d_conv - 1` on both sides and keeping the first `L` outputs, so position `t` reads
only `[t - d_conv + 1, t]`. A channel-mixing convolution would inflate the interval bounds
by roughly the channel count for no modelling gain:

| conv | groups | conv params | max L1 gain | u bound | h bound |
| --- | --- | --- | --- | --- | --- |
| depthwise | 128 | 512 | 1.6581 | 8.4267 | 1169.0270 |
| dense | 1 | 65536 | 11.9768 | 56.2254 | 8.099e+04 |

## Why delta_from_pre clamps

In exact arithmetic the exponential already lands strictly inside `(dt_min, dt_max)`, so
the clamp looks redundant. It is not, because float32 `sigmoid` returns exactly `1.0` for a
pre-activation above roughly `16.64` and exactly `0.0` below roughly `-88.72`, and
`dt_min * (dt_max/dt_min)` need not round back to exactly `dt_max`. The clamp is what makes
the closed interval hold bitwise, and the certificate assumes the closed interval. The
upper tail is reached during training, so this is a live case rather than a hypothetical
one. `test_ssm_math.py` checks that `delta` attains both endpoints.

## The scan

The recursion is an unfused Python loop over `t`. A fused kernel would be faster and would
hide every intermediate inside it. The loop keeps `delta`, `B`, `C`, `A_bar`, `B_bar` and
`h` as addressable tensors, which is what `forward(trace=True)` returns in an `SSMTrace`
and what `scripts/04_export_bounds.py` audits against the exported description.
`detach_trace=True` is the default so an `L`-step graph is not held; only
`scripts/07_attack_bounds.py` needs the attached version. The cost is throughput, and the
purchase is being able to check that the equations shipped are the equations that ran.

The scan runs in float32. Under fp16 autocast `exp(delta*A)` flushes to exactly zero at the
top of the `delta` range, which deletes transitions the certificate assumes exist, so
`training.amp` is off by default.

## LOBMamba

`LOBMamba` projects the `input_dim` features to `d_model` through
`Linear -> SiLU -> LayerNorm`, stacks `num_layers` blocks, applies a final LayerNorm, and
reads position `-1` through `Linear -> SiLU -> Dropout -> Linear`.

That final LayerNorm is what makes the output range certifiable. A LayerNorm output obeys
`|LN(x)_i - beta_i| <= |gamma_i| * sqrt(d - 1)` for every `x` in `R^d`, so the head's input
sits in a weight-determined box no matter how far the residual stream has drifted. The head
is then an affine map, a SiLU bounded on any interval, and a second affine map, which gives
a fixed output interval with no premise on the input at all. On the trained network that
interval is `[-5.12, 4.34]` in units of the standardised target. Proofs are in
[05-layernorm-bound.md](05-layernorm-bound.md).

`architecture()` returns what was actually instantiated and goes into the export manifest.
Defaults come from `config.yaml` under `model.mamba`:

| field | value |
| --- | --- |
| d_model | 64 |
| d_state | 16 |
| d_conv | 4 |
| expand | 2 |
| num_layers | 2 |
| dt_rank | null, meaning ceil(d_model/16) as in the reference |
| dt_min | 0.001 |
| dt_max | 0.1 |
| dt_parametrisation | bounded |
| head_dropout | 0.2 |

## Transformer baseline

A pre-norm encoder with sinusoidal positional encoding, GELU feedforward, and the same
head shape as the SSM, reading position `-1`. `causal: true` installs the autoregressive
attention mask.

The flag changes nothing about leakage. Every position in the lookback window precedes the
label, which sits a full prediction horizon past the end of the window, so an unmasked
encoder does not see the future. What it changes is whether the baseline is described
correctly. An unmasked encoder is bidirectional, and calling one a causal transformer, or
using it to argue about autoregressive decoding cost, misstates what was compared.
`test_transformer.py` asserts that the masked variant is causal and that the unmasked one
is not.

## LSTM baseline

Two layers, orthogonal recurrent initialisation, Xavier input weights, forget-gate bias set
to `1.0`, reading position `-1` through a single linear layer. Torch packs the gate biases
as `i, f, g, o`, so the forget slice is the second quarter. This is the conventional
sequence baseline and receives no certification treatment.
