# Discretisation

The recursion inside a Mamba block is a discretisation of a continuous-time linear system.
That system is what a certificate talks about, so the two have to agree. Reference Mamba
uses a pairing that agrees only in a regime this network never operates in, and the size of
the disagreement is a function of the timescale range alone.

Notation: `A` is diagonal, strictly negative, of shape `(d_inner, d_state)`; `dt` is the
per-channel timescale, written `delta` in the code; `u` is the block's scalar input per
channel, held constant across the step.

## The exact step

Proposition 1. For diagonal `A < 0` and `dt > 0`, the zero-order hold discretisation of
`h' = A h + B u` over one step of length `dt` is

```
A_bar = exp(dt * A)
B_bar = A^-1 (exp(dt * A) - I) B = dt * phi(dt * A) * B,     phi(v) = expm1(v)/v, phi(0) = 1.
```

Proof. Hold `u` constant at `u_t` over `[0, dt]`. Variation of constants gives

```
h(dt) = exp(dt * A) h(0) + (integral_0^dt exp((dt - s) A) ds) B u_t,
```

and the integral evaluates in closed form because `A` is diagonal. For a single diagonal
entry `a`,

```
integral_0^dt exp((dt - s) a) ds = (exp(dt * a) - 1)/a,
```

which is the `A^-1 (exp(dt A) - I)` factor read elementwise. The division is safe for every
parameter value because `a = -exp(A_log)` is strictly negative. Substituting `v = dt * a`
turns it into `dt * (exp(v) - 1)/v = dt * phi(v)`. QED

Two consequences the interval code relies on. For `a < 0` the gain `g(dt) = dt * phi(dt*a)`
is positive and increasing in `dt`, since `g'(dt) = exp(dt * a) > 0`, so evaluating at the
endpoints of a `dt` interval gives the box exactly rather than conservatively. And
`A_bar = exp(dt * a)` is decreasing in `dt`, so its box comes from the endpoints swapped.
Both are in `zoh_boxes()` in `src/verification/intervals.py`.

## What the reference does instead

Reference Mamba discretises `A` by the exact formula above and `B` by forward Euler,
`B_bar = dt * B`. That is `phi` replaced by `1`. The two halves are consistent only when
`|dt * A|` is small, and the relative error is exact rather than approximate:

```
|dt*B - B_bar| / |B_bar| = |dt - dt*phi(v)| / |dt*phi(v)| = |1/phi(v) - 1|,   v = dt * A.
```

It vanishes as `v -> 0` and grows without bound as `v -> -inf`. For `v < 0` the function
`phi(v)` lies in `(0, 1)` and behaves like `1/|v|` for large `|v|`, so the error grows
roughly linearly in `|dt * A|` once that product leaves the neighbourhood of zero. Euler
always overstates the input gain, never understates it.

## What it would have cost here

Measured over the certified `dt` range and the learned `A`, by
`ablate_discretisation()` in `scripts/06_certify.py`:

| `dt` | max `\|dt * A\|` | median rel. error | max rel. error |
| --- | --- | --- | --- |
| unconstrained softplus, median `dt` about 0.63 | ~31 | 411% | 2266% |
| bounded to `[1e-3, 1e-1]` | 1.68 | 17% | 107% |

The bounded row comes from `dt_max = 0.1` against `min A = -16.84`, giving
`max |dt * A| = 1.68`; the learned `A` spans `[-16.84, -0.965]`. The 99th percentile of the
relative error on that row is 88%. So even after bounding the timescale, `|dt * A|` is not
small and the Euler pairing is still wrong by tens of percent across most of the range.

`src/models/mamba.py` uses the exact form, so the error in the shipped model is zero by
construction. The table prices the shortcut rather than reporting a defect. It also shows a
second-order effect of the timescale bound argued for in [04-stability.md](04-stability.md):
the discretisation error is a property of the `dt` range, so constraining `dt` would have
bounded the inconsistency too, even had the Euler form been kept.

`test_euler_shortcut_is_materially_different` in `tests/test_ssm_math.py` asserts the gap is
above 1% on a freshly initialised block, so a silent regression to `B_bar = dt * B` fails
the suite rather than passing quietly.

## Why a verifier cares

A verifier handed `(A, B, C, dt)` and told the system is `h' = A h + B u` will reason about
that system. It will discretise it itself, or reason about the flow directly, and either way
it uses the exact relation between `A_bar` and `B_bar`. If the network's recurrence is not a
consistent discretisation of the stated continuous system, every conclusion the verifier
reaches is about a different object, and no amount of soundness in the abstract domain
repairs that. The mismatch is not conservative in a helpful direction either: an Euler
`B_bar` is larger than the true one, so a bound proved for the exact system understates the
state of the network that actually runs.

This is why `dynamics_spec` in the export states the discretisation as an equation rather
than naming it, and why `scripts/04_export_bounds.py` re-runs the recurrence in NumPy from
the exported numbers alone and refuses to write if it disagrees with PyTorch.

## The limit at the origin

`phi(v) = expm1(v)/v` is `0/0` at `v = 0`, with limit `1`. Evaluating the quotient directly
near zero loses precision and its gradient is `nan` at exactly zero, which reaches the
optimiser through `dt`. `zoh_phi` in `src/models/mamba.py` switches below `|v| = 1e-4` to
the truncated series

```
phi(v) = 1 + v/2 + v^2/6 + O(v^3),
```

selected with `torch.where` on a mask, with the divisor replaced by ones on the small branch
so the discarded lane never divides by zero. At the switch point the two branches agree to
better than float32 resolution. Three tests cover it: the quotient definition away from
zero over `v` in `[-30, 5]`, continuity and finiteness at and near zero, and a finite
gradient at exactly zero.

The `1e-4` threshold is reachable in normal operation. With `dt_min = 1e-3` and the smallest
learned `|A|` near `0.965`, the smallest `|dt * A|` is around `9.7e-4`, an order of magnitude
above the switch, but nothing in the parametrisation prevents `A_log` from drifting lower
during training, and a randomly initialised block can sit either side of it.

## See also

- [02-model.md](02-model.md) for where `B_bar` sits in the block, and the other two
  departures from the reference.
- [04-stability.md](04-stability.md) for the timescale bound, which this page treats as
  given.
- [`certification/ablation.md`](certification/ablation.md) section 5, the generated numbers.
