# Stability

The recursion inside a block is

```
h_t = Abar_t * h_{t-1} + Bbar_t * u_t,     h_{-1} = 0,
```

with `Abar_t = exp(delta_t A)` and `A = -exp(A_log)`, diagonal per channel. Only `A` and `D`
are parameters. `delta`, `B` and `C` are activations produced from the block input at every
timestep, so the system is linear time-varying and nothing below may lean on time
invariance. The bounds come from the parametrisation constraining those activations
uniformly over all inputs.

## The transition is always inside the unit interval

**Proposition (structural).** With `A = -exp(A_log)` and `delta > 0`,

```
Abar = exp(delta A) in (0, 1)   elementwise, for every parameter value and every input.
```

*Proof.* `exp(A_log) > 0` for every real `A_log`, so `A < 0` strictly. With `delta > 0` this
gives `delta A < 0`, and `exp` maps the negative reals onto `(0, 1)`. ∎

No training and no data enter the argument. This is the part of the folklore stability claim
that survives. It gives strict decay at each entry and says nothing about a rate: the
supremum of `Abar` over parameters and inputs can still be `1`, and a bound on the state
needs a rate.

## A uniform contraction factor requires delta bounded below

**Proposition (contraction).** Suppose additionally `delta` lies in `[dt_min, dt_max]` with
`dt_min > 0`. Put `s = exp(-dt_min * min|A|)`. Then `s < 1`, and

```
|h_t| <= M * sum_{k <= t} s^k   for all t,        |h_inf| <= M / (1 - s),
```

where `M = sup |Bbar * u|`.

*Proof.* For `a < 0` the map `delta -> exp(delta a)` is decreasing, so `delta >= dt_min`
gives `Abar_t <= exp(-dt_min * min|A|) = s`, uniformly in `t`. Taking absolute values in the
recursion, `|h_t| <= s |h_{t-1}| + M`. With `h_{-1} = 0`, induction gives the partial
geometric sum; letting `t -> infinity` gives the fixed point. ∎

`M` comes from the LayerNorm box and the fused `Linear o LayerNorm` bound, which are
input-independent, so the whole certificate is. On the trained weights `min|A| = 0.9651` and
`dt_min = 1e-3`, giving `s = 0.9990` and `1/(1 - s) = 1036.6`.

## Under softplus there is no invariant set

**Proposition (negative result).** Under the reference parametrisation `delta = softplus(z)`,

```
inf_{z in R} delta = 0,   hence   sup Abar = 1,
```

so `sum_k s^k` diverges and no finite invariant set exists for the state.

*Proof.* `softplus(z) -> 0` as `z -> -infinity`, and `exp(delta A) -> 1` as `delta -> 0+`. ∎

The supremum is attained in the limit, so the divergence is a property of the module and not
looseness in the abstract domain. There is no tighter analysis of reference Mamba that
recovers a contraction factor, because none exists to recover.

## Two claims that have to be kept apart

Structurally, as a property of the module in isolation and over arbitrary inputs to the
block, softplus admits no contraction factor at all, while bounded `delta` gives `s < 1`
unconditionally. That is what `SelectiveSSMBlock.contraction_certificate()` reports.

In context, a block is never fed an arbitrary input. It sits behind a LayerNorm, so the `dt`
pre-activation is confined to a finite box and `inf delta = softplus(pre_lo) > 0` strictly.
A finite geometric bound therefore does exist under softplus, with contraction factor
`1 - O(softplus(pre_lo))`, which leaves it about five orders of magnitude weaker than the
bounded arm. That is what `certify_block()` computes.

The claim being made is not "softplus is unbounded and bounded `delta` is bounded". Both
arms are bounded once the LayerNorm is taken into account. The claim is that softplus
carries no structural guarantee, and that what it leaves in context is a bound too weak to
certify anything useful. Constraining `delta >= dt_min > 0`, one line of the
parametrisation, restores a usable one.

## The ablation

Sequence length `L = 100`. Every entry is a property of the module, computed with no
reference to any dataset.

| `delta` | `inf delta` | `sup Abar` | `1/(1-s)` | geometric `sup\|h\|` | `sup\|h\|` at L=100 | contractive |
| --- | --- | --- | --- | --- | --- | --- |
| bounded | 1e-3 | 0.9990 | 1036.6 | 1.24e4 | 1.17e3 | yes |
| softplus | 0 | 1.0000 | infinite | infinite | 1.14e4 | no |

The softplus arm still has a finite number in the last column because the finite-horizon
form stays finite at `s = 1`, where it degenerates to `L` terms of size `M`. That number is
roughly ten times the bounded arm's, and it grows without limit in `L`, whereas the bounded
arm approaches `1.24e4` and stops.

## Finite horizon against the fixed point

The partial sum is kept separately from its limit. It is tighter at the lengths
actually used, and it is the only one of the two that survives `s = 1`.

| L | `sum_{k<L} s^k` | `1/(1-s)` | certified `sup\|h\|` at L | geometric `sup\|h\|` |
| --- | --- | --- | --- | --- |
| 10 | 9.96 | 1036.6 | 127.6 | 1.24e4 |
| 100 | 95.37 | 1036.6 | 1.17e3 | 1.24e4 |
| 1000 | 641.7 | 1036.6 | 7.78e3 | 1.24e4 |

At `L = 100`, the length this model runs at, the finite-horizon bound is 10.6 times tighter
than the fixed point.

## The parametrisation and the clamp

```
delta = dt_min * (dt_max/dt_min)^sigmoid(z),   clamped to [dt_min, dt_max]
```

with `dt_min = 1e-3` and `dt_max = 1e-1`. The map is monotone increasing in `z`, so a
verifier bounds `delta` from the endpoints of an interval on `z` and gets an exact box; the
measured looseness on `delta` is 1.07 to 1.10 times, which is what makes the contraction
proposition usable rather than only true.

In exact arithmetic `sigmoid` maps onto the open interval `(0, 1)`, `delta` lands strictly
inside `(dt_min, dt_max)`, and the clamp never fires. Float32 does not behave that way.
`sigmoid` returns exactly `1.0` above `z` around `16.6` and exactly `0.0` below `z` around
`-88.7`, and `dt_min * (dt_max/dt_min)` need not round back to `dt_max`. The clamp is what
makes the closed interval `[dt_min, dt_max]` hold bitwise. The certificate assumes the
closed interval, and the abstract domain clamps its `delta` box the same way, so the two
agree on every input. The upper tail is reached during training, so the case is a real one.

The parametrisation itself is described in [02-model.md](02-model.md), the bound on `M` in
[05-layernorm-bound.md](05-layernorm-bound.md), and the generated numbers in
[certification/ablation.md](certification/ablation.md).
