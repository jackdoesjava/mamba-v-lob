# The interval domain

Two files carry the abstraction. `src/verification/intervals.py` is the domain, one function per
operation a block performs. `src/verification/certify.py` chains them in the order
`SelectiveSSMBlock.forward` runs and returns the boxes. The contract on every function in the first
file is the same: given boxes containing the inputs, return a box containing every output those
inputs can produce. Under-approximation is a bug, and `tests/test_verification.py` samples each
operation looking for one.

## The dataclass

`Interval` holds `lo` and `hi` tensors of the same shape, checked in `__post_init__`. That shape is
the quantity at a single timestep, with no batch axis and no time axis: a box on `u` is
`(d_inner,)`, a box on `Abar` is `(d_inner, d_state)`. A box claims something about every timestep
and every input at once, which is what lets a hundred-step recursion be bounded without unrolling
any input. The helpers are thin: `abs_max` returns `max(|lo|, |hi|)` and is what the summary tables
report, `contains` tests membership with `atol=1e-5`, and `__add__` and `__mul__` promote a plain
tensor to a degenerate box.

## Elementwise maps

`sigmoid`, `softplus` and `exp` defer to `monotone(box, fn)`, which returns
`Interval(fn(lo), fn(hi))`. Sound and exact for any increasing `fn`, and it assumes increasing: a
decreasing map still needs only its endpoints, but they have to be swapped, and `monotone` will not
do that for you.

SiLU is the primitive where endpoints are not enough. `silu(x) = x * sigmoid(x)` decreases on
`(-inf, x*]` and increases on `[x*, inf)`, with

```
x*       = -1.2784645427610738
silu(x*) = -0.27846454276107395
```

so the maximum over a box is always at an endpoint, and the minimum is too unless the box straddles
`x*`. `silu` computes `straddles = (lo <= SILU_ARGMIN) & (hi >= SILU_ARGMIN)` and selects
`SILU_MIN` there. Both constants are imported from `src/models/mamba.py` rather than restated, so
the model and the domain cannot drift apart.

## Products and affine maps

`__mul__` stacks the four corner products `{lo*lo, lo*hi, hi*lo, hi*hi}` and takes the elementwise
min and max. Exact for two independent intervals, and independent is the word doing the work.
`drive` is `g * B * u`, and `B` and `u` are both functions of the same input sequence, so the corner
rule prices a combination no input can realise. The recursion then discards that dependence `L` more
times; [07-soundness.md](07-soundness.md) prices the discard.

`affine` splits the weight as `w_pos = W.clamp(min=0)`, `w_neg = W.clamp(max=0)` and pairs each part
with the endpoint that extremises it: `lo = w_pos @ box.lo + w_neg @ box.hi`, and `hi` the other way
round. Each output coordinate is a linear functional over a box, its extreme sits at a vertex, and
the split picks that vertex coordinatewise, so the result is exact for the box it is given. For the
first Linear in each block `certify_block` skips it in favour of `layernorm_linear`, which keeps the
zero-sum and `l2` constraints a box throws away; see [05-layernorm-bound.md](05-layernorm-bound.md).

## The convolution and the zero padding

`depthwise_conv1d` raises unless `groups == in_channels`; `dense_conv1d` covers the channel-mixing
variant used in the conv ablation, and `certify_block` picks between them. Both apply the same
positive and negative split, per channel over the kernel axis, and both first call `hull_with_zero`,
which widens the box to contain `0`.

That hull is a soundness requirement. The block pads `d_conv - 1` zeros on the left and keeps the
first `L` outputs, so at positions `t < d_conv - 1` some taps read exactly `0`, and `0` need not lie
in the box on `u`. Without the hull the bound fails at those positions and only those, which
sampling from the middle of a sequence would never show; the conv test builds whole sequences and
compares per-channel extrema across all `L` positions.

## The timescale and its discretisation

`delta_box(block, pre)` pushes `block.delta_from_pre` through `monotone`. Both parametrisations are
increasing in the pre-activation: `softplus` is, and `dt_min * (dt_max/dt_min)^sigmoid(z)` is a
positive power of a constant above one composed with `sigmoid`. For the bounded arm the box is then
clamped to `[dt_min, dt_max]` a second time, mirroring the clamp inside `delta_from_pre`, so the two
agree bitwise whatever float32 did at the endpoints; [04-stability.md](04-stability.md) covers why
that clamp fires.

`discretisation_boxes(A, delta)` boxes the two quantities the recursion needs, with `Bbar = g * B`:

```
Abar(dt) = exp(dt * A)
g(dt)    = (exp(dt*A) - 1)/A = dt * phi(dt*A),   phi(v) = expm1(v)/v
```

`A` is strictly negative by construction and `dt` strictly positive, which fixes both
monotonicities and the sign of `g`:

```
d/d(dt) Abar = A * exp(dt*A) < 0     so Abar decreases in dt
d/d(dt) g    = exp(dt*A)     > 0     so g increases in dt, and g(0) = 0 gives g > 0

Abar in [exp(dt_hi * A), exp(dt_lo * A)]
g    in [dt_lo * phi(dt_lo * A), dt_hi * phi(dt_hi * A)]
```

Endpoints suffice for both, with the order reversed for `Abar`. `delta` arrives per channel and is
unsqueezed to broadcast over `d_state` against an `A` of shape `(d_inner, d_state)`. Positive `g`
keeps the sign of `Bbar` equal to the sign of `B`, and `Abar > 0` is what the unrolled state bound
relies on. None of this gives `sup Abar < 1`; that needs `dt_min > 0` and nothing else delivers it.

## Two bounds on the state

The recursion is `h_t = Abar_t * h_{t-1} + drive_t` with `h_{-1} = 0` and `drive = Bbar * u`.

`state_bound_geometric` takes `s = sup Abar` and `m = sup |drive|` and returns the symmetric box of
radius `m/(1 - s)`, or an infinite one when `s >= 1`. One detail for anyone editing it:
`torch.where` evaluates both lanes, so `(1 - s)` is clamped at `1e-30` to keep the discarded lane
finite, and the selected lane returns infinity explicitly.

`state_bound_unrolled` iterates `L` times from `lo = hi = 0`, matching the zero initial state in
`forward`. Because `Abar > 0` elementwise, the extremising endpoint of `Abar` depends only on the
sign of the current bound:

```
hi <- (Abar.hi * hi  if hi >= 0 else Abar.lo * hi) + drive.hi
lo <- (Abar.lo * lo  if lo >= 0 else Abar.hi * lo) + drive.lo
```

| bound | `sup\|h\|` at L=100 | independent of L | finite at `sup Abar = 1` |
| --- | --- | --- | --- |
| unrolled horizon | 1169.0 | no | yes |
| geometric fixed point | 1.235e4 | yes | no |

The horizon bound is tighter at every finite `L`, 10.6 times so at the length this model runs at,
and it is the only one that returns a number for the softplus arm. The fixed point is the statement
that an invariant set exists, and it is the one to quote for a model running on an unbounded stream.
`certify_block` computes both and carries each through the rest of the block, which is why the boxes
dict holds a `_geometric` and a `_horizon` copy of `h`, `y` and `out`.

## Composition

`certify_block` needs no data and assumes nothing about the block input:

```
layernorm_linear(norm, in_proj)        -> split -> u_pre, gate_pre
conv(u_pre) -> silu                    -> u
affine(u, x_proj)                      -> split -> dt_pre_raw, B, C
affine(dt_pre_raw, dt_proj) -> delta_box
discretisation_boxes(A, delta)         -> Abar, g
g * B -> Bbar,  Bbar * u -> drive
state bounds                           -> h_geometric, h_horizon
(h * C).sum(-1) + D * u                -> y
affine(y * silu(gate_pre), out_proj)   -> out
```

`tight_layernorm=False` replaces the first line with `affine(layernorm(norm), in_proj)`, which is
how the relaxation ablation is produced. The returned dict keeps tensors under `boxes` so
`looseness_report` can price them against `empirical_envelope`, and flattens the headline scalars
into `summary`.

## The block boxes stop before the residual

`forward` returns `out_proj(y * silu(gate)) + x`, and `certify_block` bounds `out_proj(...)` alone,
so `out_geometric` and `out_horizon` are pre-residual. The residual term is the block input, and
nothing in the block certificate bounds it: the LayerNorm bounds what the block reads, not what the
block was handed. Those two boxes bound a block's contribution to the residual stream, and summing
them across layers is not a bound on the network output.

The whole-pass bound is `certify_output_range`, which starts at `final_norm` and needs nothing from
the layers, because a LayerNorm re-bounds the stream whatever it grew to. It indexes `head[0]` as
the first Linear and `head[3]` as the second, with `head[1]` the SiLU and `head[2]` a Dropout that
is the identity at eval. Adding a layer to the head means fixing those indices.

## Extending it

Any new operation returns a superset of the true image and gets a sampling test alongside the
existing ones. Non-monotone maps need their interior extrema found first; evaluating at the
endpoints is the failure mode SiLU illustrates. Constants shared with the model belong in the model.
The domain is elementwise and carries no dependence between quantities, so a zonotope relaxation
would replace `__mul__` and `affine` and keep what those two discard.
