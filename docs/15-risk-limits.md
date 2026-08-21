# The certified bound as a pre-trade risk limit

A machine learning signal going into a trading system is capped before it reaches the book.
Position size is some function of signal strength, and that function has a ceiling. The
ceiling is calibrated: someone runs the backtest, reads the historical distribution of the
signal, picks a level the strategy did not exceed in sample, and adds a margin. Behind it sit
runtime monitoring on the realised signal and a kill switch.

Every part of that is a statement about data that has been seen. The backtest is a sample.
The historical distribution is a sample. The monitor fires after the fact, on an order that
has already gone out. None of it constrains what the model does on data it has not seen,
which is the case the limit exists for: a corrupted or stale feed, a venue outage that empties
one side of the book, quoting designed to move the signal, a regime with no analogue in the
sample.

## What the weights alone give

`scripts/06_certify.py` reads the checkpoint and produces, offline, in seconds, with no
dataset loaded:

```
certified output range, standardised target units : [-5.1242, 4.3374], width 9.4616
one standardised unit                             : 0.7042 basis points of the 100-tick
                                                    forward return
certified output range, basis points              : [-3.600, +3.063]
```

The model cannot forecast a 100-tick move outside about `-3.6` to `+3.1` basis points. Not
"has not been observed to". Cannot, for every input in `R^{100 x 43}`, including inputs that
are not order books at all.

One caveat on the units. The standardised interval is a property of the weights. The
conversion to basis points uses the training split's target standard deviation, `7.042e-5` in
log return, so the basis point figure carries the scale of the day the model was fitted on.
The standardised interval itself does not move.

## Why the range is finite

`LOBMamba` applies a LayerNorm after the last block and before the head. For a LayerNorm of
normalised shape `d`, every coordinate of the output satisfies
`|LN(x)_i - beta_i| <= |gamma_i| * sqrt(d - 1)` for every `x` in `R^d`, with no premise on
`x` at all. The head's input therefore sits in a box determined by the gain and the bias.

The head is `Linear -> SiLU -> Linear`. An affine map of a box is bounded, `SiLU` is bounded
on any bounded interval, and a second affine map preserves that. So the head has a fixed
image, and the residual stream can drift as far as it likes without changing it. The proof,
and the fused `Linear o LayerNorm` bound that makes the interval exact rather than merely
sound, are in [05-layernorm-bound.md](05-layernorm-bound.md).

This is the one bound in the repository that needs no premise. The state bound in
[04-stability.md](04-stability.md) needs `delta` bounded below; the output range does not.

## What it composes with

Take any sizing rule that is monotone in the signal with a known gain. A proved cap on the
signal then gives a proved cap on the position, and hence on the notional exposed to a single
model output, with no reference to a backtest. What was a monitored property becomes one
checkable at build time, from the artefact, before the model is deployed.

Two limits on that, both real.

It bounds the signal, not the loss. A forecast confined to `[-3.600, +3.063]` basis points
can be wrong on every window, and a position sized from a bounded signal can lose without
limit if the signal has the wrong sign often enough.

It bounds one forward pass. A sizing rule that reads only the current signal inherits the cap
directly. A rule that accumulates, averages or integrates past signals needs its own argument,
and the bound gives that argument a per-output ceiling to start from rather than its
conclusion. This repository states no sizing rule, so it does not carry the composition
through.

## What it costs

Three constraints in [02-model.md](02-model.md) are what make the bounds finite and tight
enough to use, and none of them is unusual. `delta` is confined to `[1e-3, 1e-1]` instead of
`softplus`. The input matrix uses the exact zero-order hold rather than the forward Euler
shortcut. The convolution stays depthwise, which is what reference Mamba already does.

[certification/ablation.md](certification/ablation.md) prices them against the certified
state bound at `L = 100`, which is `1169.0` as shipped.

| change | certified `sup \|h\|` | factor |
| --- | --- | --- |
| as shipped | 1.17e3 | 1x |
| `delta = softplus(z)` | infinite | no invariant set exists |
| dense convolution | 8.099e4 | 69x |
| LayerNorm bounded coordinatewise, not fused | 5.708e4 | 49x |
| geometric fixed point, not finite horizon | 1.235e4 | 10.6x |

The first of those is the structural one: under `softplus`, `inf delta = 0`, so
`sup Abar = 1` and no finite invariant set exists for the state. The last two are choices in
the analysis rather than in the model, and they are free.

The zero-order hold is a correctness constraint rather than a tightening one. Euler and the
exact form disagree by 17% median relative error over this network's `delta` range, where
`max |delta*A| = 1.68`, so a certificate proved about the Euler recurrence would be a
certificate about a different model. See [03-discretisation.md](03-discretisation.md).

None of the three costs accuracy. The Mamba, transformer and LSTM test rank ICs cannot be
separated on this sample, so nothing here suggests the constrained model pays for its bounds
in predictive performance.

## How tight the limit is

`scripts/07_attack_bounds.py` runs Adam on the input to push the output past its certified
ends. On the trained model with unrestricted input it reaches `0.3593` against a certified
`4.3374`, and `-0.9234` against a certified `-5.1242`, so the attained range is `1.2827` wide
against a certified `9.4616`. Since `attacked <= true reachable <= certified`, the limit is
`7.4` times wider than anything the search could produce, and the true factor is at most that.

A limit seven times too conservative is still a limit: it caps the signal at `3.6` basis
points where an attacker was only able to reach about `0.65`. A limit `1e4` times too
conservative is not, and the repository has rows like that. The certified `sup |y|` inside a
block sits `2.5e4` times above what the attack reaches and the state row `2.3e3` times, both
because interval arithmetic compounds across the 100-step recursion. The output row avoids
that entirely, because `final_norm` re-bounds the residual stream before the head reads it,
which is why the output row is the one a pre-trade limit uses.
[07-soundness.md](07-soundness.md) has the full table.

## What this does not establish

The certification and the trading claim have to be read at different strengths, and the
second is much the weaker.

The signal is not shown to be worth trading. Test rank IC for the Mamba is `+0.1413` with a
block bootstrap standard error of `0.0819` and a 95% interval of `[-0.0247, +0.2904]`, which
includes zero. Overlapping labels give 2560 test observations an effective size of 25.6
independent windows, and Diebold-Mariano cannot separate the three sequence models. See
[09-statistics.md](09-statistics.md). All of the data is one instrument on one session, GME on
2021-01-28. The certificate does not depend on any of that, since it never touches the data,
but nothing here says the forecast is good.

A bounded signal is not a bounded loss. The certificate says how large the output can be. It
says nothing about whether the output is correct, and nothing in the repository bounds
`|f(x) - f(x')|` for nearby inputs, so it is not a robustness certificate either.

The bound covers a single forward pass through the network. Position accumulation over time,
execution, queue position, fill uncertainty and market impact are all outside it. A firm that
wanted an end-to-end statement about notional at risk would supply the sizing and execution
model and prove the rest itself.

No external verifier has consumed the artefact. The export ships the recurrence in
`dynamics_spec` and every weight in the generator chain so that a formal tool could take it,
and a NumPy round trip checks the description matches the running model to `8.8e-7`. Bounds
computed by this repository and checked by this repository's tests are a weaker position than
an independent check. See [11-handoff.md](11-handoff.md).

The novelty of the approach is unchecked. No literature review has been done, so the argument
about the `softplus` infimum and its repair should be read as new to the author rather than
new.

More of this is in [12-limitations.md](12-limitations.md), stated at full strength.

## What a firm would actually get

A model class whose worst-case output is provable offline, from the weights, before anything
is deployed; three architectural constraints that buy that and cost nothing measurable in
accuracy; and a pipeline that emits the proof as an artefact next to the checkpoint. The value
is in removing one class of tail risk from the deployment argument, so that the cap on a
signal stops being a calibrated guess about unseen data and becomes a fact about the model.
Whether the underlying signal earns money is a separate question, and this work does not
answer it.
