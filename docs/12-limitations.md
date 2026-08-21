# Limitations

The certificate covers reachable state and reachable output. It does not cover sensitivity to
perturbation, it does not cover loss, and the numbers it produces are nowhere near tight. The
predictive side of the paper rests on one instrument on one day and cannot separate the three
models it compares. Each of those is set out below at full strength, because the paper should
state them the same way.

## The bounds are loose

Gradient ascent on the input prices every bound. Since `attacked <= true reachable <=
certified`, the ratio between the last two columns is an upper bound on the abstraction's
slack, and on the state and output rows that slack is large.

| quantity | attacked | certified | looseness |
| --- | --- | --- | --- |
| `sup delta` | 0.0932 | 0.1000 | 1.07x |
| `inf delta` | 1.10e-3 | 1.00e-3 | 1.10x |
| `sup \|u\|` | 4.53 | 7.63 | 1.7x |
| `sup \|B\|` | 1.26 | 16.4 | 13x |
| `sup \|h\|` | 0.54 | 2.68e2 | 498x |
| `sup \|y\|` | 4.51 | 4.18e4 | 9.3e3x |

Against the realised envelope rather than the attack, the gap is wider again. The certified
`sup |h|` at `L = 100` is `1.17e3` where held-out data reaches `0.23`, a factor of `5.2e3`,
and the certified `sup |y|` sits `5.8e4` times above its realised maximum. Part of that is
the abstraction and part of it is that ordinary market data goes nowhere near the worst
case, which is why the attacked column is the one that prices the domain.

Those are layer 0 of two from the recorded run. The cause is known and is not a constant that
a sharper inequality would remove. The interval domain holds a box for `h_{t-1}` and a box for
`u_t` and combines them as though both could take their worst values at once. They cannot;
both are functions of the same input sequence. That correlation is discarded once per step,
one hundred times, and the factors of 500 and 1e4 are what the discarding compounds to. A
zonotope or CROWN-style relaxation would track linear dependence through the recursion and
move these rows without changing a single proposition. It has not been implemented here. See
[07-soundness.md](07-soundness.md).

The `delta` rows are the exception, and they are the rows Proposition 5 depends on. Both
endpoints are attainable to within 10%, so the contraction factor is read off a value the
input can actually produce.

## Not a robustness certificate

Nothing in this repository bounds `|f(x) - f(x')|` for nearby `x` and `x'`. Every result is a
statement about the reachable set over all inputs, which says how large the output can be and
says nothing about how fast it moves. A Lipschitz analysis would need the same interval
machinery pointed at a different quantity, and it has not been written. Any reading of these
bounds as adversarial robustness is wrong.

## A bounded output is not a bounded loss

The certified output range `[-5.12, 4.34]` says the model cannot emit an arbitrarily large
signal, whatever it is fed. It says nothing about whether the signal is correct. A model
confined to that interval can still be wrong on every window, and a bounded prediction placed
into a live book can still lose money without limit. No claim is made about market safety, and
none of this prevents a flash crash.

## One instrument, one day

All data is GME on 2021-01-28, one Nasdaq TotalView-ITCH feed, 4,550,476 rows. That is a
single high-volatility session on a single name. The predictive results are illustrative and
should not be generalised from. Nothing has been tested across instruments, across regimes, or
across dates.

The certification results are unaffected, because they never touch the data: they are
properties of the trained weights and hold for every input in `R^{B x L x d_in}`. The two
strands have to be read at different strengths, and conflating them would overclaim the
predictive half by borrowing the guarantees of the other. See [08-data.md](08-data.md).

## The predictive comparison does not resolve

With a 100-tick forward window sampled every tick, 2560 test observations are 25.6 independent
windows.

| model | test rank IC | 95% CI (block bootstrap) |
| --- | --- | --- |
| Causal Transformer | +0.1876 | [+0.0603, +0.2954] |
| Mamba (bounded `delta`) | +0.1413 | [-0.0247, +0.2904] |
| LSTM | +0.0717 | [-0.1168, +0.2542] |
| Ridge | +0.0176 | [-0.0608, +0.0914] |

One interval excludes zero, and it belongs to the transformer rather than to the model this
paper is about. Diebold-Mariano cannot distinguish the three sequence models from one another
(`p = 0.77` transformer against Mamba). The defensible statement is that all three beat the
ridge baseline and that the sample cannot rank them. See [09-statistics.md](09-statistics.md).

## The model plateaus early and was never tuned

The best validation rank IC, `+0.1398`, was recorded at optimiser step 250, the first
evaluation. Evaluation runs every 250 steps with `patience: 8`, so the run continued to step
2250 without improving on that first reading, against a configured budget of 3000. The
transformer peaked at step 250 as well; the LSTM at 750.

No learning-rate sweep was performed. The rate is `1e-3` in `config.yaml` and was not varied.
Neither were `d_model`, `d_state`, depth, batch size, dropout or the weight decay. There is no
evidence that any of these settings is near a good one, and the plateau is at least as
consistent with an untuned optimiser as with a real ceiling on predictability. The
architecture comparison inherits that: three models trained at one hyperparameter setting each
is not a comparison of architectures.

## No external verifier has consumed the artefact

The export states its recurrence in `dynamics_spec` and ships every weight in the generator
chain so that a formal tool could take it, and the round trip through NumPy checks the
description matches the running model. No such tool has been pointed at it. The bounds are
computed by code in this repository and checked by tests in this repository, which is a
weaker position than an independent check. See [11-handoff.md](11-handoff.md).

## The novelty claim is unchecked

No literature review has been done. The claim that the softplus infimum argument and its fix
are new is unverified, and it should be treated as new to the author rather than new. A proper
related-work review is required before this is submitted anywhere, covering at minimum
verification and reachability analysis for state space models, Lipschitz and stability
analysis of recurrent networks, and the interval, zonotope and CROWN line of neural network
verification. If the result is already in the literature, the contribution reduces to the
implementation and the measured ablation, and the paper should say so.

## See also

- [07-soundness.md](07-soundness.md) for how the looseness figures were obtained.
- [09-statistics.md](09-statistics.md) for the standard errors behind the predictive table.
