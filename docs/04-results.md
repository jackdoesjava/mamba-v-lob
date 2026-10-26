# Results

The certification numbers do not depend on the data and would hold on any weights; the
predictive numbers are one instrument on one day and are illustrative. Every number here is read
from a file in `docs/certification/`, `models/results/`, `models/bounds/` or `models/checkpoints/`.

## The invariant against the alternatives

Trained bounded arm, `L = 100`, `B` and `u` bounded input-independently. Induction holds in both
layers, the residual is exactly 0, and the zero-order hold identity checks exact.

| layer | invariant | widened fixed point | iterations | geometric | unrolled at `L=100` | tighter than widened |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 175.9475 | 1.682e4 | 11120 | 1.682e4 | 1514.25 | 95.61x |
| 1 | 137.2183 | 1.306e4 | 10880 | 1.306e4 | 1273.59 | 95.19x |

Against the realised envelope on held-out batches the certified state radius is 162.5 times the
largest realised `|h|` at layer 0 and 382.0 times at layer 1; the boxes on `u` are 3.7 and 3.5 times
loose and those on `B` 16.4 and 15.6 times. See [02-invariant.md](02-invariant.md) and
[03-certificate.md](03-certificate.md).

## Scaling with sequence length

Layer 0; the invariant is constant in `L`, and the unrolled bound crosses it between `L = 10` and
`L = 50`.

| `L` | 10 | 50 | 100 | 500 | 1000 | 5000 |
| --- | --- | --- | --- | --- | --- | --- |
| invariant | 175.95 | 175.95 | 175.95 | 175.95 | 175.95 | 175.95 |
| unrolled | 168.84 | 798.21 | 1514.25 | 6313.87 | 1.023e4 | 1.665e4 |

## The softplus arm

No lower bound on `delta` imposed; induction still holds in both layers, so the invariant needs
no floor on `delta` and only the geometric argument does, which is infinite here.

| layer | invariant | widened fixed point |
| --- | --- | --- |
| 0 | 287.3869 | 2.876e7 |
| 1 | 208.6790 | 2.086e7 |

## What sets the radius

The radius is `sup abs(B u) / abs(A)`, so it moves with the bound on `u`.

| variant | `sup abs u` | radius |
| --- | --- | --- |
| fused LayerNorm and Linear | 9.2681 | 175.9475 |
| LayerNorm box only | 62.0598 | 8075.22 |
| depthwise convolution | 9.2681 | 175.9475 |
| dense convolution | 58.5532 | 1.212e4 |

## Discretisation

Euler against exact zero-order hold, `delta` in `[1e-3, 1e-1]`, `max abs(dt A) = 1.714`. The
identity `Bbar = (1 - Abar) B / |A|` is exact for the zero-order hold and only asymptotic for the
exponential-Euler surrogate `Bbar = dt B`. That surrogate still gives a convex update with the
equilibrium enlarged by `Psi(x) = x / (1 - exp(-x))` at `x = |A| dt`, so an invariant survives with
a radius larger by at most `Psi(|A| dt_max)`: 2.0902 and 2.0859 on the two blocks of the bounded
arm, and 537.5 and 484.8 at the certified ceilings of the softplus arm.

| median relative error | maximum relative error |
| --- | --- |
| 17.3 percent | 109.0 percent |

## Certified output range

From the final LayerNorm, for any input at all; one standardised unit is 0.7042 basis points of
the 100-tick forward return, about a mean of 0.0085 basis points.

| | lower | upper | width |
| --- | --- | --- | --- |
| standardised | -3.5643 | 3.1277 | 6.6920 |
| basis points | -2.502 | +2.211 | 4.713 |

## Adversarial tightness

Gradient ascent on the trained model, 150 steps, layer 0 quantities, no violation found.

| quantity | attacked | certified | ratio |
| --- | --- | --- | --- |
| `sup delta` | 0.0967 | 0.1000 | 1.03x |
| `inf delta` | 0.001061 | 0.001000 | 1.06x |
| `sup abs u` | 7.1924 | 9.2681 | 1.29x |
| `sup abs B` | 2.3962 | 21.4481 | 8.95x |
| `sup abs C` | 2.5339 | 19.4540 | 7.68x |
| max output | 0.6553 | 3.1277 | 4.77x |
| min output | -0.8281 | -3.5643 | 4.30x |
| `sup abs h` against the invariant | 2.7868 | 175.95 | 63.1x |
| `sup abs h`, layer 1 | 2.7911 | 137.22 | 49.2x |

## Machine-checked lemmas

Z3 5.1.0, 30 second timeout, 16 of 19 discharged, 3 nlsat timeouts, zero counterexamples; `exp`
and the zero-order hold are outside decidable real arithmetic. The Isabelle theory in `isabelle/`
proves the zero-order-hold identity, the invariant and the LayerNorm and fused bounds for every `d`.

| lemma | scope | result |
| --- | --- | --- |
| interval product lies between its four corner products | all endpoints and weights | proved |
| interval square handles a box straddling zero | all endpoints | proved |
| positive and negative weight split bounds an affine map | `n <= 4` | proved |
| LayerNorm coordinate bound `z_i^2 <= d - 1` | `d = 2..5` | proved, `d = 6` times out |
| fused LayerNorm and Linear bound is tight | `d = 2, 3` | proved, `d = 4, 5` time out |
| its maximiser lies in the constraint set | `d = 2..5` | proved |

## The soundness apparatus

The artefact and its schema are in [05-handoff.md](05-handoff.md).

| check | result |
| --- | --- |
| NumPy driven only by the artefact against PyTorch | 1.167e-06 worst over every intermediate and the output, tolerance 1e-4 |
| sampled traces against the certified boxes | 0 violations |
| gradient falsification, 16 quantities | 0 violations |
| attacks on the sensitivity bounds, trained arms and released model | 0 violations |
| input box | enforced by winsorisation at inference |
| Z3 lemmas | 16 of 19, zero counterexamples |

## The sensitivity bound

Data-free, per block, with respect to a sup-norm perturbation of the block's normalised input.
Output slopes, the small-`delta` limit of the bound divided by `delta`:

| arm | layer | log-timescale | common start | additive with floor, limit in `L` | additive without floor passes it at `L` |
| --- | --- | --- | --- | --- | --- |
| bounded | 0 | 9.336e6 | 5.17e6 | 2.154e8 | 8.28 |
| bounded | 1 | 6.072e6 | 3.437e6 | 1.363e8 | 8.06 |
| softplus | 0 | 1.843e7 | 1.02e7 | 3.925e26 | 0.23 |
| softplus | 1 | 1.457e7 | 8.074e6 | 4.104e22 | 0.23 |

The log-timescale slope is the same at every length. In the last column a value below 1 means the
additive bound without a floor is the looser at every length. The certified floor of the softplus arm is
`1.27e-21` at block 0 and `1.87e-17` at block 1, which is why the floored additive bound is so
large there. Against gradient attacks on held-out windows at `L = 10` to `5000`, the data-free
slope is `3.06e5` to `3.23e6` times the attacked slope, and `1.7e5` to `1.79e6` times with the
common-start term.

The local rules on the same windows, median end-to-end width at the model input:

| arm | eps | additive | running maximum | recursion | endpoint |
| --- | --- | --- | --- | --- | --- |
| bounded | 1e-7 | 0.4799 | 0.1316 | 0.0409 | 0.0408 |
| bounded | 1e-6 | inf | 10.17 | 0.8564 | 0.8506 |
| bounded | 1e-5 | inf | 5.383e14 | 6.359e10 | 5.728e10 |
| bounded | 1e-4 | inf | inf | inf | 7.822e28 |
| softplus | 1e-7 | 0.7646 | 11.33 | 0.3306 | 0.3267 |
| softplus | 1e-6 | 1.517e12 | 3.75e12 | 67.64 | 61.17 |
| softplus | 1e-5 | inf | inf | 3.368e18 | 1.668e17 |

At `eps = 1e-6` the endpoint bound is 5.67e5 times the attacked sensitivity of the prediction on
the bounded arm and 1.61e7 times on the softplus arm. Of the `eps` values of `domains.json`,
`1e-6` to `1e-1`, it is below the global certified width only at `1e-6` on the bounded arm; at
`eps = 1e-7` it is below it on both arms. Around a block's own normalised input it is 3.37 to 12.6 times
the attacked block slope at `delta = 1e-6`, against 4.04 to 61.3 for the additive rule. See
[`certification/incremental.md`](certification/incremental.md) and
[`certification/incremental_attack.md`](certification/incremental_attack.md).

## The released model

`state-spaces/mamba-130m-hf`, 24 layers, on the first 1024 tokens of
`data/text/public_domain_sample.txt`, in float64. Ranges are minimum to maximum over layers.

| quantity | value |
| --- | --- |
| reference forward pass against transformers, with its float32 casts / without | 0 / 2.58e-5 on outputs up to 67.29 |
| realised quantities inside their certified boxes | every one, at every layer |
| certified ceiling `dt_hi` / realised largest `dt` | 1015 to 2616 / 0.644 to 9.904 |
| `G = Psi(lam dt_hi)`, largest per layer | 4.38e7 to 4.53e11 |
| invariant radius over realised `abs h`, median cell | 2e6 to 1.38e9 |
| fused RMSNorm bound against the coordinate box, largest radius | 441 to 1.06e4 times smaller |
| data-free output slope | 6.11e16 to 2.31e19, 0.9975 to 0.9991 of it through the timescale |
| data-free additive slope without floor, over the log-timescale slope | 379 to 1.69e6 at `L = 1`, 3.88e5 to 1.73e9 at `L = 1024` |
| data-free bound over the attacked change, `delta = 1e-6` | 4.79e12 to 3.82e16 |
| attacked slope, `delta = 1e-6` | 133.4 to 2.71e4 |
| local bound over attacked slope, `delta = 1e-6`: additive / running maximum / recursion / endpoint | 61.95 to 607.9 / 205.9 to 1.63e5 / 92.7 to 1134 / 59.96 to 496 |
| pairs with the endpoint rule below the additive one, `delta = 1e-6`, `1e-4`, `1e-2` | 72 of 72 at each |
| endpoint over additive where both are finite, `delta = 1e-6` | 0.679 to 0.973 |
| additive rule infinite, `delta = 1e-2` | 72 of 72 pairs |
| endpoint bound on the change over the largest realised output, `delta = 1e-6` / `1e-4` / `1e-2` | 0.0023 to 0.066 / 0.26 to 7.5 / 2.4e4 to 6.7e6 |

The data-free certificate fails because interval propagation gives the timescale pre-activation a
box 92 to 255 times wider than its realised range, and `Psi` of the resulting ceiling is
enormous; it is sound and useless. The local endpoint certificate is informative at
`delta = 1e-6`, borderline at `1e-4` and useless at `1e-2`. The bounds are for the float64 map; the
float32 rounding of transformers' slow path is not bounded. The text was typed from memory and has
not been checked against a printed edition. See
[`certification/released.md`](certification/released.md).

## Predictive performance

Held-out test split, 25,600 overlapping observations, which is 256 independent forward windows.

| model | rank IC | 95% CI | block SE | naive iid SE |
| --- | --- | --- | --- | --- |
| Mamba, bounded `delta` | +0.1349 | [+0.0752, +0.1970] | 0.0305 | 0.0063 |
| Mamba, softplus `delta` | +0.1257 | [+0.0769, +0.1766] | 0.0256 | 0.0063 |
| LSTM | +0.1218 | [+0.0665, +0.1778] | 0.0285 | 0.0063 |
| Causal Transformer | +0.1101 | [+0.0610, +0.1593] | 0.0253 | 0.0063 |
| Ridge | +0.0252 | [+0.0107, +0.0405] | 0.0076 | 0.0063 |

Diebold-Mariano, Bartlett weighted and HLN corrected, `h = 100`.

| pair | p |
| --- | --- |
| each neural model against Ridge | 0.000 |
| neural against neural, the two Mamba arms included | 0.171 to 0.951 |
| the two Mamba arms | 0.621 |

No pair of neural models can be separated, and all four separate cleanly from Ridge. Reporting
the naive standard error would put the neural models 18 to 22 standard errors from zero, while the
block bootstrap puts them 4.3 to 4.9 from zero.

The two Mamba arms are not independent models. They share the seed, the initialisation and the
batch order, their validation curves agree to within 0.0016 in rank IC at every one of the 11
shared evaluations, and they differ mainly because validation selected step 750 for one and step
2000 for the other. A comparison between them is a comparison of two selection steps, and it
supports no conclusion about the timescale parametrisation.

## Does the advantage survive a sweep

The ratio is the geometric bound over the invariant. Certification needs no data,
so untrained configurations are as informative as trained ones.

| varied | range | invariant | ratio |
| --- | --- | --- | --- |
| seed, 8 values | fixed config | 101.2 to 132.2 | 95.203 to 95.209 |
| d_state | 4 to 64 | 125.526 throughout | 95.1 to 95.2 |
| expand | 1 to 4 | 78.7 to 166.1 | 95.0 to 95.2 |
| d_model | 16 to 128 | 15.7 to 359.0 | 29.5 to 95.2 |

The ratio moves by 0.006 percent of its median across seeds while the radius moves by 30 percent,
because the ratio depends on the weights only through the slowest pole. It falls at small widths,
reaching 29.5 at `d_model = 16`, and is flat from 64 upwards.

The timescale range is the one setting that moves it, and it moves only the baseline.

| dt_min | sup Abar | invariant | geometric | ratio |
| --- | --- | --- | --- | --- |
| 1e-05 | 1.0000 | 125.526 | 1.193e+06 | 9503.3 |
| 0.0001 | 0.9999 | 125.526 | 1.194e+05 | 951.5 |
| 0.001 | 0.9990 | 125.526 | 1.195e+04 | 95.2 |
| 0.01 | 0.9900 | 125.526 | 1201 | 9.6 |

The invariant is identical to six figures at every floor, because its radius is
`sup|B u| / |A|` and contains no `dt`. The geometric bound divides by `1 - sup Abar`
and pays for the floor directly, about tenfold per decade.

On the trained checkpoints the ratio is 95.62 for the bounded arm. For the softplus arm
the geometric bound is infinite while the invariant certifies at 287.387.

The induction step holds in all 29 configurations tested.

## Training

| model | parameters | steps run | selected step | validation rank IC there |
| --- | --- | --- | --- | --- |
| Mamba, bounded `delta` | 71,361 | 2750, early stop | 750 | +0.1170 |
| Mamba, softplus `delta` | 71,361 | 3000, step budget | 2000 | +0.1173 |
| Causal Transformer | 72,001 | 2250, early stop | 250 | +0.1613 |
| LSTM | 61,249 | 2750, early stop | 750 | +0.1439 |

Selection is by validation rank IC, evaluated every 250 steps with a patience of 8 evaluations.
No learning-rate sweep was run and each model was trained from one seed, so none of these is a
tuned figure.

## Data

| quantity | value |
| --- | --- |
| source | Databento MBP-10, Nasdaq TotalView-ITCH |
| instrument and date | SPY, 2021-01-28 |
| rows after feature construction | 4,550,476 |
| model inputs | 43 |
| train, validation, test windows | 3,185,134 / 682,372 / 682,473 |
| purge at each split boundary | 100 ticks |
| normalised input box | [-22.48, 19.25] |

Features are winsorised to train-split quantiles at inference, so the input box is enforced
rather than observed. `src/dataset.py` raises on a column whose train-split standard deviation is
zero, not finite, or at most `1e-5` times its largest winsorised magnitude, instead of flooring it.
`delta_t` is an input feature and not the SSM timescale, and nothing divides by it.
