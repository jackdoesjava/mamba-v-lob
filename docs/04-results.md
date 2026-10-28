# Results

Every number here is read from a file in `docs/certification/`, `models/results/`,
`models/bounds/` or `models/checkpoints/`, or computed directly from numbers in one, except one
entry, marked below, that no script records yet. The data-free certificates are computed from the weights alone and hold for every input,
including inputs that no order book would produce. The predictive numbers come from one
instrument on one day and only show that the certified models are working models.

The first section, In the paper, has the numbers the paper reports, at the precision of the
recorded runs. The paper rounds some of them, and the few entries in that section that the paper
does not report are marked. The rest of the page covers work that the paper leaves out.

## In the paper

### One cell

These numbers come from `scripts/16_cell_tightness.py`, with `lam = 1`, `M = 1` and both runs
started from 0 ([certification/cell_tightness.md](certification/cell_tightness.md)). The search
runs Adam from random initial sequences and from initial sequences shaped like the construction
of Proposition 4.

| quantity | value |
| --- | --- |
| largest gap found over the hypothesis class of Theorem 3, against `eta + 2 M tanh(rho/2)` | never above it; for `T >= 300`, within 0.4 per cent from random initial sequences and within 0.01 per cent from initial sequences shaped like the construction |
| theorem's bound over the largest gap, `eta = 0`, `T >= 300` | `e^rho + 1` to four significant figures |
| bounded map, `rho = 0.01`: the additive baseline passes the theorem's bound | from `T = 39` without the floor and `T = 40` with it |
| bounded map: the floored baseline as `T -> inf` | 25.6 times the theorem's bound |
| softplus map, `rho = 0.01`: the floor `dt(inf Z)` | about `1.4e-21` |
| softplus map: the baseline passes the theorem's bound / the trivial bound `2M` | from `T = 2` / from `T = 114`, which no script records yet |
| softplus map: the baseline at `T = 3000` | more than 2600 times the theorem's bound |

### The data-free block certificate

Each certificate bounds one block, with respect to a sup-norm perturbation of the block's
normalised input. The numbers are output slopes, the small-`delta` limit of the bound divided by
`delta` ([certification/incremental.md](certification/incremental.md)).

| arm | block | log-timescale | common start | additive with floor, limit in `L` | additive without floor passes it at `L` |
| --- | --- | --- | --- | --- | --- |
| bounded | 0 | 9.336e6 | 5.17e6 | 2.154e8 | 8.28 |
| bounded | 1 | 6.072e6 | 3.437e6 | 1.363e8 | 8.06 |
| softplus | 0 | 1.843e7 | 1.02e7 | 3.925e26 | 0.23 |
| softplus | 1 | 1.457e7 | 8.074e6 | 4.104e22 | 0.23 |

The log-timescale column is the paper's data-free certificate (Corollary 8), the column headed
Theorem 3 in its Table 1, where the slopes are rounded up. It is the same at every length. The
common-start column is not in the paper. It uses the sharper constant `2 M tanh(rho/2)`, which
the paper does not prove (see [02-invariant.md](02-invariant.md)). In the last column, a value
below 1 means that the additive bound without a floor is the looser at every length, which the
paper's Table 1 reports as a crossing of none.

Without a floor, at `L = 5000` the baseline is 539 and 538 times ours on the bounded blocks and
1.95e4 and 1.97e4 times on the softplus blocks. The baseline is affine in `L`, and its values
at `L = 10` and `L = 5000` show that it grows by about 0.11 times our certificate per step on the
bounded arm and 3.9 times on the softplus arm. Over the cells of the bounded
arm, the baseline's time-warp term reaches ours after 3 to 42 steps, with a median of 5 (`n*`),
and on the softplus arm after at most 2. With the certified floor, the ratio of the two
time-warp terms lies between 24.1 and 25.8 over the cells of the bounded arm. The terms the two
certificates share make up 0.11 to 0.13 of ours. With the floor that an attack on the block's
input reaches in each channel, the baseline on the bounded arm is still 8.5 to 12 times ours.

The smallest certified floor of the softplus arm is `1.27e-21` at block 0 and `1.87e-17` at
block 1, which is why the floored additive bound is so large there.

### Against attacks

`scripts/13_incremental_attack.py` attacks each block on four held-out windows at each of
`L = 10`, `100`, `1000` and `5000`
([certification/incremental_attack.md](certification/incremental_attack.md)). The two rows on the
stages come instead from `scripts/12_incremental.py`. It compares the certified bound after each
stage with the largest change that random perturbations of the block input produce there
([certification/incremental.md](certification/incremental.md)). Random perturbations do not
search for the worst case, so each such ratio is an upper bound on the looseness after its stage.

| quantity | value |
| --- | --- |
| data-free slope over the attacked slope, `delta = 1e-6` | `3.06e5` to `3.23e6` (`1.7e5` to `1.79e6` with the common-start term, which the paper does not report) |
| growth of the attacked slope from `L = 10` to `5000` | a factor of 1.4 to 7.3, against at least 445 for the baseline without a floor |
| attacked slope at the last position alone | grows by a factor of 1.06 to 2.3; the last position is the worst in 1 of 16 cases |
| certified bound over the largest change from random perturbations, after each stage, block 0 of the bounded arm (an upper bound on the looseness) | 8.2 after `u`, 134 after `B`, 537 after `c`, 1.54e4 after `h`, 3.24e5 after `y`, 4.65e7 at the output |
| factor of a stage, the ratio after it over the ratio before it | the gate and output projection, 98 to 147; the scan, 18 to 39; the readout, 14 to 29; forming `c` from `B` and `u`, 4 to 7 |
| local endpoint bound over the attacked slope, `delta <= 1e-4` | at most 12.7 |
| change from dropping the log-timescale interval, `delta <= 1e-4` | at most 0.034 per cent |

### The trained model

The decay rates `lam` of the bounded arm lie between 0.912 and 17.14. The kept checkpoints are
the ones with the best rank information coefficient on a fixed slice of 12,800 validation
windows: step 750 on the bounded arm and step 2000 on the softplus arm. On the first 25,600 test
windows the bounded arm has a rank information coefficient of 0.1349, with a block bootstrap 95
per cent interval of `[0.0752, 0.1970]`.

## Beyond the paper

### The invariant against the alternatives

These are for the trained bounded arm at `L = 100`, with `B` and `u` bounded independently of
the input. Induction holds in both layers, the induction residual is exactly 0, and the
zero-order hold identity checks exactly. The paper quotes the ratio of 95.6 and 95.2 in
Appendix C.

| layer | invariant | widened fixed point | iterations | geometric | unrolled at `L=100` | tighter than widened |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 175.9475 | 1.682e4 | 11120 | 1.682e4 | 1514.25 | 95.61x |
| 1 | 137.2183 | 1.306e4 | 10880 | 1.306e4 | 1273.59 | 95.19x |

Against the realised states on held-out batches, the certified radius is 162.5 times the largest
realised `|h|` at layer 0 and 382.0 times at layer 1. The boxes on `u` are 3.7 and 3.5 times
loose, and those on `B` 16.4 and 15.6 times. See [02-invariant.md](02-invariant.md) and
[03-certificate.md](03-certificate.md).

### Scaling with sequence length

On layer 0 the invariant is the same at every `L`, and the unrolled bound passes it between
`L = 10` and `L = 50`.

| `L` | 10 | 50 | 100 | 500 | 1000 | 5000 |
| --- | --- | --- | --- | --- | --- | --- |
| invariant | 175.95 | 175.95 | 175.95 | 175.95 | 175.95 | 175.95 |
| unrolled | 168.84 | 798.21 | 1514.25 | 6313.87 | 1.023e4 | 1.665e4 |

### The softplus arm

No lower bound on `delta` is imposed. Induction still holds in both layers, so the invariant
needs no floor. Only the separated (geometric) bound does, and it is infinite here.

| layer | invariant | widened fixed point |
| --- | --- | --- |
| 0 | 287.3869 | 2.876e7 |
| 1 | 208.6790 | 2.086e7 |

### What sets the radius

The radius is `sup abs(B u) / abs(A)`, so it moves with the bound on `u`.

| variant | `sup abs u` | radius |
| --- | --- | --- |
| fused LayerNorm and Linear | 9.2681 | 175.9475 |
| LayerNorm box only | 62.0598 | 8075.22 |
| depthwise convolution | 9.2681 | 175.9475 |
| dense convolution | 58.5532 | 1.212e4 |

### Discretisation

The identity `Bbar = (1 - Abar) B / |A|` is exact for the zero-order hold and only approximate
for the exponential-Euler input `Bbar = dt B`. With `delta` in `[1e-3, 1e-1]` and
`max abs(dt A) = 1.714`, the two differ by a relative error of 17.3 per cent in median and 109.0
per cent at most. The exponential-Euler step still gives a convex update, with the equilibrium
enlarged by `Psi(x) = x / (1 - exp(-x))` at `x = |A| dt`. An invariant therefore survives with a
radius larger by at most `Psi(|A| dt_max)`: 2.0902 and 2.0859 on the two blocks of the bounded
arm, and 537.5 and 484.8 at the certified ceilings of the softplus arm.

### Certified output range

The range is certified from the final LayerNorm on, so it holds for any input at all. One
standardised unit is 0.7042 basis points of the 100-tick forward return, around a mean of 0.0085
basis points.

| | lower | upper | width |
| --- | --- | --- | --- |
| standardised | -3.5643 | 3.1277 | 6.6920 |
| basis points | -2.502 | +2.211 | 4.713 |

### How tight the boxes are

Gradient ascent ran for 150 steps on the trained model, and no violation was found. The rows are
for layer 0 unless marked.

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

### Machine-checked lemmas

Z3 5.1.0 with a 30-second timeout discharges 16 of 19 obligations, with 3 `nlsat` timeouts and no
counterexamples. `exp` and the zero-order hold are outside decidable real arithmetic. The
Isabelle theory in `isabelle/` proves the discretisation identity, the invariant, and the
LayerNorm and fused bounds for every `d`.

| lemma | scope | result |
| --- | --- | --- |
| interval product lies between its four corner products | all endpoints and weights | proved |
| interval square handles a box straddling zero | all endpoints | proved |
| positive and negative weight split bounds an affine map | `n <= 4` | proved |
| LayerNorm coordinate bound `z_i^2 <= d - 1` | `d = 2..5` | proved, `d = 6` times out |
| fused LayerNorm and Linear bound is tight | `d = 2, 3` | proved, `d = 4, 5` time out |
| its maximiser lies in the constraint set | `d = 2..5` | proved |

### Soundness checks

The exported artefact and its schema are described in [05-handoff.md](05-handoff.md).

| check | result |
| --- | --- |
| NumPy driven only by the artefact, against PyTorch | 1.167e-06 worst over every intermediate and the output, tolerance 1e-4 |
| sampled traces against the certified boxes | 0 violations |
| gradient falsification, 16 quantities | 0 violations |
| attacks on the sensitivity bounds, trained arms and released model | 0 violations |
| input box | enforced by winsorisation at inference |
| Z3 lemmas | 16 of 19, no counterexamples |

### Local rules end to end

The five local rules of [03-certificate.md](03-certificate.md), run on the four held-out windows
of `domains.json`, give the median end-to-end widths below. Each width is that of the
prediction, for a perturbation of size `eps` at the model input.

| arm | eps | additive | running maximum | recursion | additive endpoint | endpoint |
| --- | --- | --- | --- | --- | --- | --- |
| bounded | 1e-7 | 0.0490 | 0.1316 | 0.0409 | 0.0408 | 0.0408 |
| bounded | 1e-6 | 1.291 | 10.17 | 0.8564 | 0.8514 | 0.8503 |
| bounded | 1e-5 | inf | 5.383e14 | 6.359e10 | 6.43e10 | 5.525e10 |
| bounded | 1e-4 | inf | inf | inf | 8.1e28 | 7.455e28 |
| softplus | 1e-7 | 0.7646 | 11.33 | 0.3306 | 0.3276 | 0.3267 |
| softplus | 1e-6 | 1.517e12 | 3.75e12 | 67.64 | 68.42 | 61.17 |
| softplus | 1e-5 | inf | inf | 3.368e18 | 2.333e17 | 1.668e17 |

The additive rule takes the timescale's slope over the local box. On the bounded arm this makes
it finite at `eps = 1e-6`, where the slope of `lipschitz._delta` that it used earlier sent it past
the float32 range. At `eps = 1e-6` the endpoint rule's bound is 5.67e5 times the attacked
sensitivity of the prediction on the bounded arm and 1.61e7 times on the softplus arm. Of the
`eps` values of `domains.json`, `1e-6` to `1e-1`, it is below the global certified width only at
`1e-6` on the bounded arm. At `eps = 1e-7` it is below it on both arms. Around a block's own
normalised input it is 3.37 to 12.6 times the attacked block slope at `delta = 1e-6`, against 3.41
to 18.0 for the additive rule.

### The released model

These are for `state-spaces/mamba-130m-hf`, with 24 layers, run in float64 on the first 1024
tokens of `data/text/public_domain_sample.txt`. Each range runs from the smallest to the largest
value over layers.

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
| local bound over attacked slope, `delta = 1e-6`: additive / running maximum / recursion / additive endpoint / endpoint | 61.95 to 607.9 / 205.9 to 1.63e5 / 92.7 to 1134 / 59.96 to 496.1 / 59.96 to 496 |
| pairs with the endpoint rule below the additive one, `delta = 1e-6`, `1e-4`, `1e-2` | 72 of 72 at each |
| endpoint over additive where both are finite, `delta = 1e-6` | 0.679 to 0.973 |
| endpoint over additive endpoint (what the log-timescale interval adds), `delta = 1e-6` / `1e-4` / `1e-2` | 0.99935 to 0.99999 / 0.916 to 0.998 / 0.607 to 0.9995 |
| additive rule infinite, `delta = 1e-2` | 72 of 72 pairs |
| endpoint bound on the change over the largest realised output, `delta = 1e-6` / `1e-4` / `1e-2` | 0.0023 to 0.066 / 0.26 to 7.5 / 2.4e4 to 6.7e6 |

The data-free certificate is sound but uninformative, because interval propagation gives the
timescale pre-activation a box 92 to 255 times wider than its realised range, and
`G = Psi(lam dt_hi)` then reaches 4.53e11. The local endpoint bound is informative at
`delta = 1e-6`, borderline at `1e-4` and of no use at `1e-2`. The bounds are for the float64 map.
The float32 rounding of transformers' slow path is not bounded. The text was typed from memory
and has not been checked against a printed edition. The full results are in
[certification/released.md](certification/released.md).

### Predictive performance

Scores are on the held-out test split: 25,600 overlapping observations, which make 256
independent forward windows.

| model | rank IC | 95% CI | block SE | naive iid SE |
| --- | --- | --- | --- | --- |
| Mamba, bounded `delta` | +0.1349 | [+0.0752, +0.1970] | 0.0305 | 0.0063 |
| Mamba, softplus `delta` | +0.1257 | [+0.0769, +0.1766] | 0.0256 | 0.0063 |
| LSTM | +0.1218 | [+0.0665, +0.1778] | 0.0285 | 0.0063 |
| Causal Transformer | +0.1101 | [+0.0610, +0.1593] | 0.0253 | 0.0063 |
| Ridge | +0.0252 | [+0.0107, +0.0405] | 0.0076 | 0.0063 |

The Diebold-Mariano tests use Bartlett weights and the HLN correction, with `h = 100`.

| pair | p |
| --- | --- |
| each neural model against Ridge | 0.000 |
| neural against neural, the two Mamba arms included | 0.171 to 0.951 |
| the two Mamba arms | 0.621 |

No pair of neural models can be told apart, and all four separate clearly from Ridge. With the
naive standard error the neural models would sit 18 to 22 standard errors from zero. The block
bootstrap puts them 4.3 to 4.9 from zero.

The two Mamba arms are not independent models. They share the seed, the initialisation and the
batch order. Their validation curves agree to within 0.0016 in rank IC at every one of the 11
shared evaluations, and they differ mainly because validation selected step 750 for one and step
2000 for the other. Comparing them compares two selection steps, so it supports no conclusion
about the timescale parametrisation.

### The separated bound across configurations

The ratio below is the separated (geometric) bound over the invariant. Certification needs no
data, so untrained configurations are as informative as trained ones.

| varied | range | invariant | ratio |
| --- | --- | --- | --- |
| seed, 8 values | fixed config | 101.2 to 132.2 | 95.203 to 95.209 |
| d_state | 4 to 64 | 125.526 throughout | 95.1 to 95.2 |
| expand | 1 to 4 | 78.7 to 166.1 | 95.0 to 95.2 |
| d_model | 16 to 128 | 15.7 to 359.0 | 29.5 to 95.2 |

The ratio moves by 0.006 per cent of its median across seeds while the radius moves by 30 per
cent, because the ratio depends on the weights only through the slowest pole. It falls at small
widths, to 29.5 at `d_model = 16`, and is flat from 64 upwards.

The timescale range is the one setting that moves it, and it moves only the separated bound.

| dt_min | sup Abar | invariant | geometric | ratio |
| --- | --- | --- | --- | --- |
| 1e-05 | 1.0000 | 125.526 | 1.193e+06 | 9503.3 |
| 0.0001 | 0.9999 | 125.526 | 1.194e+05 | 951.5 |
| 0.001 | 0.9990 | 125.526 | 1.195e+04 | 95.2 |
| 0.01 | 0.9900 | 125.526 | 1201 | 9.6 |

The invariant is identical to six figures at every floor, because its radius is
`sup|B u| / |A|` and contains no `dt`. The separated bound divides by `1 - sup Abar` and pays for
the floor directly, about tenfold per decade. On the trained checkpoints the ratio is 95.62 for
the bounded arm. For the softplus arm the separated bound is infinite while the invariant
certifies at 287.387. The induction step holds in all 29 configurations tested.

### Training

| model | parameters | steps run | selected step | validation rank IC there |
| --- | --- | --- | --- | --- |
| Mamba, bounded `delta` | 71,361 | 2750, early stop | 750 | +0.1170 |
| Mamba, softplus `delta` | 71,361 | 3000, step budget | 2000 | +0.1173 |
| Causal Transformer | 72,001 | 2250, early stop | 250 | +0.1613 |
| LSTM | 61,249 | 2750, early stop | 750 | +0.1439 |

Selection is by validation rank IC, evaluated every 250 steps with a patience of 8 evaluations.
No learning-rate sweep was run and each model was trained from one seed, so none of these is a
tuned figure.

### Data

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
rather than observed. `src/dataset.py` raises an error on a column whose train-split standard
deviation is zero, not finite, or at most `1e-5` times its largest winsorised magnitude, instead
of flooring it. `delta_t` is an input feature and not the SSM timescale, and nothing divides by
it.
