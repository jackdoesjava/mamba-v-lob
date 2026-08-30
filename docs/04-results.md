# Results

The certification numbers do not depend on the data and would hold on any weights; the
predictive numbers are one instrument on one day and are illustrative.

## The invariant against the alternatives

Trained model, `L = 100`, `B` and `u` bounded input-independently. Induction holds in both
layers, the residual is exactly 0, and the zero-order hold identity checks exact.

| layer | invariant | widened fixed point | iterations | geometric | unrolled at `L=100` | tighter than widened |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 129.6739 | 1.235e4 | 10509 | 1.235e4 | 1169.03 | 95.23x |
| 1 | 143.5443 | 1.368e4 | 10634 | 1.368e4 | 1273.92 | 95.31x |

With `B` and `u` taken from the realised trace instead of the certified boxes, which separates
the invariant from the bounds feeding it.

| layer | invariant from realised `B`, `u` | realised `sup abs h` | ratio |
| --- | --- | --- | --- |
| 0 | 1.0020 | 0.2066 | 4.8x |
| 1 | 0.5139 | 0.1585 | 3.2x |

The machinery is 3 to 5 times loose; the rest of the certified gap is in the bounds on `B` and
`u`. In total the certified radius is about 630 times the realised state at layer 0, 130 of it
from the boxes and 4.8 from the induction. See [02-invariant.md](02-invariant.md) and [03-certificate.md](03-certificate.md).

## Scaling with sequence length

Layer 0; the invariant is constant in `L`, and the unrolled bound crosses it at about `L = 10`.

| `L` | 10 | 50 | 100 | 500 | 1000 | 5000 |
| --- | --- | --- | --- | --- | --- | --- |
| invariant | 129.67 | 129.67 | 129.67 | 129.67 | 129.67 | 129.67 |
| unrolled | 127.56 | 601.78 | 1169.03 | 4838.40 | 7781.12 | 1.226e4 |

## The softplus arm

No lower bound on `delta` imposed; induction still holds in both layers, so the invariant needs
no floor on `delta` and only the geometric argument does.

| layer | invariant | widened fixed point |
| --- | --- | --- |
| 0 | 129.7168 | 1.298e7 |
| 1 | 143.3978 | 1.433e7 |

## What sets the radius

The radius is `sup abs(B u) / abs(A)`, so it moves with the bound on `u`.

| variant | `sup abs u` | radius |
| --- | --- | --- |
| fused LayerNorm and Linear | 8.4267 | 129.6739 |
| LayerNorm box only | 61.3134 | 6332.03 |
| depthwise convolution | 8.4267 | 129.6739 |
| dense convolution | 56.3247 | 9239.72 |

## Discretisation

Euler against exact zero-order hold, `delta` in `[1e-3, 1e-1]`, `max abs(dt A) = 1.684`; the
convex form is exact for the zero-order hold and only asymptotic for Euler, so the invariant
needs the exact discretisation to exist.

| median relative error | maximum relative error |
| --- | --- |
| 17.3 percent | 106.8 percent |

## Certified output range

From the final LayerNorm, for any input at all; one standardised unit is 0.7042 basis points of
the 100-tick forward return.

| | lower | upper | width |
| --- | --- | --- | --- |
| standardised | -5.1242 | 4.3374 | 9.4616 |
| basis points | -3.600 | +3.063 | 6.663 |

## Adversarial tightness

Gradient ascent on the trained model, 60 steps, layer 0 quantities, no violation found.

| quantity | attacked | certified | ratio |
| --- | --- | --- | --- |
| `sup delta` | 0.0940 | 0.1000 | 1.06x |
| `inf delta` | 0.001088 | 0.001000 | 1.09x |
| `sup abs u` | 4.5630 | 8.4267 | 1.85x |
| `sup abs B` | 1.8159 | 17.7131 | 9.75x |
| `sup abs C` | 1.3877 | 17.5265 | 12.63x |
| max output | 0.3593 | 4.3374 | 12.07x |
| min output | -0.9234 | -5.1242 | 5.55x |
| `sup abs h` against the invariant | 0.5130 | 129.67 | 252.8x |
| output width | 1.2827 | 9.4616 | 7.4x |

## Machine-checked lemmas

Z3 5.1.0, 30 second timeout, 16 of 19 discharged, 3 nlsat timeouts, zero counterexamples; `exp`
and the zero-order hold are outside decidable real arithmetic and keep hand proofs.

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
| NumPy driven only by the artefact against PyTorch | 8.771e-07 worst over every intermediate and the output, tolerance 1e-4 |
| sampled traces against the certified boxes | 0 violations |
| gradient falsification, 16 quantities | 0 violations |
| enforced input box over 4,550,476 rows | 0 violations |
| Z3 lemmas | 16 of 19, zero counterexamples |

## Local sensitivity

Both local domains diverge while the unconditional bound does not; median over three held-out
windows, widths in standardised target units.

| `eps` | attack | norm propagation | interval | unconditional |
| --- | --- | --- | --- | --- |
| 1e-6 | 1.6242e-6 | 2.1798 | 0.5514 | 9.4616 |
| 1e-5 | 1.6302e-5 | inf | 20.402 | 9.4616 |
| 1e-4 | 1.6296e-4 | inf | 2.7576e7 | 9.4616 |
| 1e-1 | 1.6217e-1 | nan | nan | 9.4616 |

The attack width is linear in `eps` across five decades, so the model's true local Lipschitz
constant is about 1.63. The domains fail, not the model.

## Predictive performance

Held-out test split, 25,600 overlapping observations, which is 256 independent forward windows.

| model | rank IC | 95% CI | block SE | naive iid SE |
| --- | --- | --- | --- | --- |
| LSTM | +0.1111 | [+0.0516, +0.1721] | 0.0306 | 0.0063 |
| Mamba, bounded `delta` | +0.1044 | [+0.0420, +0.1690] | 0.0318 | 0.0063 |
| Mamba, softplus `delta` | +0.1043 | [+0.0420, +0.1689] | 0.0318 | 0.0063 |
| Causal Transformer | +0.1040 | [+0.0583, +0.1496] | 0.0239 | 0.0063 |
| Ridge | +0.0312 | [+0.0166, +0.0460] | 0.0076 | 0.0063 |

Diebold-Mariano, Bartlett weighted and HLN corrected, `h = 100`.

| pair | p |
| --- | --- |
| each neural model against Ridge | 0.000 |
| neural against neural | 0.556 to 0.984 |
| the two Mamba arms | 0.0017 |

The four neural models cannot be separated from each other, and all four separate cleanly from
Ridge. Reporting the naive standard error would put every neural model at 16 standard errors
from zero, while the correct figure puts them at 3 to 4. The one significant pair is the two
Mamba arms, favouring the constrained parametrisation by 3e-6 in MSE, which is detectable
across 25,600 paired observations and means nothing economically; the claim it supports is that
the parametrisation costs no accuracy, not that it gains any.

## Does the advantage survive a sweep

The ratio is the geometric bound over the invariant. Certification needs no data,
so untrained configurations are as informative as trained ones.

| varied | range | invariant | ratio |
| --- | --- | --- | --- |
| seed, 8 values | fixed config | 101.2 to 132.2 | 95.203 to 95.209 |
| d_state | 4 to 64 | 125.526 throughout | 95.1 to 95.2 |
| expand | 1 to 4 | 78.7 to 166.1 | 95.0 to 95.2 |
| d_model | 16 to 128 | 15.7 to 359.0 | 29.5 to 95.2 |

The ratio moves by 0.0 percent of its median across seeds, so it is a property of the
configuration rather than of a particular draw of the weights. It falls at small widths,
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

On the trained checkpoints the ratio is 95.24 for the bounded arm. For the softplus arm
the geometric bound is infinite while the invariant certifies at 129.717.

The induction step holds in all 29 configurations tested.

## Training

| model | parameters | steps | best validation rank IC | at step |
| --- | --- | --- | --- | --- |
| Mamba, bounded `delta` | 71,361 | 2250 | +0.1398 | 250 |
| Mamba, softplus `delta` | 71,361 | 2250 | +0.1398 | 250 |
| Causal Transformer | 72,001 | 2250 | +0.1632 | 250 |
| LSTM | 61,249 | 3000 | +0.1004 | 1750 |

All four early-stopped, and every one peaks at or near the first evaluation and then plateaus.
No learning-rate sweep was run, so none of these is a tuned figure.

## Data

| quantity | value |
| --- | --- |
| source | Databento MBP-10, Nasdaq TotalView-ITCH |
| instrument and date | GME, 2021-01-28 |
| rows after feature construction | 4,550,476 |
| model inputs | 43 |
| train, validation, test windows | 3,185,134 / 682,372 / 682,473 |
| purge at each split boundary | 100 ticks |
| normalised input box | [-6.60, 18.87] |
| rows carrying `delta_t = 0` | 2.18 percent |

Features are winsorised to train-split quantiles at inference, so the input box is enforced
rather than observed, with zero violations across all 4,550,476 rows. `delta_t` is an input
feature and not the SSM timescale, and nothing divides by it.
