# Results

Every number produced by one pass of the pipeline, on the checkpoints in
`models/checkpoints/`. Certification results do not depend on the data and would hold on any
weights; predictive results are from one instrument on one day and are illustrative only.

## What was certified

Data-free, established from the weights alone. Both layers of the trained model.

| quantity | layer 0 | layer 1 |
| --- | --- | --- |
| `sup Abar` | 0.999035 | 0.999041 |
| geometric gain `1/(1 - sup Abar)` | 1036.6 | 1043.5 |
| finite-horizon gain at `L = 100` | 95.37 | 95.42 |
| certified `sup abs(h)` at `L = 100` | 1.169e3 | 1.274e3 |
| certified `sup abs(h)`, geometric | 1.235e4 | 1.368e4 |
| contractive | yes | yes |

The prediction itself is bounded with no premise on the input at all, because `final_norm`
re-bounds the residual stream before the head reads it.

| | value |
| --- | --- |
| certified output range, standardised | `[-5.1242, 4.3374]`, width 9.4616 |
| one standardised unit | 0.7042 basis points of the 100-tick forward return |
| certified output range, basis points | `[-3.600, +3.063]` |
| enforced input box, normalised | `[-6.60, 18.87]`, zero violations in 4,550,476 rows |

## What each constraint buys

From [certification/ablation.md](certification/ablation.md), both arms trained to the same
budget.

| constraint | certified `sup abs(h)` at `L = 100` |
| --- | --- |
| `delta = softplus(z)`, reference | 1.297e4, and no finite invariant set structurally |
| `delta` bounded below | 1.169e3 |
| dense convolution | 8.099e4 |
| depthwise convolution | 1.169e3, so 69 times tighter |
| LayerNorm bounded coordinatewise | 5.708e4 |
| LayerNorm fused with the next Linear | 1.169e3, so 49 times tighter |
| geometric fixed point | 1.235e4 |
| finite horizon at `L = 100` | 1.169e3, so 10.6 times tighter |

Bounding `delta` also bounds the discretisation error it was not aimed at. The Euler surrogate
for `Bbar` carries 411 percent median relative error at the timescale the unconstrained model
learned, and 17 percent once `delta` is confined to `[1e-3, 1e-1]`. The model uses the exact
zero-order hold either way, so the figure prices the shortcut rather than describing the code.

## How tight the bounds are

`scripts/07_attack_bounds.py`, trained model, unrestricted input, 60 Adam steps per objective.
Since `attacked <= true reachable <= certified`, each ratio is an upper bound on looseness.

| quantity | attacked | certified | ratio |
| --- | --- | --- | --- |
| `sup delta` | 0.09402 | 0.10000 | 1.06 |
| `inf delta` | 1.088e-3 | 1.000e-3 | 1.09 |
| `sup abs(u)` | 4.563 | 8.427 | 1.85 |
| max output | 0.3593 | 4.3374 | 12.07 |
| min output | -0.9234 | -5.1242 | 5.55 |
| `sup abs(B)` | 1.816 | 17.713 | 9.75 |
| `sup abs(C)` | 1.388 | 17.527 | 12.63 |
| `sup abs(h)` | 0.5130 | 1.169e3 | 2.28e3 |
| `sup abs(y)` | 4.740 | 1.601e5 | 3.38e4 |

No violation was found. The attained output range is `[-0.9234, 0.3593]`, width 1.2827,
against a certified 9.4616, so the output bound is 7.4 times wider than anything the search
could reach. The `delta` rows being near-exact is what makes the contraction result usable;
the state and output rows inside a block are loose because interval arithmetic compounds over
100 steps, which [16-domains.md](16-domains.md) takes further.

## Which abstract domain works

Median over three held-out windows. Widths in standardised target units.

| eps | attack | norm propagation | interval | unconditional |
| --- | --- | --- | --- | --- |
| 1e-6 | 1.6242e-6 | 2.1798 | 0.5514 | 9.4616 |
| 1e-5 | 1.6302e-5 | inf | 20.402 | 9.4616 |
| 1e-4 | 1.6296e-4 | inf | 2.7576e7 | 9.4616 |
| 1e-3 | 1.6294e-3 | inf | 2.1698e25 | 9.4616 |
| 1e-2 | 1.6310e-2 | inf | nan | 9.4616 |
| 1e-1 | 1.6217e-1 | nan | nan | 9.4616 |

The attack width is linear in `eps` across five decades, so the model's true local Lipschitz
constant is about 1.63 and the model is locally well behaved. Both local domains diverge
anyway, and the unconditional bound is below both at every `eps` at or above 1e-5.

## Machine-checked lemmas

`scripts/09_check_lemmas.py`, Z3 5.1.0, 30 second timeout per check. 16 of 19 discharged, 3
nlsat timeouts, no counterexamples.

| lemma | scope | result |
| --- | --- | --- |
| interval product lies between its four corner products | all endpoints and points | proved |
| interval square over a box straddling zero | all endpoints and points | proved |
| positive and negative weight split bounds an affine map | all weights, `n <= 4` | proved |
| LayerNorm coordinate bound `z_i^2 <= d - 1` | `d = 2..5` | proved |
| fused LayerNorm and Linear is tight | `d = 2, 3` | proved |
| the maximiser lies in the constraint set | `d = 2..5` | proved |

The first three quantify over the weights and endpoints, so the operations the certificate is
built from are proved sound rather than sampled. `exp` and the zero-order hold sit outside
decidable real arithmetic and keep their hand proofs.

## Soundness apparatus

| check | result |
| --- | --- |
| sampling, every abstract operation | no escape found |
| NumPy round trip against PyTorch | 8.771e-07 worst over every intermediate and the output |
| realised envelope inside the certified box | no violation, 20 held-out batches |
| gradient-based falsification | no violation |

## Predictive performance

Held-out test split, loaded only by `scripts/03_evaluate_models.py`. 25,600 overlapping
observations, which is 256 independent forward windows.

| model | rank IC | 95% CI, block bootstrap | block SE | naive SE | hit rate | MSE |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM | +0.1111 | `[+0.0516, +0.1721]` | 0.0306 | 0.0063 | 53.30% | 0.335516 |
| Mamba, bounded `delta` | +0.1044 | `[+0.0420, +0.1690]` | 0.0318 | 0.0063 | 54.47% | 0.337268 |
| Mamba, softplus `delta` | +0.1043 | `[+0.0420, +0.1689]` | 0.0318 | 0.0063 | 54.46% | 0.337271 |
| Causal Transformer | +0.1040 | `[+0.0583, +0.1496]` | 0.0239 | 0.0063 | 54.97% | 0.337308 |
| Ridge | +0.0312 | `[+0.0166, +0.0460]` | 0.0076 | 0.0063 | 50.47% | 0.743454 |

The block standard error is 4 to 5 times the naive one. Reporting the naive figure would put
every neural model at 16 standard errors from zero; the correct figure puts them at 3 to 4.

Diebold-Mariano, Bartlett-weighted and HLN-corrected, `h = 100`:

| pair | p | favours |
| --- | --- | --- |
| any neural model against Ridge | 0.000 | the neural model |
| Mamba bounded against the transformer | 0.983 | neither |
| Mamba bounded against the LSTM | 0.576 | neither |
| Mamba bounded against Mamba softplus | 0.0017 | bounded |

The four neural models cannot be separated on rank IC, and all four separate cleanly from
Ridge. The one significant pair is the two Mamba arms, where the constrained parametrisation
is favoured on squared error by 3e-6, which is detectable across 25,600 paired observations
and means nothing economically. The claim the ablation supports is that bounding `delta` costs
no accuracy, not that it gains any.

## Training

| model | parameters | steps | best validation rank IC | at step |
| --- | --- | --- | --- | --- |
| Mamba, bounded `delta` | 71,361 | 2250 | +0.1398 | 250 |
| Mamba, softplus `delta` | 71,361 | 2250 | +0.1398 | 250 |
| Causal Transformer | 72,001 | 2250 | +0.1632 | 250 |
| LSTM | 61,249 | 3000 | +0.1004 | 1750 |

Every model reaches its best validation rank IC at or near the first evaluation and then
plateaus, so all four runs early-stopped. No learning-rate sweep was performed. The two Mamba
arms reach +0.139834 and +0.139839 from the same seed, which is the accuracy statement the
ablation rests on.

## Data

| | |
| --- | --- |
| rows after feature construction | 4,550,476 |
| model inputs | 43 |
| train windows | 3,185,134 |
| validation windows | 682,372 |
| test windows | 682,473 |
| purge at each split boundary | 100 ticks |
