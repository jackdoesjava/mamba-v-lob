# Data

One instrument, one day. Everything here is built from a Databento feed of Nasdaq
TotalView-ITCH for GME on 2021-01-28, market by price, ten levels a side. That is enough to
train three models and rank them, and nowhere near enough to generalise from, so the
predictive numbers in this paper are illustrative. The certification results do not depend
on the data at all: they are properties of the trained weights and hold for every input,
in-distribution or not. See [04-stability.md](04-stability.md).

| | |
| --- | --- |
| provider | Databento (US equities) |
| feed | Nasdaq TotalView-ITCH (`XNAS.ITCH`) |
| schema | MBP-10, top ten levels, both sides |
| instrument, date | GME, 2021-01-28 |
| rows | 4,550,476 |
| timestamps | nanosecond UTC |
| prices | fixed-point integers scaled by `1e9` |

`scripts/01_build_features.py` reads the DBN, applies
`OrderBookNormalizer.process_dbn_dataframe` and writes 47 columns: `ts_event`, `mid_price`,
`target_log_return`, and the 44 engineered features. `src/dataset.py` drops those first
three plus `micro_price`, leaving 43 model inputs. `micro_price` goes for the same reason
the raw levels do: it is an absolute price.

## Features

Raw book prices are `I(1)`. Fed directly, a sequence model fits the day's drift rather than
its microstructure, so the book is mapped into a relative coordinate space first.

```
delta_t[k]    = t[k] - t[k-1]                              seconds
P_mid[k]      = (P_ask[0,k] + P_bid[0,k]) / 2
D_ask[i,k]    = P_ask[i,k] - P_mid[k]        i in 0..9
D_bid[i,k]    = P_bid[i,k] - P_mid[k]
V_log[i,k]    = ln(V_raw[i,k] + 1)
y[k]          = ln(P_mid[k+H] / P_mid[k])    H = 100
```

Depth-level sizes are heavy-tailed and skewed, so they go through `log1p`; the `+1` keeps an
empty level off `log(0)`. Three scalars are derived alongside the level features:
`spread_level_0`, `obi_level_0`, `micro_price`. The label is a forward log return over an
event-time horizon of 100 ticks rather than a wall-clock one, and rows with no forward
window are dropped at build time. Consecutive labels therefore share 99 ticks of their
forward window, which is what [09-statistics.md](09-statistics.md) has to correct for.

## delta_t is a feature, not the timescale

The state space model's `Delta` is produced by `dt_proj(x_proj(u))` inside each block. It is
an activation, learned from the content of the sequence, and it has no connection to
wall-clock time. `delta_t` is an ordinary input column sitting beside the distances and the
log sizes. Conflating the two is easy and wrong; a selective SSM learns its timescales, it
does not read them off the clock. How `Delta` is parametrised, and why the certificate
rests entirely on that parametrisation, is in [02-model.md](02-model.md).

Because `delta_t` is only a feature, simultaneous messages are harmless. 2.18 percent of
rows (99,165 of 4,550,476) carry `delta_t = 0`, and nothing downstream divides by it, so no
deduplication is performed. It is also the heaviest-tailed column in the set, which is what
the input box below exists for.

## Structural invariants

| invariant | statement |
| --- | --- |
| uncrossed book | `a0[k] - b0[k] > 0` |
| ask half-space | `D_ask[i,k] >= 0` |
| bid half-space | `D_bid[i,k] <= 0` |
| book monotonicity | `D_ask[i+1,k] >= D_ask[i,k]`, `D_bid[i+1,k] <= D_bid[i,k]` |
| imbalance range | `obi in [-1, 1]` |
| time direction | `delta_t[k] >= 0` |

These hold by construction of the matching engine and of the transformations above, and
`tests/test_features.py` asserts each one against the built parquet. The certificate does
not use them. Every block opens with a LayerNorm, whose output lies in an input-independent
box, so the SSM bounds are unconditional on the features. The invariants are context for a
verifier and a cheap check that the build did not quietly corrupt the book.

## The enforced input box

```
x_norm = (clip(x, w_lo, w_hi) - mu) / sigma
```

`w_lo`, `w_hi`, `mu` and `sigma` are fitted on the training split only, at a winsorisation
quantile of 0.1 percent. The clip is part of the forward path and runs at inference, not
only during fitting, which is what makes the resulting range a deployment guarantee rather
than a historical observation.

| | global normalised range | `delta_t` range | held-out violations |
| --- | --- | --- | --- |
| train min/max, unclipped | `[-6.58, 2065.70]` | `[-0.15, 2065.70]` | 677 entries / 663 rows |
| winsorised at 0.1%, enforced | `[-6.60, 18.87]` | `[-0.21, 18.87]` | 0 |

The gap between those two rows is the argument. Taking the train-split min/max as an
observed box, 677 held-out entries fall outside it. That is a vanishing fraction of the
entries, but they are spread across 663 distinct rows, and the model consumes 100-tick
windows, so one out-of-range tick taints every window containing it: up to roughly 4.9
percent of held-out windows. Unclipped, `delta_t` reaches 2066 sigma under the 70/15/15
split and 176 sigma under the earlier 80/20 split. A figure that swings by an order of
magnitude when the split point moves is not a box anyone should certify against. After
winsorisation the global normalised box is `[-6.60, 18.87]`, with zero violations across all
4,550,476 rows.

## One dtype for the clip and the box

The box has to be computed in float32, matching the clip that produces the model inputs.
Winsorisation pins a great many values exactly on the clip edge, so the edge is where the
box is tight and there is no slack to absorb a rounding difference. Evaluating the same
affine map in float64 lands the edge one ulp inside the float32 result, and 15,262,693
float32 feature values then fall outside the box. That is an unsound export, from arithmetic
alone, with no bug in the pipeline. `src/dataset.py` routes both the training features and
`Normalisation.input_box()` through `Normalisation.apply()`, so the two cannot drift. The
exported box is the one in [11-handoff.md](11-handoff.md).

## Splits

Chronological, three-way, with a purge of `H = 100` ticks at each boundary, since a window
ending within `H` of a split carries a label reaching across it.

| split | rows | windows |
| --- | --- | --- |
| train | 0 to 3,185,333 | 3,185,134 |
| validation | 3,185,333 to 3,867,904 | 682,372 |
| test | 3,867,904 to 4,550,476 | 682,473 |

The test split needs no purge of its own: it ends at the data boundary, and
`01_build_features.py` has already dropped the unlabelled tail rows. Normalisation is fitted
on train and applied everywhere; fitting it on validation or test would be leakage. Early
stopping reads validation. `scripts/03_evaluate_models.py` is the only place the test split
is loaded.

`split_report()` returns those ranges and counts and is embedded in every checkpoint, so the
exact data a model saw is recoverable after the fact. The feature matrix is held in RAM as
float32, 4.55M by 43, about 780 MB. It is not memory-mapped; on a smaller machine, narrow
the row range or point `LOBWindows` at a `np.memmap`.
