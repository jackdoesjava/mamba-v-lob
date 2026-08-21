# Statistics

The label is a forward 100-tick log return, and it is sampled at every tick. Observation `k`
and observation `k+1` share 99 ticks of the same forward window, so consecutive labels are
close to the same number by construction. Every standard error on this data has to be built
around that. `src/utils/stats.py` exists for no other reason.

## Effective sample size

The crude correction first, because it sets the scale of the problem:

```
n_eff = max(1, n / H)
```

with `H = 100`. The recorded test evaluation has 2560 overlapping observations, which is 25.6
independent forward windows.

| quantity | value |
| --- | --- |
| observations `n` | 2560 |
| horizon `H` | 100 |
| `n_eff` | 25.6 |
| naive i.i.d. SE, `1/√n` | 0.0198 |
| window-count SE, `1/√n_eff` | 0.1976 |

The two differ by exactly `√H = 10`. A statistic quoted against the first of those is
overstated by an order of magnitude. `n_eff` is used directly only for the information ratio
standard error; the rank IC gets something less blunt.

## Rank IC with a circular block bootstrap

`block_bootstrap_ic` resamples blocks one forward window long. It draws `ceil(n / H)` start
indices uniformly, expands each into `H` consecutive positions, wraps them modulo `n` so the
series is treated as circular, truncates back to `n`, and recomputes Spearman correlation on
the resampled pairs. 2000 resamples, seed fixed, 95% interval from the 2.5th and 97.5th
percentiles.

The block length is the point. Inside a block the overlap structure is carried through
untouched; only the joins between blocks are artificial, and there are `n/H` of them rather
than `n`. Shorter blocks would break the dependence apart and reproduce the naive answer with
extra steps. Both standard errors are returned, `se_block_bootstrap` and `se_naive_iid`, and
`03_evaluate_models.py` prints them next to each other rather than picking one.

| model | rank IC | 95% CI (block) | block SE | naive SE | hit rate |
| --- | --- | --- | --- | --- | --- |
| Causal Transformer | +0.1876 | [+0.0603, +0.2954] | 0.0598 | 0.0198 | 55.90% |
| Mamba (bounded `Δ`) | +0.1413 | [−0.0247, +0.2904] | 0.0819 | 0.0198 | 56.33% |
| LSTM | +0.0717 | [−0.1168, +0.2542] | 0.0974 | 0.0198 | 49.53% |
| Ridge | +0.0176 | [−0.0608, +0.0914] | 0.0381 | 0.0198 | 51.52% |

Realised inflation runs from 1.9x to 4.9x, short of the full 10x the sample-size argument
implies but enough to change every conclusion. Read against the naive column, the top two
rank ICs clear seven standard errors. Read against the block column, one interval excludes
zero, and it is not the model this paper is about.

## Newey-West, and why the weights are not optional

`newey_west_lrv` estimates the long-run variance of a series with Bartlett weights:

```
lrv = g0 + 2 * sum_{k=1..q} (1 − k/(q+1)) * g_k
```

where `g_k` is the sample autocovariance at lag `k` and `q = H − 1 = 99`.

The weights are not a bandwidth refinement. Summing 99 raw autocovariances with weight one
gives an estimator that is not guaranteed non-negative, and on a short overlapping sample it
can come out below zero. A negative estimate then has to be floored, and the floor in this
file is `1e-30`. Dividing a nonzero mean loss differential by the square root of `1e-30 / n`
produces a statistic of order `1e14` and a p-value of zero, with nothing crashing on the way.
The Bartlett kernel is positive semi-definite, so the weighted sum is non-negative by
construction and the floor never binds. It stays in the code as a divide-by-zero guard, not
as a variance estimator.

## Diebold-Mariano

`diebold_mariano` compares two forecasts on squared-error loss:

```
d_k  = (y_k − a_k)^2 − (y_k − b_k)^2
stat = mean(d) / sqrt(lrv(d, lags = H − 1) / n)
```

then applies the Harvey-Leybourne-Newbold small-sample factor

```
stat *= sqrt((n + 1 − 2h + h(h−1)/n) / n)
```

skipping it if the factor comes out negative, and refers the result to Student-t with `n − 1`
degrees of freedom rather than to a normal. At `h = 100` the uncorrected statistic is biased
away from zero and the normal reference compounds it. A degenerate input, fewer than three
observations or an identically zero differential, returns `p = 1` instead of dividing by
nothing. Negative statistics favour model A.

| comparison | statistic | p |
| --- | --- | --- |
| Mamba vs Ridge | −11.44 | 0.000 |
| Transformer vs Ridge | −11.39 | 0.000 |
| LSTM vs Ridge | −10.62 | 0.000 |
| Transformer vs Mamba | −0.30 | 0.767 |
| Mamba vs LSTM | −0.10 | 0.923 |
| Transformer vs LSTM | −0.22 | 0.827 |

The three sequence models are indistinguishable from one another on this sample, and all
three beat the flattened-window ridge baseline on squared error by a margin the noise model
cannot explain away. That is the whole of what the test resolves at 25.6 independent windows.

## Selection is a separate trap

Loss is the wrong quantity to select on here, and the reason is not subtle.

| candidate | validation Huber | validation rank IC |
| --- | --- | --- |
| constant zero | 0.14979 | n/a |
| Causal Transformer | 0.15172 | +0.1632 |
| Mamba (bounded `Δ`) | 0.15272 | +0.1398 |
| LSTM | 0.15576 | +0.0869 |

A predictor that outputs zero for every window beats all three trained models on the
objective they were trained with. Forward returns at this horizon are almost entirely
unpredictable, so a point-forecast loss is dominated by the irreducible component, and any
model taking variance risk to capture ranking signal pays for that risk in MSE while
recovering very little of the level. Selecting on validation loss would prefer the model that
predicts nothing, every time.

`scripts/02_train_models.py` selects on validation rank IC and early-stops on it, with
`patience: 8` in `config.yaml` because rank IC is noisier than loss. Huber stays as the
training objective, since the gradient needs a smooth pointwise loss. The validation loss at
the selected step is recorded as `val_loss_at_best` and is not consulted when choosing
between checkpoints, and `selection_metric` records the choice alongside it.

## What is not reported

There is no annualised Sharpe ratio anywhere in this repository. Annualising a per-tick
signal whose forward windows overlap by 99 ticks with a factor of `√(252 · 23400)` produces a
number none of the assumptions behind the formula support; the observations are neither
independent nor uniformly spaced in time, and the horizon is event time, not clock time.

A per-decision information ratio is reported instead, in log-return units, with its standard
error taken from `n_eff` rather than `n`:

| model | mean PnL per decision | information ratio | SE |
| --- | --- | --- | --- |
| Mamba (bounded `Δ`) | 5.35e−06 | 0.125 | 0.198 |
| LSTM | 3.34e−06 | 0.078 | 0.198 |
| Causal Transformer | 2.85e−06 | 0.066 | 0.198 |
| Ridge | 4.48e−07 | 0.010 | 0.198 |

All four sit inside one standard error of zero. That is the correct reading of a 25.6-window
sample, and it is why the predictive results here are described as illustrative while the
certification results are not. None of the certificate depends on any of this; see
[04-stability.md](04-stability.md).

Label construction, the purge at each split boundary, and the enforced input box are in
[08-data.md](08-data.md). How to reproduce these tables is in [13-running.md](13-running.md).
