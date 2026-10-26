# Incremental certificates

Every certified number here is data-free. It bounds how far a block's contribution
before the residual, `out_proj(y * SiLU(gate))`, can move when its normalised input
moves by at most `delta` in sup-norm at every position, with both normalised inputs
LayerNorm outputs. Slopes are the small-`delta` limits of the bounds divided by `delta`.
The log-timescale certificate is reported with two time-warp terms: the cell theorem's
`2 M (e^rho - 1)`, and `2 M tanh(rho/2)`, the exact worst case when both runs start from
the same state, which they do here (`h_{-1} = 0`). To first order the second is half the
first.

## Block slopes

| arm | layer | log-timescale, state | log-timescale, output | log-timescale, common start, output | additive with floor, output | additive with floor / log-timescale | additive without floor passes at L | same, common start |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 5631.5735 | 4.307e+06 | 2.467e+06 | 9.521e+07 | 22.1053 | 8.0251 | 4.0126 |
| bounded | 1 | 5823.1448 | 4.27e+06 | 2.445e+06 | 9.438e+07 | 22.1030 | 8.1371 | 4.0686 |
| softplus | 0 | 5319.1150 | 3.914e+06 | 2.27e+06 | 6.955e+17 | 1.777e+11 | 0.2089 | 0.1045 |
| softplus | 1 | 5200.7033 | 3.775e+06 | 2.198e+06 | 2.585e+17 | 6.847e+10 | 0.2121 | 0.1060 |

The log-timescale columns hold at every sequence length. The additive bound without a
floor is affine in `L`, and the last two columns are the lengths at which its output
slope passes the log-timescale one, with the cell theorem and with a common start; below
them the additive bound is the tighter, and a value below 1 means it is looser at every
length.

Where the log-timescale output slope comes from, at its largest output coordinate, as
fractions of it:

| arm | layer | time-warp term | time-warp | forcing | C perturbation | skip path D | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | cell theorem | 0.8546 | 0.0724 | 0.0473 | 3.291e-05 | 0.0256 |
| bounded | 0 | common start | 0.7462 | 0.1263 | 0.0827 | 5.746e-05 | 0.0448 |
| bounded | 1 | cell theorem | 0.8547 | 0.0724 | 0.0464 | 3.235e-05 | 0.0265 |
| bounded | 1 | common start | 0.7462 | 0.1264 | 0.0811 | 5.649e-05 | 0.0462 |
| softplus | 0 | cell theorem | 0.8399 | 0.0797 | 0.0522 | 3.621e-05 | 0.0282 |
| softplus | 0 | common start | 0.7240 | 0.1374 | 0.0899 | 6.243e-05 | 0.0487 |
| softplus | 1 | cell theorem | 0.8356 | 0.0819 | 0.0525 | 3.66e-05 | 0.0299 |
| softplus | 1 | common start | 0.7177 | 0.1406 | 0.0902 | 6.286e-05 | 0.0514 |

Time-warp is the term `2 M (e^rho - 1)` (or `2 M tanh(rho/2)`) read out through `C`;
forcing is `sup|c - c~|` read out the same way; C perturbation is `M L_C`; the gate term
is `sup|y|` times the gate's own sensitivity.

## Against sequence length

bounded arm, output slope:

| layer | L | log-timescale | additive, no floor | additive, certified floor |
| --- | --- | --- | --- | --- |
| 0 | 10 | 4.307e+06 | 5.213e+06 | 5.007e+06 |
| 0 | 100 | 4.307e+06 | 4.65e+07 | 3.231e+07 |
| 0 | 1000 | 4.307e+06 | 4.593e+08 | 8.325e+07 |
| 0 | 5000 | 4.307e+06 | 2.294e+09 | 9.504e+07 |
| 0 | limit | 4.307e+06 | inf | 9.521e+07 |
| 1 | 10 | 4.27e+06 | 5.106e+06 | 4.903e+06 |
| 1 | 100 | 4.27e+06 | 4.547e+07 | 3.154e+07 |
| 1 | 1000 | 4.27e+06 | 4.491e+08 | 8.188e+07 |
| 1 | 5000 | 4.27e+06 | 2.243e+09 | 9.42e+07 |
| 1 | limit | 4.27e+06 | inf | 9.438e+07 |

softplus arm, output slope:

| layer | L | log-timescale | additive, no floor | additive, certified floor |
| --- | --- | --- | --- | --- |
| 0 | 10 | 3.914e+06 | 1.58e+08 | 1.58e+08 |
| 0 | 100 | 3.914e+06 | 1.574e+09 | 1.573e+09 |
| 0 | 1000 | 3.914e+06 | 1.574e+10 | 1.563e+10 |
| 0 | 5000 | 3.914e+06 | 7.868e+10 | 7.727e+10 |
| 0 | limit | 3.914e+06 | inf | 6.955e+17 |
| 1 | 10 | 3.775e+06 | 1.493e+08 | 1.493e+08 |
| 1 | 100 | 3.775e+06 | 1.488e+09 | 1.486e+09 |
| 1 | 1000 | 3.775e+06 | 1.487e+10 | 1.476e+10 |
| 1 | 5000 | 3.775e+06 | 7.436e+10 | 7.287e+10 |
| 1 | limit | 3.775e+06 | inf | 2.585e+17 |

In the additive bound with the certified floor, the timescale term is its one-step
cost times `sum_{k<L} exp(-k lam dt_min)`, which saturates at `1/(1 - exp(-lam dt_min))`.
On the softplus arm the certified floor is below `5e-13`, so it does not saturate
within these lengths. The additive bound keeps the factor `exp(-lam dt_min)` in its
one-step cost, which Cao et al.'s Lemma B.3 drops; their Remark 3.4 constant is larger
by `e^(lam dt_min)`, which at the certified floors is at most 1.0195 on the bounded arm and 1.0048 on the softplus arm.

## Per cell

The per-cell comparison is between the two time-warp terms, to first order in `delta`,
for each (channel, state) cell:

```
ratio = [2 M lam L_delta / (exp(lam dt_min) - 1)] / [2 M L_log_delta]
      = [L_delta / (dt_min L_log_delta)] * x / (e^x - 1),     x = lam dt_min
```

The numerator is the additive time-warp term with the certified floor `dt_min`, at its
limit in the length, with the factor `exp(-x)` kept; the denominator is the cell
theorem's term. The forcing term is the same in both and is left out, and `M` and the
Lipschitz constant of the pre-activation cancel, so the ratio depends on the
parametrisation, on the certified box of the channel's pre-activation and on `x` only.
`L_delta` and `L_log_delta` are `sup|dt'|` and `sup|(log dt)'|` over that box times the
same `L_z`. Cao et al.'s Remark 3.4 form drops `exp(-x)` and is larger by `e^x`.
Against the common-start term the ratio doubles.

| arm | layer | certified floor | x min | x max | ratio min | ratio median | ratio max | weight-free factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 0.0010 | 0.0009732 | 0.0193 | 21.3832 | 25.6753 | 25.7874 | 25.7999 |
| bounded | 1 | 0.0010 | 0.0009594 | 0.0183 | 22.3683 | 25.6822 | 25.7875 | 25.7999 |
| softplus | 0 | 2.287e-13 | 2.312e-13 | 0.0037 | 4264.8707 | 4.955e+07 | 4.372e+12 | - |
| softplus | 1 | 4.158e-13 | 4.133e-13 | 0.0047 | 1744.7053 | 7.782e+07 | 2.405e+12 | - |

The weight-free factor is the first factor with the slopes taken over all `z`:
`sup|dt'| / (dt_min sup|(log dt)'|)`, which for the bounded parametrisation depends on
`dt_max / dt_min` only. Over the certified boxes the first factor lies between 21.5909 and 25.7999 (bounded layer 0); 22.5743 and 25.7999 (bounded layer 1); 4272.7743 and 4.372e+12 (softplus layer 0); 1748.8528 and 2.405e+12 (softplus layer 1), and `x / (e^x - 1) <= 1`, so the ratio is at most the first factor of its channel.
The softplus parametrisation has no positive `dt_min`; there the certified floor plays
its part, and the first factor is about `1 / dt_min`.

Without a floor the additive time-warp term after `n` steps is `2 M lam L_delta n`,
while the cell theorem's, in its finite-horizon form with the worst case over
admissible timescales, is `2 M L_log_delta (1 - exp(-lam dt_hi n))`, `dt_hi` the
certified ceiling. The crossover horizon `n*` is the smallest `n >= 1` at which the
first is at least the second, to first order in `delta`; for fewer steps the additive
term is the smaller.

| arm | layer | n* min | n* median | n* max | additive / ours after one step, min | certified ceiling, max |
| --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 3 | 5 | 40 | 0.2707 | 0.1000 |
| bounded | 1 | 3 | 5 | 40 | 0.2706 | 0.1000 |
| softplus | 0 | 1 | 1 | 2 | 0.9653 | 19.8025 |
| softplus | 1 | 1 | 1 | 2 | 0.9594 | 17.9400 |

At the attacked floor:

| arm | layer | attacked floor | x max | ratio min | ratio median | ratio max | fraction of cells with ratio below 1 | additive / log-timescale at the block output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 0.0011 | 1.2870 | 0.1558 | 8.4911 | 23.9809 | 0.1782 | 9.8369 |
| bounded | 1 | 0.0011 | 1.3314 | 0.1483 | 6.4848 | 24.5252 | 0.1460 | 7.9539 |
| softplus | 0 | 0.0002686 | 0.8627 | 11.6634 | 329.8177 | 3722.5641 | 0.0000 | 694.5207 |
| softplus | 1 | 0.0003222 | 1.0473 | 8.5608 | 209.1829 | 3102.6654 | 0.0000 | 482.2043 |

The ratio here is the same per-cell ratio with `dt_min` set to the attacked floor of
the channel. The attacked floor is the smallest `delta` per channel that gradient descent
on the block's raw input reaches, so any valid floor is at most that; the additive term
decreases in the floor, so its ratio there is a lower bound on what the additive rule
could give with an exact floor. A cell with ratio below 1 there is one where an exact
per-channel floor could make the additive term the smaller one. They sit in channels the attack could not push down to the certified floor: bounded layer 0, attacked floor at least 0.0225; bounded layer 1, attacked floor at least 0.023.
The last column is the block-level comparison: the floored additive output slope, with
the forcing included, at the attacked floor, over the log-timescale output slope. The
floor columns show the minimum over channels. On the bounded arm the certified floor is
`dt_min`, which the clamp in `delta_from_pre` makes a hard lower bound, and the attack
comes within 7.5 percent of it in the lowest channel. No channel in any block was pushed below its certified floor.

## Second order

| arm | layer | delta | both LayerNorm outputs | perturbed input arbitrary | common start, both LayerNorm outputs | common start, perturbed input arbitrary |
| --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 1e-06 | 4.3071 | 4.3083 | 2.4666 | 2.4672 |
| bounded | 0 | 0.0001 | 431.0224 | 431.3483 | 246.6570 | 246.8314 |
| bounded | 0 | 0.0100 | 4.639e+04 | 4.873e+04 | 2.462e+04 | 2.576e+04 |
| bounded | 1 | 1e-06 | 4.2701 | 4.2704 | 2.4453 | 2.4455 |
| bounded | 1 | 0.0001 | 427.3132 | 427.5370 | 244.5324 | 244.6512 |
| bounded | 1 | 0.0100 | 4.59e+04 | 4.813e+04 | 2.441e+04 | 2.55e+04 |
| softplus | 0 | 1e-06 | 3.9142 | 3.9152 | 2.2705 | 2.2710 |
| softplus | 0 | 0.0001 | 391.6693 | 391.9585 | 227.0450 | 227.2012 |
| softplus | 0 | 0.0100 | 4.177e+04 | 4.385e+04 | 2.267e+04 | 2.37e+04 |
| softplus | 1 | 1e-06 | 3.7746 | 3.7749 | 2.1975 | 2.1977 |
| softplus | 1 | 0.0001 | 377.6880 | 377.8845 | 219.7548 | 219.8601 |
| softplus | 1 | 0.0100 | 4.012e+04 | 4.205e+04 | 2.195e+04 | 2.291e+04 |

Maximum over output coordinates of the bound at `delta`, not its slope. When the
perturbed normalised input need not be a LayerNorm output, the perturbed trajectory can
leave the certified boxes, so they are widened by `L delta` and derivative suprema are
taken globally. With the cell theorem the second reading is larger by at most 0.027 percent at `delta = 1e-06` and 5 percent at `delta = 0.01`.

## How loose the data-free bound is

| arm | layer | u | log delta | B | C | c | h | h, common start | y | output | output, common start |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 0.1186 | 0.0078 | 0.0079 | 0.0087 | 0.0015 | 7.239e-05 | 0.0001663 | 5.191e-06 | 3.752e-08 | 9.117e-08 |
| bounded | 1 | 0.1141 | 0.0090 | 0.0092 | 0.0092 | 0.0015 | 9.846e-05 | 0.0001732 | 7.377e-06 | 4.07e-08 | 7.692e-08 |
| softplus | 0 | 0.1186 | 0.0080 | 0.0079 | 0.0087 | 0.0015 | 8.284e-05 | 0.0001779 | 5.93e-06 | 4.127e-08 | 9.5e-08 |
| softplus | 1 | 0.1141 | 0.0090 | 0.0092 | 0.0092 | 0.0015 | 0.000122 | 0.0002108 | 8.557e-06 | 4.618e-08 | 8.158e-08 |

Each entry is the largest realised difference over its certified bound, sampled over
random inputs and random perturbations of the raw block input. The ratio falls at
every stage from `u` to the output, because each stage applies worst-case row sums of
absolute weights or the certified boxes to the stage before it, and the losses
multiply. Sampling does not search for the worst case, so the certificate is at most
the reciprocal of each entry times the true sensitivity, and may be much less.

## LayerNorm and the exponential-Euler gain

| arm | block | max abs gamma | eps | l2 bound, any input | l2 at a constant input | sup-norm at a constant input | exponential-Euler gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 1.0338 | 1e-05 | 326.9024 | 326.8111 | 643.5891 | 2.0681 |
| bounded | 1 | 1.0247 | 1e-05 | 324.0503 | 323.9675 | 637.9740 | 2.0462 |
| softplus | 0 | 1.0338 | 1e-05 | 326.9035 | 326.8137 | 643.5912 | 319.0057 |
| softplus | 1 | 1.0248 | 1e-05 | 324.0549 | 323.9719 | 637.9831 | 288.8383 |

With `eps > 0` the block's LayerNorm is globally Lipschitz. Its Jacobian is
`diag(gamma)(I - 1 1^T/d - z z^T/d)/s` with `s = sqrt(var + eps)`, whose l2 norm is at
most `max|gamma|/s <= max|gamma|/sqrt(eps)`. The values at a constant input, where the
variance is 0, are attained, so they bound the global constants from below, in l2 and in
the sup-norm that the block certificate uses. A data-free statement in the raw block
input therefore exists, with these factors; a local one needs a lower bound on the
spread `s`. The last column is `g(lam dt_hi) = lam dt_hi / (1 - exp(-lam dt_hi))`,
maximised over cells with the certified ceiling `dt_hi`, the factor by which
exponential-Euler (`Bbar = dt B`) enlarges the equilibrium; this repository's block uses
the exact zero-order hold.

## On held-out windows

8 windows of length 1000 from the test split, rows 3867904 to 4550475, ending at rows 3868903, 3966270, 4063638, 4161005, 4258373, 4355740, 4453108, 4550475. Each block is run in float64 from `h_{-1} = 0`.

| arm | block | realised distance, max over cells | median over cells | 2M, max over cells | 2M, median | largest distance / 2M in one cell |
| --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 0.8715 | 0.0208 | 259.3478 | 17.7214 | 0.0060 |
| bounded | 1 | 0.4324 | 0.0175 | 287.0885 | 16.7325 | 0.0052 |
| softplus | 0 | 0.8819 | 0.0207 | 259.4336 | 17.7460 | 0.0058 |
| softplus | 1 | 0.4413 | 0.0175 | 286.7957 | 16.7305 | 0.0053 |

The realised distance of a cell is `max_t |h_{t-1} - c_t|` over the windows and
positions, with `c_t = B_t u_t / lam`. The local corollary uses it where the cell
theorem has the worst case `2 M`.

| arm | block | positions | spread s, min | spread s, max | l2 norm of the Jacobian, max | max|gamma| / s, max | sup-norm of the Jacobian, max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 8000 | 0.9943 | 1.0158 | 1.0392 | 1.0396 | 3.9306 |
| bounded | 1 | 8000 | 0.9797 | 1.1662 | 1.0456 | 1.0460 | 3.8205 |
| softplus | 0 | 8000 | 0.9943 | 1.0158 | 1.0392 | 1.0396 | 3.9307 |
| softplus | 1 | 8000 | 0.9797 | 1.1658 | 1.0456 | 1.0460 | 3.8205 |

LayerNorm's Jacobian at the realised input of each block, over every position of
the windows: about 1 in l2, against the global constants above.

## The local domain with each recurrence rule

Median certified output width over 4 held-out windows of length 100, perturbing the network input by `eps`, with the norm-propagation domain of `src/verification/lipschitz.py` and its recurrence step replaced. The additive rule reproduces that domain exactly. The running maximum is the local corollary, `E_t = max(E_{t-1}, dc_t + (e^rho_t - 1)|h_{t-1} - c_t|)`. The recursive rule is the one-step inequality in its proof, `E_t = K_t + a*_t (E_{t-1} - K_t)` with `K_t` the second argument of that maximum and `a*_t` the perturbed transition at its worst value in `[exp(-lam dt e^rho), exp(-lam dt e^-rho)]`; it is never larger than the running maximum, and with `rho = 0` it is the exact recursion of the forcing term.

| arm | eps | additive | log-timescale, running maximum | log-timescale, recursive |
| --- | --- | --- | --- | --- |
| bounded | 1e-07 | 0.0487 | 0.0421 | 0.0222 |
| bounded | 1e-06 | 2.1798 | 0.7063 | 0.2877 |
| bounded | 1e-05 | inf | 3.057e+04 | 61.0876 |
| bounded | 0.0001 | inf | inf | 2.851e+25 |
| bounded | 0.0010 | inf | inf | inf |
| bounded | 0.0100 | inf | inf | inf |
| bounded | 0.1000 | inf | inf | inf |
| softplus | 1e-07 | 0.0239 | 0.0436 | 0.0223 |
| softplus | 1e-06 | 0.3228 | 0.7457 | 0.2891 |
| softplus | 1e-05 | 953.4683 | 5.481e+04 | 62.3656 |
| softplus | 0.0001 | inf | inf | 7.594e+23 |
| softplus | 0.0010 | inf | inf | inf |
| softplus | 0.0100 | inf | inf | inf |
| softplus | 0.1000 | inf | inf | inf |

Per block, the local Lipschitz bound on the block's contribution around the realised normalised input, for any perturbation of it of sup-norm `eps`, against the largest sampled ratio at `eps = 1e-4`:

| arm | layer | sampled | additive @ 1e-06 | maximum @ 1e-06 | recursive @ 1e-06 | additive @ 0.0001 | maximum @ 0.0001 | recursive @ 0.0001 | additive @ 0.01 | maximum @ 0.01 | recursive @ 0.01 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 0.5687 | 24.3412 | 23.1479 | 14.8148 | 24.8506 | 23.3092 | 14.8771 | 4.627e+15 | 44.0650 | 23.1430 |
| bounded | 1 | 0.3969 | 16.5823 | 14.7667 | 11.0549 | 16.9950 | 14.8771 | 11.0998 | 3.57e+16 | 29.2660 | 17.3475 |
| softplus | 0 | 0.5661 | 15.3651 | 24.0581 | 14.8134 | 15.4457 | 24.2298 | 14.8764 | 29.5975 | 46.2404 | 23.2895 |
| softplus | 1 | 0.3971 | 11.3067 | 15.0061 | 11.0811 | 11.3609 | 15.1191 | 11.1264 | 23.3758 | 29.7959 | 17.3844 |

## Scope

The bounds hold over the reals, not in floating point. They are per block, with respect
to the normalised input. A statement in the raw block input passes through the
LayerNorm, which is globally Lipschitz only through `eps`, with the constants above.
The same per-cell bound follows from Cao et al. (ICML 2026), Lemma 3.1 and Lemma B.2,
with a piecewise-linear time change, and it strengthens their Theorem 3.3(ii). The
additive baseline mirrors the structure of their Theorem 3.3(ii) and Remark 3.4, applied
per cell over the certified boxes, with the factor their Lemma B.3 drops kept. The
induction is Horbatko's Lemma 4.4 (arXiv:2604.18580) applied to the error.
