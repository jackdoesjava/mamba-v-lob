# Incremental certificates

Every certified number here is data-free. It bounds how far a block's contribution
before the residual, `out_proj(y * SiLU(gate))`, can move when its normalised input
moves by at most `delta` in sup-norm at every position, with both normalised inputs
LayerNorm outputs. Slopes are the small-`delta` limits of the bounds divided by `delta`.
The log-timescale certificate is reported with two time-warp terms: the cell theorem's
`2 M (e^rho - 1)`, and `2 M tanh(rho/2)`, a sharper constant for two runs that start from
the same state, which they do here (`h_{-1} = 0`). To first order the second is half the
first. The paper states only the first.

## Block slopes

| arm | layer | log-timescale, state | log-timescale, output | log-timescale, common start, output | additive with floor, output | additive with floor / log-timescale | additive without floor passes at L | same, common start |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 8774.7255 | 9.336e+06 | 5.17e+06 | 2.154e+08 | 23.0734 | 8.2758 | 4.1379 |
| bounded | 1 | 7027.4895 | 6.072e+06 | 3.437e+06 | 1.363e+08 | 22.4477 | 8.0620 | 4.0310 |
| softplus | 0 | 1.562e+04 | 1.843e+07 | 1.02e+07 | 3.925e+26 | 2.13e+19 | 0.2293 | 0.1146 |
| softplus | 1 | 1.698e+04 | 1.457e+07 | 8.074e+06 | 4.104e+22 | 2.818e+15 | 0.2263 | 0.1131 |

The log-timescale columns hold at every sequence length. The additive bound without a
floor is affine in `L`, and the last two columns are the lengths at which its output
slope passes the log-timescale one, with the cell theorem and with a common start; below
them the additive bound is the tighter, and a value below 1 means it is looser at every
length.

Where the log-timescale output slope comes from, at its largest output coordinate, as
fractions of it:

| arm | layer | time-warp term | time-warp | forcing | C perturbation | skip path D | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | cell theorem | 0.8924 | 0.0539 | 0.0347 | 1.815e-05 | 0.0190 |
| bounded | 0 | common start | 0.8057 | 0.0972 | 0.0627 | 3.278e-05 | 0.0343 |
| bounded | 1 | cell theorem | 0.8678 | 0.0660 | 0.0422 | 2.587e-05 | 0.0240 |
| bounded | 1 | common start | 0.7665 | 0.1166 | 0.0746 | 4.57e-05 | 0.0423 |
| softplus | 0 | cell theorem | 0.8928 | 0.0535 | 0.0349 | 1.15e-05 | 0.0188 |
| softplus | 0 | common start | 0.8063 | 0.0966 | 0.0630 | 2.078e-05 | 0.0340 |
| softplus | 1 | cell theorem | 0.8914 | 0.0540 | 0.0348 | 1.287e-05 | 0.0198 |
| softplus | 1 | common start | 0.8040 | 0.0975 | 0.0628 | 2.323e-05 | 0.0356 |

Time-warp is the term `2 M (e^rho - 1)` (or `2 M tanh(rho/2)`) read out through `C`;
forcing is `sup|c - c~|` read out the same way; C perturbation is `M L_C`; the gate term
is `sup|y|` times the gate's own sensitivity.

## Against sequence length

bounded arm, output slope:

| layer | L | log-timescale | additive, no floor | additive, certified floor |
| --- | --- | --- | --- | --- |
| 0 | 10 | 9.336e+06 | 1.107e+07 | 1.063e+07 |
| 0 | 100 | 9.336e+06 | 1.017e+08 | 7.117e+07 |
| 0 | 1000 | 9.336e+06 | 1.008e+09 | 1.873e+08 |
| 0 | 5000 | 9.336e+06 | 5.034e+09 | 2.15e+08 |
| 0 | limit | 9.336e+06 | inf | 2.154e+08 |
| 1 | 10 | 6.072e+06 | 7.338e+06 | 7.039e+06 |
| 1 | 100 | 6.072e+06 | 6.616e+07 | 4.564e+07 |
| 1 | 1000 | 6.072e+06 | 6.544e+08 | 1.178e+08 |
| 1 | 5000 | 6.072e+06 | 3.269e+09 | 1.36e+08 |
| 1 | limit | 6.072e+06 | inf | 1.363e+08 |

softplus arm, output slope:

| layer | L | log-timescale | additive, no floor | additive, certified floor |
| --- | --- | --- | --- | --- |
| 0 | 10 | 1.843e+07 | 7.196e+08 | 7.196e+08 |
| 0 | 100 | 1.843e+07 | 7.178e+09 | 7.178e+09 |
| 0 | 1000 | 1.843e+07 | 7.176e+10 | 7.176e+10 |
| 0 | 5000 | 1.843e+07 | 3.588e+11 | 3.586e+11 |
| 0 | limit | 1.843e+07 | inf | 3.925e+26 |
| 1 | 10 | 1.457e+07 | 5.753e+08 | 5.753e+08 |
| 1 | 100 | 1.457e+07 | 5.739e+09 | 5.738e+09 |
| 1 | 1000 | 1.457e+07 | 5.738e+10 | 5.725e+10 |
| 1 | 5000 | 1.457e+07 | 2.869e+11 | 2.851e+11 |
| 1 | limit | 1.457e+07 | inf | 4.104e+22 |

In the additive bound with the certified floor, the timescale term is its one-step
cost times `sum_{k<L} exp(-k lam dt_min)`, which saturates at `1/(1 - exp(-lam dt_min))`.
On the softplus arm the certified floor is below `5e-13`, so it does not saturate
within these lengths. The additive bound keeps the factor `exp(-lam dt_min)` in its
one-step cost, which Cao et al.'s Lemma B.3 drops; their Remark 3.4 constant is larger
by `e^(lam dt_min)`, which at the certified floors is at most 1.0173 on the bounded arm and 1.0011 on the softplus arm.

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
| bounded | 0 | 0.0010 | 0.0009121 | 0.0171 | 25.0900 | 25.6879 | 25.7882 | 25.7999 |
| bounded | 1 | 0.0010 | 0.0009135 | 0.0172 | 24.1183 | 25.6836 | 25.7882 | 25.7999 |
| softplus | 0 | 1.265e-21 | 1.298e-21 | 0.0001206 | 1.282e+05 | 9.064e+10 | 7.902e+20 | - |
| softplus | 1 | 1.867e-17 | 2.024e-17 | 0.0011 | 1.304e+04 | 3.955e+10 | 5.358e+16 | - |

The weight-free factor is the first factor with the slopes taken over all `z`:
`sup|dt'| / (dt_min sup|(log dt)'|)`, which for the bounded parametrisation depends on
`dt_max / dt_min` only. Over the certified boxes the first factor lies between 25.2939 and 25.7999 (bounded layer 0); 24.3156 and 25.7999 (bounded layer 1); 1.282e+05 and 7.902e+20 (softplus layer 0); 1.305e+04 and 5.358e+16 (softplus layer 1), and `x / (e^x - 1) <= 1`, so the ratio is at most the first factor of its channel.
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
| bounded | 0 | 3 | 5 | 42 | 0.2699 | 0.1000 |
| bounded | 1 | 3 | 5 | 42 | 0.2700 | 0.1000 |
| softplus | 0 | 1 | 1 | 2 | 0.8026 | 32.5263 |
| softplus | 1 | 1 | 1 | 2 | 0.8617 | 29.2931 |

At the attacked floor:

| arm | layer | attacked floor | x max | ratio min | ratio median | ratio max | fraction of cells with ratio below 1 | additive / log-timescale at the block output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 0.0010 | 1.3258 | 0.1565 | 11.4337 | 25.4188 | 0.1025 | 12.3136 |
| bounded | 1 | 0.0011 | 1.2378 | 0.1697 | 7.2007 | 24.5498 | 0.1196 | 8.5440 |
| softplus | 0 | 8.361e-06 | 0.4995 | 24.4576 | 704.8776 | 1.196e+05 | 0.0000 | 6280.3297 |
| softplus | 1 | 0.0001003 | 0.4618 | 26.7074 | 507.6782 | 9972.4859 | 0.0000 | 1222.5962 |

The ratio here is the same per-cell ratio with `dt_min` set to the attacked floor of
the channel. The attacked floor is the smallest `delta` per channel that gradient descent
on the block's raw input reaches, so any valid floor is at most that; the additive term
decreases in the floor, so its ratio there is a lower bound on what the additive rule
could give with an exact floor. A cell with ratio below 1 there is one where an exact
per-channel floor could make the additive term the smaller one. They sit in channels the attack could not push down to the certified floor: bounded layer 0, attacked floor at least 0.0221; bounded layer 1, attacked floor at least 0.0225.
The last column is the block-level comparison: the floored additive output slope, with
the forcing included, at the attacked floor, over the log-timescale output slope. The
floor columns show the minimum over channels. On the bounded arm the certified floor is
`dt_min`, which the clamp in `delta_from_pre` makes a hard lower bound, and the attack
comes within 5.0 percent of it in the lowest channel. No channel in any block was pushed below its certified floor.

## Second order

| arm | layer | delta | both LayerNorm outputs | perturbed input arbitrary | common start, both LayerNorm outputs | common start, perturbed input arbitrary |
| --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 1e-06 | 9.3358 | 9.3383 | 5.1702 | 5.1715 |
| bounded | 0 | 0.0001 | 934.5739 | 935.2776 | 517.0234 | 517.3911 |
| bounded | 0 | 0.0100 | 1.043e+05 | 1.096e+05 | 5.149e+04 | 5.394e+04 |
| bounded | 1 | 1e-06 | 6.0717 | 6.0725 | 3.4372 | 3.4376 |
| bounded | 1 | 0.0001 | 607.6545 | 608.0122 | 343.7209 | 343.9096 |
| bounded | 1 | 0.0100 | 6.589e+04 | 6.912e+04 | 3.429e+04 | 3.585e+04 |
| softplus | 0 | 1e-06 | 18.4302 | 18.4339 | 10.2030 | 10.2049 |
| softplus | 0 | 0.0001 | 1844.9825 | 1846.2462 | 1020.2952 | 1020.9576 |
| softplus | 0 | 0.0100 | 2.06e+05 | 2.163e+05 | 1.016e+05 | 1.064e+05 |
| softplus | 1 | 1e-06 | 14.5659 | 14.5676 | 8.0740 | 8.0749 |
| softplus | 1 | 0.0001 | 1458.0904 | 1458.9559 | 807.4004 | 807.8513 |
| softplus | 1 | 0.0100 | 1.622e+05 | 1.702e+05 | 8.042e+04 | 8.411e+04 |

Maximum over output coordinates of the bound at `delta`, not its slope. When the
perturbed normalised input need not be a LayerNorm output, the perturbed trajectory can
leave the certified boxes, so they are widened by `L delta` and derivative suprema are
taken globally. With the cell theorem the second reading is larger by at most 0.026 percent at `delta = 1e-06` and 5.1 percent at `delta = 0.01`.

## How loose the data-free bound is

| arm | layer | u | log delta | B | C | c | h | h, common start | y | output | output, common start |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 0.1223 | 0.0085 | 0.0075 | 0.0089 | 0.0019 | 6.508e-05 | 0.0001316 | 3.087e-06 | 2.153e-08 | 6.491e-08 |
| bounded | 1 | 0.1188 | 0.0093 | 0.0092 | 0.0090 | 0.0013 | 7.358e-05 | 0.0001362 | 5.425e-06 | 3.679e-08 | 6.697e-08 |
| softplus | 0 | 0.1255 | 0.0094 | 0.0088 | 0.0090 | 0.0022 | 5.664e-05 | 0.0001767 | 1.98e-06 | 2.027e-08 | 4.137e-08 |
| softplus | 1 | 0.1214 | 0.0081 | 0.0095 | 0.0086 | 0.0018 | 6.768e-05 | 0.0001802 | 2.981e-06 | 2.245e-08 | 4.917e-08 |

Each entry is the largest realised difference over its certified bound, sampled over
random inputs and random perturbations of the raw block input. The ratio falls at
every stage from `u` to the output, because each stage applies worst-case row sums of
absolute weights or the certified boxes to the stage before it, and the losses
multiply. Sampling does not search for the worst case, so the certificate is at most
the reciprocal of each entry times the true sensitivity, and may be much less.

## LayerNorm and the exponential-Euler gain

| arm | block | max abs gamma | eps | l2 bound, any input | l2 at a constant input | sup-norm at a constant input | exponential-Euler gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 1.0484 | 1e-05 | 331.5441 | 331.3924 | 652.7275 | 2.0902 |
| bounded | 1 | 1.0467 | 1e-05 | 330.9961 | 330.8781 | 651.6486 | 2.0859 |
| softplus | 0 | 1.0852 | 1e-05 | 343.1624 | 342.8449 | 675.6009 | 537.4571 |
| softplus | 1 | 1.0571 | 1e-05 | 334.2735 | 334.1287 | 658.1009 | 484.7989 |

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
| bounded | 0 | 2.9017 | 0.0314 | 351.8950 | 20.4778 | 0.0104 |
| bounded | 1 | 0.7882 | 0.0183 | 274.4366 | 18.2590 | 0.0089 |
| softplus | 0 | 11.0413 | 0.0540 | 574.7738 | 26.0247 | 0.0244 |
| softplus | 1 | 4.2600 | 0.0315 | 417.3579 | 24.8642 | 0.0162 |

The realised distance of a cell is `max_t |h_{t-1} - c_t|` over the windows and
positions, with `c_t = B_t u_t / lam`. The local corollary uses it where the cell
theorem has the worst case `2 M`.

| arm | block | positions | spread s, min | spread s, max | l2 norm of the Jacobian, max | max|gamma| / s, max | sup-norm of the Jacobian, max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 8000 | 0.9999 | 1.0513 | 1.0477 | 1.0486 | 3.8453 |
| bounded | 1 | 8000 | 0.9982 | 1.3493 | 1.0477 | 1.0486 | 3.5566 |
| softplus | 0 | 8000 | 0.9906 | 1.0713 | 1.0939 | 1.0954 | 3.8435 |
| softplus | 1 | 8000 | 1.0129 | 2.4249 | 1.0430 | 1.0436 | 3.2816 |

LayerNorm's Jacobian at the realised input of each block, over every position of
the windows: about 1 in l2, against the global constants above.

## The local domain with each recurrence rule

Median certified output width over 4 held-out windows of length 100, perturbing the network input by `eps`, with the norm-propagation domain of `src/verification/lipschitz.py` and its recurrence step replaced. The additive rule is that domain with the timescale's slope taken over the local box, the same as `lipschitz._delta` on softplus and never larger on the bounded map. The running maximum is the local corollary, `E_t = max(E_{t-1}, dc_t + (e^rho_t - 1)|h_{t-1} - c_t|)`. The recursive rule is the one-step inequality in its proof, `E_t = K_t + a*_t (E_{t-1} - K_t)` with `K_t` the second argument of that maximum and `a*_t` the perturbed transition at its worst value in `[exp(-lam dt e^rho), exp(-lam dt e^-rho)]`; it is never larger than the running maximum, and with `rho = 0` it is the exact recursion of the forcing term. The endpoint rule, the reported one, bounds the error identity itself, `|e_t| <= s E_{t-1} + (1 - s) dc_t + |Abar_t - s| |h_{t-1} - c_t|` with `s` the perturbed transition, which is convex in `s`, and takes its larger value at the two ends of the interval that holds `s`: the intersection of the interval above with the one the additive bound on `|dt - dt~|` gives. It is never larger than the recursive rule or the additive rule, and it is finite for every `rho`. The additive interval, endpoint rule takes the same maximum over the additive interval alone; it lies between the endpoint rule and the additive rule and isolates what `rho` adds.

| arm | eps | additive | log-timescale, running maximum | log-timescale, recursive | additive interval, endpoint | log-timescale, endpoint |
| --- | --- | --- | --- | --- | --- | --- |
| bounded | 1e-07 | 0.0490 | 0.1316 | 0.0409 | 0.0408 | 0.0408 |
| bounded | 1e-06 | 1.2915 | 10.1726 | 0.8564 | 0.8514 | 0.8503 |
| bounded | 1e-05 | inf | 5.383e+14 | 6.359e+10 | 6.43e+10 | 5.525e+10 |
| bounded | 0.0001 | inf | inf | inf | 8.1e+28 | 7.455e+28 |
| bounded | 0.0010 | inf | inf | inf | inf | inf |
| bounded | 0.0100 | inf | inf | inf | inf | inf |
| bounded | 0.1000 | inf | inf | inf | inf | inf |
| softplus | 1e-07 | 0.7646 | 11.3295 | 0.3306 | 0.3276 | 0.3267 |
| softplus | 1e-06 | 1.517e+12 | 3.75e+12 | 67.6375 | 68.4174 | 61.1721 |
| softplus | 1e-05 | inf | inf | 3.368e+18 | 2.333e+17 | 1.668e+17 |
| softplus | 0.0001 | inf | inf | inf | inf | inf |
| softplus | 0.0010 | inf | inf | inf | inf | inf |
| softplus | 0.0100 | inf | inf | inf | inf | inf |
| softplus | 0.1000 | inf | inf | inf | inf | inf |

Per block, the local Lipschitz bound on the block's contribution around the realised normalised input, for any perturbation of it of sup-norm `eps`, against the largest sampled ratio at `eps = 1e-4`:

| arm | layer | sampled | additive @ 1e-06 | maximum @ 1e-06 | recursive @ 1e-06 | additive endpoint @ 1e-06 | endpoint @ 1e-06 | additive @ 0.0001 | maximum @ 0.0001 | recursive @ 0.0001 | additive endpoint @ 0.0001 | endpoint @ 0.0001 | additive @ 0.01 | maximum @ 0.01 | recursive @ 0.01 | additive endpoint @ 0.01 | endpoint @ 0.01 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 1.0634 | 53.5224 | 91.2983 | 46.8824 | 46.8002 | 46.8002 | 53.8259 | 91.8144 | 47.0774 | 46.9930 | 46.9913 | 105.1590 | 156.6958 | 72.0186 | 71.6299 | 71.1441 |
| bounded | 1 | 1.0443 | 20.6427 | 30.3014 | 20.0030 | 19.9889 | 19.9888 | 20.7559 | 30.5293 | 20.0947 | 20.0801 | 20.0799 | 39.9342 | 59.2551 | 32.2204 | 32.1173 | 32.0478 |
| softplus | 0 | 3.1906 | 221.1907 | 505.3958 | 168.9724 | 168.2226 | 168.2221 | 222.7077 | 507.5770 | 169.6445 | 168.8898 | 168.8430 | 574.4966 | 769.7997 | 254.5843 | 256.7469 | 244.9514 |
| softplus | 1 | 1.2526 | 90.4989 | 174.4459 | 73.0029 | 72.8103 | 72.8102 | 91.1185 | 175.2929 | 73.3266 | 73.1282 | 73.1183 | 219.1212 | 278.0274 | 114.7608 | 114.1230 | 111.7384 |

## Scope

The bounds hold over the reals, not in floating point. They are per block, with respect
to the normalised input. A statement in the raw block input passes through the
LayerNorm, which is globally Lipschitz only through `eps`, with the constants above.
The same per-cell bound follows from Cao et al. (ICML 2026), Lemma 3.1 and Lemma B.2,
with a piecewise-linear time change, and it strengthens their Theorem 3.3(ii). The
additive baseline mirrors the structure of their Theorem 3.3(ii) and Remark 3.4, applied
per cell over the certified boxes, with the factor their Lemma B.3 drops kept. The
induction is Horbatko's Lemma 4.4 (arXiv:2604.18580) applied to the error.
