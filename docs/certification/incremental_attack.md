# Attacking the incremental bound

Every attacked number comes from 4 held-out windows of `data/processed/GME_features_20210128.parquet`, the test split, rows 3867904 to 4550475. The windows end at the same rows (3872903, 4098760, 4324618, 4550475) at every length, so a longer window adds history before the same final tick. An attacked value is reached, so it is a lower bound on the true sensitivity, and every bound must lie above it. The attacks run in float64, and every reported deviation is scored again with the reference forward pass.

Per window and block, the attack ranks every (position, output coordinate) pair with 32 random sign probes, takes the exact gradient row of the best-ranked pairs, and then runs sign-gradient ascent for 12 steps on the largest change over all positions. It starts from the gradient signs of the two best rows and from one random corner, and a fourth restart targets the last position. The number of ranked candidates is 24 at L = 10, 24 at L = 100, 32 at L = 1000, 48 at L = 5000. The model attack starts from the sign of the input gradient, for each sign of the prediction, and runs 8 steps.

Both checkpoints are the weights at training step 250 of 3000, where validation rank IC peaked, from the same seed. Outside the timescale parameters (`dt_proj` and the timescale rows of `x_proj`) their weights differ by at most 2.1 percent of each tensor's largest entry (`layers.0.x_proj.weight`). The attacked sensitivities of the two arms are therefore close, and the arms differ mainly in the bounds, where the timescale parametrisation enters.

## Block output against sequence length

The perturbation is any point within `delta` of the block's realised normalised input in sup-norm. The attacked slope is the largest `|F(xn + xi) - F(xn)|` over positions, output coordinates and windows, divided by `delta`. The first two columns are at `delta = 1e-06`, where the block is linear to the digits shown, so they are comparable with the certified slopes; the third is the largest over the four values of `delta`, which includes the nonlinear `delta = 0.1`. The certified columns are the data-free slopes of `incremental.json`, with the cell theorem's time-warp term and with the common-start one, and with the additive ones taken at the window's length. The local columns are the local block bound around each window at `delta = 1e-06`, maximised over windows, positions and coordinates, with the additive recurrence step and with the two log-timescale steps: the running maximum of the local corollary, `E_t = max(E_{t-1}, dc_t + (e^rho_t - 1)|h_{t-1} - c_t|)`, and the recursion in its proof, `E_t = K_t + a*_t (E_{t-1} - K_t)` with `K_t` the second argument of that maximum and `a*_t` the perturbed transition at its worst admissible value.

| arm | block | L | attack, worst position | attack, last position | attack, largest over delta | data-free, log-timescale | data-free, common start | additive, no floor | additive, certified floor | local, additive | local, maximum | local, recursive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 10 | 2.161 | 1.553 | 2.643 | 4.31e+06 | 2.47e+06 | 5.21e+06 | 5.01e+06 | 9.446 | 13.26 | 9.086 |
| bounded | 0 | 100 | 3.421 | 1.56 | 3.422 | 4.31e+06 | 2.47e+06 | 4.65e+07 | 3.23e+07 | 23.64 | 21.53 | 15.09 |
| bounded | 0 | 1000 | 4.116 | 1.611 | 4.116 | 4.31e+06 | 2.47e+06 | 4.59e+08 | 8.33e+07 | 63.06 | 34.11 | 23.65 |
| bounded | 0 | 5000 | 4.237 | 1.627 | 4.237 | 4.31e+06 | 2.47e+06 | 2.29e+09 | 9.5e+07 | 82.67 | 41.98 | 26.97 |
| bounded | 1 | 10 | 1.355 | 1.22 | 1.53 | 4.27e+06 | 2.45e+06 | 5.11e+06 | 4.9e+06 | 9.301 | 11.04 | 8.749 |
| bounded | 1 | 100 | 1.533 | 1.227 | 1.728 | 4.27e+06 | 2.45e+06 | 4.55e+07 | 3.15e+07 | 14.72 | 13.93 | 10.57 |
| bounded | 1 | 1000 | 1.943 | 1.231 | 1.943 | 4.27e+06 | 2.45e+06 | 4.49e+08 | 8.19e+07 | 22.46 | 21.73 | 15.27 |
| bounded | 1 | 5000 | 2.012 | 1.233 | 2.367 | 4.27e+06 | 2.45e+06 | 2.24e+09 | 9.42e+07 | 22.67 | 22.96 | 15.31 |
| softplus | 0 | 10 | 2.158 | 1.555 | 2.158 | 3.91e+06 | 2.27e+06 | 1.58e+08 | 1.58e+08 | 9.197 | 13.61 | 9.168 |
| softplus | 0 | 100 | 3.416 | 1.559 | 3.814 | 3.91e+06 | 2.27e+06 | 1.57e+09 | 1.57e+09 | 15.84 | 22.24 | 15.34 |
| softplus | 0 | 1000 | 4.037 | 1.611 | 4.436 | 3.91e+06 | 2.27e+06 | 1.57e+10 | 1.56e+10 | 25.49 | 35.54 | 24.16 |
| softplus | 0 | 5000 | 4.234 | 1.625 | 4.83 | 3.91e+06 | 2.27e+06 | 7.87e+10 | 7.73e+10 | 29.5 | 44.32 | 27.36 |
| softplus | 1 | 10 | 1.355 | 1.22 | 1.53 | 3.77e+06 | 2.2e+06 | 1.49e+08 | 1.49e+08 | 8.854 | 11.14 | 8.801 |
| softplus | 1 | 100 | 1.519 | 1.227 | 1.711 | 3.77e+06 | 2.2e+06 | 1.49e+09 | 1.49e+09 | 10.74 | 14.03 | 10.58 |
| softplus | 1 | 1000 | 1.949 | 1.231 | 1.949 | 3.77e+06 | 2.2e+06 | 1.49e+10 | 1.48e+10 | 16.01 | 22 | 15.46 |
| softplus | 1 | 5000 | 2.01 | 1.232 | 2.363 | 3.77e+06 | 2.2e+06 | 7.44e+10 | 7.29e+10 | 16.06 | 23.26 | 15.5 |

On the bounded arm, block 0, the attacked slope at `delta = 1e-06` is 2.161 at L = 10 and 4.237 at L = 5000, a factor of 1.961, while the additive bound without a floor grows by a factor of 440.1. At the last position the attacked slope ranges from 1.553 to 1.627. On the bounded arm, block 1, the attacked slope at `delta = 1e-06` is 1.355 at L = 10 and 2.012 at L = 5000, a factor of 1.485, while the additive bound without a floor grows by a factor of 439.4. At the last position the attacked slope ranges from 1.22 to 1.233. On the softplus arm, block 0, the attacked slope at `delta = 1e-06` is 2.158 at L = 10 and 4.234 at L = 5000, a factor of 1.962, while the additive bound without a floor grows by a factor of 498. At the last position the attacked slope ranges from 1.555 to 1.625. On the softplus arm, block 1, the attacked slope at `delta = 1e-06` is 1.355 at L = 10 and 2.01 at L = 5000, a factor of 1.484, while the additive bound without a floor grows by a factor of 497.9. At the last position the attacked slope ranges from 1.22 to 1.232.

At `delta = 1e-06` the worst position is the last one in 0 of 16 (arm, block, length) cases; otherwise it lies 4 to 3560 positions before the end of the window. Most of the change with `L` therefore comes from the attack finding a more sensitive position among more positions, rather than from a longer history acting on one position; the last-position column isolates the second effect. Across arms, blocks and lengths the data-free slope, which is the same at every length, is 9.24e+05 to 3.15e+06 times the attacked slope, and 5.36e+05 to 1.8e+06 times with the common-start term. At L = 5000 the additive bound without a floor is 5.41e+08 to 3.7e+10 times the attacked slope. The local bound is 4.261 to 19.51 times the attacked slope with the additive rule; 6.139 to 11.57 times the attacked slope with the log-timescale, running maximum rule; 4.205 to 7.931 times the attacked slope with the log-timescale, recursive rule.

Per `delta`:

| arm | block | L | delta | attacked slope | attacked slope, last position | data-free bound / attacked | local additive / attacked | local maximum / attacked | local recursive / attacked | worst position, from the end | ascent gain over its start |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 10 | 1e-06 | 2.161 | 1.553 | 1.99e+06 | 4.372 | 6.139 | 4.205 | 5 | 1 |
| bounded | 0 | 10 | 0.0001 | 2.16 | 1.553 | 2e+06 | 4.389 | 6.185 | 4.216 | 5 | 1 |
| bounded | 0 | 10 | 0.01 | 2.116 | 1.584 | 2.3e+06 | 14.67 | 12.38 | 5.691 | 5 | 1.001 |
| bounded | 0 | 10 | 0.1 | 2.643 | 1.879 | 6.52e+06 | 4.4e+08 | 506.9 | 268.8 | 5 | 1.541 |
| bounded | 0 | 100 | 1e-06 | 3.421 | 1.56 | 1.26e+06 | 6.911 | 6.294 | 4.411 | 88 | 1 |
| bounded | 0 | 100 | 0.0001 | 3.422 | 1.559 | 1.26e+06 | 7.057 | 6.341 | 4.43 | 88 | 1 |
| bounded | 0 | 100 | 0.01 | 3.369 | 1.588 | 1.45e+06 | 1.89e+15 | 12.64 | 7.104 | 88 | 1 |
| bounded | 0 | 100 | 0.1 | 2.903 | 1.884 | 5.94e+06 | 4.18e+99 | 596.5 | 429.4 | 88 | 1.025 |
| bounded | 0 | 1000 | 1e-06 | 4.116 | 1.611 | 1.05e+06 | 15.32 | 8.287 | 5.746 | 277 | 1 |
| bounded | 0 | 1000 | 0.0001 | 4.115 | 1.61 | 1.05e+06 | 16.16 | 8.354 | 5.778 | 277 | 1 |
| bounded | 0 | 1000 | 0.01 | 4.062 | 1.59 | 1.2e+06 | 7.93e+184 | 16.68 | 9.902 | 277 | 1.085 |
| bounded | 0 | 1000 | 0.1 | 4.085 | 1.888 | 4.22e+06 | inf | 617.5 | 422.4 | 643 | 1.141 |
| bounded | 0 | 5000 | 1e-06 | 4.237 | 1.627 | 1.02e+06 | 19.51 | 9.91 | 6.366 | 3560 | 1 |
| bounded | 0 | 5000 | 0.0001 | 4.236 | 1.626 | 1.02e+06 | 20.6 | 9.98 | 6.402 | 3560 | 1 |
| bounded | 0 | 5000 | 0.01 | 4.175 | 1.591 | 1.17e+06 | inf | 18.94 | 10.95 | 3560 | 1 |
| bounded | 0 | 5000 | 0.1 | 4.056 | 1.888 | 4.25e+06 | inf | 734.7 | 451.3 | 1932 | 1.12 |
| bounded | 1 | 10 | 1e-06 | 1.355 | 1.22 | 3.15e+06 | 6.864 | 8.145 | 6.457 | 4 | 1 |
| bounded | 1 | 10 | 0.0001 | 1.355 | 1.22 | 3.16e+06 | 6.907 | 8.201 | 6.477 | 4 | 1 |
| bounded | 1 | 10 | 0.01 | 1.372 | 1.212 | 3.51e+06 | 28.13 | 15.49 | 9.198 | 4 | 1 |
| bounded | 1 | 10 | 0.1 | 1.53 | 1.327 | 1.07e+07 | 7.84e+08 | 788.2 | 461.7 | 4 | 1.047 |
| bounded | 1 | 100 | 1e-06 | 1.533 | 1.227 | 2.79e+06 | 9.602 | 9.085 | 6.894 | 74 | 1 |
| bounded | 1 | 100 | 0.0001 | 1.533 | 1.227 | 2.79e+06 | 9.813 | 9.161 | 6.927 | 74 | 1.052 |
| bounded | 1 | 100 | 0.01 | 1.552 | 1.219 | 3.1e+06 | 1.07e+17 | 19.62 | 11.46 | 74 | 1.001 |
| bounded | 1 | 100 | 0.1 | 1.728 | 1.343 | 9.47e+06 | 4.59e+98 | 912 | 670.7 | 74 | 1.066 |
| bounded | 1 | 1000 | 1e-06 | 1.943 | 1.231 | 2.2e+06 | 11.56 | 11.18 | 7.857 | 347 | 1 |
| bounded | 1 | 1000 | 0.0001 | 1.943 | 1.231 | 2.2e+06 | 11.93 | 11.3 | 7.903 | 347 | 1 |
| bounded | 1 | 1000 | 0.01 | 1.922 | 1.223 | 2.5e+06 | 6.04e+187 | 26.9 | 14.01 | 347 | 1 |
| bounded | 1 | 1000 | 0.1 | 1.86 | 1.356 | 8.8e+06 | inf | 1314 | 881.9 | 813 | 1.013 |
| bounded | 1 | 5000 | 1e-06 | 2.012 | 1.233 | 2.12e+06 | 11.27 | 11.41 | 7.61 | 1674 | 1 |
| bounded | 1 | 5000 | 0.0001 | 2.012 | 1.232 | 2.12e+06 | 11.64 | 11.53 | 7.652 | 1674 | 1 |
| bounded | 1 | 5000 | 0.01 | 2.045 | 1.224 | 2.35e+06 | inf | 27.31 | 13.25 | 1674 | 1 |
| bounded | 1 | 5000 | 0.1 | 2.367 | 1.359 | 6.92e+06 | inf | 1119 | 699.4 | 1674 | 1.014 |
| softplus | 0 | 10 | 1e-06 | 2.158 | 1.555 | 1.81e+06 | 4.261 | 6.304 | 4.248 | 5 | 1 |
| softplus | 0 | 10 | 0.0001 | 2.158 | 1.556 | 1.82e+06 | 4.274 | 6.353 | 4.26 | 5 | 1 |
| softplus | 0 | 10 | 0.01 | 2.113 | 1.587 | 2.08e+06 | 6.343 | 12.92 | 5.776 | 5 | 1.001 |
| softplus | 0 | 10 | 0.1 | 1.979 | 1.882 | 6.97e+06 | 5.68e+12 | 656.2 | 335.4 | 1 | 1.153 |
| softplus | 0 | 100 | 1e-06 | 3.416 | 1.559 | 1.15e+06 | 4.637 | 6.511 | 4.492 | 88 | 1 |
| softplus | 0 | 100 | 0.0001 | 3.415 | 1.559 | 1.15e+06 | 4.664 | 6.563 | 4.513 | 88 | 1 |
| softplus | 0 | 100 | 0.01 | 3.364 | 1.59 | 1.3e+06 | 9.384 | 13.28 | 7.377 | 88 | 1 |
| softplus | 0 | 100 | 0.1 | 3.814 | 1.887 | 3.62e+06 | 1.29e+158 | 445.1 | 310.1 | 90 | 1.319 |
| softplus | 0 | 1000 | 1e-06 | 4.037 | 1.611 | 9.7e+05 | 6.314 | 8.805 | 5.985 | 279 | 1 |
| softplus | 0 | 1000 | 0.0001 | 4.036 | 1.611 | 9.71e+05 | 6.364 | 8.878 | 6.021 | 279 | 1 |
| softplus | 0 | 1000 | 0.01 | 3.983 | 1.593 | 1.1e+06 | 16.35 | 18.07 | 10.63 | 279 | 1 |
| softplus | 0 | 1000 | 0.1 | 4.436 | 1.89 | 3.11e+06 | inf | 549 | 381.8 | 347 | 1.27 |
| softplus | 0 | 5000 | 1e-06 | 4.234 | 1.625 | 9.25e+05 | 6.967 | 10.47 | 6.461 | 3560 | 1 |
| softplus | 0 | 5000 | 0.0001 | 4.234 | 1.625 | 9.26e+05 | 7.022 | 10.54 | 6.498 | 3560 | 1 |
| softplus | 0 | 5000 | 0.01 | 4.172 | 1.593 | 1.05e+06 | 19.05 | 20.15 | 11.26 | 3560 | 1 |
| softplus | 0 | 5000 | 0.1 | 4.83 | 1.89 | 2.86e+06 | inf | 591.6 | 366.1 | 3558 | 1.336 |
| softplus | 1 | 10 | 1e-06 | 1.355 | 1.22 | 2.79e+06 | 6.535 | 8.224 | 6.496 | 4 | 1 |
| softplus | 1 | 10 | 0.0001 | 1.355 | 1.219 | 2.79e+06 | 6.558 | 8.282 | 6.517 | 4 | 1 |
| softplus | 1 | 10 | 0.01 | 1.372 | 1.211 | 3.07e+06 | 10.31 | 15.7 | 9.303 | 4 | 1.002 |
| softplus | 1 | 10 | 0.1 | 1.53 | 1.327 | 8.12e+06 | 5.01e+11 | 761.7 | 424.1 | 4 | 1.058 |
| softplus | 1 | 100 | 1e-06 | 1.519 | 1.227 | 2.48e+06 | 7.071 | 9.235 | 6.967 | 19 | 1 |
| softplus | 1 | 100 | 0.0001 | 1.52 | 1.227 | 2.49e+06 | 7.111 | 9.312 | 7 | 19 | 1 |
| softplus | 1 | 100 | 0.01 | 1.538 | 1.219 | 2.73e+06 | 15.13 | 19.95 | 11.61 | 19 | 1 |
| softplus | 1 | 100 | 0.1 | 1.711 | 1.343 | 7.26e+06 | 2e+142 | 884.6 | 638.3 | 19 | 1.04 |
| softplus | 1 | 1000 | 1e-06 | 1.949 | 1.231 | 1.94e+06 | 8.213 | 11.29 | 7.931 | 347 | 1 |
| softplus | 1 | 1000 | 0.0001 | 1.949 | 1.231 | 1.94e+06 | 8.271 | 11.41 | 7.977 | 347 | 1 |
| softplus | 1 | 1000 | 0.01 | 1.928 | 1.223 | 2.18e+06 | 19.95 | 26.99 | 14.19 | 347 | 1 |
| softplus | 1 | 1000 | 0.1 | 1.927 | 1.356 | 6.45e+06 | inf | 1185 | 775.4 | 428 | 1.102 |
| softplus | 1 | 5000 | 1e-06 | 2.01 | 1.232 | 1.88e+06 | 7.988 | 11.57 | 7.712 | 1674 | 1 |
| softplus | 1 | 5000 | 0.0001 | 2.011 | 1.232 | 1.88e+06 | 8.043 | 11.69 | 7.755 | 1674 | 1 |
| softplus | 1 | 5000 | 0.01 | 2.044 | 1.224 | 2.06e+06 | 19.05 | 27.53 | 13.47 | 1674 | 1 |
| softplus | 1 | 5000 | 0.1 | 2.363 | 1.36 | 5.26e+06 | inf | 1045 | 638.6 | 1674 | 1.011 |

The data-free bound here is the arbitrary-perturbation reading of `incremental.json` at that `delta`, since the attacked point need not be a LayerNorm output. The ascent gain is the best value after the sign-gradient steps divided by the best starting value, so a gain of 1 means the gradient-sign start was already the best point found.

No attacked value exceeds its certificate. There are 0 violations of the data-free bound, checked per output coordinate, 0 of the local block bounds, checked per position and coordinate where they are finite, and 0 of the local end-to-end bounds.

The local block bound at `delta = 1e-2`, against the length:

| arm | block | rule | L = 10 | L = 100 | L = 1000 | L = 5000 |
| --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | additive | 31.04 | 6.38e+15 | 3.22e+185 | inf |
| bounded | 0 | log_delta | 26.2 | 42.59 | 67.75 | 79.09 |
| bounded | 0 | log_delta_recursive | 12.04 | 23.93 | 40.23 | 45.71 |
| bounded | 1 | additive | 38.59 | 1.66e+17 | 1.16e+188 | inf |
| bounded | 1 | log_delta | 21.24 | 30.45 | 51.7 | 55.85 |
| bounded | 1 | log_delta_recursive | 12.62 | 17.79 | 26.94 | 27.1 |
| softplus | 0 | additive | 13.4 | 31.57 | 65.11 | 79.47 |
| softplus | 0 | log_delta | 27.31 | 44.68 | 71.96 | 84.06 |
| softplus | 0 | log_delta_recursive | 12.21 | 24.82 | 42.32 | 46.99 |
| softplus | 1 | additive | 14.15 | 23.27 | 38.46 | 38.92 |
| softplus | 1 | log_delta | 21.54 | 30.68 | 52.03 | 56.26 |
| softplus | 1 | log_delta_recursive | 12.76 | 17.85 | 27.36 | 27.54 |

## The prediction against sequence length

The spread of the prediction over an `eps` ball at the model input is what `scripts/08_domains.py` measures. Below it is divided by `2 eps`, so a constant row means a linear response, and it is the median over the windows. The last column is `|grad|_1` at the nominal input, the first-order value of every other column.

| arm | L | eps = 1e-06 | eps = 1e-05 | eps = 0.0001 | eps = 0.001 | eps = 0.01 | eps = 0.1 | first order |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 10 | 0.7579 | 0.7579 | 0.7579 | 0.7578 | 0.758 | 0.7565 | 0.7579 |
| bounded | 100 | 0.759 | 0.759 | 0.759 | 0.759 | 0.7591 | 0.7576 | 0.759 |
| bounded | 1000 | 0.7593 | 0.7593 | 0.7593 | 0.7593 | 0.7595 | 0.758 | 0.7593 |
| bounded | 5000 | 0.7593 | 0.7593 | 0.7593 | 0.7593 | 0.7595 | 0.7581 | 0.7593 |
| softplus | 10 | 0.758 | 0.758 | 0.758 | 0.758 | 0.7582 | 0.7567 | 0.758 |
| softplus | 100 | 0.7592 | 0.7592 | 0.7592 | 0.7592 | 0.7593 | 0.7578 | 0.7592 |
| softplus | 1000 | 0.7596 | 0.7596 | 0.7596 | 0.7596 | 0.7597 | 0.7582 | 0.7596 |
| softplus | 5000 | 0.7595 | 0.7595 | 0.7595 | 0.7595 | 0.7597 | 0.7582 | 0.7595 |

At `eps = 1e-06` the attacked width divided by `2 eps` ranges from 0.7579 to 0.7593 on the bounded arm and from 0.758 to 0.7596 on the softplus arm across L = 10, 100, 1000, 5000. Up to `eps = 1e-3` the attacked width differs from the first-order width by at most 0.000141 percent, so the prediction is locally linear at these scales; at `eps = 0.1` the attacked width is 0.9982 to 0.9983 times the first-order width.

The local end-to-end bound against the length, as the median width over the windows:

| arm | rule | eps | L = 10 | L = 100 | L = 1000 | L = 5000 |
| --- | --- | --- | --- | --- | --- | --- |
| bounded | additive | 1e-06 | 0.1134 | 0.7207 | inf | inf |
| bounded | additive | 1e-05 | 15.72 | inf | inf | inf |
| bounded | maximum | 1e-06 | 0.1656 | 0.2815 | 0.6293 | 0.8918 |
| bounded | maximum | 1e-05 | 20.03 | 141.5 | 7.54e+09 | 2.49e+10 |
| bounded | recursive | 1e-06 | 0.09029 | 0.1575 | 0.201 | 0.2047 |
| bounded | recursive | 1e-05 | 3.325 | 15.23 | 32.55 | 34.99 |
| softplus | additive | 1e-06 | 0.09297 | 0.171 | 0.2304 | 0.2368 |
| softplus | additive | 1e-05 | 4.03 | 35.66 | 7.78e+05 | inf |
| softplus | maximum | 1e-06 | 0.1693 | 0.2895 | 0.6661 | 0.9494 |
| softplus | maximum | 1e-05 | 21.17 | 163.4 | 9.18e+09 | 3.03e+10 |
| softplus | recursive | 1e-06 | 0.09078 | 0.1571 | 0.2026 | 0.2068 |
| softplus | recursive | 1e-05 | 3.385 | 15.17 | 33.67 | 36.51 |

At `eps = 1e-06` the local end-to-end bound is non-finite with the additive rule on the bounded arm from L = 1000 on; it stays finite with the log-timescale, running maximum rule on the bounded arm, going from 0.1656 at L = 10 to 0.8918 at L = 5000; it stays finite with the log-timescale, recursive rule on the bounded arm, going from 0.09029 at L = 10 to 0.2047 at L = 5000; it stays finite with the additive rule on the softplus arm, going from 0.09297 at L = 10 to 0.2368 at L = 5000; it stays finite with the log-timescale, running maximum rule on the softplus arm, going from 0.1693 at L = 10 to 0.9494 at L = 5000; it stays finite with the log-timescale, recursive rule on the softplus arm, going from 0.09078 at L = 10 to 0.2068 at L = 5000.

## The local end-to-end certificate on the windows of domains.json

bounded arm: 4 windows of length 100, the first of the unshuffled test loader, as `scripts/08_domains.py` takes them. The global certified width of the prediction is 9.462.

| eps | attack width | additive | maximum | recursive | additive / attack | maximum / attack | recursive / attack |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-06 | 1.63e-06 | 2.18 | 0.7063 | 0.2877 | 1.34e+06 | 4.33e+05 | 1.77e+05 |
| 1e-05 | 1.63e-05 | inf | 3.06e+04 | 61.09 | inf | 1.88e+09 | 3.75e+06 |
| 0.0001 | 0.000163 | inf | inf | 2.85e+25 | inf | inf | 1.75e+29 |
| 0.001 | 0.00163 | inf | inf | inf | inf | inf | inf |
| 0.01 | 0.01631 | inf | inf | inf | inf | inf | inf |
| 0.1 | 0.1623 | nan | inf | inf | nan | inf | inf |

With the additive rule the bound is first non-finite at `eps = 1e-05`, from the stage `L1 recurrence` on, and first above the global certified width at `eps = 1e-05`. There the step multiplier in block 1 reaches 2.123, and 67.79 percent of the steps enlarge the carried bound. With the log-timescale, running maximum rule the bound is first non-finite at `eps = 0.0001`, from the stage `final LayerNorm` on, and first above the global certified width at `eps = 1e-05`. With the log-timescale, recursive rule the bound is first non-finite at `eps = 0.001`, from the stage `L1 recurrence` on, and first above the global certified width at `eps = 1e-05`. The step never scales the carried bound up; there the largest relative timescale perturbation in block 1 is `rho = 4.64e+13`, beyond 88.7, where `e^rho - 1` overflows float32. The largest `eps` at which the bound is below the global certified width is 1e-06 with the additive rule, 1e-06 with the log-timescale, running maximum rule, 1e-06 with the log-timescale, recursive rule.

The additive rule reproduces the `lipschitz` column of `domains.json`: yes.

softplus arm: 4 windows of length 100, the first of the unshuffled test loader, as `scripts/08_domains.py` takes them. The global certified width of the prediction is 9.462.

| eps | attack width | additive | maximum | recursive | additive / attack | maximum / attack | recursive / attack |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-06 | 1.63e-06 | 0.3228 | 0.7457 | 0.2891 | 1.98e+05 | 4.58e+05 | 1.77e+05 |
| 1e-05 | 1.63e-05 | 953.5 | 5.48e+04 | 62.37 | 5.85e+07 | 3.36e+09 | 3.83e+06 |
| 0.0001 | 0.000163 | inf | inf | 7.59e+23 | inf | inf | 4.66e+27 |
| 0.001 | 0.00163 | inf | inf | inf | inf | inf | inf |
| 0.01 | 0.01631 | inf | inf | inf | inf | inf | inf |
| 0.1 | 0.1624 | nan | inf | inf | nan | inf | inf |

With the additive rule the bound is first non-finite at `eps = 0.0001`, from the stage `L1 recurrence` on, and first above the global certified width at `eps = 1e-05`. There the step multiplier in block 1 reaches 223.3, and 97.83 percent of the steps enlarge the carried bound. With the log-timescale, running maximum rule the bound is first non-finite at `eps = 0.0001`, from the stage `final LayerNorm` on, and first above the global certified width at `eps = 1e-05`. With the log-timescale, recursive rule the bound is first non-finite at `eps = 0.001`, from the stage `L1 recurrence` on, and first above the global certified width at `eps = 1e-05`. The step never scales the carried bound up; there the largest relative timescale perturbation in block 1 is `rho = 3.99e+13`, beyond 88.7, where `e^rho - 1` overflows float32. The largest `eps` at which the bound is below the global certified width is 1e-06 with the additive rule, 1e-06 with the log-timescale, running maximum rule, 1e-06 with the log-timescale, recursive rule.

## Where the end-to-end bound comes from

The domain carries one sensitivity per coordinate from the input, where it is 1, to the prediction. The stages form a chain, so the bound is the product of the stage ratios `max(after) / max(before)`, and its `log10` is the sum of theirs. The groups add those logs. LayerNorm is the stem, block and final norms. Feed-forward is every Linear, the convolution, every SiLU, the gate product and the head. Recurrence is the scan and its readout `C h + D u`. Residual is the skip connection. A negative entry is a stage whose largest sensitivity is smaller than its input's. Only `eps` values with a finite bound appear.

| arm | rule | eps | LayerNorm | feed-forward | recurrence | residual | total, log10 of the bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | additive | 1e-06 | 2.841 | 0.8322 | 2.466 | 0.00969 | 6.149 |
| bounded | maximum | 1e-06 | 2.78 | 1.349 | 1.455 | 0.01522 | 5.599 |
| bounded | recursive | 1e-06 | 2.788 | 1.325 | 1.044 | 0.02717 | 5.184 |
| bounded | maximum | 1e-05 | 7.712 | 0.4638 | 1.931 | 0.00879 | 10.12 |
| bounded | recursive | 1e-05 | 3.448 | 1.719 | 1.321 | 0.01671 | 6.505 |
| bounded | recursive | 0.0001 | 21.23 | 2.04 | 5.981 | 0.00686 | 29.26 |
| softplus | additive | 1e-06 | 2.794 | 1.308 | 1.103 | 0.0262 | 5.232 |
| softplus | maximum | 1e-06 | 2.777 | 1.376 | 1.456 | 0.01484 | 5.624 |
| softplus | recursive | 1e-06 | 2.787 | 1.331 | 1.04 | 0.02714 | 5.185 |
| softplus | additive | 1e-05 | 4.418 | 1.678 | 1.795 | 0.01147 | 7.903 |
| softplus | maximum | 1e-05 | 10.58 | -1.716 | 1.94 | 0.00845 | 10.81 |
| softplus | recursive | 1e-05 | 3.459 | 1.725 | 1.317 | 0.01664 | 6.518 |
| softplus | recursive | 0.0001 | 20.18 | 2.209 | 5.278 | 0.00705 | 27.68 |

On the bounded arm at `eps = 1e-06`, the recurrence group splits into 1.763 decades from the scans and 0.7028 from the readouts with the additive rule; 1.196 decades from the scans and 0.2581 from the readouts with the log-timescale, running maximum rule; 0.1917 decades from the scans and 0.8525 from the readouts with the log-timescale, recursive rule. The timescale rule changes the scan stages only; LayerNorm and the feed-forward stages differ between the rules only through the sensitivity that reaches them.

The attacked sensitivity of the prediction on these windows is 0.8147 on the bounded arm and 0.8149 on the softplus arm (width over `2 eps` at `eps = 1e-06`). The local end-to-end bound exceeds it by a factor of 4.33e+05 on the bounded arm, 4.58e+05 on the softplus arm with the log-timescale, running maximum rule; 1.77e+05 on the bounded arm, 1.77e+05 on the softplus arm with the log-timescale, recursive rule. On the attack windows of length 100 the factor is 1.85e+05 on the bounded arm, 1.91e+05 on the softplus arm with the log-timescale, running maximum rule; 1.04e+05 on the bounded arm, 1.03e+05 on the softplus arm with the log-timescale, recursive rule. On those same windows, around one block's own normalised input at `delta = 1e-06`, the same domain is 6.294 to 9.235 times the attacked block slope with the log-timescale, running maximum rule; 4.411 to 6.967 times the attacked block slope with the log-timescale, recursive rule, and there it covers the block's linear layers, convolution, gate and recurrence. The two measurements place most of the end-to-end excess outside the block interiors: in the LayerNorms, the stem and the head, and in chaining stage bounds that are each computed on their own.

Stage by stage on the bounded arm at `eps = 1e-06`, as the largest sensitivity over the windows, positions and coordinates:

| stage | group | additive | log10 ratio | maximum | log10 ratio | recursive | log10 ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stem Linear | feed-forward | 4.022 | 0.6044 | 4.022 | 0.6044 | 4.022 | 0.6044 |
| stem SiLU | feed-forward | 3.959 | -0.00684 | 3.959 | -0.00684 | 3.959 | -0.00684 |
| stem LayerNorm | LayerNorm | 55.89 | 1.15 | 55.89 | 1.15 | 55.89 | 1.15 |
| L0 LayerNorm | LayerNorm | 162.4 | 0.4632 | 162.4 | 0.4632 | 162.4 | 0.4632 |
| L0 in_proj | feed-forward | 362.5 | 0.3488 | 362.5 | 0.3488 | 362.5 | 0.3488 |
| L0 conv | feed-forward | 527 | 0.1625 | 527 | 0.1625 | 527 | 0.1625 |
| L0 SiLU | feed-forward | 532.9 | 0.00487 | 532.9 | 0.00487 | 532.9 | 0.00487 |
| L0 recurrence | recurrence | 2327 | 0.6401 | 2198 | 0.6154 | 676.6 | 0.1036 |
| L0 readout | recurrence | 4457 | 0.2823 | 2851 | 0.1129 | 1961 | 0.4621 |
| L0 gate | feed-forward | 2934 | -0.1816 | 2703 | -0.02321 | 1753 | -0.04867 |
| L0 out_proj | feed-forward | 1379 | -0.3278 | 1347 | -0.3023 | 864.7 | -0.3068 |
| L0 residual | residual | 1397 | 0.00552 | 1366 | 0.00584 | 882.4 | 0.00878 |
| L1 LayerNorm | LayerNorm | 4853 | 0.5408 | 4457 | 0.5137 | 3234 | 0.564 |
| L1 in_proj | feed-forward | 1.44e+04 | 0.4721 | 1.32e+04 | 0.4725 | 9033 | 0.4461 |
| L1 conv | feed-forward | 2.35e+04 | 0.2139 | 2.18e+04 | 0.2162 | 1.48e+04 | 0.2155 |
| L1 SiLU | feed-forward | 1.68e+04 | -0.1456 | 1.64e+04 | -0.1227 | 1.14e+04 | -0.1144 |
| L1 recurrence | recurrence | 2.23e+05 | 1.123 | 6.25e+04 | 0.5811 | 1.4e+04 | 0.08805 |
| L1 readout | recurrence | 5.89e+05 | 0.4205 | 8.74e+04 | 0.1451 | 3.43e+04 | 0.3904 |
| L1 gate | feed-forward | 1.1e+05 | -0.7278 | 3.87e+04 | -0.3539 | 2.03e+04 | -0.2287 |
| L1 out_proj | feed-forward | 9.18e+04 | -0.07905 | 3.8e+04 | -0.00769 | 1.84e+04 | -0.0417 |
| L1 residual | residual | 9.27e+04 | 0.00417 | 3.88e+04 | 0.00937 | 1.92e+04 | 0.01839 |
| final LayerNorm | LayerNorm | 4.51e+05 | 0.6874 | 1.75e+05 | 0.6535 | 7.84e+04 | 0.6107 |
| last position | feed-forward | 4.51e+05 | 0 | 1.75e+05 | 0 | 7.84e+04 | 0 |
| head Linear and SiLU | feed-forward | 1.02e+06 | 0.3525 | 3.4e+05 | 0.2893 | 1.4e+05 | 0.2509 |
| head Linear | feed-forward | 1.41e+06 | 0.1419 | 3.97e+05 | 0.06727 | 1.53e+05 | 0.03877 |

Inside each recurrence, per block: `rho` is the largest relative timescale perturbation, `L_log_delta eps`. The step multiplier is the largest factor that one step applies to the carried bound: `Abar + |A| exp(-|A| dt_lo) L_dt eps` for the additive rule, which can exceed 1; 1 for the running maximum; and the slowest admissible perturbed transition `exp(-|A| dt e^-rho)`, below 1, for the recursive rule. The fraction growing counts the (cell, step) pairs with a multiplier above 1; the fraction raised counts those at which the carried bound increases at all. The last two columns are the largest state sensitivity with the timescale's own sensitivity and without it. The run without it is not a bound; it measures how much of the recurrence's growth comes through the timescale.

| arm | rule | eps | block | rho | step multiplier | fraction of steps growing | fraction of steps raised | state sensitivity | without the timescale term |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | additive | 1e-06 | 0 | - | 0.9989 | 0 | 0.7077 | 2327 | 611.2 |
| bounded | additive | 1e-06 | 1 | - | 1.02 | 0.01016 | 0.7975 | 2.23e+05 | 2.16e+04 |
| bounded | maximum | 1e-06 | 0 | 0.00134 | 1 | 0 | 0.09132 | 2198 | 1956 |
| bounded | maximum | 1e-06 | 1 | 0.0575 | 1 | 0 | 0.1442 | 6.25e+04 | 5.84e+04 |
| bounded | recursive | 1e-06 | 0 | 0.00134 | 0.9989 | 0 | 0.7017 | 676.6 | 611.2 |
| bounded | recursive | 1e-06 | 1 | 0.03898 | 0.9989 | 0 | 0.6968 | 1.4e+04 | 1.37e+04 |
| bounded | additive | 1e-05 | 0 | - | 0.999 | 0 | 0.7386 | 2546 | 615 |
| bounded | additive | 1e-05 | 1 | - | 2.123 | 0.6779 | 0.9433 | inf | 3.22e+04 |
| bounded | maximum | 1e-05 | 0 | 0.0134 | 1 | 0 | 0.09152 | 2212 | 1968 |
| bounded | maximum | 1e-05 | 1 | 0.6754 | 1 | 0 | 0.1689 | 8.31e+04 | 7.68e+04 |
| bounded | recursive | 1e-05 | 0 | 0.0134 | 0.9989 | 0 | 0.7021 | 682.8 | 615 |
| bounded | recursive | 1e-05 | 1 | 0.4348 | 0.9989 | 0 | 0.7026 | 1.73e+04 | 1.67e+04 |
| bounded | additive | 0.0001 | 0 | - | 1.127 | 0.4272 | 0.8513 | 1.2e+11 | 654.2 |
| bounded | additive | 0.0001 | 1 | - | 6.81e+23 | 0.9924 | 0.3071 | inf | nan |
| bounded | maximum | 0.0001 | 0 | 0.1404 | 1 | 0 | 0.0935 | 2353 | 2093 |
| bounded | maximum | 0.0001 | 1 | 29.34 | 1 | 0 | 0.2079 | 1.79e+15 | 1.97e+06 |
| bounded | recursive | 0.0001 | 0 | 0.1404 | 0.9989 | 0 | 0.7047 | 748.6 | 654.2 |
| bounded | recursive | 0.0001 | 1 | 12.87 | 1 | 0 | 0.3074 | 1.26e+08 | 2.38e+05 |
| bounded | additive | 0.001 | 0 | - | 4.277 | 0.8869 | 0.897 | inf | 1230 |
| bounded | additive | 0.001 | 1 | - | inf | 1 | 0.0325 | inf | nan |
| bounded | maximum | 0.001 | 0 | 2.078 | 1 | 0 | 0.1058 | 4361 | 3804 |
| bounded | maximum | 0.001 | 1 | 1.99e+14 | 1 | 0 | 0.01002 | inf | 8.9e+30 |
| bounded | recursive | 0.001 | 0 | 2.078 | 0.9998 | 0 | 0.5849 | 1988 | 1230 |
| bounded | recursive | 0.001 | 1 | 4.64e+13 | 1 | 0 | 0.02002 | inf | 1.07e+29 |
| bounded | additive | 0.01 | 0 | - | 487.2 | 1 | 0.1965 | inf | 6.57e+05 |
| bounded | additive | 0.01 | 1 | - | inf | 1 | 0.01 | inf | nan |
| bounded | maximum | 0.01 | 0 | 303.4 | 1 | 0 | 0.08017 | inf | 1.99e+06 |
| bounded | maximum | 0.01 | 1 | inf | 1 | 0 | 0.01 | inf | inf |
| bounded | recursive | 0.01 | 0 | 303.4 | 1 | 0 | 0.08119 | inf | 6.57e+05 |
| bounded | recursive | 0.01 | 1 | inf | 1 | 0 | 0.01 | inf | inf |
| bounded | additive | 0.1 | 0 | - | 1.77e+28 | 1 | 0.01 | inf | nan |
| bounded | additive | 0.1 | 1 | - | inf | 1 | 0 | nan | nan |
| bounded | maximum | 0.1 | 0 | 1.1e+28 | 1 | 0 | 0.01 | inf | inf |
| bounded | maximum | 0.1 | 1 | inf | 1 | 0 | 0.01 | inf | inf |
| bounded | recursive | 0.1 | 0 | 1.1e+28 | 1 | 0 | 0.01 | inf | inf |
| bounded | recursive | 0.1 | 1 | inf | 1 | 0 | 0.01 | inf | inf |
| softplus | additive | 1e-06 | 0 | - | 0.9992 | 0 | 0.7203 | 835.9 | 582.5 |
| softplus | additive | 1e-06 | 1 | - | 0.9991 | 0 | 0.7227 | 1.74e+04 | 1.45e+04 |
| softplus | maximum | 1e-06 | 0 | 0.00128 | 1 | 0 | 0.08939 | 2188 | 1962 |
| softplus | maximum | 1e-06 | 1 | 0.0508 | 1 | 0 | 0.1433 | 6.49e+04 | 6.01e+04 |
| softplus | recursive | 1e-06 | 0 | 0.00128 | 0.9992 | 0 | 0.6982 | 641.9 | 582.5 |
| softplus | recursive | 1e-06 | 1 | 0.0336 | 0.9991 | 0 | 0.6932 | 1.44e+04 | 1.4e+04 |
| softplus | additive | 1e-05 | 0 | - | 0.9992 | 0 | 0.7227 | 862 | 586.1 |
| softplus | additive | 1e-05 | 1 | - | 0.9994 | 0 | 0.8056 | 5.15e+04 | 1.79e+04 |
| softplus | maximum | 1e-05 | 0 | 0.01282 | 1 | 0 | 0.08957 | 2202 | 1974 |
| softplus | maximum | 1e-05 | 1 | 0.6 | 1 | 0 | 0.1687 | 8.61e+04 | 7.96e+04 |
| softplus | recursive | 1e-05 | 0 | 0.01282 | 0.9992 | 0 | 0.6984 | 647.6 | 586.1 |
| softplus | recursive | 1e-05 | 1 | 0.3748 | 0.9993 | 0 | 0.6985 | 1.8e+04 | 1.71e+04 |
| softplus | additive | 0.0001 | 0 | - | 0.9992 | 0 | 0.7486 | 1214 | 623.3 |
| softplus | additive | 0.0001 | 1 | - | 223.3 | 0.9783 | 0.4185 | inf | 3.45e+05 |
| softplus | maximum | 0.0001 | 0 | 0.1326 | 1 | 0 | 0.09157 | 2340 | 2099 |
| softplus | maximum | 0.0001 | 1 | 26.7 | 1 | 0 | 0.2108 | 1.37e+14 | 2.19e+06 |
| softplus | recursive | 0.0001 | 0 | 0.1326 | 0.9992 | 0 | 0.6991 | 708.1 | 623.3 |
| softplus | recursive | 0.0001 | 1 | 11.17 | 1 | 0 | 0.3355 | 2.49e+07 | 2.48e+05 |
| softplus | additive | 0.001 | 0 | - | 9.728 | 0.83 | 0.9704 | inf | 1292 |
| softplus | additive | 0.001 | 1 | - | inf | 1 | 0.04 | inf | nan |
| softplus | maximum | 0.001 | 0 | 1.855 | 1 | 0 | 0.1051 | 4587 | 3815 |
| softplus | maximum | 0.001 | 1 | 1.77e+14 | 1 | 0 | 0.01006 | inf | 9.43e+30 |
| softplus | recursive | 0.001 | 0 | 1.855 | 0.9997 | 0 | 0.5905 | 1832 | 1292 |
| softplus | recursive | 0.001 | 1 | 3.99e+13 | 1 | 0 | 0.02002 | inf | 1.2e+29 |
| softplus | additive | 0.01 | 0 | - | 4326 | 1 | 0.1314 | inf | 6.98e+05 |
| softplus | additive | 0.01 | 1 | - | inf | 1 | 0.01 | inf | nan |
| softplus | maximum | 0.01 | 0 | 269.9 | 1 | 0 | 0.09062 | inf | 2e+06 |
| softplus | maximum | 0.01 | 1 | inf | 1 | 0 | 0.01 | inf | inf |
| softplus | recursive | 0.01 | 0 | 269.9 | 1 | 0 | 0.09191 | inf | 6.98e+05 |
| softplus | recursive | 0.01 | 1 | inf | 1 | 0 | 0.01 | inf | inf |
| softplus | additive | 0.1 | 0 | - | 1.57e+29 | 1 | 0.01 | inf | nan |
| softplus | additive | 0.1 | 1 | - | inf | 1 | 0 | nan | nan |
| softplus | maximum | 0.1 | 0 | 9.78e+27 | 1 | 0 | 0.01 | inf | inf |
| softplus | maximum | 0.1 | 1 | inf | 1 | 0 | 0.01 | inf | inf |
| softplus | recursive | 0.1 | 0 | 9.78e+27 | 1 | 0 | 0.01 | inf | inf |
| softplus | recursive | 0.1 | 1 | inf | 1 | 0 | 0.01 | inf | inf |

## Runtime and what was not run

The run took 42.2 minutes on Windows-11-10.0.26200-SP0, on the CPU. Per arm and length, the block and model attacks took:

| arm | L | block attack, seconds | model attack, seconds |
| --- | --- | --- | --- |
| bounded | 10 | 1.994 | 1.025 |
| bounded | 100 | 15.03 | 6.664 |
| bounded | 1000 | 141.9 | 61.84 |
| bounded | 5000 | 726.1 | 302.8 |
| softplus | 10 | 1.903 | 0.9867 |
| softplus | 100 | 14.48 | 6.502 |
| softplus | 1000 | 137.1 | 61.03 |
| softplus | 5000 | 714.9 | 300.1 |

The block attack has one random-corner restart per `delta` and the model attack has none. The same 4 windows are used at every length, and no further windows were run. The attack is a heuristic search, so every attacked value is a lower bound and the true worst case may be larger.

## Scope

The data-free and local bounds hold over the reals. The local end-to-end bound is evaluated in float32, as `domains.json` was, and the local block bound in float64. The data-free incremental certificate is per block, with respect to the normalised input. A data-free end-to-end counterpart would pass each LayerNorm with its global Lipschitz constant, which exists because `eps > 0`: `max|gamma| / sqrt(eps)` in l2 is 326.9, 324.1 for the blocks of the bounded arm; 326.9, 324.1 for the blocks of the softplus arm, against about 1 at realised inputs (`incremental.json`). It is not computed here.
