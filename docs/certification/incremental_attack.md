# Attacking the incremental bound

Every attacked number comes from 4 held-out windows of `data/processed/SPY_features_20210128.parquet`, the test split, rows 3867904 to 4550475. The windows end at the same rows (3872903, 4098760, 4324618, 4550475) at every length, so a longer window adds history before the same final tick. An attacked value is reached, so it is a lower bound on the true sensitivity, and every bound must lie above it. The attacks run in float64, and every reported deviation is scored again with the reference forward pass.

Per window and block, the attack ranks every (position, output coordinate) pair with 32 random sign probes, takes the exact gradient row of the best-ranked pairs, and then runs sign-gradient ascent for 12 steps on the largest change over all positions. It starts from the gradient signs of the two best rows and from one random corner, and a fourth restart targets the last position. The number of ranked candidates is 24 at L = 10, 24 at L = 100, 32 at L = 1000, 48 at L = 5000. The model attack starts from the sign of the input gradient, for each sign of the prediction, and runs 8 steps.

The checkpoints are the weights at training steps 750 (bounded) and 2000 (softplus) of 3000, where validation rank IC peaked, from the same seed. Outside the timescale parameters (`dt_proj` and the timescale rows of `x_proj`) their weights differ by at most 99 percent of each tensor's largest entry (`layers.0.norm.bias`). The arms therefore differ in their weights as well as in the timescale parametrisation, so a difference between them, attacked or certified, is not due to the parametrisation alone.

## Block output against sequence length

The perturbation is any point within `delta` of the block's realised normalised input in sup-norm. The attacked slope is the largest `|F(xn + xi) - F(xn)|` over positions, output coordinates and windows, divided by `delta`. The first two columns are at `delta = 1e-06`, where the block is linear to the digits shown, so they are comparable with the certified slopes; the third is the largest over the four values of `delta`, which includes the nonlinear `delta = 0.1`. The certified columns are the data-free slopes of `incremental.json`, with the cell theorem's time-warp term and with the common-start one, and with the additive ones taken at the window's length. The local columns are the local block bound around each window at `delta = 1e-06`, maximised over windows, positions and coordinates, with the additive recurrence step and with the three log-timescale steps: the running maximum of the local corollary, `E_t = max(E_{t-1}, dc_t + (e^rho_t - 1)|h_{t-1} - c_t|)`; the recursion in its proof, `E_t = K_t + a*_t (E_{t-1} - K_t)` with `K_t` the second argument of that maximum and `a*_t` the perturbed transition at its worst admissible value; and the endpoint form, the reported one, `E_t = max_s [s E_{t-1} + (1 - s) dc_t + |Abar_t - s| |h_{t-1} - c_t|]` over the two ends of the interval that holds the perturbed transition `s`, the intersection of the log-timescale interval with the one the additive bound on `|dt - dt~|` gives.

| arm | block | L | attack, worst position | attack, last position | attack, largest over delta | data-free, log-timescale | data-free, common start | additive, no floor | additive, certified floor | local, additive | local, maximum | local, recursive | local, additive endpoint | local, endpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 10 | 3.061 | 2.361 | 3.586 | 9.34e+06 | 5.17e+06 | 1.11e+07 | 1.06e+07 | 18.87 | 40.78 | 18.21 | 18.17 | 18.17 |
| bounded | 0 | 100 | 5.167 | 2.855 | 5.827 | 9.34e+06 | 5.17e+06 | 1.02e+08 | 7.12e+07 | 42.78 | 69.63 | 37.3 | 37.24 | 37.24 |
| bounded | 0 | 1000 | 7.963 | 3.757 | 9.239 | 9.34e+06 | 5.17e+06 | 1.01e+09 | 1.87e+08 | 84.72 | 121.7 | 72.02 | 71.95 | 71.95 |
| bounded | 0 | 5000 | 9.992 | 4.045 | 11.96 | 9.34e+06 | 5.17e+06 | 5.03e+09 | 2.15e+08 | 112.5 | 154.9 | 87.84 | 87.6 | 87.6 |
| bounded | 1 | 10 | 3.875 | 3.875 | 4.281 | 6.07e+06 | 3.44e+06 | 7.34e+06 | 7.04e+06 | 13.2 | 24.13 | 13.05 | 13.04 | 13.04 |
| bounded | 1 | 100 | 4.228 | 3.927 | 4.647 | 6.07e+06 | 3.44e+06 | 6.62e+07 | 4.56e+07 | 23.4 | 32.3 | 22.4 | 22.38 | 22.38 |
| bounded | 1 | 1000 | 5.29 | 4.049 | 5.689 | 6.07e+06 | 3.44e+06 | 6.54e+08 | 1.18e+08 | 34.23 | 47.84 | 32.87 | 32.85 | 32.85 |
| bounded | 1 | 5000 | 5.575 | 4.122 | 5.99 | 6.07e+06 | 3.44e+06 | 3.27e+09 | 1.36e+08 | 41.14 | 58.91 | 39.81 | 39.79 | 39.79 |
| softplus | 0 | 10 | 8.246 | 7.176 | 8.246 | 1.84e+07 | 1.02e+07 | 7.2e+08 | 7.2e+08 | 54.06 | 226 | 50.56 | 50.16 | 50.16 |
| softplus | 0 | 100 | 19.51 | 13.44 | 23.16 | 1.84e+07 | 1.02e+07 | 7.18e+09 | 7.18e+09 | 336.5 | 568.2 | 246.5 | 245.5 | 245.5 |
| softplus | 0 | 1000 | 25.69 | 17.73 | 31.37 | 1.84e+07 | 1.02e+07 | 7.18e+10 | 7.18e+10 | 461.7 | 649.8 | 312.9 | 311.9 | 311.9 |
| softplus | 0 | 5000 | 60.19 | 16.3 | 73.63 | 1.84e+07 | 1.02e+07 | 3.59e+11 | 3.59e+11 | 658.6 | 893.2 | 405.8 | 403.7 | 403.7 |
| softplus | 1 | 10 | 4.516 | 3.97 | 4.953 | 1.46e+07 | 8.07e+06 | 5.75e+08 | 5.75e+08 | 25.31 | 81.43 | 23.34 | 23.25 | 23.25 |
| softplus | 1 | 100 | 5.408 | 4.272 | 5.858 | 1.46e+07 | 8.07e+06 | 5.74e+09 | 5.74e+09 | 56.68 | 107.4 | 47.25 | 47.14 | 47.14 |
| softplus | 1 | 1000 | 6.943 | 6.365 | 7.763 | 1.46e+07 | 8.07e+06 | 5.74e+10 | 5.72e+10 | 101 | 189.7 | 81.19 | 81.07 | 81.07 |
| softplus | 1 | 5000 | 14.31 | 7.587 | 18.16 | 1.46e+07 | 8.07e+06 | 2.87e+11 | 2.85e+11 | 178.8 | 243.2 | 136.1 | 135.9 | 135.9 |

On the bounded arm, block 0, the attacked slope at `delta = 1e-06` is 3.061 at L = 10 and 9.992 at L = 5000, a factor of 3.264, while the additive bound without a floor grows by a factor of 454.7. At the last position the attacked slope ranges from 2.361 to 4.045. On the bounded arm, block 1, the attacked slope at `delta = 1e-06` is 3.875 at L = 10 and 5.575 at L = 5000, a factor of 1.439, while the additive bound without a floor grows by a factor of 445.4. At the last position the attacked slope ranges from 3.875 to 4.122. On the softplus arm, block 0, the attacked slope at `delta = 1e-06` is 8.246 at L = 10 and 60.19 at L = 5000, a factor of 7.299, while the additive bound without a floor grows by a factor of 498.6. At the last position the attacked slope ranges from 7.176 to 17.73. On the softplus arm, block 1, the attacked slope at `delta = 1e-06` is 4.516 at L = 10 and 14.31 at L = 5000, a factor of 3.168, while the additive bound without a floor grows by a factor of 498.6. At the last position the attacked slope ranges from 3.97 to 7.587.

At `delta = 1e-06` the worst position is the last one in 1 of 16 (arm, block, length) cases; otherwise it lies 3 to 3058 positions before the end of the window. The last-position column isolates a longer history acting on one position: from L = 10 to L = 5000 it grows by a factor of 1.064 to 2.271, against 1.439 to 7.299 over all positions, so the change with `L` comes partly from the attack finding a more sensitive position among more positions and partly from the longer history. Across arms, blocks and lengths the data-free slope, which is the same at every length, is 3.06e+05 to 3.23e+06 times the attacked slope, and 1.7e+05 to 1.79e+06 times with the common-start term. At L = 5000 the additive bound without a floor is 5.04e+08 to 2.01e+10 times the attacked slope. The local bound is 3.407 to 17.98 times the attacked slope with the additive rule; 6.227 to 29.13 times the attacked slope with the log-timescale, running maximum rule; 3.369 to 12.64 times the attacked slope with the log-timescale, recursive rule; 3.365 to 12.58 times the attacked slope with the additive interval, endpoint rule; 3.365 to 12.58 times the attacked slope with the log-timescale, endpoint rule.

Per `delta`:

| arm | block | L | delta | attacked slope | attacked slope, last position | data-free bound / attacked | local additive / attacked | local maximum / attacked | local recursive / attacked | local additive endpoint / attacked | local endpoint / attacked | worst position, from the end | ascent gain over its start |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | 10 | 1e-06 | 3.061 | 2.361 | 3.05e+06 | 6.165 | 13.32 | 5.948 | 5.936 | 5.936 | 6 | 1 |
| bounded | 0 | 10 | 0.0001 | 3.062 | 2.361 | 3.05e+06 | 6.192 | 13.4 | 5.969 | 5.957 | 5.957 | 6 | 1 |
| bounded | 0 | 10 | 0.01 | 3.108 | 2.408 | 3.53e+06 | 10.01 | 23.07 | 8.777 | 8.68 | 8.664 | 6 | 1.001 |
| bounded | 0 | 10 | 0.1 | 3.586 | 2.865 | 1.81e+07 | 2.81e+05 | 992.9 | 602.1 | 355.5 | 340.5 | 5 | 1.08 |
| bounded | 0 | 100 | 1e-06 | 5.167 | 2.855 | 1.81e+06 | 8.28 | 13.47 | 7.219 | 7.207 | 7.207 | 81 | 1 |
| bounded | 0 | 100 | 0.0001 | 5.168 | 2.855 | 1.81e+06 | 8.338 | 13.55 | 7.253 | 7.241 | 7.24 | 81 | 1 |
| bounded | 0 | 100 | 0.01 | 5.232 | 2.91 | 2.09e+06 | 20.76 | 23.2 | 11.83 | 11.78 | 11.65 | 81 | 1 |
| bounded | 0 | 100 | 0.1 | 5.827 | 3.494 | 1.11e+07 | 7.53e+61 | 810.3 | 659 | 409.9 | 394.6 | 82 | 1.043 |
| bounded | 0 | 1000 | 1e-06 | 7.963 | 3.757 | 1.17e+06 | 10.64 | 15.28 | 9.043 | 9.035 | 9.035 | 279 | 1 |
| bounded | 0 | 1000 | 0.0001 | 7.964 | 3.758 | 1.17e+06 | 10.7 | 15.36 | 9.081 | 9.072 | 9.072 | 279 | 1 |
| bounded | 0 | 1000 | 0.01 | 8.083 | 3.848 | 1.36e+06 | 22.97 | 25.02 | 13.76 | 13.78 | 13.66 | 279 | 1 |
| bounded | 0 | 1000 | 0.1 | 9.239 | 4.828 | 7.02e+06 | inf | 813.4 | 604.4 | 411.8 | 323.7 | 279 | 1.017 |
| bounded | 0 | 5000 | 1e-06 | 9.992 | 4.045 | 9.35e+05 | 11.26 | 15.5 | 8.791 | 8.767 | 8.767 | 3052 | 1 |
| bounded | 0 | 5000 | 0.0001 | 9.993 | 4.046 | 9.36e+05 | 11.33 | 15.58 | 8.825 | 8.8 | 8.799 | 3052 | 1 |
| bounded | 0 | 5000 | 0.01 | 10.17 | 4.148 | 1.08e+06 | 26.66 | 25.28 | 13 | 13.03 | 12.95 | 3052 | 1 |
| bounded | 0 | 5000 | 0.1 | 11.96 | 5.285 | 5.42e+06 | inf | 814.5 | 501.4 | 838.6 | 278.6 | 3052 | 1.005 |
| bounded | 1 | 10 | 1e-06 | 3.875 | 3.875 | 1.57e+06 | 3.407 | 6.227 | 3.369 | 3.365 | 3.365 | 0 | 1 |
| bounded | 1 | 10 | 0.0001 | 3.875 | 3.875 | 1.57e+06 | 3.417 | 6.273 | 3.378 | 3.374 | 3.374 | 0 | 1 |
| bounded | 1 | 10 | 0.01 | 3.916 | 3.916 | 1.77e+06 | 5.117 | 11.99 | 4.662 | 4.641 | 4.635 | 0 | 1 |
| bounded | 1 | 10 | 0.1 | 4.281 | 4.281 | 6.21e+06 | 3.7e+04 | 391.2 | 209.9 | 169.8 | 164.5 | 0 | 1.013 |
| bounded | 1 | 100 | 1e-06 | 4.228 | 3.927 | 1.44e+06 | 5.535 | 7.639 | 5.298 | 5.294 | 5.294 | 19 | 1 |
| bounded | 1 | 100 | 0.0001 | 4.229 | 3.927 | 1.44e+06 | 5.564 | 7.693 | 5.321 | 5.317 | 5.317 | 19 | 1 |
| bounded | 1 | 100 | 0.01 | 4.27 | 3.969 | 1.62e+06 | 10.48 | 14.31 | 8.303 | 8.294 | 8.275 | 19 | 1 |
| bounded | 1 | 100 | 0.1 | 4.647 | 4.351 | 5.72e+06 | 1.29e+52 | 494.3 | 344 | 303.7 | 283.5 | 19 | 1.01 |
| bounded | 1 | 1000 | 1e-06 | 5.29 | 4.049 | 1.15e+06 | 6.47 | 9.043 | 6.213 | 6.209 | 6.209 | 523 | 1 |
| bounded | 1 | 1000 | 0.0001 | 5.291 | 4.05 | 1.15e+06 | 6.51 | 9.105 | 6.245 | 6.242 | 6.242 | 523 | 1 |
| bounded | 1 | 1000 | 0.01 | 5.331 | 4.093 | 1.3e+06 | 12.78 | 16.88 | 10.29 | 10.28 | 10.26 | 523 | 1 |
| bounded | 1 | 1000 | 0.1 | 5.689 | 4.496 | 4.67e+06 | inf | 558.5 | 383.6 | 460 | 318.9 | 523 | 1.006 |
| bounded | 1 | 5000 | 1e-06 | 5.575 | 4.122 | 1.09e+06 | 7.378 | 10.57 | 7.141 | 7.137 | 7.137 | 1053 | 1 |
| bounded | 1 | 5000 | 0.0001 | 5.576 | 4.123 | 1.09e+06 | 7.423 | 10.64 | 7.178 | 7.175 | 7.174 | 1053 | 1 |
| bounded | 1 | 5000 | 0.01 | 5.616 | 4.167 | 1.23e+06 | 14.26 | 19.91 | 11.79 | 11.79 | 11.76 | 1053 | 1 |
| bounded | 1 | 5000 | 0.1 | 5.99 | 4.587 | 4.44e+06 | inf | 575.8 | 373.5 | 876.5 | 312.3 | 1057 | 1.003 |
| softplus | 0 | 10 | 1e-06 | 8.246 | 7.176 | 2.24e+06 | 6.556 | 27.41 | 6.131 | 6.083 | 6.083 | 3 | 1 |
| softplus | 0 | 10 | 0.0001 | 8.245 | 7.177 | 2.24e+06 | 6.596 | 27.58 | 6.159 | 6.11 | 6.109 | 3 | 1 |
| softplus | 0 | 10 | 0.01 | 8.131 | 7.041 | 2.66e+06 | 13.92 | 49.26 | 10.21 | 10 | 9.744 | 3 | 1.001 |
| softplus | 0 | 10 | 0.1 | 7.068 | 5.911 | 1.81e+07 | 2.84e+16 | 2613 | 1710 | 688.1 | 586.9 | 3 | 1.043 |
| softplus | 0 | 100 | 1e-06 | 19.51 | 13.44 | 9.45e+05 | 17.25 | 29.13 | 12.64 | 12.58 | 12.58 | 15 | 1 |
| softplus | 0 | 100 | 0.0001 | 19.51 | 13.44 | 9.46e+05 | 17.39 | 29.26 | 12.69 | 12.64 | 12.63 | 15 | 1 |
| softplus | 0 | 100 | 0.01 | 19.83 | 13.2 | 1.09e+06 | 55.17 | 45.53 | 19.56 | 19.69 | 18.73 | 15 | 1 |
| softplus | 0 | 100 | 0.1 | 23.16 | 11.17 | 5.52e+06 | 2.32e+195 | 1491 | 1227 | 398.3 | 352.6 | 15 | 1.022 |
| softplus | 0 | 1000 | 1e-06 | 25.69 | 17.73 | 7.18e+05 | 17.98 | 25.3 | 12.18 | 12.14 | 12.14 | 14 | 1 |
| softplus | 0 | 1000 | 0.0001 | 25.69 | 17.73 | 7.19e+05 | 18.14 | 25.41 | 12.23 | 12.2 | 12.19 | 14 | 1 |
| softplus | 0 | 1000 | 0.01 | 26.19 | 17.37 | 8.26e+05 | 82.03 | 38.52 | 18.48 | 19.41 | 17.94 | 14 | 1 |
| softplus | 0 | 1000 | 0.1 | 31.37 | 14.51 | 4.08e+06 | inf | 1139 | 904.5 | 598.2 | 300.4 | 14 | 1.019 |
| softplus | 0 | 5000 | 1e-06 | 60.19 | 16.3 | 3.06e+05 | 10.94 | 14.84 | 6.742 | 6.707 | 6.707 | 3058 | 1.017 |
| softplus | 0 | 5000 | 0.0001 | 59.21 | 16.3 | 3.12e+05 | 11.2 | 15.15 | 6.878 | 6.841 | 6.839 | 3060 | 1 |
| softplus | 0 | 5000 | 0.01 | 60.52 | 17.52 | 3.57e+05 | 39.71 | 22.34 | 9.66 | 10.1 | 9.386 | 3060 | 1 |
| softplus | 0 | 5000 | 0.1 | 73.63 | 20.3 | 1.74e+06 | inf | 578 | 446.1 | 1048 | 147.9 | 3060 | 1.008 |
| softplus | 1 | 10 | 1e-06 | 4.516 | 3.97 | 3.23e+06 | 5.605 | 18.03 | 5.169 | 5.148 | 5.148 | 4 | 1 |
| softplus | 1 | 10 | 0.0001 | 4.516 | 3.97 | 3.23e+06 | 5.635 | 18.13 | 5.19 | 5.168 | 5.168 | 4 | 1 |
| softplus | 1 | 10 | 0.01 | 4.56 | 4.015 | 3.73e+06 | 10.67 | 30.44 | 7.946 | 7.869 | 7.794 | 4 | 1 |
| softplus | 1 | 10 | 0.1 | 4.953 | 4.417 | 1.97e+07 | 3.23e+15 | 971.5 | 521 | 419.8 | 355.1 | 4 | 1.015 |
| softplus | 1 | 100 | 1e-06 | 5.408 | 4.272 | 2.69e+06 | 10.48 | 19.86 | 8.736 | 8.716 | 8.716 | 19 | 1 |
| softplus | 1 | 100 | 0.0001 | 5.409 | 4.273 | 2.7e+06 | 10.55 | 19.98 | 8.776 | 8.756 | 8.755 | 19 | 1 |
| softplus | 1 | 100 | 0.01 | 5.453 | 4.322 | 3.12e+06 | 25.45 | 34.72 | 13.93 | 13.9 | 13.68 | 19 | 1 |
| softplus | 1 | 100 | 0.1 | 5.858 | 4.765 | 1.66e+07 | 5.12e+191 | 1190 | 846.6 | 614.8 | 523.9 | 19 | 1.081 |
| softplus | 1 | 1000 | 1e-06 | 6.943 | 6.365 | 2.1e+06 | 14.55 | 27.33 | 11.69 | 11.68 | 11.68 | 329 | 1 |
| softplus | 1 | 1000 | 0.0001 | 6.944 | 6.366 | 2.1e+06 | 14.65 | 27.46 | 11.75 | 11.73 | 11.73 | 329 | 1 |
| softplus | 1 | 1000 | 0.01 | 7.019 | 6.475 | 2.42e+06 | 40.96 | 45.12 | 18.96 | 19.2 | 18.57 | 329 | 1 |
| softplus | 1 | 1000 | 0.1 | 7.763 | 7.593 | 1.26e+07 | inf | 1392 | 965.5 | 955 | 562.5 | 476 | 1.008 |
| softplus | 1 | 5000 | 1e-06 | 14.31 | 7.587 | 1.02e+06 | 12.5 | 17 | 9.516 | 9.497 | 9.497 | 3055 | 1 |
| softplus | 1 | 5000 | 0.0001 | 14.31 | 7.588 | 1.02e+06 | 12.59 | 17.08 | 9.554 | 9.536 | 9.534 | 3055 | 1 |
| softplus | 1 | 5000 | 0.01 | 14.64 | 7.732 | 1.16e+06 | 31.5 | 26.98 | 14.3 | 14.44 | 14.09 | 3055 | 1 |
| softplus | 1 | 5000 | 0.1 | 18.16 | 9.188 | 5.37e+06 | inf | 647.4 | 418.8 | 1168 | 261.7 | 3055 | 1.012 |

The data-free bound here is the arbitrary-perturbation reading of `incremental.json` at that `delta`, since the attacked point need not be a LayerNorm output. The ascent gain is the best value after the sign-gradient steps divided by the best starting value, so a gain of 1 means the gradient-sign start was already the best point found.

No attacked value exceeds its certificate. There are 0 violations of the data-free bound, checked per output coordinate, 0 of the local block bounds, checked per position and coordinate where they are finite, and 0 of the local end-to-end bounds.

The local block bound at `delta = 1e-2`, against the length:

| arm | block | rule | L = 10 | L = 100 | L = 1000 | L = 5000 |
| --- | --- | --- | --- | --- | --- | --- |
| bounded | 0 | additive | 31.12 | 108.6 | 185.7 | 271.2 |
| bounded | 0 | log_delta | 71.7 | 121.4 | 202.2 | 257.1 |
| bounded | 0 | log_delta_recursive | 27.27 | 61.88 | 111.2 | 132.3 |
| bounded | 0 | additive_endpoint | 26.98 | 61.63 | 111.4 | 132.5 |
| bounded | 0 | log_delta_endpoint | 26.92 | 60.97 | 110.4 | 131.7 |
| bounded | 1 | additive | 20.03 | 44.75 | 68.12 | 80.09 |
| bounded | 1 | log_delta | 46.94 | 61.1 | 90.01 | 111.8 |
| bounded | 1 | log_delta_recursive | 18.25 | 35.45 | 54.83 | 66.22 |
| bounded | 1 | additive_endpoint | 18.17 | 35.41 | 54.82 | 66.21 |
| bounded | 1 | log_delta_endpoint | 18.15 | 35.34 | 54.68 | 66.07 |
| softplus | 0 | additive | 113.2 | 1094 | 2149 | 2403 |
| softplus | 0 | log_delta | 400.6 | 902.8 | 1009 | 1352 |
| softplus | 0 | log_delta_recursive | 83.02 | 387.9 | 484.1 | 584.6 |
| softplus | 0 | additive_endpoint | 81.32 | 390.4 | 508.5 | 611.4 |
| softplus | 0 | log_delta_endpoint | 79.23 | 371.4 | 469.9 | 568 |
| softplus | 1 | additive | 48.66 | 138.8 | 287.5 | 461.2 |
| softplus | 1 | log_delta | 138.8 | 189.3 | 316.7 | 395.1 |
| softplus | 1 | log_delta_recursive | 36.23 | 75.94 | 133.1 | 209.4 |
| softplus | 1 | additive_endpoint | 35.88 | 75.77 | 134.8 | 211.4 |
| softplus | 1 | log_delta_endpoint | 35.54 | 74.59 | 130.3 | 206.4 |

## The prediction against sequence length

The spread of the prediction over an `eps` ball at the model input is what `scripts/08_domains.py` measures. Below it is divided by `2 eps`, so a constant row means a linear response, and it is the median over the windows. The last column is `|grad|_1` at the nominal input, the first-order value of every other column.

| arm | L | eps = 1e-06 | eps = 1e-05 | eps = 0.0001 | eps = 0.001 | eps = 0.01 | eps = 0.1 | first order |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 10 | 0.5394 | 0.5394 | 0.5394 | 0.5394 | 0.5394 | 0.547 | 0.5394 |
| bounded | 100 | 0.5737 | 0.5737 | 0.5737 | 0.5737 | 0.5736 | 0.5825 | 0.5737 |
| bounded | 1000 | 0.5768 | 0.5768 | 0.5768 | 0.5768 | 0.5768 | 0.5878 | 0.5768 |
| bounded | 5000 | 0.5748 | 0.5748 | 0.5748 | 0.5748 | 0.5749 | 0.5865 | 0.5748 |
| softplus | 10 | 1.006 | 1.006 | 1.006 | 1.006 | 1.006 | 1.029 | 1.006 |
| softplus | 100 | 1.262 | 1.262 | 1.262 | 1.262 | 1.263 | 1.297 | 1.262 |
| softplus | 1000 | 1.299 | 1.299 | 1.299 | 1.299 | 1.3 | 1.382 | 1.299 |
| softplus | 5000 | 1.289 | 1.289 | 1.289 | 1.289 | 1.289 | 1.384 | 1.289 |

At `eps = 1e-06` the attacked width divided by `2 eps` ranges from 0.5394 to 0.5768 on the bounded arm and from 1.006 to 1.299 on the softplus arm across L = 10, 100, 1000, 5000. Up to `eps = 1e-3` the attacked width differs from the first-order width by at most 0.000743 percent, so the prediction is locally linear at these scales; at `eps = 0.1` the attacked width is 1.014 to 1.073 times the first-order width.

The local end-to-end bound against the length, as the median width over the windows:

| arm | rule | eps | L = 10 | L = 100 | L = 1000 | L = 5000 |
| --- | --- | --- | --- | --- | --- | --- |
| bounded | additive | 1e-06 | 0.1081 | 0.3694 | 0.7558 | 0.8574 |
| bounded | additive | 1e-05 | 8.811 | 4.52e+14 | inf | inf |
| bounded | maximum | 1e-06 | 0.4617 | 0.895 | 6.659 | 13.35 |
| bounded | maximum | 1e-05 | 2.11e+07 | 6.52e+10 | 1.36e+14 | 1.95e+15 |
| bounded | recursive | 1e-06 | 0.1022 | 0.3082 | 0.5651 | 0.6191 |
| bounded | recursive | 1e-05 | 6.639 | 792 | 1.31e+10 | 1.86e+10 |
| bounded | additive endpoint | 1e-06 | 0.1019 | 0.3075 | 0.5641 | 0.6181 |
| bounded | additive endpoint | 1e-05 | 6.546 | 780.7 | 1.99e+10 | 8.75e+10 |
| bounded | endpoint | 1e-06 | 0.1019 | 0.3074 | 0.5637 | 0.6176 |
| bounded | endpoint | 1e-05 | 6.531 | 734.2 | 1.26e+10 | 1.79e+10 |
| softplus | additive | 1e-06 | 0.3336 | 7.06 | 49.43 | 188.6 |
| softplus | additive | 1e-05 | 1.95e+10 | inf | inf | inf |
| softplus | maximum | 1e-06 | 22.7 | 366 | 4.24e+10 | 1.06e+12 |
| softplus | maximum | 1e-05 | 2.59e+15 | 1.09e+18 | inf | inf |
| softplus | recursive | 1e-06 | 0.2922 | 2.864 | 8.6 | 10.61 |
| softplus | recursive | 1e-05 | 407 | 4.14e+12 | 3.27e+14 | 2.53e+15 |
| softplus | additive endpoint | 1e-06 | 0.2882 | 2.83 | 8.578 | 10.66 |
| softplus | additive endpoint | 1e-05 | 388.6 | 6.86e+12 | 5.81e+15 | 2.38e+17 |
| softplus | endpoint | 1e-06 | 0.2879 | 2.802 | 8.438 | 10.41 |
| softplus | endpoint | 1e-05 | 324.5 | 3.12e+12 | 1.71e+14 | 5.83e+14 |

At `eps = 1e-06` the local end-to-end bound stays finite with the additive rule on the bounded arm, going from 0.1081 at L = 10 to 0.8574 at L = 5000; it stays finite with the log-timescale, running maximum rule on the bounded arm, going from 0.4617 at L = 10 to 13.35 at L = 5000; it stays finite with the log-timescale, recursive rule on the bounded arm, going from 0.1022 at L = 10 to 0.6191 at L = 5000; it stays finite with the additive interval, endpoint rule on the bounded arm, going from 0.1019 at L = 10 to 0.6181 at L = 5000; it stays finite with the log-timescale, endpoint rule on the bounded arm, going from 0.1019 at L = 10 to 0.6176 at L = 5000; it stays finite with the additive rule on the softplus arm, going from 0.3336 at L = 10 to 188.6 at L = 5000; it stays finite with the log-timescale, running maximum rule on the softplus arm, going from 22.7 at L = 10 to 1.06e+12 at L = 5000; it stays finite with the log-timescale, recursive rule on the softplus arm, going from 0.2922 at L = 10 to 10.61 at L = 5000; it stays finite with the additive interval, endpoint rule on the softplus arm, going from 0.2882 at L = 10 to 10.66 at L = 5000; it stays finite with the log-timescale, endpoint rule on the softplus arm, going from 0.2879 at L = 10 to 10.41 at L = 5000.

## The local end-to-end certificate on the windows of domains.json

bounded arm: 4 windows of length 100, the first of the unshuffled test loader, as `scripts/08_domains.py` takes them. The global certified width of the prediction is 6.692.

| eps | attack width | additive | maximum | recursive | additive endpoint | endpoint | additive / attack | maximum / attack | recursive / attack | additive endpoint / attack | endpoint / attack |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-06 | 1.5e-06 | 1.291 | 10.17 | 0.8564 | 0.8514 | 0.8503 | 8.62e+05 | 6.79e+06 | 5.71e+05 | 5.68e+05 | 5.67e+05 |
| 1e-05 | 1.5e-05 | inf | 5.38e+14 | 6.36e+10 | 6.43e+10 | 5.53e+10 | inf | 3.59e+19 | 4.24e+15 | 4.29e+15 | 3.69e+15 |
| 0.0001 | 0.00015 | inf | inf | inf | 8.1e+28 | 7.46e+28 | inf | inf | inf | 5.4e+32 | 4.97e+32 |
| 0.001 | 0.0015 | inf | inf | inf | inf | inf | inf | inf | inf | inf | inf |
| 0.01 | 0.01499 | inf | inf | inf | inf | inf | inf | inf | inf | inf | inf |
| 0.1 | 0.1528 | nan | inf | inf | inf | inf | nan | inf | inf | inf | inf |

With the additive rule the bound is first non-finite at `eps = 1e-05`, from the stage `final LayerNorm` on, and first above the global certified width at `eps = 1e-05`. There the step multiplier in block 1 reaches 1.465, and 23.2 percent of the steps enlarge the carried bound. With the log-timescale, running maximum rule the bound is first non-finite at `eps = 0.0001`, from the stage `L1 recurrence` on, and first above the global certified width at `eps = 1e-06`. The step never scales the carried bound up; there the largest relative timescale perturbation in block 1 is `rho = 656.5`, beyond 88.7, where `e^rho - 1` overflows float32. With the log-timescale, recursive rule the bound is first non-finite at `eps = 0.0001`, from the stage `final LayerNorm` on, and first above the global certified width at `eps = 1e-05`. With the additive interval, endpoint rule the bound is first non-finite at `eps = 0.001`, from the stage `L1 readout` on, and first above the global certified width at `eps = 1e-05`. The step never scales the carried bound up; there the largest relative timescale perturbation in block 1 is `rho = 2.38e+14`, beyond 88.7, where `e^rho - 1` overflows float32. With the log-timescale, endpoint rule the bound is first non-finite at `eps = 0.001`, from the stage `L1 readout` on, and first above the global certified width at `eps = 1e-05`. The step never scales the carried bound up; there the largest relative timescale perturbation in block 1 is `rho = 1.69e+14`. The largest `eps` at which the bound is below the global certified width is 1e-06 with the additive rule, none of the tested with the log-timescale, running maximum rule, 1e-06 with the log-timescale, recursive rule, 1e-06 with the additive interval, endpoint rule, 1e-06 with the log-timescale, endpoint rule.

The additive rule here uses the tight slope of the timescale map, which on the bounded map is smaller than the slope behind the `lipschitz` column of `domains.json`; it is never above that column: yes.

softplus arm: 4 windows of length 100, the first of the unshuffled test loader, as `scripts/08_domains.py` takes them. The global certified width of the prediction is 5.818.

| eps | attack width | additive | maximum | recursive | additive endpoint | endpoint | additive / attack | maximum / attack | recursive / attack | additive endpoint / attack | endpoint / attack |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-06 | 3.8e-06 | 1.52e+12 | 3.75e+12 | 67.64 | 68.42 | 61.17 | 3.99e+17 | 9.87e+17 | 1.78e+07 | 1.8e+07 | 1.61e+07 |
| 1e-05 | 3.8e-05 | inf | inf | 3.37e+18 | 2.33e+17 | 1.67e+17 | inf | inf | 8.86e+22 | 6.14e+21 | 4.39e+21 |
| 0.0001 | 0.00038 | inf | inf | inf | inf | inf | inf | inf | inf | inf | inf |
| 0.001 | 0.0038 | inf | inf | inf | inf | inf | inf | inf | inf | inf | inf |
| 0.01 | 0.03803 | nan | inf | inf | inf | inf | nan | inf | inf | inf | inf |
| 0.1 | 0.3945 | nan | inf | inf | inf | inf | nan | inf | inf | inf | inf |

With the additive rule the bound is first non-finite at `eps = 1e-05`, from the stage `L1 recurrence` on, and first above the global certified width at `eps = 1e-06`. There the step multiplier in block 1 reaches 137.6, and 95.88 percent of the steps enlarge the carried bound. With the log-timescale, running maximum rule the bound is first non-finite at `eps = 1e-05`, from the stage `final LayerNorm` on, and first above the global certified width at `eps = 1e-06`. With the log-timescale, recursive rule the bound is first non-finite at `eps = 0.0001`, from the stage `L1 recurrence` on, and first above the global certified width at `eps = 1e-06`. The step never scales the carried bound up; there the largest relative timescale perturbation in block 1 is `rho = 1213`, beyond 88.7, where `e^rho - 1` overflows float32. With the additive interval, endpoint rule the bound is first non-finite at `eps = 0.0001`, from the stage `final LayerNorm` on, and first above the global certified width at `eps = 1e-06`. With the log-timescale, endpoint rule the bound is first non-finite at `eps = 0.0001`, from the stage `final LayerNorm` on, and first above the global certified width at `eps = 1e-06`. The largest `eps` at which the bound is below the global certified width is none of the tested with the additive rule, none of the tested with the log-timescale, running maximum rule, none of the tested with the log-timescale, recursive rule, none of the tested with the additive interval, endpoint rule, none of the tested with the log-timescale, endpoint rule.

## Where the end-to-end bound comes from

The domain carries one sensitivity per coordinate from the input, where it is 1, to the prediction. The stages form a chain, so the bound is the product of the stage ratios `max(after) / max(before)`, and its `log10` is the sum of theirs. The groups add those logs. LayerNorm is the stem, block and final norms. Feed-forward is every Linear, the convolution, every SiLU, the gate product and the head. Recurrence is the scan and its readout `C h + D u`. Residual is the skip connection. A negative entry is a stage whose largest sensitivity is smaller than its input's. Only `eps` values with a finite bound appear.

| arm | rule | eps | LayerNorm | feed-forward | recurrence | residual | total, log10 of the bound |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | additive | 1e-06 | 2.487 | 1.508 | 1.814 | 0.01456 | 5.823 |
| bounded | maximum | 1e-06 | 2.731 | 1.805 | 2.176 | 0.00636 | 6.718 |
| bounded | recursive | 1e-06 | 2.46 | 1.439 | 1.727 | 0.01616 | 5.643 |
| bounded | additive endpoint | 1e-06 | 2.46 | 1.438 | 1.726 | 0.01619 | 5.641 |
| bounded | endpoint | 1e-06 | 2.46 | 1.438 | 1.726 | 0.01619 | 5.64 |
| bounded | maximum | 1e-05 | 14.2 | 2.171 | 3.198 | 0.0016 | 19.57 |
| bounded | recursive | 1e-05 | 11.46 | 1.905 | 2.23 | 0.00667 | 15.6 |
| bounded | additive endpoint | 1e-05 | 11.44 | 1.92 | 2.226 | 0.0067 | 15.59 |
| bounded | endpoint | 1e-05 | 11.41 | 1.893 | 2.226 | 0.00675 | 15.54 |
| bounded | additive endpoint | 0.0001 | 24.56 | 2.972 | 5.318 | 0.00409 | 32.86 |
| bounded | endpoint | 0.0001 | 24.53 | 2.968 | 5.313 | 0.0041 | 32.82 |
| softplus | additive | 1e-06 | 12.21 | 0.5959 | 5.448 | 0.00155 | 18.26 |
| softplus | maximum | 1e-06 | 12.39 | 1.991 | 3.968 | 0.000644 | 18.35 |
| softplus | recursive | 1e-06 | 3.577 | 0.7767 | 3.196 | 0.00454 | 7.555 |
| softplus | additive endpoint | 1e-06 | 3.556 | 0.8046 | 3.195 | 0.00457 | 7.56 |
| softplus | endpoint | 1e-06 | 3.516 | 0.8016 | 3.187 | 0.00462 | 7.508 |
| softplus | recursive | 1e-05 | 16.39 | 1.791 | 5.089 | 0.00193 | 23.27 |
| softplus | additive endpoint | 1e-05 | 15.58 | 2.407 | 4.089 | 0.00196 | 22.08 |
| softplus | endpoint | 1e-05 | 15.51 | 2.369 | 4.049 | 0.00196 | 21.94 |

On the bounded arm at `eps = 1e-06`, the recurrence group splits into 0.9455 decades from the scans and 0.8686 from the readouts with the additive rule; 1.509 decades from the scans and 0.6669 from the readouts with the log-timescale, running maximum rule; 0.7659 decades from the scans and 0.9614 from the readouts with the log-timescale, recursive rule; 0.7633 decades from the scans and 0.9626 from the readouts with the additive interval, endpoint rule; 0.7633 decades from the scans and 0.9626 from the readouts with the log-timescale, endpoint rule. The timescale rule changes the scan stages only; LayerNorm and the feed-forward stages differ between the rules only through the sensitivity that reaches them.

The attacked sensitivity of the prediction on these windows is 0.7494 on the bounded arm and 1.9 on the softplus arm (width over `2 eps` at `eps = 1e-06`). The local end-to-end bound exceeds it by a factor of 6.79e+06 on the bounded arm, 9.87e+17 on the softplus arm with the log-timescale, running maximum rule; 5.71e+05 on the bounded arm, 1.78e+07 on the softplus arm with the log-timescale, recursive rule; 5.68e+05 on the bounded arm, 1.8e+07 on the softplus arm with the additive interval, endpoint rule; 5.67e+05 on the bounded arm, 1.61e+07 on the softplus arm with the log-timescale, endpoint rule. On the attack windows of length 100 the factor is 7.8e+05 on the bounded arm, 1.45e+08 on the softplus arm with the log-timescale, running maximum rule; 2.69e+05 on the bounded arm, 1.13e+06 on the softplus arm with the log-timescale, recursive rule; 2.68e+05 on the bounded arm, 1.12e+06 on the softplus arm with the additive interval, endpoint rule; 2.68e+05 on the bounded arm, 1.11e+06 on the softplus arm with the log-timescale, endpoint rule. On those same windows, around one block's own normalised input at `delta = 1e-06`, the same domain is 7.639 to 29.13 times the attacked block slope with the log-timescale, running maximum rule; 5.298 to 12.64 times the attacked block slope with the log-timescale, recursive rule; 5.294 to 12.58 times the attacked block slope with the additive interval, endpoint rule; 5.294 to 12.58 times the attacked block slope with the log-timescale, endpoint rule, and there it covers the block's linear layers, convolution, gate and recurrence. The two measurements place most of the end-to-end excess outside the block interiors: in the LayerNorms, the stem and the head, and in chaining stage bounds that are each computed on their own.

Stage by stage on the bounded arm at `eps = 1e-06`, as the largest sensitivity over the windows, positions and coordinates:

| stage | group | additive | log10 ratio | maximum | log10 ratio | recursive | log10 ratio | additive endpoint | log10 ratio | endpoint | log10 ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stem Linear | feed-forward | 4.355 | 0.639 | 4.355 | 0.639 | 4.355 | 0.639 | 4.355 | 0.639 | 4.355 | 0.639 |
| stem SiLU | feed-forward | 4.176 | -0.01821 | 4.176 | -0.01821 | 4.176 | -0.01821 | 4.176 | -0.01821 | 4.176 | -0.01821 |
| stem LayerNorm | LayerNorm | 50.75 | 1.085 | 50.75 | 1.085 | 50.75 | 1.085 | 50.75 | 1.085 | 50.75 | 1.085 |
| L0 LayerNorm | LayerNorm | 139.3 | 0.4385 | 139.3 | 0.4385 | 139.3 | 0.4385 | 139.3 | 0.4385 | 139.3 | 0.4385 |
| L0 in_proj | feed-forward | 345.9 | 0.3951 | 345.9 | 0.3951 | 345.9 | 0.3951 | 345.9 | 0.3951 | 345.9 | 0.3951 |
| L0 conv | feed-forward | 502.8 | 0.1624 | 502.8 | 0.1624 | 502.8 | 0.1624 | 502.8 | 0.1624 | 502.8 | 0.1624 |
| L0 SiLU | feed-forward | 518.1 | 0.01301 | 518.1 | 0.01301 | 518.1 | 0.01301 | 518.1 | 0.01301 | 518.1 | 0.01301 |
| L0 recurrence | recurrence | 2003 | 0.5874 | 4328 | 0.9219 | 1371 | 0.4227 | 1362 | 0.4199 | 1362 | 0.4199 |
| L0 readout | recurrence | 5404 | 0.4309 | 8806 | 0.3085 | 4550 | 0.521 | 4535 | 0.5223 | 4535 | 0.5223 |
| L0 gate | feed-forward | 3225 | -0.2242 | 6580 | -0.1265 | 2987 | -0.1827 | 2974 | -0.1832 | 2974 | -0.1832 |
| L0 out_proj | feed-forward | 3065 | -0.02205 | 5082 | -0.1122 | 2705 | -0.04314 | 2700 | -0.042 | 2700 | -0.04201 |
| L0 residual | residual | 3098 | 0.00461 | 5100 | 0.00151 | 2738 | 0.00521 | 2733 | 0.00522 | 2733 | 0.00522 |
| L1 LayerNorm | LayerNorm | 9819 | 0.5011 | 1.77e+04 | 0.5396 | 8632 | 0.4987 | 8618 | 0.4988 | 8618 | 0.4988 |
| L1 in_proj | feed-forward | 2.91e+04 | 0.4723 | 5.1e+04 | 0.46 | 2.59e+04 | 0.4773 | 2.59e+04 | 0.4772 | 2.59e+04 | 0.4772 |
| L1 conv | feed-forward | 3.92e+04 | 0.129 | 6.69e+04 | 0.1186 | 3.53e+04 | 0.134 | 3.52e+04 | 0.1342 | 3.52e+04 | 0.1342 |
| L1 SiLU | feed-forward | 2.98e+04 | -0.1189 | 5.25e+04 | -0.1054 | 2.67e+04 | -0.1214 | 2.66e+04 | -0.1217 | 2.66e+04 | -0.1217 |
| L1 recurrence | recurrence | 6.8e+04 | 0.3582 | 2.03e+05 | 0.587 | 5.88e+04 | 0.3433 | 5.87e+04 | 0.3434 | 5.87e+04 | 0.3435 |
| L1 readout | recurrence | 1.86e+05 | 0.4377 | 4.63e+05 | 0.3584 | 1.62e+05 | 0.4405 | 1.62e+05 | 0.4404 | 1.62e+05 | 0.4404 |
| L1 gate | feed-forward | 2.03e+05 | 0.03665 | 9.75e+05 | 0.3233 | 1.79e+05 | 0.04317 | 1.79e+05 | 0.04303 | 1.79e+05 | 0.04302 |
| L1 out_proj | feed-forward | 1.25e+05 | -0.209 | 4.25e+05 | -0.3608 | 1.03e+05 | -0.2389 | 1.03e+05 | -0.2393 | 1.03e+05 | -0.2394 |
| L1 residual | residual | 1.28e+05 | 0.00995 | 4.29e+05 | 0.00485 | 1.06e+05 | 0.01095 | 1.06e+05 | 0.01096 | 1.06e+05 | 0.01097 |
| final LayerNorm | LayerNorm | 3.72e+05 | 0.4624 | 2e+06 | 0.6679 | 2.9e+05 | 0.4383 | 2.89e+05 | 0.438 | 2.89e+05 | 0.438 |
| last position | feed-forward | 3.22e+05 | -0.06321 | 1.59e+06 | -0.09826 | 2.41e+05 | -0.08175 | 2.4e+05 | -0.08207 | 2.39e+05 | -0.08236 |
| head Linear and SiLU | feed-forward | 7.68e+05 | 0.3783 | 3.96e+06 | 0.395 | 5.45e+05 | 0.355 | 5.43e+05 | 0.3549 | 5.42e+05 | 0.3548 |
| head Linear | feed-forward | 6.65e+05 | -0.06245 | 5.22e+06 | 0.1203 | 4.39e+05 | -0.0936 | 4.37e+05 | -0.09394 | 4.36e+05 | -0.09404 |

Inside each recurrence, per block: `rho` is the largest relative timescale perturbation, `L_log_delta eps`. The step multiplier is the largest factor that one step applies to the carried bound: `Abar + |A| exp(-|A| dt_lo) L_dt eps` for the additive rule, which can exceed 1; 1 for the running maximum; and the slowest admissible perturbed transition `exp(-|A| dt e^-rho)`, below 1, for the recursive rule; and for the endpoint rule the upper end of the interval it maximises over, the slowest transition that both the log and the additive timescale budgets admit, at most the recursive rule's. The fraction growing counts the (cell, step) pairs with a multiplier above 1; the fraction raised counts those at which the carried bound increases at all. The last two columns are the largest state sensitivity with the timescale's own sensitivity and without it. The run without it is not a bound; it measures how much of the recurrence's growth comes through the timescale.

| arm | rule | eps | block | rho | step multiplier | fraction of steps growing | fraction of steps raised | state sensitivity | without the timescale term |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | additive | 1e-06 | 0 | - | 0.9989 | 0 | 0.6995 | 2003 | 1119 |
| bounded | additive | 1e-06 | 1 | - | 0.9989 | 0 | 0.7586 | 6.8e+04 | 6.36e+04 |
| bounded | maximum | 1e-06 | 0 | 0.00171 | 1 | 0 | 0.0776 | 4328 | 2477 |
| bounded | maximum | 1e-06 | 1 | 0.203 | 1 | 0 | 0.148 | 2.03e+05 | 2e+05 |
| bounded | recursive | 1e-06 | 0 | 0.00171 | 0.9989 | 0 | 0.6598 | 1371 | 1119 |
| bounded | recursive | 1e-06 | 1 | 0.1054 | 0.9989 | 0 | 0.711 | 5.88e+04 | 5.82e+04 |
| bounded | additive endpoint | 1e-06 | 0 | 0.00171 | 0.9989 | 0 | 0.6629 | 1362 | 1119 |
| bounded | additive endpoint | 1e-06 | 1 | 0.1053 | 0.9989 | 0 | 0.7162 | 5.87e+04 | 5.81e+04 |
| bounded | endpoint | 1e-06 | 0 | 0.00171 | 0.9989 | 0 | 0.6629 | 1362 | 1119 |
| bounded | endpoint | 1e-06 | 1 | 0.1053 | 0.9989 | 0 | 0.7155 | 5.87e+04 | 5.81e+04 |
| bounded | additive | 1e-05 | 0 | - | 0.9989 | 0 | 0.7015 | 2040 | 1126 |
| bounded | additive | 1e-05 | 1 | - | 1.465 | 0.232 | 0.9281 | 4.67e+20 | 9.59e+04 |
| bounded | maximum | 1e-05 | 0 | 0.01714 | 1 | 0 | 0.07757 | 4357 | 2491 |
| bounded | maximum | 1e-05 | 1 | 3.072 | 1 | 0 | 0.1866 | 4.72e+05 | 4.53e+05 |
| bounded | recursive | 1e-05 | 0 | 0.01714 | 0.9989 | 0 | 0.6591 | 1383 | 1126 |
| bounded | recursive | 1e-05 | 1 | 1.335 | 0.9993 | 0 | 0.6728 | 8.44e+04 | 8.11e+04 |
| bounded | additive endpoint | 1e-05 | 0 | 0.01714 | 0.9989 | 0 | 0.6629 | 1372 | 1126 |
| bounded | additive endpoint | 1e-05 | 1 | 1.333 | 1 | 0 | 0.7815 | 8.36e+04 | 8.09e+04 |
| bounded | endpoint | 1e-05 | 0 | 0.01714 | 0.9989 | 0 | 0.6627 | 1372 | 1126 |
| bounded | endpoint | 1e-05 | 1 | 1.333 | 0.9993 | 0 | 0.7141 | 8.36e+04 | 8.09e+04 |
| bounded | additive | 0.0001 | 0 | - | 0.9989 | 0 | 0.7229 | 2462 | 1192 |
| bounded | additive | 0.0001 | 1 | - | 48.68 | 0.9833 | 0.4726 | inf | 1.16e+07 |
| bounded | maximum | 0.0001 | 0 | 0.1775 | 1 | 0 | 0.07728 | 4665 | 2640 |
| bounded | maximum | 0.0001 | 1 | 656.5 | 1 | 0 | 0.09748 | inf | 8.1e+08 |
| bounded | recursive | 0.0001 | 0 | 0.1775 | 0.9989 | 0 | 0.6497 | 1518 | 1192 |
| bounded | recursive | 0.0001 | 1 | 68.77 | 1 | 0 | 0.247 | 8.73e+31 | 4.54e+06 |
| bounded | additive endpoint | 0.0001 | 0 | 0.1775 | 0.9989 | 0 | 0.6613 | 1472 | 1192 |
| bounded | additive endpoint | 0.0001 | 1 | 68.24 | 1 | 0 | 0.9861 | 8.57e+06 | 4.47e+06 |
| bounded | endpoint | 0.0001 | 0 | 0.1775 | 0.9989 | 0 | 0.6597 | 1472 | 1192 |
| bounded | endpoint | 0.0001 | 1 | 67.82 | 1 | 0 | 0.9696 | 8.46e+06 | 4.41e+06 |
| bounded | additive | 0.001 | 0 | - | 1.893 | 0.6819 | 0.9411 | inf | 2093 |
| bounded | additive | 0.001 | 1 | - | inf | 1 | 0.03026 | inf | nan |
| bounded | maximum | 0.001 | 0 | 2.519 | 1 | 0 | 0.06721 | 1.83e+04 | 4655 |
| bounded | maximum | 0.001 | 1 | 2.55e+15 | 1 | 0 | 0.01 | inf | 1.22e+33 |
| bounded | recursive | 0.001 | 0 | 2.519 | 0.9999 | 0 | 0.4097 | 1.29e+04 | 2093 |
| bounded | recursive | 0.001 | 1 | 3.35e+14 | 1 | 0 | 0.01047 | inf | 8.59e+30 |
| bounded | additive endpoint | 0.001 | 0 | 2.519 | 1 | 0 | 0.8539 | 2857 | 2093 |
| bounded | additive endpoint | 0.001 | 1 | 2.38e+14 | 1 | 0 | 0.3548 | 1.07e+31 | 4.5e+30 |
| bounded | endpoint | 0.001 | 0 | 2.519 | 0.9999 | 0 | 0.5567 | 2857 | 2093 |
| bounded | endpoint | 0.001 | 1 | 1.69e+14 | 1 | 0 | 0.3068 | 5.37e+30 | 2.23e+30 |
| bounded | additive | 0.01 | 0 | - | 104.5 | 1 | 0.2605 | inf | 3.72e+05 |
| bounded | additive | 0.01 | 1 | - | inf | 1 | 0.01 | inf | nan |
| bounded | maximum | 0.01 | 0 | 244.8 | 1 | 0 | 0.03355 | inf | 7.98e+05 |
| bounded | maximum | 0.01 | 1 | inf | 1 | 0 | 0.01 | inf | inf |
| bounded | recursive | 0.01 | 0 | 244.8 | 1 | 0 | 0.03364 | inf | 3.72e+05 |
| bounded | recursive | 0.01 | 1 | inf | 1 | 0 | 0.01 | inf | inf |
| bounded | additive endpoint | 0.01 | 0 | 244.8 | 1 | 0 | 0.8623 | 7.79e+05 | 3.72e+05 |
| bounded | additive endpoint | 0.01 | 1 | 8.96e+35 | 1 | 0 | 0.01 | inf | inf |
| bounded | endpoint | 0.01 | 0 | 244.8 | 1 | 0 | 0.8623 | 7.79e+05 | 3.72e+05 |
| bounded | endpoint | 0.01 | 1 | 8.96e+35 | 1 | 0 | 0.01 | inf | inf |
| bounded | additive | 0.1 | 0 | - | 6.03e+27 | 1 | 0.01 | inf | nan |
| bounded | additive | 0.1 | 1 | - | inf | 1 | 0 | nan | nan |
| bounded | maximum | 0.1 | 0 | 1.42e+28 | 1 | 0 | 0.01 | inf | inf |
| bounded | maximum | 0.1 | 1 | inf | 1 | 0 | 0.01 | inf | inf |
| bounded | recursive | 0.1 | 0 | 1.42e+28 | 1 | 0 | 0.01 | inf | inf |
| bounded | recursive | 0.1 | 1 | inf | 1 | 0 | 0.01 | inf | inf |
| bounded | additive endpoint | 0.1 | 0 | 1.42e+28 | 1 | 0 | 0.01 | inf | inf |
| bounded | additive endpoint | 0.1 | 1 | inf | 1 | 0 | 0.01 | inf | inf |
| bounded | endpoint | 0.1 | 0 | 1.42e+28 | 1 | 0 | 0.01 | inf | inf |
| bounded | endpoint | 0.1 | 1 | inf | 1 | 0 | 0.01 | inf | inf |
| softplus | additive | 1e-06 | 0 | - | 0.9999 | 0 | 0.6984 | 9101 | 2069 |
| softplus | additive | 1e-06 | 1 | - | 0.9995 | 0 | 0.8496 | 1.02e+08 | 6.2e+05 |
| softplus | maximum | 1e-06 | 0 | 0.00204 | 1 | 0 | 0.06977 | 1.54e+04 | 5758 |
| softplus | maximum | 1e-06 | 1 | 1.23 | 1 | 0 | 0.09717 | 4.69e+06 | 2.43e+06 |
| softplus | recursive | 1e-06 | 0 | 0.00204 | 0.9999 | 0 | 0.644 | 4320 | 2069 |
| softplus | recursive | 1e-06 | 1 | 0.311 | 0.9994 | 0 | 0.6594 | 6.75e+05 | 4.64e+05 |
| softplus | additive endpoint | 1e-06 | 0 | 0.00204 | 0.9999 | 0 | 0.649 | 4250 | 2069 |
| softplus | additive endpoint | 1e-06 | 1 | 0.3098 | 0.9995 | 0 | 0.6875 | 6.67e+05 | 4.63e+05 |
| softplus | endpoint | 1e-06 | 0 | 0.00204 | 0.9999 | 0 | 0.6489 | 4247 | 2069 |
| softplus | endpoint | 1e-06 | 1 | 0.3098 | 0.9994 | 0 | 0.6793 | 6.42e+05 | 4.62e+05 |
| softplus | additive | 1e-05 | 0 | - | 0.9999 | 0 | 0.6999 | 9434 | 2078 |
| softplus | additive | 1e-05 | 1 | - | 137.6 | 0.9588 | 0.5242 | inf | 1.85e+06 |
| softplus | maximum | 1e-05 | 0 | 0.02046 | 1 | 0 | 0.0698 | 1.55e+04 | 5784 |
| softplus | maximum | 1e-05 | 1 | 38.35 | 1 | 0 | 0.09337 | 5.04e+20 | 2.87e+07 |
| softplus | recursive | 1e-05 | 0 | 0.02046 | 0.9999 | 0 | 0.6428 | 4370 | 2078 |
| softplus | recursive | 1e-05 | 1 | 5.29 | 1 | 0 | 0.3445 | 2.2e+07 | 1.04e+06 |
| softplus | additive endpoint | 1e-05 | 0 | 0.02046 | 0.9999 | 0 | 0.6487 | 4309 | 2078 |
| softplus | additive endpoint | 1e-05 | 1 | 5.261 | 1 | 0 | 0.9991 | 1.75e+06 | 1.03e+06 |
| softplus | endpoint | 1e-05 | 0 | 0.02046 | 0.9999 | 0 | 0.6486 | 4272 | 2078 |
| softplus | endpoint | 1e-05 | 1 | 5.25 | 1 | 0 | 0.6174 | 1.59e+06 | 1.03e+06 |
| softplus | additive | 0.0001 | 0 | - | 0.9999 | 0 | 0.7201 | 1.44e+04 | 2175 |
| softplus | additive | 0.0001 | 1 | - | 5.55e+05 | 0.992 | 0.1342 | inf | 7.79e+11 |
| softplus | maximum | 0.0001 | 0 | 0.2104 | 1 | 0 | 0.0701 | 1.65e+04 | 6055 |
| softplus | maximum | 0.0001 | 1 | 1.1e+11 | 1 | 0 | 0.03 | inf | 2.2e+25 |
| softplus | recursive | 0.0001 | 0 | 0.2104 | 0.9999 | 0 | 0.6294 | 4899 | 2175 |
| softplus | recursive | 0.0001 | 1 | 1213 | 1 | 0 | 0.08125 | inf | 1.47e+09 |
| softplus | additive endpoint | 0.0001 | 0 | 0.2104 | 0.9999 | 0 | 0.646 | 5034 | 2175 |
| softplus | additive endpoint | 0.0001 | 1 | 1187 | 1 | 0 | 0.5593 | 2.56e+09 | 1.46e+09 |
| softplus | endpoint | 0.0001 | 0 | 0.2104 | 0.9999 | 0 | 0.6438 | 4539 | 2175 |
| softplus | endpoint | 0.0001 | 1 | 1101 | 1 | 0 | 0.578 | 2.2e+09 | 1.25e+09 |
| softplus | additive | 0.001 | 0 | - | 15.96 | 0.9536 | 0.9132 | inf | 3434 |
| softplus | additive | 0.001 | 1 | - | inf | 1 | 0.03 | inf | nan |
| softplus | maximum | 0.001 | 0 | 2.774 | 1 | 0 | 0.06641 | 3.81e+04 | 9574 |
| softplus | maximum | 0.001 | 1 | 7.48e+16 | 1 | 0 | 0.01 | inf | 1.04e+36 |
| softplus | recursive | 0.001 | 0 | 2.774 | 1 | 0 | 0.4171 | 2.65e+04 | 3434 |
| softplus | recursive | 0.001 | 1 | 5.29e+15 | 1 | 0 | 0.01 | inf | 2.78e+33 |
| softplus | additive endpoint | 0.001 | 0 | 2.774 | 1 | 0 | 0.986 | 1.41e+04 | 3434 |
| softplus | additive endpoint | 0.001 | 1 | 7.47e+15 | 1 | 0 | 0.4573 | 1.03e+34 | 5.5e+33 |
| softplus | endpoint | 0.001 | 0 | 2.774 | 1 | 0 | 0.5551 | 8700 | 3434 |
| softplus | endpoint | 0.001 | 1 | 2.01e+15 | 1 | 0 | 0.3774 | 7.68e+32 | 3.97e+32 |
| softplus | additive | 0.01 | 0 | - | 3735 | 1 | 0.1238 | inf | 4.31e+05 |
| softplus | additive | 0.01 | 1 | - | inf | 1 | 0 | nan | nan |
| softplus | maximum | 0.01 | 0 | 226 | 1 | 0 | 0.04614 | inf | 9.69e+05 |
| softplus | maximum | 0.01 | 1 | inf | 1 | 0 | 0.01 | inf | inf |
| softplus | recursive | 0.01 | 0 | 226 | 1 | 0 | 0.0465 | inf | 4.31e+05 |
| softplus | recursive | 0.01 | 1 | inf | 1 | 0 | 0.01 | inf | inf |
| softplus | additive endpoint | 0.01 | 0 | 226 | 1 | 0 | 0.8957 | 9.69e+05 | 4.31e+05 |
| softplus | additive endpoint | 0.01 | 1 | 1.48e+36 | 1 | 0 | 0.01 | inf | inf |
| softplus | endpoint | 0.01 | 0 | 226 | 1 | 0 | 0.8957 | 9.69e+05 | 4.31e+05 |
| softplus | endpoint | 0.01 | 1 | 1.48e+36 | 1 | 0 | 0.01 | inf | inf |
| softplus | additive | 0.1 | 0 | - | 3.74e+29 | 1 | 0.01 | inf | nan |
| softplus | additive | 0.1 | 1 | - | inf | 1 | 0 | nan | nan |
| softplus | maximum | 0.1 | 0 | 2.26e+28 | 1 | 0 | 0.01 | inf | inf |
| softplus | maximum | 0.1 | 1 | inf | 1 | 0 | 0.01 | inf | inf |
| softplus | recursive | 0.1 | 0 | 2.26e+28 | 1 | 0 | 0.01 | inf | inf |
| softplus | recursive | 0.1 | 1 | inf | 1 | 0 | 0.01 | inf | inf |
| softplus | additive endpoint | 0.1 | 0 | 2.26e+28 | 1 | 0 | 0.01 | inf | inf |
| softplus | additive endpoint | 0.1 | 1 | inf | 1 | 0 | 0.01 | inf | inf |
| softplus | endpoint | 0.1 | 0 | 2.26e+28 | 1 | 0 | 0.01 | inf | inf |
| softplus | endpoint | 0.1 | 1 | inf | 1 | 0 | 0.01 | inf | inf |

## Runtime and what was not run

The run took 47.5 minutes on Windows-11-10.0.26200-SP0, on the CPU. Per arm and length, the block and model attacks took:

| arm | L | block attack, seconds | model attack, seconds |
| --- | --- | --- | --- |
| bounded | 10 | 2.119 | 1.041 |
| bounded | 100 | 15.53 | 6.91 |
| bounded | 1000 | 152.3 | 67.53 |
| bounded | 5000 | 834.6 | 364.4 |
| softplus | 10 | 2.692 | 1.068 |
| softplus | 100 | 15.99 | 7.247 |
| softplus | 1000 | 151.5 | 67.64 |
| softplus | 5000 | 777.6 | 330.9 |

The block attack has one random-corner restart per `delta` and the model attack has none. The same 4 windows are used at every length, and no further windows were run. The attack is a heuristic search, so every attacked value is a lower bound and the true worst case may be larger.

## Scope

The data-free and local bounds hold over the reals. The local end-to-end bound is evaluated in float32, as `domains.json` was, and the local block bound in float64. The data-free incremental certificate is per block, with respect to the normalised input. A data-free end-to-end counterpart would pass each LayerNorm with its global Lipschitz constant, which exists because `eps > 0`: `max|gamma| / sqrt(eps)` in l2 is 331.5, 331 for the blocks of the bounded arm; 343.2, 334.3 for the blocks of the softplus arm, against about 1 at realised inputs (`incremental.json`). It is not computed here.
