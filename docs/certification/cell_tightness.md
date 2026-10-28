# Tightness of the cell theorem

Written by `scripts/16_cell_tightness.py`. Everything is in float64 with `lam = 1.0`, `M = 1.0` and both runs started from `h_{-1} = 0`. Each gap is the largest `|h_{T-1} - h~_{T-1}|` that Adam ascent found over a batch of restarts (600 steps, learning rate 0.1 decaying to 0.002): 64 random starts and alternating starts in which `c~` follows a square wave with runs of length 1 to 16 and the perturbed timescale is the faster step on the `+` runs and the slower on the `-` runs, at a grid of base timescales. The best iterate of every restart is also evaluated with every variable that enters through `tanh` set to its sign. The winning sequence is re-run with the plain recursion, and that value is the one reported. Part A searches the hypothesis class of Theorem 3 with the nominal timescale `x_t = lam dt_t` free in `[0.0001, 10]`. Part B searches the perturbation model of the paper, `dt = map(z)` with `z` in a box `Z` and `|z~ - z| <= zeta` in `Z`, with `eta = 0`. `B = eta + 2 M (e^rho - 1)` is the theorem's bound and `eta + 2 M tanh(rho/2)` is the value that the construction of Proposition 4 approaches, below which the paper proves no bound holds; `cell_bound_common_start` also uses it as an upper bound, which the paper does not state. A search is a lower bound on the worst case, not the worst case itself.

No gap exceeded a bound it should respect: largest `G/B` in Part A `0.9159`, largest gap `|h_t - h~_t|` over the smallest valid bound at its own length `t + 1` in Part B `0.4975`. Run time 290 s.

## Part A: `G/B` on the theorem's hypothesis class

| rho | eta | T = 10 | T = 30 | T = 100 | T = 300 | T = 1000 | common-start constant / B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0100 | 0.0000 | 0.4923 | 0.4969 | 0.4974 | 0.4975 | 0.4975 | 0.4975 |
| 0.0100 | 0.1000 | 0.9150 | 0.9158 | 0.9159 | 0.9159 | 0.9159 | 0.9159 |
| 0.0300 | 0.0000 | 0.4873 | 0.4919 | 0.4924 | 0.4925 | 0.4925 | 0.4925 |
| 0.0300 | 0.1000 | 0.8058 | 0.8077 | 0.8078 | 0.8079 | 0.8079 | 0.8079 |
| 0.1000 | 0.0000 | 0.4700 | 0.4744 | 0.4749 | 0.4750 | 0.4750 | 0.4750 |
| 0.1000 | 0.1000 | 0.6407 | 0.6438 | 0.6441 | 0.6442 | 0.6442 | 0.6442 |
| 0.3000 | 0.0000 | 0.4212 | 0.4250 | 0.4255 | 0.4256 | 0.4256 | 0.4256 |
| 0.3000 | 0.1000 | 0.4935 | 0.4969 | 0.4973 | 0.4974 | 0.4974 | 0.4974 |
| 0.6000 | 0.0000 | 0.3509 | 0.3539 | 0.3543 | 0.3543 | 0.3543 | 0.3543 |
| 0.6000 | 0.1000 | 0.3881 | 0.3910 | 0.3913 | 0.3914 | 0.3914 | 0.3914 |

At the longest length, `T = 1000`:

| rho | eta | G | B | common-start constant | G / constant | median x | sign changes of c~ | best start |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0100 | 0.0000 | 0.0100 | 0.0201 | 0.0100 | 1.0000 | 0.0116 | 999 | k=1, x0=0.01, constant, w=plus |
| 0.0100 | 0.1000 | 0.1100 | 0.1201 | 0.1100 | 1.0000 | 0.0183 | 999 | k=1, x0=0.01, constant, w=minus |
| 0.0300 | 0.0000 | 0.0300 | 0.0609 | 0.0300 | 1.0000 | 0.0108 | 999 | k=1, x0=0.01, constant, w=plus |
| 0.0300 | 0.1000 | 0.1300 | 0.1609 | 0.1300 | 1.0000 | 0.0115 | 999 | k=1, x0=0.01, constant, w=minus |
| 0.1000 | 0.0000 | 0.0999 | 0.2103 | 0.0999 | 1.0000 | 0.0105 | 999 | k=1, x0=0.01, constant, w=plus |
| 0.1000 | 0.1000 | 0.1999 | 0.3103 | 0.1999 | 1.0000 | 0.0054 | 999 | k=1, x0=0.0001, constant, w=wave |
| 0.3000 | 0.0000 | 0.2978 | 0.6997 | 0.2978 | 1.0000 | 0.0095 | 999 | k=1, x0=0.00316, constant, w=plus |
| 0.3000 | 0.1000 | 0.3978 | 0.7997 | 0.3978 | 1.0000 | 0.0077 | 999 | k=1, x0=0.00316, constant, w=plus |
| 0.6000 | 0.0000 | 0.5826 | 1.6442 | 0.5826 | 1.0000 | 0.0071 | 999 | k=1, x0=0.00316, constant, w=plus |
| 0.6000 | 0.1000 | 0.6826 | 1.7442 | 0.6826 | 1.0000 | 0.0068 | 999 | k=1, x0=0.00316, constant, w=plus |

## Part B: the length picture under the perturbation model

### bounded map, `Z = [-8, 8]`

`zeta = 0.00868589`, `sup_Z |(log dt)'| = 1.15129`, `sup_Z |dt'| = 0.0297033`, so `rho = 0.01` and `eta_dt = 0.000257999`. Floor `dt(inf Z) = 0.00100155`. Theorem 3: `0.0201003` at every length; common-start constant `0.00999992`; floored additive baseline as `L -> inf`: `0.514945`. The additive baseline reaches the theorem's bound at `L = 38.9542` without the floor and at `L = 39.7548` with it.

| T | G | theorem | additive, no floor | additive, floor | G / theorem | median dt | best start |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.0007201 | 0.0201 | 0.0015 | 0.0015 | 0.0358 | 0.0419 | random |
| 10 | 0.0021 | 0.0201 | 0.0052 | 0.0051 | 0.1049 | 0.0392 | random |
| 30 | 0.0047 | 0.0201 | 0.0155 | 0.0152 | 0.2317 | 0.0326 | k=1, z0=1, constant |
| 100 | 0.0081 | 0.0201 | 0.0516 | 0.0491 | 0.4012 | 0.0210 | k=1, z0=0, constant |
| 300 | 0.0098 | 0.0201 | 0.1548 | 0.1336 | 0.4868 | 0.0129 | k=1, z0=0, balanced |
| 1000 | 0.0100 | 0.0201 | 0.5160 | 0.3258 | 0.4975 | 0.0101 | k=1, z0=1, balanced |
| 3000 | 0.0100 | 0.0201 | 1.5480 | 0.4894 | 0.4975 | 0.0100 | k=1, z0=0, constant |

### softplus map, `Z = [-48, 2]`

`zeta = 0.01`, `sup_Z |(log dt)'| = 1`, `sup_Z |dt'| = 0.880797`, so `rho = 0.01` and `eta_dt = 0.00880797`. Floor `dt(inf Z) = 1.42516e-21`. Theorem 3: `0.0201003` at every length; common-start constant `0.00999992`; floored additive baseline as `L -> inf`: `1.23606e+19`. The additive baseline reaches the theorem's bound at `L = 1.1410` without the floor and at `L = 1.1410` with it.

| T | G | theorem | additive, no floor | additive, floor | G / theorem | median dt | best start |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.0062 | 0.0201 | 0.0528 | 0.0528 | 0.3099 | 0.8852 | k=1, z0=-9, balanced |
| 10 | 0.0084 | 0.0201 | 0.1762 | 0.1762 | 0.4201 | 0.2927 | k=1, z0=-2, balanced |
| 30 | 0.0094 | 0.0201 | 0.5285 | 0.5285 | 0.4677 | 0.1155 | k=1, z0=-2, balanced |
| 100 | 0.0098 | 0.0201 | 1.7616 | 1.7616 | 0.4879 | 0.0378 | k=1, z0=-3, balanced |
| 300 | 0.0099 | 0.0201 | 5.2848 | 5.2848 | 0.4942 | 0.0131 | k=1, z0=-4, balanced |
| 1000 | 0.0100 | 0.0201 | 17.6159 | 17.6159 | 0.4965 | 0.0042 | k=1, z0=-3, constant |
| 3000 | 0.0100 | 0.0201 | 52.8478 | 52.8478 | 0.4969 | 0.0052 | k=4, z0=-6, constant |

Figure: `docs\figures\cell_tightness.pdf`.
