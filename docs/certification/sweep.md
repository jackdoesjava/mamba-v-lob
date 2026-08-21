# Does the advantage survive a sweep

The ratio is the geometric bound over the invariant, at L = 100. Certification needs no data, so the untrained rows say as much as the
trained ones. The geometric bound stands in for the widened fixed point, which it
agrees with.

## Seeds, configuration held fixed

| seed | invariant | geometric | ratio | induction holds |
| --- | --- | --- | --- | --- |
| 0 | 125.526 | 1.195e+04 | 95.209 | yes |
| 1 | 127.132 | 1.21e+04 | 95.209 | yes |
| 2 | 132.210 | 1.259e+04 | 95.209 | yes |
| 3 | 101.228 | 9637.245 | 95.203 | yes |
| 4 | 120.918 | 1.151e+04 | 95.209 | yes |
| 5 | 119.362 | 1.136e+04 | 95.209 | yes |
| 6 | 125.689 | 1.197e+04 | 95.203 | yes |
| 7 | 107.669 | 1.025e+04 | 95.209 | yes |

Across 8 seeds the ratio moves by 0.0 percent of its median, so it is not a
property of a particular draw of the weights.

## State dimension

| d_state | max abs(A) | invariant | geometric | ratio | induction holds |
| --- | --- | --- | --- | --- | --- |
| 4 | 4.000 | 125.526 | 1.195e+04 | 95.183 | yes |
| 8 | 8.000 | 125.526 | 1.195e+04 | 95.209 | yes |
| 16 | 16.000 | 125.526 | 1.195e+04 | 95.209 | yes |
| 32 | 32.000 | 125.526 | 1.194e+04 | 95.105 | yes |
| 64 | 64.000 | 125.526 | 1.195e+04 | 95.209 | yes |

## Model width

| d_model | invariant | geometric | ratio | induction holds |
| --- | --- | --- | --- | --- |
| 16 | 15.672 | 462.437 | 29.507 | yes |
| 32 | 31.290 | 2782.591 | 88.928 | yes |
| 64 | 125.526 | 1.195e+04 | 95.209 | yes |
| 128 | 359.030 | 3.418e+04 | 95.209 | yes |

## Expansion factor

| expand | invariant | geometric | ratio | induction holds |
| --- | --- | --- | --- | --- |
| 1 | 78.748 | 7483.234 | 95.028 | yes |
| 2 | 125.526 | 1.195e+04 | 95.209 | yes |
| 4 | 166.102 | 1.581e+04 | 95.209 | yes |

## Timescale floor

| dt_min | sup Abar | invariant | geometric | ratio | induction holds |
| --- | --- | --- | --- | --- | --- |
| 1e-05 | 1.000 | 125.526 | 1.193e+06 | 9503.333 | yes |
| 0.0001 | 1.000 | 125.526 | 1.194e+05 | 951.466 | yes |
| 0.001 | 0.999 | 125.526 | 1.195e+04 | 95.209 | yes |
| 0.010 | 0.990 | 125.526 | 1200.513 | 9.564 | yes |

The geometric bound is the term that moves. It grows as the floor falls, because it
divides by 1 - sup Abar, while the invariant does not depend on the floor at all.

## Timescale ceiling

| dt_max | invariant | geometric | ratio | induction holds |
| --- | --- | --- | --- | --- |
| 0.010 | 125.526 | 1249.616 | 9.955 | yes |
| 0.100 | 125.526 | 1.195e+04 | 95.209 | yes |
| 0.500 | 125.526 | 4.941e+04 | 393.662 | yes |

## Trained checkpoints

| checkpoint | invariant | geometric | ratio | induction holds |
| --- | --- | --- | --- | --- |
| mamba | 129.674 | 1.235e+04 | 95.236 | yes |
| mamba_softplus | 129.717 | inf | inf | yes |

The induction step holds in all 29 configurations.
