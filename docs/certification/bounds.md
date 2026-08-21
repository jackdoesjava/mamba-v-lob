# State bounds

Sequence length L = 100 where it applies. Every number is input-independent.

## The comparison

| layer | invariant | widened fixed point | geometric | unrolled at L=100 | invariant is tighter by | induction holds |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 129.6739 | 1.235e+04 | 1.235e+04 | 1169.0270 | 95.2343 | yes |
| 1 | 143.5443 | 1.368e+04 | 1.368e+04 | 1273.9186 | 95.3109 | yes |

The widened column iterates generic interval propagation to convergence, so it is not
a truncated baseline. It agrees with the geometric fixed point, as it should. The
unrolled column is lower only because it stops early, and it grows with L.

## Scaling with sequence length

| L | invariant | unrolled |
| --- | --- | --- |
| 10 | 129.6739 | 127.5603 |
| 50 | 129.6739 | 601.7822 |
| 100 | 129.6739 | 1169.0270 |
| 500 | 129.6739 | 4838.4028 |
| 1000 | 129.6739 | 7781.1172 |
| 5000 | 129.6739 | 1.226e+04 |

One induction step covers every length. The unrolled bound has to be recomputed per
length and climbs towards the fixed point.

## The softplus arm

| layer | invariant | widened fixed point | induction holds |
| --- | --- | --- | --- |
| 0 | 129.7168 | 1.298e+07 | yes |
| 1 | 143.3978 | 1.433e+07 | yes |

No lower bound on delta is imposed here and the invariant still holds. The convex
form does not use one; only the geometric argument does.

## What loosens the radius

| variant | sup abs(u) | invariant radius |
| --- | --- | --- |
| fused LayerNorm and Linear | 8.4267 | 129.6739 |
| LayerNorm box only | 61.3134 | 6332.0322 |
| depthwise convolution | 8.4267 | 129.6739 |
| dense convolution | 56.3247 | 9239.7197 |

The radius is sup|B u| / |A|, so anything that loosens the bounds on B or u loosens it
by the same factor. Neither choice affects whether the invariant exists.

## Discretisation

The convex form is exact for the zero-order hold and only asymptotic for the Euler
surrogate, so the invariant needs the exact discretisation to exist at all.

```json
{
  "delta_range": [
    0.001,
    0.1
  ],
  "max_abs_dtA": 1.6843674182891846,
  "median_rel_error": 0.17306886613368988,
  "max_rel_error": 1.068134069442749
}
```
