# State bounds

Sequence length L = 100 where it applies. Every number is input-independent.

## The comparison

| layer | invariant | widened fixed point | geometric | unrolled at L=100 | invariant is tighter by | induction holds |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 175.9475 | 1.682e+04 | 1.682e+04 | 1514.2534 | 95.6086 | yes |
| 1 | 137.2183 | 1.306e+04 | 1.306e+04 | 1273.5948 | 95.1944 | yes |

The widened column iterates generic interval propagation to convergence, so it is not
a truncated baseline. It agrees with the geometric fixed point, as it should. The
unrolled column is lower only because it stops early, and it grows with L.

## Scaling with sequence length

| L | invariant | unrolled |
| --- | --- | --- |
| 10 | 175.9475 | 168.8444 |
| 50 | 175.9475 | 798.2133 |
| 100 | 175.9475 | 1514.2534 |
| 500 | 175.9475 | 6313.8657 |
| 1000 | 175.9475 | 1.023e+04 |
| 5000 | 175.9475 | 1.665e+04 |

One induction step covers every length. The unrolled bound has to be recomputed per
length and climbs towards the fixed point.

## The softplus arm

| layer | invariant | widened fixed point | induction holds |
| --- | --- | --- | --- |
| 0 | 287.3869 | 2.876e+07 | yes |
| 1 | 208.6790 | 2.086e+07 | yes |

No lower bound on delta is imposed here and the invariant still holds. The convex
form does not use one; only the geometric argument does.

## What loosens the radius

| variant | sup abs(u) | invariant radius |
| --- | --- | --- |
| fused LayerNorm and Linear | 9.2681 | 175.9475 |
| LayerNorm box only | 62.0598 | 8075.2227 |
| depthwise convolution | 9.2681 | 175.9475 |
| dense convolution | 58.5532 | 1.212e+04 |

The radius is sup|B u| / |A|, so anything that loosens the bounds on B or u loosens it
by the same factor. Neither choice affects whether the invariant exists.

## Discretisation

The identity Bbar = (1 - Abar) B / |A| is exact for the zero-order hold and only
asymptotic for the exponential-Euler surrogate Bbar = dt B. That surrogate still gives a
convex update, with the equilibrium enlarged by x / (1 - exp(-x)) at x = |A| dt, so an
invariant survives with a radius larger by that factor at the timescale ceiling.

```json
{
  "delta_range": [
    0.001,
    0.1
  ],
  "max_abs_dtA": 1.7135084867477417,
  "median_rel_error": 0.1731559783220291,
  "max_rel_error": 1.090236783027649
}
```
