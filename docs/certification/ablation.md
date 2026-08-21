# Certification ablation

Sequence length L = 100. Every number below is input-independent: it is a
property of the module, established without reference to any dataset.

## 1. Delta parametrisation (the main result)

| delta | inf delta | sup Abar | 1/(1-sup) | sum_{k<100} | |h| geometric | |h| at L=100 | contractive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bounded | 0.0010 | 0.9990 | 1036.6341 | 95.3719 | 1.235e+04 | 1169.0270 | yes |
| softplus | 0.0000 | 1.0000 | inf | 100.0000 | inf | 1.297e+04 | no |

`softplus` is reference Mamba. Because softplus has infimum 0, delta can approach
zero, Abar can approach 1, and no finite invariant set exists for the state. This
is not conservatism in the abstraction: the supremum is genuinely attained in the
limit. Bounding delta below by dt_min repairs it structurally.

## 2. Convolution structure

| conv | groups | params | max L1 gain | |u| bound | |B| bound | |h| bound |
| --- | --- | --- | --- | --- | --- | --- |
| depthwise | 128 | 512 | 1.6581 | 8.4267 | 17.7131 | 1169.0270 |
| dense | 1 | 65536 | 11.9768 | 56.2254 | 182.3051 | 8.099e+04 |

## 3. Relaxation of LayerNorm

| relaxation | |u| bound | |B| bound | |h| bound | output range |
| --- | --- | --- | --- | --- |
| layernorm_linear_fused (exact) | 8.4267 | 17.7131 | 1169.0270 | [-5.1242, 4.3374] |
| layernorm_box (sound) | 61.3134 | 117.9816 | 5.708e+04 | [-33.9526, 29.0941] |

The fused bound is exact for the composition: over {z : sum z = 0, ||z||_2 <= sqrt(d)}
a linear functional attains ||v - mean(v)||_2 * sqrt(d), and that supremum is reached.

## 4. Finite horizon vs geometric fixed point

| L | sum_{k<L} s^k | 1/(1-s) | |h| at L | |h| geometric |
| --- | --- | --- | --- | --- |
| 10 | 9.9567 | 1036.6341 | 127.5603 | 1.235e+04 |
| 50 | 48.8363 | 1036.6341 | 601.7822 | 1.235e+04 |
| 100 | 95.3719 | 1036.6341 | 1169.0270 | 1.235e+04 |
| 500 | 396.8241 | 1036.6341 | 4838.4028 | 1.235e+04 |
| 1000 | 641.7437 | 1036.6341 | 7781.1172 | 1.235e+04 |

## 5. Discretisation

```json
{
  "delta_range": [
    0.001,
    0.1
  ],
  "A_range": [
    -16.843673706054688,
    -0.9651261568069458
  ],
  "max_abs_dtA": 1.6843674182891846,
  "euler_vs_zoh_rel_error": {
    "median": 0.17306886613368988,
    "p99": 0.8804996013641357,
    "max": 1.068134069442749
  },
  "quantity": "ZOH input gain (exp(dt A) - 1)/A against the Euler surrogate dt",
  "note": "A is discretised exactly either way; what differs is the factor multiplying B. Euler is consistent with that exact Abar only when |delta*A| << 1. The error is a property of the delta range, so bounding delta bounds it as well."
}
```
