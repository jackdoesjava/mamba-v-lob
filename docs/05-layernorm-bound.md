# The LayerNorm bound

Two facts about `LayerNorm` carry most of this certificate. Its output lies in a box fixed
by the gain and the bias alone, and composing it with the Linear layer that follows admits a
bound that is exact rather than merely sound. Together they take the input distribution out
of every argument downstream.

Notation: `d` is the normalised shape, `eps > 0` the epsilon, `gamma` the gain, `beta` the
bias, `1` the all-ones vector, `W_k` the `k`th row of a weight matrix. Here `d = d_model = 64`,
so `sqrt(d - 1) = 7.94`.

## The box

Proposition 2. For a LayerNorm with normalised shape `d` and `eps > 0`, every `x` in `R^d`
and every coordinate `i` satisfy

```
|LN(x)_i - beta_i| <= |gamma_i| * sqrt(d - 1).
```

Proof. Write `z = (x - mu)/sqrt(sigma^2 + eps)`, so `LN(x) = gamma * z + beta`. The
normalisation forces two constraints on `z`:

```
sum_j z_j = 0,        sum_j z_j^2 = d * sigma^2/(sigma^2 + eps) <= d.
```

Fix a coordinate `i` and put `c = z_i`. Zero sum gives `sum_{j != i} z_j = -c`.
Cauchy-Schwarz on the remaining `d - 1` coordinates gives

```
sum_{j != i} z_j^2 >= (sum_{j != i} z_j)^2 / (d - 1) = c^2/(d - 1).
```

Add `c^2` to both sides and compare with the second constraint:
`d >= c^2 + c^2/(d - 1) = c^2 * d/(d - 1)`, hence `c^2 <= d - 1`. Scaling by `gamma_i` and
shifting by `beta_i` gives the claim. QED

Nothing in the proof mentions `x`, the scale of the incoming activations, or the value of
`eps` beyond its positivity. This is `layernorm()` in `src/verification/intervals.py`. The
test suite samples it at `d` in `{8, 64, 256}`, at input scales spanning eight orders of
magnitude, and against one-hot directions chosen to be extreme.

## Fusing with the next Linear

The box above is stated coordinatewise, which throws away the two constraints that produced
it. Keeping them costs nothing and gives a bound that cannot be improved.

Proposition 3. Let `W` be in `R^{m x d}`, let `v_k = W_k * gamma` elementwise, and let

```
S = {z : 1^T z = 0, ||z||_2 <= sqrt(d)}.
```

Then

```
sup_{z in S} (W(gamma * z))_k = ||v_k - mean(v_k) 1||_2 * sqrt(d),
```

and the supremum is attained at `z* = sqrt(d) (v_k - mean(v_k) 1) / ||v_k - mean(v_k) 1||_2`.

Proof. `(W(gamma * z))_k = <v_k, z>`. Since `1^T z = 0`, the constant vector `mean(v_k) 1`
is orthogonal to `z`, so the mean subtracts for free:
`<v_k, z> = <v_k - mean(v_k) 1, z>`. Cauchy-Schwarz over `||z||_2 <= sqrt(d)` bounds this by
`||v_k - mean(v_k) 1||_2 * sqrt(d)`. The candidate `z*` is a multiple of a vector whose
coordinates sum to zero, so `1^T z* = 0`, and `||z*||_2 = sqrt(d)` exactly, so `z*` is in
`S`; substituting it returns the bound. QED

Bounding the two layers separately gives `||v_k||_1 * sqrt(d - 1)` instead: an `l1` quantity
in place of an `l2` one, with the zero-sum constraint discarded as well. The ratio between
the two grows like `sqrt(d)` in the worst case. This is `layernorm_linear()` in
`src/verification/intervals.py`, and the test that matters constructs `z*` and checks the
realised output matches the bound to `1e-3` relative, which is the empirical form of the
exactness claim.

## What the two relaxations cost

Every quantity below is a property of the trained weights at `L = 100`, computed without
reference to any dataset. The only difference between the rows is whether `LayerNorm` and
the Linear after it are bounded together or one at a time.

| relaxation | `sup \|u\|` | `sup \|B\|` | `sup \|h\|` | certified output range |
| --- | --- | --- | --- | --- |
| fused, Proposition 3 | 8.43 | 17.7 | 1.17e3 | [-5.12, 4.34] |
| separate boxes | 61.3 | 118.0 | 5.71e4 | [-33.95, 29.09] |

That is 49 times on the state bound and 7 times on the output range, from a change of
algebra rather than a change of model. The looseness compounds because `u` feeds `B` and `B`
feeds a hundred steps of recursion; see [04-stability.md](04-stability.md) for how the state
bound accumulates.

`S` is itself a relaxation. With `eps > 0` the reachable set has `||z||_2 < sqrt(d)`
strictly, so the supremum over `S` is approached but never reached by an actual LayerNorm
output. Exactness in Proposition 3 is exactness over `S`, and the residual gap is whatever
`S` contains that LayerNorm cannot produce.

## The input distribution drops out

Each block begins with a LayerNorm, and `LOBMamba` applies one more before the head. So the
operating range of the SSM does not depend on the block's input at all: not on the data
distribution, not on how far the residual stream has drifted through the preceding layers,
not on whether the input is in distribution or is arbitrary. Everything the block computes
downstream, `u`, `delta`, `B` and `C`, inherits a box determined by the weights.

The input box in [08-data.md](08-data.md) is therefore not a premise of any of
these bounds. It is needed to restate them in market units, and for any later analysis that
propagates forward from the features, but the certificate holds without it.

## The output range

Proposition 7. `LOBMamba`'s scalar output lies in a fixed interval determined by the weights
alone, for every input in `R^{B x L x d_in}`.

Proof. `final_norm` is a LayerNorm, so by Propositions 2 and 3 the head's input lies in a
weight-determined box whatever the residual stream did. The head is `Linear`, `SiLU`,
`Linear`. Interval affine arithmetic is sound for the two Linear layers, and `SiLU` is
bounded on any interval: it decreases on `(-inf, x*]` and increases on `[x*, inf)` with
minimum `-0.278465` at `x* = -1.278465`, so the maximum sits at an endpoint and the minimum
is interior exactly when the interval straddles `x*`. QED

The measured range on the trained network is `[-5.12, 4.34]`, in units of the standardised
target. It holds at input scales from `1e-3` to `1e6`, which the test suite checks directly
rather than assuming.

Claimed at its actual strength, this says the model cannot emit an arbitrarily large signal,
whatever it is fed. It says nothing about whether the signal is correct, and a bounded
output is not a bounded loss.

## See also

- [`certification/ablation.md`](certification/ablation.md) section 3, the generated table.
- `src/verification/intervals.py` for `layernorm` and `layernorm_linear`,
  `src/verification/certify.py` for `certify_output_range`.
