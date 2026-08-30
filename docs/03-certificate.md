# The certificate around the invariant

The invariant needs one premise: `|B_t u_t| <= M` uniformly, for every admissible input and every
`t`. No property of the data supplies that. The architecture does, because every block opens with a
LayerNorm, and a LayerNorm's output lies in a box fixed by its own parameters.

## The LayerNorm box

**Proposition.** For a LayerNorm with normalised shape `d` and `eps > 0`, every `x` in `R^d` and
every coordinate `i` satisfy `|LN(x)_i - beta_i| <= |gamma_i| sqrt(d - 1)`.

Proof. Write `z = (x - mu)/sqrt(sigma^2 + eps)`, so `LN(x) = gamma * z + beta`. Normalisation forces
`sum_j z_j = 0` and `sum_j z_j^2 = d sigma^2/(sigma^2 + eps) <= d`. Fix `i` and put `c = z_i`. Zero
sum gives `sum_{j != i} z_j = -c`, and Cauchy-Schwarz over the other `d - 1` coordinates gives
`sum_{j != i} z_j^2 >= c^2/(d - 1)`. Add `c^2` to both sides and compare with the second constraint:
`d >= c^2 d/(d - 1)`, so `c^2 <= d - 1`. Scaling by `gamma_i` and shifting by `beta_i` gives the
claim. QED

The proof never mentions `x`, its scale, or the value of `eps` beyond positivity. Here
`d = d_model = 64`, so `sqrt(d - 1) = 7.94`. This is `layernorm()` in
`src/verification/intervals.py`.

## Fusing the norm with the Linear after it

Stating the box coordinatewise discards the two constraints that produced it. Keeping them costs
nothing and gives a bound that cannot be improved.

**Proposition.** Let `v_k = W_k * gamma` elementwise and `S = {z : 1^T z = 0, ||z||_2 <= sqrt(d)}`.
Then `sup_{z in S} <v_k, z> = ||v_k - mean(v_k) 1||_2 sqrt(d)`, attained at
`z* = sqrt(d) (v_k - mean(v_k) 1) / ||v_k - mean(v_k) 1||_2`.

Proof. Since `1^T z = 0`, the constant vector `mean(v_k) 1` is orthogonal to `z`, so the mean
subtracts for free and `<v_k, z> = <v_k - mean(v_k) 1, z>`. Cauchy-Schwarz over `||z||_2 <= sqrt(d)`
gives the bound. The candidate `z*` sums to zero and has norm exactly `sqrt(d)`, so it lies in `S`,
and substituting it returns the bound. QED

Sound and attained, so composing LayerNorm with the Linear after it is tight over `S` rather than
merely valid. With `eps > 0` the reachable set has `||z||_2 < sqrt(d)` strictly, so the residual gap
is whatever `S` holds that a LayerNorm cannot produce. Bounding the two layers separately gives
`||v_k||_1 sqrt(d - 1)`, an `l1` quantity where an `l2` one will do and the zero-sum constraint
discarded as well. The invariant's radius is `sup|B u| / |A|`, so the cost lands on it directly.

| variant | `sup \|u\|` | radius |
| --- | --- | --- |
| fused LayerNorm and Linear | 8.4267 | 129.6739 |
| LayerNorm box only | 61.3134 | 6332.03 |
| depthwise convolution | 8.4267 | 129.6739 |
| dense convolution | 56.3247 | 9239.72 |

Fusing saves a factor of 48.8 on the radius and changes no weights. The convolution rows compare
architectures rather than relaxations: a channel-mixing convolution sums boxes over all `d_inner`
channels where the depthwise one sums `d_conv = 4` taps of a single channel, and costs 71 times.

## The interval domain

`src/verification/intervals.py` holds one function per operation, each obeying one contract: given
boxes containing the inputs, return a box containing every output they can produce.
Under-approximation is a bug, so anything added there needs a sampling test beside it. Monotone
elementwise maps need only their endpoints, which covers `sigmoid`, `softplus`, `exp` and the
discretisation: with `A < 0` and `dt > 0`, `Abar = exp(dt A)` decreases in `dt` while
`g(dt) = (exp(dt A) - 1)/A` increases, so the `delta` endpoints give both boxes, reversed for `Abar`.

Three operations need more than that. SiLU dips to `-0.2784645` at `x* = -1.2784645`, so its minimum
is interior whenever the box straddles `x*` and the endpoints alone are unsound; the constants are
imported from `src/models/mamba.py` so the model and the domain cannot drift apart. The convolution
hulls its input box with zero first, since the block left-pads `d_conv - 1` zeros and `0` need not
lie in the box on `u`; without the hull the bound fails at `t < d_conv - 1` and nowhere else, which
sampling from the middle of a sequence would never find. `affine` splits the weight into positive
and negative parts and pairs each with the endpoint that extremises it, exact because every output
coordinate is a linear functional over a box and its extreme sits at a vertex. Products take the
four corners, exact for independent intervals; `B` and `u` are not independent, and that discard is
where most of the certified looseness comes from.

## The whole-network certificate

`certify_block` chains the domain in the order `forward` runs, needing no data and assuming nothing
about the block's input, so the stem never enters the argument: the fused norm and `in_proj` give
`u_pre` and the gate, the convolution and SiLU give `u`, `x_proj` and `dt_proj` give `B`, `C` and
`delta`, the discretisation gives `Abar` and `g`, the invariant radius `sup|B u| / |A|` gives `h`,
and `C`, `D` and the gate carry that to the block's contribution.

That contribution is pre-residual. `forward` returns `out_proj(...) + x` and nothing inside the
block bounds `x`, so summing block boxes across layers is not a bound on the network output. The
whole-pass bound starts later, at `final_norm`, and needs nothing from the layers, because a
LayerNorm re-bounds the residual stream whatever it grew to. The head is `Linear`, `SiLU`, `Linear`,
all three of which the domain handles soundly, and on the trained model the scalar output lies in
`[-5.1242, 4.3374]`, a width of 9.4616 standardised units, for any input in `R^{L x 43}` at all.

One standardised unit is 0.7042 basis points of the 100-tick forward return, so the model cannot
forecast a move outside `[-3.600, +3.063]` basis points. That licenses a pre-trade position limit
computed from the weights before the model sees a tick. It says nothing about whether the forecast
is any good, and a bounded output is not a bounded loss.

## Checking it, weakest first

Sampling. Every abstract operation has a test that draws concrete values from an input box and
asserts the result lands inside the computed box, including a box laid across SiLU's minimum and
whole sequences for the convolution's padded positions. The LayerNorm box is sampled at `d` in
`{8, 64, 256}` over eight orders of magnitude of input scale, and the fused bound is checked to be
attained at `z*`. Sampling shows a bound is plausible; the sampler is not looking for the worst case.

Round trip. `src/verification/reference.py` re-runs the forward pass in NumPy from the exported
artefact alone, importing nothing from the model. Worst disagreement against PyTorch is `8.771e-07`
over every intermediate and the output, against a tolerance of `1e-4`. This checks what nothing else
here checks, that the artefact describes the model at all; an earlier reference reversed the
convolution kernel, because `Conv1d` cross-correlates, and every bound and every test still passed.

Machine-checked lemmas. Z3 5.1.0 with a 30 second timeout discharges 16 of 19 obligations, with 3
`nlsat` timeouts and zero counterexamples. Three are proved in general, quantified over all endpoints
and weights: the interval product lies between its four corner products, the interval square handles
a box straddling zero, and the positive and negative weight split bounds an affine map for `n <= 4`.
The rest are proved at small widths: the LayerNorm coordinate bound `z_i^2 <= d - 1` for `d = 2..5`,
the fused bound's tightness for `d = 2, 3`, and its maximiser lying in the constraint set for
`d = 2..5`. `exp` and the zero-order hold fall outside decidable real arithmetic and keep their hand
proofs. The run is in [`certification/smt_lemmas.json`](certification/smt_lemmas.json).

Falsification. `scripts/07_attack_bounds.py` runs Adam on the input to push each quantity out of its
box, in the unrestricted regime where the LayerNorm bound is the only thing holding the certificate
up. No violation has been found. Optimising against a bound also prices it, since
`attacked <= true reachable <= certified`.

| quantity | attacked | certified | ratio |
| --- | --- | --- | --- |
| `sup delta` | 0.0940 | 0.1000 | 1.06x |
| `inf delta` | 1.088e-3 | 1.000e-3 | 1.09x |
| `sup \|u\|` | 4.5630 | 8.4267 | 1.85x |
| `sup \|B\|` | 1.8159 | 17.7131 | 9.75x |
| `sup \|C\|` | 1.3877 | 17.5265 | 12.63x |
| output max | 0.3593 | 4.3374 | 12.07x |
| output min | -0.9234 | -5.1242 | 5.55x |
| `sup \|h\|` against the invariant | 0.5130 | 129.67 | 252.8x |
| attained output width | 1.2827 | 9.4616 | 7.4x |

The internal rows are layer 0 of two, and layer 1 tracks them closely. The `delta` rows are
near-exact because the box there is a property of the parametrisation and its clamp rather than of
an accumulation. The `B` and `C` rows are the interval product discarding correlation, and that is
where the certified gap lives: driven by `B` and `u` from a realised trace instead of the certified
boxes, the invariant sits 3 to 5 times above the realised `sup|h|`. Against that trace the
certified radius is about 630 times the realised state, so the boxes on `B` and `u` account for
a factor of 130 and the induction for the remaining 4.8.

## Local sensitivity, the negative result

Asking a narrower question makes the answer worse. Both local domains diverge while the
unconditional bound stays where it is. Median over three held-out windows, widths in standardised
target units:

| eps | attack | norm propagation | interval | unconditional |
| --- | --- | --- | --- | --- |
| 1e-6 | 1.6242e-6 | 2.1798 | 0.5514 | 9.4616 |
| 1e-5 | 1.6302e-5 | inf | 20.402 | 9.4616 |
| 1e-4 | 1.6296e-4 | inf | 2.7576e7 | 9.4616 |
| 1e-1 | 1.6217e-1 | nan | nan | 9.4616 |

The attack column is a search over inputs that exist, so it lower-bounds the true width, and it is
linear in `eps` across five decades. The model's local Lipschitz constant is about 1.63 and the
model is locally well behaved; the domains fail, not the model. The interval column at `eps = 1e-6`
is the one entry below the unconditional bound, and it is small rather than tight, since the
attacked width in that row is `1.62e-6`.

Interval propagation collapses to a box at every step and norm propagation collapses to a magnitude
at every layer, so both throw away the sign structure of the dependence on the input. A relaxation
that carries linear coefficients instead, a zonotope or a CROWN-style backward pass, is the standard
remedy, and it is not implemented here.

## See also

[02-invariant.md](02-invariant.md) for the invariant these bounds feed. The recorded runs are
[`certification/bounds.json`](certification/bounds.json),
[`certification/attack.json`](certification/attack.json) and
[`certification/domains.json`](certification/domains.json); the code is
`src/verification/intervals.py` and `src/verification/certify.py`.
