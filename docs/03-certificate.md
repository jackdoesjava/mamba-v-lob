# The certificate around the invariant

The invariant needs one premise: `|B_t u_t| <= M` uniformly, for every admissible input and every
`t`. No property of the data supplies that. The architecture does, because every block opens with a
normalisation, and a normalisation's output lies in a box fixed by its own parameters.

## The LayerNorm box

**Proposition.** For a LayerNorm with normalised shape `d` and `eps > 0`, every `x` in `R^d` and
every coordinate `i` satisfy `|LN(x)_i - beta_i| <= |gamma_i| sqrt(d - 1)`.

Proof. Write `z = (x - mu)/sqrt(sigma^2 + eps)`, so `LN(x) = gamma * z + beta`. Normalisation forces
`sum_j z_j = 0` and `sum_j z_j^2 = d sigma^2/(sigma^2 + eps) <= d`. Fix `i` and put `c = z_i`. Zero
sum gives `sum_{j != i} z_j = -c`, and Cauchy-Schwarz over the other `d - 1` coordinates gives
`sum_{j != i} z_j^2 >= c^2/(d - 1)`. Add `c^2` to both sides and compare with the second constraint:
`d >= c^2 d/(d - 1)`, so `c^2 <= d - 1`. Scaling by `gamma_i` and shifting by `beta_i` gives the
claim. QED

This is the Laguerre-Samuelson inequality: no one of `d` numbers lies more than `sqrt(d - 1)`
standard deviations from their mean, with the divisor-`d` variance that PyTorch's LayerNorm uses.
With `eps > 0` the bound is a supremum that no input attains. Here `d = d_model = 64`, so
`sqrt(d - 1) = 7.94`. This is `layernorm()` in `src/verification/intervals.py`.

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

Sound and attained over `S`, so composing LayerNorm with the Linear after it is tight over `S`
rather than merely valid. With `eps > 0` the reachable set has `||z||_2 < sqrt(d)` strictly, so the
residual gap is whatever `S` holds that a LayerNorm cannot produce. Bounding the two layers
separately gives `||v_k||_1 sqrt(d - 1)`, an `l1` quantity where an `l2` one will do and the
zero-sum constraint discarded as well. The invariant's radius is `sup|B u| / |A|`, so the cost lands
on it directly.

| variant | `sup \|u\|` | radius |
| --- | --- | --- |
| fused LayerNorm and Linear | 9.2681 | 175.9475 |
| LayerNorm box only | 62.0598 | 8075.22 |
| depthwise convolution | 9.2681 | 175.9475 |
| dense convolution | 58.5532 | 1.212e4 |

Fusing saves a factor of 45.9 on the radius and changes no weights. The convolution rows compare
architectures rather than relaxations: a channel-mixing convolution sums boxes over all `d_inner`
channels where the depthwise one sums `d_conv = 4` taps of a single channel, and costs 68.9 times.

## RMSNorm

The released model normalises with RMSNorm, `RMSNorm(x)_i = w_i r_i` with
`r = x / sqrt(mean(x^2) + eps)`. Then `||r||_2^2 = d mean(x^2)/(mean(x^2) + eps) < d`, so `r` lies in
the open ball of radius `sqrt(d)` and `|r_i| < sqrt(d)`, the coordinate bound of MXNorm
(arXiv:2603.13180, Lemma 1). Without the mean subtraction there is no zero-sum constraint, and the
fused bound on a row `k` of `in_proj` is `||W_k * w||_2 sqrt(d)`, Cauchy-Schwarz over the ball with
no centring. On `mamba-130m-hf` the fused bound gives a largest invariant radius 441 to 1.06e4 times
smaller than the coordinate box followed by interval propagation (median over layers 8979).

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
`[-3.5643, 3.1277]`, a width of 6.6920 standardised units, for any input in `R^{L x 43}` at all.

One standardised unit is 0.7042 basis points of the 100-tick forward return, about a mean of
0.0085 basis points, so the model cannot forecast a move outside `[-2.502, +2.211]` basis points.
That licenses a pre-trade position limit computed from the weights before the model sees a tick.
It says nothing about whether the forecast is any good, and a bounded output is not a bounded loss.

## Checking it, weakest first

Sampling. Every abstract operation has a test that draws concrete values from an input box and
asserts the result lands inside the computed box, including a box laid across SiLU's minimum and
whole sequences for the convolution's padded positions. The LayerNorm box is sampled at `d` in
`{8, 64, 256}` over eight orders of magnitude of input scale, and the fused bound is checked to be
attained at `z*`. Sampling shows a bound is plausible; the sampler is not looking for the worst case.

Round trip. `src/verification/reference.py` re-runs the forward pass in NumPy from the exported
artefact alone, importing nothing from the model. Worst disagreement against PyTorch is `1.167e-06`
over every intermediate and the output, against a tolerance of `1e-4`. This checks what nothing else
here checks, that the artefact describes the model at all; an earlier reference reversed the
convolution kernel, because `Conv1d` cross-correlates, and every bound and every test still passed.

Machine-checked lemmas. Z3 5.1.0 with a 30 second timeout discharges 16 of 19 obligations, with 3
`nlsat` timeouts and zero counterexamples. Three are proved in general, quantified over all endpoints
and weights: the interval product lies between its four corner products, the interval square handles
a box straddling zero, and the positive and negative weight split bounds an affine map for `n <= 4`.
The rest are proved at small widths: the LayerNorm coordinate bound `z_i^2 <= d - 1` for `d = 2..5`,
the fused bound's tightness for `d = 2, 3`, and its maximiser lying in the constraint set for
`d = 2..5`. The Isabelle theory in `isabelle/` proves the LayerNorm and fused bounds for every `d`,
together with the zero-order-hold identity and the invariant. The run is in
[`certification/smt_lemmas.json`](certification/smt_lemmas.json).

Falsification. `scripts/07_attack_bounds.py` runs Adam on the input to push each quantity out of its
box, in the unrestricted regime where the LayerNorm bound is the only thing holding the certificate
up. No violation has been found. Optimising against a bound also prices it, since
`attacked <= true reachable <= certified`.

| quantity | attacked | certified | ratio |
| --- | --- | --- | --- |
| `sup delta` | 0.0967 | 0.1000 | 1.03x |
| `inf delta` | 1.061e-3 | 1.000e-3 | 1.06x |
| `sup \|u\|` | 7.1924 | 9.2681 | 1.29x |
| `sup \|B\|` | 2.3962 | 21.4481 | 8.95x |
| `sup \|C\|` | 2.5339 | 19.4540 | 7.68x |
| output max | 0.6553 | 3.1277 | 4.77x |
| output min | -0.8281 | -3.5643 | 4.30x |
| `sup \|h\|` against the invariant | 2.7868 | 175.95 | 63.1x |

The internal rows are layer 0 of two, and layer 1 tracks them closely. The `delta` rows are
near-exact because the box there is a property of the parametrisation and its clamp rather than of
an accumulation. The `B` and `C` rows are the interval product discarding correlation, and that is
where the certified gap lives.

## Local sensitivity

Asking a narrower question with a generic domain makes the answer worse. Median over four held-out
windows, widths of the prediction in standardised target units:

| eps | attack | norm propagation | interval | unconditional |
| --- | --- | --- | --- | --- |
| 1e-6 | 1.5199e-6 | inf | 1.2481 | 6.6920 |
| 1e-5 | 1.4983e-5 | inf | 4.2265e4 | 6.6920 |
| 1e-4 | 1.4989e-4 | inf | 3.0382e9 | 6.6920 |
| 1e-1 | 1.5276e-1 | nan | nan | 6.6920 |

The attack column is a search over inputs that exist, so it lower-bounds the true width, and it is
linear in `eps` up to `1e-2`. The model is locally well behaved; the domains fail, not the model.
The interval column at `eps = 1e-6` is the one entry below the unconditional bound, and it is
8.2e5 times the attacked width.

Interval propagation collapses to a box at every step and norm propagation collapses to a magnitude
at every layer, so both throw away the sign structure of the dependence on the input. Most of the
norm-propagation failure is its recurrence step, which perturbs `Abar` additively in `delta` and so
loses the factor `1 - Abar`. `src/verification/incremental.py` replaces that step with one of three
log-timescale rules from the cell theorem, with `rho_t` the relative timescale perturbation and
`h_{t-1} - c_t` taken on the nominal run:

- running maximum, the local corollary: `E_t = max(E_{t-1}, dc_t + (e^rho_t - 1)|h_{t-1} - c_t|)`;
- recursion, the one-step inequality of its proof: `E_t = K_t + a*_t (E_{t-1} - K_t)`, with `K_t`
  the second argument above and `a*_t` the perturbed transition at its worst admissible value;
- endpoint form, the reported rule: the exact one-step error identity bounded by
  `s E_{t-1} + (1 - s) dc_t + |Abar_t - s| |h_{t-1} - c_t|`, with `s` the perturbed transition,
  maximised over the interval `I_t` that holds `s`. `I_t` is the intersection of the log-timescale
  interval with the one the additive bound on `|dt - dt~|` gives. The bound is convex in `s`, so its
  maximum is at an end of `I_t`. It is never above the recursion or the additive step, and it is
  finite for every `rho`.

For the exponential-Euler cell of the released model the corresponding bound,
`s E_{t-1} + x~ db_t + |(Abar_t - s) h_{t-1} + (x - x~) b|` with `x = lam dt` and `b = B u / lam`, is
not convex in `s`, because the Euler equilibrium moves with `dt`. Its exact maximum over `I_t` is at
an end or at one closed-form stationary point per sign branch, and without the additive interval it
can be infinite. The derivation is in the docstring of `src/verification/released.py`.

On the trained bounded arm at `eps = 1e-6` the end-to-end width is 0.8506 with the endpoint rule,
0.8564 with the recursion, 10.17 with the running maximum and infinite with the additive step; at
`eps = 1e-4` only the endpoint rule is finite, at 7.8e28. Around one block's own normalised input the endpoint
rule is 3.37 to 12.6 times the attacked block slope, but the end-to-end bound is 5.7e5 times the
attack on the bounded arm and 1.6e7 times on the softplus arm. The two measurements place most of the
excess outside the block interiors, in the LayerNorms, the stem and the head, and in chaining stage
bounds that are each computed on their own. A relaxation that carries linear coefficients, a zonotope
or a CROWN-style backward pass, is the standard remedy, and it is not implemented here. The numbers
are in [`certification/incremental.md`](certification/incremental.md) and
[`certification/incremental_attack.md`](certification/incremental_attack.md).

## The released model

`src/verification/released.py` certifies `state-spaces/mamba-130m-hf` as transformers runs it:
RMSNorm, the exponential-Euler step and an unclamped softplus timescale, in float64 without the
float32 casts of transformers' slow path. Each layer is certified with respect to its RMSNorm
output for every input of the layer, with the fused RMSNorm bound above, the interval domain for the
rest, and the Euler invariant radius `G sup|B u| / lam`, `G = Psi(lam dt_hi)`.

The data-free certificate is sound and vacuous. Every realised quantity on 1024 tokens of
`data/text/public_domain_sample.txt` lies in its box at every layer, but interval propagation gives
the timescale pre-activation a box reaching about `[-2800, 2616]` against a realised `[-27.5, 9.9]`.
The ceiling `dt_hi` is then 1015 to 2616 per layer, `lam` reaches `2.8e8`, `G` reaches `4.5e11`, and
the radius at the median cell is `2e6` to `1.4e9` times the realised state. The incremental slope is
`6.1e16` to `2.3e19`, almost all of it through the timescale terms. The certified floor underflows
to 0 in float64 in 8 to 1511 of 1536 channels per layer, so every floored bound is infinite in
floating point.

The local certificate around a realised input is where the endpoint rule matters. It is below the
additive rule in all 72 (layer, length) pairs at `delta = 1e-6`, `1e-4` and `1e-2`. At
`delta = 1e-6` it is 60 to 496 times the attacked slope. At `delta = 1e-2` the additive rule is
infinite in every pair and the endpoint rule is the only one with no vacuous pair, but its bound on
the change is still 2.4e4 to 6.7e6 times the largest realised output, so it says nothing useful
there. The numbers are
in [04-results.md](04-results.md) and [`certification/released.md`](certification/released.md).

## See also

[02-invariant.md](02-invariant.md) for the invariant and the cell theorem these bounds feed. The
recorded runs are [`certification/bounds.json`](certification/bounds.json),
[`certification/attack.json`](certification/attack.json),
[`certification/domains.json`](certification/domains.json),
[`certification/incremental.json`](certification/incremental.json),
[`certification/incremental_attack.json`](certification/incremental_attack.json) and
[`certification/released.json`](certification/released.json); the code is
`src/verification/intervals.py`, `certify.py`, `incremental.py` and `released.py`.
