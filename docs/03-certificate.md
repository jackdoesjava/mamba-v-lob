# From the cell to a block certificate

The cell theorem needs a bound `M` on the equilibrium `c_t = B_t u_t / lam` for every admissible
input. No property of the data gives one. The architecture does: every block opens with a
normalisation, and the output of a normalisation lies in a set fixed by its own parameters. This
page derives that set, carries it through the block by interval arithmetic, and composes the cell
theorem into a bound on a whole block, as the paper does. The last sections cover the output
range of the network, the local bounds and the released model. Of these, the paper reports only
the endpoint-rule local bound and its ablation.

## The LayerNorm box

**Coordinate bound (the paper's Lemma 9).** For a LayerNorm with normalised shape `d` and
`eps > 0`, every `x` in `R^d` and every coordinate `i` satisfy
`|LN(x)_i - beta_i| <= |gamma_i| sqrt(d - 1)`.

*Proof.* Write `z = (x - mu)/sqrt(sigma^2 + eps)`, so `LN(x) = gamma * z + beta`. Normalisation
forces `sum_j z_j = 0` and `sum_j z_j^2 = d sigma^2/(sigma^2 + eps) <= d`. Fix `i` and put
`c = z_i`. The zero sum gives `sum_{j != i} z_j = -c`, and Cauchy-Schwarz over the other `d - 1`
coordinates gives `sum_{j != i} z_j^2 >= c^2/(d - 1)`. Adding `c^2` to both sides and comparing
with the second constraint gives `d >= c^2 d/(d - 1)`, so `c^2 <= d - 1`. Scaling by `gamma_i`
and shifting by `beta_i` gives the claim. ∎

With `eps = 0` this is the Laguerre-Samuelson inequality: none of `d` numbers lies more than
`sqrt(d - 1)` standard deviations from their mean, with the divisor-`d` variance that PyTorch's
LayerNorm uses.
With `eps > 0` the bound is a supremum that no input attains. Here `d = d_model = 64`, so
`sqrt(d - 1) = 7.94`. The code is `layernorm()` in `src/verification/intervals.py`.

## Fusing the norm with the linear map after it

The coordinate box forgets the two constraints that produced it, the zero mean and the bounded
norm. Keeping them gives a bound that cannot be improved.

**Fused bound (the paper's Proposition 10).** Let `v_k = W_k * gamma` elementwise and
`S = {z : 1^T z = 0, ||z||_2 <= sqrt(d)}`. Then
`sup_{z in S} <v_k, z> = ||v_k - mean(v_k) 1||_2 sqrt(d)`, attained at
`z* = sqrt(d) (v_k - mean(v_k) 1) / ||v_k - mean(v_k) 1||_2`.

*Proof.* Since `1^T z = 0`, the constant vector `mean(v_k) 1` is orthogonal to `z`, so
`<v_k, z> = <v_k - mean(v_k) 1, z>`. Cauchy-Schwarz over `||z||_2 <= sqrt(d)` gives the bound.
The candidate `z*` sums to zero and has norm exactly `sqrt(d)`, so it lies in `S`, and
substituting it returns the bound. ∎

The bound is attained over `S`, so composing the LayerNorm with the following linear map loses
nothing over `S`. With `eps > 0` the reachable set has `||z||_2 < sqrt(d)` strictly, so the bound
is not attained, but by the second part of the paper's Proposition 10 it is still the supremum
over all inputs. Bounding the two layers separately gives `||v_k||_1 sqrt(d - 1)` instead, an
`l1` quantity where an `l2` one suffices, and it drops the zero-sum constraint as well. The
difference lands directly on the radius `M = sup|B u| / lam`:

| variant | `sup \|u\|` | radius |
| --- | --- | --- |
| fused LayerNorm and Linear | 9.2681 | 175.9475 |
| LayerNorm box only | 62.0598 | 8075.22 |
| depthwise convolution | 9.2681 | 175.9475 |
| dense convolution | 58.5532 | 1.212e4 |

Fusing shrinks the radius by a factor of 45.9 without changing any weight. The convolution rows
compare two architectures rather than two relaxations. A channel-mixing convolution sums boxes
over all `d_inner` channels, while the depthwise one sums the `d_conv = 4` taps of one channel,
and the channel-mixing version costs a factor of 68.9.

## The interval domain

`src/verification/intervals.py` has one function per operation. Each takes boxes that contain its
inputs and returns a box that contains every output they can produce. A box that is too small is
a bug, so every function has a sampling test beside it. Monotone maps need only their endpoints,
which covers `sigmoid`, `softplus`, `exp` and the discretisation: with `A < 0` and `dt > 0`,
`Abar = exp(dt A)` decreases in `dt` and `g(dt) = (exp(dt A) - 1)/A` increases, so the endpoints
of the `delta` box give both boxes, reversed for `Abar`.

Four operations need more care. SiLU has its minimum `-0.2784645` at `x* = -1.2784645`, so when
the box contains `x*` the lower bound is `-0.2784645`, not the value at an endpoint. The constants
are imported from `src/models/mamba.py`, so the model and the domain cannot drift apart. The
convolution widens its input box to contain zero, because the block pads the start of the
sequence with `d_conv - 1` zeros and `0` need not lie in the box on `u`. Without this the bound
fails at `t < d_conv - 1`, which sampling from the middle of a sequence would never find.
`affine` splits the weight into positive and negative parts and pairs each with the endpoint that
extremises it, which is exact because each output is a linear function over a box. Products take
the four corners, which is exact for independent intervals. `B` and `u` are not independent, and
the product ignores that. Taking the worst case over each box in this way is where most of the
certificate's looseness comes from (the paper's Section 5.4).

## The block certificate

The paper's Corollary 8 follows a perturbation of a block's normalised input through the stages
in order. The input projection, the convolution and the SiLU bound the change of the scan input
`u`. The projections that follow bound the changes of `B`, `C` and the timescale pre-activation
`z`. The slope of `log dt` turns the bound on `z` into `rho`, and a product rule gives the bound
`eta` on the change of the equilibrium. The cell theorem then bounds the change of each cell's
state, and the readout, the gate and the output projection carry it to the block's output. Each
stage uses a sup-norm Lipschitz constant over its certified box, so the certificate holds, in real
arithmetic, for every input of the block. It needs no data, so the paper calls it data-free.

The certificate bounds the block's output before the residual connection. `forward` returns
`out_proj(...) + x`, and each certificate is taken with respect to the block's normalised input.
Chaining block certificates across layers would therefore also need a Lipschitz bound for each
LayerNorm, as the paper's Section 7 explains. The certificates for both trained arms are in
[certification/incremental.md](certification/incremental.md).

## Checking the boxes

Sampling. Every operation of the domain has a test that draws concrete values from an input box
and checks that the result lands in the computed box. The tests include a box across SiLU's
minimum and whole sequences for the convolution's padded positions. The LayerNorm box is sampled
at `d` in `{8, 64, 256}` over eight orders of magnitude of input scale, and the fused bound is
checked to be attained at `z*`. Sampling makes a bound plausible but does not search for the
worst case.

Round trip. `src/verification/reference.py` reruns the forward pass in NumPy from the exported
artefact alone, importing nothing from the model. Its largest disagreement with PyTorch, over
every intermediate and the output, is `1.167e-06` against a tolerance of `1e-4`. This checks that
the artefact describes the model at all. An earlier reference reversed the convolution kernel,
because `Conv1d` cross-correlates, and every bound and every test still passed.

Machine-checked lemmas. Z3 5.1.0 with a 30-second timeout discharges 16 of 19 obligations, with
3 `nlsat` timeouts and no counterexamples. Three are proved in general, for all endpoints and
weights: the interval product lies between its four corner products, the interval square handles
a box straddling zero, and the split into positive and negative weights bounds an affine map for
`n <= 4`. The rest are proved at small widths: the LayerNorm coordinate bound for `d = 2..5`, the
fused bound's tightness for `d = 2, 3`, and its maximiser lying in the constraint set for
`d = 2..5`. The Isabelle theory in `isabelle/` proves the LayerNorm and fused bounds for every
`d`, together with the discretisation identity and the invariant. The Z3 run is in
[certification/smt_lemmas.json](certification/smt_lemmas.json).

Falsification. `scripts/07_attack_bounds.py` runs Adam on the input to push each quantity out of
its box. The input is unrestricted, so the LayerNorm bound is the only thing holding the boxes
up. No violation has been found. An attack also measures how loose a bound is, since
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

The rows are for layer 0 of two, and layer 1 is close to them. The `delta` rows are almost exact,
because that box comes from the parametrisation and its clamp rather than from an accumulation.
The `B` and `C` rows show the interval product ignoring correlation, which is where most of the
gap in the boxes lies.

## Beyond the paper: the output range

`certify_block` chains the same domain without any perturbation and gives a box for every
intermediate quantity. A bound on the whole network starts at `final_norm`, because a LayerNorm
re-bounds the residual stream however large it has grown. The head is `Linear`, `SiLU`,
`Linear`, which the domain handles soundly. On the trained model the prediction lies in
`[-3.5643, 3.1277]`, a width of 6.6920 standardised units, for any input in `R^{L x 43}`. One
standardised unit is 0.7042 basis points of the 100-tick forward return, around a mean of 0.0085
basis points, so the model cannot forecast a move outside `[-2.502, +2.211]` basis points. A
limit like this can be computed from the weights before the model sees any data. It says nothing
about whether the forecast is good, and a bounded output is not a bounded loss.

## Local bounds, mostly beyond the paper

The data-free certificate holds for every input. A local bound holds only around one realised
input, and it can use what that input fixes, such as the realised timescale and the realised
distance `|h_{t-1} - c_t|`. Generic local domains do badly here. The medians over four held-out
windows, as widths of the prediction in standardised units, are:

| eps | attack | norm propagation | interval | unconditional |
| --- | --- | --- | --- | --- |
| 1e-6 | 1.5199e-6 | inf | 1.2481 | 6.6920 |
| 1e-5 | 1.4983e-5 | inf | 4.2265e4 | 6.6920 |
| 1e-4 | 1.4989e-4 | inf | 3.0382e9 | 6.6920 |
| 1e-1 | 1.5276e-1 | nan | nan | 6.6920 |

The attack column searches over real inputs, so it is a lower bound on the true width, and it is
linear in `eps` up to `1e-2`. The model behaves well locally, and the looseness comes from the
domains. The interval column at `eps = 1e-6` is the only entry below the unconditional bound, and
it is 8.2e5 times the attacked width. Interval propagation collapses to a box at every step and
norm propagation collapses to a magnitude at every layer, so both lose the sign structure of the
dependence on the input. Most of the norm-propagation failure is its recurrence step, which
perturbs `Abar` additively in `delta` and so loses the factor `1 - Abar`.

`src/verification/incremental.py` replaces that step with rules built on the cell theorem, with
`rho_t` the bound on the change of `log dt_t` and `h_{t-1} - c_t` taken on the nominal run:

- running maximum, the local form of the theorem:
  `E_t = max(E_{t-1}, dc_t + (e^rho_t - 1)|h_{t-1} - c_t|)`;
- recursion, the one-step inequality of its proof: `E_t = K_t + a*_t (E_{t-1} - K_t)`, with `K_t`
  the second argument above and `a*_t` the perturbed transition at its worst admissible value;
- endpoint rule, the one the paper reports: the one-step error identity bounded by
  `s E_{t-1} + (1 - s) dc_t + |Abar_t - s| |h_{t-1} - c_t|`, with `s` the perturbed transition,
  maximised over the interval `I_t` that holds `s`. `I_t` is the intersection of the
  log-timescale interval and the interval that the additive bound on `|dt - dt~|` gives. The
  bound is convex in `s`, so its maximum is at an end of `I_t`. It is never above the recursion or
  the additive step, and it is finite for every `rho`.

A fourth rule, the additive endpoint rule, is the endpoint rule on the additive interval alone
(`rho = inf`). It lies between the endpoint rule and the additive step, so the ratio of the
endpoint rule to it measures what the log-timescale bound adds along an input. As `delta -> 0`
the two intervals agree to first order, because `rho_t` and the additive bound come from one
bound on the change of `z`, through `(log dt)'` and `dt' = dt (log dt)'`. For `delta <= 1e-4`
the log-timescale interval changes the endpoint rule by at most 0.034 per cent, so along an input
the gain over the additive step comes from the convex step of the proof, not from the log scale.
The paper reports this ablation.

Every rule takes the timescale's slope over the local box (`_delta_tight`, `delta_slope`), the
same slope as the data-free baseline. An earlier version of the local additive step used
`dt_max ln R sigma'(z)` on the bounded map, which is up to `dt_max / dt` times too large and made
the additive rules look worse than they are.

Around one block's normalised input, at `delta = 1e-6`, the endpoint rule is 3.37 to 12.6 times
the attacked block slope and the additive step 3.41 to 18.0 times. End to end, through the stem,
both blocks, the normalisations and the head, the bounds are much looser. On the trained bounded
arm at `eps = 1e-6` the end-to-end width is 0.8503 with the endpoint rule, 0.8514 with the
additive endpoint rule, 0.8564 with the recursion, 1.291 with the additive step and 10.17 with the
running maximum. At `eps = 1e-4` only the two endpoint rules are finite, at 7.5e28 and 8.1e28.
The end-to-end bound is 5.7e5 times the attack on the bounded arm and 1.6e7 times on the softplus
arm. Most of the excess therefore lies outside the block interiors: in the LayerNorms, the stem
and the head, and in chaining stage bounds that are each computed separately. A relaxation that
carries linear coefficients, such as a zonotope or a CROWN-style backward pass, is the standard
remedy, and it is not implemented here. The numbers are in
[certification/incremental.md](certification/incremental.md) and
[certification/incremental_attack.md](certification/incremental_attack.md).

## Beyond the paper: the released model

`src/verification/released.py` certifies `state-spaces/mamba-130m-hf` as transformers runs it:
RMSNorm, the exponential-Euler step and an unclamped softplus timescale, in float64 without the
float32 casts of transformers' slow path. Each layer is certified with respect to its RMSNorm
output, for every input of the layer.

RMSNorm computes `RMSNorm(x)_i = w_i r_i` with `r = x / sqrt(mean(x^2) + eps)`. Then
`||r||_2^2 = d mean(x^2)/(mean(x^2) + eps) < d`, so `r` lies in the open ball of radius
`sqrt(d)` and `|r_i| < sqrt(d)`, the coordinate bound of MXNorm (arXiv:2603.13180, Lemma 1).
There is no mean subtraction and so no zero-sum constraint, and the fused bound on a row `k` of
`in_proj` is `||W_k * w||_2 sqrt(d)`. On `mamba-130m-hf` the fused bound makes the largest
invariant radius 441 to 1.06e4 times smaller than the coordinate box followed by interval
propagation, with a median over layers of 8979.

For the exponential-Euler step the equilibrium moves with `dt`, so the local endpoint bound,
`s E_{t-1} + x~ db_t + |(Abar_t - s) h_{t-1} + (x - x~) b|` with `x = lam dt` and
`b = B u / lam`, is not convex in `s`. Its maximum over `I_t` lies at an end or at one
closed-form stationary point per sign branch, and without the additive interval it can be
infinite. The derivation is in the docstring of `src/verification/released.py`.

The data-free certificate is sound but uninformative. Every realised quantity on 1024 tokens of
`data/text/public_domain_sample.txt` lies in its box at every layer, but interval propagation
gives the timescale pre-activation a box reaching about `[-2800, 2616]`, against a realised range
of `[-27.5, 9.9]`. The ceiling `dt_hi` is then 1015 to 2616 per layer, `lam` reaches `2.8e8`, `G`
reaches `4.5e11`, and the radius at the median cell is `2e6` to `1.4e9` times the realised state.
The data-free slope is `6.1e16` to `2.3e19`, almost all of it through the timescale. The certified
floor underflows to 0 in float64 in 8 to 1511 of the 1536 channels per layer, so every floored
bound is infinite in floating point.

The local bounds are where the endpoint rule matters on this model. It is below the additive
rule in all 72 (layer, length) pairs at `delta = 1e-6`, `1e-4` and `1e-2`, and at `delta = 1e-6`
it is 60 to 496 times the attacked slope. The ablation shows where the gain comes from. The
log-timescale interval lowers the endpoint rule by at most 0.07 per cent at `delta = 1e-6`, 8.4
per cent at `1e-4` and 39.3 per cent at `1e-2`. At small `delta` the gain over the additive rule
therefore comes from the convex step. At `delta = 1e-2` the additive rule is infinite in every
pair, and only the two endpoint rules have no vacuous pair. Even so, the endpoint rule's bound on
the change is 2.4e4 to 6.7e6 times the largest realised output, so it says nothing useful there.
The numbers are in [04-results.md](04-results.md) and
[certification/released.md](certification/released.md).

## See also

[02-invariant.md](02-invariant.md) has the invariant and the cell theorem that these bounds use.
The recorded runs are [certification/bounds.json](certification/bounds.json),
[certification/attack.json](certification/attack.json),
[certification/domains.json](certification/domains.json),
[certification/incremental.json](certification/incremental.json),
[certification/incremental_attack.json](certification/incremental_attack.json) and
[certification/released.json](certification/released.json). The code is in
`src/verification/intervals.py`, `certify.py`, `incremental.py` and `released.py`.
