# Tests

`python -m pytest` from the repository root. Sixty tests. One skips without CUDA, and the
seven data tests skip if `data/processed/GME_features_20210128.parquet` has not been built;
the path in the fixture is relative, which is the other reason to run from the root.

| file | tests | what it attacks |
| --- | --- | --- |
| `tests/test_ssm_math.py` | 24 | the recurrence, independent of any trained weights |
| `tests/test_verification.py` | 19 | soundness of the interval abstraction |
| `tests/test_features.py` | 7 | invariants of the processed feature matrix |
| `tests/test_transformer.py` | 6 | baseline plumbing, and the causality label |
| `tests/test_lstm.py` | 4 | baseline plumbing |

A certificate is worth nothing unless its soundness is checked. A bound that is wrong is
worse than no bound at all, because it reads like a guarantee. So most of the suite tries to
break the abstraction rather than exercising the happy path: sample inside an input box, run
the concrete layer, assert the concrete output landed inside the computed output box.

## The state-space mathematics

`zoh_phi` matches `expm1(u)/u` to `1e-6` away from the origin, is continuous with value 1 at
the origin, and has a finite gradient there. `B_bar` equals `A^-1 (exp(delta*A) - I) B`
exactly, read off a traced forward pass. A separate test asserts the Euler surrogate
`delta*B` differs from it by more than 1 percent relative somewhere in the tensor. That one
is a regression guard: reverting to the shortcut would leave every other test green while
invalidating the certificate. The argument is in
[03-discretisation.md](03-discretisation.md).

The structural guarantees each get a test. `A < 0` for `A_log` drawn uniformly from
`[-20, 20]`, far outside anything training produces. `delta` stays inside `[dt_min, dt_max]`
at pre-activation scales of `1e-3`, `1`, `1e3` and `1e6`, and is monotone in the
pre-activation across `[-50, 50]`, which is what lets the verifier bound it by endpoints.
`A_bar` lies strictly inside `(0, 1)` on inputs scaled by ten.

One test documents a floating-point fact rather than a mathematical one. In exact arithmetic
sigmoid maps onto the open interval, so `delta` never reaches its endpoints; in float32 it
saturates, and at a pre-activation of `1e4` the parametrisation returns `dt_max` on the nose.
The clamp makes the closed interval hold bitwise, and closed is the form the certificate
assumes.

The two parametrisations are compared directly. Softplus at a pre-activation of `-60` gives
`delta < 1e-20`, `sup A_bar = 1`, infinite geometric gain, `contractive` false. The bounded
parametrisation gives `0 < sup A_bar < 1` with a finite gain. The finite-horizon gain
separates them too: at `L = 100` the softplus arm returns exactly `100.0`, one unit per step
with no decay, while the bounded arm sits strictly below its own geometric gain.

Perturbing the input at position 9 of 16 leaves every output before 9 unchanged and changes
the output at 9. That catches an off-by-one in the causal conv truncation, which is
otherwise invisible. Sequences within a batch do not interact. `forward(trace=True)` is
replayed step by step in Python and checked against the traced tensors, so the exported
trace describes the computation that actually ran rather than a reconstruction of it.

The rest is plumbing: output shape, gradients reaching `A_log`, `dt_proj.bias`, the stem and
the head, determinism in eval mode, `groups == d_inner` and `d_inner * d_conv` conv weights,
the SiLU minimum constants against a 200001-point grid, and that `config.yaml` keys are
honoured. That last one exists because an earlier implementation ignored `d_model`,
`d_state`, `expand`, `num_layers` and `dt_min` in silence.

## Soundness of the abstraction

SiLU, sigmoid, softplus and exp are each sampled against 50 random input boxes with 400
points per box. A dedicated test checks that a box straddling SiLU's minimum reaches down to
`-0.27846454` rather than stopping at the smaller endpoint value, the one case where naive
endpoint evaluation is unsound.

Interval products, affine maps and the depthwise conv are sampled the same way. The conv
test builds 64 concrete sequences from the per-channel box and compares the realised
per-channel extrema against the bound.

The LayerNorm box is checked at `d` in `{8, 64, 256}`, at input scales from `1e-4` to `1e4`,
and against one-hot directions scaled by `1e3`, which are the extremal shape for the
normaliser. The fused LayerNorm-Linear bound is checked to be sound, to be strictly narrower
than bounding the two layers separately, and to be attained: the test constructs
`z* = sqrt(d) (v - mean(v)) / ||v - mean(v)||_2` for each output row and asserts the realised
output matches the bound to `1e-3` relative. That is the empirical form of the exactness
claim in [05-layernorm-bound.md](05-layernorm-bound.md).

At whole-model level, the certified output range is checked to hold at input scales from
`1e-3` to `1e6`, which is the testable face of "for every input whatsoever". The block
certificate is checked to contain every realised `u`, `delta`, `B`, `C`, `A_bar`, `h` and
`y`. The empirical envelope measured over sampled batches is checked to sit inside the
certificate on all eight quantities, so the two ways of reporting a range cannot disagree in
the unsound direction. The geometric and finite-horizon state bounds are checked to agree at
400 steps and to be strictly smaller at 3.

## The main result gets two tests

The structural claim and the in-context claim are different, and running them together
overclaims, so they are separate tests.

`test_softplus_has_no_structural_contraction_factor` takes the block on its own. Over
arbitrary inputs, `inf softplus = 0`, so `sup A_bar = 1`, the geometric series diverges and
`contractive` is false. Nothing about the surrounding network enters.

`test_bounding_delta_tightens_the_in_context_state_bound_enormously` puts the block back in
the model, where it sits behind a LayerNorm and `inf delta = softplus(pre_lo) > 0`. A finite
geometric bound does exist there. It is just far weaker. The test certifies one-layer models
both ways at `seq_len = 50`, asserts `all_layers_contractive` for the bounded arm and not
for the softplus one, and asserts the softplus state bound exceeds the bounded one by a
factor of at least 100. It also checks the finite-horizon bound stays finite in both arms,
since that is the fallback when no contraction factor exists. See
[04-stability.md](04-stability.md).

## Data invariants

The feature tests run against the first 10,000 rows of the processed parquet, which is
enough to trip any of these and keeps the suite fast. They assert no nulls anywhere,
`delta_t >= 0`, `spread_level_0 > 0`, signed distances in their own half-spaces
(`ask_dist_00 >= 0`, `bid_dist_00 <= 0`), a book that gets worse with depth on both sides,
`obi_level_0` inside `[-1, 1]`, and the 47-column schema. A zero or crossed top-of-book
spread is a parsing bug rather than a market event, which is why it is an assertion and not
a filter. The columns are described in [08-data.md](08-data.md).

## Baselines

The LSTM tests cover forward shape, a gradient reaching the final layer, batch independence
and determinism in eval mode. The transformer tests add config plumbing and an fp16 entry
cast, since LayerNorm on CPU rejects half precision, plus a device-routing test that skips
without a GPU.

Two of the transformer tests exist to keep the baseline labelled honestly.
`test_causal_mask_is_applied_when_configured` perturbs position 9 of 12 with `causal: true`
and asserts the hidden states before 9 do not move.
`test_bidirectional_variant_is_not_causal` runs the same perturbation with `causal: false`
and asserts they do. The pair pins the two arms apart, so neither can drift into the other
without a failure. Calling a bidirectional encoder a causal transformer, or citing it in an
argument about autoregressive decoding latency, misstates the comparison.

## What the suite does not do

No test asserts that any model predicts anything. Accuracy lives in the evaluation scripts,
described in [13-running.md](13-running.md), not here. And sampling can only refute a bound,
never establish one; every soundness test above is a search for a counterexample that came
back empty. What that buys and what it does not is in [07-soundness.md](07-soundness.md),
and the proofs themselves are in [04-stability.md](04-stability.md) and
[05-layernorm-bound.md](05-layernorm-bound.md).
