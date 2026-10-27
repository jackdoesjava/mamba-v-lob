# Handoff and limitations

Read this before you open the artefact. One fact governs everything else, and it is the one
that trips people up.

Only `A` and `D` are weights. The timescale `delta`, the input matrix `B` and the output
matrix `C` are activations: `x_proj` and `dt_proj` compute them from the layer input at every
timestep, so no checkpoint here contains them and no amount of reading the `state_dict` will
recover them. Each block is linear time-varying. Treat it as a fixed linear system, reach for
the usual LTI machinery, and you will prove something about a different object.

What you get instead is the generator chain, every weight that turns a layer input into
`delta`, `B` and `C`, plus certified intervals those three provably lie in. The intervals come
from the parametrisation and from the LayerNorm at the front of each block, not from the data,
so they hold for any input at all.

The property to discharge is one step. With `Abar = exp(delta A)` in `(0, 1)` and
`c[t] = B[t] u[t] / |A|`, the update `h[t] = Abar[t] h[t-1] + (1 - Abar[t]) c[t]` is a convex
combination, so `|c[t]| <= M` for every admissible `B` and `u` gives `|h[t-1]| <= M` implies
`|h[t]| <= M`. The state starts at `h[-1] = 0`, inside the box. There is no induction over
sequence length to carry out and no constraint on `delta`. This is Horbatko's Lemma 4.4 and
follows from Cao et al.'s Theorem 3.3(i). Induction holds in both layers with residual exactly
0, and the zero-order hold identity checks exact.

| layer | radius `M` | widened fixed point | unrolled at `L = 100` |
| --- | --- | --- | --- |
| 0 | 175.9475 | 1.682e4 | 1514.25 |
| 1 | 137.2183 | 1.306e4 | 1273.59 |

## The artefact

[../scripts/04_export_bounds.py](../scripts/04_export_bounds.py) writes
[../models/bounds/mamba_certificate.json](../models/bounds/mamba_certificate.json) (2.7 MB)
and a matching `mamba_certificate.npz` (286 KB, 38 float32 arrays keyed `layer{i}.{name}` plus
the stem and head names). The `.npz` mirrors the `parameters` section, so you need not parse
megabytes of nested lists to get the tensors. It is the bounded arm, the checkpoint selected at
training step 750.

| key | contents |
| --- | --- |
| `schema_version` | currently `"1.0"` |
| `provenance` | export environment and seed, training provenance, checkpoint path, training summary |
| `architecture` | `d_model` 64, `d_state` 16, `d_conv` 4, `expand` 2, `d_inner` 128, `dt_rank` 4, `num_layers` 2, `dt_min` 1e-3, `dt_max` 0.1, `dt_parametrisation` |
| `dynamics_spec` | the equations as strings, grouped `pre_ssm`, `selective_parameters`, `discretisation`, `recurrence`, `post_ssm`, `stability` |
| `normalisation` | feature names, winsorisation limits, means, standard deviations, target scaling |
| `input_box` | per-feature `lo` and `hi` over the 43 features, plus how the box is enforced |
| `certificate` | `seq_len`, `relaxation`, `all_layers_contractive`, `output_range`, and the per-layer boxes |
| `empirical_envelope` | realised ranges for `u`, `delta`, `B`, `C`, `Abar`, `Bbar`, `h`, `y` and the output, on held-out batches |
| `looseness` | 16 rows, one per layer and quantity: certified abs max, empirical abs max, ratio; `h` and `y` are priced against the invariant |
| `checks` | the round trip and the soundness check, both of which must pass before the file is written |
| `parameters` | `stem`, `layers`, `head`; weights as nested lists of floats, which round-trip float32 exactly |

Inside `certificate.layers[i]` you get `contraction`, `horizon_gain`, `summary`, then
`boxes_full` holding `delta`, `B`, `C` and `u` per element, and `boxes_reduced` collapsing
the rest to their extrema: `A_bar`, `B_bar`, the shipped `h_invariant`, `y_invariant` and
`out_invariant`, and the baselines `h_widened`, `h_geometric`, `h_horizon`, `y_geometric`,
`y_horizon`, `out_geometric` and `out_horizon`.

One caution. The three `out_*` boxes are pre-residual, bounding `out_proj(y * SiLU(gate))`
before the `+ x`; the only whole-network output bound is `certificate.output_range`,
`[-3.5643, 3.1277]`, through `final_norm`.

## The recurrence

Per block, per channel `d`, with diagonal state index `n`:

```
xn      = LayerNorm(x)
[u0, g] = in_proj(xn)
u       = SiLU(DepthwiseCausalConv1d(u0))     # position t reads u0[t-d_conv+1 .. t]

[dt_pre, B, C] = x_proj(u)                    # split dt_rank, d_state, d_state
z              = dt_proj(dt_pre)
delta          = dt_min * (dt_max/dt_min)**sigmoid(z)   # bounded arm; softplus arm is softplus(z)

A     = -exp(A_log)                           # strictly negative, (d_inner, d_state)
Abar  = exp(delta * A)
Bbar  = (exp(delta * A) - 1) / A * B
      = delta * phi(delta * A) * B,   phi(v) = expm1(v)/v,  phi(0) = 1

h[t] = Abar[t] * h[t-1] + Bbar[t] * u[t]      # h[-1] = 0
y[t] = sum_n C[t, n] * h[t, :, n] + D * u[t]
out  = out_proj(y * SiLU(g)) + x              # the residual is the block input, pre-norm
```

`Bbar` is the exact zero-order hold, not the exponential-Euler step `Bbar = delta * B` that the
released Mamba implementations use. The identity `Bbar = (1 - Abar) B / |A|` holds only for the
zero-order hold, but the Euler step is also a convex combination, with the equilibrium enlarged by
`Psi(lam delta) = lam delta / (1 - exp(-lam delta))`, so it has an invariant too, with a radius that
needs a ceiling on `delta` and no floor. Over `delta` in `[1e-3, 1e-1]`, where
`max |delta A| = 1.714`, the two disagree by 17.3 percent median and 109.0 percent maximum relative
error.

Two details are easy to get wrong. `Conv1d` cross-correlates rather than convolves, so the
depthwise kernel is not reversed: `out[t] = sum_k w[k] u[t + k - d_conv + 1]`. And
`phi(v) = expm1(v)/v` needs a Taylor branch `1 + v/2 + v*v/6` below `|v| = 1e-4`, where it is
0/0. [../src/verification/reference.py](../src/verification/reference.py) is a NumPy
implementation reading only the JSON, so check your encoding against it; `04_export_bounds.py`
runs it against PyTorch and refuses to write on disagreement, worst difference 1.167e-06 over
every intermediate and the output.

## The sensitivity bound and the released model

`src/verification/incremental.py` bounds the difference of two runs of a block: data-free, from
the cell theorem with the time-warp term `2 M (e^rho - 1)`, or with the sharper common-start
term `2 M tanh(rho/2)`, which the paper states only as a lower bound (Proposition 3.3); and
locally, around a realised input, with one of five recurrence rules. The endpoint rule is the
reported one. The additive rule, the running maximum, the recursion and the additive endpoint
rule are kept as `COMPARISON_RULES`; the last is the endpoint step on the interval of transitions
that the additive timescale bound alone allows, the ablation that measures what the log-timescale
bound adds along an input. Every local rule takes the timescale's slope over the local box
(`_delta_tight`), the same slope that the data-free baseline uses. Scripts 12 and 13 write
[`certification/incremental.md`](certification/incremental.md) and
[`certification/incremental_attack.md`](certification/incremental_attack.md).

`src/verification/released.py` and `scripts/14_released.py` do the same for
`state-spaces/mamba-130m-hf` on `data/text/public_domain_sample.txt` and write
[`certification/released.md`](certification/released.md). The model is not in the repository. It
is loaded with `local_files_only=True` from `MAMBA_130M_DIR`, or from
`~/hf_models/mamba-130m-hf` when that is unset, and the tests that need it skip without
it. The run used transformers 5.15.1 and records the SHA-256 of the weights.

`scripts/15_local_figures.py` draws the paper's figures of the local certificates
(`sensitivity_local`, `end_to_end` and `released_local` in `docs/figures/`) from the JSON that
scripts 13 and 14 write. It recomputes nothing and runs in seconds.

## Open tasks

1. Check the Isabelle sensitivity draft. `isabelle/drafts/Mamba_Sensitivity.thy` states the
   timescale rescaling bound, the error recursion and the cell bound, but it was written without
   access to Isabelle and has never been loaded. It is not in `ROOT`.

2. Discharge the induction step with a solver. `exp` is transcendental, so Z3 cannot take it;
   the 16 of 19 lemmas it did discharge stop at the interval and LayerNorm algebra. dReal is
   delta-complete over the reals with transcendental functions, and is the right tool here.

3. Tighten the local end-to-end bound. Around one block the endpoint rule is within a factor of
   13 of the attack, but end to end it is 5.7e5 times the attack on the bounded arm. The excess is
   in the LayerNorms, the stem, the head and the chaining of stage bounds. A zonotope or a
   CROWN-style backward pass is the standard remedy.

4. Bound the released model's timescale more tightly. Interval propagation gives the timescale
   pre-activation a box 92 to 255 times wider than its realised range, which makes the data-free
   certificate vacuous. The float32 rounding of transformers' slow path is also unbounded.

5. Train independent arms. The two Mamba arms share a seed and differ mainly in the selected
   step. `scripts/02_train_models.py` takes `--seed`; one Mamba run takes about 72 minutes on a CPU.

6. Prove soundness of the composition. The invariant covers the recurrence; the certificate
   covers the stem, both blocks, the gate and the head, and those are stitched together by
   interval arithmetic in this repository rather than by a proof.

## Limitations

The certified radius is 175.9475 at layer 0 against a largest realised state of 1.0830 on
held-out batches, 162 times, and 63 times what a gradient attack reaches; at layer 1 the factors
are 382 and 49. The boxes on `B` are about 16 times loose and those on `u` about 3.5 times, and the
interval product of the two discards their correlation.

The invariant bounds reachable state, and the certified output range comes from `final_norm`
and the head alone. Neither is a robustness certificate. The data-free sensitivity bound holds at
every length but is `3.06e5` to `3.23e6` times the attacked block slope. The local bounds are
closer per block and much looser end to end, and on the softplus arm no rule is below the global
certified width at any `eps` from `1e-6` up. Any reading of these numbers as adversarial robustness
is wrong. A bounded signal is also not a bounded loss: the certified output range says the model
cannot emit an arbitrarily large prediction, and says nothing about whether the prediction is
right or what happens when it is traded.

On the released model the data-free certificate is sound and vacuous, the local certificate is
informative at `delta = 1e-6`, borderline at `1e-4` and useless at `1e-2`, the bounds cover the float64 map and not
transformers' float32 rounding, and the text was typed from memory without being checked against
a printed edition.

One instrument on one day, SPY on 2021-01-28, so the predictive results are illustrative and
should not be generalised from. No pair of neural models is separated by a Diebold-Mariano test.
Each model was trained from one seed at one learning rate. No external verifier has consumed the
artefact, which leaves the bounds computed and checked by code in this repository. The novelty
statements rest on a literature search that was not exhaustive.

## What must not be claimed

These have been retracted or overreached and must not come back.

- That the softplus timescale forces `sup Abar = 1` and so an unbounded state, repaired by a lower
  bound on `delta`. The state is bounded either way; the divergence was an artefact of bounding
  `Abar` and `Bbar` separately, and the softplus arm carries invariants of 287.3869 and 208.6790
  with no floor imposed.
- That the forward invariant is new. It is Horbatko's Lemma 4.4 and follows from Cao et al.'s
  Theorem 3.3(i). Invariants for neural sequence models are older still (Jacoby, Barrett and Katz,
  2020), and the gain and time-constant relation is classical linear systems theory.
- That the invariant exists only under the exact zero-order hold. The exponential-Euler step has
  one too, with a ceiling in place of a floor.
- That interval arithmetic cannot check the induction step for any box with width. It fails at the
  invariant's own radius whenever the transition box has width, and can pass for a smaller
  equilibrium bound.
- That the gap between the geometric bound and the invariant is independent of the weights. It
  depends on them through the slowest pole.
- That one Mamba arm beats the other, or that the comparison shows anything about the timescale
  parametrisation. The arms are not independent, and their Diebold-Mariano p-value is 0.621.
- That the data is anything other than SPY.

What can be claimed, as far as a search that was not exhaustive found, is one new result: the cell
theorem with the timescale perturbation measured in `log dt`, a strengthening of the time-warp
part of Cao et al.'s Theorem 3.3(ii) whose technique is theirs and Horbatko's, together with a
construction showing that its constant is within a factor `e^rho + 1` of the best possible. The
endpoint form of the local rule and the certification of the released model apply it. That the
sharper common-start constant `tanh(rho/2)` that the code computes is also an upper bound is not
part of the paper.
