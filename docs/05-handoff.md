# Reproducing the paper, the artefact and what is left

This page is for someone who wants to rerun the paper's results, use the exported certificate,
or pick up the remaining work. It ends with the limitations and with the claims this project has
retracted.

## Where the paper's numbers come from

| script | what it gives the paper | recorded in |
| --- | --- | --- |
| `scripts/02_train_models.py`, `scripts/03_evaluate_models.py` | Section 5.1: training, the kept checkpoints and the rank IC | `models/results/test_metrics.json` |
| `scripts/16_cell_tightness.py` | Section 5.2 and Figure 2: the one-cell search | [certification/cell_tightness.md](certification/cell_tightness.md) |
| `scripts/12_incremental.py` | Section 5.3, Table 1 and Figure 3: the data-free certificates; also the stage ratios from random perturbations in Section 5.4, the widening of the boxes in Section 5.1, the LayerNorm constants in Section 7 and the crossover horizon and per-cell ratios in Appendix B | [certification/incremental.md](certification/incremental.md) |
| `scripts/13_incremental_attack.py` | Section 5.4: the attacks and the local bound | [certification/incremental_attack.md](certification/incremental_attack.md) |
| `scripts/06_certify.py` | Appendix C: the separated bound against the invariant | [certification/bounds.md](certification/bounds.md) |
| `scripts/17_mechanism_figure.py` | Figure 1: the mechanism and the lower-bound construction | `docs/figures/mechanism.pdf` |

`scripts/13_incremental_attack.py` also reads `certification/domains.json`, which
`scripts/08_domains.py` writes. Some numbers in the paper do not yet come from a script. In
Section 5.2 they are the length `T = 114` at which the additive baseline passes the trivial bound
and the length `T = 3` at `lam = 17.14`. In Section 7 they are the float32 and bfloat16 rounding
estimates. They were computed by one-off checks while the paper was reviewed.

Before quoting a number, run `python -m scripts.11_check_artefacts`. It reports any recorded
result that is older than the code that produces it. By default it compares commit times, so it
cannot see uncommitted edits. Before committing, run
`python -m scripts.11_check_artefacts --mtime`, which compares file times instead.

## The exported certificate

[../scripts/04_export_bounds.py](../scripts/04_export_bounds.py) writes
[../models/bounds/mamba_certificate.json](../models/bounds/mamba_certificate.json) (2.7 MB) and a
matching `mamba_certificate.npz` (286 KB, 38 float32 arrays keyed `layer{i}.{name}`, plus the stem
and head names). The `.npz` mirrors the `parameters` section, so the tensors can be read without
parsing megabytes of nested lists. It is the bounded arm, from the checkpoint selected at training
step 750.

| key | contents |
| --- | --- |
| `schema_version` | currently `"1.0"` |
| `provenance` | export environment and seed, training provenance, checkpoint path, training summary |
| `architecture` | `d_model` 64, `d_state` 16, `d_conv` 4, `expand` 2, `d_inner` 128, `dt_rank` 4, `num_layers` 2, `dt_min` 1e-3, `dt_max` 0.1, `dt_parametrisation` |
| `dynamics_spec` | the equations as strings, grouped `pre_ssm`, `selective_parameters`, `discretisation`, `recurrence`, `post_ssm`, `stability` |
| `normalisation` | feature names, winsorisation limits, means, standard deviations, target scaling |
| `input_box` | per-feature `lo` and `hi` over the 43 features, and how the box is enforced |
| `certificate` | `seq_len`, `relaxation`, `all_layers_contractive`, `output_range`, and the per-layer boxes |
| `empirical_envelope` | realised ranges for `u`, `delta`, `B`, `C`, `Abar`, `Bbar`, `h`, `y` and the output, on held-out batches |
| `looseness` | 16 rows, one per layer and quantity: certified abs max, empirical abs max, ratio; `h` and `y` are compared with the invariant |
| `checks` | the round trip and the soundness check, both of which must pass before the file is written |
| `parameters` | `stem`, `layers`, `head`; weights as nested lists of floats, which round-trip float32 exactly |

`certificate.layers[i]` holds `contraction`, `horizon_gain` and `summary`, then `boxes_full` with
`delta`, `B`, `C` and `u` per element, and `boxes_reduced` with the rest collapsed to their
extremes: `A_bar`, `B_bar`, the invariant boxes `h_invariant`, `y_invariant` and `out_invariant`,
and the baselines `h_widened`, `h_geometric`, `h_horizon`, `y_geometric`, `y_horizon`,
`out_geometric` and `out_horizon`.

The three `out_*` boxes are pre-residual: they bound `out_proj(y * SiLU(gate))` before the
`+ x`. The only bound on the whole network's output is `certificate.output_range`,
`[-3.5643, 3.1277]`, which goes through `final_norm`.

## Re-implementing the block

The block is written out in [01-problem.md](01-problem.md). It uses the exact zero-order hold,
`Bbar = (exp(delta * A) - 1) / A * B = delta * phi(delta * A) * B`, not the exponential-Euler step
`Bbar = delta * B` of the released Mamba implementations. Over `delta` in `[1e-3, 1e-1]`, where
`max |delta A| = 1.714`, the two differ by 17.3 per cent in median and 109.0 per cent at most,
in relative error. The Euler step is also a convex combination, with the equilibrium enlarged by
`Psi(lam delta) = lam delta / (1 - exp(-lam delta))`, so it has an invariant too, with a radius
that needs a ceiling on the timescale but still no floor.

Two details are easy to get wrong. `Conv1d` cross-correlates rather than convolves, so the
depthwise kernel is not reversed: `out[t] = sum_k w[k] u[t + k - d_conv + 1]`. The function
`phi(v) = expm1(v)/v` needs a Taylor branch, `1 + v/2 + v*v/6` below `|v| = 1e-4`, because the
quotient is 0/0 at `v = 0`. [../src/verification/reference.py](../src/verification/reference.py)
is a NumPy version that reads only the JSON, so an encoding can be checked against it.
`04_export_bounds.py` runs it against PyTorch and refuses to write the file if they disagree. The
worst difference over every intermediate and the output is 1.167e-06.

## The sensitivity code

`src/verification/incremental.py` bounds the difference of two runs of a block. The data-free
certificate uses the cell theorem with the time-warp term `2 M (e^rho - 1)`. It can also use the
sharper common-start term `2 M tanh(rho/2)`, which appears in the paper only as the value that
its lower-bound construction approaches (Proposition 4). The local bound, around a realised
input, uses one of five recurrence rules. The endpoint rule is the one the paper reports, in
Section 5.4. The additive rule, the running maximum, the recursion and the additive endpoint rule
are kept as `COMPARISON_RULES`. The last of these is the endpoint step on the interval of
transitions that the additive timescale bound alone allows. It is the ablation that measures
what the log-timescale bound adds along an input, and the paper reports it too. Every local rule
takes the timescale's slope over the local box (`_delta_tight`), the same slope that the
data-free baseline uses. Scripts 12 and 13 write
[certification/incremental.md](certification/incremental.md) and
[certification/incremental_attack.md](certification/incremental_attack.md).

`src/verification/released.py` and `scripts/14_released.py` do the same for
`state-spaces/mamba-130m-hf` on `data/text/public_domain_sample.txt` and write
[certification/released.md](certification/released.md). The model is not in the repository. It is
loaded with `local_files_only=True` from `MAMBA_130M_DIR`, or from `~/hf_models/mamba-130m-hf`
when that is unset, and the tests that need it are skipped without it. The run used transformers
5.15.1 and records the SHA-256 of the weights.

`scripts/15_local_figures.py` draws the figures of the local bounds (`sensitivity_local`,
`end_to_end` and `released_local` in `docs/figures/`) from the JSON that scripts 13 and 14 write.
These figures are not in the paper. The script recomputes nothing and runs in seconds.

## Open tasks

1. Record the numbers above that do not yet come from a script, in Sections 5.2 and 7.

2. Machine-check the paper's own result. The Isabelle theory covers the discretisation identity,
   the invariant and the normalisation bounds, but not Lemma 2, Theorem 3 or Proposition 4.
   `isabelle/drafts/Mamba_Sensitivity.thy` states the timescale rescaling bound, the error
   recursion and the cell bound, but it was written without access to Isabelle and has never been
   loaded. It is not in `ROOT`.

3. Settle the largest gap from a common start. The paper does not settle whether
   `eta + 2 M tanh(rho/2)` is the largest gap, nor what the largest gap is when the equilibria of
   both runs lie in `[-M, M]`. An earlier draft of the paper has an unpublished proof of the upper
   bound for the time-warp term, described in [02-invariant.md](02-invariant.md), and checking it
   is the natural first step.

4. Tighten the local end-to-end bound. Around one block the endpoint rule is within a factor of
   13 of the attack, but end to end it is 5.7e5 times the attack on the bounded arm. The excess
   is in the LayerNorms, the stem, the head and the chaining of stage bounds. A zonotope or a
   CROWN-style backward pass is the standard remedy.

5. Bound the released model's timescale more tightly. Interval propagation gives the timescale
   pre-activation a box 92 to 255 times wider than its realised range, which makes the data-free
   certificate uninformative. The float32 rounding of transformers' slow path is also not
   bounded.

6. Train independent arms. The two Mamba arms share a seed and differ mainly in the selected
   step. `scripts/02_train_models.py` takes `--seed`, and one Mamba run takes about 72 minutes on
   a CPU.

7. Prove the composition sound. The invariant covers the recurrence, and the interval boxes cover
   the stem, both blocks, the gate and the head, but these are joined by interval arithmetic in
   this repository rather than by a proof.

## Limitations

The cell theorem covers one scalar cell with a real transition in `(0, 1)`. It needs a diagonal
`A` and does not cover the complex-valued update of Mamba-3. The block certificates are per block,
with respect to the normalised input, and are not composed through the residual stream and the
head. They are computed in floating point without outward rounding, so they are real-number
bounds only up to rounding. At `delta = 1e-6` a float32 scan is covered only once a rounding term
is added.

The data-free certificate holds at every length but is `3.06e5` to `3.23e6` times the attacked
block slope. The local bounds are much closer for one block and much looser end to end, and on
the softplus arm no rule is below the global certified width at any `eps` from `1e-6` up. None of
these numbers is a statement about adversarial robustness. At layer 0 the certified radius of the
state is 175.9475. That is 162 times the largest realised state on held-out batches, 1.0830, and
63 times what a gradient attack reaches. At layer 1 the factors are 382 and 49. The certified
output range shows that the model cannot emit an arbitrarily large prediction. It says nothing
about whether the prediction is right, and a bounded output is not a bounded loss.

On the released model the data-free certificate is sound but uninformative. The local bound is
informative at `delta = 1e-6`, borderline at `1e-4` and of no use at `1e-2`. The bounds cover the
float64 map, not transformers' float32 rounding, and the text was typed from memory without being
checked against a printed edition.

The data cover one instrument on one day, SPY on 2021-01-28, so the predictive results are
illustrative and should not be generalised. No pair of neural models is separated by a
Diebold-Mariano test, and each model was trained from one seed at one learning rate. No external
verifier has used the artefact, so the bounds are computed and checked only by code in this
repository. The novelty statements rest on a literature search that was not exhaustive.

## What must not be claimed

These claims were retracted or went too far, and should not come back.

- That the softplus timescale forces `sup Abar = 1` and so an unbounded state, repaired by a lower
  bound on `delta`. The state is bounded either way. The divergence came from bounding `Abar` and
  `Bbar` separately, and the softplus arm has invariants of 287.3869 and 208.6790 with no floor
  imposed.
- That the forward invariant is new. It is Horbatko's Lemma 4.4 and follows from Cao et al.'s
  Theorem 3.3(i). Invariants for neural sequence models are older still (Jacoby, Barrett and Katz,
  2020), and the link between gain and time constant is classical linear systems theory.
- That the invariant exists only under the exact zero-order hold. The exponential-Euler step has
  one too, with a ceiling in place of a floor.
- That interval arithmetic cannot check the induction step for any box with width. It fails at
  the invariant's own radius whenever the transition box has width, and can pass for a smaller
  bound on the equilibrium.
- That the gap between the separated bound and the invariant does not depend on the weights. It
  depends on them through the slowest pole.
- That one Mamba arm beats the other, or that the comparison shows anything about the timescale
  parametrisation. The arms are not independent, and their Diebold-Mariano p-value is 0.621.
- That the data are anything other than SPY.

What can be claimed, as far as a search that was not exhaustive found, is one new result: the
cell theorem with the timescale change measured in `log dt`. Its technique is Horbatko's
invariant argument applied to the error, or, from a common start, Cao et al.'s lemmas after a
change of time variable. Unlike that of Cao et al.'s Theorem 3.3(ii), its time-warp term does not
grow with the length, and a construction shows that its constant is within a factor `e^rho + 1`
of the best possible. The local endpoint rule applies it, and the paper reports it in Section
5.4. The certification of the released model also applies it but is not part of the paper. That
the sharper common-start constant `2 M tanh(rho/2)` is also an upper bound is not part of the
paper.
