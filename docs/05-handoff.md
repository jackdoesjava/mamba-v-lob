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
sequence length to carry out and no constraint on `delta`. Induction holds in both layers with
residual exactly 0, and the zero-order hold identity checks to 1e-14.

| layer | radius `M` | widened fixed point | unrolled at `L = 100` |
| --- | --- | --- | --- |
| 0 | 129.6739 | 1.235e4 | 1169.03 |
| 1 | 143.5443 | 1.368e4 | 1273.92 |

## The artefact

[../scripts/04_export_bounds.py](../scripts/04_export_bounds.py) writes
[../models/bounds/mamba_certificate.json](../models/bounds/mamba_certificate.json) (2.7 MB)
and a matching `mamba_certificate.npz` (285 KB, 38 float32 arrays keyed `layer{i}.{name}` plus
the stem and head names). The `.npz` mirrors the `parameters` section, so you need not parse
megabytes of nested lists to get the tensors.

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
`[-5.1242, 4.3374]`, through `final_norm`.

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

`Bbar` is the exact zero-order hold, not the forward Euler shortcut `Bbar ~ delta * B` that
reference Mamba uses. The convex form is exact for the zero-order hold and only asymptotic for
Euler, so the invariant needs this discretisation to exist. Over `delta` in `[1e-3, 1e-1]`,
where `max |delta A| = 1.684`, the two disagree by 17.3 percent median and 106.8 percent
maximum relative error.

Two details are easy to get wrong. `Conv1d` cross-correlates rather than convolves, so the
depthwise kernel is not reversed: `out[t] = sum_k w[k] u[t + k - d_conv + 1]`. And
`phi(v) = expm1(v)/v` needs a Taylor branch `1 + v/2 + v*v/6` below `|v| = 1e-4`, where it is
0/0. [../src/verification/reference.py](../src/verification/reference.py) is a NumPy
implementation reading only the JSON, so check your encoding against it; `04_export_bounds.py`
runs it against PyTorch and refuses to write on disagreement, worst difference 8.771e-07 over
every intermediate and the output.

## Open tasks

1. Discharge the induction step with a solver. `exp` is transcendental, so Z3 cannot take it;
   the 16 of 19 lemmas it did discharge stop at the interval and LayerNorm algebra. dReal is
   delta-complete over the reals with transcendental functions, and is the right tool here.

2. Position the result against the existing verifiers, either by running them on this model or
   by characterising them honestly. Jacoby, Barrett and Katz infer invariants for RNNs with a
   verifier, which can return unknown and is not parameter-explicit; Tran et al. do star
   reachability. The claim is analytic derivation, not first use of invariants, and the
   comparison has to be written so that distinction survives review.

3. Push past the box to a tighter shape. A weighted box, an ellipsoid or a polyhedron all
   admit the same one-step argument, and the question is synthesising the tightest member of
   the family rather than checking one you were handed. This part is research, not plumbing.

4. Prove soundness of the composition. The invariant covers the recurrence; the certificate
   covers the stem, both blocks, the gate and the head, and those are stitched together by
   interval arithmetic in this repository rather than by a proof.

## Limitations

The certified radius is 129.6739 at layer 0 against a realised state of 0.2066, so about 600
times conservative. The gap is not in the invariant: feeding it `B` and `u` from the realised
trace instead of the certified boxes gives 1.0020 at layer 0 against 0.2066, and 0.5139 at
layer 1 against 0.1585, so the induction accounts for a factor of 3 to 5 and the boxes on
`B` and `u` for the remaining 130.

This bounds reachable state and reachable output. It is not a robustness certificate. Nothing
here bounds `|f(x) - f(x')|` for nearby inputs, and any reading of these numbers as
adversarial robustness is wrong. A bounded signal is also not a bounded loss: the certified
output range says the model cannot emit an arbitrarily large prediction, and says nothing
about whether the prediction is right or what happens when it is traded.

One instrument on one day, GME on 2021-01-28, so the predictive results are illustrative and
should not be generalised from. No external verifier has consumed the artefact, which leaves
the bounds computed and checked by code in this repository. Every model plateaus at the first
evaluation, step 250, and no learning-rate sweep was run, so three architectures at one
setting each is not a comparison of architectures. The novelty claim rests on a literature
search that was not exhaustive.

## What must not be claimed

Three things have been retracted or overreached and must not come back. That
`inf softplus = 0` implies `sup Abar = 1` implies an unbounded state, repaired by a lower
bound on `delta`: the state is bounded either way, the divergence was an artefact of bounding
`Abar` and `Bbar` separately, and the softplus arm carries invariants of 129.7168 and
143.3978 with no floor imposed. That invariants for neural sequence models are new, or that
length-independent certification is new: Jacoby, Barrett and Katz did invariant inference for
RNNs in 2020. That the gain and time-constant relation is novel, when it is classical linear
systems theory. What can be claimed is the first closed-form, architecture-specific inductive
invariant for an input-selective state space model, derived analytically rather than inferred
by a verifier, whose assumptions are discharged by the model's own normalisation bounds.
