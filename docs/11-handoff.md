# Handoff to formal verification

Read this before opening the artefact. One fact governs everything else, and it is the one
that trips people up.

Only `A` and `D` are weights. The timescale `delta`, the input matrix `B` and the output
matrix `C` are activations: `x_proj` and `dt_proj` compute them from the layer input at
every timestep. No `state_dict` in this repository contains `delta`, `B` or `C`, and no
amount of reading the checkpoint will recover them. The block is linear time-varying, so if
you treat it as a fixed linear system and reach for the usual LTI machinery you will prove
something about a different object.

What you get instead is the generator chain, meaning every weight that produces `delta`, `B`
and `C` from the layer input, together with certified intervals those three provably lie in.
The intervals come from the parametrisation and from the LayerNorm at the front of each
block, not from the data, so they hold for any input at all. Proofs are in
[03-discretisation.md](03-discretisation.md) and [04-stability.md](04-stability.md); the short form of the contraction
argument is [04-stability.md](04-stability.md).

## The recurrence

Per block, per channel `d` in `0..d_inner-1`, with diagonal state index `n` in
`0..d_state-1`:

```
xn      = LayerNorm(x)
[u0, g] = in_proj(xn)
u       = SiLU(DepthwiseCausalConv1d(u0))     # position t reads u0[t-d_conv+1 .. t]

[dt_pre, B, C] = x_proj(u)                    # split dt_rank, d_state, d_state
z              = dt_proj(dt_pre)
delta          = dt_min * (dt_max/dt_min)**sigmoid(z), clamped to [dt_min, dt_max]

A     = -exp(A_log)                           # strictly negative, shape (d_inner, d_state)
Abar  = exp(delta * A)
Bbar  = (exp(delta * A) - 1) / A * B
      = delta * phi(delta * A) * B,   phi(v) = expm1(v)/v,  phi(0) = 1

h[t] = Abar[t] * h[t-1] + Bbar[t] * u[t]      # h[-1] = 0
y[t] = sum_n C[t, n] * h[t, :, n] + D * u[t]
out  = out_proj(y * SiLU(g)) + x              # the residual is the block input, pre-norm
```

`Bbar` is the exact zero-order hold form, not the forward Euler shortcut `Bbar ~ delta * B`
that reference Mamba uses. Implement the form above. The two disagree by 17% median relative
error over this network's `delta` range and by 411% over the unconstrained range it first
learned, so the shortcut would have you verifying a different recurrence. Use the Taylor
branch `1 + v/2 + v*v/6` for `|v| < 1e-4` so `phi` stays finite at the origin.

The `delta` line above is the `bounded` parametrisation. The `softplus` arm exists for the
ablation and has no contraction factor at all, so check `dt_parametrisation` in the file
before assuming which one you have.

## The artefact

`scripts/04_export_bounds.py` writes `models/bounds/mamba_certificate.json` (2.7 MB) and a
matching `mamba_certificate.npz` (285 KB, 38 float32 arrays, keyed `layer{i}.{name}` plus the
stem and head names). The `.npz` mirrors the `parameters` section of the JSON and exists so
you do not have to parse megabytes of nested lists to get the tensors.

Top-level keys:

| key | contents |
| --- | --- |
| `schema_version` | currently `"1.0"` |
| `provenance` | export environment and seed, training provenance, checkpoint path, training summary |
| `architecture` | what was instantiated: `d_model` 64, `d_state` 16, `d_conv` 4, `expand` 2, `d_inner` 128, `dt_rank` 4, `num_layers` 2, `dt_min` 1e-3, `dt_max` 0.1 |
| `dynamics_spec` | the equations above as strings, grouped `pre_ssm`, `selective_parameters`, `discretisation`, `recurrence`, `post_ssm`, `stability` |
| `normalisation` | feature names, winsorisation limits, means, standard deviations, target scaling |
| `input_box` | per-feature `lo` and `hi` over 43 features, plus a note on how it is enforced |
| `certificate` | `seq_len`, `relaxation`, `all_layers_contractive`, `output_range`, and per-layer boxes |
| `empirical_envelope` | realised ranges for `u`, `delta`, `B`, `C`, `Abar`, `Bbar`, `h`, `y` and the model output, measured on held-out batches |
| `looseness` | 16 rows, one per layer and quantity, giving certified abs max, empirical abs max, and their ratio |
| `checks` | the round trip and the soundness check, both of which must pass before the file is written |
| `parameters` | `stem`, `layers`, `head`; weights as nested lists of Python floats, which round-trip float32 exactly |

Inside `certificate.layers[i]`, `boxes_full` carries `delta`, `B`, `C` and `u` at full
per-element resolution, since those are what you will reason about. `boxes_reduced` collapses
`Abar`, `Bbar`, `h_geometric`, `h_horizon`, `y_geometric`, `y_horizon`, `out_geometric` and
`out_horizon` to their extrema, because the per-(channel, state) tensors are large and the
extrema are what the bounds use.

`dynamics_spec` is there so the file explains itself. You should not need `src/` to know what
the numbers mean.

## Caveats

`out_geometric` and `out_horizon` are pre-residual. They bound `out_proj(y * SiLU(g))` before
the `+ x`, and the residual stream is not bounded at that point, so they are not bounds on
the block output. The only whole-network output bound is `certificate.output_range`, which is
`[-5.124, 4.337]` here and comes through `final_norm` rather than through the blocks.

The input box is float32 and has to be used as float32. It is computed by pushing the
winsorisation limits through the same `apply()` the model uses, in float32 throughout.
Recomputing `(w - mean)/std` in float64 can land the edge one ulp inside the float32 result,
which would make the box unsound against the values the network actually consumes.

The state bound is loose. Against the realised envelope it runs about `5.2e3` at layer 0 and
`7.6e3` at layer 1, order `1e4`, and `y` is looser still at `5.8e4` and `9.7e4`. That is the
cost of interval arithmetic over a 100-step recursion, where each step discards the
correlation between `h[t]` and `u[t]`. The `delta` boxes, by contrast, come out at 1.09 to
1.10, near exact. Expect the certified state and output boxes to say something about what
cannot happen, and nothing useful about typical behaviour.

## Already checked

Three things are done, so you need not redo them.

The NumPy round trip re-runs block 0's recurrence in float64 from the exported `A` and `D`
and the traced `delta`, `B`, `C`, `u`, and compares against PyTorch. Maximum absolute
disagreement is `2.086e-07` on `y` and `8.725e-08` on the final state, against a tolerance of
`1e-4`. The export refuses to write the file if this fails, so a certificate that exists is
one whose equations reproduce the model.

Sampling soundness is covered by `tests/test_verification.py`. Concrete values drawn from
random input boxes are asserted to land inside every computed output box, the LayerNorm box
is checked at `d` in `{8, 64, 256}` across eight orders of magnitude of input scale, the SiLU
box is checked to reach its interior minimum of `-0.278465`, and the fused LayerNorm-Linear
bound is checked to be attained rather than merely sound.

Gradient-based falsification is `scripts/07_attack_bounds.py`. Adam runs on the input to
maximise `|u|`, `delta`, `|B|`, `|C|`, `|h|`, `|y|` and the model output, in both threat
models: input restricted to the enforced feature box, and input unrestricted. No violation
has been found, and the script exits non-zero if one is. The export additionally checks that
every realised value from the held-out batches sits inside its certified box, and refuses to
write if any escapes.

## The reference implementation

`src/verification/reference.py` reads only the exported JSON. It imports numpy and nothing
else, so it is the ground truth to check your encoding against:

```python
from src.verification.reference import ReferenceModel

model = ReferenceModel.from_json("models/bounds/mamba_certificate.json")
model.check_input_box(x)          # raises if x leaves the certified box
y = model.forward(x)              # x is (batch, L, 43), already normalised
y, trace = model.forward(x, trace=True)   # delta, B, C, A_bar, B_bar, h, y per layer
```

`scripts/04_export_bounds.py` runs it against PyTorch before writing and refuses to write on
disagreement, so a shipped artefact has always passed. Worst measured difference across every
intermediate and the output is 8.821e-07.

Two details it encodes that are easy to get wrong. `Conv1d` cross-correlates rather than
convolves, so the depthwise kernel is not reversed: `out[t] = sum_k w[k] u[t + k - d_conv + 1]`.
And `phi(v) = expm1(v)/v` needs a Taylor branch below `|v| = 1e-4`, where it is 0/0.

## What would help, not yet done

An unrolled feedforward export at fixed `L`. Most verification tooling handles feedforward
networks rather than loops, and at `L = 100` with 2 layers, `d_inner` 128 and `d_state` 16
the scan flattens to a finite graph of depth `L`. The per-timestep `delta`, `B` and `C` stay
input-dependent under unrolling, so this buys a shape the tools accept rather than a simpler
system. Both items are open.
