# What makes a selective SSM hard to bound

A state space layer with fixed matrices is a solved problem. Discretise `A` once, read off
`Ā = exp(ΔA)`, check that its spectral radius sits below one, and the reachable state
follows from a geometric series. Mamba does not present that problem. Most of the
recurrence is recomputed from the layer input at every timestep, so there is no single `Ā`
whose spectral radius answers the question.

## Which tensors are parameters

Only `A` and `D` live in the state dict. The timescale `Δ`, the input matrix `B` and the
output matrix `C` are activations, produced from the layer input by `x_proj` and `dt_proj`
at each position. In `src/models/mamba.py`, with `d_model = 64`, `expand = 2`,
`d_state = 16` and `dt_rank = ceil(d_model/16)`:

| quantity | shape | where it comes from |
| --- | --- | --- |
| `A` | `(128, 16)` | parameter, `A = −exp(A_log)`, strictly negative |
| `D` | `(128,)` | parameter, skip path |
| `Δ` | `(batch, L, 128)` | activation, `dt_proj` applied to the first `dt_rank` columns of `x_proj(u)` |
| `B` | `(batch, L, 16)` | activation, `x_proj(u)` |
| `C` | `(batch, L, 16)` | activation, `x_proj(u)` |

The recurrence the block runs is

```
h_t = exp(Δ_t ⊙ A) ⊙ h_{t−1} + B̄_t ⊙ u_t
y_t = Σ_n C_{t,n} · h_{t,:,n} + D ⊙ u_t
```

with a different `Δ_t`, `B̄_t` and `C_t` at every `t`. The system is linear time-varying.

## The claim that has to go

"Mamba is an LTI system and therefore boundable" is false, and it is the first thing a
reviewer will check. The state dict of a block contains `A_log`, `D` and the weights of
`norm`, `in_proj`, `conv1d`, `x_proj`, `dt_proj` and `out_proj`. It contains no `Δ`, no `B`
and no `C`, at any position. A verifier handed a checkpoint and told to reason about
`ḣ = Ah + Bu` cannot fill in `B`; there is nothing there to read.

Time-varying is harder than time-invariant for a concrete reason, not a formal one. A
bound on the state has to hold for a product of transition matrices that the input chooses,
one factor per step, and the input also chooses `B̄_t` and `u_t` in the same breath. A single
eigenvalue computation covers none of that. Whatever is proved has to be proved uniformly
over every sequence the layer could be fed.

Because the scan is unrolled in Python rather than dispatched to a fused kernel, all of
these intermediates are addressable: `forward(trace=True)` returns `Δ`, `B`, `C`, `Ā`, `B̄`,
`h`, `u` and `y`. The time-varying structure is checked against the running computation
rather than taken on faith. The block itself is laid out in [02-model.md](02-model.md).

## What survives

The parametrisation constrains the activations uniformly over all inputs, which is enough
to bound the state without assuming time-invariance anywhere. Three facts do the work.

The sign structure is fixed. `A = −exp(A_log)` is strictly negative for every real
`A_log`, and `Δ > 0`, so `Ā = exp(ΔA) ∈ (0, 1)` elementwise for every parameter value and
every input. No training and no data are involved.

The operating range is fixed. Each block opens with a LayerNorm, and a LayerNorm output
satisfies `|LN(x)_i − β_i| ≤ |γ_i| · √(d − 1)` for every `x ∈ ℝ^d`. Everything downstream
of it, `u`, `Δ`, `B` and `C` alike, inherits a box determined by the weights alone. The
SSM's operating range does not depend on the block's input, on the data distribution, or on
how far the residual stream has drifted.

The timescale is fixed within a range, once the parametrisation is changed. Under
`Δ = δ_min · (δ_max/δ_min)^σ(z)` clamped to `[δ_min, δ_max]`, here `[1e-3, 1e-1]`, the
contraction factor `s = exp(−δ_min · min|A|)` is `0.9990`, below one for every input. Under
the reference `Δ = softplus(z)` it is not, because `softplus` has infimum zero and `sup Ā`
is therefore exactly `1`.

Given those, the per-step inequality

```
|h_t| ≤ s · |h_{t−1}| + M,     M = sup |B̄ ⊙ u|
```

holds at every step whatever the input did, since `s` and `M` are suprema over all inputs.
Summing gives a finite-horizon bound `|h_t| ≤ M · Σ_{k<t+1} sᵏ` and, when `s < 1`, a fixed
point `M/(1 − s)`. Neither step uses time-invariance. At `L = 100` the certified state bound
is `1.17e3` with `Δ` bounded and unbounded structurally with softplus, and the scalar output
of the trained network lies in `[−5.12, 4.34]` for any input in `ℝ^{B×L×d_in}`.

What the export ships is shaped by all of this. Not `Δ`, `B` and `C`, which do not exist
until an input arrives, but the generator chain that produces them, together with certified
intervals those activations provably lie in.

## Where each part is proved

| page | result |
| --- | --- |
| [02-model.md](02-model.md) | the block, and the three deviations from reference Mamba |
| [03-discretisation.md](03-discretisation.md) | exact zero-order hold, and what forward Euler on `B` costs |
| [04-stability.md](04-stability.md) | `Ā ∈ (0,1)` unconditionally, then contraction and the invariant set |
| [05-layernorm-bound.md](05-layernorm-bound.md) | the box, and fusing LayerNorm with the next Linear exactly |

The negative result under softplus, the certified output range, the enforced input box, the
soundness and looseness measurements, and the list of what none of this establishes follow
in the later pages, in that order.

The proofs live on the pages named above; [notation.md](notation.md) fixes the symbols they
use.
