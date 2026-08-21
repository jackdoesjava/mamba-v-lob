# Notation

Every symbol below is the one the code uses, so a name in a proof can be grepped for in
`src/models/mamba.py` or `src/verification/intervals.py` and found. Tensor shapes are given
batch first, with `batch` for the batch axis and `L` for time. Products written `*` are
elementwise and broadcast over the trailing axes; `exp` acts entrywise; `sup` and `inf` run
over all parameter values and all inputs unless a page says otherwise.

## Sizes

| symbol | in code | meaning | value here |
| --- | --- | --- | --- |
| `d_model` | `d_model` | width of the residual stream, and the normalised shape of every LayerNorm | 64 |
| `d_state` | `d_state` | states per channel, written `n` in the proofs | 16 |
| `expand` | `expand` | inner width multiplier | 2 |
| `d_inner` | `d_inner` | `expand * d_model`, the number of SSM channels | 128 |
| `d_conv` | `d_conv` | kernel width of the causal depthwise convolution | 4 |
| `dt_rank` | `dt_rank` | rank of the timescale projection, `ceil(d_model/16)` when left unset | 4 |
| `L` | `seq_length` | lookback window, in ticks | 100 |
| `input_dim` | `input_dim` | features per tick | 43 |
| `num_layers` | `num_layers` | stacked blocks | 2 |

## The recursion

`A_log` and `D` are the only parameters of the recursion. Everything else in this table is
an activation, recomputed from the block's own input at every timestep, which is what makes
the block linear time-varying. See [02-model.md](02-model.md).

| symbol | in code | shape | meaning |
| --- | --- | --- | --- |
| `A_log` | `A_log` | `(d_inner, d_state)` | parameter; initialised S4D-Real to `log(n+1)` |
| `A` | `A` | `(d_inner, d_state)` | `-exp(A_log)`, the continuous-time state matrix, strictly negative and diagonal per channel; on the trained weights it spans `[-16.8437, -0.9651]` |
| `D` | `D` | `(d_inner,)` | parameter; the skip path `D * u`, initialised to ones |
| `delta` | `delta` | `(batch, L, d_inner)` | timescale, one per channel per step |
| `dt_min`, `dt_max` | `dt_min`, `dt_max` | scalars | the closed range `delta` is confined to, `1e-3` and `1e-1` |
| `pre` | `pre` | `(batch, L, d_inner)` | pre-activation feeding `delta_from_pre`; written `z` in the parametrisation formulas |
| `B` | `B` | `(batch, L, d_state)` | continuous-time input matrix, shared across channels |
| `C` | `C` | `(batch, L, d_state)` | output matrix, shared across channels |
| `Abar` | `A_bar` | `(batch, L, d_inner, d_state)` | `exp(delta * A)`, in `(0, 1)` entrywise |
| `Bbar` | `B_bar` | `(batch, L, d_inner, d_state)` | `delta * phi(delta * A) * B`, the exact zero-order hold input matrix |
| `h` | `h` | `(batch, L, d_inner, d_state)` | state; the recursion runs on `(batch, d_inner, d_state)` slices from `h_{-1} = 0` |
| `u` | `u` | `(batch, L, d_inner)` | SSM branch input, after the depthwise convolution and SiLU |
| `gate` | `gate` | `(batch, L, d_inner)` | second half of `in_proj`, entering as `SiLU(gate)` |
| `y` | `y` | `(batch, L, d_inner)` | SSM output before the gate, `sum_n C[t,n] h[t,:,n] + D * u[t]` |
| `phi` | `zoh_phi` | elementwise | `phi(v) = expm1(v)/v`, extended by `phi(0) = 1`, with a Taylor branch below `\|v\| = 1e-4` |
| `g` | `g` | `(d_inner, d_state)` box | ZOH gain `(exp(delta A) - 1)/A = delta * phi(delta * A)`, kept apart from `B` in `discretisation_boxes` |

## Certificate quantities

| symbol | in code | meaning | value here |
| --- | --- | --- | --- |
| `s` | `sup_A_bar` | contraction factor `exp(-dt_min * min\|A\|)`, the supremum of `Abar` over all inputs | 0.9990 |
| `1/(1-s)` | `geometric_gain` | gain of the fixed point `\|h_inf\| <= M/(1-s)` | 1036.6 |
| `sum_{k<L} s^k` | `horizon_gain` | finite-horizon gain, which stays finite at `s = 1` | 95.37 at `L = 100` |
| `M` | `drive.abs_max()` | drive bound `sup \|Bbar * u\|`, from the LayerNorm box and the fused `Linear o LayerNorm` bound | |
| `x*` | `SILU_ARGMIN` | argmin of SiLU, where the minimum can be interior to a box | -1.278465 |
| `SiLU(x*)` | `SILU_MIN` | the SiLU minimum itself | -0.278465 |
| `gamma`, `beta` | `ln.weight`, `ln.bias` | LayerNorm gain and bias, both `(d_model,)` | |
| `eps` | `ln.eps` | LayerNorm epsilon; the proofs use only `eps > 0` | |
| `lo`, `hi` | `Interval.lo`, `Interval.hi` | endpoints of an interval box, always the same shape | |

The LayerNorm proofs also use `z` for the normalised output `(x - mu)/sqrt(sigma^2 + eps)`
and `S` for the set `{z : sum z = 0, \|\|z\|\|_2 <= sqrt(d)}` it lives in. That `z` is
unrelated to the `z` in `delta = softplus(z)`; the two never appear on the same page. See
[05-layernorm-bound.md](05-layernorm-bound.md) and [04-stability.md](04-stability.md).

## Data side

| symbol | in code | meaning | value here |
| --- | --- | --- | --- |
| `w_lo`, `w_hi` | `winsor_lo`, `winsor_hi` | winsorisation limits, per feature, the 0.1% and 99.9% train-split quantiles, applied at inference | `(43,)` each |
| `mu`, `sigma` | `mean`, `std` | train-split standardisation statistics, per feature | `(43,)` each |
| `x_norm` | `input_box()` | the enforced input box `[(w_lo - mu)/sigma, (w_hi - mu)/sigma]` | `[-6.60, 18.87]` globally |
| `H` | `prediction_horizon` | forward horizon of the target, in ticks | 100 |
| `delta_t` | `delta_t` | inter-arrival time between messages, in seconds, a feature | |

`delta_t` is an input column and has nothing to do with the SSM's `delta`, which is produced
by `dt_proj(x_proj(u))` inside each block and never reads the clock. The box, the splits and
the features are in [08-data.md](08-data.md).

## Bars

`A` and `B` name the continuous-time system `h' = A h + B u`. `Abar` and `Bbar` name its
exact zero-order hold over one step of length `delta`, so the bar marks a discretisation and
not a normalisation or an average. `C` and `D` carry no bar because the output map is the
same in both descriptions. The recursion the network runs uses the barred pair; the
equations a verifier is handed in `dynamics_spec` state the unbarred one, and the two agree
only because the discretisation is exact rather than the forward Euler shortcut the
reference takes; that comparison is [03-discretisation.md](03-discretisation.md).

The proof pages write these with overbars
and Greek letters, `Ā`, `B̄`, `Δ`, `γ`, `β`. The numbered pages use the plain ASCII names
above throughout.
