# The cell theorem

The paper's one result bounds how far apart two runs of one cell can be, at every sequence
length, with no floor on the timescale. It rests on two facts that are not new: the exact
zero-order hold makes each update a convex combination, and a convex recursion with a bounded
equilibrium stays bounded. This page goes through both, then through the new part: the ratio
lemma, the cell theorem, the construction that shows how tight it is, and the comparison with
Cao et al.

Throughout, `A` is diagonal and strictly negative, `A = -exp(A_log)`, and `lam = |A|` for one
cell. The timescale `dt > 0` is written `delta` in the code. The timescale, `B_t` and `u_t` are
activations, so the cell is linear and time-varying. The state starts at `h_{-1} = 0`.

## The discretisation identity

**Discretisation identity (the paper's equation (1)).** For diagonal `A < 0` and `dt > 0`, the
exact zero-order hold gives

```
Abar = exp(dt A)
Bbar = A^-1 (exp(dt A) - I) B = (1 - Abar) B / |A|.
```

*Proof.* Read elementwise, with `a < 0` a diagonal entry of `A`. Holding `u` constant over a step
of length `dt` gives the input multiplier `(exp(dt a) - 1) / a`. Since `a = -|a|`,

```
(exp(dt a) - 1) / a = (1 - exp(dt a)) / |a| = (1 - Abar) / |a|.
```

∎

The identity is a rearrangement. It puts the factor `1 - Abar` both in the input gain and in the
complement of the transition. `test_zoh_gives_the_convex_form` in `tests/test_invariant.py`
checks it in float64 for `dt` from `1e-8` to `5.0`, to within `1e-14`.

## The convex form

Substituting the identity into `h_t = Abar_t h_{t-1} + Bbar_t u_t` gives

```
h_t = Abar_t h_{t-1} + (1 - Abar_t) c_t,     c_t = B_t u_t / |A|.
```

Since `A < 0` and `dt > 0`, every entry of `Abar_t` lies in `(0, 1)` for every parameter value
and every input. Each step therefore moves the state part of the way towards `c_t`, which we call
the equilibrium, and `1 - Abar_t` is the weight of that step's input, the input weight.

## The forward invariant

**Forward invariant (the paper's Proposition 1, from `h_{-1} = 0`).** If `|c_t| <= M` for every
admissible input and every `t`, then `|h_t| <= M` for every `t`.

*Proof.* Write `a` for a coordinate of `Abar_t`, so `a` lies in `(0, 1)`. `h_{-1} = 0` gives the
base case. If `|h_{t-1}| <= M`, then

```
|h_t| = |a h_{t-1} + (1 - a) c_t| <= a |h_{t-1}| + (1 - a) |c_t| <= a M + (1 - a) M = M.
```

∎

This is Horbatko's Lemma 4.4 (arXiv:2604.18580), stated there from any start as
`|h_t| <= max(|h_{-1}|, M)`, and the same per-cell radius follows from Theorem 3.3(i) of Cao et
al. (ICML 2026). The paper states it in this general form as Proposition 1. It holds at every
length, because the argument is one step repeated and `M` does not depend on `t`, and it assumes
nothing about `dt` beyond `dt > 0`. [03-certificate.md](03-certificate.md) shows how the model's
own normalisation gives `M`.

## The ratio lemma

The difference of two runs needs one more fact. Write `a = exp(-x)` with `x = lam dt` for the
nominal transition, and `a~ = exp(-x e^s)` for a transition whose timescale is `e^s` times the
nominal one.

**Ratio lemma (the paper's Lemma 2).** For `x > 0` and real `s`,
`|e^-x - e^-(x e^s)| <= |e^s - 1| (1 - e^-x)`.

The proof compares chord slopes of the concave function `y -> 1 - e^-y`. So if
`|log dt - log dt~| <= rho`, then `|a - a~| <= (e^rho - 1)(1 - a)`: a change of the timescale on a
log scale moves the transition by at most a fixed fraction of the input weight. When the
timescale shrinks, the input weight shrinks, and so does the possible change of the transition.
Cao et al.'s Lemma B.3 bounds the same change by `lam |dt - dt~|`, which does not shrink with the
input weight. Figure 1(a) of the paper shows the two side by side.

## The cell theorem

Let `h` and `h~` be a nominal and a perturbed run, with error `e_t = h_t - h~_t`. Collecting terms
gives

```
e_t = a_t e_{t-1} + (1 - a_t)(c_t - c~_t) + (a_t - a~_t)(h~_{t-1} - c~_t).
```

**Cell theorem (the paper's Theorem 3).** Suppose `|log dt_t - log dt~_t| <= rho` and
`|c~_t| <= M` for every `t`, and `|h~_{-1}| <= M`. With `eta = sup |c_t - c~_t|`,
`E = eta + 2 M (e^rho - 1)` and `P_t = a_0 a_1 ... a_t`,

```
|e_t| <= P_t |e_{-1}| + (1 - P_t) E <= max(|e_{-1}|, E).
```

The invariant gives `|h~_{t-1} - c~_t| <= 2M`, and the ratio lemma then turns the error identity
into `|e_t| <= a_t |e_{t-1}| + (1 - a_t) E`, the same convex recursion as the state. The invariant
argument applied to this recursion gives the bound. It holds at every length and needs no floor,
because the disturbance at each step carries the same factor `1 - a_t` as the contraction, so
their ratio stays bounded however small the timescale gets.

From a common start the bound is `|e_t| <= E` at every length, and in that case it also follows
from Cao et al.'s Lemmas 3.1 and B.2 after a change of time variable that aligns the time grids
of the two runs (Appendix B of the paper).

The local bounds replace `2 M` by the realised distance `|h_{t-1} - c_t|` of a nominal run. The
paper reports one of them, the endpoint rule, in Section 5.4, and
[03-certificate.md](03-certificate.md) describes all of them.

## How tight it is

**Lower bound (the paper's Proposition 4).** For every `M`, `eta` and `rho > 0` there are runs
from `h_{-1} = h~_{-1} = 0` that satisfy the hypotheses of the theorem and whose gap comes
arbitrarily close to `eta + 2 M tanh(rho/2)`.

In the construction the perturbed equilibrium alternates between `M` and `-M`. The perturbed
timescale is `e^-rho` times the nominal one on the steps with `c~ = M` and `e^rho` times it on
the others. As `t` grows and then the timescales go to zero, the gap after a step with `c~ = M`
approaches `eta + 2 M tanh(rho/2)`. Since `tanh(rho/2) = (e^rho - 1)/(e^rho + 1)`, the theorem's
constant is within a factor `e^rho + 1` of the best possible from a common start, and this factor
tends to 2 as `rho -> 0`. Figure 1(b) of the paper shows the construction at `rho = 0.5`.

A gradient search over the theorem's hypothesis class in one cell, with both runs started from 0
(`scripts/16_cell_tightness.py`), finds no gap above `eta + 2 M tanh(rho/2)`. For `T >= 300` it
comes within 0.4 per cent of that value from random initial sequences and within 0.01 per cent
from initial sequences shaped like the construction
([certification/cell_tightness.md](certification/cell_tightness.md)).

## Comparison with Cao et al.

Replace the bound on `log dt` by an additive bound `|dt - dt~| <= eta_dt` and a lower bound
`dt_lo >= 0` on both timescales, which is a floor when it is positive. The mean value theorem
then gives `|a - a~| <= lam exp(-lam dt_lo) eta_dt`, and carrying the last term of the error
identity forward gives the additive baseline

```
|h_{L-1} - h~_{L-1}| <= eta + 2 M lam exp(-lam dt_lo) eta_dt * sum_{k=0}^{L-1} exp(-k lam dt_lo).
```

It has the structure of Cao et al.'s Theorem 3.3(ii) and Remark 3.4, and it is slightly smaller
than their constant. Without a floor the sum is `L`, so the bound grows linearly with the length.
With a floor it levels off, and its time-warp term is the one-step cost times a factor of order
`1/(lam dt_lo)` as the floor shrinks. The cell theorem replaces this term by `2 M (e^rho - 1)`,
which depends on neither the length nor a floor.

When only the additive bound and a floor are known, `rho = log(1 + eta_dt / dt_lo)` is
admissible, and our time-warp term then lies between the baseline's time-warp term as
`L -> inf` and that of Cao et al.'s Remark 3.4. The gain in our experiments comes from bounding
`log dt` directly from the bound on the pre-activation `z`, through `d log dt / dz`, which is at
most 1 for softplus and at most `ln(dt_max/dt_min)/4` for the bounded map. For the bounded map,
once the baseline has saturated, the ratio of the two time-warp terms in a cell is, to first
order, at most `25.8` for `dt_max/dt_min = 100` when the cell's pre-activation box contains 0, as
every box of the trained bounded arm does (the paper's Remark 5). On the trained model it lies
between 24.1 and 25.8. `src/verification/incremental.py` computes both bounds.

## The exponential-Euler step

The released Mamba implementations use the input term `dt B u` instead of the zero-order hold.
The identity no longer holds, but the update is still a convex combination. With
`Psi(x) = x / (1 - exp(-x))` we have `dt = (1 - Abar) Psi(lam dt) / lam`, so the equilibrium
becomes `c_t = Psi(lam dt_t) B_t u_t / lam`. `Psi` grows without bound, so the invariant and the
theorem now need a ceiling `dt_hi` on the timescale, though still no floor. With
`G = Psi(lam dt_hi)` the invariant radius is `G sup|B u| / lam`. From a common zero start the
theorem holds with `eta` replaced by `G eta_b`, where `eta_b` bounds the change of
`B_t u_t / lam`, with `M` replaced by `G M` and with the factor 2 replaced by 3 (the paper's
Corollary 7). On the trained bounded arm `G` is at most 2.0902; on the released model it reaches
`4.5e11`.

## Why bounding the two factors separately fails

A verifier that bounds `Abar` and `Bbar` on their own gets `|h_t| <= s |h_{t-1}| + M_drive`, with
`s = sup Abar` and `M_drive = sup |Bbar u|`, and hence `|h| <= M_drive / (1 - s)`. `M_drive`
contains the factor `1 - inf Abar`, while the denominator has `1 - sup Abar`, so the cancellation
between them is lost. As `dt -> 0` both factors go to zero together and the true ratio stays
stable, but the separated bound diverges.

With `[d_lo, d_hi]` the propagated timescale box, the ratio of the separated bound to the
invariant has a closed form:

```
gap(lam) = (1 - exp(-lam d_hi)) / (1 - exp(-lam d_lo))
```

It depends on the weights only through `lam`, it is largest for the slowest pole, and it grows
like `1/d_lo`. It is 95.20 to 95.21 across eight untrained seeds, whose slowest pole is
`lam = 1`, and 95.62 on the trained checkpoint, whose slowest pole is `lam = 0.9121`. On the
trained model the invariant radius is `175.9475` at layer 0 and `137.2183` at layer 1. Generic
interval propagation iterated to convergence lands on the separated bound, `95.61` and `95.19`
times larger. `abstraction_gap` in `src/verification/invariant.py` computes the gap, and
`tests/test_invariant.py` checks it against the measured ratio.

Interval arithmetic loses the same dependence when it checks the induction step. In
`a h + (1 - a) c` the variable `a` occurs twice, and an interval evaluator returns
`a_hi M + (1 - a_lo) M_c`, which stays within `M` only when `(1 - a_lo) M_c <= (1 - a_hi) M`. At
the invariant's own radius, `M = M_c`, the check fails whenever `a_lo < a_hi`.
`induction_residual` in `src/verification/invariant.py` keeps `a` as one variable. The supremum
of `a M + (1 - a) M_c` over `a` in `[0, 1]` is `max(M, M_c)`, so the box is preserved exactly when
`M_c <= M`, and the residual it reports is `max(M, M_c) - M`.

## A sharper constant from a common start (not in the paper)

When both runs start from the same state, a proof in an earlier draft of the paper improves the
time-warp term from `2 M (e^rho - 1)` to `2 M tanh(rho/2)`, the value that the construction above
approaches. The paper proves only the lower bound. The upper bound has a short description in
terms of hazards. Unrolled, `h_t` is a weighted sum of the equilibria `c_s` with weights
`(1 - a_s) prod_{r>s} a_r`, an age distribution whose per-step hazard is `1 - a_s`. The
time-warp error of two runs is `2 M` times the total-variation distance between their age
distributions. The hazards of the two runs differ at each step by at most a factor `e^rho`, and
the total-variation distance is then at most `tanh(rho/2)`. The constant equals the classical
bound for two laws whose density ratio lies in `[e^-rho, e^rho]` (Kairouz, Oh and Viswanath,
Remark A.1(b)). That bound does not apply directly, because the density ratio of the two age
distributions is unbounded. The proof, a potential-function argument whose only inequality is
weighted AM-GM, is not published here. `cell_bound` and `cell_bound_common_start` compute the two
forms, and [certification/incremental.md](certification/incremental.md) reports the block
certificate built from each.

## Prior work

Invariants for neural sequence models are not new. Jacoby, Barrett and Katz (ATVA 2020) verify
recurrent networks by inferring an invariant with a verifier, and Bonassi, Farina and Scattolini
(2021) obtain a convex-combination invariant for gated recurrent units, where the convex weights
are a learned gate. For the selective scan the invariant is Horbatko's Lemma 4.4 and follows from
Cao et al.'s Theorem 3.3(i). Gu and Dao (2023, Theorem 1) give the convex form for `lam = 1` and
`B_t = 1` with a softplus timescale, and the link between gain and time constant is classical
linear systems theory.

A search that was not exhaustive found no earlier statement of the cell theorem with the
timescale change measured in `log dt`. It is the paper's one new result. Its technique is
Horbatko's invariant argument applied to the error, or, from a common start, Cao et al.'s lemmas
after a change of time variable. Unlike that of their Theorem 3.3(ii), its time-warp term does
not grow with the length.
