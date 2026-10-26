# A forward-invariant box for the state

The hidden state of a Mamba block never leaves a box whose radius is fixed by the parameters
and the input bounds. That radius is the same at length 10 and at length 5000, and it follows
from one induction step. All of it rests on an identity in the discretisation. Applied to the
difference of two runs, the same step bounds the sensitivity of the state at every length.

`A` is diagonal and strictly negative, from `A = -exp(A_log)`. `dt > 0` is the per-channel
timescale, written `delta` in the code. `B_t` and `u_t` are activations produced from the
block input at every step, so the system is linear time-varying. The state starts at
`h_{-1} = 0`.

## The discretisation identity

**Proposition 1.** For diagonal `A < 0` and `dt > 0`, the exact zero-order hold gives

```
Abar = exp(dt A)
Bbar = A^-1 (exp(dt A) - I) B = (1 - Abar) B / |A|.
```

*Proof.* Read elementwise, with `a < 0` a diagonal entry of `A`. Holding `u` constant over a
step of length `dt` gives the input multiplier `(exp(dt a) - 1) / a`. Since `a = -|a|`,

```
(exp(dt a) - 1) / a = (1 - exp(dt a)) / |a| = (1 - Abar) / |a|.
```

∎

It is a rearrangement, and its consequence is that `1 - Abar` sits in the input gain and in
the complement of the transition at once. `test_zoh_gives_the_convex_form` in
`tests/test_invariant.py` checks it in float64 at `dt` from `1e-8` to `5.0` to `1e-14`.

The exponential-Euler step `Bbar = dt B`, which the released Mamba implementations use, does
not satisfy the identity, but it is still a convex combination. With `lam = |A|` and
`Psi(x) = x / (1 - exp(-x))`, `dt = (1 - Abar) Psi(lam dt) / lam`, so the equilibrium becomes
`c_t = Psi(lam dt_t) B_t u_t / lam`. An invariant exists with radius `G sup|B u| / lam`,
`G = Psi(lam dt_hi)`, which needs a ceiling `dt_hi` on the timescale and no floor. On the trained
bounded arm `G` is at most 2.0902; on the released model it reaches `4.5e11`.

## The convex form

Substituting Proposition 1 into `h_t = Abar_t h_{t-1} + Bbar_t u_t`:

```
h_t = Abar_t h_{t-1} + (1 - Abar_t) c_t,     c_t = B_t u_t / |A|.
```

`Abar_t = exp(dt_t A)` lies in `(0, 1)` elementwise for every parameter value and every
input, because `A < 0` strictly and `dt > 0`. Each coordinate of `h_t` is a convex
combination of the previous state and `c_t`, the value the state is pulled towards.

## The invariant

**Proposition 2.** Suppose `|c_t| <= M` elementwise for every admissible input and every `t`.
Then `|h_t| <= M` for every `t`.

*Proof.* Write `a` for a coordinate of `Abar_t`, so `a` lies in `(0, 1)`. `h_{-1} = 0` gives
the base case. Assume `|h_{t-1}| <= M`. Then

```
|h_t| = |a h_{t-1} + (1 - a) c_t| <= a |h_{t-1}| + (1 - a) |c_t| <= a M + (1 - a) M = M.
```

∎

This is Horbatko's Lemma 4.4 (arXiv:2604.18580), which states it from an arbitrary start as
`|h_t| <= max(|h_{-1}|, M)`, and the same per-cell radius follows from Theorem 3.3(i) of Cao et
al. (ICML 2026). The bound holds at every sequence length, because the argument is one step
applied `t` times and `M` does not depend on `t`. The only assumption on `dt` is `dt > 0`.

## Why bounding the two factors separately fails

The alternative argument bounds `Abar` and `Bbar` on their own, giving
`|h_t| <= s |h_{t-1}| + M_drive` with `s = sup Abar` and `M_drive = sup |Bbar u|`, hence
`|h| <= M_drive / (1 - s)`. `M_drive` carries the factor `1 - Abar` and is taken at its
maximum, so the numerator holds `1 - inf Abar` while the denominator holds `1 - sup Abar`.
The cancellation is lost. As `dt -> 0` both go to zero at the same rate, so the true ratio is
stable; the separated bound diverges while nothing about the state is diverging.

The same loss of dependency weakens the induction step under interval arithmetic. In
`a h + (1 - a) c` the variable `a` occurs in both terms. With `a` in `[a_lo, a_hi]`, an interval
evaluator returns `a_hi M + (1 - a_lo) M_c`, which stays within `M` only when
`(1 - a_lo) M_c <= (1 - a_hi) M`. At the invariant's own radius, `M = M_c`, the check therefore
fails whenever `a_lo < a_hi`.

`induction_residual` in `src/verification/invariant.py` keeps `a` as one variable. The
supremum over `a` in `[0, 1]` of `a M + (1 - a) M_c` is `max(M, M_c)`, so the box is
preserved exactly when `M_c <= M`, and the reported residual is `max(M, M_c) - M`.

## What it measures

Trained bounded arm, `L = 100`, input-independent throughout.

| layer | invariant | widened fixed point | geometric | unrolled at L=100 | tighter by |
| --- | --- | --- | --- | --- | --- |
| 0 | 175.9475 | 1.682e4 | 1.682e4 | 1514.25 | 95.61x |
| 1 | 137.2183 | 1.306e4 | 1.306e4 | 1273.59 | 95.19x |

The widened column iterates generic interval propagation to convergence, 11,120 iterations at
layer 0 and 10,880 at layer 1, and it lands on the geometric fixed point as it should.
Induction holds in both layers, residual exactly `0`, zero-order hold confirmed exact.

Scaling with sequence length, layer 0:

| L | 10 | 50 | 100 | 500 | 1000 | 5000 |
| --- | --- | --- | --- | --- | --- | --- |
| invariant | 175.95 | 175.95 | 175.95 | 175.95 | 175.95 | 175.95 |
| unrolled | 168.84 | 798.21 | 1514.25 | 6313.87 | 1.023e4 | 1.665e4 |

The unrolled bound is recomputed at each length and crosses the invariant between `L = 10` and
`L = 50`.
Above that the invariant is the tighter of the two, and it stops moving.

With `delta = softplus(z)` and no lower bound imposed on it at all, induction still holds:

| layer | invariant | widened fixed point |
| --- | --- | --- |
| 0 | 287.3869 | 2.876e7 |
| 1 | 208.6790 | 2.086e7 |

The invariant needs no floor on `delta`; only the geometric argument does.

## What sets the radius

`M` is `sup |B u| / |A|` over the certified boxes for `B` and `u`. Whether the invariant
exists depends on `A < 0` and `dt > 0` alone; how tight it is depends on those two boxes, and
two structural choices dominate.

| variant | sup abs(u) | radius |
| --- | --- | --- |
| fused LayerNorm and Linear | 9.2681 | 175.9475 |
| LayerNorm box only | 62.0598 | 8075.22 |
| depthwise convolution | 9.2681 | 175.9475 |
| dense convolution | 58.5532 | 1.212e4 |

Fusing the LayerNorm with the Linear after it, rather than boxing the LayerNorm and
propagating through, tightens the radius by 46x; keeping the convolution depthwise gives 69x.
Neither affects whether the invariant exists. Both bounds are derived in
[03-certificate.md](03-certificate.md).

Against held-out batches the certified radius is 162 times the largest realised `|h|` at layer
0 and 382 times at layer 1; against a gradient attack it is 63 and 49 times
([04-results.md](04-results.md)).

## Where the loss comes from, exactly

The gap between the two bounds has a closed form. Writing `lam = |A|` and the propagated
timescale box as `[d_lo, d_hi]`,

```
invariant = sup|B u| / lam
geometric = [ (1 - exp(-lam d_hi)) sup|B u| / lam ] / (1 - exp(-lam d_lo))
```

because `sup|Bbar u|` carries the factor `1 - inf Abar` while the denominator carries
`1 - sup Abar`. Everything except `lam` cancels:

```
gap(lam) = (1 - exp(-lam d_hi)) / (1 - exp(-lam d_lo))
```

Three things follow, and all three are visible in the sweep. The gap depends on the weights
only through `lam`, and it grows as `lam` falls, so the slowest pole sets it: it is 95.20 to
95.21 across eight untrained seeds, whose slowest pole is `lam = 1`, while the radius moves by 30
percent, and 95.62 on the trained checkpoint, whose slowest pole is `lam = 0.9121`. It does not
move with `d_state`, which adds only faster poles. And it grows like `1/d_lo`, which is why
lowering the timescale floor by a decade costs the geometric bound a factor of ten and costs the
invariant nothing.

`abstraction_gap` in `src/verification/invariant.py` computes it, and
`tests/test_invariant.py` checks it elementwise against the measured ratio.

## Two runs: the cell theorem

The same step bounds the difference `e_t = h_t - h~_t` between a nominal run and one driven by a
perturbed input. Collecting terms,

```
e_t = a_t e_{t-1} + (1 - a_t)(c_t - c~_t) + (a_t - a~_t)(h~_{t-1} - c~_t).
```

Measuring the timescale perturbation in `log dt` keeps the factor that the separated bound lost:
for `x > 0` and real `s`, `|e^-x - e^-(x e^s)| <= |e^s - 1| (1 - e^-x)`, so
`|log dt - log dt~| <= rho` gives `|a - a~| <= (e^rho - 1)(1 - a)`. The error recursion is then a
convex combination again, and Proposition 2 applied to it gives, at every length and with no floor,

```
|e_t| <= max(|e_{-1}|, eta + 2 M (e^rho - 1)),     eta = sup|c - c~|.
```

This strengthens the time-warp part of Cao et al.'s Theorem 3.3(ii), whose time-warp term grows
linearly in the position, or with a floor `dt_min` carries `1/(1 - exp(-lam dt_min))`. The same
bound follows from their Lemmas 3.1 and B.2 through a piecewise-linear time change, and the
discrete argument is Horbatko's Lemma 4.4 applied to the error. Measuring the timescale
perturbation additively, `|a - a~| <= lam exp(-lam dt_min) |dt - dt~|`, is the baseline in
`src/verification/incremental.py`.

**The sharp constant.** When both runs start from the same state, as they do here, the time-warp
term improves to `2 M tanh(rho/2)`, and that is attained in a chattering limit. The cleanest
statement is in terms of hazards. Unrolled, `h_t` is a weighted sum of the equilibria `c_s` with
weights `(1 - a_s) prod_{r>s} a_r`, an age distribution whose per-step hazard is `1 - a_s`. The
time-warp error of two runs is `2 M` times the total-variation distance between their age
distributions, the hazards of the two runs differ at each step by at most a factor `e^rho`, and
the total-variation distance is then at most `tanh(rho/2)`. The constant equals the classical
bound for two laws whose density ratio lies in `[e^-rho, e^rho]` (Kairouz, Oh and Viswanath,
Remark A.1(b)), but that bound does not apply here, because the density ratio of the two age
distributions is unbounded. The proof is a potential-function argument whose only inequality is
weighted AM-GM. `cell_bound` and `cell_bound_common_start` compute the two forms, and the data-free
block certificate built from them is in [certification/incremental.md](certification/incremental.md).

The local version replaces `2 M` by the realised distance `|h_{t-1} - c_t|` of a nominal run. Its
rules, including the endpoint form that is reported, are in [03-certificate.md](03-certificate.md).

## Against prior work

Invariants for neural sequence models are not new. Jacoby, Barrett and Katz (ATVA 2020)
verify RNNs by inferring an invariant with a verifier, and Bonassi, Farina and Scattolini
(2021) get a convex-combination invariant for GRUs, where the convex weights are a learned gate.
For the selective scan the closed-form invariant is Horbatko's Lemma 4.4 and follows from Cao et
al.'s Theorem 3.3(i). The scalar case of the discretisation identity is Theorem 1 of Gu and Dao
(2023), and the gain and time-constant relation is classical linear systems theory.

What we did not find, in a search that was not exhaustive, is the cell theorem with the timescale
perturbation measured in `log dt`, its sharp hazard-ratio constant, the discharge of the
invariant's premise from the block's own normalisation, the closed form of the gap, or an
evaluation of the invariant and the sensitivity bound on trained and released models. The cell
theorem follows quickly from Cao et al.'s lemmas and Horbatko's; the sharp constant is the new
piece of mathematics. The premise is discharged in [03-certificate.md](03-certificate.md); the
block itself is in [01-problem.md](01-problem.md).
