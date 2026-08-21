# A forward-invariant box for the state

The hidden state of a Mamba block never leaves a box whose radius is fixed by the parameters
and the input bounds. That radius is the same at length 10 and at length 5000, and it follows
from one induction step. All of it rests on an identity in the discretisation.

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
`tests/test_invariant.py` checks it in float64 at `dt` from `1e-8` to `5.0`, eight orders of
magnitude, to `1e-14`. The Euler surrogate `Bbar = dt B` satisfies it only as `dt -> 0`, so
the identity belongs to the exact discretisation and not to state space models in general.

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

The bound holds at every sequence length, because the argument is one step applied `t` times
and `M` does not depend on `t`. Nothing is unrolled and no geometric series is summed. The
only assumption on `dt` is `dt > 0`; a lower bound on it buys nothing here.

## Why bounding the two factors separately fails

The alternative argument bounds `Abar` and `Bbar` on their own, giving
`|h_t| <= s |h_{t-1}| + M_drive` with `s = sup Abar` and `M_drive = sup |Bbar u|`, hence
`|h| <= M_drive / (1 - s)`. `M_drive` carries the factor `1 - Abar` and is taken at its
maximum, so the numerator holds `1 - inf Abar` while the denominator holds `1 - sup Abar`.
The cancellation is lost. As `dt -> 0` both go to zero at the same rate, so the true ratio is
stable; the separated bound diverges while nothing about the state is diverging.

The same loss of dependency breaks the induction step, so that check cannot go through
interval arithmetic either. In `a h + (1 - a) c` the variable `a` occurs in both terms. An
interval evaluator bounds the two products independently, letting the first take `sup Abar`
and the second `1 - inf Abar`; the weights sum past one, the step reports growth, and the
check fails for any box that is not a point.

`induction_residual` in `src/verification/invariant.py` keeps `a` as one variable. The
supremum over `a` in `[0, 1]` of `a M + (1 - a) M_c` is `max(M, M_c)`, so the box is
preserved exactly when `M_c <= M`, and the reported residual is `max(M, M_c) - M`.

## What it measures

Trained model, `L = 100`, input-independent throughout.

| layer | invariant | widened fixed point | geometric | unrolled at L=100 | tighter by |
| --- | --- | --- | --- | --- | --- |
| 0 | 129.6739 | 1.235e4 | 1.235e4 | 1169.03 | 95.23x |
| 1 | 143.5443 | 1.368e4 | 1.368e4 | 1273.92 | 95.31x |

The widened column iterates generic interval propagation to convergence, 10,509 iterations at
layer 0 and 10,634 at layer 1, and it lands on the geometric fixed point as it should.
Induction holds in both layers, residual exactly `0`, zero-order hold confirmed exact.

Scaling with sequence length, layer 0:

| L | 10 | 50 | 100 | 500 | 1000 | 5000 |
| --- | --- | --- | --- | --- | --- | --- |
| invariant | 129.67 | 129.67 | 129.67 | 129.67 | 129.67 | 129.67 |
| unrolled | 127.56 | 601.78 | 1169.03 | 4838.40 | 7781.12 | 1.226e4 |

The unrolled bound is recomputed at each length and crosses the invariant at about `L = 10`.
Above that the invariant is the tighter of the two, and it stops moving.

With `delta = softplus(z)` and no lower bound imposed on it at all, induction still holds:

| layer | invariant | widened fixed point |
| --- | --- | --- |
| 0 | 129.7168 | 1.298e7 |
| 1 | 143.3978 | 1.433e7 |

The invariant needs no floor on `delta`; only the geometric argument does.

## What sets the radius

`M` is `sup |B u| / |A|` over the certified boxes for `B` and `u`. Whether the invariant
exists depends on `A < 0` and `dt > 0` alone; how tight it is depends on those two boxes, and
two structural choices dominate.

| variant | sup abs(u) | radius |
| --- | --- | --- |
| fused LayerNorm and Linear | 8.4267 | 129.6739 |
| LayerNorm box only | 61.3134 | 6332.03 |
| depthwise convolution | 8.4267 | 129.6739 |
| dense convolution | 56.3247 | 9239.72 |

Fusing the LayerNorm with the Linear after it, rather than boxing the LayerNorm and
propagating through, tightens the radius by 49x; keeping the convolution depthwise gives 71x.
Neither affects whether the invariant exists. Both bounds are derived in
[03-certificate.md](03-certificate.md).

Recomputing `M` from a realised trace instead of the certified boxes gives 1.0020 against a
realised `sup |h|` of 0.2066 at layer 0, and 0.5139 against 0.1585 at layer 1. The invariant
is 3 to 5 times loose; the rest of the certified gap is in the bounds on `B` and `u`.

## Against prior work

Invariants for neural sequence models are not new. Jacoby, Barrett and Katz (ATVA 2020)
verify RNNs by inferring an invariant with a verifier: the invariant is the output of a
search, the procedure can return unknown, and what it returns is not a formula in the
parameters. Bonassi, Farina and Scattolini (2021) get a convex-combination invariant for
GRUs, where the convex weights are a learned gate. The gain and time-constant relation in
Proposition 1 is classical linear systems theory, and the architecture is Gu and Dao (2023).

What differs here is where the convex combination comes from. It is neither learned nor
searched for. It is the exact zero-order hold, so it holds for every parameter value the
block could take. The claim is the first closed-form, architecture-specific inductive
invariant for an input-selective state space model, derived analytically rather than inferred
by a verifier, whose assumptions are discharged by the model's own normalisation bounds.
Those assumptions are the boxes on `B` and `u` above, discharged in
[03-certificate.md](03-certificate.md); the block itself is in [01-problem.md](01-problem.md).

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

Three things follow, and all three are visible in the sweep. The gap contains no trained
weight, so it is the same to three figures across seeds while the radius itself moves by 30
percent. It grows as `lam` falls, so the slowest pole attains it, which is why it does not
move with `d_state`. And it grows like `1/d_lo`, which is why lowering the timescale floor by
a decade costs the geometric bound a factor of ten and costs the invariant nothing.

Measured against the certificate the expression is exact elementwise to `5e-7` on the trained
model. `abstraction_gap` in `src/verification/invariant.py` computes it and
`tests/test_invariant.py` checks it.

