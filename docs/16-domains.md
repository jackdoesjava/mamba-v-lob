# Which abstract domain works

Four ways of answering one question, on the same three held-out windows of the same trained
model. The question is how far the prediction can move when every feature of the 100-tick
input window moves by at most `eps` in max norm. Widths are in standardised target units, and
each entry is the median over the three windows. `scripts/08_domains.py` produces the table
and writes [`certification/domains.json`](certification/domains.json).

| eps | attack | lipschitz | interval | global |
| --- | --- | --- | --- | --- |
| 1e-6 | 1.6242e-6 | 2.1798 | 0.5514 | 9.4616 |
| 1e-5 | 1.6302e-5 | inf | 20.402 | 9.4616 |
| 1e-4 | 1.6296e-4 | inf | 2.7576e7 | 9.4616 |
| 1e-3 | 1.6294e-3 | inf | 2.1698e25 | 9.4616 |
| 1e-2 | 1.6310e-2 | inf | nan | 9.4616 |
| 1e-1 | 1.6217e-1 | nan | nan | 9.4616 |

## The four columns

The attack column is a search, not a bound. `attack_width` evaluates the model at 60 random
vertices of the box, each coordinate independently `+eps` or `-eps`, then runs 25 Adam steps
towards each extreme with the step projected back onto the ball. Every value it finds comes
from an input that exists, so the column is a lower bound on the true width. It is the
yardstick for the other three.

The lipschitz column is `src/verification/lipschitz.py`. It keeps the concrete forward pass as
a centre and propagates a bound `Lambda_i >= ||d(component i)/dx||_1` alongside it, so the
certified interval is the nominal value plus or minus `Lambda * eps`. Values never widen; only
the sensitivity does.

The interval column is `src/verification/sensitivity.py`. It pushes the box
`[x - eps, x + eps]` through the whole network in the domain of
[06-interval-domain.md](06-interval-domain.md), with the centre-radius LayerNorm rather than
the input-independent one.

The global column is `certify_model` from `src/verification/certify.py`, the width of the
unconditional output range. It assumes nothing whatever about the input, so it does not depend
on `eps` and is the same number in every row.

All three bounds sit above the attack by construction, as in
[07-soundness.md](07-soundness.md):

```
attacked <= true reachable width <= any sound bound.
```

## The model is locally well behaved

Divide the attack column by `eps` and it is flat: 1.6242, 1.6302, 1.6296, 1.6294, 1.6310,
1.6217. Five decades of perturbation size, and the response stays linear to under one percent.
The model's true local behaviour is a near-constant 1.63 units of prediction width per unit of
max-norm feature move, which under the `width / (2 eps)` convention of `certified_sensitivity`
is a local Lipschitz constant of about 0.81.

Nothing in the rest of this page is caused by the model being difficult. It is not. Whatever
goes wrong in the two local columns is a property of the analysis.

## Both local domains fail, and for the same reason

Interval propagation loses the correlation between a quantity and itself. The domain holds a
box for `h_{t-1}` and a box for `u_t` and combines them as if the two could take their worst
values at once, when both are functions of the same input sequence; that discard happens once
per step for a hundred steps. LayerNorm then makes it worse. `layernorm_input_box` floors the
variance using the concrete centred spread, and once the box radius covers that spread the
floor clamps to zero and the only thing left holding `sigma` up is the LayerNorm's own
`eps = 1e-5`. So `1/sigma` saturates near `1/sqrt(1e-5) = 316`, and a layer that should have
contracted the box multiplies it by a few hundred instead. Block 1 starts wider than block 0
finished, and the column runs from 0.55 to 2.17e25 over three decades of `eps` before
overflowing to `nan`.

Norm-based sensitivity avoids the value blow-up, because the centre is always the concrete
forward pass. It loses on a different quantity. Every `linear` maps `Lambda` to
`Lambda @ |W|^T`, a row sum of absolute weights, which assumes worst-case sign alignment at
every one of them; no cancellation survives a single layer. Then `mul` keeps the second-order
term,

```
Lambda(a*b) = (|a| + Lambda_a eps) Lambda_b + (|b| + Lambda_b eps) Lambda_a,
```

which is what makes the bound sound at finite `eps` rather than only in the limit, and also
what turns it into a runaway: once `Lambda * eps` is comparable with the values it multiplies,
each product feeds its own sensitivity back into the next one. Instrumented on this model,
`Lambda` reaches 4.0 at the stem linear, 93 after the stem LayerNorm, 8996 by the end of
block 0, and diverges inside block 1.

Neither failure is a matter of tuning. The script counts the perturbation sizes at which
either local domain lands within 10 times the attack: none of the six.

## The unconditional bound is the tighter one

At every `eps` from `1e-5` upwards, the constant 9.4616 is below both local columns, in one
case by twenty-four orders of magnitude. That ordering is structural. Each block begins with a
LayerNorm, and the output box of a LayerNorm is fixed by `gamma`, `beta` and `d` alone
([05-layernorm-bound.md](05-layernorm-bound.md)); it does not depend on what reached it. The
global analysis is therefore re-anchored at the start of every block, and no width accumulated
upstream can widen what the next block starts from. The two local analyses have no such reset.
They have to carry the incoming width through the norm, and that is the step at which both of
them lose.

One row goes the other way. At `eps = 1e-6` the interval column returns 0.5514, below the
global 9.4616. It is small rather than tight: at that `eps` the attacked width is 1.62e-6, so
the interval bound is 3.4e5 times the truth and the norm-based one is 1.3e6 times. Priced the
same way at the largest perturbation tested, `eps = 1e-1`, the global bound is 58 times the
attained width, the same order as the looseness already recorded for the certified output
range in [07-soundness.md](07-soundness.md).

## The practical consequence

For this architecture there is no reason to run a local analysis. The bound that assumes
nothing about the input is cheaper to compute, needs no observed window, and is also the
tighter of the available answers at any perturbation size a trading system would care about.
That is not the result one expects. It holds here because the architecture puts a
width-resetting operation at the top of every block, and it should be checked again on any
model that does not.

## What would change the picture

Both failing domains throw away the same thing, which is the sign structure of the dependence
on the input. Interval propagation collapses to a box at every step; norm propagation collapses
to a magnitude at every layer. A relaxation that carries linear coefficients instead, a
zonotope or a CROWN-style backward pass, keeps the cancellation and is the standard remedy in
the neural network verification literature. It is not implemented here.

That is the obvious next step, and the table above is the motivation for it. It is not a claim
that local certification of this model is impossible, only that neither of the two domains
implemented in this repository gets close to it.

## See also

- [06-interval-domain.md](06-interval-domain.md) for the domain the interval column uses.
- [07-soundness.md](07-soundness.md) for the attack methodology and the looseness it measures
  on internal quantities.
- [04-stability.md](04-stability.md) for the contraction factor the sensitivity recursion
  inherits, and [05-layernorm-bound.md](05-layernorm-bound.md) for the reset that makes the
  global column flat.
- [15-risk-limits.md](15-risk-limits.md) for what the global width is used for downstream, and
  [12-limitations.md](12-limitations.md) for what none of this establishes.
