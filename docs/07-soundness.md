# Soundness

A certificate is a claim about every input, and no finite procedure checks every input. What
can be done is to attack the claim from several directions and record what survives. Three
checks run against this one, in increasing order of strength: sampling, a round trip through
NumPy, and gradient ascent aimed at breaking the bounds.

## Sampling

Every abstract operation in `src/verification/intervals.py` has a test that draws concrete
values from an input box and asserts the concrete output lands inside the computed output
box. For the elementwise primitives, `SiLU`, `sigmoid`, `softplus` and `exp`, that is 50
random boxes each. A separate test covers a box straddling `SiLU`'s minimum, checking the
output box reaches down to `-0.278465` rather than stopping at the endpoints; that is the
one case where evaluating at the endpoints alone is unsound. Interval products, affine maps
and the depthwise convolution are sampled the same way, with the convolution test building
concrete sequences from the per-channel box and comparing the realised per-channel extrema.

The LayerNorm box is checked at `d` in `{8, 64, 256}`, at input scales spanning eight orders
of magnitude, and against one-hot directions chosen to be extreme. Proposition 2 is a
statement about all of `R^d`, so the failure to look for is a direction the proof missed,
not a scale it cannot reach.

The fused LayerNorm-Linear bound gets a stronger test than the others. It is checked to be
sound, to be strictly tighter than bounding the two layers separately, and to be attained:
the test constructs the adversarial maximiser

```
z* = sqrt(d) * (v - mean(v) 1) / ||v - mean(v) 1||_2
```

and asserts the realised output matches the bound to `1e-3` relative. A bound that is both
sound and attained cannot be improved, so this pins it from below as well as above. See
[05-layernorm-bound.md](05-layernorm-bound.md) for the statement.

At whole-model level the certified output range is checked to hold at input scales from
`1e-3` to `1e6`, and the block certificate is checked to contain every realised `u`,
`delta`, `B`, `C`, `Abar`, `h` and `y`. Sampling shows a bound is plausible. It cannot show
the bound holds, because the sampler is not looking for the worst case.

## Round trip

`scripts/04_export_bounds.py` runs two self-checks before it writes anything, and exits
non-zero on either. The first re-runs block 0's recurrence in NumPy from the exported `A`
and `D` and the traced `delta`, `B`, `C`, then compares the outputs and the final state
against PyTorch. The second is the soundness check: every realised value must sit inside its
certified box.

| round-trip quantity | measured | tolerance |
| --- | --- | --- |
| max abs error in `y` | 2.086e-07 | 1e-4 |
| max abs error in final `h` | 8.724e-08 | 1e-4 |

The residual is float32 rounding accumulated over the sequence and re-run in float64. This
checks a different property from soundness, namely that the artefact describes the model at
all. A certificate proved about a recurrence the network does not run would be sound and
useless, and that failure is invisible to everything else on this page, since sampling and
the attack both go through the same PyTorch module. The export ships its equations in
`dynamics_spec` so an external tool can read them; the round trip is what makes the
description trustworthy.

## Gradient-based falsification

`scripts/07_attack_bounds.py` runs Adam on the input to maximise each internal quantity,
with the goal of pushing it outside its box: `|u|`, `delta` at both ends, `|B|`, `|C|`,
`|h|`, `|y|` and the model output in both directions, per layer. The script exits non-zero
if anything escapes.

Two threat models. With `--box` the input is restricted to the enforced normalised feature
box, which is the deployment regime. Without it the input is unrestricted, which is the
regime Propositions 2 to 7 claim to cover with no premise at all, and where LayerNorm is
doing the work; a violation there would be the more damaging of the two. No violation has
been found in either.

## What the attack measures

Optimising against a bound prices it. Since

```
attacked <= true reachable <= certified,
```

the gap between the attacked and certified columns upper-bounds the abstraction's
looseness, and where the attack lands on the certified value the bound is near-exact.

| quantity | attacked | certified | looseness |
| --- | --- | --- | --- |
| `sup delta` | 0.0932 | 0.1000 | 1.07x |
| `inf delta` | 1.10e-3 | 1.00e-3 | 1.10x |
| `sup \|u\|` | 4.53 | 7.63 | 1.7x |
| `sup \|B\|` | 1.26 | 16.4 | 13x |
| `sup \|C\|` | 1.30 | 16.5 | 12.7x |
| output max | 0.91 | 6.30 | 6.9x |
| output min | -1.43 | -8.06 | 5.6x |
| `sup \|h\|` | 0.54 | 2.68e2 | 498x |
| `sup \|y\|` | 4.51 | 4.18e4 | 9.3e3x |

These are layer 0 of two. Layer 1 tracks them closely apart from the state row, where the
attacked value is 0.30 against a certified 2.94e2. The recorded run is in
[`certification/attack.json`](certification/attack.json).

## Which rows matter

The `delta` rows are the ones the argument depends on. The bound there is a property of the
parametrisation and its clamp rather than of an accumulation, so nothing is given away in
reaching it, and the attack confirms both endpoints are attainable to within 10%. That is
what makes Proposition 5 usable rather than merely true: the contraction factor
`s = exp(-dt_min * min|A|) = 0.9990` is read off `dt_min`, and `dt_min` is a value the input
can actually produce. See [04-stability.md](04-stability.md).

The state and output rows are the cost of interval arithmetic over a 100-step recursion. At
each step the domain holds a box for `h_{t-1}` and a box for `u_t` and combines them as
though the two were free to take their worst values at once. They are not; both are
functions of the same input sequence. That correlation is discarded once per step, a hundred
times, and the factors of 500 and 1e4 are the discard compounding rather than slack a better
constant would remove.

A zonotope or CROWN-style relaxation tracks linear dependence on the input through the
recursion instead of collapsing to a box at every step, which keeps the correlation the
interval domain drops. It is the obvious next step, and it would move the state and output
rows without changing any proposition.

## See also

- [14-tests.md](14-tests.md) for what the 60 tests cover.
- [13-running.md](13-running.md) for the flags on `04_export_bounds.py` and
  `07_attack_bounds.py`.
- [04-stability.md](04-stability.md) and [05-layernorm-bound.md](05-layernorm-bound.md) for
  the propositions these checks are aimed at.
