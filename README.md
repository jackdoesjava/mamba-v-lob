# A closed-form invariant for a selective state space model

The hidden state of a Mamba-style selective state space model stays inside a box we can write
down in closed form from the model's own parameters. It holds at every step of every sequence,
at any length, and it is derived by hand rather than found by a verifier. The reason is that the
exact zero-order hold makes the state update a convex combination, so a bound on the drive is
already a bound on the state. Limit order book forecasting is the demonstration; the claim is
about the architecture, not about finance.

For diagonal `A < 0` the exact discretisation gives

```
Bbar = A^-1 (exp(dt A) - I) B = (1 - Abar) B / |A|       Abar = exp(dt A) in (0, 1)

h_t = Abar_t h_{t-1} + (1 - Abar_t) c_t,     c_t = B_t u_t / |A|
```

If `|c_t| <= M` for every admissible input, then `|h_{t-1}| <= M` gives `|h_t| <= M`, and
`h_0 = 0` starts inside. No unrolling, no geometric series, no constraint on `dt`. The usual
bound `M_drive / (1 - sup Abar)` bounds `Abar` and `Bbar` separately, discarding the
`(1 - Abar)` factor, which is why it diverges as `dt -> 0` when nothing is diverging.

As far as we can tell this is the first closed-form, architecture-specific inductive invariant
for an input-selective state space model, derived analytically rather than inferred by a
verifier, whose assumptions are discharged by the model's own normalisation bounds. Invariants
for neural sequence models are not new; Jacoby, Barrett and Katz (ATVA 2020) infer them for RNNs
with a verifier, and Bonassi, Farina and Scattolini (2021) get a convex combination for GRUs out
of a learned gate rather than out of a discretisation.

## Headline numbers

Trained model, `L = 100`, layer 0, input-independent unless stated.

| quantity | value |
| --- | --- |
| certified `sup abs(h)` from the invariant | `129.67` |
| converged fixed point of the interval recursion | `1.235e4`, so the invariant is `95x` tighter |
| dependence on sequence length | none; `129.67` at `L = 10` and at `L = 5000` |
| dependence on the weights | none to three figures; the ratio is `95.2` across 8 seeds |
| induction residual | exactly `0`, and the zero-order hold identity is exact to `1e-14` |
| certified output range, for any input at all | `[-5.1242, 4.3374]` |
| adversarial search against the trained model | no violation found |
| machine-checked lemmas, Z3 | 16 of 19 discharged, zero counterexamples |

The invariant needs no floor on `delta`: the softplus arm certifies at `129.72` while the
geometric argument diverges. Numbers are in [docs/03-certificate.md](docs/03-certificate.md).

## Layout

```
src/models/        the selective SSM and the transformer and LSTM baselines
src/verification/  interval domain, the invariant, certificate export, NumPy reference
src/dataset.py     purged chronological splits, enforced input box
src/features.py    order book feature construction
src/utils/         config, seeding and provenance, overlap-aware statistics
scripts/           01 features, 02 train, 03 evaluate, 04 export, 05 figures, 06 certify,
                   07 attack, 08 domains, 09 lemmas, 10 sweep, 11 artefact check
tests/             unit tests for the maths, the domain and the models
docs/              the five pages listed at the bottom
```

## Running

```bash
pip install -r requirements.txt
python -m scripts.01_build_features          # needs data/raw/*.dbn.zst
python -m scripts.02_train_models --model mamba
python -m scripts.02_train_models --model transformer
python -m scripts.02_train_models --model lstm
python -m scripts.02_train_models --model mamba --tag mamba_softplus \
    --dt-parametrisation softplus
python -m scripts.03_evaluate_models         # test split, block bootstrap
python -m scripts.04_export_bounds           # certificate, with self-checks
python -m scripts.06_certify                 # invariant against the alternatives
python -m scripts.07_attack_bounds           # try to break the certificate
python -m scripts.08_domains                 # local domains, the negative result
python -m scripts.09_check_lemmas            # Z3 on the interval lemmas
python -m scripts.10_sweep                   # seeds and configurations
python -m scripts.05_figures
python -m scripts.11_check_artefacts         # nothing in docs/ is stale
python -m pytest
```

`11_check_artefacts` is the one to run before quoting a number or submitting anything. Nothing
fails when a figure or a results table goes stale, which is how a plot arguing a claim this
project had already dropped survived here for several hours.

Scripts run as modules from the repository root, not as files. Everything is seeded from
`config.yaml`, and checkpoints carry their architecture, normalisation, split report and git SHA,
so an export can rebuild its model without being told what it was.

## Data

Databento MBP-10 from Nasdaq TotalView-ITCH, GME on 2021-01-28. One instrument, one day,
4,550,476 rows after feature construction and 43 model inputs. The purged three-way chronological
split, 100 ticks at each boundary, gives 3,185,134 train windows, 682,372 validation and 682,473
test. Features are winsorised to train-split quantiles at inference, so the input box
`[-6.60, 18.87]` is enforced rather than observed, with zero violations across all 4.55M rows.

None of the certification depends on any of this. The invariant is a statement about the
architecture and the trained weights, and it holds for inputs no order book would produce. The
predictive numbers only show that the certified model is a real model; one instrument on one day
supports nothing beyond that.

## Limitations

Against a realised trace the certified radius is about 630 times the realised state; fed the
realised `B` and `u` instead of the certified boxes the invariant is 3 to 5 times loose, so the
induction accounts for a factor of 5 and the boxes on `B` and `u` for the rest. This bounds reachable state and output; it
is not a robustness certificate, and a bounded output is not a bounded loss. Both local domains we
tried diverge on inputs where the model is measurably well behaved, a failure of the domains and
not of the model. Three lemmas are still hand proofs after Z3 timed out, and no external verifier
has consumed the artefact. Set out in [docs/05-handoff.md](docs/05-handoff.md).

## Documentation

- [docs/01-problem.md](docs/01-problem.md), the block, what is time-varying, and why bounded verification struggles
- [docs/02-invariant.md](docs/02-invariant.md), the discretisation identity, the invariant and its proof
- [docs/03-certificate.md](docs/03-certificate.md), tightness, adversarial search, machine-checked lemmas
- [docs/04-results.md](docs/04-results.md), data, baselines and the predictive comparison
- [docs/05-handoff.md](docs/05-handoff.md), limitations and what the verification side needs next
