# A closed-form invariant for a selective state space model

The hidden state of a Mamba-style selective state space model stays inside a box we can write
down in closed form from the model's own parameters. It holds at every step of every sequence,
at any length. The reason is that the exact zero-order hold makes the state update a convex
combination, so a bound on the drive is already a bound on the state. The same argument applied
to the difference of two runs bounds how far a perturbed input can move the state, again at every
length and with no floor on the timescale. Limit order book forecasting is the demonstration, and
the released language model `state-spaces/mamba-130m-hf` is the second test case; the claims are
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

The invariant is not new. It is Horbatko's Lemma 4.4 (arXiv:2604.18580), and the same per-cell
radius follows from Theorem 3.3(i) of Cao, Liu, Chen and Qin (ICML 2026). The LayerNorm
coordinate bound that discharges its premise is the Laguerre-Samuelson inequality. The paper's
one new result is the cell theorem: with the timescale perturbation measured in `log dt`, two
runs of one cell that start together stay within `eta + 2 M (e^rho - 1)` of each other at every
length, with no lower bound on `dt`. It strengthens the time-warp part of Cao et al.'s Theorem
3.3(ii), whose bound grows with the position unless `dt` has a floor, and it can be derived from
their lemmas by a change of time variable. A construction in the paper shows that no bound below
`eta + 2 M tanh(rho/2)` holds, so the constant is within a factor `e^rho + 1` of the best
possible. The endpoint form of the local rule and the certification of the released model are
applications of it. The code also uses `2 M tanh(rho/2)` as an upper bound for two runs from a
common start ([docs/02-invariant.md](docs/02-invariant.md)); that direction is not part of the
paper. The literature search behind these statements was not exhaustive.

## Headline numbers

Trained bounded-timescale model, `L = 100`, layer 0, input-independent unless stated.

| quantity | value |
| --- | --- |
| certified `sup abs(h)` from the invariant | `175.95` (layer 1: `137.22`) |
| converged fixed point of the interval recursion | `1.682e4`, so the invariant is `95.6x` tighter |
| dependence on sequence length | none; `175.95` at `L = 10` and at `L = 5000` |
| dependence on the weights | only through the slowest pole; the ratio is `95.20` to `95.21` across 8 untrained seeds and `95.62` on the trained checkpoint |
| induction residual | exactly `0`, and the zero-order hold identity checks exact |
| certified output range, for any input at all | `[-3.5643, 3.1277]` |
| adversarial search against the certificate | no violation found |
| data-free sensitivity bound per block | the same at every length, `3.06e5` to `3.23e6` times the attacked slope |
| local sensitivity bound per block, endpoint rule, `delta = 1e-6` | `3.37` to `12.6` times the attacked slope (additive rule `3.41` to `18.0`) |
| what the log-timescale interval adds to that local bound, `delta <= 1e-4` | at most `0.034` per cent; the gain over the additive rule is the convex step's |
| released model, local endpoint rule against the additive rule | tighter in 72 of 72 (layer, length) pairs at each of `delta = 1e-6, 1e-4, 1e-2` |
| released model, what the log-timescale interval adds to the endpoint rule | at most `0.07` per cent at `delta = 1e-6`; the gain over the additive rule is the convex step's |
| machine-checked lemmas, Z3 | 16 of 19 discharged, zero counterexamples |

The invariant needs no floor on `delta`: the softplus arm certifies at `287.39` while the
geometric argument diverges. Numbers are in [docs/04-results.md](docs/04-results.md).

## Layout

```
src/models/        the selective SSM and the transformer and LSTM baselines
src/verification/  interval domain, the invariant, the sensitivity bound (incremental.py),
                   the released-model certificate (released.py), export, NumPy reference
src/dataset.py     purged chronological splits, enforced input box
src/features.py    order book feature construction
src/utils/         config, seeding and provenance, overlap-aware statistics
scripts/           01 features, 02 train, 03 evaluate, 04 export, 05 figures, 06 certify,
                   07 attack, 08 domains, 09 lemmas, 10 sweep, 11 artefact check,
                   12 sensitivity bound, 13 attack on it, 14 released model,
                   15 local-certificate figures, 16 worst cases of the cell theorem
tests/             unit tests for the maths, the domain and the models
data/text/         public_domain_sample.txt, the text the released model is certified on
isabelle/          the machine-checked invariant and LayerNorm bounds, Isabelle/HOL, and an
                   unchecked draft of the sensitivity lemmas in isabelle/drafts
docs/              the five pages listed at the bottom, the recorded runs in docs/certification
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
python -m scripts.12_incremental             # sensitivity to input perturbations, any length
python -m scripts.13_incremental_attack      # attack that bound, about an hour on a CPU
python -m scripts.14_released                # mamba-130m-hf, about 1.5 hours on a CPU
python -m scripts.15_local_figures           # the local-certificate figures, from 13 and 14's output
python -m scripts.16_cell_tightness          # worst cases found against the cell theorem, minutes on a CPU
python -m scripts.05_figures
python -m scripts.11_check_artefacts         # nothing in docs/ is stale
python -m pytest
```

`11_check_artefacts` is the one to run before quoting a number or submitting anything. Nothing
fails when a figure or a results table goes stale, which is how a plot arguing a claim this
project had already dropped survived here for several hours. It also rejects phrases from
retracted claims. By default it compares commit times, so a fresh clone does not look stale; it
therefore cannot see edits that are not yet committed, and before committing it should be run as
`python -m scripts.11_check_artefacts --mtime`, which compares file times.

`14_released` needs `transformers` (5.15.1 was used) and a local copy of
`state-spaces/mamba-130m-hf`. The model is not committed and nothing is downloaded:
`src/verification/released.py` loads it with `local_files_only=True` from `MAMBA_130M_DIR`, or from
`~/hf_models/mamba-130m-hf` when that is unset, and the run records the SHA-256 of the weights.
Its tests skip without the model.

Scripts run as modules from the repository root, not as files. Everything is seeded from
`config.yaml`, and checkpoints carry their architecture, normalisation, split report and git SHA,
so an export can rebuild its model without being told what it was.

## Data

Databento MBP-10 from Nasdaq TotalView-ITCH, SPY (the S&P 500 ETF) on 2021-01-28. One
instrument, one day, 4,550,476 rows after feature construction in
`data/processed/SPY_features_20210128.parquet`, and 43 model inputs. Raw and processed data stay
local. The purged three-way chronological split, 100 ticks at each boundary, gives 3,185,134 train
windows, 682,372 validation and 682,473 test. Features are winsorised to train-split quantiles at
inference, so the input box `[-22.48, 19.25]` is enforced rather than observed.

None of the certification depends on any of this. The invariant is a statement about the
architecture and the trained weights, and it holds for inputs no order book would produce. The
predictive numbers only show that the certified model is a real model; one instrument on one day
supports nothing beyond that.

## Limitations

The certified radius is 162 times the largest state realised on held-out batches at layer 0 and
382 times at layer 1, and 63 and 49 times what a gradient attack reaches. This bounds reachable
state and output; it is not a robustness certificate, and a bounded output is not a bounded
loss. The data-free sensitivity bound is sound at every length and 3.1e5 to 3.2e6 times the
attacked sensitivity. With the endpoint rule the local block bound is within a factor of 13 of
the attack, but end to end it is 5.7e5 times the attack on the bounded arm and 1.6e7 times on the
softplus arm, and of the `eps` values tested end to end from `1e-6` up it is below the global
certified width only at `1e-6` on the bounded arm. The two Mamba arms share seed, initialisation
and batch order and were selected at different steps, so they are not independent models and
support no conclusion about the parametrisation. On the released model the data-free certificate
is finite and uninformative, and the bounds cover the float64 map, not the float32 rounding in
transformers' implementation. The Isabelle sensitivity draft has never been checked, and no
external verifier has consumed the artefact. Set out in [docs/05-handoff.md](docs/05-handoff.md).

## Documentation

- [docs/01-problem.md](docs/01-problem.md), the block, what is time-varying, and why bounded verification struggles
- [docs/02-invariant.md](docs/02-invariant.md), the discretisation identity, the invariant, the cell theorem, the construction that bounds its constant from below, and the sharper common-start constant
- [docs/03-certificate.md](docs/03-certificate.md), the normalisation bounds, tightness, adversarial search, local rules, the released model
- [docs/04-results.md](docs/04-results.md), every recorded number in one place, with the data and the predictive comparison
- [docs/05-handoff.md](docs/05-handoff.md), the artefact, limitations and what the verification side needs next
