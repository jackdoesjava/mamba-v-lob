# A log-timescale perturbation bound for selective state space models

Code, results and LaTeX source for the paper *A Log-Timescale Perturbation Bound for Selective
State Space Models at Every Sequence Length* by Jack Humphries and Daniel Midgley
([paper/main.tex](paper/main.tex)).

A Mamba model run on a stream, such as limit order book updates, carries its state far beyond its
training windows. A bound on how far a small change of input moves that state therefore has to
hold at every length. The closest bound we know of, Theorem 3.3(ii) of Cao, Liu, Chen and Qin
(ICML 2026), measures the change of the timescale `dt` additively. It grows linearly with the
length unless `dt` has a floor, and for Mamba's softplus timescale that floor can be tiny: about
`1e-21` on one of our trained models.

The paper's one result measures the change in `log dt` instead. Under the exact zero-order hold,
one cell of the selective scan is

```
h_t = Abar_t h_{t-1} + (1 - Abar_t) c_t,     Abar_t = exp(-lambda dt_t),   c_t = B_t u_t / lambda
```

If `|log dt_t - log dt~_t| <= rho`, `|c~_t| <= M` and `|c_t - c~_t| <= eta`, two runs that start
from the same state in `[-M, M]` stay within `eta + 2 M (e^rho - 1)` of each other at every
length, with no floor (Theorem 3). The reason is Lemma 2: a change of `log dt` by at most `rho`
moves the transition by at most `e^rho - 1` times the input weight `1 - Abar_t`, so the
disturbance at each step shrinks with the step and the disturbances cannot pile up. A construction
(Proposition 4) gives gaps arbitrarily close to `eta + 2 M tanh(rho/2)`, so the constant is within
a factor `e^rho + 1` of the best possible, about 2 for small `rho`.

The bound on the state itself, `|h_t| <= M`, is not ours. It is Horbatko's Lemma 4.4
(arXiv:2604.18580), and it also follows from Cao et al.'s Theorem 3.3(i). The theorem can also be
derived from Cao et al.'s Lemma B.2 after a change of time variable (Appendix B of the paper). The
literature search behind these statements was not exhaustive.

## Results in the paper

A two-block model trained on one day of SPY order book data, once with a bounded timescale and
once with Mamba's softplus timescale. Certificates are per block and given as slopes, the limit of
bound over perturbation size.

| quantity | value |
| --- | --- |
| one-cell search against `eta + 2 M tanh(rho/2)`, `T >= 300` | no gap above it; within 0.4 per cent from random starts and 0.01 per cent from starts shaped like the construction |
| bounded timescale: steps for which the additive baseline is tighter | at most the first 8 |
| bounded timescale: additive baseline at `L = 5000`, no floor | about 540 times ours |
| bounded timescale: additive baseline with its certified floor, `L -> inf` | 22 to 23 times ours |
| softplus timescale: additive baseline | larger at every length, about `2e4` times ours at `L = 5000` |
| our certificate against gradient attacks, `delta = 1e-6` | `3.06e5` to `3.23e6` times the attack slope |
| local bound around a realised input, `delta <= 1e-4` | at most 12.7 times the attack slope |

Figure 1 of the paper ([scripts/17_mechanism_figure.py](scripts/17_mechanism_figure.py)) shows the
idea and the construction. Every recorded number is in [docs/04-results.md](docs/04-results.md)
and [docs/certification](docs/certification).

## Beyond the paper

The repository also holds work that the paper leaves out:

- the invariant certificate on the trained model: certified `sup |h|` of `175.95` on layer 0,
  `95.6` times tighter than the converged fixed point of generic interval propagation;
- local certificates with four update rules, and the ablation showing that the log-timescale
  interval changes the local bound by at most 0.034 per cent for `delta <= 1e-4`;
- the released `state-spaces/mamba-130m-hf`, where the data-free certificate is finite but
  uninformative;
- an end-to-end local chain through the whole order book model;
- Isabelle/HOL proofs of the invariant and the LayerNorm bounds, and Z3 checks of the interval
  lemmas.

The code also uses `2 M tanh(rho/2)` as an upper bound for two runs from a common start. Its
proof is in an earlier draft and is not part of the paper ([docs/02-invariant.md](docs/02-invariant.md)).

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
                   15 local-certificate figures, 16 worst cases of the cell theorem,
                   17 the figure of the theorem's idea and the lower-bound construction
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
python -m scripts.17_mechanism_figure        # the theorem's idea and the lower-bound construction, seconds
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

None of the theory depends on any of this. The theorem is a statement about the architecture,
and the certificates hold for inputs no order book would produce. The predictive numbers only
show that the certified model is a real model; one instrument on one day supports nothing beyond
that.

## Limitations

The theorem covers one scalar cell with a real transition in `(0, 1)`. It needs a diagonal `A`
and does not cover the complex-valued update of Mamba-3. The certificates are per block, with
respect to the normalised input, and are not composed through the residual stream and the head.
They are computed in floating point without outward rounding, so they are real-number bounds only
up to rounding; at `delta = 1e-6` a float32 scan is covered only once a rounding term is added.
The data-free certificates are `3e5` to `3e6` times what gradient attacks find. The trained models
use the exact zero-order hold, not the exponential-Euler step of the released Mamba code, and the
two arms are single training runs from one set-up on one instrument and one day. The Isabelle
sensitivity draft has never been checked. Set out in Section 7 of the paper and in
[docs/05-handoff.md](docs/05-handoff.md).

## Documentation

- [docs/01-problem.md](docs/01-problem.md), the block, what is time-varying, and why bounded verification struggles
- [docs/02-invariant.md](docs/02-invariant.md), the discretisation identity, the invariant, the cell theorem, the construction that bounds its constant from below, and the sharper common-start constant
- [docs/03-certificate.md](docs/03-certificate.md), the normalisation bounds, tightness, adversarial search, local rules, the released model
- [docs/04-results.md](docs/04-results.md), every recorded number in one place, with the data and the predictive comparison
- [docs/05-handoff.md](docs/05-handoff.md), the artefact, limitations and what the verification side needs next
