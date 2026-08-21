# Certified bounds for a selective state space model on limit order book data

A proof of concept. It asks what can be **proved** about a Mamba-style selective state space
model forecasting short-horizon returns from a limit order book, and what has to change in the
architecture before anything can be.

The answer is a negative result with a constructive fix.

> Mamba computes its timescale as `delta = softplus(z)`, and softplus has infimum zero. Since
> `Abar = exp(delta A)`, that makes `sup Abar = 1`, so the state recursion has no uniform
> contraction factor and no finite invariant set. Selectivity, the thing that makes Mamba work,
> is what removes the stability guarantee people assume it has. Constraining `delta >= dt_min > 0`
> restores one, and costs a single line of the parametrisation.

Proofs are in [docs/04-stability.md](docs/04-stability.md). Start at
[docs/README.md](docs/README.md).

## What is established

Every row holds for **all** inputs, in distribution or not, and none of it depends on the
dataset. Numbers are for the trained model in `models/checkpoints/best_mamba.pt`.

| | result |
| --- | --- |
| `Abar` in `(0,1)` elementwise | for all parameters and all inputs, by construction |
| `sup Abar < 1` | only if `delta` is bounded below; `0.9990` here |
| certified `sup abs(h)` at `L = 100` | `1.17e3` and `1.27e3`, against infinite under softplus |
| certified output range | `[-5.1242, 4.3374]`, for any input whatsoever |
| enforced input box | `[-6.60, 18.87]`, zero violations across 4.55M rows |
| export round trip against PyTorch | `8.821e-07` worst over every intermediate |
| gradient-based falsification | no violation found |

Four architectural choices, each priced in [docs/certification/ablation.md](docs/certification/ablation.md):

| constraint | effect on certified `sup abs(h)` |
| --- | --- |
| `delta` bounded below | infinite to `1.17e3` |
| depthwise rather than dense convolution | 69 times tighter, 128 times fewer conv parameters |
| fuse LayerNorm with the following Linear | 49 times tighter, and the fused bound is exact |
| finite horizon rather than geometric sum | 10.6 times tighter at `L = 100` |

## Layout

```
src/
  models/          the selective SSM and two baselines
  verification/    interval domain, certificate, NumPy reference implementation
  dataset.py       purged chronological splits, enforced input box
  features.py      order book feature engineering
  utils/           config, seeding and provenance, overlap-aware statistics
scripts/           01 build, 02 train, 03 evaluate, 04 export, 05 figures,
                   06 ablation, 07 adversarial attack
tests/             65 tests; soundness is checked by sampling, not assumed
docs/              see docs/README.md
models/            checkpoints, exported certificate, results
```

## Running

```bash
pip install -r requirements.txt

python -m scripts.01_build_features          # needs data/raw/*.dbn.zst
python -m scripts.02_train_models --model mamba
python -m scripts.02_train_models --model transformer
python -m scripts.02_train_models --model lstm
python -m scripts.02_train_models --model mamba \
    --tag mamba_softplus --dt-parametrisation softplus

python -m scripts.03_evaluate_models         # test split, block bootstrap
python -m scripts.04_export_bounds           # certificate, with self-checks
python -m scripts.06_certify                 # ablation table
python -m scripts.07_attack_bounds           # try to break the certificate
python -m scripts.05_figures

python -m pytest
```

Everything is seeded from `config.yaml`. Checkpoints carry their architecture, normalisation,
split report and git SHA, so an export can rebuild its model without guessing. Full detail in
[docs/13-running.md](docs/13-running.md).

## Three things to know before reading the code

`B`, `C` and `delta` are activations, not parameters. Only `A` and `D` live in the state dict.
A selective SSM is linear time-varying, so the common claim that Mamba is an LTI system and
therefore boundable is false. See [docs/01-problem.md](docs/01-problem.md).

The discretisation is exact. Reference Mamba discretises `A` exactly but `B` by forward Euler,
which is consistent only when `abs(delta A) << 1`. At the unconstrained timescale this model
first learned, that shortcut carried 411 percent median error. See
[docs/03-discretisation.md](docs/03-discretisation.md).

The statistics account for overlap. The label is a 100-tick-ahead return sampled every tick, so
naive standard errors understate uncertainty by roughly ten times here. See
[docs/09-statistics.md](docs/09-statistics.md).

## Limitations

Set out in full in [docs/12-limitations.md](docs/12-limitations.md). The short version: the
state bound is loose by roughly `1e4` against the realised envelope; this bounds reachable
state and output and is not a robustness certificate; a bounded output is not a bounded loss;
the data is one instrument on one day; no external verifier has consumed the artefact yet; and
the novelty claim has not been checked against the literature.
