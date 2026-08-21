# Running the pipeline

Everything runs as a module from the repository root:

```
python -m scripts.02_train_models --model mamba
```

The package is `src/`, the seven pipeline scripts are in `scripts/`, the tests in `tests/`.
`load_config` opens `config.yaml` relative to the working directory and every path the
scripts touch is relative too, so running from elsewhere fails at the first file access.
Dependencies are in `requirements.txt`.

## 01_build_features.py

No flags. It reads `data/raw/xnas-itch-20210128.mbp-10.dbn.zst`, a fixed path in the script,
and raises `FileNotFoundError` if it is absent. `OrderBookNormalizer` turns the raw MBP-10
frame into `ts_event`, `mid_price`, `target_log_return` and the 44 engineered features,
written to `config.yaml`'s `data.processed_file` as zstd parquet. Nothing is normalised or
split here; both depend on the train boundary and live in `src/dataset.py`. Feature
definitions are in [08-data.md](08-data.md). `data/raw/` and `data/processed/` are
gitignored, so a fresh clone cannot skip this step.

## 02_train_models.py

```
--model {lstm,mamba,transformer}    required
--max-steps N                       override training.max_steps
--tag NAME                          checkpoint suffix (default: the model name)
--dt-parametrisation {bounded,softplus}
                                    override the config; used for the ablation arm
```

Trains one architecture on purged chronological windows and saves to
`models/checkpoints/best_<tag>.pt`, printing the split report as JSON before the first step.
Budget, evaluation cadence, patience, learning rate and device come from `training` in
`config.yaml`: 3000 optimiser steps by default, evaluated every 250, stopping early after 8
evaluations without improvement.

Selection reads validation rank IC, not validation loss: a constant-zero predictor attains
Huber `0.14979` here while a model at rank IC `+0.14` attains `0.15272`, so loss prefers the
model that predicts nothing. Huber stays the training objective; see
[09-statistics.md](09-statistics.md). `training.amp` is `false` and should stay so, because
fp16 autocast flushes `exp(delta*A)` to exactly zero for around 31% of state entries at the
top of the `delta` range.

## 03_evaluate_models.py

```
--test-batches N     default 200
--ridge-batches N    default 40
```

The only script that loads the test split. It looks for the tags `mamba`, `mamba_softplus`,
`transformer` and `lstm`, skipping any that are missing or fail to load, fits a Ridge
baseline on flattened training windows, and runs on CPU. Rank IC with a circular block
bootstrap interval, hit rate, MSE, a per-decision information ratio and a Diebold-Mariano
matrix go to `models/results/test_metrics.json`, beside
`out_of_sample_predictions.parquet`, with `model_comparison.png` at the repository root.
With no usable checkpoint it exits with an error.

## 04_export_bounds.py

```
--checkpoint PATH        default models/checkpoints/best_mamba.pt
--out PATH               default models/bounds/mamba_certificate.json
--envelope-batches N     default verification.envelope_batches
```

Certification itself needs no data; the dataset is loaded only for the realised envelope,
measured on `N` held-out batches of 32. The artefact carries `dynamics_spec`, the
materialised `A` and `D`, every weight in the chain that generates `delta`, `B` and `C`, the
normalisation and enforced input box, the certified intervals and the looseness ratios, with
an `.npz` of the raw tensors beside the JSON. Contents and caveats:
[11-handoff.md](11-handoff.md).

Two self-checks run first, and either failure exits non-zero with no output file. The round
trip re-runs the recurrence in NumPy from the exported numbers alone against PyTorch, at a
tolerance of `1e-4`; the soundness check requires every realised value to sit inside its
certified box.

## 05_figures.py

```
--certificate PATH    default models/bounds/mamba_certificate.json
--skip-latency
```

Writes `docs/figures/` as matched PDF and PNG pairs. Four certification figures always run:
contraction factor against `dt_min`, bound against sequence length, certified output range
against input scale, and certified against realised looseness. The Diebold-Mariano heatmap
and the regime-conditioned rank IC need the predictions parquet from `03_evaluate_models.py`
and are skipped with a printed message if it is absent. A missing checkpoint falls back to
fresh weights, which will not reproduce the paper's numbers, and the script says so. The
latency figure states its caveat in its own title: the scan is an unfused Python loop, so it
times this implementation rather than `O(L)` against `O(L^2)`.

## 06_certify.py

```
--checkpoint-dir DIR    default models/checkpoints
--seq-len N             default verification.seq_len
```

Produces `docs/certification/ablation.{json,md}`, the paper's headline table, in five
sections: `delta` parametrisation, convolution structure, LayerNorm relaxation, finite
horizon against the geometric fixed point, and discretisation error. Every number is
input-independent, so this runs with no dataset present. A missing checkpoint falls back to
randomly initialised weights and the `source` field records which was used; see
[04-stability.md](04-stability.md).

## 07_attack_bounds.py

```
--checkpoint PATH    default models/checkpoints/best_mamba.pt
--steps N            Adam steps per objective, default 150
--seq-len N          default 100
--box                restrict the input to the enforced normalised feature box
```

Gradient ascent on the input against each certified quantity in turn: `|u|`, both ends of
`delta`, `|B|`, `|C|`, `|h|`, `|y|` and the model output in both directions, per layer, with
the restarts sharing the batch dimension. Results go to `docs/certification/attack.json`, and
the exit code is 1 if anything escaped its box and 0 otherwise, which makes the script usable
as a gate. The two threat models are in [07-soundness.md](07-soundness.md).

## The full sequence

```bash
pip install -r requirements.txt

python -m scripts.01_build_features                      # needs data/raw/*.dbn.zst
python -m scripts.02_train_models --model mamba
python -m scripts.02_train_models --model transformer
python -m scripts.02_train_models --model lstm
python -m scripts.02_train_models --model mamba \
    --tag mamba_softplus --dt-parametrisation softplus   # ablation arm

python -m scripts.03_evaluate_models                     # test split, block bootstrap
python -m scripts.04_export_bounds                       # certificate plus self-checks
python -m scripts.06_certify                             # ablation table
python -m scripts.07_attack_bounds                       # try to break the certificate
python -m scripts.05_figures                             # last: reads 03 and 04

python -m pytest
```

Scripts 04 and 07 need no dataset once a checkpoint exists, and 06 needs neither, so the
certification half reproduces from a clone plus checkpoints with no market data present.
What the suite covers is in [14-tests.md](14-tests.md).

## Seeds and provenance

`config.yaml` carries a single `seed`, 1337. `set_seed` in `src/utils/repro.py` applies it to
`random`, NumPy, torch and CUDA, exports `PYTHONHASHSEED`, and turns off cuDNN benchmarking
so autotuning cannot pick a different algorithm between runs; loader shuffling and the Ridge
sample take explicitly seeded generators. `provenance()` records the git SHA with a `-dirty`
flag, the torch, NumPy and Python versions, the platform and the seed. It is embedded in
every checkpoint and in `test_metrics.json`, `ablation.json`, `attack.json` and the export.

A checkpoint holds `model_state_dict`, `model_name`, `architecture`, the full config,
`normalisation`, `split_report`, `provenance` and the evaluation history. That is enough for
`04_export_bounds.py` to rebuild the model, including its input dimension and `delta`
parametrisation, without consulting `config.yaml` or guessing, and enough to recover which
rows a given model saw.
