# Execution Pipeline

The `python/scripts/` directory contains the core executable pipeline. I split this up into modular, sequential steps so we don't have to re-run the entire data ingestion process every time we want to tweak a model hyperparameter or test a new evaluation metric.

---

## Data Generation
**File:** `01_build_features.py`

This script ingests the raw Databento order book feeds and applies the transformations defined in our utility files. It handles the heavy lifting of calculating the relative price distances and log-scaling the depth volumes, ultimately outputting normalized `.pt` tensor files to disk. Running this once locally saves a massive amount of computational overhead during the training loop.

---

## Training Loop
**File:** `02_train_models.py`

The main PyTorch training script. It manages the optimization (using AdamW) and implements early stopping based on validation loss. Since we're dealing with high-frequency LOB data and varying convergence times, it's set up to save the best-performing weights to `models/checkpoints/` dynamically as soon as a new lowest validation loss is hit, rather than waiting for all epochs to finish. 

*Note: When training the custom Mamba model locally on a CPU, you need to prefix the execution with `OMP_NUM_THREADS=1`. This prevents OpenMP from deadlocking your processor threads during the sequential scan.*

---

## Model Evaluation
**File:** `03_evaluate_models.py`

This script handles out-of-sample inference. It freezes all model weights using `torch.no_grad()`, runs the validation set through the Ridge baseline, LSTM, Transformer, and Mamba models, and calculates the key execution metrics (Hit Rate, Pearson/Rank IC, Pseudo-Sharpe, and Max Drawdown). Finally, it plots the cumulative return curves and exports the tear sheet to `model_comparison.png`.

---

## Verification Matrix Export 
**File:** `04_export_bounds.py`

Because Daniel is handling the formal verification proofs, he needs the exact mathematical boundaries of the trained model, not just PyTorch weights. This script extracts the extreme empirical limits of the feature space ($[x_{\min}, x_{\max}]$) alongside Mamba's continuous-time transition matrices ($A, B, C, \Delta$), dumping them into a clean, parsed format that can be directly ingested by the theorem prover.