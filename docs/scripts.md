# Execution Pipeline

This directory (`python/scripts/`) defines the executable research pipeline. The architecture is modularized to decouple data preprocessing from optimization and inference, enabling rapid hyperparameter iteration without redundant I/O overhead.

---

## Data Generation
**File:** `01_build_features.py`

Executes the deterministic preprocessing pipeline on Databento MBP-10 feeds. It computes the spatial relative coordinate mappings and log-scaled volume regularizations, serializing the fully normalized feature space into PyTorch (`.pt`) tensors. This offline materialization is strictly required to prevent CPU-bound I/O bottlenecks during subsequent mini-batch generation.

---

## Model Optimization
**File:** `02_train_models.py`

Implements the primary PyTorch training loop. It utilizes AdamW optimization with dynamic early stopping predicated on out-of-sample validation loss. To capture the varying convergence dynamics across the causal architectures, the script asynchronously caches model state dictionaries to `models/checkpoints/` immediately upon achieving new validation minimums.

*Note:* Due to the explicitly unrolled SSM transitions, CPU-bound execution requires prefixing the run with `OMP_NUM_THREADS=1` to prevent OpenMP thread deadlocks during the sequential scan.

---

## Out-of-Sample Evaluation
**File:** `03_evaluate_models.py`

Conducts strict out-of-sample inference and statistical benchmarking. Operating entirely under `torch.no_grad()` contexts, it evaluates the Mamba SSM against the Transformer, LSTM, and a Ridge regression baseline. It computes core quantitative metrics—Hit Rate, Information Coefficient (Pearson/Rank IC), Pseudo-Sharpe, and Maximum Drawdown—and exports the cumulative return distributions to `model_comparison.png`.

---

## Verification Matrix Export 
**File:** `04_export_bounds.py`

Bridges the empirical models with the downstream formal verification engine. This script extracts the strict empirical bounding hyper-rectangles of the feature space:

$$
[x_{\min}, x_{\max}]
$$

alongside the converged continuous-time SSM parameters ($A, B, C, \Delta$). These matrices are serialized into standard data structures, allowing Daniel to ingest them directly into the theorem prover to establish the topological stability boundaries of the network.