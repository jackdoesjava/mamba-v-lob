# Utility & Data Ingestion Modules

This directory (`python/utils/`) contains the data ingestion and PyTorch loading infrastructure, mapping the raw Databento MBP-10 feeds into stationary, model-ready tensors.

---

## Stationary Feature Engineering (`features.py`)

Raw limit order book prices are $I(1)$ non-stationary processes. Feeding absolute prices directly into sequential models causes them to fit spurious correlations to macroscopic price drift rather than learning the high-frequency microstructure dynamics. 

This module enforces strict stationarity across the feature space:
* **Spatial Centering (Price):** All 10 levels of bid and ask prices are transformed into relative coordinate distances from the instantaneous mid-price.
* **Volume Regularization:** Deep-book liquidity exhibits heavy-tailed variance and extreme queue imbalances. We apply $\ln(x+1)$ scaling to all volume depths to compress outliers and stabilize gradient flow during training.
* **Target Generation:** Computes the forward 100-tick log return of the mid-price. Strict index masking and temporal purging are enforced during this step to mathematically guarantee zero look-ahead bias (temporal leakage) in the training targets.

---

## High-Throughput Dataloaders (`dataset.py`)

This module implements the PyTorch `Dataset` and `DataLoader` classes, optimized for the memory constraints of high-frequency ITCH data.

* **Temporal Striding:** Implements a sliding window algorithm to generate overlapping context sequences (e.g., $L=100$). This constructs the strict chronological event horizons required by the causal models (Mamba, Causal Transformer, LSTM) for Backpropagation Through Time (BPTT).
* **Memory Management:** Because granular LOB data easily exceeds local RAM limits, the loader relies on efficient memory-mapped arrays and strictly downcasts batched views to `torch.float32` immediately before yielding. This stabilizes the memory footprint and prevents Out-of-Memory (OOM) faults during the local training loop.