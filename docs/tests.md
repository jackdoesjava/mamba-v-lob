# Testing & Validation Suite

This directory (`python/tests/`) contains the unit testing suite executed via `pytest`. The objective is to proactively isolate silent failures, such as dimensional broadcasting faults and temporal leakage in the data pipeline, prior to executing full optimization loops.

---

## Feature Invariants (`test_features.py`)

These tests enforce the strict mathematical integrity of the MBP-10 transformations:
* **Numerical Stability:** Asserts that the coordinate normalizations and log-scaled volume regularizations do not introduce `NaN` or `Inf` values across sparse edge cases, such as zero-volume depth updates.
* **Microstructure Constraints:** Verifies the physical invariants of the limit order book. This confirms the top-of-book spread remains strictly positive ($P_{\text{ask}, 0} > P_{\text{bid}, 0}$) and the spatial transformations maintain strictly bounded half-spaces ($D_{\text{ask}} \ge 0, D_{\text{bid}} \le 0$).
* **Temporal Purging:** Confirms the forward return calculations perfectly align with the tick indexing, mathematically guaranteeing zero look-ahead bias in the optimization targets.

---

## Architectural Unit Tests (`test_mamba.py`, `test_transformer.py`, `test_lstm.py`)

Each sequence model requires rigorous unit testing to guarantee topological stability:
* **Dimensional Contracts:** Asserts that passing a batched sequence tensor of shape `(Batch, Sequence, Features)` deterministically reduces to a `(Batch,)` prediction vector without triggering arbitrary PyTorch broadcasting.
* **Gradient Propagation:** Executes isolated forward and backward passes to verify that loss gradients flow completely through the parameter graphs. This is explicitly required to validate the custom unrolled state transitions in the Mamba SSM.
* **Precision Casting:** Verifies that all modules instantiate and execute without precision errors under standard `torch.float32` mappings.

---

## Execution Context

To execute the validation suite from the project root:

```bash
PYTHONPATH=. pytest python/tests/