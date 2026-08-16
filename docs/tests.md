# Testing & Validation Suite

This directory (`python/tests/`) contains the unit tests run via `pytest`. The primary goal here is catching silent bugs—like subtle dimension broadcasting errors or data pipeline leakage—before running full training loops.

---

## Feature Invariants (`test_features.py`)

These tests validate the mathematical integrity of the raw limit order book transformations:
* **No Data Corruption:** Asserts that normalization and log-scaling do not introduce `NaN` or `Inf` values across edge cases (such as zero-volume book updates).
* **Order Book Invariants:** Verifies physical market constraints, confirming that the best ask is strictly greater than the best bid ($P_{\text{ask}, 0} > P_{\text{bid}, 0}$) and that price distances from the mid-price maintain the correct sign ($D_{\text{ask}} \ge 0, D_{\text{bid}} \le 0$).
* **Target Alignment:** Confirms that forward return calculations match exact tick indexing and do not leak future information into current observations.

---

## Model Unit Tests (`test_mamba.py`, `test_transformer.py`, `test_lstm.py`)

Each model implementation has a dedicated test file to guarantee architectural stability:
* **Tensor Shape Contracts:** Verifies that passing an input batch of shape `(Batch, Sequence, Features)` consistently yields a 1D prediction tensor of shape `(Batch,)` without triggering unintentional PyTorch broadcasting.
* **Gradient Flow:** Runs a dummy forward and backward pass to ensure that loss gradients propagate through all parameter layers (particularly through the custom unrolled state transitions in Mamba) without vanishing or detaching.
* **Precision & Type Handling:** Verifies that modules execute cleanly when cast to `torch.float32`.

---

## Running the Suite

To run all unit tests from the project root:

```bash
PYTHONPATH=. pytest python/tests/