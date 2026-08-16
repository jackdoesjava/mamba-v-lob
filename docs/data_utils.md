# Data Processing & Utilities

This folder (`python/utils/`) contains the deterministic pipeline for getting raw Nasdaq ITCH data into a format that the neural networks can actually learn from.

---

## Feature Engineering (`features.py`)

Raw limit order book (MBP-10) data is highly non-stationary. If you feed absolute price levels directly into a neural network, it will just memorize the random walk of the price drift instead of learning the actual microstructure dynamics.

To fix this, this script processes the raw feeds into stationary inputs:
* **Price Distances:** Instead of absolute bids and asks, all price levels are calculated as a relative distance from the current mid-price.
* **Volume Scaling:** Deep order book sizes have massive variance and can spike randomly. We apply log-scaling to the volume depth to compress extreme outliers so they don't blow up the network gradients during training.
* **Target Formulation:** Calculates the forward 100-tick log return of the mid-price. I put strict indexing checks in place here to guarantee there is zero look-ahead bias (which is the easiest way to accidentally fake a good backtest).

---

## Dataset Management (`dataset.py`)

This contains the PyTorch `Dataset` and `DataLoader` implementations, optimized to handle high-frequency data without crashing the system.

* **Sequence Windowing:** Implements a sliding window to generate overlapping temporal sequences (e.g., $L=100$) so the recurrent layers (LSTM/Mamba) and attention mechanisms (Transformer) have historical context for every prediction.
* **Memory Efficiency:** ITCH data gets massive very quickly. The loader explicitly casts the batched arrays down to `torch.float32` before yielding them. This keeps the memory mapping stable and stops the system RAM from overflowing when running the training loop locally.