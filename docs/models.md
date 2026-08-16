# Model Architecture Zoo

This directory contains the PyTorch implementations for the three sequential models evaluated in the project. The primary focus is exploring the Mamba State Space Model, with a standard Transformer and LSTM serving as performance and stability baselines.

---

## Mamba (Pure PyTorch Implementation)
**Location:** `python/models/mamba.py`

This is a pure PyTorch implementation of the Mamba State Space Model (SSM) adapted for limit order book time-series. 

Standard Mamba relies on custom CUDA-fused kernels for hardware-aware parallel scans. While highly efficient, those kernels obscure the internal state matrices and can cause thread deadlocks in standard CPU environments. Because the downstream formal verification stage requires proving properties about the model's stability boundaries, we need explicit access to the underlying parameters. 

To achieve this, this implementation drops the fused kernels and manually unrolls the sequential discretization:

$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$

While this sacrifices the training speed of the official implementation, keeping the continuous-time matrices ($A, B, C, \Delta$) explicitly exposed in memory allows us to extract them for the external verification engine. To maintain gradient flow during this sequential unrolling, the block uses standard Pre-Layer Normalization and SiLU (Swish) activations.

---

## Causal Transformer (Expressivity Baseline)
**Location:** `python/models/transformer.py`

A standard Multi-Head Self-Attention architecture acting as our expressivity baseline. It uses strict causal masking to prevent look-ahead bias into future LOB states.

While Transformers generally capture strong correlations (high Information Coefficient), their attention matrices are unbounded and highly non-linear. This makes them notoriously difficult to formally verify and prone to instability during market regime shifts, serving as a foil to the contractive stability we are testing in the SSM.

---

## LSTM (Sanity Check)
**Location:** `python/models/lstm.py`

A standard Long Short-Term Memory network. Included purely as a sanity check to benchmark classical recurrent memory against the modern attention and state-space architectures.