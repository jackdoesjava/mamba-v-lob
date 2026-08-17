# Model Architecture Zoo

This directory (`python/models/`) contains the PyTorch implementations for the three sequential architectures evaluated in this pipeline. The primary focus is the explicitly unrolled Mamba State Space Model (SSM), benchmarked against Transformer and LSTM baselines for expressivity and theoretical stability.

---

## Mamba SSM (Verification-Exposed Implementation)
**Location:** `python/models/mamba.py`

This is a custom PyTorch implementation of the Mamba architecture tailored for high-frequency limit order book dynamics. 

The official Mamba release relies on custom CUDA-fused kernels to execute hardware-aware parallel scans. While computationally optimal for GPU throughput, fused kernels abstract away intermediate state transition dynamics and frequently trigger OpenMP deadlocks in CPU-bound environments. More importantly, the downstream formal verification stage requires topological access to the internal continuous-time parameters.

To satisfy the verification constraints, this implementation explicitly bypasses the parallel scan, opting for a manual, sequential unrolling of the discretized state:

$$
h_t = \bar{A}h_{t-1} + \bar{B}x_t
$$

Sacrificing raw training throughput allows the continuous-time parameter matrices ($A, B, C, \Delta$) to remain explicitly exposed in memory for extraction. The architecture maintains gradient flow during unrolling via standard Pre-Layer Normalization and SiLU (Swish) gating.

---

## Causal Transformer (Expressivity Benchmark)
**Location:** `python/models/transformer.py`

An autoregressive Multi-Head Self-Attention (MHSA) architecture serving as the theoretical upper bound for sequence expressivity. It enforces strict causal masking to prevent temporal look-ahead leakage across the order book events.

Empirically, MHSA networks achieve high Information Coefficients but utilize unbounded, highly non-linear attention matrices. This makes them computationally intractable to formally verify with standard SMT solvers and highly susceptible to drawdowns during aggressive market regime shifts. It serves as a direct foil to the contractive stability properties we hypothesize in the SSM.

---

## LSTM (Recurrent Baseline)
**Location:** `python/models/lstm.py`

A standard Long Short-Term Memory network. It is included to benchmark classical gated recurrent memory against modern routing mechanisms (attention and state-spaces), establishing a baseline for gradient stability and constant-time inference latency.