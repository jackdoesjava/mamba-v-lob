# Data Pipeline Specification: MBP-10 to Stationary State Space Inputs

## 1. Source Data Specifications
* **Provider:** Databento US Equities
* **Feed:** Nasdaq TotalView-ITCH (XNAS.ITCH)
* **Schema:** Market By Price (MBP-10), top 10 bids and asks reconstructed.
* **Timestamping:** Nanosecond-resolution UTC, hardware-timestamped at Equinix NY4.
* **Price Encoding:** Fixed-point integer representation scaled by factor $10^9$.

## 2. Deterministic Transformations & Normalization

To satisfy the assumptions of continuous-time State Space Models (SSMs) and allow formal verification via SMT solvers, raw inputs are transformed into a stationary coordinate space.

### Temporal Discretization ($\Delta t$)
Tick data arrives asynchronously. The model requires explicit time deltas to parameterize the state transition matrix discretization step ($\Delta$).
$$\Delta t_k = t_k - t_{k-1}$$
* Bounded domain: $\Delta t_k \ge 0$.
* Missing values: Initial sequence step enforces $\Delta t_0 = 0.0$.

### Spatial Stationarity (Price Dimension)
Raw asset prices are non-stationary $I(1)$ processes. Absolute prices are mapped to relative coordinate distances from the instantaneous mid-price ($P_{\text{mid}}$).

1. **Mid-Price Definition:**
   $$P_{\text{mid}, k} = \frac{P_{\text{ask}, 0, k} + P_{\text{bid}, 0, k}}{2}$$
2. **Distance Vector Transformation:**
   $$\forall i \in [0, 9]: \quad D_{\text{ask}, i, k} = P_{\text{ask}, i, k} - P_{\text{mid}, k}$$
   $$\forall i \in [0, 9]: \quad D_{\text{bid}, i, k} = P_{\text{bid}, i, k} - P_{\text{mid}, k}$$

### Volumetric Regularization (Size Dimension)
Order book sizes at deep levels exhibit heavy-tailed distributions. A natural logarithmic transformation stabilizes variance for the neural network gradients while preserving order-matching invariants.
$$V_{\text{scaled}, i, k} = \ln(V_{\text{raw}, i, k} + 1.0)$$

## 3. Downstream Invariants for Verification

Let $a_i(k)$ and $b_i(k)$ represent the asset price at order book depth level $i$ (where $i=0$ is top-of-book) at tick time $k$.

* **Spread Invariant:** The bid-ask spread must remain strictly positive under nominal matching engine conditions:
  $$\forall k: \quad a_0(k) - b_0(k) > 0$$

* **Distance Bounds:** Spatial distance vectors must maintain their respective geometric half-spaces:
  $$\forall i, k: \quad D^a_{i,k} \ge 0 \quad \text{and} \quad D^b_{i,k} \le 0$$

* **Target Mapping:** The optimization target is the forward log return over an event horizon of $H=100$ ticks:
  $$y_k = \ln\left(\frac{P_{\text{mid}, k+H}}{P_{\text{mid}, k}}\right)$$
