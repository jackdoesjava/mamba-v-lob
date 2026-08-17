# LOB Data Pipeline: MBP-10 Preprocessing & Verification Invariants

Here is the breakdown of how we map raw Databento order book feeds into stationary features for the Mamba SSM, along with the strict invariants required for the formal verification stage.

## 1. Source Data
* **Provider:** Databento (US Equities)
* **Feed:** Nasdaq TotalView-ITCH (XNAS.ITCH)
* **Schema:** Market By Price (MBP-10) — top 10 levels for both bids and asks.
* **Timestamps:** Nanosecond-resolution UTC (Equinix NY4 hardware timestamps).
* **Prices:** Fixed-point integers scaled by $10^9$.

## 2. Feature Engineering & Stationarity
Raw limit order book data is highly non-stationary $I(1)$, which breaks continuous-time state space assumptions. We map the raw feeds into a stationary, relative coordinate space before passing them to the network.

### Time Discretization ($\Delta t$)
Unlike standard transformers, Mamba's continuous-time formulation requires explicit time deltas to parameterize the state transition matrix ($\Delta$). Since tick data is asynchronous:

$$
\Delta t_k = t_k - t_{k-1}
$$

*Note: Because NY4 hardware logs multiple matching engine events at the exact same nanosecond, feeding $\Delta t = 0$ directly into the SSM discretization creates degenerate transition matrices. Instead, we aggregate simultaneous messages at the ingestion layer, taking the final order book snapshot for that nanosecond to ensure time is strictly monotonically increasing ($\Delta t_k > 0$).*

### Spatial Stationarity (Price)
We center the book around the instantaneous mid-price to remove price drift.
1. **Mid-Price:**
   
   $$
   P_{\text{mid}, k} = \frac{P_{\text{ask}, 0, k} + P_{\text{bid}, 0, k}}{2}
   $$

2. **Relative Distances:**
   Instead of absolute prices, the network ingests the distance from the mid-price across all 10 depth levels:
   
   $$
   \forall i \in [0, 9]: \quad D_{\text{ask}, i, k} = P_{\text{ask}, i, k} - P_{\text{mid}, k}
   $$
   
   $$
   \forall i \in [0, 9]: \quad D_{\text{bid}, i, k} = P_{\text{bid}, i, k} - P_{\text{mid}, k}
   $$

### Volume Regularization
LOB sizes at deeper levels are massively skewed and heavy-tailed. To prevent the Mamba gradients from collapsing or exploding, we squash the raw sizes with a log transform:

$$
V_{\text{scaled}, i, k} = \ln(V_{\text{raw}, i, k} + 1)
$$

## 3. Invariants (The Verification Contract)
For Daniel to actually prove the model's stability boundaries in the verification engine, he needs hard geometric guarantees about the input space $\mathcal{X}$. Let $a_i(k)$ and $b_i(k)$ be the ask and bid prices at depth $i$, time $k$.

* **Strictly Positive Spread:** The matching engine guarantees the top-of-book spread is never crossed under normal market conditions:
  
  $$
  \forall k: \quad a_0(k) - b_0(k) > 0
  $$

* **Distance Half-Spaces:** The relative price transformations are strictly bounded to their respective sides of the mid-price:
  
  $$
  \forall i, k: \quad D^a_{i,k} \ge 0 \quad \text{and} \quad D^b_{i,k} \le 0
  $$

* **Optimization Target:** The loss function optimizes for the forward log return over an event-time horizon of $H=100$ ticks:
  
  $$
  y_k = \ln\left(\frac{P_{\text{mid}, k+H}}{P_{\text{mid}, k}}\right)
  $$
