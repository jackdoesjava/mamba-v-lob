"""Block bootstrap, Newey-West and Diebold-Mariano for overlapping labels.

Consecutive labels share H-1 ticks of their forward window. See docs/09-statistics.md.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def effective_sample_size(n: int, horizon: int) -> float:
    return max(1.0, n / horizon)


def rank_ic(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(stats.spearmanr(preds, targets).statistic)


def block_bootstrap_ic(
    preds: np.ndarray,
    targets: np.ndarray,
    horizon: int = 100,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Circular block bootstrap CI for the rank IC.

    Blocks are one forward window long so resampling keeps the dependence the overlap induces.
    """
    rng = np.random.default_rng(seed)
    n = len(preds)
    block = int(horizon)
    n_blocks = int(math.ceil(n / block))

    point = rank_ic(preds, targets)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        # % n wraps blocks past the end; the slice drops the overshoot of the last block
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        idx = idx[:n]
        boots[b] = stats.spearmanr(preds[idx], targets[idx]).statistic

    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "rank_ic": point,
        "se_block_bootstrap": float(boots.std(ddof=1)),
        "se_naive_iid": float(1.0 / math.sqrt(n)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n": int(n),
        "n_effective": effective_sample_size(n, horizon),
        "significant_at_5pct": bool(lo > 0 or hi < 0),
    }


def newey_west_lrv(d: np.ndarray, lags: int) -> float:
    """Bartlett-weighted long-run variance."""
    d = np.asarray(d, dtype=np.float64)
    d = d - d.mean()
    n = len(d)
    lrv = float(np.dot(d, d) / n)
    # lrv = g0 + 2 * sum_k (1 - k/(lags+1)) * g_k
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        cov = float(np.dot(d[:-k], d[k:]) / n)
        lrv += 2.0 * w * cov
    # Bartlett weights keep this non-negative; a raw autocovariance sum can come out negative.
    # The floor is only a divide-by-zero guard for the DM statistic.
    return max(lrv, 1e-30)


def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    horizon: int = 100,
) -> dict:
    """Diebold-Mariano on squared-error loss, HLN corrected. Negative stat favours model A."""
    y_true = np.asarray(y_true, dtype=np.float64)
    d = (y_true - np.asarray(pred_a, dtype=np.float64)) ** 2 - (
        y_true - np.asarray(pred_b, dtype=np.float64)
    ) ** 2
    n = len(d)
    if n < 3 or np.allclose(d, 0.0):
        return {"stat": 0.0, "p_value": 1.0, "n": int(n), "horizon": int(horizon)}

    lrv = newey_west_lrv(d, lags=horizon - 1)
    stat = float(d.mean() / math.sqrt(lrv / n))

    h = horizon
    # HLN (1997) small-sample factor: (n + 1 - 2h + h(h-1)/n) / n, skipped if it goes negative
    correction = (n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n
    if correction > 0:
        stat *= math.sqrt(correction)

    # t reference rather than normal, also per HLN
    p = float(2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1)))
    return {
        "stat": stat,
        "p_value": p,
        "n": int(n),
        "horizon": int(horizon),
        "favours": "A" if stat < 0 else "B",
    }


def summarise_predictions(
    preds: np.ndarray, targets: np.ndarray, horizon: int = 100, seed: int = 0
) -> dict:
    boot = block_bootstrap_ic(preds, targets, horizon=horizon, seed=seed)
    pearson = float(stats.pearsonr(preds, targets).statistic)
    hit = float((np.sign(preds) == np.sign(targets)).mean())
    return {
        **boot,
        "pearson_ic": pearson,
        "hit_rate": hit,
        "mse": float(np.mean((preds - targets) ** 2)),
        "pred_std": float(np.std(preds)),
    }
