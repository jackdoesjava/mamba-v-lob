"""How close the cell theorem comes to the worst case, and how the worst case grows with length.

Part A searches the theorem's own hypothesis class for the largest gap |h_{T-1} - h~_{T-1}|
between two runs of one cell under the exact zero-order hold, h_t = a_t h_{t-1} + (1 - a_t) c_t
with a_t = exp(-x_t) and x_t = lam dt_t. The nominal timescale x_t is free in [1e-4, 10], the
perturbed one satisfies |log x~_t - log x_t| <= rho, |c~_t| <= M, |c_t - c~_t| <= eta, and both
runs start from h_{-1} = 0. The gap G is set against the theorem's bound
B = eta + 2 M (e^rho - 1) and against eta + 2 M tanh(rho/2), the value that the construction of
Proposition 4 approaches. The paper proves only that no smaller bound holds;
`cell_bound_common_start` also uses the value as an upper bound, which the paper does not state.

Part B does the same under the paper's perturbation model: dt = map(z) for the bounded map and
for softplus, the nominal pre-activation z_t in a box Z and the perturbed one within zeta of it
in Z. The theorem's bound uses rho = zeta sup_Z |(log dt)'|, and the additive baseline
(eq:additive) uses eta_dt = zeta sup_Z |dt'| with and without the floor dt(inf Z).

Both parts maximise by Adam over a batch of restarts: random starts and structured starts in
which c~ alternates between +M and -M in blocks and the timescale perturbation alternates with
it. Every gap found is checked against every bound it should respect; a gap above a bound stops
the run and writes the sequence, because it would be a counterexample or a bug.

Writes docs/certification/cell_tightness.{json,md} and docs/figures/cell_tightness.{pdf,png}.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.models.mamba import SelectiveSSMBlock  # noqa: E402
from src.utils.repro import provenance, set_seed  # noqa: E402
from src.verification.incremental import (  # noqa: E402
    cell_bound,
    cell_bound_common_start,
    delta_slope,
    log_delta_slope,
)
from src.verification.intervals import Interval  # noqa: E402

DTYPE = torch.float64
OUT_DIR = Path("docs/certification")
FIG_DIR = Path("docs/figures")

LAM, M = 1.0, 1.0
X_LO, X_HI = 1e-4, 10.0
LOGX_LO, LOGX_HI = math.log(X_LO), math.log(X_HI)
RHOS = [0.01, 0.03, 0.1, 0.3, 0.6]
ETAS = [0.0, 0.1]
LENGTHS_A = [10, 30, 100, 300, 1000]
LENGTHS_B = [3, 10, 30, 100, 300, 1000, 3000]
RHO_B = 0.01  # zeta is chosen so that zeta sup_Z |(log dt)'| equals this
DT_MIN, DT_MAX = 1e-3, 1e-1
BOXES = {"bounded": (-8.0, 8.0), "softplus": (-48.0, 2.0)}
BLOCKS = [1, 2, 4, 8, 16]  # lengths of the runs of equal sign in the structured starts
SAT = 3.0  # raw value of a saturated start, tanh(3) = 0.995
TOL = 1e-9  # relative slack of the bound checks, far above float64 rounding of the scan

# the colours and type of scripts/15_local_figures.py
OURS = "#1f77b4"
OTHER = "#ff7f0e"
LOCAL = "#6f42c1"
INK = "#212529"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)
WIDE = (6.5, 2.7)


DEFINITIONS = {
    "G": "largest |h_{T-1} - h~_{T-1}| found, re-run by the plain recursion",
    "B": "eta + 2 M (e^rho - 1), Theorem 3 from a common start",
    "common_start_constant": "eta + 2 M tanh(rho/2), the value of the construction of "
                             "Proposition 4 (cell_bound_common_start)",
    "additive_no_floor": "eq:additive with dt_lo = 0: eta + 2 M lam eta_dt T",
    "additive_floor": "eq:additive with dt_lo = dt(inf Z)",
    "crossing": "real L at which eq:additive reaches the theorem's bound",
    "part_b_model": "z_t in Z, |z~_t - z_t| <= zeta with z~_t in Z, |c_t| <= M, eta = 0; "
                    "rho = zeta sup_Z |(log dt)'|, eta_dt = zeta sup_Z |dt'|",
}


class BoundViolation(RuntimeError):
    """A search found a gap above a bound that should hold; carries the witness."""

    def __init__(self, message: str, witness: dict):
        super().__init__(message)
        self.witness = witness


# --- the cell and the bounds ------------------------------------------------------------------


def final_state(x: torch.Tensor, c: torch.Tensor, h0: float = 0.0) -> torch.Tensor:
    """h_{T-1} of h_t = a_t h_{t-1} + (1 - a_t) c_t with a_t = exp(-x_t), over the last axis.

    Unrolled, h_{T-1} = h0 exp(-sum_r x_r) + sum_k (1 - a_k) c_k exp(-sum_{r > k} x_r). Every
    exponent is a non-positive suffix sum, so nothing overflows, and the whole batch is one pass
    instead of T sequential steps, which is what makes the search affordable at T = 3000.
    """
    tail = torch.flip(torch.cumsum(torch.flip(x, (-1,)), -1), (-1,)) - x
    tail = tail.clamp(min=0.0)  # the suffix sum of the last step is 0 up to rounding
    out = (-torch.expm1(-x) * c * torch.exp(-tail)).sum(-1)
    return out + h0 * torch.exp(-x.sum(-1))


def scan_loop(x, c, h0: float = 0.0) -> np.ndarray:
    """Every state h_0, ..., h_{T-1} of the same cell by the plain recursion, in float64."""
    x, c = np.asarray(x, dtype=np.float64), np.asarray(c, dtype=np.float64)
    out = np.empty(len(x))
    h = float(h0)
    for t in range(len(x)):
        w = -math.expm1(-x[t])
        h = (1.0 - w) * h + w * c[t]
        out[t] = h
    return out


def theorem_bound(eta: float, m: float, rho: float) -> float:
    """eta + 2 M (e^rho - 1), Theorem 3 from a common start (`cell_bound` with e0 = 0)."""
    return float(cell_bound(eta, m, rho))


def common_start_constant(eta: float, m: float, rho: float) -> float:
    """eta + 2 M tanh(rho/2), which Proposition 4's construction approaches."""
    return float(cell_bound_common_start(eta, m, rho))


def additive_baseline(eta: float, m: float, lam: float, eta_dt: float, dt_lo: float,
                      length: float) -> float:
    """eq:additive: eta + 2 M lam e^{-lam dt_lo} eta_dt sum_{k=0}^{L-1} e^{-k lam dt_lo}.

    dt_lo = 0 means no floor, where the sum is L. With a floor the sum is
    (1 - q^L)/(1 - q) with q = e^{-lam dt_lo}, evaluated with expm1 so that a floor of 1e-21
    still gives L and not 0/0. A non-integer L continues the sum, for the crossing length.
    """
    if dt_lo == 0.0:
        total = float(length)
    else:
        total = math.expm1(-lam * dt_lo * length) / math.expm1(-lam * dt_lo)
    return eta + 2.0 * m * lam * math.exp(-lam * dt_lo) * eta_dt * total


def crossing_length(eta: float, m: float, lam: float, eta_dt: float, dt_lo: float,
                    bound: float) -> float | None:
    """The real L at which the additive baseline reaches `bound`, None if it never does.

    Without a floor the baseline is affine in L. With one, solving
    2 M lam q eta_dt (1 - q^L)/(1 - q) = bound - eta gives
    L = -log(1 - r) / (lam dt_lo) with r = (bound - eta)(1 - q) / (2 M lam q eta_dt), and no
    crossing when r >= 1, where the saturated baseline stays below the bound.
    """
    need = bound - eta
    if need <= 0.0:
        return 0.0
    slope = 2.0 * m * lam * math.exp(-lam * dt_lo) * eta_dt
    if dt_lo == 0.0:
        return need / slope
    r = need * -math.expm1(-lam * dt_lo) / slope
    if r >= 1.0:
        return None
    return -math.log1p(-r) / (lam * dt_lo)


# --- the search -------------------------------------------------------------------------------


def ascend(raw0: list[torch.Tensor], gap_fn, steps: int, lr: float, lr_final: float):
    """Adam ascent on the gap of every restart at once.

    gap_fn(raw, snap) returns the gap per restart; with snap=True every variable that enters
    through tanh is replaced by its sign, the vertex of the admissible box that the ascent can
    only approach. Returns the best gap of each restart over the iterates and the snapped best
    iterate, with the raw parameters and whether the snapped version won.
    """
    raw = [r.detach().clone().requires_grad_(True) for r in raw0]
    opt = torch.optim.Adam(raw, lr=lr)
    gamma = (lr_final / lr) ** (1.0 / max(steps - 1, 1))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    best = torch.full((raw0[0].shape[0],), -math.inf, dtype=DTYPE)
    best_raw = [r.detach().clone() for r in raw]
    for step in range(steps + 1):
        g = gap_fn(raw, snap=False)
        with torch.no_grad():
            better = g > best
            best = torch.where(better, g.detach(), best)
            for b, r in zip(best_raw, raw):
                b[better] = r.detach()[better]
        if step == steps:
            break
        opt.zero_grad()
        (-g.sum()).backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        snapped = gap_fn(best_raw, snap=True)
    use_snap = snapped > best
    return torch.where(use_snap, snapped, best), best_raw, use_snap


def _logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(p) - torch.log1p(-p)


def _unit(raw: torch.Tensor, snap: bool) -> torch.Tensor:
    return torch.sign(raw) if snap else torch.tanh(raw)


# Part A: the hypothesis class of the theorem


def runs_a(raw, rho: float, eta: float, snap: bool, m: float = M):
    """(x, x~, c, c~) from the raw parameters (P, Q, R, W) of a batch of restarts."""
    P, Q, R, W = raw
    logx = LOGX_LO + (LOGX_HI - LOGX_LO) * torch.sigmoid(P)
    x = torch.exp(logx)
    xt = torch.exp(logx + rho * _unit(Q, snap))
    ct = m * _unit(R, snap)
    c = ct + eta * _unit(W, snap)
    return x, xt, c, ct


def gap_a(raw, rho: float, eta: float, snap: bool = False) -> torch.Tensor:
    x, xt, c, ct = runs_a(raw, rho, eta, snap)
    return (final_state(x, c) - final_state(xt, ct)).abs()


def starts_a(T: int, rho: float, eta: float, n_random: int, gen: torch.Generator,
             n_grid: int = 11):
    """Random starts, then alternating ones.

    In the alternating starts c~ takes the sign of a square wave with runs of length k, the
    perturbed step is the faster one on the + runs and the slower one on the - runs, and the
    nominal log-timescale either stays at a base value ("constant") or sits rho/2 below it on
    + runs and rho/2 above it on - runs ("balanced"), the configuration whose fine-step limit
    is the common-start constant. With eta > 0, c - c~ is +eta, -eta or follows the wave.
    """
    t = torch.arange(T, dtype=DTYPE)
    P, Q, R, W, kinds = [], [], [], [], []
    rnd = [torch.randn(n_random, T, dtype=DTYPE, generator=gen) * 2.0 for _ in range(4)]
    P.append(rnd[0]), Q.append(rnd[1]), R.append(rnd[2]), W.append(rnd[3])
    kinds += ["random"] * n_random
    w_modes = ("plus", "minus", "wave") if eta > 0 else ("plus",)
    for k in [b for b in BLOCKS if b < T]:
        wave = 1.0 - 2.0 * (torch.div(t, k, rounding_mode="floor") % 2)
        for logx0 in np.linspace(LOGX_LO, LOGX_HI, n_grid):
            for mode in ("constant", "balanced"):
                logx = logx0 - (rho / 2.0) * wave if mode == "balanced" else torch.full_like(t, logx0)
                p = _logit((logx - LOGX_LO) / (LOGX_HI - LOGX_LO))
                for wm in w_modes:
                    w = {"plus": torch.ones_like(t), "minus": -torch.ones_like(t), "wave": wave}[wm]
                    P.append(p[None]), Q.append(SAT * wave[None]), R.append(SAT * wave[None])
                    W.append(SAT * w[None])
                    kinds.append(f"k={k}, x0={math.exp(logx0):.3g}, {mode}, w={wm}")
    return [torch.cat(v) for v in (P, Q, R, W)], kinds


def search_a(T: int, rho: float, eta: float, steps: int, n_random: int, gen: torch.Generator,
             lr: float = 0.1, lr_final: float = 0.002) -> dict:
    """The largest gap over the hypothesis class at length T, verified by the plain loop."""
    raw0, kinds = starts_a(T, rho, eta, n_random, gen)
    best, best_raw, use_snap = ascend(raw0, lambda r, snap: gap_a(r, rho, eta, snap),
                                      steps, lr, lr_final)
    i = int(torch.argmax(best))
    with torch.no_grad():
        x, xt, c, ct = (v[0].numpy() for v in
                        runs_a([r[i:i + 1] for r in best_raw], rho, eta, bool(use_snap[i])))
    gaps = np.abs(scan_loop(x, c) - scan_loop(xt, ct))
    G = float(gaps[-1])
    B = theorem_bound(eta, M, rho)
    witness = {"x": x.tolist(), "x_tilde": xt.tolist(), "c": c.tolist(), "c_tilde": ct.tolist()}
    admissible = (np.all(x >= X_LO * (1 - TOL)) and np.all(x <= X_HI * (1 + TOL))
                  and np.all(np.abs(np.log(xt) - np.log(x)) <= rho * (1 + TOL))
                  and np.all(np.abs(ct) <= M * (1 + TOL))
                  and np.all(np.abs(c - ct) <= eta * (1 + TOL) + 1e-15))
    if not admissible:
        raise BoundViolation(f"part A witness not admissible (rho={rho}, eta={eta}, T={T})", witness)
    if gaps.max() > B * (1 + TOL):
        raise BoundViolation(
            f"part A: gap {gaps.max():.17g} above B = {B:.17g} (rho={rho}, eta={eta}, T={T})",
            witness)
    plus = ct > 0
    out = {
        "rho": rho, "eta": eta, "T": T,
        "G": G, "G_max_over_t": float(gaps.max()), "B": B, "G_over_B": G / B,
        "common_start_constant": common_start_constant(eta, M, rho),
        "G_over_common_start_constant": G / common_start_constant(eta, M, rho),
        "scan_vs_loop_abs_diff": abs(float(best[i]) - G),
        "restarts": len(kinds), "best_start": kinds[i], "best_snapped": bool(use_snap[i]),
        "best_random": float(best[: n_random].max()),
        "best_structured": float(best[n_random:].max()) if len(kinds) > n_random else None,
        "witness_summary": {
            "x_median": float(np.median(x)), "x_min": float(x.min()), "x_max": float(x.max()),
            "x_median_on_plus": float(np.median(x[plus])) if plus.any() else None,
            "x_median_on_minus": float(np.median(x[~plus])) if (~plus).any() else None,
            "mean_abs_log_ratio_over_rho": float(np.mean(np.abs(np.log(xt / x))) / rho),
            "mean_abs_c_tilde": float(np.mean(np.abs(ct))),
            "sign_changes_of_c_tilde": int(np.sum(np.sign(ct[1:]) != np.sign(ct[:-1]))),
        },
    }
    if T <= 30:
        out["witness"] = witness
    return out


# Part B: the paper's perturbation model


def make_map(name: str) -> SelectiveSSMBlock:
    """A block used only for its timescale map, so the map is the model's own."""
    return SelectiveSSMBlock(d_model=4, d_state=1, dt_min=DT_MIN, dt_max=DT_MAX,
                             dt_parametrisation=name)


def model_constants(name: str, rho_target: float = RHO_B, lam: float = LAM, m: float = M) -> dict:
    """zeta, rho, eta_dt and the floor of one map on its box, from the repository's slopes.

    sup_Z |(log dt)'| and sup_Z |dt'| come from `log_delta_slope` and `delta_slope`, the
    functions the data-free certificate uses, so Part B compares the bounds the paper computes.
    """
    lo, hi = BOXES[name]
    block = make_map(name)
    box = Interval(torch.tensor([lo], dtype=DTYPE), torch.tensor([hi], dtype=DTYPE))
    log_slope = float(log_delta_slope(block, box)[0])
    dt_slope = float(delta_slope(block, box)[0])
    zeta = rho_target / log_slope
    with torch.no_grad():
        dt_lo = float(block.delta_from_pre(torch.tensor(lo, dtype=DTYPE)))
        dt_hi = float(block.delta_from_pre(torch.tensor(hi, dtype=DTYPE)))
    rho = zeta * log_slope
    eta_dt = zeta * dt_slope
    theorem = theorem_bound(0.0, m, rho)
    return {
        "map": name, "Z": [lo, hi], "zeta": zeta, "sup_abs_log_dt_slope": log_slope,
        "sup_abs_dt_slope": dt_slope, "rho": rho, "eta_dt": eta_dt,
        "dt_floor": dt_lo, "dt_ceiling": dt_hi, "theorem_bound": theorem,
        "common_start_constant": common_start_constant(0.0, m, rho),
        "additive_floor_limit": (2.0 * m * lam * eta_dt / math.expm1(lam * dt_lo)),
        "crossing_no_floor": crossing_length(0.0, m, lam, eta_dt, 0.0, theorem),
        "crossing_floor": crossing_length(0.0, m, lam, eta_dt, dt_lo, theorem),
    }


def runs_b(raw, block: SelectiveSSMBlock, Z, zeta: float, snap: bool, lam: float = LAM,
           m: float = M):
    """(x, x~, c, c~) with z in Z, z~ within zeta of z in Z, and eta = 0."""
    P, Q, R = raw
    lo, hi = Z
    z = lo + (hi - lo) * torch.sigmoid(P)
    zt = torch.clamp(z + zeta * _unit(Q, snap), lo, hi)
    x = lam * block.delta_from_pre(z)
    xt = lam * block.delta_from_pre(zt)
    c = m * _unit(R, snap)
    return x, xt, c, c, z, zt


def gap_b(raw, block, Z, zeta: float, snap: bool = False) -> torch.Tensor:
    x, xt, c, ct, _, _ = runs_b(raw, block, Z, zeta, snap)
    return (final_state(x, c) - final_state(xt, ct)).abs()


def starts_b(T: int, Z, zeta: float, grid, n_random: int, gen: torch.Generator):
    """Random starts and the alternating starts of Part A, built in z rather than in log x."""
    lo, hi = Z
    t = torch.arange(T, dtype=DTYPE)
    P, Q, R, kinds = [], [], [], []
    rnd = [torch.randn(n_random, T, dtype=DTYPE, generator=gen) * 2.0 for _ in range(3)]
    P.append(rnd[0]), Q.append(rnd[1]), R.append(rnd[2])
    kinds += ["random"] * n_random
    for k in [b for b in BLOCKS if b < T]:
        wave = 1.0 - 2.0 * (torch.div(t, k, rounding_mode="floor") % 2)
        for z0 in grid:
            for mode in ("constant", "balanced"):
                z = z0 - (zeta / 2.0) * wave if mode == "balanced" else torch.full_like(t, z0)
                P.append(_logit((z - lo) / (hi - lo))[None])
                Q.append(SAT * wave[None]), R.append(SAT * wave[None])
                kinds.append(f"k={k}, z0={z0:.3g}, {mode}")
    return [torch.cat(v) for v in (P, Q, R)], kinds


def search_b(name: str, consts: dict, T: int, steps: int, n_random: int, gen: torch.Generator,
             lr: float = 0.1, lr_final: float = 0.002) -> dict:
    block = make_map(name)
    Z, zeta = tuple(consts["Z"]), consts["zeta"]
    lo, hi = Z
    grid = np.linspace(-6.0, 6.0, 13) if name == "bounded" else np.linspace(-10.0, hi, 13)
    raw0, kinds = starts_b(T, Z, zeta, grid, n_random, gen)
    best, best_raw, use_snap = ascend(raw0, lambda r, snap: gap_b(r, block, Z, zeta, snap),
                                      steps, lr, lr_final)
    i = int(torch.argmax(best))
    with torch.no_grad():
        x, xt, c, ct, z, zt = (v[0].numpy() for v in runs_b([r[i:i + 1] for r in best_raw],
                                                             block, Z, zeta, bool(use_snap[i])))
    gaps = np.abs(scan_loop(x, c) - scan_loop(xt, ct))
    G = float(gaps[-1])
    theorem = consts["theorem_bound"]
    no_floor = additive_baseline(0.0, M, LAM, consts["eta_dt"], 0.0, T)
    floor = additive_baseline(0.0, M, LAM, consts["eta_dt"], consts["dt_floor"], T)
    witness = {"z": z.tolist(), "z_tilde": zt.tolist(), "c": c.tolist()}
    # every state h_t is checked against the bounds at its own length L = t + 1
    tight = np.array([min(theorem,
                          additive_baseline(0.0, M, LAM, consts["eta_dt"], 0.0, t + 1),
                          additive_baseline(0.0, M, LAM, consts["eta_dt"], consts["dt_floor"], t + 1))
                      for t in range(T)])
    ratio = gaps / tight
    if ratio.max() > 1.0 + TOL:
        t_bad = int(np.argmax(ratio))
        raise BoundViolation(
            f"part B ({name}): gap {gaps[t_bad]:.17g} above min(theorem, additive) = "
            f"{tight[t_bad]:.17g} at t={t_bad} (T={T})", witness)
    if np.any(np.abs(zt - z) > zeta * (1 + TOL)) or np.any(z < lo) or np.any(z > hi):
        raise BoundViolation(f"part B ({name}) witness not admissible at T={T}", witness)
    plus = c > 0
    return {
        "T": T, "G": G, "G_max_over_t": float(gaps.max()), "theorem_bound": theorem,
        "additive_no_floor": no_floor, "additive_floor": floor, "G_over_theorem": G / theorem,
        "max_over_t_gap_over_min_bound": float(ratio.max()),
        "G_over_common_start_constant": G / consts["common_start_constant"],
        "scan_vs_loop_abs_diff": abs(float(best[i]) - G),
        "restarts": len(kinds), "best_start": kinds[i], "best_snapped": bool(use_snap[i]),
        "witness_summary": {
            "z_median": float(np.median(z)),
            "dt_median": float(np.median(x / LAM)),
            "z_median_on_plus": float(np.median(z[plus])) if plus.any() else None,
            "z_median_on_minus": float(np.median(z[~plus])) if (~plus).any() else None,
            "mean_abs_z_shift_over_zeta": float(np.mean(np.abs(zt - z)) / zeta),
            "sign_changes_of_c": int(np.sum(np.sign(c[1:]) != np.sign(c[:-1]))),
        },
    }


# --- outputs ----------------------------------------------------------------------------------


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isinf(v):
            return "inf"
        return f"{v:.4g}" if (abs(v) >= 1e4 or (v and abs(v) < 1e-3)) else f"{v:.4f}"
    return str(v)


def table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def markdown(results: dict) -> str:
    s = results["settings"]
    a_rows = results["part_a"]["rows"]
    lengths_a = s["lengths_a"]
    md = [
        "# Tightness of the cell theorem",
        "",
        "Written by `scripts/16_cell_tightness.py`. Everything is in float64 with "
        f"`lam = {s['lam']}`, `M = {s['M']}` and both runs started from `h_{{-1}} = 0`. Each gap is "
        "the largest `|h_{T-1} - h~_{T-1}|` that Adam ascent found over a batch of restarts "
        f"({s['steps']} steps, learning rate {s['lr']} decaying to {s['lr_final']}): "
        f"{s['n_random']} random starts and alternating starts in which `c~` follows a square "
        "wave with runs of length 1 to 16 and the perturbed timescale is the faster step on the "
        "`+` runs and the slower on the `-` runs, at a grid of base timescales. The best iterate "
        "of every restart is also evaluated with every variable that enters through `tanh` set "
        "to its sign. The winning sequence is re-run with the plain recursion, and that value is "
        "the one reported. Part A searches the hypothesis class of Theorem 3 with the nominal "
        f"timescale `x_t = lam dt_t` free in `[{s['x_range'][0]:g}, {s['x_range'][1]:g}]`. "
        "Part B searches the perturbation model of the paper, `dt = map(z)` with `z` in a box `Z` "
        "and `|z~ - z| <= zeta` in `Z`, with `eta = 0`. `B = eta + 2 M (e^rho - 1)` is the "
        "theorem's bound and `eta + 2 M tanh(rho/2)` is the value that the construction of "
        "Proposition 4 approaches, below which the paper proves no bound holds; "
        "`cell_bound_common_start` also uses it as an upper bound, which the paper does not state. "
        "A search is a lower bound on the worst case, not the worst case itself.",
        "",
        f"No gap exceeded a bound it should respect: largest `G/B` in Part A "
        f"`{results['checks']['part_a_max_G_over_B']:.4f}`, largest gap `|h_t - h~_t|` over the "
        "smallest valid bound at its own length `t + 1` in Part B "
        f"`{results['checks']['part_b_max_G_over_min_bound']:.4f}`. "
        f"Run time {results['run_seconds']:.0f} s.",
        "",
        "## Part A: `G/B` on the theorem's hypothesis class",
        "",
    ]
    rows = []
    for rho in s["rhos"]:
        for eta in s["etas"]:
            r = {row["T"]: row for row in a_rows if row["rho"] == rho and row["eta"] == eta}
            ref = r[lengths_a[-1]]["common_start_constant"] / r[lengths_a[-1]]["B"]
            rows.append([rho, eta, *[r[T]["G_over_B"] for T in lengths_a], ref])
    md += [table(rows, ["rho", "eta", *[f"T = {T}" for T in lengths_a],
                        "common-start constant / B"]), ""]
    md += [f"At the longest length, `T = {lengths_a[-1]}`:", ""]
    rows = []
    for row in a_rows:
        if row["T"] != lengths_a[-1]:
            continue
        w = row["witness_summary"]
        rows.append([row["rho"], row["eta"], row["G"], row["B"], row["common_start_constant"],
                     row["G_over_common_start_constant"], w["x_median"],
                     w["sign_changes_of_c_tilde"], row["best_start"]])
    md += [table(rows, ["rho", "eta", "G", "B", "common-start constant", "G / constant",
                        "median x", "sign changes of c~", "best start"]), ""]
    md += ["## Part B: the length picture under the perturbation model", ""]
    for name, part in results["part_b"].items():
        k = part["constants"]
        md += [
            f"### {name} map, `Z = [{k['Z'][0]:g}, {k['Z'][1]:g}]`",
            "",
            f"`zeta = {k['zeta']:.6g}`, `sup_Z |(log dt)'| = {k['sup_abs_log_dt_slope']:.6g}`, "
            f"`sup_Z |dt'| = {k['sup_abs_dt_slope']:.6g}`, so `rho = {k['rho']:.6g}` and "
            f"`eta_dt = {k['eta_dt']:.6g}`. Floor `dt(inf Z) = {k['dt_floor']:.6g}`. Theorem 3: "
            f"`{k['theorem_bound']:.6g}` at every length; common-start constant "
            f"`{k['common_start_constant']:.6g}`; floored additive baseline as `L -> inf`: "
            f"`{k['additive_floor_limit']:.6g}`. The additive baseline reaches the theorem's bound "
            f"at `L = {fmt(k['crossing_no_floor'])}` without the floor and at "
            f"`L = {fmt(k['crossing_floor'])}` with it.",
            "",
        ]
        rows = [[r["T"], r["G"], r["theorem_bound"], r["additive_no_floor"], r["additive_floor"],
                 r["G_over_theorem"], r["witness_summary"]["dt_median"], r["best_start"]]
                for r in part["rows"]]
        md += [table(rows, ["T", "G", "theorem", "additive, no floor", "additive, floor",
                            "G / theorem", "median dt", "best start"]), ""]
    md += [f"Figure: `{results['figure']}`.", ""]
    return "\n".join(md)


def save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def figure(results: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=WIDE, layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, (name, part) in zip(axes[:2], results["part_b"].items()):
        k = part["constants"]
        T = np.array([r["T"] for r in part["rows"]], dtype=float)
        Tf = np.geomspace(T[0], T[-1], 200)
        # eta = 0 in Part B, so the trivial bound 2M + eta of two runs from a common start is 2M
        ax.loglog(Tf, np.full_like(Tf, 2.0 * M), color="0.6", linewidth=1.0, linestyle=":",
                  label=r"trivial bound $2M$")
        ax.loglog(Tf, np.full_like(Tf, k["theorem_bound"]), color=OURS, linewidth=2.0,
                  label="Theorem 3")
        ax.loglog(Tf, [additive_baseline(0.0, M, LAM, k["eta_dt"], 0.0, L) for L in Tf],
                  color=OTHER, linewidth=1.4, linestyle="--", label="additive, no floor")
        ax.loglog(Tf, [additive_baseline(0.0, M, LAM, k["eta_dt"], k["dt_floor"], L) for L in Tf],
                  color=OTHER, linewidth=1.0, label="additive, floor")
        ax.loglog(T, [r["G"] for r in part["rows"]], color=INK, linestyle="none", marker="o",
                  markersize=4.5, label="largest gap found")
        ax.set_title(f"{name} map", fontsize=9, loc="left", color=INK)
        ax.set_xlabel("sequence length $T$")
        ax.set_xlim(T[0] / 1.5, T[-1] * 1.5)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel(r"$|h_{T-1} - \tilde h_{T-1}|$")
    ax = axes[2]
    s = results["settings"]
    T_max = s["lengths_a"][-1]
    rho_f = np.geomspace(min(s["rhos"]), max(s["rhos"]), 100)
    for eta, marker, fill in zip(s["etas"], ("o", "s"), (INK, "white")):
        ax.semilogx(rho_f, [common_start_constant(eta, M, r) / theorem_bound(eta, M, r)
                            for r in rho_f], color=LOCAL, linewidth=2.4, alpha=0.45,
                    solid_capstyle="butt",
                    label=r"$(\eta + 2M\tanh(\rho/2))/E$" if eta == s["etas"][0] else None)
        rows = sorted((r for r in results["part_a"]["rows"]
                       if r["eta"] == eta and r["T"] == T_max), key=lambda r: r["rho"])
        ax.semilogx([r["rho"] for r in rows], [r["G_over_B"] for r in rows], color=INK,
                    linestyle="none", marker=marker, markersize=4.5, markerfacecolor=fill,
                    label=rf"largest gap, $\eta = {eta:g}$")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(rf"largest gap $/\, E$ at $T = {T_max}$")
    ax.set_title("hypothesis class", fontsize=9, loc="left", color=INK)
    ax.legend(loc="lower left", frameon=False, fontsize=6.5, handlelength=1.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--random", type=int, default=64, help="random restarts per search")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quick", action="store_true", help="short lengths and few steps")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--fig-dir", default=str(FIG_DIR))
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    gen = torch.Generator().manual_seed(args.seed)
    lengths_a = [10, 30] if args.quick else LENGTHS_A
    lengths_b = [3, 10, 30] if args.quick else LENGTHS_B
    steps = 100 if args.quick else args.steps
    out_dir, fig_dir = Path(args.out_dir), Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "lam": LAM, "M": M, "x_range": [X_LO, X_HI], "rhos": RHOS, "etas": ETAS,
        "lengths_a": lengths_a, "lengths_b": lengths_b, "rho_b": RHO_B,
        "dt_min": DT_MIN, "dt_max": DT_MAX, "boxes": BOXES, "blocks": BLOCKS,
        "steps": steps, "lr": 0.1, "lr_final": 0.002, "n_random": args.random,
        "h_init": 0.0, "dtype": "float64", "tolerance": TOL,
    }
    start = time.perf_counter()
    try:
        part_a = []
        for rho in RHOS:
            for eta in ETAS:
                for T in lengths_a:
                    row = search_a(T, rho, eta, steps, args.random, gen)
                    part_a.append(row)
                    print(f"A rho={rho:<5g} eta={eta:<4g} T={T:<5d} G={row['G']:.6g} "
                          f"B={row['B']:.6g} G/B={row['G_over_B']:.4f} "
                          f"[{time.perf_counter() - start:.0f} s]", flush=True)
        part_b = {}
        for name in BOXES:
            consts = model_constants(name)
            rows = []
            for T in lengths_b:
                row = search_b(name, consts, T, steps, args.random, gen)
                rows.append(row)
                print(f"B {name:<8s} T={T:<5d} G={row['G']:.6g} theorem={row['theorem_bound']:.6g} "
                      f"additive={row['additive_no_floor']:.6g}/{row['additive_floor']:.6g} "
                      f"[{time.perf_counter() - start:.0f} s]", flush=True)
            part_b[name] = {"constants": consts, "rows": rows}
    except BoundViolation as err:
        path = out_dir / "cell_tightness_violation.json"
        path.write_text(json.dumps({"provenance": provenance(args.seed), "settings": settings,
                                    "error": str(err), "witness": err.witness}, indent=2))
        print(f"STOPPED: {err}\nwitness written to {path}")
        raise SystemExit(1) from err
    run_seconds = time.perf_counter() - start

    fig_path = fig_dir / "cell_tightness.pdf"
    results = {
        "provenance": provenance(args.seed),
        "settings": settings,
        "definitions": DEFINITIONS,
        "part_a": {"rows": part_a},
        "part_b": part_b,
        "checks": {
            "any_violation": False,
            "part_a_max_G_over_B": max(r["G_max_over_t"] / r["B"] for r in part_a),
            "part_b_max_G_over_min_bound": max(
                r["max_over_t_gap_over_min_bound"] for p in part_b.values() for r in p["rows"]),
            "max_scan_vs_loop_abs_diff": max(
                [r["scan_vs_loop_abs_diff"] for r in part_a]
                + [r["scan_vs_loop_abs_diff"] for p in part_b.values() for r in p["rows"]]),
        },
        "run_seconds": run_seconds,
        "figure": str(fig_path),
    }
    figure(results, fig_path)
    md = markdown(results)
    (out_dir / "cell_tightness.json").write_text(json.dumps(results, indent=2))
    (out_dir / "cell_tightness.md").write_text(md)
    print("\n" + md)
    print(f"wrote {out_dir / 'cell_tightness.json'}, {out_dir / 'cell_tightness.md'} and {fig_path}")


if __name__ == "__main__":
    main()
