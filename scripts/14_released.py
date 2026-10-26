"""Certify and attack the released language model state-spaces/mamba-130m-hf on public-domain text.

Four measurements, per layer of the released checkpoint, through src/verification/released.py:

  data-free  the certificate of every layer with respect to its RMSNorm output: the
             exponential-Euler invariant radius M = G sup|B u| / lam with G = Psi(lam dt_hi), the
             separated geometric bound, the timescale box, the incremental output slope with the
             log-timescale rule (Corollary cor:block with Corollary cor:euler), and the additive
             baseline with and without the certified floor.
  realised   every intermediate quantity of each layer on the text, against its certified box.
  local      the norm-propagation domain around the realised normalised input of a layer, with the
             additive recurrence step and the steps of Corollary cor:local adapted to the Euler
             equilibrium (running maximum, recursion, and the endpoint form, the reported one),
             at several perturbation sizes and prefix lengths of the text.
  attack     sign-gradient projected ascent on the largest change of a layer's mixer output over
             positions and coordinates, for perturbations of its realised normalised input within
             delta in sup-norm, as scripts/13_incremental_attack.py runs it on the trained model.

An attacked value is reached, so attacked <= true <= certified; an attacked value above a
certificate would falsify the certificate. Everything runs in float64 on the map without the
float32 casts of transformers' slow path; released.py documents both.

Writes docs/certification/released.{json,md} and docs/figures/released_layers.{pdf,png}.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.repro import REPO_ROOT, provenance, set_seed
from src.verification.released import (
    LOCAL_RULES,
    REPORTED_RULE,
    MAX_TOKENS,
    MODEL_ID,
    SAMPLE_TEXT,
    ReleasedLayer,
    certify_layer,
    incremental_certificate,
    load_released,
    load_tokenizer,
    local_mixer_sensitivity,
    mixer_trace,
    model_dir,
    read_sample_text,
    realised_layer,
    reference_forward,
    released_layers,
    rms_norm,
    tokenise,
)

OUT_DIR = Path("docs/certification")
FIG_DIR = Path("docs/figures")
ATTACK_LAYERS = [0, 6, 12, 18, 20, 23]   # 20 has the largest data-free slope
ATTACK_EVERY_LAYER_UP_TO = 256            # every layer is attacked at the lengths up to this
LENGTHS = [64, 256, 1024]
DELTAS = [1e-6, 1e-4, 1e-2]
DTYPE = torch.float64
CHUNK = 128
SEED = 1337
RULE_NAMES = {"additive": "additive", "log_delta": "log-timescale, running maximum",
              "log_delta_recursive": "log-timescale, recursion",
              "log_delta_endpoint": "log-timescale, endpoint"}
SHORT = {"additive": "additive", "log_delta": "maximum", "log_delta_recursive": "recursion",
         "log_delta_endpoint": "endpoint"}
LEGEND = {"additive": "local, additive", "log_delta": r"local, log-$\Delta$, max",
          "log_delta_recursive": r"local, log-$\Delta$, recursion",
          "log_delta_endpoint": r"local, log-$\Delta$, endpoint"}

# the colours and type of scripts/05_figures.py and scripts/13_incremental_attack.py
OURS = "#1f77b4"
OTHER = "#ff7f0e"
LOCAL = "#6f42c1"
INK = "#212529"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)
FIGSIZE = (3.4, 5.4)   # one column


# --- a differentiable float64 mixer with the Euler scan's adjoint ----------------------------------


def _scan(delta, B, C, u, A, D, keep_states: bool):
    """y_t = C_t . h_t + D u_t with h_t = exp(dt_t A) h_{t-1} + dt_t B_t u_t, from h_{-1} = 0.

    The exponential-Euler scan of modeling_mamba.py, lines 249-268, run a chunk of time steps at
    once, time-major. Returns y as (batch, L, d) and, if keep_states, every h_t as (L, batch, d, n).
    """
    dt = delta.transpose(0, 1).contiguous()
    Bt = B.transpose(0, 1).contiguous()
    Ct = C.transpose(0, 1).contiguous()
    ut = u.transpose(0, 1).contiguous()
    L, b, d = ut.shape
    n = A.shape[1]
    hs = ut.new_empty(L, b, d, n) if keep_states else None
    y = ut.new_empty(L, b, d)
    h = ut.new_zeros(b, d, n)
    for t0 in range(0, L, CHUNK):
        t1 = min(L, t0 + CHUNK)
        a = torch.exp(dt[t0:t1].unsqueeze(-1) * A)
        bu = (dt[t0:t1] * ut[t0:t1]).unsqueeze(-1) * Bt[t0:t1].unsqueeze(2)
        buf = hs[t0:t1] if keep_states else torch.empty_like(bu)
        for i in range(t1 - t0):
            h = torch.addcmul(bu[i], a[i], h, out=buf[i])
        y[t0:t1] = (buf * Ct[t0:t1].unsqueeze(2)).sum(-1)
    return (y + D * ut).transpose(0, 1), hs


class _Scan(torch.autograd.Function):
    """The Euler scan with a hand-written adjoint that stores h_t once. With a_t = exp(dt_t A),
    d h_t / d dt_t = A a_t h_{t-1} + B_t u_t. tests/test_released.py checks it with gradcheck."""

    @staticmethod
    def forward(ctx, delta, B, C, u, A, D):
        y, hs = _scan(delta, B, C, u, A, D, keep_states=True)
        ctx.save_for_backward(delta, B, C, u, A, D, hs)
        return y

    @staticmethod
    def backward(ctx, gy):
        delta, B, C, u, A, D, hs = ctx.saved_tensors
        dt = delta.transpose(0, 1).contiguous()
        Bt = B.transpose(0, 1).contiguous()
        Ct = C.transpose(0, 1).contiguous()
        ut = u.transpose(0, 1).contiguous()
        gyt = gy.transpose(0, 1).contiguous()
        L, b, d = ut.shape
        n = A.shape[1]
        g_dt, g_B = torch.empty_like(dt), torch.empty_like(Bt)
        g_C, g_u = torch.empty_like(Ct), torch.empty_like(ut)
        carry = hs.new_zeros(b, d, n)               # a_{t+1} times the adjoint of h_{t+1}
        for t0 in reversed(range(0, L, CHUNK)):
            t1 = min(L, t0 + CHUNK)
            a = torch.exp(dt[t0:t1].unsqueeze(-1) * A)
            gh = gyt[t0:t1].unsqueeze(-1) * Ct[t0:t1].unsqueeze(2)
            for i in range(t1 - t0 - 1, -1, -1):
                torch.add(gh[i], carry, out=gh[i])
                carry = gh[i] * a[i]
            if t0 > 0:
                h_prev = hs[t0 - 1 : t1 - 1]
            else:
                h_prev = torch.cat([hs.new_zeros(1, b, d, n), hs[: t1 - 1]], 0)
            Bc, uc = Bt[t0:t1], ut[t0:t1]
            g_dt[t0:t1] = (gh * (A * a * h_prev + Bc.unsqueeze(2) * uc.unsqueeze(-1))).sum(-1)
            ghd = gh * dt[t0:t1].unsqueeze(-1)
            g_B[t0:t1] = (ghd * uc.unsqueeze(-1)).sum(2)
            g_u[t0:t1] = (ghd * Bc.unsqueeze(2)).sum(-1) + gyt[t0:t1] * D
            g_C[t0:t1] = (hs[t0:t1] * gyt[t0:t1].unsqueeze(-1)).sum(2)
        return (g_dt.transpose(0, 1), g_B.transpose(0, 1), g_C.transpose(0, 1),
                g_u.transpose(0, 1), None, None)


def mixer_contribution(layer: ReleasedLayer, v: torch.Tensor) -> torch.Tensor:
    """The mixer output out_proj(y * SiLU(gate)) from the normalised input v, differentiable in v.

    Equal to released.mixer_trace(layer, v).out to rounding; that function, not this one, scores
    every reported number.
    """
    L = v.shape[1]
    x, gate = (v @ layer.W_in.t()).chunk(2, dim=-1)
    conv = F.conv1d(x.transpose(1, 2), layer.conv.weight, layer.conv.bias,
                    padding=layer.d_conv - 1, groups=layer.d_inner)[..., :L].transpose(1, 2)
    u = F.silu(conv)
    W_p, W_B, W_C = layer.split_x()
    B, C = u @ W_B.t(), u @ W_C.t()
    delta = F.softplus((u @ W_p.t()) @ layer.W_dt.t() + layer.b_dt)
    A = -layer.lam
    if torch.is_grad_enabled() and v.requires_grad:
        y = _Scan.apply(delta, B, C, u, A, layer.D)
    else:
        y, _ = _scan(delta, B, C, u, A, layer.D, keep_states=False)
    return (y * F.silu(gate)) @ layer.W_out.t()


def _signs(shape, gen: torch.Generator) -> torch.Tensor:
    return (torch.randint(0, 2, shape, generator=gen) * 2 - 1).to(DTYPE)


# --- the attack, as scripts/13_incremental_attack.py runs it --------------------------------------


@dataclass
class Budget:
    """What one attack on one (layer, length) runs. Recorded in the JSON."""

    probes: int           # random sign vectors at the smallest delta, to rank the targets
    candidates: int       # targets over all positions whose exact gradient row is taken
    last_candidates: int  # targets at the last position whose exact gradient row is taken
    steps: int            # sign-gradient steps per restart
    restarts: int         # restarts per delta: best rows' signs, random corners, last position
    batch: int            # replicas per forward pass


def budget_for(seq_len: int) -> Budget:
    """Four restarts per delta at every length: the two best gradient rows' signs, one random
    corner, and the best row at the last position. Only the batch shrinks with the length, to
    bound the memory of the stored states (about 200 MB per replica at L = 1024)."""
    return Budget(probes=32, candidates=24 if seq_len <= 256 else 32, last_candidates=8,
                  steps=12, restarts=4, batch=16 if seq_len <= 256 else 8)


def _step_size(radius: torch.Tensor, step: int, steps: int) -> torch.Tensor:
    # from 2 delta, which flips a coordinate across the box, down to delta/8
    return radius * 2.0 * 0.5 ** (4.0 * step / max(steps, 1))


def _pgd(objective, xi: torch.Tensor, radius: torch.Tensor, steps: int, batch: int):
    """Sign-gradient ascent on objective(xi) -> (elements,), projected onto |xi| <= radius.
    Returns the best point visited per element, its value, and the value at the start."""
    best_xi, best, start = [], [], []
    for i0 in range(0, xi.shape[0], batch):
        x = xi[i0 : i0 + batch].clone()
        r = radius[i0 : i0 + batch]
        top = torch.full((x.shape[0],), -math.inf, dtype=DTYPE)
        keep = x.clone()
        for s in range(steps + 1):
            x.requires_grad_(True)
            val = objective(x, slice(i0, i0 + x.shape[0]))
            v = val.detach()
            if s == 0:
                start.append(v.clone())
            better = v > top
            top = torch.where(better, v, top)
            keep[better] = x.detach()[better]
            if s == steps:
                break
            (g,) = torch.autograd.grad(val.sum(), x)
            with torch.no_grad():
                x = x.detach() + _step_size(r, s, steps) * torch.sign(g)
                x = torch.maximum(torch.minimum(x, r), -r)
        best_xi.append(keep)
        best.append(top)
    return torch.cat(best_xi), torch.cat(best), torch.cat(start)


def attack_layer(layer: ReleasedLayer, v: torch.Tensor, deltas: list[float], budget: Budget,
                 seed: int) -> dict:
    """The largest |F(v + xi) - F(v)| found over |xi|_inf <= delta, for one sequence v (1, L, d).

    1. budget.probes random sign vectors at the smallest delta rank every (position, coordinate)
       by the root mean square of its change.
    2. The exact gradient row of the top budget.candidates targets, and of the top
       budget.last_candidates at the last position, gives each its first-order optimum
       delta |row|_1, at xi = delta sign(row).
    3. Per delta, sign-gradient ascent on the largest change over all positions and coordinates,
       from the two best rows' signs and from a random corner, and on the largest change at the
       last position, from that position's best row.
    4. The best point of every restart is scored again with released.mixer_trace. Those are the
       reported numbers, and `map` holds the largest change per (position, coordinate).
    """
    gen = torch.Generator().manual_seed(seed)
    L, d_in = v.shape[1], v.shape[2]
    d_out = layer.W_out.shape[0]
    with torch.no_grad():
        base = mixer_contribution(layer, v)

    d0 = min(deltas)
    sq = torch.zeros(L * d_out, dtype=DTYPE)
    random_corner = 0.0
    for p0 in range(0, budget.probes, budget.batch):
        p = min(budget.batch, budget.probes - p0)
        with torch.no_grad():
            ch = mixer_contribution(layer, v + d0 * _signs((p, L, d_in), gen)) - base
        sq += (ch * ch).sum(0).flatten()
        random_corner = max(random_corner, float(ch.abs().max()) / d0)

    top = torch.topk(sq, min(budget.candidates, sq.numel())).indices
    last_scores = sq.view(L, d_out)[-1]
    last = torch.topk(last_scores, min(budget.last_candidates, d_out)).indices + (L - 1) * d_out
    targets = torch.unique(torch.cat([top, last]))
    rows = []
    for i0 in range(0, len(targets), budget.batch):
        tg = targets[i0 : i0 + budget.batch]
        xi = torch.zeros(len(tg), L, d_in, dtype=DTYPE, requires_grad=True)
        out = mixer_contribution(layer, v + xi).flatten(1)
        (g,) = torch.autograd.grad(out.gather(1, tg.unsqueeze(1)).sum(), xi)
        rows.append(g)
    rows = torch.cat(rows)
    l1 = rows.abs().flatten(1).sum(1)
    order = torch.argsort(l1, descending=True)
    at_last = targets // d_out == L - 1
    best_last = order[at_last[order]][0]

    starts, radius, last_flags = [], [], []
    for d in deltas:
        for j in order[:2]:
            starts.append(d * torch.sign(rows[j]))
            last_flags.append(False)
        starts.append(d * _signs((L, d_in), gen))
        last_flags.append(False)
        starts.append(d * torch.sign(rows[best_last]))
        last_flags.append(True)
        radius += [d] * budget.restarts
    xi0 = torch.stack(starts)
    r = torch.tensor(radius, dtype=DTYPE).view(-1, 1, 1)
    flags = torch.tensor(last_flags)

    def objective(x, sl):
        ch = (mixer_contribution(layer, v + x) - base).abs()
        return torch.where(flags[sl], ch[:, -1].amax(-1), ch.flatten(1).amax(1))

    best_xi, best_val, start_val = _pgd(objective, xi0, r, budget.steps, budget.batch)

    with torch.no_grad():
        ref_base = mixer_trace(layer, v).out
        dev = torch.cat([(mixer_trace(layer, v + best_xi[i0 : i0 + budget.batch]).out - ref_base).abs()
                         for i0 in range(0, best_xi.shape[0], budget.batch)])

    per_delta = []
    k = budget.restarts
    for i, d in enumerate(deltas):
        sl = slice(k * i, k * i + k)
        m = dev[sl].amax(0)                                   # (L, d_out)
        flat = int(m.flatten().argmax())
        per_delta.append({
            "delta": d,
            "map": m,
            "max": float(m.max()),
            "last": float(m[-1].max()),
            "argmax": [flat // d_out, flat % d_out],
            "pgd_gain": float(best_val[sl].max() / start_val[sl].max()),
        })
    return {
        "per_delta": per_delta,
        "first_order": float(l1.max()),
        "first_order_last": float(l1[at_last].max()),
        "first_order_target": [int(targets[order[0]]) // d_out, int(targets[order[0]]) % d_out],
        "random_corner": random_corner,
    }


# --- the data-free certificate and the realised quantities ----------------------------------------


def _stats(t: torch.Tensor) -> dict:
    """min, median and max over the finite entries, and the count of the others."""
    t = t.flatten()
    ok = torch.isfinite(t)
    f = t[ok]
    if f.numel() == 0:
        return {"min": math.nan, "median": math.nan, "max": math.nan, "non_finite": int((~ok).sum())}
    return {"min": float(f.min()), "median": float(f.median()), "max": float(f.max()),
            "non_finite": int((~ok).sum())}


@torch.no_grad()
def data_free_layer(layer: ReleasedLayer, x: torch.Tensor,
                    lengths: list[int]) -> tuple[dict, object, torch.Tensor]:
    """The data-free certificate of one layer and the realised quantities on its input x. Returns
    the record, the incremental certificate and the width of the certified output box.

    The additive baseline with the certified floor has a limit in the length of
    2 M lam exp(-lam dt_lo) L_delta / (1 - exp(-lam dt_lo)) per cell, which is infinite in float64
    wherever the floor underflows. Its base-10 logarithm is recorded per cell from
    log 2 M L_delta - log dt_lo, which is the limit to within a factor 1 + O(lam dt_lo) when
    lam dt_lo < 1e-8.
    """
    cert = certify_layer(layer)
    coord = certify_layer(layer, fused=False)
    inc = incremental_certificate(layer, cert)
    real = realised_layer(layer, x, cert)
    b = cert.boxes
    ln10 = math.log(10.0)
    lam_lo = cert.lam * cert.delta_lo.unsqueeze(1)
    tiny = lam_lo < 1e-8
    log_limit = (torch.log(2.0 * cert.M * inc.L_delta.unsqueeze(1)) - cert.log_delta_lo.unsqueeze(1)) / ln10
    slope = inc.output_slope("cell")
    parts = inc.output_slope_parts("cell")
    row = {
        "layer": layer.index,
        "lam": _stats(cert.lam),
        "u_sup_fused": float(b["u"].abs_max().max()),
        "u_sup_coordinate": float(coord.boxes["u"].abs_max().max()),
        "M_max_fused": float(cert.M.max()),
        "M_max_coordinate": float(coord.M.max()),
        "z_lo_min": float(b["z"].lo.min()),
        "z_hi_max": float(b["z"].hi.max()),
        "delta_hi": _stats(cert.delta_hi),
        "log10_delta_lo": _stats(cert.log_delta_lo / ln10),
        "floor_underflow_channels": int((cert.delta_lo == 0).sum()),
        "channels": int(cert.delta_lo.numel()),
        "sup_Bu": _stats(cert.sup_Bu),
        "G": _stats(cert.G),
        "M": _stats(cert.M),
        "log10_geometric": _stats(cert.log10_geometric),
        "log10_gap": _stats(cert.log10_gap),
        "geometric_non_finite_cells": int((~torch.isfinite(cert.geometric)).sum()),
        "cells": int(cert.M.numel()),
        "out_box_sup": float(b["out"].abs_max().max()),
        "ell_min": float((inc.L_log_delta / inc.L_z).min()),
        "L_z": _stats(inc.L_z),
        "L_u": _stats(inc.L_u),
        "out_slope_cell_max": float(slope.max()),
        "out_slope_common_start_max": float(inc.output_slope("common_start").max()),
        "warp_share_cell": float((parts["warp"] / slope).max()),
        "forcing_share_cell_max": float((parts["forcing"] / slope).max()),
        "additive_no_floor_max": {str(L): float(inc.additive_output_slope(L, floor=False).max())
                                  for L in [1, *lengths]},
        "additive_certified_floor_max": {str(L): float(inc.additive_output_slope(L, floor=True).max())
                                         for L in lengths},
        "additive_certified_floor_limit_max": float(inc.additive_output_slope(None, floor=True).max()),
        "log10_additive_floor_limit_state_max": float(log_limit[tiny].max()) if tiny.any() else None,
        "realised": real,
    }
    return row, inc, b["out"].width()


def _over(rows: list[dict], fn) -> dict:
    v = torch.tensor([fn(r) for r in rows], dtype=DTYPE)
    return _stats(v)


def data_free_summary(rows: list[dict], lengths: list[int]) -> dict:
    Lmax = str(max(lengths))
    return {
        "u_sup_fused": _over(rows, lambda r: r["u_sup_fused"]),
        "u_sup_coordinate": _over(rows, lambda r: r["u_sup_coordinate"]),
        "coordinate_over_fused_M": _over(rows, lambda r: r["M_max_coordinate"] / r["M_max_fused"]),
        "z_lo_min": _over(rows, lambda r: r["z_lo_min"]),
        "z_hi_max": _over(rows, lambda r: r["z_hi_max"]),
        "realised_z_min": _over(rows, lambda r: r["realised"]["z"]["realised_min"]),
        "realised_z_max": _over(rows, lambda r: r["realised"]["z"]["realised_max"]),
        "delta_hi_max": _over(rows, lambda r: r["delta_hi"]["max"]),
        "realised_delta_max": _over(rows, lambda r: r["realised"]["delta"]["realised_max"]),
        "realised_delta_min": _over(rows, lambda r: r["realised"]["delta"]["realised_min"]),
        "log10_delta_lo_min": _over(rows, lambda r: r["log10_delta_lo"]["min"]),
        "floor_underflow_channels": _over(rows, lambda r: r["floor_underflow_channels"]),
        "lam_max": _over(rows, lambda r: r["lam"]["max"]),
        "G_max": _over(rows, lambda r: r["G"]["max"]),
        "G_median": _over(rows, lambda r: r["G"]["median"]),
        "realised_G_max": _over(rows, lambda r: r["realised"]["G"]["realised_max"]),
        "M_max": _over(rows, lambda r: r["M"]["max"]),
        "realised_h_max": _over(rows, lambda r: r["realised"]["h"]["realised_max"]),
        "h_certified_over_realised_median": _over(
            rows, lambda r: r["realised"]["h"]["certified_over_realised"]["median"]),
        "h_certified_over_realised_min": _over(
            rows, lambda r: r["realised"]["h"]["certified_over_realised"]["min"]),
        "log10_geometric_max": _over(rows, lambda r: r["log10_geometric"]["max"]),
        "log10_gap_median": _over(rows, lambda r: r["log10_gap"]["median"]),
        "out_slope_cell_max": _over(rows, lambda r: r["out_slope_cell_max"]),
        "out_slope_common_start_max": _over(rows, lambda r: r["out_slope_common_start_max"]),
        "warp_share_cell": _over(rows, lambda r: r["warp_share_cell"]),
        "additive_no_floor_L1_over_ours": _over(
            rows, lambda r: r["additive_no_floor_max"]["1"] / r["out_slope_cell_max"]),
        f"additive_no_floor_L{Lmax}_over_ours": _over(
            rows, lambda r: r["additive_no_floor_max"][Lmax] / r["out_slope_cell_max"]),
        "log10_additive_floor_limit_state_max": _over(
            rows, lambda r: r["log10_additive_floor_limit_state_max"]
            if r["log10_additive_floor_limit_state_max"] is not None else math.nan),
        "ell_min": _over(rows, lambda r: r["ell_min"]),
        "all_contained": all(all(r["realised"]["contained"].values()) for r in rows),
        "hardest_layer_by_slope": max(rows, key=lambda r: r["out_slope_cell_max"])["layer"],
        "easiest_layer_by_slope": min(rows, key=lambda r: r["out_slope_cell_max"])["layer"],
    }


# --- the local certificate and the attack per (layer, length) -------------------------------------


def _max(t: torch.Tensor) -> float:
    if torch.isnan(t).any():
        return math.nan
    return float(t.max())


@torch.no_grad()
def local_layer(layer: ReleasedLayer, v: torch.Tensor, deltas: list[float]) -> dict:
    """local_mixer_sensitivity with every rule at every delta, keeping the per-(position,
    coordinate) sensitivity for the violation check."""
    out = {}
    for d in deltas:
        out[f"{d:g}"] = {}
        for rule in LOCAL_RULES:
            t0 = time.time()
            r = local_mixer_sensitivity(layer, v, d, rule)
            out["out_abs_max"] = float(r["nominal"].abs().max())
            lip = r["lipschitz"][0]
            out[f"{d:g}"][rule] = {
                "lip": lip,
                "lip_max": _max(lip),
                "lip_last_max": _max(lip[-1]),
                "finite": bool(torch.isfinite(lip).all()),
                "diagnostics": r["diagnostics"],
                "seconds": time.time() - t0,
            }
    return out


def _ratio(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)) or b == 0:
        return math.inf if (math.isinf(a) and b > 0) else math.nan
    return a / b


def experiment(layer: ReleasedLayer, inc, out_width: torch.Tensor, v: torch.Tensor,
               deltas: list[float], attack: bool, seed: int) -> dict:
    """The local certificate at one (layer, length), and, if attack, the attack priced against it
    and against the data-free bound in the reading arbitrary=True, since the attacked point need
    not be an RMSNorm output.

    A local bound lip_{t,k} delta at least the width out_width_k of the certified output box says
    nothing the box does not; the fraction of (position, coordinate) pairs where that happens, or
    where the bound is infinite, is recorded as vacuous_fraction.
    """
    t0 = time.time()
    local = local_layer(layer, v, deltas)
    t1 = time.time()
    att = attack_layer(layer, v, deltas, budget_for(v.shape[1]), seed) if attack else None
    t2 = time.time()
    per_delta = []
    for i, d in enumerate(deltas):
        key = f"{d:g}"
        df = inc.output_bound(d, arbitrary=True)
        row = {
            "delta": d,
            "data_free_bound_max": float(df.max()),
            "data_free_rmsnorm_bound_max": float(inc.output_bound(d).max()),
            "local": {},
        }
        a = att["per_delta"][i] if att is not None else None
        if a is not None:
            m = a["map"]
            row.update({
                "attacked_max": a["max"],
                "slope_max": a["max"] / d,
                "last_slope_max": a["last"] / d,
                "worst_position": a["argmax"][0],
                "worst_position_from_end": m.shape[0] - 1 - a["argmax"][0],
                "worst_coordinate": a["argmax"][1],
                "pgd_gain": a["pgd_gain"],
                "data_free_violations": int((m.amax(0) > df * (1 + 1e-9)).sum()),
            })
        for rule in LOCAL_RULES:
            loc = local[key][rule]
            entry = {k: loc[k] for k in ("lip_max", "lip_last_max", "finite", "diagnostics", "seconds")}
            lip = loc["lip"]
            entry["vacuous_fraction"] = float(((lip * d >= out_width) | ~torch.isfinite(lip))
                                              .to(DTYPE).mean())
            entry["over_realised_out"] = loc["lip_max"] * d / local["out_abs_max"]
            if a is not None:
                ok = torch.isfinite(lip)
                entry["violations"] = int(((a["map"] > lip * d * (1 + 1e-9) + 1e-14) & ok).sum())
                entry["over_attack"] = _ratio(loc["lip_max"], row["slope_max"])
            row["local"][rule] = entry
        per_delta.append(row)
    res = {"per_delta": per_delta, "out_abs_max": local["out_abs_max"],
           "local_seconds": t1 - t0, "attack_seconds": t2 - t1, "attacked": att is not None}
    if att is not None:
        res.update({k: att[k] for k in ("first_order", "first_order_last", "first_order_target",
                                        "random_corner")})
    return res


# --- the figure -------------------------------------------------------------------------------------


def _plot(v) -> float:
    return v if (v is not None and math.isfinite(v) and v > 0) else np.nan


def figure(results: dict, path: Path) -> None:
    """Per layer: the data-free slopes, the local bounds and the attacked slope at the longest
    length, at the smallest delta (top) and at delta = 1e-4 (bottom). Values per unit delta."""
    rows = results["data_free"]["layers"]
    layers = [r["layer"] for r in rows]
    Lmax = str(max(results["run"]["lengths"]))
    deltas = results["run"]["deltas"]
    panels = [min(deltas)] + ([1e-4] if 1e-4 in deltas else [])
    fig, axes = plt.subplots(len(panels), 1, figsize=FIGSIZE, sharex=True, layout="constrained")
    axes = np.atleast_1d(axes)
    exp = {(e["layer"], str(e["seq_len"])): e for e in results["experiments"]}
    df_add = [r["additive_no_floor_max"][Lmax] for r in rows]
    top = 30.0 * max(df_add)
    styles = {"additive": dict(color=OTHER, marker="s", markersize=3, linewidth=0.9),
              "log_delta": dict(color=LOCAL, marker="D", markersize=3, linewidth=0.9,
                                linestyle=":", markerfacecolor="white"),
              "log_delta_recursive": dict(color=LOCAL, marker="D", markersize=3, linewidth=0.9,
                                          linestyle="--", markerfacecolor="white"),
              "log_delta_endpoint": dict(color=LOCAL, marker="D", markersize=3, linewidth=1.1)}
    for ax, d in zip(axes, panels):
        j = deltas.index(d)
        ax.semilogy(layers, df_add, color=OTHER, linewidth=1.4, linestyle="--",
                    label=f"data-free, additive, $L = {Lmax}$")
        ax.semilogy(layers, [r["out_slope_cell_max"] for r in rows], color=OURS, linewidth=1.6,
                    label=r"data-free, log-$\Delta$")
        loc_layers = [l for l in layers if (l, Lmax) in exp]
        off = []
        for rule in LOCAL_RULES:
            vals = [_plot(exp[(l, Lmax)]["per_delta"][j]["local"][rule]["lip_max"]) for l in loc_layers]
            shown = [v if v < top else np.nan for v in vals]
            ax.semilogy(loc_layers, shown, label=LEGEND[rule], **styles[rule])
            above = [l for l, v in zip(loc_layers, vals) if not (v < top)]
            if above:
                off.append((rule, above))
        att_layers = [l for l in loc_layers if exp[(l, Lmax)]["attacked"]]
        ax.semilogy(att_layers, [exp[(l, Lmax)]["per_delta"][j]["slope_max"] for l in att_layers],
                    color=INK, linestyle="none", marker="o", markersize=4.5,
                    label=f"attack, $L = {Lmax}$")
        # the longest shorter length at which more layers were attacked
        shorter = [L for L, ls in results["run"]["attack_layers_by_length"].items()
                   if L != Lmax and len(ls) > len(att_layers)]
        lows = [exp[(l, Lmax)]["per_delta"][j]["slope_max"] for l in att_layers]
        if shorter:
            Ls = max(shorter, key=int)
            ls = results["run"]["attack_layers_by_length"][Ls]
            vals = [exp[(l, Ls)]["per_delta"][j]["slope_max"] for l in ls]
            lows += vals
            ax.semilogy(ls, vals, color=INK, linestyle="none", marker="o", markersize=3.5,
                        markerfacecolor="white", markeredgewidth=0.8, label=f"attack, $L = {Ls}$")
        for rule, above in off:
            # a bound above the axis, or infinite, is marked at the top edge
            ax.plot(above, [top] * len(above), linestyle="none", marker="^", markersize=4.5,
                    color=styles[rule]["color"], clip_on=False,
                    label=f"{LEGEND[rule]}, off scale")
        ax.set_ylim(min(lows) / 5.0 if lows else None, top)
        ax.set_ylabel(r"$\|F(v) - F(\tilde v)\|_\infty / \delta$")
        ax.set_title(rf"$\delta = 10^{{{int(round(math.log10(d)))}}}$, local bounds at $L = {Lmax}$",
                     fontsize=8, loc="left", color=INK)
    axes[-1].set_xlabel("layer")
    axes[-1].set_xticks(list(range(0, len(layers), 4)) + [len(layers) - 1])
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=6.5,
               handlelength=1.8, columnspacing=1.0)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


# --- the markdown twin ----------------------------------------------------------------------------


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf"
        if v == 0.0:
            return "0"
        return f"{v:.3g}" if (abs(v) >= 1e4 or abs(v) < 1e-2) else f"{v:.4g}"
    return str(v)


def table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def _span(values: list[float]) -> str:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return "not finite"
    lo, hi = fmt(min(finite)), fmt(max(finite))
    return lo if lo == hi else f"{lo} to {hi}"


def _layers(ls: list[int]) -> str:
    """A list of layer indices, with a full range written as one."""
    if len(ls) > 2 and ls == list(range(ls[0], ls[-1] + 1)):
        return f"{ls[0]} to {ls[-1]}"
    return ", ".join(str(l) for l in ls)


def _mmm(s: dict) -> str:
    return f"{fmt(s['min'])}, {fmt(s['median'])} and {fmt(s['max'])}"


def _attacked(results: dict) -> list[dict]:
    return [e for e in results["experiments"] if e["attacked"]]


def _violations(results: dict) -> tuple[int, int]:
    df = local = 0
    for e in _attacked(results):
        for pd in e["per_delta"]:
            df += pd["data_free_violations"]
            local += sum(pd["local"][rule]["violations"] for rule in LOCAL_RULES)
    return df, local


def markdown(results: dict) -> str:
    run, dfree = results["run"], results["data_free"]
    rows, s = dfree["layers"], dfree["summary"]
    lengths, deltas = run["lengths"], run["deltas"]
    Lmax = max(lengths)
    d0 = min(deltas)
    ag = results["agreement"]
    text = results["text"]
    budget = results["budget"]
    exps = results["experiments"]
    att = _attacked(results)
    md = [
        "# The released model mamba-130m-hf",
        "",
        f"Every number here is for `{results['model']['id']}`, {results['model']['layers']} layers "
        f"with `d_model = {results['model']['d_model']}`, `d_inner = {results['model']['d_inner']}` "
        f"and `d_state = {results['model']['d_state']}`, loaded from a local copy with transformers "
        f"{results['model']['transformers_version']}. The text is `{text['file']}`, "
        f"{text['tokens']} tokens. The certified map is the float64 forward pass of "
        "`src/verification/released.py` without the float32 casts of transformers' slow path. With "
        "those casts reproduced the reference forward pass differs from transformers' float64 "
        f"output by at most {fmt(ag['faithful_max_abs_diff'])} on {ag['tokens']} tokens, and "
        f"without them by at most {fmt(ag['pure_max_abs_diff'])}, against a largest output of "
        f"{fmt(ag['final_abs_max'])}.",
        "",
        "## The data-free certificate",
        "",
        "Each layer is certified with respect to its RMSNorm output, for every input of the layer. "
        "The scan of the released model is the exponential-Euler step, so the invariant radius is "
        "`M = G sup|B u| / lam` with `G = Psi(lam dt_hi)`, which needs the ceiling `dt_hi` of the "
        "timescale and no floor (Corollary cor:euler). The separated geometric bound "
        "`dt_hi sup|B u| / (1 - exp(-lam dt_lo))` needs the floor `dt_lo`, and is given as a "
        "base-10 logarithm because the floor underflows in float64.",
        "",
    ]
    md += [table([[r["layer"], r["z_lo_min"], r["z_hi_max"], r["realised"]["z"]["realised_min"],
                   r["realised"]["z"]["realised_max"], r["delta_hi"]["max"],
                   r["realised"]["delta"]["realised_max"], r["log10_delta_lo"]["min"],
                   r["floor_underflow_channels"], r["G"]["max"], r["realised"]["G"]["realised_max"],
                   r["M"]["max"], r["realised"]["h"]["realised_max"],
                   r["realised"]["h"]["certified_over_realised"]["median"], r["log10_geometric"]["max"],
                   r["log10_gap"]["median"]] for r in rows],
                 ["layer", "z box, low", "z box, high", "realised z, min", "realised z, max",
                  "dt_hi, max", "realised dt, max", "log10 dt_lo, min", "channels with dt_lo = 0",
                  "G, max", "realised Psi(lam dt), max", "M, max", "realised abs h, max",
                  "M / realised abs h, median cell", "log10 geometric bound, max",
                  "log10 gap, median cell"]), ""]
    md += [
        f"Over the {len(rows)} layers the certified pre-activation box of the timescale has its low "
        f"end between {fmt(s['z_lo_min']['min'])} and {fmt(s['z_lo_min']['max'])} and its high end "
        f"between {fmt(s['z_hi_max']['min'])} and {fmt(s['z_hi_max']['max'])}, while the realised "
        f"pre-activations on the text lie between {fmt(s['realised_z_min']['min'])} and "
        f"{fmt(s['realised_z_max']['max'])}. The certified ceiling `dt_hi` reaches "
        f"{_span([s['delta_hi_max']['min'], s['delta_hi_max']['max']])} per layer, against a "
        f"realised largest timescale of {_span([s['realised_delta_max']['min'], s['realised_delta_max']['max']])}. "
        f"With `lam` up to {_span([s['lam_max']['min'], s['lam_max']['max']])} per layer, "
        f"`G = Psi(lam dt_hi)` reaches {_span([s['G_max']['min'], s['G_max']['max']])}, against a "
        f"realised largest `Psi(lam dt)` of "
        f"{_span([s['realised_G_max']['min'], s['realised_G_max']['max']])}. The invariant radius at "
        "the median cell is "
        f"{_span([s['h_certified_over_realised_median']['min'], s['h_certified_over_realised_median']['max']])} "
        "times the largest realised state of that cell, and at the closest cell "
        f"{_span([s['h_certified_over_realised_min']['min'], s['h_certified_over_realised_min']['max']])} times.",
        "",
        f"The certified floor is between `10^{fmt(s['log10_delta_lo_min']['min'])}` and "
        f"`10^{fmt(s['log10_delta_lo_min']['max'])}` at its smallest channel per layer, and it "
        f"underflows to 0 in {_span([s['floor_underflow_channels']['min'], s['floor_underflow_channels']['max']])} "
        f"of {rows[0]['channels']} channels per layer. The separated geometric bound has a base-10 "
        f"logarithm of up to {_span([s['log10_geometric_max']['min'], s['log10_geometric_max']['max']])} "
        "per layer, and its ratio to the invariant radius at the median cell has a base-10 "
        f"logarithm of {_mmm(s['log10_gap_median'])} (minimum, median and maximum over the layers).",
        "",
        "The input projection is bounded through the fused form `||W_k * w||_2 sqrt(d)`. The "
        "coordinate box `|w_i| sqrt(d)` followed by interval propagation gives a largest invariant "
        f"radius {_mmm(s['coordinate_over_fused_M'])} times larger (minimum, median and maximum over "
        f"the layers). Every realised quantity on the {text['tokens']} tokens lies in its certified "
        f"box at every layer: {fmt(s['all_contained'])}. Realised against certified, per layer:",
        "",
        table([[r["layer"], r["realised"]["u"]["realised_max"], r["realised"]["u"]["certified_max"],
                r["realised"]["Bu"]["realised_max"], r["realised"]["Bu"]["certified_max"],
                r["realised"]["out"]["realised_max"], r["realised"]["out"]["certified_max"],
                all(r["realised"]["contained"].values())] for r in rows],
              ["layer", "abs u, realised", "abs u, certified", "abs Bu, realised", "abs Bu, certified",
               "abs out, realised", "abs out, certified", "all inside"]),
        "",
        "## The data-free incremental certificate",
        "",
        "The slope is the sup-norm Lipschitz constant of the mixer output with respect to the "
        "normalised input that the certificate gives, as the largest over output coordinates, in "
        "the reading where both inputs are RMSNorm outputs. The additive baseline replaces the "
        "relative timescale bound by `|a - a~| <= lam exp(-lam dt_min) |dt - dt~|`, and is taken at "
        "the length shown; with the certified floor its limit in the length is given per cell as a "
        "base-10 logarithm of the state slope, since it is infinite in float64.",
        "",
        table([[r["layer"], r["out_slope_cell_max"], r["out_slope_common_start_max"], r["warp_share_cell"],
                r["additive_no_floor_max"]["1"], *[r["additive_no_floor_max"][str(L)] for L in lengths],
                r["additive_certified_floor_max"][str(Lmax)], r["additive_certified_floor_limit_max"],
                r["log10_additive_floor_limit_state_max"]] for r in rows],
              ["layer", "log-timescale", "log-timescale, common start", "share of the timescale terms",
               "additive, no floor, L = 1", *[f"additive, no floor, L = {L}" for L in lengths],
               f"additive, certified floor, L = {Lmax}", "additive, certified floor, limit",
               "log10 of the floored limit, state slope"]),
        "",
        f"The log-timescale slope is {_mmm(s['out_slope_cell_max'])} over the layers (minimum, "
        f"median and maximum), largest at layer {s['hardest_layer_by_slope']} and smallest at layer "
        f"{s['easiest_layer_by_slope']}. The terms through the timescale, the time-warp term and the "
        f"change of `Psi`, make up {_span([s['warp_share_cell']['min'], s['warp_share_cell']['max']])} "
        "of it at its largest coordinate. The additive slope without a floor is "
        f"{_span([s['additive_no_floor_L1_over_ours']['min'], s['additive_no_floor_L1_over_ours']['max']])} "
        "times larger already at L = 1, because `lam` reaches "
        f"{fmt(s['lam_max']['max'])}, and {_span([s[f'additive_no_floor_L{Lmax}_over_ours']['min'], s[f'additive_no_floor_L{Lmax}_over_ours']['max']])} "
        f"times larger at L = {Lmax}. With the certified floor the additive slope at these lengths "
        "equals the one without, since `lam dt_lo` is below `1/L` by hundreds of orders of magnitude, "
        "and its limit in the length has a base-10 logarithm of "
        f"{_span([s['log10_additive_floor_limit_state_max']['min'], s['log10_additive_floor_limit_state_max']['max']])} "
        "at the largest cell per layer. The factor `ell = sigmoid(z_lo) / softplus(z_lo)` of Lemma "
        f"lem:param is {fmt(s['ell_min']['min'])} at its smallest channel over all layers, so on "
        "these boxes the lemma gives no gain over the constant 1.",
        "",
        "## The local certificate and the attack",
        "",
        "The perturbation is any point within `delta` in sup-norm of a layer's realised normalised "
        "input on the first L tokens of the text. The local certificate is the norm-propagation "
        "domain around that input, with the additive recurrence step and with the two steps of "
        "Corollary cor:local adapted to the Euler equilibrium `c = Psi(lam dt) B u / lam`: the "
        "running maximum `E_t = max(E_{t-1}, K_t)` and the recursion `E_t = K_t + a*_t (E_{t-1} - K_t)`, "
        "with `K_t = dc_t + (e^rho_t - 1)|h_{t-1} - c_t|`, and the endpoint form, the reported one. "
        "The endpoint form bounds the one-step error identity "
        "`e_t = s e_{t-1} + x~ (b - b~) + (Abar_t - s) h_{t-1} + (x - x~) b`, with `x = lam dt`, "
        "`b = B u / lam` and `s = exp(-x~)` the perturbed transition, by "
        "`s E_{t-1} + x~ db_t + |(Abar_t - s) h_{t-1} + (x - x~) b|` and takes its largest value over "
        "the interval that holds `s`, the intersection of the intervals given by `rho_t` and by the "
        "additive bound on `|dt - dt~|`. That value is at an end of the interval or at one "
        "stationary point in closed form, and it is never larger than the recursion or the "
        "additive step. It is computed at layers "
        f"{_layers(run['local_layers'])}. The attack runs at "
        + "; ".join(f"layers {_layers(ls)} at L = {L}" for L, ls in run['attack_layers_by_length'].items())
        + ". Per (layer, length) it "
        f"ranks every (position, output coordinate) pair with {budget[str(lengths[0])]['probes']} "
        "random sign probes, takes the exact gradient row of the best-ranked pairs ("
        + ", ".join(f"{budget[str(L)]['candidates']} at L = {L}" for L in lengths)
        + f", plus {budget[str(lengths[0])]['last_candidates']} at the last position), and runs "
        f"sign-gradient ascent for {budget[str(lengths[0])]['steps']} steps from "
        f"{budget[str(lengths[0])]['restarts']} restarts per `delta`: the signs of the two best "
        "rows, one random corner, and the best row at the last position, the last on the largest "
        "change at the last position. Every best point is scored again with the reference forward "
        "pass. Values are per unit `delta`: the largest change over positions and coordinates "
        "divided by `delta`.",
        "",
    ]
    trows = []
    for e in att:
        for pd in e["per_delta"]:
            trows.append([e["layer"], e["seq_len"], pd["delta"], pd["slope_max"], pd["last_slope_max"],
                          *[pd["local"][rule]["lip_max"] for rule in LOCAL_RULES],
                          pd["data_free_bound_max"] / pd["delta"], pd["worst_position_from_end"],
                          pd["pgd_gain"]])
    md += [table(trows, ["layer", "L", "delta", "attack", "attack, last position",
                         *[f"local, {SHORT[r]}" for r in LOCAL_RULES], "data-free, any point",
                         "worst position, from the end", "ascent gain"]), ""]

    small = [(e, e["per_delta"][deltas.index(d0)]) for e in att]
    ratio = {rule: [pd["local"][rule]["over_attack"] for _, pd in small] for rule in LOCAL_RULES}
    df_ratio = [pd["data_free_bound_max"] / pd["attacked_max"] for _, pd in small]
    first = [e["first_order"] / pd["slope_max"] for e, pd in small]
    gain = [pd["pgd_gain"] for e in att for pd in e["per_delta"]]
    at_long = [(e, e["per_delta"][deltas.index(d0)]) for e in att if e["seq_len"] == Lmax]
    hardest_att = max(at_long, key=lambda p: p[1]["slope_max"]) if at_long else None
    md += [
        f"At `delta = {fmt(d0)}` the attacked slope ranges over "
        f"{_span([pd['slope_max'] for _, pd in small])} across the attacked layers and lengths"
        + (f", and at L = {Lmax} it is largest at layer {hardest_att[0]['layer']}, "
           f"{fmt(hardest_att[1]['slope_max'])}" if hardest_att else "")
        + ". The local bound is "
        + "; ".join(f"{_span(ratio[rule])} times the attacked slope with the {RULE_NAMES[rule]} rule"
                    for rule in LOCAL_RULES)
        + f". The data-free bound is {_span(df_ratio)} times the attacked change. The largest "
        f"first-order value `|row|_1` over the gradient rows divided by the attacked slope is "
        f"{_span(first)}, and the ascent improves on its best start by a factor of {_span(gain)} "
        "across all `delta`"
        + (", so the attacked change is the first-order optimum of the best-ranked target and "
           "the mixer is linear at these `delta` to the digits shown."
           if (max(gain) < 1.001 and all(abs(f - 1.0) < 1e-3 for f in first)) else "."),
        "",
    ]
    df_v, local_v = _violations(results)
    md += [
        ("No attacked value exceeds its certificate: there are "
         f"{df_v} violations of the data-free bound, checked per output coordinate in the reading "
         "where the perturbed input is any point within `delta`, and "
         f"{local_v} of the local bounds, checked per position and coordinate where they are finite.")
        if df_v + local_v == 0 else
        (f"Violations found: {df_v} of the data-free bound and {local_v} of the local bounds. A "
         "violation falsifies that bound."),
        "",
        "The local certificate at every layer, against the length and `delta`, as the largest "
        "sensitivity over positions and coordinates:",
        "",
    ]
    lrows = []
    for e in exps:
        lrows.append([e["layer"], e["seq_len"], *[e["per_delta"][j]["local"][rule]["lip_max"]
                                                  for j in range(len(deltas)) for rule in LOCAL_RULES]])
    md += [table(lrows, ["layer", "L", *[f"{SHORT[r]}, delta = {fmt(d)}" for d in deltas
                                          for r in LOCAL_RULES]]), ""]

    def wins(d, L=None):
        sel = [e for e in exps if L is None or e["seq_len"] == L]
        j = deltas.index(d)
        end = [e["per_delta"][j]["local"][REPORTED_RULE]["lip_max"] for e in sel]
        rec = [e["per_delta"][j]["local"]["log_delta_recursive"]["lip_max"] for e in sel]
        add = [e["per_delta"][j]["local"]["additive"]["lip_max"] for e in sel]
        mx = [e["per_delta"][j]["local"]["log_delta"]["lip_max"] for e in sel]
        n = len(sel)
        end_beats = sum(1 for r, a in zip(end, add) if r < a)
        add_inf = sum(1 for a in add if not math.isfinite(a))
        end_inf = sum(1 for r in end if not math.isfinite(r))
        ratio_ea = [r / a for r, a in zip(end, add) if math.isfinite(r) and math.isfinite(a)]
        ratio_re = [r / x for r, x in zip(rec, end) if math.isfinite(r) and math.isfinite(x)]
        ratio_mr = [m / r for m, r in zip(mx, rec) if math.isfinite(r) and math.isfinite(m)]
        return n, end_beats, add_inf, end_inf, ratio_ea, ratio_re, ratio_mr

    sentences = []
    for d in deltas:
        n, eb, ai, ei, rea, rre, rmr = wins(d)
        sentences.append(
            f"At `delta = {fmt(d)}` the endpoint form is below the additive bound in {eb} of the "
            f"{n} (layer, length) pairs; the additive bound is infinite in {ai} of them and the "
            f"endpoint form in {ei}; "
            + (f"where both are finite the endpoint form over the additive bound is {_span(rea)}"
               if rea else "the two are never both finite")
            + (f", the recursion over the endpoint form is {_span(rre)}" if rre else "")
            + (f", and the running maximum over the recursion is {_span(rmr)}" if rmr else "")
            + ".")
    md += [" ".join(sentences), ""]
    by_d = {d: [e["per_delta"][deltas.index(d)]["local"][REPORTED_RULE]["diagnostics"]["rho_max"]
                for e in exps] for d in deltas}
    local_long = [(e["layer"], e["per_delta"][deltas.index(d0)]["local"][REPORTED_RULE]["lip_max"])
                  for e in exps if e["seq_len"] == Lmax]
    if local_long:
        hard = max(local_long, key=lambda p: p[1] if math.isfinite(p[1]) else math.inf)
        easy = min(local_long, key=lambda p: p[1])
        md += [
            "The largest relative timescale perturbation `rho = L_log_delta delta` on the local box "
            "is " + ", ".join(f"{_span(v)} at `delta = {fmt(d)}`" for d, v in by_d.items())
            + f". At L = {Lmax} and `delta = {fmt(d0)}` the local certificate, the endpoint form, is "
            f"largest at layer {hard[0]}, {fmt(hard[1])}, and smallest at layer {easy[0]}, "
            f"{fmt(easy[1])}.",
            "",
        ]
    vac = []
    for d in deltas:
        j = deltas.index(d)
        vac.append(f"at `delta = {fmt(d)}` " + ", ".join(
            f"{_span([100 * e['per_delta'][j]['local'][rule]['vacuous_fraction'] for e in exps])} "
            f"percent with the {SHORT[rule]} rule" for rule in LOCAL_RULES))
    rel = []
    for d in deltas:
        j = deltas.index(d)
        rel.append(f"at `delta = {fmt(d)}` " + ", ".join(
            f"{_span([e['per_delta'][j]['local'][rule]['over_realised_out'] for e in exps])} with "
            f"the {SHORT[rule]} rule" for rule in LOCAL_RULES))
    md += [
        "A local bound at a (position, coordinate) pair is vacuous when `lip delta` is at least the "
        "width of the certified output box of that coordinate, or infinite. Over the (layer, length) "
        "pairs, the share of vacuous pairs is " + "; ".join(vac) + ". The certified output box is "
        f"itself {_span([r['realised']['out']['certified_max'] / r['realised']['out']['realised_max'] for r in rows])} "
        "times the largest realised output, so this test is weak. Against the realised output, "
        "the largest local bound on the change, `lip delta`, over the largest realised `|F(v)|` on "
        "the same sequence is " + "; ".join(rel) + ".",
        "",
    ]
    by_len = run["attack_layers_by_length"]
    full = [L for L, ls in by_len.items() if len(ls) == len(rows)]
    if full:
        Lf = max(full, key=int)
        j0 = deltas.index(d0)
        sel = {e["layer"]: e for e in exps if str(e["seq_len"]) == Lf}
        rank_att = sorted(sel, key=lambda l: -sel[l]["per_delta"][j0]["slope_max"])
        rank_loc = sorted(sel, key=lambda l: -sel[l]["per_delta"][j0]["local"][REPORTED_RULE]["lip_max"])
        rank_df = sorted(sel, key=lambda l: -rows[l]["out_slope_cell_max"])
        md += [
            f"At L = {Lf}, where every layer is attacked, and `delta = {fmt(d0)}`, the three layers "
            f"with the largest attacked slope are {', '.join(str(l) for l in rank_att[:3])} "
            f"({', '.join(fmt(sel[l]['per_delta'][j0]['slope_max']) for l in rank_att[:3])}), and "
            f"the smallest is layer {rank_att[-1]} ({fmt(sel[rank_att[-1]]['per_delta'][j0]['slope_max'])}). "
            f"The three largest local certificates are at layers {', '.join(str(l) for l in rank_loc[:3])}, "
            f"and the three largest data-free slopes at layers {', '.join(str(l) for l in rank_df[:3])}.",
            "",
        ]
    md += [
        f"`{results['figure']}` plots, per layer, the two data-free slopes, the local bound with "
        "each rule "
        f"at L = {Lmax} and the attacked slopes, at "
        + " and at ".join(f"`delta = {fmt(d)}`" for d in [d0] + ([1e-4] if 1e-4 in deltas else []))
        + ".",
        "",
    ]
    md += [
        "## What was run",
        "",
        f"The run took {results['seconds_total'] / 60:.1f} minutes on "
        f"{results['provenance']['platform']}, on the CPU with {run['threads']} threads. The "
        f"data-free certificate and the realised quantities cover all {len(rows)} layers on "
        f"{text['tokens']} tokens. The local certificate covers layers "
        f"{_layers(run['local_layers'])} at L = {', '.join(str(L) for L in lengths)} "
        f"and `delta` = {', '.join(fmt(d) for d in deltas)}. The attack covers "
        + "; ".join(f"layers {_layers(ls)} at L = {L}" for L, ls in run['attack_layers_by_length'].items())
        + ", at the same `delta`, one sequence each: the first L tokens of the text. Per attacked "
        "(layer, length):",
        "",
        table([[e["layer"], e["seq_len"], e["local_seconds"], e["attack_seconds"]] for e in att],
              ["layer", "L", "local certificate, seconds", "attack, seconds"]),
        "",
        "The attack is a heuristic search on one sequence per length, so every attacked value is a "
        "lower bound and the true worst case may be larger.",
        "",
        "## Scope",
        "",
        "All bounds hold over the reals for the float64 map. The released model as run by "
        "transformers rounds the residual stream, RMSNorm, `A`, the timescale bias, `B`, `u` and `D` "
        "to float32, and nothing here bounds that rounding. The data-free certificate covers every "
        "input of a layer; the local certificate and the attack cover the sup-norm ball of radius "
        "`delta` around one realised normalised input, which is not restricted to RMSNorm outputs. "
        "The incremental certificates are per layer with respect to the normalised input and do not "
        "compose through the residual stream and the RMSNorm in front of the next layer. The text "
        f"file states in its header that it was transcribed to ASCII, so dashes and quotation marks "
        "may differ from a printed edition.",
        "",
    ]
    return "\n".join(md)


# --- main -------------------------------------------------------------------------------------------


def _strip(obj):
    """Drop tensors kept for the violation checks before writing the JSON."""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if not isinstance(v, torch.Tensor)}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Certify and attack mamba-130m-hf on public-domain text.")
    parser.add_argument("--attack-layers", type=int, nargs="+", default=ATTACK_LAYERS)
    parser.add_argument("--attack-every-layer-up-to", type=int, default=ATTACK_EVERY_LAYER_UP_TO,
                        help="attack every layer at the lengths up to this; 0 for none")
    parser.add_argument("--local-layers", type=int, nargs="*", default=None,
                        help="layers for the local certificate; default every layer")
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--deltas", type=float, nargs="+", default=DELTAS)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--fig-dir", default=str(FIG_DIR))
    args = parser.parse_args()

    set_seed(SEED)
    torch.set_num_threads(args.threads)
    start = time.time()
    lengths = sorted(set(args.lengths))
    deltas = sorted(set(args.deltas))
    if max(lengths) > MAX_TOKENS:
        raise SystemExit(f"lengths above {MAX_TOKENS} tokens are not available")

    print("loading the model ...", flush=True)
    model = load_released()
    tok = load_tokenizer()
    layers = released_layers(model)
    ids = tokenise(tok, read_sample_text())
    n_tokens = int(ids.shape[1])
    if max(lengths) > n_tokens:
        raise SystemExit(f"the text has {n_tokens} tokens, fewer than {max(lengths)}")
    attack_by_length = {
        L: sorted(set(range(len(layers))) if L <= args.attack_every_layer_up_to
                  else set(args.attack_layers))
        for L in lengths
    }
    attack_layers = sorted(set(args.attack_layers))
    local_layers = sorted(set(args.local_layers if args.local_layers is not None
                              else range(len(layers))) | set().union(*attack_by_length.values()))

    print("reference forward pass against transformers ...", flush=True)
    with torch.no_grad():
        hf = model.backbone(input_ids=ids, use_cache=False).last_hidden_state
    faithful, _ = reference_forward(model, layers, ids, faithful=True)
    pure, inputs_full = reference_forward(model, layers, ids)
    agreement = {
        "tokens": n_tokens,
        "final_abs_max": float(hf.abs().max()),
        "faithful_max_abs_diff": float((faithful - hf).abs().max()),
        "pure_max_abs_diff": float((pure - hf).abs().max()),
    }
    del faithful, pure, hf

    print("data-free certificates ...", flush=True)
    t0 = time.time()
    df_rows, incs, widths = [], [], []
    for layer, x in zip(layers, inputs_full):
        row, inc, width = data_free_layer(layer, x, lengths)
        widths.append(width)
        df_rows.append(row)
        incs.append(inc)
        print(f"  layer {layer.index}: slope {row['out_slope_cell_max']:.3e}", flush=True)
    df_seconds = time.time() - t0

    experiments = []
    for li_len, L in enumerate(lengths):
        inputs = inputs_full if L == n_tokens else reference_forward(model, layers, ids[:, :L])[1]
        for l in local_layers:
            layer = layers[l]
            v = rms_norm(inputs[l], layer.norm_weight, layer.eps)
            do_attack = l in attack_by_length[L]
            print(f"  L={L} layer {l}: local{' and attack' if do_attack else ''} ...", flush=True)
            e = experiment(layer, incs[l], widths[l], v, deltas, do_attack,
                           SEED + 1000 * l + 100 * li_len)
            experiments.append({"layer": l, "seq_len": L, **e})

    import transformers

    cfg = model.config
    results = {
        "provenance": provenance(SEED),
        "model": {
            "id": MODEL_ID,
            "local_dir": str(model_dir()),
            "transformers_version": transformers.__version__,
            "layers": len(layers),
            "d_model": int(cfg.hidden_size),
            "d_inner": layers[0].d_inner,
            "d_state": layers[0].d_state,
            "dt_rank": layers[0].dt_rank,
            "d_conv": layers[0].d_conv,
            "rmsnorm_eps": layers[0].eps,
        },
        "text": {"file": str(SAMPLE_TEXT.relative_to(REPO_ROOT)).replace("\\", "/"), "tokens": n_tokens},
        "agreement": agreement,
        "run": {
            "lengths": lengths,
            "deltas": deltas,
            "attack_layers": attack_layers,
            "attack_layers_by_length": {str(L): ls for L, ls in attack_by_length.items()},
            "local_layers": local_layers,
            "threads": args.threads,
            "sequences_per_length": 1,
            "sequence": "the first L tokens of the text",
            "perturbation": "any point within delta in sup-norm of the layer's realised normalised input",
        },
        "budget": {str(L): asdict(budget_for(L)) for L in lengths},
        "attack": "per (layer, length): rank targets by random sign probes at the smallest delta; "
                  "exact gradient rows of the best-ranked targets; per delta, sign-gradient ascent "
                  "on the largest change over all positions from the two best rows' signs and from "
                  "a random corner, and on the largest change at the last position from its best "
                  "row; step size from 2 delta down to delta / 8; every best point rescored with "
                  "released.mixer_trace in float64",
        "data_free": {"layers": df_rows, "summary": data_free_summary(df_rows, lengths),
                      "seconds": df_seconds},
        "experiments": _strip(experiments),
        "seconds_total": None,
    }
    fig_dir, out_dir = Path(args.fig_dir), Path(args.out_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "released_layers.pdf"
    results["figure"] = str(fig_path).replace("\\", "/")
    results["seconds_total"] = time.time() - start
    figure(results, fig_path)
    md = markdown(results)
    (out_dir / "released.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (out_dir / "released.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"wrote {out_dir / 'released.json'}, {out_dir / 'released.md'} and {fig_path}")


if __name__ == "__main__":
    main()
