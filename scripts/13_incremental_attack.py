"""Attack the incremental bound on held-out windows, and trace the local domain stage by stage.

Three measurements, on both trained arms:

  block  sign-gradient ascent on |F(xn + xi) - F(xn)| over |xi|_inf <= delta, where
         F = out_proj(y * SiLU(gate)) is a block's contribution before the residual and xn is
         its realised normalised input, at sequence lengths 10, 100, 1000 and 5000. The result is
         set against the data-free certificate of src/verification/incremental.py, the additive
         baseline at the same length, and the local block bound with each recurrence rule.
  model  the spread of the prediction over an eps ball at the model input, the quantity
         scripts/08_domains.py attacks, at the same lengths.
  local  the norm-propagation certificate of src/verification/lipschitz.py with the additive
         recurrence step and with the three log-timescale steps (the local corollary's running
         maximum, the recursion in its proof, and the endpoint form, the reported one), on the
         windows and eps values of
         docs/certification/domains.json, traced stage by stage so that the growth of the bound
         can be assigned to LayerNorm, the feed-forward layers and the recurrence.

An attacked value is reached, so attacked <= true <= certified. An attacked value is a lower bound
on the true sensitivity, and an attacked value above its certificate would falsify the certificate.

Writes docs/certification/incremental_attack.{json,md} and
docs/figures/sensitivity_vs_length.{pdf,png}.
"""

from __future__ import annotations

import argparse
import copy
import itertools
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

from src.dataset import LOBDataModule
from src.models.mamba import LOBMamba, SelectiveSSMBlock
from src.utils.config import load_config
from src.utils.repro import provenance, set_seed
from src.verification.certify import certify_output_range
from src.verification.incremental import (
    REPORTED_RULE,
    RULES,
    BlockIncrement,
    _delta_tight,
    _log_delta_lip,
    additive_step,
    contribution,
    incremental_certificate,
    layernorm_lipschitz,
    local_block_sensitivity,
    local_sensitivity_rule,
    log_delta_step,
    timescale_offsets,
)
from src.verification.lipschitz import Sens, _causal_conv, add, layer_norm, linear, mul, silu

OUT_DIR = Path("docs/certification")
FIG_DIR = Path("docs/figures")
DOMAINS = OUT_DIR / "domains.json"
ARMS = {"bounded": "best_mamba.pt", "softplus": "best_mamba_softplus.pt"}
RULE_NAMES = {"additive": "additive", "log_delta": "log-timescale, running maximum",
              "log_delta_recursive": "log-timescale, recursive",
              "additive_endpoint": "additive interval, endpoint",
              "log_delta_endpoint": "log-timescale, endpoint"}
SHORT = {"additive": "additive", "log_delta": "maximum", "log_delta_recursive": "recursive",
         "additive_endpoint": "additive endpoint", "log_delta_endpoint": "endpoint"}
LENGTHS = [10, 100, 1000, 5000]
BLOCK_DELTAS = [1e-6, 1e-4, 1e-2, 1e-1]
MODEL_EPS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]   # the eps values of docs/certification/domains.json
LENGTH_EPS = [1e-6, 1e-5]                            # the local end-to-end bound against length
DTYPE = torch.float64
CHUNK = 256

# the colours and type of scripts/05_figures.py, so the figure sits with the paper's others
OURS = "#1f77b4"
OTHER = "#ff7f0e"
LOCAL = "#6f42c1"
INK = "#212529"
FAINT = "#adb5bd"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
        "pdf.fonttype": 42,
    }
)
FIGSIZE = (6.5, 6.4)


# --- a float64 forward pass that autograd can afford at L = 5000 ----------------------------


def _scan(delta, B, C, u, A, D, keep_states: bool):
    """y_t = C_t . h_t + D u_t, with h_t = exp(dt_t A) h_{t-1} + (expm1(dt_t A) / A) B_t u_t.

    expm1(dt A) / A is the exact zero-order hold gain dt phi(dt A), written the other way, so the
    result agrees with SelectiveSSMBlock.forward to rounding. The elementwise work runs a chunk of
    time steps at once, time-major, and the loop does one fused multiply-add per step. Returns y
    as (batch, L, d) and, if keep_states, every h_t as (L, batch, d, n).
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
        em = torch.expm1(dt[t0:t1].unsqueeze(-1) * A)
        a = em + 1.0
        bu = (em / A) * Bt[t0:t1].unsqueeze(2) * ut[t0:t1].unsqueeze(-1)
        buf = hs[t0:t1] if keep_states else torch.empty_like(bu)
        for i in range(t1 - t0):
            h = torch.addcmul(bu[i], a[i], h, out=buf[i])
        y[t0:t1] = (buf * Ct[t0:t1].unsqueeze(2)).sum(-1)
    return (y + D * ut).transpose(0, 1), hs


class _Scan(torch.autograd.Function):
    """The scan with a hand-written adjoint, which stores h_t once instead of every
    intermediate of every step. tests/test_incremental_attack.py checks it with gradcheck."""

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
            em = torch.expm1(dt[t0:t1].unsqueeze(-1) * A)
            a = em + 1.0
            gh = gyt[t0:t1].unsqueeze(-1) * Ct[t0:t1].unsqueeze(2)
            for i in range(t1 - t0 - 1, -1, -1):
                torch.add(gh[i], carry, out=gh[i])
                carry = gh[i] * a[i]
            if t0 > 0:
                h_prev = hs[t0 - 1 : t1 - 1]
            else:
                h_prev = torch.cat([hs.new_zeros(1, b, d, n), hs[: t1 - 1]], 0)
            Bc, uc = Bt[t0:t1], ut[t0:t1]
            # d h_t / d dt_t = a_t (A h_{t-1} + B_t u_t), since d/d dt of expm1(dt A)/A is exp(dt A)
            g_dt[t0:t1] = (gh * a * (A * h_prev + Bc.unsqueeze(2) * uc.unsqueeze(-1))).sum(-1)
            ghg = gh * (em / A)
            g_B[t0:t1] = (ghg * uc.unsqueeze(-1)).sum(2)
            g_u[t0:t1] = (ghg * Bc.unsqueeze(2)).sum(-1) + gyt[t0:t1] * D
            g_C[t0:t1] = (hs[t0:t1] * gyt[t0:t1].unsqueeze(-1)).sum(2)
        return (g_dt.transpose(0, 1), g_B.transpose(0, 1), g_C.transpose(0, 1),
                g_u.transpose(0, 1), None, None)


def ssm_contribution(block: SelectiveSSMBlock, xn: torch.Tensor) -> torch.Tensor:
    """F(xn) = out_proj(y * SiLU(gate)) from the normalised input, differentiable in xn.

    Equal to incremental.contribution to rounding; that function, not this one, scores every
    reported number.
    """
    L = xn.shape[1]
    u, gate = block.in_proj(xn).chunk(2, dim=-1)
    u = F.silu(block.conv1d(u.transpose(1, 2))[:, :, :L].transpose(1, 2))
    delta, B, C = block.ssm_parameters(u)
    A = -torch.exp(block.A_log)
    if torch.is_grad_enabled() and xn.requires_grad:
        y = _Scan.apply(delta, B, C, u, A, block.D)
    else:
        y, _ = _scan(delta, B, C, u, A, block.D, keep_states=False)
    return block.out_proj(y * F.silu(gate))


def predict(model: LOBMamba, x: torch.Tensor) -> torch.Tensor:
    """LOBMamba.forward in the model's own dtype, through ssm_contribution."""
    h = model.proj(x)
    for block in model.layers:
        h = h + ssm_contribution(block, block.norm(h))
    return model.head(model.final_norm(h)[:, -1, :]).squeeze(-1)


@torch.no_grad()
def predict_reference(model: LOBMamba, x: torch.Tensor) -> torch.Tensor:
    """LOBMamba.forward without its cast to float32, through SelectiveSSMBlock.forward."""
    h = model.proj(x)
    for block in model.layers:
        h = block(h)
    return model.head(model.final_norm(h)[:, -1, :]).squeeze(-1)


@torch.no_grad()
def block_inputs(model: LOBMamba, x: torch.Tensor) -> list[torch.Tensor]:
    """The realised normalised input of every block."""
    h = model.proj(x)
    out = []
    for block in model.layers:
        out.append(block.norm(h))
        h = block(h)
    return out


def float64_copy(model: LOBMamba) -> LOBMamba:
    m = copy.deepcopy(model).to(DTYPE).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _signs(shape, gen: torch.Generator) -> torch.Tensor:
    return (torch.randint(0, 2, shape, generator=gen) * 2 - 1).to(DTYPE)


# --- the attacks ----------------------------------------------------------------------------


@dataclass
class Budget:
    """What one attack on one window runs. Recorded in the JSON as run."""

    probes: int           # random sign vectors at the smallest delta, to rank the targets
    candidates: int       # targets over all positions whose exact gradient row is taken
    last_candidates: int  # targets at the last position whose exact gradient row is taken
    block_steps: int      # sign-gradient steps per restart of the block attack
    model_steps: int      # sign-gradient steps per restart of the model attack
    batch: int            # replicas per forward pass


def budget_for(seq_len: int) -> Budget:
    """The same probes and steps at every length. Only the number of candidate targets grows,
    since a longer window has more positions to rank, and the batch shrinks to bound memory."""
    candidates = 24 if seq_len <= 100 else (32 if seq_len <= 1000 else 48)
    return Budget(probes=32, candidates=candidates, last_candidates=8, block_steps=12,
                  model_steps=8, batch=32 if seq_len <= 100 else 16)


def _step_size(radius: torch.Tensor, step: int, steps: int) -> torch.Tensor:
    # from 2 delta, which flips a coordinate across the box, down to delta/8
    return radius * 2.0 * 0.5 ** (4.0 * step / max(steps, 1))


def _pgd(objective, xi: torch.Tensor, radius: torch.Tensor, steps: int, batch: int):
    """Sign-gradient ascent on objective(xi) -> (elements,), projected onto |xi| <= radius.

    Returns the best point visited per element, its value, and the value at the start.
    """
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


def attack_block(block: SelectiveSSMBlock, xn: torch.Tensor, deltas: list[float],
                 budget: Budget, seed: int) -> dict:
    """The largest |F(xn + xi) - F(xn)| found over |xi|_inf <= delta, for one window xn (1, L, d).

    1. budget.probes random sign vectors at the smallest delta rank every (position, coordinate)
       by the root mean square of its change, which estimates the l2 norm of its gradient row.
    2. The exact gradient row of the top budget.candidates targets, and of the top
       budget.last_candidates at the last position, gives each its first-order optimum
       delta |row|_1, at xi = delta sign(row).
    3. Per delta, sign-gradient ascent on the largest change over all positions and coordinates,
       started from the two best rows' signs and from a random corner, and on the largest change
       at the last position, started from that position's best row.
    4. The best point of every restart is scored again with incremental.contribution in float64.
       Those are the reported numbers.
    """
    gen = torch.Generator().manual_seed(seed)
    L, d_in = xn.shape[1], xn.shape[2]
    d_out = block.d_model
    with torch.no_grad():
        base = ssm_contribution(block, xn)

    # 1. rank the targets
    d0 = min(deltas)
    sq = torch.zeros(L * d_out, dtype=DTYPE)
    random_corner = 0.0
    for p0 in range(0, budget.probes, budget.batch):
        p = min(budget.batch, budget.probes - p0)
        with torch.no_grad():
            ch = ssm_contribution(block, xn + d0 * _signs((p, L, d_in), gen)) - base
        sq += (ch * ch).sum(0).flatten()
        random_corner = max(random_corner, float(ch.abs().max()) / d0)

    # 2. exact gradient rows of the best-ranked targets
    top = torch.topk(sq, min(budget.candidates, sq.numel())).indices
    last_scores = sq.view(L, d_out)[-1]
    last = torch.topk(last_scores, min(budget.last_candidates, d_out)).indices + (L - 1) * d_out
    targets = torch.unique(torch.cat([top, last]))
    rows = []
    for i0 in range(0, len(targets), budget.batch):
        tg = targets[i0 : i0 + budget.batch]
        xi = torch.zeros(len(tg), L, d_in, dtype=DTYPE, requires_grad=True)
        out = ssm_contribution(block, xn + xi).flatten(1)
        (g,) = torch.autograd.grad(out.gather(1, tg.unsqueeze(1)).sum(), xi)
        rows.append(g)
    rows = torch.cat(rows)
    l1 = rows.abs().flatten(1).sum(1)
    order = torch.argsort(l1, descending=True)
    at_last = targets // d_out == L - 1
    best_last = order[at_last[order]][0]

    # 3. restarts per delta: two best rows, a random corner, and the best row at the last position
    starts, radius, last_flags = [], [], []
    for d in deltas:
        for j in order[:2]:
            starts.append(d * torch.sign(rows[j]))
            last_flags.append(False)
        starts.append(d * _signs((L, d_in), gen))
        last_flags.append(False)
        starts.append(d * torch.sign(rows[best_last]))
        last_flags.append(True)
        radius += [d] * 4
    xi0 = torch.stack(starts)
    r = torch.tensor(radius, dtype=DTYPE).view(-1, 1, 1)
    flags = torch.tensor(last_flags)

    def objective(x, sl):
        ch = (ssm_contribution(block, xn + x) - base).abs()
        return torch.where(flags[sl], ch[:, -1].amax(-1), ch.flatten(1).amax(1))

    best_xi, best_val, start_val = _pgd(objective, xi0, r, budget.block_steps, budget.batch)

    # 4. score the best points with the reference pass
    with torch.no_grad():
        ref_base = contribution(block, xn)
        dev = []
        for i0 in range(0, best_xi.shape[0], budget.batch):
            dev.append((contribution(block, xn + best_xi[i0 : i0 + budget.batch]) - ref_base).abs())
        dev = torch.cat(dev)                                  # (restarts, L, d_out)

    per_delta = []
    for i, d in enumerate(deltas):
        sl = slice(4 * i, 4 * i + 4)
        m = dev[sl].amax(0)                                   # (L, d_out)
        flat = int(m.flatten().argmax())
        per_delta.append(
            {
                "delta": d,
                "map": m,
                "max": float(m.max()),
                "last": float(m[-1].max()),
                "argmax": [flat // d_out, flat % d_out],
                # how much the ascent added to the best of the four starts
                "pgd_gain": float(best_val[sl].max() / start_val[sl].max()),
            }
        )
    return {
        "per_delta": per_delta,
        "first_order": float(l1.max()),
        "first_order_last": float(l1[at_last].max()),
        "first_order_target": [int(targets[order[0]]) // d_out, int(targets[order[0]]) % d_out],
        "random_corner": random_corner,
    }


def attack_model(model: LOBMamba, x: torch.Tensor, eps_list: list[float], budget: Budget) -> dict:
    """The spread of the prediction over |x' - x|_inf <= eps, for one window x (1, L, inputs).

    The exact input gradient gives the first-order spread 2 eps |grad|_1. Per eps, sign-gradient
    ascent on +f from x + eps sign(grad) and on -f from x - eps sign(grad). The best points are
    scored again with predict_reference, and the spread is the largest minus the smallest
    prediction among them and the nominal one, as scripts/08_domains.py takes it. Random
    corners are left out: on the windows of docs/certification/domains.json they never beat
    the gradient sign, which is the first-order optimum.
    """
    xg = x.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(predict(model, xg).sum(), xg)
    l1 = float(g.abs().sum())
    direction = torch.sign(g[0])

    starts, radius, sign = [], [], []
    for e in eps_list:
        for s in (1.0, -1.0):
            starts.append(s * e * direction)
            radius.append(e)
            sign.append(s)
    xi0 = torch.stack(starts)
    r = torch.tensor(radius, dtype=DTYPE).view(-1, 1, 1)
    sgn = torch.tensor(sign, dtype=DTYPE)

    def objective(xi, sl):
        return sgn[sl] * predict(model, x + xi)

    best_xi, _, _ = _pgd(objective, xi0, r, budget.model_steps, budget.batch)
    with torch.no_grad():
        nominal = float(predict_reference(model, x))
        vals = torch.cat([predict_reference(model, x + best_xi[i0 : i0 + budget.batch])
                          for i0 in range(0, best_xi.shape[0], budget.batch)])
    per = len(starts) // len(eps_list)
    rows = []
    for i, e in enumerate(eps_list):
        v = vals[per * i : per * (i + 1)]
        hi, lo = max(float(v.max()), nominal), min(float(v.min()), nominal)
        rows.append({"eps": e, "width": hi - lo, "first_order_width": 2.0 * e * l1})
    return {"nominal": nominal, "grad_l1": l1, "per_eps": rows}


# --- the local domain, stage by stage ---------------------------------------------------------


GROUPS = ("LayerNorm", "feed-forward", "recurrence", "residual")


def _lip_max(s) -> float:
    """Largest sensitivity, nan if any entry is nan; a float passes through."""
    if isinstance(s, float):
        return s
    lip = s.lip if isinstance(s, Sens) else s
    if torch.isnan(lip).any():
        return math.nan
    return float(lip.max())


def _fmax(a: float, b: float) -> float:
    """max that keeps a nan, which Python's max drops when it comes second."""
    return math.nan if (math.isnan(a) or math.isnan(b)) else max(a, b)


@torch.no_grad()
def _block_trace(block: SelectiveSSMBlock, x: Sens, eps: float, rule: str, rec, li: int):
    """incremental._block_rule line for line, recording each stage and the recurrence's state.

    The recurrence is also run with the timescale's sensitivity set to zero. That run is not a
    bound; it measures how much of the recurrence stage's growth comes through the timescale.
    The step multiplier is the largest factor one step applies to the carried bound: for the
    additive rule Abar plus its sensitivity, which can exceed 1; for the running maximum 1; for
    the recursive rule the slowest admissible perturbed transition, below 1; for the endpoint rule
    the upper end of the interval it maximises over, the slowest transition that both the log and
    the additive timescale budgets admit, at most that of the recursive rule.
    """
    L = x.val.shape[1]
    xn = layer_norm(x, block.norm, eps)
    rec(f"L{li} LayerNorm", "LayerNorm", xn)
    xz = linear(xn, block.in_proj.weight, block.in_proj.bias)
    u = Sens(xz.val[..., : block.d_inner], xz.lip[..., : block.d_inner])
    gate = Sens(xz.val[..., block.d_inner :], xz.lip[..., block.d_inner :])
    rec(f"L{li} in_proj", "feed-forward", u)

    u = _causal_conv(u, block)
    rec(f"L{li} conv", "feed-forward", u)
    u = silu(u, eps)
    rec(f"L{li} SiLU", "feed-forward", u)

    proj = linear(u, block.x_proj.weight, block.x_proj.bias)
    r, n = block.dt_rank, block.d_state
    pre = linear(Sens(proj.val[..., :r], proj.lip[..., :r]),
                 block.dt_proj.weight, block.dt_proj.bias)
    delta = _delta_tight(block, pre, eps)
    log_lip = _log_delta_lip(block, pre, eps) if rule != "additive" else None
    B = Sens(proj.val[..., r : r + n], proj.lip[..., r : r + n])
    C = Sens(proj.val[..., r + n :], proj.lip[..., r + n :])

    A = -torch.exp(block.A_log)
    batch = x.val.shape[0]
    zeros = torch.zeros(batch, block.d_inner, block.d_state, dtype=x.val.dtype)
    h = Sens(zeros, zeros.clone())
    h_frozen = Sens(zeros.clone(), zeros.clone())
    ys_v, ys_l = [], []
    h_max, frozen_max = 0.0, 0.0
    growth_max, growth_cells, raised_cells, cells = 0.0, 0, 0, 0

    for t in range(L):
        dt_t = Sens(delta.val[:, t].unsqueeze(-1), delta.lip[:, t].unsqueeze(-1))
        u_t = Sens(u.val[:, t].unsqueeze(-1), u.lip[:, t].unsqueeze(-1))
        B_t = Sens(B.val[:, t].unsqueeze(1), B.lip[:, t].unsqueeze(1))
        C_t = Sens(C.val[:, t].unsqueeze(1), C.lip[:, t].unsqueeze(1))
        still = Sens(dt_t.val, torch.zeros_like(dt_t.lip))
        before = h.lip

        if rule == "additive":
            # mul(A_bar, h) multiplies h.lip by A_bar.abs_max(eps); above 1 the bound grows
            dtA = dt_t.val * A.unsqueeze(0)
            dtA_lo = (dt_t.val - dt_t.lip * eps).clamp(min=0.0) * A.unsqueeze(0)
            growth = torch.exp(dtA) + A.abs() * torch.exp(dtA_lo) * dt_t.lip * eps
            h = additive_step(h, dt_t, u_t, B_t, A, eps)
            h_frozen = additive_step(h_frozen, still, u_t, B_t, A, eps)
        else:
            recursive = rule == "log_delta_recursive"
            endpoint = rule == "log_delta_endpoint"
            additive_only = rule == "additive_endpoint"
            ll = log_lip[:, t].unsqueeze(-1)
            lam_dt = dt_t.val * A.abs().unsqueeze(0)
            if endpoint or additive_only:
                # f is affine in E_{t-1} with coefficient the chosen end, at most the upper one;
                # the ablation drops the log-timescale bound, so only the additive interval binds
                rho = torch.full_like(ll, math.inf) if additive_only else ll * eps
                _, down = timescale_offsets(lam_dt, rho, A.abs().unsqueeze(0) * (dt_t.lip * eps))
                growth = torch.exp(-(lam_dt - down))
            elif recursive:
                # the carried bound is scaled by the perturbed transition, at most the slowest
                growth = torch.exp(-lam_dt * torch.exp(-ll * eps))
            else:
                # a running maximum carries the bound over unscaled
                growth = torch.ones_like(h.val)
            h = log_delta_step(h, dt_t, ll, u_t, B_t, A, eps, recursive=recursive,
                               endpoint=endpoint, additive_only=additive_only)
            h_frozen = log_delta_step(h_frozen, still, torch.zeros_like(ll), u_t, B_t, A, eps,
                                      recursive=recursive, endpoint=endpoint,
                                      additive_only=additive_only)
        growth_max = _fmax(growth_max, _lip_max(growth))
        growth_cells += int((growth > 1.0).sum())
        raised_cells += int((h.lip > before).sum())
        cells += h.lip.numel()
        h_max = _fmax(h_max, _lip_max(h))
        frozen_max = _fmax(frozen_max, _lip_max(h_frozen))

        cy = mul(h, C_t, eps)
        ys_v.append(cy.val.sum(-1) + block.D * u.val[:, t])
        ys_l.append(cy.lip.sum(-1) + block.D.abs() * u.lip[:, t])

    rec(f"L{li} recurrence", "recurrence", h_max)
    y = Sens(torch.stack(ys_v, 1), torch.stack(ys_l, 1))
    rec(f"L{li} readout", "recurrence", y)
    gated = mul(y, silu(gate, eps), eps)
    rec(f"L{li} gate", "feed-forward", gated)
    out = linear(gated, block.out_proj.weight, block.out_proj.bias)
    rec(f"L{li} out_proj", "feed-forward", out)
    res = add(out, x)
    rec(f"L{li} residual", "residual", res)

    diag = {
        "layer": li,
        "u_lip_max": _lip_max(u),
        "delta_lip_max": _lip_max(delta),
        "log_delta_lip_max": _lip_max(log_lip) if log_lip is not None else None,
        "rho_max": _lip_max(log_lip) * eps if log_lip is not None else None,
        "h_lip_max": h_max,
        "h_lip_max_without_timescale": frozen_max,
        "step_multiplier_max": growth_max,
        "fraction_of_steps_growing": growth_cells / max(cells, 1),
        "fraction_of_steps_raised": raised_cells / max(cells, 1),
    }
    return res, diag


@torch.no_grad()
def trace_local(model: LOBMamba, x: torch.Tensor, eps: float, rule: str) -> dict:
    """incremental.local_sensitivity_rule, recording the largest sensitivity after every stage.

    The stages form one chain from the input, whose sensitivity is 1, to the prediction, so the
    product of the stage ratios max(after) / max(before) is the final bound. Its log is
    therefore the sum of the stage logs exactly, and grouping the stages assigns the bound's
    orders of magnitude to LayerNorm, the feed-forward layers and the recurrence.
    """
    model.eval()
    stages: list[dict] = []
    prev = [1.0]

    def rec(name: str, group: str, s) -> None:
        v = _lip_max(s)
        ratio = v / prev[0] if prev[0] > 0 else math.nan
        log10 = math.log10(ratio) if (math.isfinite(ratio) and ratio > 0) else (
            math.inf if ratio == math.inf else math.nan)
        stages.append({"stage": name, "group": group, "lip_max": v, "log10_ratio": log10})
        prev[0] = v

    h = linear(Sens(x, torch.ones_like(x)), model.proj[0].weight, model.proj[0].bias)
    rec("stem Linear", "feed-forward", h)
    h = silu(h, eps)
    rec("stem SiLU", "feed-forward", h)
    h = layer_norm(h, model.proj[2], eps)
    rec("stem LayerNorm", "LayerNorm", h)
    blocks = []
    for li, block in enumerate(model.layers):
        h, diag = _block_trace(block, h, eps, rule, rec, li)
        blocks.append(diag)
    h = layer_norm(h, model.final_norm, eps)
    rec("final LayerNorm", "LayerNorm", h)
    last = Sens(h.val[:, -1, :], h.lip[:, -1, :])
    rec("last position", "feed-forward", last)
    z = silu(linear(last, model.head[0].weight, model.head[0].bias), eps)
    rec("head Linear and SiLU", "feed-forward", z)
    out = linear(z, model.head[3].weight, model.head[3].bias)
    rec("head Linear", "feed-forward", out)

    lip = out.lip.squeeze(-1)
    first_bad = next((s["stage"] for s in stages if not math.isfinite(s["lip_max"])), None)
    groups = {g: 0.0 for g in GROUPS}
    for s in stages:
        groups[s["group"]] += s["log10_ratio"] if math.isfinite(s["log10_ratio"]) else 0.0
    return {
        "eps": eps,
        "rule": rule,
        "lipschitz": lip,
        "width": 2.0 * lip * eps,
        "stages": stages,
        "blocks": blocks,
        "first_non_finite_stage": first_bad,
        "log10_by_group": groups if first_bad is None else None,
    }


# --- data -------------------------------------------------------------------------------------


def held_out_windows(dm: LOBDataModule, seq_len: int, ends: list[int]) -> torch.Tensor:
    """Held-out windows of length seq_len ending at the given rows, through src/dataset.py."""
    old = dm.seq_len
    dm.seq_len = seq_len
    try:
        ds = dm.dataset("test")
    finally:
        dm.seq_len = old
    xs = []
    for e in ends:
        idx = e - seq_len + 1 - ds.start
        if not 0 <= idx < len(ds):
            raise ValueError(f"row {e} cannot end a test window of length {seq_len}")
        xs.append(ds[idx][0])
    return torch.stack(xs)


def window_ends(dm: LOBDataModule, n: int, longest: int) -> list[int]:
    """n end rows spread evenly over the test split, late enough for the longest window."""
    lo, hi = dm.idx_test[0] + longest - 1, dm.idx_test[1] - 1
    if n == 1:
        return [hi]
    return [int(round(lo + i * (hi - lo) / (n - 1))) for i in range(n)]


def domain_windows(dm: LOBDataModule, windows: int) -> torch.Tensor:
    """The windows of scripts/08_domains.py: the first of the unshuffled test loader."""
    x, _ = next(iter(dm.loader("test", shuffle=False, batch_size=max(8, windows))))
    return x[:windows]


def load_model(path: Path, arm: str) -> LOBMamba:
    if not path.exists():
        raise SystemExit(f"{path} not found; the attack needs the trained checkpoint")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(ckpt["config"])
    cfg["model"]["mamba"]["dt_parametrisation"] = arm
    model = LOBMamba(int(ckpt["architecture"]["input_dim"]), cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


# --- the experiments --------------------------------------------------------------------------


def _max(t: torch.Tensor) -> float:
    """Largest entry, inf if any entry is inf, nan if any is nan."""
    if torch.isnan(t).any():
        return math.nan
    return float(t.max())


def _median(t: torch.Tensor) -> float:
    """torch's median, the lower middle value for an even count, as scripts/08_domains.py takes
    it; nan if any entry is nan."""
    if torch.isnan(t).any():
        return math.nan
    return float(t.median())


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())


def certified_block(inc: BlockIncrement, lengths: list[int]) -> dict:
    """The data-free numbers the block attack is set against, maxima over output coordinates."""
    return {
        "ours_slope_max": float(inc.output_slope().max()),
        "ours_common_start_slope_max": float(inc.output_slope("common_start").max()),
        "ours_bound_max": {
            f"{d:g}": {
                "arbitrary": float(inc.output_bound(d, arbitrary=True).max()),
                "layernorm_outputs": float(inc.output_bound(d).max()),
            }
            for d in BLOCK_DELTAS
        },
        "additive_slope_max": {
            str(L): {
                "no_floor": float(inc.additive_output_slope(L, floor=False).max()),
                "certified_floor": float(inc.additive_output_slope(L, floor=True).max()),
            }
            for L in lengths
        },
    }


def block_experiment(block: SelectiveSSMBlock, inc: BlockIncrement, xn: torch.Tensor,
                     budget: Budget, seed: int) -> dict:
    """attack_block on every window, priced against the data-free and the local bounds.

    The attacked perturbation is any point within delta of the realised normalised input, so the
    data-free bound it must respect is the arbitrary reading, checked per output coordinate. The
    local bound is checked per position and coordinate wherever it is finite.
    """
    n = xn.shape[0]
    attacks = [attack_block(block, xn[w : w + 1], BLOCK_DELTAS, budget, seed + w) for w in range(n)]
    per_delta = []
    for i, d in enumerate(BLOCK_DELTAS):
        maps = torch.stack([a["per_delta"][i]["map"] for a in attacks])      # (n, L, d_model)
        bound = inc.output_bound(d, arbitrary=True)                            # (d_model,)
        slopes = maps.flatten(1).amax(1) / d
        worst = int(slopes.argmax())
        local = {}
        for rule in RULES:
            lip = local_block_sensitivity(block, xn, d, rule).lip
            ok = torch.isfinite(lip)
            over = (maps > lip * d * (1 + 1e-9) + 1e-14) & ok
            local[rule] = {
                "lip_max": _max(lip),
                "lip_median_over_windows": _median(lip.flatten(1).amax(1)),
                "finite": bool(ok.all()),
                "violations": int(over.sum()),
            }
        t, k = attacks[worst]["per_delta"][i]["argmax"]
        per_delta.append(
            {
                "delta": d,
                "attacked_max": float(maps.max()),
                "slope_max": float(slopes.max()),
                "slope_median": _median(slopes),
                "last_slope_max": float(maps[:, -1].max()) / d,
                "worst_window": worst,
                "worst_position": t,
                "worst_position_from_end": maps.shape[1] - 1 - t,
                "worst_coordinate": k,
                "pgd_gain_max": max(a["per_delta"][i]["pgd_gain"] for a in attacks),
                "data_free_bound_max": float(bound.max()),
                "data_free_violations": int((maps.amax(dim=(0, 1)) > bound * (1 + 1e-9)).sum()),
                "local": local,
            }
        )
    return {
        "per_delta": per_delta,
        "measured_slope_max": max(r["slope_max"] for r in per_delta),
        "last_slope_max": max(r["last_slope_max"] for r in per_delta),
        "first_order_slope_max": max(a["first_order"] for a in attacks),
        "first_order_slope_per_window": [a["first_order"] for a in attacks],
        "first_order_last_max": max(a["first_order_last"] for a in attacks),
        "random_corner_slope_max": max(a["random_corner"] for a in attacks),
    }


def model_experiment(m64: LOBMamba, model: LOBMamba, x: torch.Tensor, budget: Budget,
                     local_eps: list[float]) -> dict:
    """attack_model on every window, and the local end-to-end bound with each rule at local_eps.

    The local bound runs on the float32 model, as docs/certification/domains.json was made.
    """
    x64 = x.to(DTYPE)
    attacks = [attack_model(m64, x64[w : w + 1], MODEL_EPS, budget) for w in range(x.shape[0])]
    rows = []
    for j, e in enumerate(MODEL_EPS):
        widths = torch.tensor([a["per_eps"][j]["width"] for a in attacks], dtype=DTYPE)
        first = torch.tensor([a["per_eps"][j]["first_order_width"] for a in attacks], dtype=DTYPE)
        row = {
            "eps": e,
            "attack_width_median": _median(widths),
            "attack_width_max": _max(widths),
            "first_order_width_median": _median(first),
            "first_order_width_max": _max(first),
        }
        if e in local_eps:
            for rule in RULES:
                w = local_sensitivity_rule(model, x.to(torch.float32), e, rule)["width"].to(DTYPE)
                ok = torch.isfinite(w)
                row[rule] = {
                    "width_median": _median(w),
                    "width_max": _max(w),
                    "finite": bool(ok.all()),
                    "violations": int(((widths > w * (1 + 1e-6)) & ok).sum()),
                }
        rows.append(row)
    return {"per_eps": rows, "grad_l1_per_window": [a["grad_l1"] for a in attacks]}


def local_end_to_end(model: LOBMamba, m64: LOBMamba, x: torch.Tensor, budget: Budget,
                     global_width: float, domains: dict | None) -> dict:
    """The norm-propagation certificate with each rule on the windows and eps values of
    docs/certification/domains.json, against the attack on the same windows."""
    attack = model_experiment(m64, model, x, budget, local_eps=[])
    rows, traces = [], {}
    for j, e in enumerate(MODEL_EPS):
        row = {"eps": e, "attack_width_median": attack["per_eps"][j]["attack_width_median"]}
        traces[f"{e:g}"] = {}
        for rule in RULES:
            tr = trace_local(model, x.to(torch.float32), e, rule)
            ref = local_sensitivity_rule(model, x.to(torch.float32), e, rule)
            if not _same(tr["lipschitz"], ref["lipschitz"]):
                raise AssertionError("the stage trace no longer reproduces local_sensitivity_rule")
            w = tr["width"].to(DTYPE)
            row[rule] = _median(w)
            row[f"{rule}_finite"] = bool(torch.isfinite(w).all())
            row[f"{rule}_over_attack"] = row[rule] / row["attack_width_median"]
            traces[f"{e:g}"][rule] = {k: tr[k] for k in
                                      ("stages", "blocks", "first_non_finite_stage", "log10_by_group")}
        if domains is not None:
            match = [r for r in domains["rows"] if r["eps"] == e]
            if match:
                row["domains_json_attack"] = match[0]["attack"]
                row["domains_json_lipschitz"] = match[0]["lipschitz"]
        rows.append(row)

    divergence = {}
    for rule in RULES:
        bad = [r["eps"] for r in rows if not r[f"{rule}_finite"]]
        vacuous = [r["eps"] for r in rows
                   if not (math.isfinite(r[rule]) and r[rule] <= global_width)]
        first = min(bad) if bad else None
        divergence[rule] = {
            "first_non_finite_eps": first,
            "first_vacuous_eps": min(vacuous) if vacuous else None,
            "largest_useful_eps": max([r["eps"] for r in rows if r["eps"] not in vacuous],
                                      default=None),
            "first_non_finite_stage": (traces[f"{first:g}"][rule]["first_non_finite_stage"]
                                       if first is not None else None),
        }
    # The additive rule uses the tight timescale slope (_delta_tight), and domains.json the
    # looser one of lipschitz._delta, so on the bounded map it can only be smaller; on softplus
    # the two agree.
    at_most = None
    if domains is not None:
        pairs = [(r["additive"], r["domains_json_lipschitz"]) for r in rows
                 if "domains_json_lipschitz" in r]
        at_most = all(
            (a == b) or (math.isnan(a) and math.isnan(b)) or math.isinf(b)
            or (math.isfinite(a) and a <= b * (1 + 1e-6))
            for a, b in pairs
        ) if pairs else None
    return {
        "windows": int(x.shape[0]),
        "seq_len": int(x.shape[1]),
        "global_certified_width": global_width,
        "rows": rows,
        "divergence": divergence,
        "additive_at_most_domains_json": at_most,
        "traces": traces,
    }


def compare_checkpoints(paths: dict[str, Path]) -> dict:
    """How far apart the two arms' weights are outside the timescale parameters.

    dt_proj and the timescale rows of x_proj carry the parametrisation, so they are left out.
    """
    ckpts = {arm: torch.load(p, map_location="cpu", weights_only=False) for arm, p in paths.items()}
    (a, ca), (b, cb) = ckpts.items()
    sa, sb = ca["model_state_dict"], cb["model_state_dict"]
    r = int(ca["architecture"]["dt_rank"])
    worst, where = 0.0, None
    for key in sa:
        if "dt_proj" in key:
            continue
        ta, tb = sa[key].double(), sb[key].double()
        if "x_proj" in key:
            ta, tb = ta[r:], tb[r:]
        rel = float((ta - tb).abs().max() / ta.abs().max().clamp(min=1e-30))
        if rel > worst:
            worst, where = rel, key
    steps = {arm: int(c["training"]["step"]) for arm, c in ckpts.items()}
    budget = {arm: int(c["training"]["max_steps"]) for arm, c in ckpts.items()}
    when = (f"Both checkpoints are the weights at training step {steps[a]}" if steps[a] == steps[b]
            else f"The checkpoints are the weights at training steps {steps[a]} ({a}) and "
                 f"{steps[b]} ({b})")
    close = worst <= 0.05 and steps[a] == steps[b]
    note = (
        f"{when} of {budget[a]}, where validation rank IC peaked, from the same seed. "
        "Outside the timescale parameters (`dt_proj` and the timescale rows of `x_proj`) their "
        f"weights differ by at most {100 * worst:.2g} percent of each tensor's largest entry "
        f"(`{where}`). "
        + ("The attacked sensitivities of the two arms are therefore close, and the arms differ "
           "mainly in the bounds, where the timescale parametrisation enters."
           if close else
           "The arms therefore differ in their weights as well as in the timescale "
           "parametrisation, so a difference between them, attacked or certified, is not due to "
           "the parametrisation alone.")
    )
    return {"selected_at_step": steps, "max_steps": budget,
            "max_relative_difference_outside_timescale": worst, "where": where, "note": note}


def run_arm(arm: str, model: LOBMamba, dm: LOBDataModule, ends: list[int], lengths: list[int],
            domain_x: torch.Tensor, domains: dict | None, seed: int) -> dict:
    m64 = float64_copy(model)
    incs = [incremental_certificate(b) for b in model.layers]
    out_box = certify_output_range(model)
    global_width = float(out_box.hi - out_box.lo)
    layers = [{"layer": li, "certified": certified_block(inc, lengths), "lengths": {}}
              for li, inc in enumerate(incs)]
    model_rows, timing = {}, {}

    for li_len, L in enumerate(lengths):
        budget = budget_for(L)
        x = held_out_windows(dm, L, ends)
        t0 = time.time()
        xns = block_inputs(m64, x.to(DTYPE))
        for li, (block, inc) in enumerate(zip(m64.layers, incs)):
            print(f"  {arm} L={L} block {li}: attack ...", flush=True)
            layers[li]["lengths"][str(L)] = block_experiment(
                block, inc, xns[li], budget, seed + 1000 * li + 100 * li_len)
        t1 = time.time()
        print(f"  {arm} L={L} model: attack ...", flush=True)
        model_rows[str(L)] = model_experiment(m64, model, x, budget, LENGTH_EPS)
        timing[str(L)] = {"block_seconds": t1 - t0, "model_seconds": time.time() - t1}

    print(f"  {arm}: local end-to-end certificate on the domains.json windows ...", flush=True)
    t0 = time.time()
    local = local_end_to_end(model, m64, domain_x, budget_for(domain_x.shape[1]), global_width,
                             domains if arm == "bounded" else None)
    timing["local_end_to_end_seconds"] = time.time() - t0
    grid = np.unique(np.round(np.logspace(math.log10(lengths[0]), math.log10(lengths[-1]), 120))
                     .astype(int)).tolist()
    curves = {
        "L": grid,
        "ours": float(incs[0].output_slope().max()),
        "no_floor": [float(incs[0].additive_output_slope(L, floor=False).max()) for L in grid],
        "certified_floor": [float(incs[0].additive_output_slope(L, floor=True).max()) for L in grid],
    }
    return {
        "_curves": curves,
        "global_certified_width": global_width,
        "layernorm": [layernorm_lipschitz(b.norm) for b in model.layers],
        "block": {"layers": layers},
        "model": {"lengths": model_rows},
        "local_end_to_end": local,
        "seconds": timing,
    }


# --- the figure -------------------------------------------------------------------------------


def _plot(v: float) -> float:
    return v if (math.isfinite(v) and v > 0) else np.nan


def figure(arms: dict, lengths: list[int], path: Path, curves: dict) -> None:
    """Measured and certified sensitivity against length: block 0 above, the prediction below.

    curves holds, per arm, the data-free slopes of block 0 on a dense grid of lengths; they are
    closed forms in L, so the lines between the measured lengths are exact.
    """
    fig = plt.figure(figsize=FIGSIZE, layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.12, 1.0])
    ax_top = top.subplots(1, 2, sharey=True)
    ax_bot = bottom.subplots(1, 2, sharey=True)
    # the data-dependent series at the smallest delta, the linear regime the slopes describe
    k = BLOCK_DELTAS.index(min(BLOCK_DELTAS))
    e = LENGTH_EPS[0]
    j = MODEL_EPS.index(e)
    for col, (arm, a) in enumerate(arms.items()):
        c = curves[arm]
        rows = [a["block"]["layers"][0]["lengths"][str(L)] for L in lengths]
        ax = ax_top[col]
        ax.loglog(c["L"], c["no_floor"], color=OTHER, linewidth=1.6, linestyle="--",
                  label="certified, additive, no floor")
        ax.loglog(c["L"], c["certified_floor"], color=OTHER, linewidth=2.0,
                  label="certified, additive, certified floor")
        ax.loglog(c["L"], [c["ours"]] * len(c["L"]), color=OURS, linewidth=2.2,
                  label="certified, log-timescale, data-free")
        ax.loglog(lengths, [_plot(r["per_delta"][k]["local"]["log_delta"]["lip_max"]) for r in rows],
                  color=LOCAL, linewidth=1.4, marker="D", markersize=4,
                  label="local, log-timescale, maximum")
        ax.loglog(lengths, [_plot(r["per_delta"][k]["local"][REPORTED_RULE]["lip_max"])
                            for r in rows],
                  color=LOCAL, linewidth=1.2, linestyle="--", marker="D", markersize=4,
                  markerfacecolor="white", label=f"local, {RULE_NAMES[REPORTED_RULE]}")
        ax.loglog(lengths, [r["per_delta"][k]["slope_max"] for r in rows], color=INK,
                  linewidth=1.2, marker="o", markersize=5, label="attack, worst position")
        ax.loglog(lengths, [r["per_delta"][k]["last_slope_max"] for r in rows], color=INK,
                  linewidth=1.0, linestyle=":", marker="o", markersize=5, markerfacecolor="white",
                  label="attack, last position")
        ax.text(0.04, 0.96, f"{arm} arm, block 0\nattack and local bound at "
                + r"$\delta = 10^{-6}$", transform=ax.transAxes, va="top", fontsize=8.5,
                color=INK, linespacing=1.3)
        gap = max(abs(math.log10(a_ / b_)) for a_, b_ in zip(c["no_floor"], c["certified_floor"]))
        if gap < 0.05:
            # the certified floor is far below 1/(lam L), so the two additive curves coincide;
            # the note goes below the line, clear of the panel label
            i = (3 * len(c["L"])) // 5
            ax.annotate("additive, with or\nwithout the floor", xy=(c["L"][i], c["no_floor"][i]),
                        xytext=(c["L"][i], c["no_floor"][i] / 60), fontsize=7.5,
                        color=INK, ha="center", va="top",
                        arrowprops={"arrowstyle": "-", "color": INK, "linewidth": 0.6})

        ax = ax_bot[col]
        m = a["model"]["lengths"]
        ax.loglog(lengths, [_plot(m[str(L)]["per_eps"][j]["additive"]["width_median"] / (2 * e))
                            for L in lengths],
                  color=OTHER, linewidth=2.0, marker="s", markersize=4,
                  label="local, additive")
        ax.loglog(lengths, [_plot(m[str(L)]["per_eps"][j]["log_delta"]["width_median"] / (2 * e))
                            for L in lengths],
                  color=LOCAL, linewidth=1.4, marker="D", markersize=4,
                  label="local, log-timescale, maximum")
        ax.loglog(lengths, [_plot(m[str(L)]["per_eps"][j][REPORTED_RULE]["width_median"]
                                  / (2 * e)) for L in lengths],
                  color=LOCAL, linewidth=1.2, linestyle="--", marker="D", markersize=4,
                  markerfacecolor="white", label=f"local, {RULE_NAMES[REPORTED_RULE]}")
        ax.loglog(lengths, [m[str(L)]["per_eps"][j]["attack_width_median"] / (2 * e)
                            for L in lengths],
                  color=INK, linewidth=1.2, marker="o", markersize=5, label="attack")
        colours = {"additive": OTHER, "log_delta": LOCAL, REPORTED_RULE: LOCAL}
        for rule in colours:
            vals = [m[str(L)]["per_eps"][j][rule]["width_median"] / (2 * e) for L in lengths]
            bad = [L for L, v in zip(lengths, vals) if not math.isfinite(v)]
            good = [(L, v) for L, v in zip(lengths, vals) if math.isfinite(v)]
            if bad and good:
                # a bound that stops being finite has no point to draw; say where it stops
                L_last, v_last = good[-1]
                ax.annotate(f"{SHORT[rule]} bound infinite\nfrom L = {bad[0]}", xy=(L_last, v_last),
                            xytext=(bad[0], v_last / 100), fontsize=7.5, color=INK,
                            ha="center", va="top",
                            arrowprops={"arrowstyle": "-", "color": colours[rule], "linewidth": 0.8})
        ax.text(0.04, 0.96, f"{arm} arm, prediction\n" + r"$\kappa = 10^{-6}$",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK, linespacing=1.3)
        ax.set_xlabel("sequence length $L$")
    for ax in (*ax_top, *ax_bot):
        ax.set_xlim(lengths[0] / 1.5, lengths[-1] * 1.5)
    for axes in (ax_top, ax_bot):
        lo, hi = axes[0].get_ylim()
        axes[0].set_ylim(lo, hi * 100)                # room for the two-line panel label
    ax_top[0].set_ylabel(r"block: $\|F(x) - F(\tilde x)\|_\infty / \delta$")
    ax_bot[0].set_ylabel(r"prediction: width $/\, 2\kappa$")
    for sub, ax in ((top, ax_top[0]), (bottom, ax_bot[0])):
        handles, labels = ax.get_legend_handles_labels()
        sub.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False,
                   fontsize=7.5)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


# --- the markdown twin ------------------------------------------------------------------------


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


def _ratio(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)) or b == 0:
        return math.inf if (math.isinf(a) and b > 0) else math.nan
    return a / b


def _span(values: list[float]) -> str:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return "not finite"
    lo, hi = fmt(min(finite)), fmt(max(finite))
    return lo if lo == hi else f"{lo} to {hi}"


def _block_rows(results: dict):
    """(arm, layer index, length, layer dict, length dict) for every block measurement."""
    for arm, a in results["arms"].items():
        for lay in a["block"]["layers"]:
            for L in results["data"]["lengths"]:
                yield arm, lay["layer"], L, lay, lay["lengths"][str(L)]


def _violations(results: dict) -> tuple[int, int, int]:
    data_free = local = model = 0
    for _, _, _, _, r in _block_rows(results):
        for pd in r["per_delta"]:
            data_free += pd["data_free_violations"]
            local += sum(pd["local"][rule]["violations"] for rule in RULES)
    for a in results["arms"].values():
        for m in a["model"]["lengths"].values():
            for row in m["per_eps"]:
                model += sum(row[rule]["violations"] for rule in RULES if rule in row)
    return data_free, local, model


def markdown(results: dict) -> str:
    data, arms = results["data"], results["arms"]
    lengths = data["lengths"]
    L0, L1 = lengths[0], lengths[-1]
    k0_delta = min(BLOCK_DELTAS)
    k0 = BLOCK_DELTAS.index(k0_delta)
    k2 = BLOCK_DELTAS.index(1e-2)
    j6 = MODEL_EPS.index(LENGTH_EPS[0])
    budget = results["budget"]
    n_win = data["windows"]
    md = [
        "# Attacking the incremental bound",
        "",
        f"Every attacked number comes from {n_win} held-out windows of `{data['file']}`, the test "
        f"split, rows {data['test_rows'][0]} to {data['test_rows'][1] - 1}. The windows end at "
        f"the same rows ({', '.join(str(e) for e in data['window_end_rows'])}) at every length, "
        "so a longer window adds history before the same final tick. An attacked value is "
        "reached, so it is a lower bound on the true sensitivity, and every bound must lie above "
        "it. The attacks run in float64, and every reported deviation is scored again with the "
        "reference forward pass.",
        "",
        "Per window and block, the attack ranks every (position, output coordinate) pair with "
        f"{budget[str(L0)]['probes']} random sign probes, takes the exact gradient row of the "
        "best-ranked pairs, and then runs sign-gradient ascent for "
        f"{budget[str(L0)]['block_steps']} steps on the largest change over all positions. It "
        "starts from the gradient signs of the two best rows and from one random corner, and a "
        "fourth restart targets the last position. The number of ranked candidates is "
        + ", ".join(f"{budget[str(L)]['candidates']} at L = {L}" for L in lengths)
        + ". The model attack starts from the sign of the input gradient, for each sign of the "
        f"prediction, and runs {budget[str(L0)]['model_steps']} steps.",
        "",
        results["checkpoints"]["note"],
        "",
        "## Block output against sequence length",
        "",
        "The perturbation is any point within `delta` of the block's realised normalised input in "
        "sup-norm. The attacked slope is the largest `|F(xn + xi) - F(xn)|` over positions, output "
        "coordinates and windows, divided by `delta`. The first two columns are at "
        f"`delta = {k0_delta:g}`, where the block is linear to the digits shown, so they are "
        "comparable with the certified slopes; the third is the largest over the four values of "
        "`delta`, which includes the nonlinear `delta = 0.1`. The certified columns are the "
        "data-free slopes of `incremental.json`, with the cell theorem's time-warp term and with "
        "the common-start one, and with the additive ones taken at the window's length. The local "
        "columns are the local block bound around each window at "
        f"`delta = {k0_delta:g}`, maximised over windows, positions and coordinates, with the "
        "additive recurrence step and with the three log-timescale steps: the running maximum of "
        "the local corollary, `E_t = max(E_{t-1}, dc_t + (e^rho_t - 1)|h_{t-1} - c_t|)`; the "
        "recursion in its proof, `E_t = K_t + a*_t (E_{t-1} - K_t)` with `K_t` the second argument "
        "of that maximum and `a*_t` the perturbed transition at its worst admissible value; and "
        "the endpoint form, the reported one, "
        "`E_t = max_s [s E_{t-1} + (1 - s) dc_t + |Abar_t - s| |h_{t-1} - c_t|]` over the two ends "
        "of the interval that holds the perturbed transition `s`, the intersection of the "
        "log-timescale interval with the one the additive bound on `|dt - dt~|` gives.",
        "",
    ]
    rows = []
    for arm, li, L, lay, r in _block_rows(results):
        cert = lay["certified"]
        small = r["per_delta"][k0]
        rows.append([
            arm, li, L, small["slope_max"], small["last_slope_max"], r["measured_slope_max"],
            cert["ours_slope_max"], cert["ours_common_start_slope_max"],
            cert["additive_slope_max"][str(L)]["no_floor"],
            cert["additive_slope_max"][str(L)]["certified_floor"],
            *[small["local"][rule]["lip_max"] for rule in RULES],
        ])
    md += [
        table(rows, ["arm", "block", "L", "attack, worst position", "attack, last position",
                     "attack, largest over delta", "data-free, log-timescale",
                     "data-free, common start", "additive, no floor", "additive, certified floor",
                     *[f"local, {SHORT[rule]}" for rule in RULES]]),
        "",
    ]

    sentences = []
    growth_all, growth_last = [], []
    for arm, a in arms.items():
        for lay in a["block"]["layers"]:
            s0 = lay["lengths"][str(L0)]["per_delta"][k0]["slope_max"]
            s1 = lay["lengths"][str(L1)]["per_delta"][k0]["slope_max"]
            add0 = lay["certified"]["additive_slope_max"][str(L0)]["no_floor"]
            add1 = lay["certified"]["additive_slope_max"][str(L1)]["no_floor"]
            lasts = [lay["lengths"][str(L)]["per_delta"][k0]["last_slope_max"] for L in lengths]
            growth_all.append(s1 / s0)
            growth_last.append(lasts[-1] / lasts[0])
            sentences.append(
                f"On the {arm} arm, block {lay['layer']}, the attacked slope at "
                f"`delta = {k0_delta:g}` is {fmt(s0)} at L = {L0} and {fmt(s1)} at L = {L1}, a "
                f"factor of {fmt(s1 / s0)}, while the additive bound without a floor grows by a "
                f"factor of {fmt(add1 / add0)}. At the last position the attacked slope ranges "
                f"from {_span(lasts)}."
            )
    md += [" ".join(sentences), ""]

    ours_ratio = [lay["certified"]["ours_slope_max"] / r["per_delta"][k0]["slope_max"]
                  for _, _, _, lay, r in _block_rows(results)]
    common_ratio = [lay["certified"]["ours_common_start_slope_max"] / r["per_delta"][k0]["slope_max"]
                    for _, _, _, lay, r in _block_rows(results)]
    local_ratio = {rule: [r["per_delta"][k0]["local"][rule]["lip_max"] / r["per_delta"][k0]["slope_max"]
                          for _, _, _, _, r in _block_rows(results)] for rule in RULES}
    additive_ratio = [lay["certified"]["additive_slope_max"][str(L)]["no_floor"]
                      / r["per_delta"][k0]["slope_max"]
                      for _, _, L, lay, r in _block_rows(results) if L == L1]
    pos = [r["per_delta"][k0]["worst_position_from_end"] for _, _, _, _, r in _block_rows(results)]
    at_last = sum(p == 0 for p in pos)
    away = [float(p) for p in pos if p > 0]
    md += [
        f"At `delta = {k0_delta:g}` the worst position is the last one in {at_last} of {len(pos)} "
        "(arm, block, length) cases"
        + (f"; otherwise it lies {_span(away)} positions before the end of the window" if away else "")
        + ". The last-position column isolates a longer history acting on one position: from "
        f"L = {L0} to L = {L1} it grows by a factor of {_span(growth_last)}, against "
        f"{_span(growth_all)} over all positions, so the change with `L` comes partly from the "
        "attack finding a more sensitive position among more positions and partly from the longer "
        "history. Across arms, blocks and "
        "lengths the data-free slope, which is the same at every length, is "
        + _span(ours_ratio) + " times the attacked slope, and "
        + _span(common_ratio) + f" times with the common-start term. At L = {L1} the additive "
        "bound without a floor is "
        + _span(additive_ratio) + " times the attacked slope. The local bound is "
        + "; ".join(f"{_span(local_ratio[rule])} times the attacked slope with the "
                    f"{RULE_NAMES[rule]} rule" for rule in RULES) + ".",
        "",
        "Per `delta`:",
        "",
    ]
    rows = []
    for arm, li, L, lay, r in _block_rows(results):
        for pd in r["per_delta"]:
            rows.append([
                arm, li, L, pd["delta"], pd["slope_max"], pd["last_slope_max"],
                _ratio(pd["data_free_bound_max"], pd["attacked_max"]),
                *[_ratio(pd["local"][rule]["lip_max"], pd["slope_max"]) for rule in RULES],
                pd["worst_position_from_end"], pd["pgd_gain_max"],
            ])
    md += [
        table(rows, ["arm", "block", "L", "delta", "attacked slope", "attacked slope, last position",
                     "data-free bound / attacked",
                     *[f"local {SHORT[rule]} / attacked" for rule in RULES],
                     "worst position, from the end", "ascent gain over its start"]),
        "",
        "The data-free bound here is the arbitrary-perturbation reading of `incremental.json` at "
        "that `delta`, since the attacked point need not be a LayerNorm output. The ascent gain is "
        "the best value after the sign-gradient steps divided by the best starting value, so a "
        "gain of 1 means the gradient-sign start was already the best point found.",
        "",
    ]

    df_v, local_v, model_v = _violations(results)
    md += [
        "No attacked value exceeds its certificate. There are "
        f"{df_v} violations of the data-free bound, checked per output coordinate, "
        f"{local_v} of the local block bounds, checked per position and coordinate where they "
        f"are finite, and {model_v} of the local end-to-end bounds."
        if df_v + local_v + model_v == 0 else
        f"Violations found: {df_v} of the data-free bound, {local_v} of the local block bounds "
        f"and {model_v} of the local end-to-end bounds. A violation falsifies that bound.",
        "",
    ]
    rows = []
    for arm, a in arms.items():
        for lay in a["block"]["layers"]:
            for rule in RULES:
                rows.append([arm, lay["layer"], rule] + [
                    lay["lengths"][str(L)]["per_delta"][k2]["local"][rule]["lip_max"]
                    for L in lengths])
    md += [
        "The local block bound at `delta = 1e-2`, against the length:",
        "",
        table(rows, ["arm", "block", "rule", *[f"L = {L}" for L in lengths]]),
        "",
        "## The prediction against sequence length",
        "",
        "The spread of the prediction over an `eps` ball at the model input is what "
        "`scripts/08_domains.py` measures. Below it is divided by `2 eps`, so a constant row means "
        "a linear response, and it is the median over the windows. The last column is "
        "`|grad|_1` at the nominal input, the first-order value of every other column.",
        "",
    ]
    rows = []
    for arm, a in arms.items():
        for L in lengths:
            m = a["model"]["lengths"][str(L)]
            rows.append([arm, L, *[row["attack_width_median"] / (2 * row["eps"])
                                   for row in m["per_eps"]],
                         m["per_eps"][0]["first_order_width_median"] / (2 * m["per_eps"][0]["eps"])])
    md += [
        table(rows, ["arm", "L", *[f"eps = {e:g}" for e in MODEL_EPS], "first order"]),
        "",
    ]
    slopes = {arm: [a["model"]["lengths"][str(L)]["per_eps"][j6]["attack_width_median"]
                    / (2 * LENGTH_EPS[0]) for L in lengths] for arm, a in arms.items()}
    gaps = [abs(row["attack_width_median"] / row["first_order_width_median"] - 1.0)
            for a in arms.values() for m in a["model"]["lengths"].values()
            for row in m["per_eps"] if row["eps"] <= 1e-3]
    big = [row["attack_width_median"] / row["first_order_width_median"]
           for a in arms.values() for m in a["model"]["lengths"].values()
           for row in m["per_eps"] if row["eps"] == MODEL_EPS[-1]]
    md += [
        f"At `eps = {LENGTH_EPS[0]:g}` the attacked width divided by `2 eps` ranges "
        + " and ".join(f"from {_span(v)} on the {arm} arm" for arm, v in slopes.items())
        + f" across L = {', '.join(str(L) for L in lengths)}. Up to `eps = 1e-3` the attacked "
        f"width differs from the first-order width by at most {fmt(100 * max(gaps))} percent, "
        f"so the prediction is locally linear at these scales; at `eps = {MODEL_EPS[-1]:g}` the "
        f"attacked width is {_span(big)} times the first-order width.",
        "",
        "The local end-to-end bound against the length, as the median width over the windows:",
        "",
    ]
    rows, verdicts = [], []
    for arm, a in arms.items():
        for rule in RULES:
            for e in LENGTH_EPS:
                jj = MODEL_EPS.index(e)
                vals = [a["model"]["lengths"][str(L)]["per_eps"][jj][rule]["width_median"]
                        for L in lengths]
                rows.append([arm, SHORT[rule], e, *vals])
                if e != LENGTH_EPS[0]:
                    continue
                name = RULE_NAMES[rule]
                bad = [L for L, v in zip(lengths, vals) if not math.isfinite(v)]
                verdicts.append(
                    f"is non-finite with the {name} rule on the {arm} arm from L = {bad[0]} on"
                    if bad else
                    f"stays finite with the {name} rule on the {arm} arm, going from "
                    f"{fmt(vals[0])} at L = {L0} to {fmt(vals[-1])} at L = {L1}"
                )
    md += [
        table(rows, ["arm", "rule", "eps", *[f"L = {L}" for L in lengths]]),
        "",
        f"At `eps = {LENGTH_EPS[0]:g}` the local end-to-end bound " + "; it ".join(verdicts) + ".",
        "",
        "## The local end-to-end certificate on the windows of domains.json",
        "",
    ]
    for arm, a in arms.items():
        loc = a["local_end_to_end"]
        md += [f"{arm} arm: {loc['windows']} windows of length {loc['seq_len']}, the first of the "
               "unshuffled test loader, as `scripts/08_domains.py` takes them. The global certified "
               f"width of the prediction is {fmt(loc['global_certified_width'])}.", ""]
        rows = [[r["eps"], r["attack_width_median"], *[r[rule] for rule in RULES],
                 *[r[f"{rule}_over_attack"] for rule in RULES]] for r in loc["rows"]]
        md += [table(rows, ["eps", "attack width", *[SHORT[rule] for rule in RULES],
                            *[f"{SHORT[rule]} / attack" for rule in RULES]]), ""]
        parts = []
        for rule in RULES:
            dv = loc["divergence"][rule]
            first = dv["first_non_finite_eps"]
            text = (f"With the {RULE_NAMES[rule]} rule the bound is "
                    + (f"first non-finite at `eps = {fmt(first)}`, from the stage "
                       f"`{dv['first_non_finite_stage']}` on" if first is not None
                       else "finite at every `eps` here")
                    + (f", and first above the global certified width at "
                       f"`eps = {fmt(dv['first_vacuous_eps'])}`." if dv["first_vacuous_eps"] is not None
                       else ", and never above the global certified width."))
            if first is not None:
                blocks = loc["traces"][f"{first:g}"][rule]["blocks"]
                if rule == "additive":
                    b = max(blocks, key=lambda d: d["fraction_of_steps_growing"])
                    text += (f" There the step multiplier in block {b['layer']} reaches "
                             f"{fmt(b['step_multiplier_max'])}, and "
                             f"{fmt(100 * b['fraction_of_steps_growing'])} percent of the steps "
                             "enlarge the carried bound.")
                else:
                    stage = dv["first_non_finite_stage"] or ""
                    bad = int(stage[1]) if stage[:1] == "L" and stage[1:2].isdigit() else None
                    b = next((d for d in blocks if d["layer"] == bad), None)
                    if b is not None and b["rho_max"] is not None:
                        rho = b["rho_max"]
                        overflow = rule != "log_delta_endpoint" and rho > 88.72
                        text += (" The step never scales the carried bound up; there the largest "
                                 f"relative timescale perturbation in block {b['layer']} is "
                                 + ("not finite, because the sensitivity that reaches it is not."
                                    if not math.isfinite(rho) else
                                    f"`rho = {fmt(rho)}`"
                                    + (", beyond 88.7, where `e^rho - 1` overflows float32."
                                       if overflow else ".")))
            parts.append(text)
        useful = {rule: loc["divergence"][rule]["largest_useful_eps"] for rule in RULES}
        parts.append(
            "The largest `eps` at which the bound is below the global certified width is "
            + ", ".join((f"{useful[rule]:g}" if useful[rule] is not None else "none of the tested")
                        + f" with the {RULE_NAMES[rule]} rule" for rule in RULES)
            + "."
        )
        md += [" ".join(parts), ""]
        if loc.get("additive_at_most_domains_json") is not None:
            md += ["The additive rule here uses the tight slope of the timescale map, which on "
                   "the bounded map is smaller than the slope behind the `lipschitz` column of "
                   "`domains.json`; it is never above that column: "
                   + fmt(loc["additive_at_most_domains_json"]) + ".", ""]

    md += [
        "## Where the end-to-end bound comes from",
        "",
        "The domain carries one sensitivity per coordinate from the input, where it is 1, to the "
        "prediction. The stages form a chain, so the bound is the product of the stage ratios "
        "`max(after) / max(before)`, and its `log10` is the sum of theirs. The groups add those "
        "logs. LayerNorm is the stem, block and final norms. Feed-forward is every Linear, the "
        "convolution, every SiLU, the gate product and the head. Recurrence is the scan and its "
        "readout `C h + D u`. Residual is the skip connection. A negative entry is a stage whose "
        "largest sensitivity is smaller than its input's. Only `eps` values with a finite bound "
        "appear.",
        "",
    ]
    rows = []
    for arm, a in arms.items():
        for e, by_rule in a["local_end_to_end"]["traces"].items():
            for rule in RULES:
                g = by_rule[rule]["log10_by_group"]
                if g is None:
                    continue
                rows.append([arm, SHORT[rule], float(e), *[g[k] for k in GROUPS], sum(g.values())])
    md += [table(rows, ["arm", "rule", "eps", *GROUPS, "total, log10 of the bound"]), ""]

    first_arm = next(iter(arms))
    tr = arms[first_arm]["local_end_to_end"]["traces"][f"{LENGTH_EPS[0]:g}"]
    split = {}
    for rule in RULES:
        st = tr[rule]["stages"]
        split[rule] = (sum(s["log10_ratio"] for s in st if s["stage"].endswith("recurrence")),
                       sum(s["log10_ratio"] for s in st if s["stage"].endswith("readout")))
    attack6 = {arm: a["local_end_to_end"]["rows"][j6]["attack_width_median"]
               for arm, a in arms.items()}
    log_rules = [rule for rule in RULES if rule != "additive"]
    e2e = {rule: {arm: a["local_end_to_end"]["rows"][j6][f"{rule}_over_attack"]
                  for arm, a in arms.items()} for rule in log_rules}
    # the block-level and end-to-end comparison on the same windows: the attack windows of the
    # domains.json length
    L_ref = data["domains_seq_len"]
    e2e_same = {rule: {arm: _ratio(a["model"]["lengths"][str(L_ref)]["per_eps"][j6][rule]["width_median"],
                                   a["model"]["lengths"][str(L_ref)]["per_eps"][j6]["attack_width_median"])
                       for arm, a in arms.items() if str(L_ref) in a["model"]["lengths"]}
                for rule in log_rules}
    block_l = {rule: [r["per_delta"][k0]["local"][rule]["lip_max"] / r["per_delta"][k0]["slope_max"]
                      for _, _, L, _, r in _block_rows(results) if L == L_ref] for rule in log_rules}
    md += [
        f"On the {first_arm} arm at `eps = {LENGTH_EPS[0]:g}`, the recurrence group splits into "
        + "; ".join(f"{fmt(split[rule][0])} decades from the scans and {fmt(split[rule][1])} from "
                    f"the readouts with the {RULE_NAMES[rule]} rule" for rule in RULES)
        + ". The timescale rule changes the scan stages only; LayerNorm and the feed-forward "
        "stages differ between the rules only through the sensitivity that reaches them.",
        "",
        "The attacked sensitivity of the prediction on these windows is "
        + " and ".join(f"{fmt(v / (2 * LENGTH_EPS[0]))} on the {arm} arm" for arm, v in attack6.items())
        + f" (width over `2 eps` at `eps = {LENGTH_EPS[0]:g}`). The local end-to-end bound exceeds "
        "it by a factor of "
        + "; ".join(", ".join(f"{fmt(v)} on the {arm} arm" for arm, v in e2e[rule].items())
                    + f" with the {RULE_NAMES[rule]} rule" for rule in log_rules)
        + f". On the attack windows of length {L_ref} the factor is "
        + "; ".join(", ".join(f"{fmt(v)} on the {arm} arm" for arm, v in e2e_same[rule].items())
                    + f" with the {RULE_NAMES[rule]} rule" for rule in log_rules)
        + ". On those same windows, around one block's own normalised input at "
        f"`delta = {k0_delta:g}`, the same domain is "
        + "; ".join(f"{_span(block_l[rule])} times the attacked block slope with the "
                    f"{RULE_NAMES[rule]} rule" for rule in log_rules)
        + ", and there it covers the block's linear layers, convolution, gate and recurrence. "
        "The two measurements place most of the end-to-end excess outside the block interiors: "
        "in the LayerNorms, the stem and the head, and in chaining stage bounds that are each "
        "computed on their own.",
        "",
        f"Stage by stage on the {first_arm} arm at `eps = {LENGTH_EPS[0]:g}`, as the largest "
        "sensitivity over the windows, positions and coordinates:",
        "",
    ]
    rows = []
    for stages in zip(*(tr[rule]["stages"] for rule in RULES)):
        rows.append([stages[0]["stage"], stages[0]["group"],
                     *itertools.chain.from_iterable((s["lip_max"], s["log10_ratio"]) for s in stages)])
    md += [
        table(rows, ["stage", "group",
                     *itertools.chain.from_iterable((SHORT[rule], "log10 ratio") for rule in RULES)]),
        "",
        "Inside each recurrence, per block: `rho` is the largest relative timescale perturbation, "
        "`L_log_delta eps`. The step multiplier is the largest factor that one step applies to "
        "the carried bound: `Abar + |A| exp(-|A| dt_lo) L_dt eps` for the additive rule, which can "
        "exceed 1; 1 for the running maximum; and the slowest admissible perturbed transition "
        "`exp(-|A| dt e^-rho)`, below 1, for the recursive rule; and for the endpoint rule the "
        "upper end of the interval it maximises over, the slowest transition that both the log "
        "and the additive timescale budgets admit, at most the recursive rule's. The fraction "
        "growing counts the "
        "(cell, step) pairs with a multiplier above 1; the fraction raised counts those at which "
        "the carried bound increases at all. The last two columns are the largest state "
        "sensitivity with the timescale's own sensitivity and without it. The run without it is "
        "not a bound; it measures how much of the recurrence's growth comes through the "
        "timescale.",
        "",
    ]
    rows = []
    for arm, a in arms.items():
        for e, by_rule in a["local_end_to_end"]["traces"].items():
            for rule in RULES:
                for b in by_rule[rule]["blocks"]:
                    rows.append([arm, SHORT[rule], float(e), b["layer"], b["rho_max"],
                                 b["step_multiplier_max"], b["fraction_of_steps_growing"],
                                 b["fraction_of_steps_raised"],
                                 b["h_lip_max"], b["h_lip_max_without_timescale"]])
    md += [
        table(rows, ["arm", "rule", "eps", "block", "rho", "step multiplier",
                     "fraction of steps growing", "fraction of steps raised", "state sensitivity",
                     "without the timescale term"]),
        "",
        "## Runtime and what was not run",
        "",
        f"The run took {results['seconds_total'] / 60:.1f} minutes on "
        f"{results['provenance']['platform']}, on the CPU. Per arm and length, the block and "
        "model attacks took:",
        "",
        table([[arm, L, a["seconds"][str(L)]["block_seconds"], a["seconds"][str(L)]["model_seconds"]]
               for arm, a in arms.items() for L in lengths],
              ["arm", "L", "block attack, seconds", "model attack, seconds"]),
        "",
        "The block attack has one random-corner restart per `delta` and the model attack has "
        f"none. The same {n_win} windows are used at every length, and no further windows were "
        "run. The attack is a heuristic search, so every attacked value is a lower bound and the "
        "true worst case may be larger.",
        "",
        "## Scope",
        "",
        "The data-free and local bounds hold over the reals. The local end-to-end bound is "
        "evaluated in float32, as `domains.json` was, and the local block bound in float64. The "
        "data-free incremental certificate is per block, with respect to the normalised input. "
        "A data-free end-to-end counterpart would pass each LayerNorm with its global Lipschitz "
        "constant, which exists because `eps > 0`: `max|gamma| / sqrt(eps)` in l2 is "
        + "; ".join(f"{', '.join(fmt(ln['l2_global_bound']) for ln in a['layernorm'])} for the "
                    f"blocks of the {arm} arm" for arm, a in arms.items())
        + ", against about 1 at realised inputs (`incremental.json`). It is not computed here.",
        "",
    ]
    return "\n".join(md)


# --- main -------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Attack the incremental bound on held-out windows.")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints")
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--fig-dir", default=str(FIG_DIR))
    args = parser.parse_args()

    config = load_config()
    seed = int(config.get("seed", 1337))
    set_seed(seed)
    start = time.time()
    lengths = sorted(set(args.lengths))

    dm = LOBDataModule(config)
    ends = window_ends(dm, args.windows, max(lengths))
    domain_x = domain_windows(dm, 4)
    domains = json.loads(DOMAINS.read_text()) if DOMAINS.exists() else None

    arms = {}
    for i, (arm, name) in enumerate(ARMS.items()):
        print(f"{arm} arm ...", flush=True)
        path = Path(args.checkpoint_dir) / name
        model = load_model(path, arm)
        arms[arm] = {"checkpoint": str(path),
                     **run_arm(arm, model, dm, ends, lengths, domain_x, domains, seed + 100000 * i)}

    fig_dir, out_dir = Path(args.fig_dir), Path(args.out_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "sensitivity_vs_length.pdf"
    curves = {arm: a.pop("_curves") for arm, a in arms.items()}
    figure(arms, lengths, fig_path, curves)

    results = {
        "provenance": provenance(seed),
        "data": {
            "file": config["data"]["processed_file"],
            "split": "test",
            "test_rows": [int(dm.idx_test[0]), int(dm.idx_test[1])],
            "windows": len(ends),
            "window_end_rows": ends,
            "lengths": lengths,
            "domains_json_windows": "the first 4 windows of the unshuffled test loader, length "
                                    f"{int(domain_x.shape[1])}, as scripts/08_domains.py takes them",
            "domains_seq_len": int(domain_x.shape[1]),
        },
        "checkpoints": compare_checkpoints(
            {arm: Path(args.checkpoint_dir) / name for arm, name in ARMS.items()}),
        "perturbation": {
            "block": "any point within delta in sup-norm of the block's realised normalised input",
            "model": "any point within eps in sup-norm of the model input",
        },
        "deltas": BLOCK_DELTAS,
        "eps": MODEL_EPS,
        "length_eps": LENGTH_EPS,
        "budget": {str(L): asdict(budget_for(L)) for L in sorted(set(lengths) | {int(domain_x.shape[1])})},
        "attack": {
            "block": "per window: rank targets by random sign probes at the smallest delta; exact "
                     "gradient rows of the best-ranked targets; per delta, sign-gradient ascent "
                     "on the largest change over all positions from the two best rows' signs and "
                     "from a random corner, and on the largest change at the last position from "
                     "its best row; step size from 2 delta down to delta / 8; every best point "
                     "rescored with incremental.contribution in float64",
            "model": "per window and eps: sign-gradient ascent on +f and on -f from the input "
                     "gradient's sign; step size from 2 eps down to eps / 8; best points rescored "
                     "with the float64 reference forward pass; width = max - min including the "
                     "nominal prediction",
        },
        "arms": arms,
        "figure": str(fig_path),
        "seconds_total": time.time() - start,
    }
    md = markdown(results)
    (out_dir / "incremental_attack.json").write_text(json.dumps(results, indent=2))
    (out_dir / "incremental_attack.md").write_text(md)
    print("\n" + md)
    print(f"wrote {out_dir / 'incremental_attack.json'}, {out_dir / 'incremental_attack.md'} "
          f"and {fig_path}")


if __name__ == "__main__":
    main()
