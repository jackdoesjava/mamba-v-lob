"""Incremental bound for the selective scan: how far two runs of one block can drift apart.

invariant.py bounds one trajectory. This module bounds the difference between two, h driven by
one input and h~ by a perturbed one, uniformly in sequence length and with no floor on the
timescale. Per cell, with lam = |A|, a = exp(-lam dt) and c = B u / lam, collecting the
coefficients of h_{t-1}, h~_{t-1}, c_t and c~_t gives two forms of the error e = h - h~,

    e_t = a_t e_{t-1} + (1 - a_t)(c_t - c~_t) + (a_t - a~_t)(h~_{t-1} - c~_t)        (1)
    e_t = a~_t e_{t-1} + (1 - a~_t)(c_t - c~_t) + (a_t - a~_t)(h_{t-1} - c_t)        (2)

Measured in log(dt), a timescale perturbation moves the decay by a multiple of the weight the step
gives its equilibrium: |a - a~| <= (e^rho - 1)(1 - a) whenever |log dt - log dt~| <= rho
(`decay_gap_bound`), and the same with 1 - a~ in place of 1 - a. Form (1) with the invariant
radius M bounding h~ and c~ is a convex combination again, so

    |e_t| <= max(|e_{-1}|, sup|c - c~| + 2 M (e^rho - 1))     for every t      (`cell_bound`)

With a common start the time-warp term improves to its exact worst case 2 M tanh(rho/2)
(`cell_bound_common_start`). Form (2) puts the realised distance |h_{t-1} - c_t| of a nominal run
in place of 2 M and needs nothing about the perturbed run (`local_step`, and the recursion of its
proof, `local_step_recursive`). Form (2) without the lemma, maximised over the interval that holds
a~_t, is the endpoint form (`local_step_endpoint`), the step of the reported local certificate.

Bounding the decay additively, |a - a~| <= lam exp(-lam dt_min) |dt - dt~|, loses the (1 - a)
factor. That bound grows linearly in t when there is no floor, and with a floor it is
proportional to 1/(1 - exp(-lam dt_min)). Both forms are computed here as the baseline.

The same per-cell bound follows from Cao et al. (ICML 2026), Lemma 3.1 and Lemma B.2 with a
piecewise-linear time change, and it strengthens their Theorem 3.3(ii). The discrete induction is
Horbatko's Lemma 4.4 (arXiv:2604.18580) applied to the error recursion. Everything here holds over
the reals, not in floating point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mamba import LOBMamba, SelectiveSSMBlock, zoh_phi
from src.verification import intervals as iv
from src.verification.certify import _split
from src.verification.intervals import Interval
from src.verification.invariant import certify_block_state, equilibrium_box
from src.verification.lipschitz import (
    SILU_DERIV_MAX,
    Sens,
    _causal_conv,
    _delta,
    _sigmoid_deriv_sup,
    _silu_deriv_sup,
    add,
    layer_norm,
    linear,
    mul,
    silu,
)

# The certificate arithmetic runs in float64. The softplus floor on the trained model is about
# 2e-13, where float32 rounds exp(-lam dt) to exactly 1.
DTYPE = torch.float64

# Recurrence steps of the local domain. "additive" is the step of lipschitz.py with the timescale's
# slope taken over the local box (`_delta_tight`). "log_delta" is the local corollary, a running
# maximum. "log_delta_recursive" is the one-step inequality in that corollary's proof with the
# perturbed transition at its worst admissible value; it is never larger than the running
# maximum, and unlike it the carried bound decays. "log_delta_endpoint" maximises error form (2)
# itself over the interval that holds the perturbed transition, the intersection of the intervals
# given by rho_t and by the additive bound on |dt - dt~|; it is never larger than the recursion or
# than the additive step, and it is the reported rule. "additive_endpoint" does the same over the
# additive interval alone, the ablation that isolates what rho_t adds. The first four are kept for
# comparison.
COMPARISON_RULES = ("additive", "log_delta", "log_delta_recursive", "additive_endpoint")
REPORTED_RULE = "log_delta_endpoint"
RULES = (*COMPARISON_RULES, REPORTED_RULE)

# The two time-warp terms of the data-free certificate: the cell theorem, and the exact constant
# with a common start (both runs of the block start from h_{-1} = 0).
VARIANTS = ("cell", "common_start")


def _f64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(DTYPE)


def _f64_box(box: Interval) -> Interval:
    return Interval(_f64(box.lo), _f64(box.hi))


# --- the scalar lemma ---------------------------------------------------------------------


def relative_factor(rho):
    """kappa(rho) = e^rho - 1, the factor in |a - a~| <= kappa (1 - a)."""
    if isinstance(rho, torch.Tensor):
        return torch.expm1(rho)
    return math.expm1(rho)


def decay_gap(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """|exp(-x) - exp(-x e^s)|, evaluated without cancellation.

    exp(-x) - exp(-x~) = -exp(-x) expm1(-(x~ - x)) with x~ - x = x expm1(s). Subtracting the two
    exponentials directly loses digits as x falls and returns exactly 0 once exp(-x) rounds to 1,
    below about 1e-16 in float64 and 6e-8 in float32.
    """
    return torch.exp(-x) * torch.expm1(-x * torch.expm1(s)).abs()


def decay_gap_bound(x: torch.Tensor, rho) -> torch.Tensor:
    """(e^rho - 1)(1 - exp(-x)), the lemma's right-hand side.

    For x > 0 and x~ = x e^s with |s| <= rho, |exp(-x) - exp(-x~)| is at most this. With
    m(y) = (1 - exp(-y))/y, which decreases on (0, inf), exp(-x) - exp(-x~) = x~ m(x~) - x m(x).
    For s >= 0 that is at most (x~ - x) m(x) = (e^s - 1)(1 - exp(-x)); for s < 0 the same
    argument gives (1 - e^s)(1 - exp(-x)), and 1 - e^s <= e^|s| - 1. The ratio tends to
    e^rho - 1 as x -> 0 with s = rho, so the factor cannot be improved.
    """
    return relative_factor(rho) * -torch.expm1(-x)


# --- the cell theorem, its sharp constant and its local form ----------------------------------


def warp_term(M, rho, variant: str = "cell"):
    """The time-warp part of the state bound: 2 M (e^rho - 1), or 2 M tanh(rho/2) with a common
    start. The second is smaller by the factor e^rho + 1, which tends to 2 as rho -> 0."""
    if variant == "cell":
        return 2.0 * M * relative_factor(rho)
    if variant == "common_start":
        t = torch.tanh(rho / 2.0) if isinstance(rho, torch.Tensor) else math.tanh(rho / 2.0)
        return 2.0 * M * t
    raise ValueError(f"unknown variant: {variant!r}")


def cell_bound(dc, M, rho, e0=0.0):
    """sup_t |h_t - h~_t| <= max(|e_{-1}|, sup|c - c~| + 2 M (e^rho - 1)).

    Premises: |c~_t| <= M and |h~_{t-1}| <= M for every t, which the invariant gives once
    |h~_{-1}| <= M, and |log dt_t - log dt~_t| <= rho for every t. The step is
    |e_t| <= a_t |e_{t-1}| + (1 - a_t) K with K the second argument of the max, a convex
    combination, so the bound holds at every t with no lower bound on dt.
    """
    K = torch.as_tensor(dc + warp_term(M, rho, "cell"), dtype=DTYPE)
    return torch.maximum(torch.as_tensor(e0, dtype=DTYPE).abs(), K)


def cell_bound_common_start(dc, M, rho):
    """sup_t |h_t - h~_t| <= sup|c - c~| + 2 M tanh(rho/2), when h_{-1} = h~_{-1}.

    The premises of `cell_bound`, plus a common start. Split h - h~ = (h - h^) + (h^ - h~), with
    h^ driven by c~ on the nominal timescales: the first part is at most sup|c - c~| by the
    convex-combination argument, and the second, the time-warp part, is at most 2 M tanh(rho/2),
    because the pair (h^, h~)/M stays in a convex region whose diagonal width is that. A two-step
    chattering input approaches the constant, so it cannot be reduced.
    """
    return torch.as_tensor(dc + warp_term(M, rho, "common_start"), dtype=DTYPE)


def _warp_local(kappa: torch.Tensor, gap: torch.Tensor) -> torch.Tensor:
    # kappa * |h_{t-1} - c_t|, with 0 where the distance is 0 even if kappa overflowed
    return torch.where(gap > 0, kappa * gap, torch.zeros_like(gap))


def local_step(E_prev, kappa, dc, gap):
    """One step of the local corollary along a nominal run (h, c):

        E_t = max(E_{t-1}, dc_t + kappa_t |h_{t-1} - c_t|)

    with E_{t-1} >= |e_{t-1}|, dc_t >= |c_t - c~_t|, kappa_t = e^{rho_t} - 1 for
    |log dt_t - log dt~_t| <= rho_t, and the nominal distance |h_{t-1} - c_t|. Form (2) of the
    module docstring and the lemma with the two runs exchanged, |a - a~| <= kappa (1 - a~), give
    |e_t| <= a~_t |e_{t-1}| + (1 - a~_t)(dc_t + kappa_t |h_{t-1} - c_t|), and a~_t is in (0, 1).
    Valid for every rho_t, and nothing about the perturbed run is assumed.
    """
    return torch.maximum(E_prev, dc + _warp_local(kappa, gap))


def local_step_recursive(E_prev, x, rho, dc, gap):
    """The inequality in the proof of `local_step`, with a~_t at its worst admissible value.

        E_t = K_t + a*_t (E_{t-1} - K_t),     K_t = dc_t + (e^{rho_t} - 1) |h_{t-1} - c_t|

    Here x = lam dt_t on the nominal run. |log dt_t - log dt~_t| <= rho_t puts a~_t = exp(-lam dt~_t)
    in [exp(-x e^rho), exp(-x e^-rho)], and K_t + a~_t (E_{t-1} - K_t) is affine in a~_t, so its
    largest value takes a*_t at the upper end when E_{t-1} >= K_t and at the lower end otherwise.
    The result is never larger than max(E_{t-1}, K_t), and with rho = 0 it is the exact recursion
    of the forcing term.
    """
    K = dc + _warp_local(torch.expm1(rho), gap)
    slowest = torch.exp(-x * torch.exp(-rho))
    fastest = torch.exp(-x * torch.exp(rho))
    a = torch.where(E_prev >= K, slowest, fastest)
    out = K + a * (E_prev - K)
    # an infinite input is no bound, and inf - inf or 0 * inf must not turn it into one
    return torch.where(torch.isfinite(E_prev) & torch.isfinite(K), out, torch.full_like(out, math.inf))


def timescale_offsets(x, rho, dx=None) -> tuple[torch.Tensor, torch.Tensor]:
    """The interval of x~ = lam dt~ around the nominal x = lam dt, as offsets (up, down):
    x~ lies in [x - down, x + up].

    |log dt - log dt~| <= rho gives x~ in [x e^-rho, x e^rho], so up = x (e^rho - 1) and
    down = x (1 - e^-rho). An additive bound |dt - dt~| <= beta, passed as dx = lam beta, gives
    x~ in [x - dx, x + dx], and dt~ > 0 gives x~ >= 0. The interval is the intersection, so up is
    the smaller of the two upward offsets and down the smallest of the downward ones and x. Each
    offset is formed directly rather than as a difference of two ends, so it keeps its relative
    precision when it is far below x.
    """
    x = torch.as_tensor(x)
    rho = torch.as_tensor(rho, dtype=x.dtype)
    pos = x > 0
    # x = 0 with rho = inf would give 0 * inf; the interval is then the point 0
    up = torch.where(pos, x * torch.expm1(rho), torch.zeros_like(x))
    down = x * -torch.expm1(-rho)
    if dx is not None:
        dx = torch.as_tensor(dx, dtype=x.dtype)
        up = torch.minimum(up, dx)
        down = torch.minimum(down, dx)
    return up, torch.minimum(down, x)


def local_step_endpoint(E_prev, x, rho, dc, gap, dx=None):
    """The endpoint form of the local corollary: error form (2), maximised over the transitions
    the perturbed run can take.

    Form (2) with |e_{t-1}| <= E_{t-1}, |c_t - c~_t| <= dc_t and the nominal distance
    gap_t = |h_{t-1} - c_t| gives, with a = exp(-x) and the perturbed transition s = a~_t,

        |e_t| <= f(s) = s E_{t-1} + (1 - s) dc_t + |a - s| gap_t.

    f is convex in s, the sum of two affine terms and |a - s| times a constant, so its largest
    value over the interval I_t that holds a~_t is at one of the two ends, and

        E_t = max(f(exp(-(x + up))), f(exp(-(x - down))))

    with (up, down) from `timescale_offsets`. I_t is the intersection of the interval given by
    rho_t and, when dx = lam beta_t is passed, the one given by |dt - dt~| <= beta_t, so the step
    uses whichever is tighter in each cell. The ends enter through 1 - s and |a - s| formed with
    expm1, a - exp(-(x + up)) = exp(-x)(1 - exp(-up)) and exp(-(x - down)) - a =
    exp(-(x - down))(1 - exp(-down)), so a timescale far below 1/lam is not lost to rounding.

    Properties, all over the reals. (i) The lemma with the runs exchanged gives
    |a - s| <= (e^rho - 1)(1 - s) on the rho interval, so f(s) <= K_t + s (E_{t-1} - K_t) there,
    and E_t is never larger than `local_step_recursive`; with gap = 0 and no dx the two are
    equal, and with rho = 0 both are the exact recursion of the forcing term. (ii) |a - s| <= 1,
    so E_t <= E_{t-1} + dc_t + gap_t is finite whenever its inputs are, for every rho_t; e^rho
    overflowing makes up infinite and the lower end 0, which f takes as it is. (iii) For a given
    (E_{t-1}, dc_t, gap_t, I_t) the step is attained: at the maximising end, e_{t-1} and
    c_t - c~_t of the sign of (a - s)(h_{t-1} - c_t) give |e_t| = E_t.
    """
    up, down = timescale_offsets(x, rho, dx)
    fast = torch.exp(-(x + up))                       # the lower end of I_t
    slow = torch.exp(-(x - down))                     # the upper end
    f_fast = fast * E_prev + -torch.expm1(-(x + up)) * dc + torch.exp(-x) * -torch.expm1(-up) * gap
    f_slow = slow * E_prev + -torch.expm1(-(x - down)) * dc + slow * -torch.expm1(-down) * gap
    out = torch.maximum(f_fast, f_slow)
    ok = torch.isfinite(torch.as_tensor(E_prev)) & torch.isfinite(torch.as_tensor(dc))
    return torch.where(ok, out, torch.full_like(out, math.inf))


def euler_gain(x: torch.Tensor) -> torch.Tensor:
    """g(x) = x / (1 - exp(-x)), with g(0) = 1.

    Exponential-Euler, the discretisation of released Mamba-1/2, uses Bbar = dt B, so its forcing
    is c = g(lam dt) B u / lam. d log g / d log x = 1 - x/(e^x - 1) lies in [0, 1), so g inherits
    the timescale's log-perturbation: g(x~)/g(x) lies in [e^-rho, e^rho].
    """
    x = torch.as_tensor(x, dtype=DTYPE)
    safe = torch.where(x > 0, x, torch.ones_like(x))
    return torch.where(x > 0, safe / -torch.expm1(-safe), torch.ones_like(x))


def euler_cell_bound(dc_bu, M0, rho, x_hi):
    """The cell theorem under exponential-Euler. This repository's block uses the exact ZOH.

    With b = B u / lam, |b| <= M0 and |b - b~| <= dc_bu, the forcing c = g(x) b satisfies
    |c| <= g(x_hi) M0 and |c - c~| <= g(x_hi) (dc_bu + kappa M0), so the bound is
    g(x_hi) (dc_bu + 3 kappa M0), where x_hi = lam sup dt. It needs that ceiling and no floor.
    """
    kappa = relative_factor(rho)
    return euler_gain(x_hi) * (dc_bu + 3.0 * kappa * M0)


# --- timescale slopes -------------------------------------------------------------------------


def softplus_log_slope(z: torch.Tensor) -> torch.Tensor:
    """d log softplus(z) / dz = sigmoid(z) / softplus(z), in (0, 1).

    The ratio is below 1 because log(1 + y) > y/(1 + y), and it decreases in z because
    log(1 + y) < y. Below z = -40 it is 1 - O(e^z), within 1e-17 of 1, and it is returned as 1,
    its supremum, rather than as a ratio of two numbers that underflow.
    """
    r = torch.sigmoid(z) / F.softplus(z)
    r = torch.where(torch.isfinite(r), r, torch.ones_like(r)).clamp(max=1.0)
    return torch.where(z < -40.0, torch.ones_like(r), r)


def log_delta_slope(block: SelectiveSSMBlock, z: Interval) -> torch.Tensor:
    """sup over the box of |d log(dt) / dz|, per channel.

    softplus: decreasing in z, so the supremum is at the lower end, and it is never above 1.
    bounded: log dt = log dt_min + sigmoid(z) ln(dt_max/dt_min), so the slope is
    sigmoid'(z) ln(dt_max/dt_min) <= ln(dt_max/dt_min)/4. The clamp is 1-Lipschitz in log dt.
    """
    lo, hi = _f64(z.lo), _f64(z.hi)
    if block.dt_parametrisation == "bounded":
        return math.log(block.dt_max / block.dt_min) * _sigmoid_deriv_sup(lo, hi)
    return softplus_log_slope(lo)


def log_delta_slope_global(block: SelectiveSSMBlock) -> float:
    """The supremum of |d log(dt) / dz| over all z: 1 for softplus, ln(dt_max/dt_min)/4 bounded."""
    if block.dt_parametrisation == "bounded":
        return math.log(block.dt_max / block.dt_min) / 4.0
    return 1.0


def _bounded_delta_slope(block: SelectiveSSMBlock, s: torch.Tensor) -> torch.Tensor:
    lr = math.log(block.dt_max / block.dt_min)
    return block.dt_min * lr * torch.exp(lr * s) * s * (1.0 - s)


def _bounded_delta_peak(block: SelectiveSSMBlock) -> float:
    # d/ds [r^s s (1 - s)] has the sign of -lr s^2 + (lr - 2) s + 1, which is positive at s = 0
    # and negative at s = 1, so the slope is unimodal in s = sigmoid(z) and peaks at the root
    lr = math.log(block.dt_max / block.dt_min)
    return (lr - 2.0 + math.sqrt(lr * lr + 4.0)) / (2.0 * lr)


def delta_slope(block: SelectiveSSMBlock, z: Interval) -> torch.Tensor:
    """sup over the box of |d dt / dz|, the additive baseline's sensitivity of dt itself.

    softplus: softplus' = sigmoid, increasing, so the upper end. bounded:
    dt' = dt_min lr r^s s (1 - s) with s = sigmoid(z) and lr = ln r, evaluated at the peak in s
    clamped into the box. lipschitz.py bounds the same derivative by dt_max lr / 4, about four
    times larger at the trained range; the baseline is given the tight one.
    """
    lo, hi = _f64(z.lo), _f64(z.hi)
    if block.dt_parametrisation == "softplus":
        return torch.sigmoid(hi)
    peak = torch.full_like(lo, _bounded_delta_peak(block))
    s = torch.clamp(peak, min=torch.sigmoid(lo), max=torch.sigmoid(hi))
    return _bounded_delta_slope(block, s)


def delta_slope_global(block: SelectiveSSMBlock) -> float:
    if block.dt_parametrisation == "softplus":
        return 1.0
    s = torch.tensor(_bounded_delta_peak(block), dtype=DTYPE)
    return float(_bounded_delta_slope(block, s))


def weight_free_factor(block: SelectiveSSMBlock) -> float | None:
    """sup|dt'| / (dt_min sup|(log dt)'|) over all z, for the bounded parametrisation.

    To first order, the floored additive time-warp term over the cell theorem's is this factor
    times x/(e^x - 1) with x = lam dt_min, once the slopes are taken over all z. It depends on
    dt_max/dt_min only: 25.80 at a ratio of 100. The softplus parametrisation has no positive
    dt_min, so there is no such factor; the certified floor takes its place.
    """
    if block.dt_parametrisation != "bounded":
        return None
    return delta_slope_global(block) / (block.dt_min * log_delta_slope_global(block))


@torch.no_grad()
def timescale_floor(block: SelectiveSSMBlock, z: Interval) -> torch.Tensor:
    """Per-channel inf dt over the box, evaluated in float64.

    In float32 the softplus floor survives, about 2e-13 on the trained model, but
    exp(-lam dt_min) rounds to 1, so 1 - exp(-lam dt_min) would be 0.
    """
    return block.delta_from_pre(_f64(z.lo))


@torch.no_grad()
def timescale_ceiling(block: SelectiveSSMBlock, z: Interval) -> torch.Tensor:
    """Per-channel sup dt over the box, evaluated in float64."""
    return block.delta_from_pre(_f64(z.hi))


# --- the data-free block certificate ----------------------------------------------------------


@torch.no_grad()
def block_boxes(block: SelectiveSSMBlock, tight_layernorm: bool = True) -> dict[str, Interval]:
    """The boxes certify_block computes on the way, including the ones it does not return.

    Built in the same order with the same functions, so u, B, C and delta agree with
    certify_block exactly; tests/test_incremental.py checks that.
    """
    if tight_layernorm:
        in_box = iv.layernorm_linear(block.norm, block.in_proj)
    else:
        in_box = iv.affine(iv.layernorm(block.norm), block.in_proj.weight, block.in_proj.bias)
    u_pre, gate_pre = _split(in_box, [block.d_inner, block.d_inner])

    depthwise = block.conv1d.groups == block.conv1d.in_channels
    conv = (iv.depthwise_conv1d if depthwise else iv.dense_conv1d)(u_pre, block.conv1d)
    u = iv.silu(conv)

    proj = iv.affine(u, block.x_proj.weight, block.x_proj.bias)
    dt_pre_raw, B, C = _split(proj, [block.dt_rank, block.d_state, block.d_state])
    z = iv.affine(dt_pre_raw, block.dt_proj.weight, block.dt_proj.bias)
    return {
        "u_pre": u_pre,
        "gate_pre": gate_pre,
        "conv": conv,
        "u": u,
        "B": B,
        "C": C,
        "z": z,
        "delta": iv.delta_box(block, z),
    }


@torch.no_grad()
def block_lipschitz(
    block: SelectiveSSMBlock, boxes: dict[str, Interval] | None = None
) -> dict[str, torch.Tensor]:
    """Per-coordinate Lipschitz constants with respect to the block's normalised input.

    A perturbation of size at most delta at every coordinate and position of the normalised input
    moves coordinate i of each quantity by at most L_i delta. The convolution reads d_conv
    positions, each perturbed independently, so its constant is the kernel's l1 norm times the
    input's. x_proj and dt_proj meet with nothing in between, so they are composed before taking
    absolute values. Derivative suprema are taken over `boxes`, which both trajectories must lie
    in; with boxes=None they are the global suprema, valid for any input.
    """
    d = block.d_inner
    W_in = _f64(block.in_proj.weight)
    L_u_pre = W_in[:d].abs().sum(1)
    L_gate_pre = W_in[d:].abs().sum(1)

    w = _f64(block.conv1d.weight)
    if block.conv1d.groups == block.conv1d.in_channels:
        L_conv = w.squeeze(1).abs().sum(1) * L_u_pre
    else:
        L_conv = w.abs().sum(2) @ L_u_pre

    if boxes is None:
        silu_u = torch.full_like(L_conv, SILU_DERIV_MAX)
        silu_gate = torch.full_like(L_gate_pre, SILU_DERIV_MAX)
    else:
        silu_u = _silu_deriv_sup(_f64(boxes["conv"].lo), _f64(boxes["conv"].hi))
        silu_gate = _silu_deriv_sup(_f64(boxes["gate_pre"].lo), _f64(boxes["gate_pre"].hi))
    L_u = silu_u * L_conv

    W_x = _f64(block.x_proj.weight)
    r, n = block.dt_rank, block.d_state
    L_B = W_x[r : r + n].abs() @ L_u
    L_C = W_x[r + n :].abs() @ L_u
    W_z = _f64(block.dt_proj.weight) @ W_x[:r]
    L_z = W_z.abs() @ L_u

    if boxes is None:
        log_slope = torch.full_like(L_z, log_delta_slope_global(block))
        dt_slope = torch.full_like(L_z, delta_slope_global(block))
    else:
        log_slope = log_delta_slope(block, boxes["z"])
        dt_slope = delta_slope(block, boxes["z"])

    return {
        "u_pre": L_u_pre,
        "conv": L_conv,
        "u": L_u,
        "gate_pre": L_gate_pre,
        "gate": silu_gate * L_gate_pre,     # of SiLU(gate)
        "B": L_B,
        "C": L_C,
        "z": L_z,
        "log_delta": log_slope * L_z,
        "delta": dt_slope * L_z,
    }


def _widen(box: Interval, r: torch.Tensor) -> Interval:
    return Interval(box.lo - r, box.hi + r)


def _warp_coefficient(variant: str) -> float:
    # first order in rho: 2 (e^rho - 1) -> 2 rho, and 2 tanh(rho/2) -> rho
    if variant == "cell":
        return 2.0
    if variant == "common_start":
        return 1.0
    raise ValueError(f"unknown variant: {variant!r}")


@dataclass
class BlockIncrement:
    """The incremental certificate of one block with respect to its normalised input, in float64.

    Per cell (channel d, state n): lam, M. Per channel: floor, ceiling, L_u, L_gate, L_z,
    L_log_delta, L_delta, sup_u, sup_y, sup_gate, D_abs. Per state: L_B, L_C, sup_B, sup_C.
    W_out_abs is |out_proj.weight|, (d_model, d_inner).

    Two readings of "perturb the normalised input by delta":

      arbitrary=False  both normalised inputs are LayerNorm outputs, as they are inside the
                       network. Both trajectories then lie in the certified boxes and every
                       bound is first order in delta except through the time-warp term.
      arbitrary=True   the perturbed one is any point within delta of a LayerNorm output. It can
                       leave the boxes, which are widened by L delta, so second-order terms enter.
                       Derivative suprema are global.

    Two time-warp terms, chosen by `variant`: "cell", the cell theorem's 2 M (e^rho - 1), and
    "common_start", 2 M tanh(rho/2), which holds because both runs start from h_{-1} = 0.

    The slopes are reported for the first reading. The second's slope is at least as large, and
    equal when the boxes already reach the suprema of SiLU' and of the log-timescale slope.
    """

    parametrisation: str
    lam: torch.Tensor
    M: torch.Tensor
    floor: torch.Tensor
    ceiling: torch.Tensor
    L_u: torch.Tensor
    L_gate: torch.Tensor
    L_B: torch.Tensor
    L_C: torch.Tensor
    L_z: torch.Tensor
    L_log_delta: torch.Tensor
    L_delta: torch.Tensor
    sup_u: torch.Tensor
    sup_B: torch.Tensor
    sup_C: torch.Tensor
    sup_y: torch.Tensor
    sup_gate: torch.Tensor
    D_abs: torch.Tensor
    W_out_abs: torch.Tensor
    # what arbitrary=True needs: the magnitude boxes and the global Lipschitz constants
    boxes: dict
    global_lipschitz: dict

    # ours

    def forcing_slope(self) -> torch.Tensor:
        """sup|c - c~| / delta per cell, from c - c~ = (B (u - u~) + (B - B~) u~) / lam."""
        return (
            self.sup_B.unsqueeze(0) * self.L_u.unsqueeze(1)
            + self.sup_u.unsqueeze(1) * self.L_B.unsqueeze(0)
        ) / self.lam

    def warp_slope(self, variant: str = "cell") -> torch.Tensor:
        """Small-delta slope of the time-warp term per cell: 2 M L_log_delta for the cell theorem,
        M L_log_delta with a common start."""
        return _warp_coefficient(variant) * self.M * self.L_log_delta.unsqueeze(1)

    def state_slope(self, variant: str = "cell") -> torch.Tensor:
        """Small-delta slope of sup_t |h_t - h~_t| per cell: forcing plus time-warp."""
        return self.forcing_slope() + self.warp_slope(variant)

    def state_bound(self, delta: float, arbitrary: bool = False, variant: str = "cell") -> torch.Tensor:
        """sup_t |h_t - h~_t| per cell at perturbation size delta."""
        if not arbitrary:
            rho = self.L_log_delta.unsqueeze(1) * delta
            return self.forcing_slope() * delta + warp_term(self.M, rho, variant)
        g = self.global_lipschitz
        u_w = _widen(self.boxes["u"], g["u"] * delta)
        B_w = _widen(self.boxes["B"], g["B"] * delta)
        M_w = equilibrium_box(-self.lam, B_w, u_w).abs_max()
        dc = (
            self.sup_B.unsqueeze(0) * g["u"].unsqueeze(1)
            + u_w.abs_max().unsqueeze(1) * g["B"].unsqueeze(0)
        ) * delta / self.lam
        rho = g["log_delta"].unsqueeze(1) * delta
        return dc + warp_term(M_w, rho, variant)

    def output_bound(self, delta: float, arbitrary: bool = False, variant: str = "cell") -> torch.Tensor:
        """Bound on |F(x) - F(x~)| per output coordinate, F the block's contribution before the
        residual, out_proj(y * SiLU(gate)), at perturbation size delta."""
        return self.propagate(self.state_bound(delta, arbitrary, variant), delta, arbitrary)

    def output_slope(self, variant: str = "cell") -> torch.Tensor:
        """Small-delta slope of output_bound, per output coordinate: the sup-norm Lipschitz
        constant of the block's contribution with respect to its normalised input."""
        return self.propagate(self.state_slope(variant), 1.0)

    def output_slope_parts(self, variant: str = "cell") -> dict[str, torch.Tensor]:
        """output_slope split by origin, per output coordinate; the parts add up to it.

        warp and forcing are the two terms of the state bound read out through C; C is the
        perturbation of C itself, M L_C; D is the skip path; gate is sup|y| times the gate's
        own sensitivity.
        """
        g = self.sup_gate
        parts = {
            "warp": g * (self.sup_C.unsqueeze(0) * self.warp_slope(variant)).sum(-1),
            "forcing": g * (self.sup_C.unsqueeze(0) * self.forcing_slope()).sum(-1),
            "C": g * (self.M * self.L_C.unsqueeze(0)).sum(-1),
            "D": g * self.D_abs * self.L_u,
            "gate": self.sup_y * self.L_gate,
        }
        return {k: self.W_out_abs @ v for k, v in parts.items()}

    def y_bound(self, delta: float, arbitrary: bool = False, variant: str = "cell") -> torch.Tensor:
        """sup_t |y_t - y~_t| per channel at perturbation size delta, before the gate."""
        return self._readout(self.state_bound(delta, arbitrary, variant), delta, arbitrary)

    def y_slope(self, variant: str = "cell") -> torch.Tensor:
        return self._readout(self.state_slope(variant), 1.0, False)

    def _readout(self, E: torch.Tensor, delta: float, arbitrary: bool) -> torch.Tensor:
        # y_d = sum_n C_n h_dn + D_d u_d and C h - C~ h~ = C~ (h - h~) + (C - C~) h
        if arbitrary:
            g = self.global_lipschitz
            sup_C = _widen(self.boxes["C"], g["C"] * delta).abs_max()
            L_u, L_C = g["u"], g["C"]
        else:
            sup_C, L_u, L_C = self.sup_C, self.L_u, self.L_C
        dy = (sup_C.unsqueeze(0) * E + self.M * L_C.unsqueeze(0) * delta).sum(-1)
        return dy + self.D_abs * L_u * delta

    def propagate(self, E: torch.Tensor, delta: float, arbitrary: bool = False) -> torch.Tensor:
        """Carry a per-cell state bound E through C, D, the SiLU gate and out_proj.

        |y - y~| <= sum_n (sup|C~_n| E_dn + M_dn L_C delta) + |D| L_u delta, and the gate gives
        |y g - y~ g~| <= |y - y~| sup|g~| + sup|y| |g - g~|. Linear in (E, delta), so a slope in
        gives a slope out.
        """
        dy = self._readout(E, delta, arbitrary)
        if arbitrary:
            g = self.global_lipschitz
            sup_gate = iv.silu(_widen(self.boxes["gate_pre"], g["gate_pre"] * delta)).abs_max()
            L_gate = g["gate"]
        else:
            sup_gate, L_gate = self.sup_gate, self.L_gate
        dv = dy * sup_gate + self.sup_y * L_gate * delta
        return self.W_out_abs @ dv

    # the additive baseline

    def additive_state_slope(self, seq_len: int | None = None, floor: bool = True) -> torch.Tensor:
        """Per-cell slope of the same recursion with |a - a~| <= lam exp(-lam dt_min) |dt - dt~|.

        The step is |e_t| <= a_t |e_{t-1}| + (1 - a_t) dc + 2 M eta with eta = lam exp(-lam dt_min)
        L_delta delta, and from e_{-1} = 0 that gives |e_{L-1}| <= dc + 2 M eta S_L with
        S_L = sum_{k<L} sup(a)^k. With floor=False, dt_min = 0, sup a = 1 and S_L = L, linear in
        the length. With floor=True, dt_min is the certified per-channel floor and
        S_L = (1 - exp(-L lam dt_min))/(1 - exp(-lam dt_min)); seq_len=None gives the limit
        1/(1 - exp(-lam dt_min)). Cao et al.'s Lemma B.3 drops the factor exp(-lam dt_min);
        keeping it makes this the smaller of the two, so the baseline is given its better form.
        """
        dc = self.forcing_slope()
        L_delta = self.L_delta.unsqueeze(1)
        if not floor:
            if seq_len is None:
                return torch.full_like(dc, math.inf)
            return dc + 2.0 * self.M * self.lam * L_delta * float(seq_len)
        x = self.lam * self.floor.unsqueeze(1)
        eta = self.lam * torch.exp(-x) * L_delta
        positive = x > 0
        x_safe = torch.where(positive, x, torch.ones_like(x))
        if seq_len is None:
            S = torch.where(positive, 1.0 / -torch.expm1(-x_safe), torch.full_like(x, math.inf))
        else:
            S = torch.where(
                positive,
                torch.expm1(-float(seq_len) * x_safe) / torch.expm1(-x_safe),
                torch.full_like(x, float(seq_len)),
            )
        return dc + 2.0 * self.M * eta * S

    def additive_output_slope(self, seq_len: int | None = None, floor: bool = True) -> torch.Tensor:
        return self.propagate(self.additive_state_slope(seq_len, floor), 1.0)

    # per-cell comparison of the two time-warp terms, first order in delta

    def warp_ratio(self, floor: torch.Tensor | None = None) -> torch.Tensor:
        """The floored additive time-warp term over the cell theorem's, per cell.

            additive  2 M lam exp(-x) L_delta / (1 - exp(-x)) = 2 M lam L_delta / (e^x - 1)
            ours      2 M L_log_delta

        with x = lam dt_min, the additive term taken at its limit in the length. The forcing term
        is the same in both and is left out, and M cancels. With the slopes over the certified
        box of the channel, the ratio is [L_delta / (dt_min L_log_delta)] x / (e^x - 1): the
        first factor is the parametrisation's, and x / (e^x - 1) <= 1. Cao et al.'s Remark 3.4
        term drops exp(-x) and is larger by e^x. `floor` replaces the certified floor.
        """
        f = self.floor if floor is None else floor.to(self.floor.dtype)
        x = self.lam * f.unsqueeze(1)
        return self.lam * self.L_delta.unsqueeze(1) / (torch.expm1(x) * self.L_log_delta.unsqueeze(1))

    def first_factor(self) -> torch.Tensor:
        """L_delta / (dt_min L_log_delta) per channel: sup|dt'| / (dt_min sup|(log dt)'|) over the
        certified box of z, with dt_min the certified floor."""
        return self.L_delta / (self.floor * self.L_log_delta)

    def crossover_horizon(self, max_steps: int = 10**6) -> torch.Tensor:
        """Per cell, the number of steps below which the additive term without a floor is smaller.

        To first order in delta, after n steps the additive time-warp term without a floor is
        2 M lam L_delta n, and the cell theorem's, in its finite-horizon form with the worst
        case over admissible timescales, is 2 M L_log_delta (1 - exp(-lam dt_hi n)), dt_hi the
        certified ceiling. The first is linear and the second concave with the same value 0 at
        n = 0, so they cross once. Returns the smallest integer n >= 1 at which the additive term
        is at least ours, inf if that is beyond max_steps.
        """
        a = self.lam * self.L_delta.unsqueeze(1)
        b = self.L_log_delta.unsqueeze(1).expand_as(a)
        y = self.lam * self.ceiling.unsqueeze(1)
        # at n >= b/a the linear term alone exceeds the bounded concave one
        reach = torch.where(a > 0, torch.ceil(b / a), torch.full_like(a, math.inf))
        finite = reach[torch.isfinite(reach)]
        top = float(finite.max()) if finite.numel() else 1.0
        N = int(min(max(top, 1.0), float(max_steps)))
        n = torch.arange(1, N + 1, dtype=DTYPE)
        ok = a.unsqueeze(-1) * n >= b.unsqueeze(-1) * -torch.expm1(-y.unsqueeze(-1) * n)
        first = ok.to(torch.int64).argmax(-1).to(DTYPE) + 1.0
        return torch.where(ok.any(-1), first, torch.full_like(first, math.inf))

    def additive_over_ours_one_step(self) -> torch.Tensor:
        """Per cell, the additive time-warp term without a floor over the cell theorem's
        finite-horizon one after a single step, first order in delta."""
        a = self.lam * self.L_delta.unsqueeze(1)
        y = self.lam * self.ceiling.unsqueeze(1)
        return a / (self.L_log_delta.unsqueeze(1) * -torch.expm1(-y))

    def euler_gain_max(self) -> float:
        """g(lam dt_hi), the largest factor exponential-Euler puts on the equilibrium, over cells."""
        return float(euler_gain(self.lam * self.ceiling.unsqueeze(1)).max())


@torch.no_grad()
def incremental_certificate(
    block: SelectiveSSMBlock, tight_layernorm: bool = True, cert: dict | None = None
) -> BlockIncrement:
    """Data-free incremental certificate for one block.

    M, sup|u|, sup|B|, sup|C| and sup|y| come from the certificate's own boxes: pass the dict
    certify_block returned, or leave cert=None to rebuild the same boxes without certify_block's
    baselines, whose widening runs to its iteration cap on the softplus arm.
    """
    boxes = block_boxes(block, tight_layernorm)
    if cert is not None:
        cb = cert["boxes"]
        u, B, C = cb["u"], cb["B"], cb["C"]
        h_box, y_box = cb["h_invariant"], cb["y_invariant"]
    else:
        u, B, C = boxes["u"], boxes["B"], boxes["C"]
        h_box = certify_block_state(block, u, B)["invariant"]
        cy = h_box * Interval(C.lo.unsqueeze(0), C.hi.unsqueeze(0))
        y_box = cy.sum(dim=-1) + (u * block.D)

    lip = block_lipschitz(block, boxes)
    glob = block_lipschitz(block, None)
    lam = torch.exp(_f64(block.A_log))

    return BlockIncrement(
        parametrisation=block.dt_parametrisation,
        lam=lam,
        M=_f64(h_box.abs_max()),
        floor=timescale_floor(block, boxes["z"]),
        ceiling=timescale_ceiling(block, boxes["z"]),
        L_u=lip["u"],
        L_gate=lip["gate"],
        L_B=lip["B"],
        L_C=lip["C"],
        L_z=lip["z"],
        L_log_delta=lip["log_delta"],
        L_delta=lip["delta"],
        sup_u=_f64(u.abs_max()),
        sup_B=_f64(B.abs_max()),
        sup_C=_f64(C.abs_max()),
        sup_y=_f64(y_box.abs_max()),
        sup_gate=_f64(iv.silu(boxes["gate_pre"]).abs_max()),
        D_abs=_f64(block.D).abs(),
        W_out_abs=_f64(block.out_proj.weight).abs(),
        boxes={k: _f64_box(boxes[k]) for k in ("u", "B", "C", "gate_pre")},
        global_lipschitz=glob,
    )


@torch.no_grad()
def contribution(block: SelectiveSSMBlock, xn: torch.Tensor) -> torch.Tensor:
    """F(xn) = out_proj(y * SiLU(gate)), the block's output before the residual, run from the
    normalised input. Mirrors SelectiveSSMBlock.forward line for line, minus the LayerNorm."""
    L = xn.shape[1]
    u, gate = block.in_proj(xn).chunk(2, dim=-1)
    u = F.silu(block.conv1d(u.transpose(1, 2))[:, :, :L].transpose(1, 2))
    delta, B, C = block.ssm_parameters(u)
    A = -torch.exp(block.A_log)

    h = xn.new_zeros(xn.shape[0], block.d_inner, block.d_state)
    ys = []
    for t in range(L):
        dtA = delta[:, t].unsqueeze(-1) * A.unsqueeze(0)
        B_bar = delta[:, t].unsqueeze(-1) * zoh_phi(dtA) * B[:, t].unsqueeze(1)
        h = torch.exp(dtA) * h + B_bar * u[:, t].unsqueeze(-1)
        ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]) + block.D * u[:, t])
    y = torch.stack(ys, dim=1)
    return block.out_proj(y * F.silu(gate))


@torch.no_grad()
def realised_gap(block: SelectiveSSMBlock, xn: torch.Tensor) -> torch.Tensor:
    """Per cell, the largest |h_{t-1} - c_t| over the batch and the positions, c_t = B_t u_t / lam.

    The distance between the state and its equilibrium along the run from h_{-1} = 0 that the
    normalised input xn drives; the local corollary puts it where the cell theorem has 2 M.
    The state is updated as SelectiveSSMBlock.forward updates it.
    """
    L = xn.shape[1]
    u, _ = block.in_proj(xn).chunk(2, dim=-1)
    u = F.silu(block.conv1d(u.transpose(1, 2))[:, :, :L].transpose(1, 2))
    delta, B, _ = block.ssm_parameters(u)
    A = -torch.exp(block.A_log)
    lam = A.abs()

    h = xn.new_zeros(xn.shape[0], block.d_inner, block.d_state)
    worst = torch.zeros_like(lam)
    for t in range(L):
        c = B[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1) / lam
        worst = torch.maximum(worst, (h - c).abs().amax(0))
        dtA = delta[:, t].unsqueeze(-1) * A.unsqueeze(0)
        B_bar = delta[:, t].unsqueeze(-1) * zoh_phi(dtA) * B[:, t].unsqueeze(1)
        h = torch.exp(dtA) * h + B_bar * u[:, t].unsqueeze(-1)
    return worst


# --- LayerNorm in front of each block ---------------------------------------------------------


def layernorm_jacobian(ln: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    """The Jacobian of LayerNorm at every row of x, (..., d, d), in closed form.

    With s = sqrt(var + eps) and z = (x - mean)/s, it is diag(gamma)(I - 1 1^T/d - z z^T/d)/s.
    The bracket is symmetric with eigenvalues 1, eps/s^2 and 0, so the l2 operator norm is at
    most max|gamma|/s, and at most max|gamma|/sqrt(eps) for any x.
    """
    d = ln.normalized_shape[0]
    centred = x - x.mean(-1, keepdim=True)
    s = torch.sqrt((centred * centred).mean(-1, keepdim=True) + ln.eps)
    z = centred / s
    eye = torch.eye(d, dtype=x.dtype)
    bracket = eye - 1.0 / d - z.unsqueeze(-1) * z.unsqueeze(-2) / d
    return ln.weight.to(x.dtype).unsqueeze(-1) * bracket / s.unsqueeze(-1)


@torch.no_grad()
def layernorm_lipschitz(ln: nn.LayerNorm) -> dict[str, float]:
    """Global Lipschitz constants of LayerNorm, which exist because eps > 0.

    l2_global_bound is max|gamma|/sqrt(eps), an upper bound for every input. The values at a
    constant input, where the variance is 0, are attained, so they are lower bounds on the global
    constants in l2 and in the sup-norm.
    """
    d = ln.normalized_shape[0]
    gamma = _f64(ln.weight)
    J = layernorm_jacobian(ln, torch.zeros(1, d, dtype=DTYPE))[0]
    return {
        "eps": float(ln.eps),
        "gamma_abs_max": float(gamma.abs().max()),
        "l2_global_bound": float(gamma.abs().max()) / math.sqrt(ln.eps),
        "l2_at_constant_input": float(torch.linalg.matrix_norm(J, ord=2)),
        "linf_at_constant_input": float(torch.linalg.matrix_norm(J, ord=float("inf"))),
    }


@torch.no_grad()
def layernorm_realised(ln: nn.LayerNorm, x: torch.Tensor, chunk: int = 1024) -> dict[str, float]:
    """The same constants at realised inputs x (..., d): the largest l2 and sup-norm operator norm
    of the Jacobian over the rows, the bound max|gamma|/s, and the range of the spread s."""
    x = x.to(DTYPE).reshape(-1, x.shape[-1])
    l2, linf = 0.0, 0.0
    for i0 in range(0, x.shape[0], chunk):
        J = layernorm_jacobian(ln, x[i0 : i0 + chunk])
        l2 = max(l2, float(torch.linalg.matrix_norm(J, ord=2).max()))
        linf = max(linf, float(torch.linalg.matrix_norm(J, ord=float("inf")).max()))
    centred = x - x.mean(-1, keepdim=True)
    s = torch.sqrt((centred * centred).mean(-1) + ln.eps)
    return {
        "rows": int(x.shape[0]),
        "l2_max": l2,
        "l2_bound_max": float(_f64(ln.weight).abs().max() / s.min()),
        "linf_max": linf,
        "spread_min": float(s.min()),
        "spread_max": float(s.max()),
    }


# --- the local version, as a drop-in recurrence step for the norm-propagation domain ----------


def additive_step(h: Sens, dt_t: Sens, u_t: Sens, B_t: Sens, A: torch.Tensor, eps: float) -> Sens:
    """The recurrence step of lipschitz._block, unchanged: Abar and the ZOH gain are perturbed
    additively in delta, with sensitivities |A| exp(delta_lo A) and exp(delta_lo A)."""
    dtA = dt_t.val * A.unsqueeze(0)
    # both exp(delta A) and its delta-derivative are largest at the smallest delta
    dtA_lo = (dt_t.val - dt_t.lip * eps).clamp(min=0.0) * A.unsqueeze(0)
    decay_sup = torch.exp(dtA_lo)
    A_bar = Sens(torch.exp(dtA), A.abs() * decay_sup * dt_t.lip)
    g = Sens(dt_t.val * zoh_phi(dtA), decay_sup * dt_t.lip)
    return add(mul(A_bar, h, eps), mul(mul(g, B_t, eps), u_t, eps))


def log_delta_step(
    h: Sens, dt_t: Sens, log_lip_t: torch.Tensor, u_t: Sens, B_t: Sens, A: torch.Tensor,
    eps: float, recursive: bool = False, endpoint: bool = False, additive_only: bool = False,
) -> Sens:
    """The local corollary as a recurrence step. h.lip * eps carries E_t; the value is the
    nominal state, computed as SelectiveSSMBlock.forward computes it.

    dc_t is the bound mul gives on |B u - B~ u~| divided by lam, rho_t = log_lip_t eps, and the
    distance |h_{t-1} - c_t| is taken on the nominal values. recursive=False applies
    `local_step`, the running maximum, and recursive=True `local_step_recursive`, which also
    lets the carried bound decay. Neither grows with rho_t beyond e^{rho_t} - 1 times a realised
    distance. endpoint=True applies `local_step_endpoint` with both the log-timescale bound rho_t
    and the additive one the domain carries, |dt - dt~| <= dt_t.lip eps, the bound the additive
    step uses; it overrides recursive. additive_only=True applies `local_step_endpoint` with the
    additive bound alone (rho = inf): the convex step of the proof without the log-timescale
    bound. It is the ablation that isolates what rho_t adds, and it overrides the other two.
    """
    dtA = dt_t.val * A.unsqueeze(0)
    lam = A.abs().unsqueeze(0)
    Bu = mul(B_t, u_t, eps)
    c = Bu.val / lam
    E_prev = h.lip * eps
    rho = log_lip_t * eps
    dc = Bu.lip * eps / lam
    gap = (h.val - c).abs()
    if additive_only:
        E = local_step_endpoint(E_prev, dt_t.val * lam, torch.full_like(rho, math.inf), dc, gap,
                                dx=lam * (dt_t.lip * eps))
    elif endpoint:
        E = local_step_endpoint(E_prev, dt_t.val * lam, rho, dc, gap, dx=lam * (dt_t.lip * eps))
    elif recursive:
        E = local_step_recursive(E_prev, dt_t.val * lam, rho, dc, gap)
    else:
        E = local_step(E_prev, torch.expm1(rho), dc, gap)
    # 0 * inf arises once an upstream bound is already infinite; report no bound, not nan
    E = torch.nan_to_num(E, nan=math.inf)
    val = torch.exp(dtA) * h.val + (dt_t.val * zoh_phi(dtA)) * B_t.val * u_t.val
    return Sens(val, E / eps)


def _log_delta_lip(block: SelectiveSSMBlock, pre: Sens, eps: float) -> torch.Tensor:
    """Sensitivity of log(delta), from the pre-activation's and the slope over its local box."""
    lo, hi = pre.box(eps)
    return log_delta_slope(block, Interval(lo, hi)).to(pre.lip.dtype) * pre.lip


def _delta_tight(block: SelectiveSSMBlock, pre: Sens, eps: float) -> Sens:
    """The timescale and its sensitivity over the pre-activation's local box, with the slope of
    `delta_slope`, the supremum of |dt'| over the box that the data-free baseline also uses.

    lipschitz._delta bounds the bounded map's slope dt ln(r) sigmoid' by dt_max ln(r) sigmoid',
    which is up to dt_max/dt_min times larger where the realised dt is small, and that looseness
    would pass to every rule that uses the additive bound on the timescale. For softplus the
    two agree.
    """
    if block.dt_parametrisation != "bounded":
        return _delta(block, pre, eps)
    lo, hi = pre.box(eps)
    slope = delta_slope(block, Interval(lo, hi)).to(pre.lip.dtype)
    return Sens(block.delta_from_pre(pre.val), slope * pre.lip)


def _block_rule(block: SelectiveSSMBlock, x: Sens, eps: float, rule: str,
                normalised: bool = False) -> Sens:
    """lipschitz._block with its recurrence step chosen by `rule`.

    The timescale's sensitivity is `_delta_tight`, so rule="additive" is lipschitz._block with
    the bounded map's slope tightened, and is lipschitz._block exactly for softplus. With
    normalised=True, x is the block's normalised input and the result is its contribution before
    the residual, the map the data-free certificate bounds.
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule: {rule!r}")
    L = x.val.shape[1]
    xn = x if normalised else layer_norm(x, block.norm, eps)
    xz = linear(xn, block.in_proj.weight, block.in_proj.bias)
    u = Sens(xz.val[..., : block.d_inner], xz.lip[..., : block.d_inner])
    gate = Sens(xz.val[..., block.d_inner :], xz.lip[..., block.d_inner :])

    u = silu(_causal_conv(u, block), eps)

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
    ys_v, ys_l = [], []

    for t in range(L):
        dt_t = Sens(delta.val[:, t].unsqueeze(-1), delta.lip[:, t].unsqueeze(-1))
        u_t = Sens(u.val[:, t].unsqueeze(-1), u.lip[:, t].unsqueeze(-1))
        B_t = Sens(B.val[:, t].unsqueeze(1), B.lip[:, t].unsqueeze(1))
        C_t = Sens(C.val[:, t].unsqueeze(1), C.lip[:, t].unsqueeze(1))

        if rule == "additive":
            h = additive_step(h, dt_t, u_t, B_t, A, eps)
        else:
            h = log_delta_step(h, dt_t, log_lip[:, t].unsqueeze(-1), u_t, B_t, A, eps,
                               recursive=rule == "log_delta_recursive",
                               endpoint=rule == "log_delta_endpoint",
                               additive_only=rule == "additive_endpoint")

        cy = mul(h, C_t, eps)
        ys_v.append(cy.val.sum(-1) + block.D * u.val[:, t])
        ys_l.append(cy.lip.sum(-1) + block.D.abs() * u.lip[:, t])

    y = Sens(torch.stack(ys_v, 1), torch.stack(ys_l, 1))
    gated = mul(y, silu(gate, eps), eps)
    out = linear(gated, block.out_proj.weight, block.out_proj.bias)
    return out if normalised else add(out, x)


@torch.no_grad()
def local_sensitivity_rule(model: LOBMamba, x: torch.Tensor, eps: float,
                           rule: str = REPORTED_RULE) -> dict:
    """lipschitz.local_sensitivity with the recurrence step chosen by `rule`.

    rule="additive" returns what lipschitz.local_sensitivity returns with the bounded map's
    timescale slope tightened (`_delta_tight`): never more, and exactly the same for softplus.
    """
    model.eval()
    h = linear(Sens(x, torch.ones_like(x)), model.proj[0].weight, model.proj[0].bias)
    h = layer_norm(silu(h, eps), model.proj[2], eps)

    for block in model.layers:
        h = _block_rule(block, h, eps, rule)

    h = layer_norm(h, model.final_norm, eps)
    last = Sens(h.val[:, -1, :], h.lip[:, -1, :])

    z = silu(linear(last, model.head[0].weight, model.head[0].bias), eps)
    out = linear(z, model.head[3].weight, model.head[3].bias)

    val, lip = out.val.squeeze(-1), out.lip.squeeze(-1)
    return {
        "eps": eps,
        "rule": rule,
        "nominal": val,
        "lipschitz": lip,
        "lo": val - lip * eps,
        "hi": val + lip * eps,
        "width": 2.0 * lip * eps,
    }


@torch.no_grad()
def local_block_sensitivity(block: SelectiveSSMBlock, xn: torch.Tensor, eps: float,
                            rule: str = REPORTED_RULE) -> Sens:
    """Local bound on the block's contribution around a nominal normalised input xn, for any
    perturbation of xn of sup-norm at most eps, LayerNorm output or not."""
    return _block_rule(block, Sens(xn, torch.ones_like(xn)), eps, rule, normalised=True)
