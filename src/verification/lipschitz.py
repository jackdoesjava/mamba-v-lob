"""Certified local sensitivity: how far the prediction can move when the book moves a little.

certify.py bounds the model over every input it could ever see. That is the right object for
an unconditional safety claim and useless for a question about one book state. Propagating a
narrow box through the network instead does not work: the LayerNorm variance lower bound
collapses once the box widens, 1/sigma runs away, and the widths compound. Measured on this
model, plain interval propagation reaches 1e24 at eps = 1e-3 and overflows above it.

What works is to keep the concrete forward pass as the centre and propagate only a bound on
the sensitivity around it. For a max-norm budget eps this tracks

    Lambda_i  >=  || d(component i) / dx ||_1

so |component_i(x + delta) - component_i(x)| <= Lambda_i * eps for every ||delta||_inf <= eps.
The certified interval is the nominal value plus or minus Lambda * eps.

The state recursion inherits the contraction factor from docs/04-stability.md:

    Lambda_h(t) <= sup|Abar| Lambda_h(t-1) + |h(t-1)| Lambda_Abar + Lambda_drive

so the dt_min that gives a finite reachable set also gives a finite sensitivity. See
docs/16-domains.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.models.mamba import LOBMamba, SelectiveSSMBlock, zoh_phi

# Extrema of silu'(x) = sigmoid(x) (1 + x (1 - sigmoid(x))), located at +-2.3993580.
SILU_DERIV_MAX = 1.0998393201
SILU_DERIV_MIN = -0.0998393201
SILU_DERIV_ARGMAX = 2.3993580
SILU_DERIV_ARGMIN = -2.3993580


@dataclass
class Sens:
    """A concrete value and a bound on its max-norm sensitivity to the network input."""

    val: torch.Tensor
    lip: torch.Tensor

    def box(self, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.lip * eps
        return self.val - r, self.val + r

    def abs_max(self, eps: float) -> torch.Tensor:
        return self.val.abs() + self.lip * eps


def _silu_deriv_sup(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    s_lo, s_hi = torch.sigmoid(lo), torch.sigmoid(hi)
    d_lo = s_lo * (1 + lo * (1 - s_lo))
    d_hi = s_hi * (1 + hi * (1 - s_hi))
    out = torch.maximum(d_lo.abs(), d_hi.abs())
    out = torch.where((lo <= SILU_DERIV_ARGMAX) & (hi >= SILU_DERIV_ARGMAX),
                      torch.full_like(out, SILU_DERIV_MAX), out)
    return torch.where((lo <= SILU_DERIV_ARGMIN) & (hi >= SILU_DERIV_ARGMIN),
                       torch.maximum(out, torch.full_like(out, abs(SILU_DERIV_MIN))), out)


def _sigmoid_deriv_sup(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    s_lo, s_hi = torch.sigmoid(lo), torch.sigmoid(hi)
    out = torch.maximum(s_lo * (1 - s_lo), s_hi * (1 - s_hi))
    return torch.where((lo <= 0) & (hi >= 0), torch.full_like(out, 0.25), out)


def silu(a: Sens, eps: float) -> Sens:
    lo, hi = a.box(eps)
    return Sens(F.silu(a.val), _silu_deriv_sup(lo, hi) * a.lip)


def linear(a: Sens, weight: torch.Tensor, bias: torch.Tensor | None) -> Sens:
    val = a.val @ weight.t()
    if bias is not None:
        val = val + bias
    return Sens(val, a.lip @ weight.abs().t())


def mul(a: Sens, b: Sens, eps: float) -> Sens:
    """Product rule keeping the second-order term, so the bound holds at finite eps."""
    return Sens(a.val * b.val, a.abs_max(eps) * b.lip + b.abs_max(eps) * a.lip)


def add(a: Sens, b: Sens) -> Sens:
    return Sens(a.val + b.val, a.lip + b.lip)


def layer_norm(a: Sens, ln, eps: float) -> Sens:
    """LayerNorm value and sensitivity.

    Row i of the Jacobian of x -> (x - mean)/sigma is (1/sigma)(e_i - 1/d - z_i z / d), whose
    L1 norm contracted against Lambda is Lambda_i + mean(Lambda) + |z_i| sum_j |z_j| Lambda_j / d.
    sigma is floored using the concrete spread: the perturbation moves each centred coordinate
    by at most Lambda_i eps + mean(Lambda) eps, so shrinking |centred| by that much and
    re-averaging is a valid floor.
    """
    d = ln.normalized_shape[0]
    centred = a.val - a.val.mean(-1, keepdim=True)
    sigma = torch.sqrt((centred * centred).mean(-1, keepdim=True) + ln.eps)
    z = centred / sigma

    reach = a.lip * eps + (a.lip * eps).mean(-1, keepdim=True)
    shrunk = (centred.abs() - reach).clamp(min=0.0)
    sigma_lo = torch.sqrt((shrunk * shrunk).mean(-1, keepdim=True) + ln.eps)

    z_abs = z.abs() + reach / sigma_lo

    # Row i of the Jacobian is (1/sigma)(e_i - 1/d - z_i z / d), so contract it against
    # Lambda coordinate by coordinate rather than bounding it by its L1 norm times max Lambda.
    weighted = (z_abs * a.lip).sum(-1, keepdim=True) / d
    row = a.lip + a.lip.mean(-1, keepdim=True) + z_abs * weighted

    lip = ln.weight.abs() * row / sigma_lo
    return Sens(z * ln.weight + ln.bias, lip)


def _causal_conv(u: Sens, block: SelectiveSSMBlock) -> Sens:
    """Depthwise causal conv. The left padding is exact zero and carries no sensitivity."""
    L = u.val.shape[1]
    w = block.conv1d.weight.reshape(block.d_inner, block.d_conv)
    pad = block.d_conv - 1
    zeros = torch.zeros(u.val.shape[0], pad, block.d_inner)
    v = torch.cat([zeros, u.val], 1)
    lp = torch.cat([zeros, u.lip], 1)

    cv = torch.zeros_like(u.val)
    cl = torch.zeros_like(u.lip)
    for k in range(block.d_conv):
        cv = cv + v[:, k : k + L, :] * w[:, k]
        cl = cl + lp[:, k : k + L, :] * w[:, k].abs()
    if block.conv1d.bias is not None:
        cv = cv + block.conv1d.bias
    return Sens(cv, cl)


def _delta(block: SelectiveSSMBlock, pre: Sens, eps: float) -> Sens:
    lo, hi = pre.box(eps)
    if block.dt_parametrisation == "bounded":
        # delta = dt_min ratio**sigmoid(z), so d(delta)/dz = delta ln(ratio) sigmoid'(z)
        gain = block.dt_max * math.log(block.dt_max / block.dt_min)
        return Sens(block.delta_from_pre(pre.val), gain * _sigmoid_deriv_sup(lo, hi) * pre.lip)
    return Sens(F.softplus(pre.val), torch.sigmoid(hi) * pre.lip)


def _block(block: SelectiveSSMBlock, x: Sens, eps: float) -> Sens:
    L = x.val.shape[1]
    xn = layer_norm(x, block.norm, eps)
    xz = linear(xn, block.in_proj.weight, block.in_proj.bias)
    u = Sens(xz.val[..., : block.d_inner], xz.lip[..., : block.d_inner])
    gate = Sens(xz.val[..., block.d_inner :], xz.lip[..., block.d_inner :])

    u = silu(_causal_conv(u, block), eps)

    proj = linear(u, block.x_proj.weight, block.x_proj.bias)
    r, n = block.dt_rank, block.d_state
    delta = _delta(
        block,
        linear(Sens(proj.val[..., :r], proj.lip[..., :r]),
               block.dt_proj.weight, block.dt_proj.bias),
        eps,
    )
    B = Sens(proj.val[..., r : r + n], proj.lip[..., r : r + n])
    C = Sens(proj.val[..., r + n :], proj.lip[..., r + n :])

    A = -torch.exp(block.A_log)
    batch = x.val.shape[0]
    zeros = torch.zeros(batch, block.d_inner, block.d_state)
    h = Sens(zeros, zeros.clone())
    ys_v, ys_l = [], []

    for t in range(L):
        dt_t = Sens(delta.val[:, t].unsqueeze(-1), delta.lip[:, t].unsqueeze(-1))
        u_t = Sens(u.val[:, t].unsqueeze(-1), u.lip[:, t].unsqueeze(-1))
        B_t = Sens(B.val[:, t].unsqueeze(1), B.lip[:, t].unsqueeze(1))
        C_t = Sens(C.val[:, t].unsqueeze(1), C.lip[:, t].unsqueeze(1))

        dtA = dt_t.val * A.unsqueeze(0)
        # both exp(delta A) and its delta-derivative are largest at the smallest delta
        dtA_lo = (dt_t.val - dt_t.lip * eps).clamp(min=0.0) * A.unsqueeze(0)
        decay_sup = torch.exp(dtA_lo)
        A_bar = Sens(torch.exp(dtA), A.abs() * decay_sup * dt_t.lip)
        g = Sens(dt_t.val * zoh_phi(dtA), decay_sup * dt_t.lip)

        h = add(mul(A_bar, h, eps), mul(mul(g, B_t, eps), u_t, eps))

        cy = mul(h, C_t, eps)
        ys_v.append(cy.val.sum(-1) + block.D * u.val[:, t])
        ys_l.append(cy.lip.sum(-1) + block.D.abs() * u.lip[:, t])

    y = Sens(torch.stack(ys_v, 1), torch.stack(ys_l, 1))
    gated = mul(y, silu(gate, eps), eps)
    return add(linear(gated, block.out_proj.weight, block.out_proj.bias), x)


@torch.no_grad()
def local_sensitivity(model: LOBMamba, x: torch.Tensor, eps: float) -> dict:
    """Certified prediction interval and max-norm Lipschitz bound at x for a budget eps."""
    model.eval()
    h = linear(Sens(x, torch.ones_like(x)), model.proj[0].weight, model.proj[0].bias)
    h = layer_norm(silu(h, eps), model.proj[2], eps)

    for block in model.layers:
        h = _block(block, h, eps)

    h = layer_norm(h, model.final_norm, eps)
    last = Sens(h.val[:, -1, :], h.lip[:, -1, :])

    z = silu(linear(last, model.head[0].weight, model.head[0].bias), eps)
    out = linear(z, model.head[3].weight, model.head[3].bias)

    val, lip = out.val.squeeze(-1), out.lip.squeeze(-1)
    return {
        "eps": eps,
        "nominal": val,
        "lipschitz": lip,
        "lo": val - lip * eps,
        "hi": val + lip * eps,
        "width": 2.0 * lip * eps,
    }
