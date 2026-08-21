"""Local sensitivity by box propagation, the other baseline that does not survive the model.

Sound, and reaches 1e24 at eps = 1e-3 before overflowing. See docs/03-certificate.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.mamba import LOBMamba, SelectiveSSMBlock
from src.verification import intervals as iv
from src.verification.intervals import Interval


def affine_seq(box: Interval, weight: torch.Tensor, bias: torch.Tensor | None) -> Interval:
    """Interval image of a Linear applied over the last axis of a (batch, L, in) box."""
    w_pos, w_neg = weight.clamp(min=0.0).t(), weight.clamp(max=0.0).t()
    lo = box.lo @ w_pos + box.hi @ w_neg
    hi = box.hi @ w_pos + box.lo @ w_neg
    if bias is not None:
        lo, hi = lo + bias, hi + bias
    return Interval(lo, hi)


def depthwise_causal_conv_seq(box: Interval, conv: nn.Conv1d) -> Interval:
    """Depthwise causal convolution over a (batch, L, channels) box.

    The left padding is exact zero rather than part of the perturbation, so it contributes a
    degenerate interval and not a hull with zero.
    """
    batch, L, ch = box.lo.shape
    kernel = conv.kernel_size[0]
    w = conv.weight.reshape(ch, kernel)
    pad = kernel - 1

    zeros = torch.zeros(batch, pad, ch, dtype=box.lo.dtype, device=box.lo.device)
    lo = torch.cat([zeros, box.lo], dim=1)
    hi = torch.cat([zeros, box.hi], dim=1)

    out_lo = torch.zeros(batch, L, ch, dtype=box.lo.dtype, device=box.lo.device)
    out_hi = torch.zeros_like(out_lo)
    for k in range(kernel):
        w_pos, w_neg = w[:, k].clamp(min=0.0), w[:, k].clamp(max=0.0)
        seg_lo, seg_hi = lo[:, k : k + L, :], hi[:, k : k + L, :]
        out_lo += seg_lo * w_pos + seg_hi * w_neg
        out_hi += seg_hi * w_pos + seg_lo * w_neg
    if conv.bias is not None:
        out_lo, out_hi = out_lo + conv.bias, out_hi + conv.bias
    return Interval(out_lo, out_hi)


def propagate_block(block: SelectiveSSMBlock, x: Interval) -> Interval:
    """One block, in the order SelectiveSSMBlock.forward runs it."""
    L = x.lo.shape[1]

    xn = iv.layernorm_input_box(x, block.norm)
    xz = affine_seq(xn, block.in_proj.weight, block.in_proj.bias)
    u = Interval(xz.lo[..., : block.d_inner], xz.hi[..., : block.d_inner])
    gate = Interval(xz.lo[..., block.d_inner :], xz.hi[..., block.d_inner :])

    u = iv.silu(depthwise_causal_conv_seq(u, block.conv1d))

    proj = affine_seq(u, block.x_proj.weight, block.x_proj.bias)
    r, n = block.dt_rank, block.d_state
    dt_pre = Interval(proj.lo[..., :r], proj.hi[..., :r])
    B = Interval(proj.lo[..., r : r + n], proj.hi[..., r : r + n])
    C = Interval(proj.lo[..., r + n :], proj.hi[..., r + n :])
    delta = iv.delta_box(block, affine_seq(dt_pre, block.dt_proj.weight, block.dt_proj.bias))

    A = -torch.exp(block.A_log)
    batch = x.lo.shape[0]
    h = Interval(
        torch.zeros(batch, block.d_inner, block.d_state, dtype=x.lo.dtype),
        torch.zeros(batch, block.d_inner, block.d_state, dtype=x.lo.dtype),
    )
    ys_lo, ys_hi = [], []

    for t in range(L):
        dt_t = Interval(delta.lo[:, t], delta.hi[:, t])
        u_t = Interval(u.lo[:, t], u.hi[:, t])
        B_t = Interval(B.lo[:, t].unsqueeze(1), B.hi[:, t].unsqueeze(1))
        C_t = Interval(C.lo[:, t].unsqueeze(1), C.hi[:, t].unsqueeze(1))

        A_bar, g = iv.discretisation_boxes(A, dt_t)
        drive = (g * B_t) * Interval(u_t.lo.unsqueeze(-1), u_t.hi.unsqueeze(-1))
        h = A_bar * h + drive

        y_t = (h * C_t).sum(dim=-1) + u_t * block.D
        ys_lo.append(y_t.lo)
        ys_hi.append(y_t.hi)

    y = Interval(torch.stack(ys_lo, 1), torch.stack(ys_hi, 1))
    gated = y * iv.silu(gate)
    return affine_seq(gated, block.out_proj.weight, block.out_proj.bias) + x


@torch.no_grad()
def certified_output(model: LOBMamba, x_box: Interval) -> Interval:
    """Certified interval for the scalar prediction, given a box on the input window."""
    h = affine_seq(x_box, model.proj[0].weight, model.proj[0].bias)
    h = iv.layernorm_input_box(iv.silu(h), model.proj[2])

    for block in model.layers:
        h = propagate_block(block, h)

    h = iv.layernorm_input_box(h, model.final_norm)
    last = Interval(h.lo[:, -1, :], h.hi[:, -1, :])

    head_in, head_out = model.head[0], model.head[3]
    z = iv.silu(affine_seq(last, head_in.weight, head_in.bias))
    return affine_seq(z, head_out.weight, head_out.bias)


@torch.no_grad()
def certified_sensitivity(
    model: LOBMamba,
    x: torch.Tensor,
    eps: float,
    input_box: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict:
    """Certified prediction interval and Lipschitz constant at x for a max-norm budget eps.

    `input_box` intersects the perturbation with the enforced feature box, which is what a
    deployment actually admits: winsorisation clips anything outside it, so a perturbation
    cannot push a feature past the edge.
    """
    lo, hi = x - eps, x + eps
    if input_box is not None:
        box_lo, box_hi = input_box
        lo = torch.maximum(lo, box_lo)
        hi = torch.minimum(hi, box_hi)

    out = certified_output(model, Interval(lo, hi))
    nominal = model(x)
    width = (out.hi - out.lo).squeeze(-1)

    return {
        "eps": eps,
        "nominal": nominal.detach(),
        "lo": out.lo.squeeze(-1),
        "hi": out.hi.squeeze(-1),
        "width": width,
        "lipschitz": width / (2.0 * eps),
    }


@torch.no_grad()
def sensitivity_sweep(
    model: LOBMamba,
    x: torch.Tensor,
    epsilons,
    input_box: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> list[dict]:
    rows = []
    for eps in epsilons:
        r = certified_sensitivity(model, x, float(eps), input_box)
        rows.append(
            {
                "eps": float(eps),
                "median_width": float(r["width"].median()),
                "max_width": float(r["width"].max()),
                "median_lipschitz": float(r["lipschitz"].median()),
                "max_lipschitz": float(r["lipschitz"].max()),
            }
        )
    return rows
