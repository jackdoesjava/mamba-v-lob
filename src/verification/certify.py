"""End-to-end certificate for `LOBMamba`, plus the empirical envelope it is priced against.

Structural vs in-context bounds, and what each one licenses: see docs/02-invariant.md.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch

from src.models.mamba import LOBMamba, SelectiveSSMBlock
from src.verification import intervals as iv
from src.verification.intervals import Interval
from src.verification.invariant import certify_block_state


def _split(box: Interval, sizes: list[int], dim: int = -1) -> list[Interval]:
    los = torch.split(box.lo, sizes, dim=dim)
    his = torch.split(box.hi, sizes, dim=dim)
    return [Interval(a, b) for a, b in zip(los, his)]


# Realised quantities and the certified box each is priced against. The state and its
# readout are priced against the invariant, which is the bound the certificate ships; the
# unrolled boxes stay in the certificate as baselines.
PRICED: list[tuple[str, str]] = [
    ("u", "u"),
    ("delta", "delta"),
    ("B", "B"),
    ("C", "C"),
    ("A_bar", "A_bar"),
    ("B_bar", "B_bar"),
    ("h", "h_invariant"),
    ("y", "y_invariant"),
]


@torch.no_grad()
def certify_block(
    block: SelectiveSSMBlock,
    seq_len: int,
    tight_layernorm: bool = True,
) -> dict:
    """Certify one block. Requires no data and no assumption on the block's input."""
    # fusing the norm into in_proj gives a tighter box than bounding the two separately
    if tight_layernorm:
        in_box = iv.layernorm_linear(block.norm, block.in_proj)
    else:
        in_box = iv.affine(
            iv.layernorm(block.norm), block.in_proj.weight, block.in_proj.bias
        )
    u_pre, gate_pre = _split(in_box, [block.d_inner, block.d_inner])

    conv_fn = (
        iv.depthwise_conv1d
        if block.conv1d.groups == block.conv1d.in_channels
        else iv.dense_conv1d
    )
    u_box = iv.silu(conv_fn(u_pre, block.conv1d))

    proj_box = iv.affine(u_box, block.x_proj.weight, block.x_proj.bias)
    dt_pre_raw, B_box, C_box = _split(
        proj_box, [block.dt_rank, block.d_state, block.d_state]
    )
    dt_pre = iv.affine(dt_pre_raw, block.dt_proj.weight, block.dt_proj.bias)
    delta = iv.delta_box(block, dt_pre)

    # A < 0 by construction, so Abar is decreasing in dt and g stays positive; that is what
    # discretisation_boxes needs. It does NOT give sup Abar < 1: that needs dt_min > 0.
    A = -torch.exp(block.A_log)
    A_bar, g = iv.discretisation_boxes(A, delta)

    # both unsqueezes broadcast up to (d_inner, d_state): B over channels, u over states
    B_bar = g * Interval(B_box.lo.unsqueeze(0), B_box.hi.unsqueeze(0))
    drive = B_bar * Interval(u_box.lo.unsqueeze(-1), u_box.hi.unsqueeze(-1))

    # The bound the certificate ships. The two below are baselines it is measured against.
    state = certify_block_state(block, u_box, B_box)
    h_invariant = state["invariant"]

    h_geom = iv.state_bound_geometric(A_bar, drive)
    h_horizon = iv.state_bound_unrolled(A_bar, drive, seq_len)
    h_widened, widen_iters = iv.state_bound_widened(A_bar, drive)

    def block_output(h_box: Interval) -> tuple[Interval, Interval]:
        # The second box is pre-residual: forward() returns out_proj(...) + x, and the
        # residual stream is not bounded here. Only certify_output_range bounds a whole
        # forward pass, via final_norm.
        cy = h_box * Interval(C_box.lo.unsqueeze(0), C_box.hi.unsqueeze(0))
        y = cy.sum(dim=-1) + (u_box * block.D)
        y_gated = y * iv.silu(gate_pre)
        return y, iv.affine(y_gated, block.out_proj.weight, block.out_proj.bias)

    y_invariant, out_invariant = block_output(h_invariant)
    y_geom, out_geom = block_output(h_geom)
    y_horizon, out_horizon = block_output(h_horizon)

    cert = block.contraction_certificate()
    return {
        "contraction": cert,
        "horizon_gain": block.horizon_gain(seq_len),
        "boxes": {
            "u": u_box,
            "delta": delta,
            "B": B_box,
            "C": C_box,
            "A_bar": A_bar,
            "B_bar": B_bar,
            "h_invariant": h_invariant,
            "h_geometric": h_geom,
            "h_horizon": h_horizon,
            "h_widened": h_widened,
            "y_invariant": y_invariant,
            "y_geometric": y_geom,
            "y_horizon": y_horizon,
            "out_invariant": out_invariant,
            "out_geometric": out_geom,
            "out_horizon": out_horizon,
        },
        "summary": {
            "sup_A_bar": cert["sup_A_bar"],
            "geometric_gain": cert["geometric_gain"],
            "u_abs_max": u_box.abs_max().max().item(),
            "delta_lo": delta.lo.min().item(),
            "delta_hi": delta.hi.max().item(),
            "B_abs_max": B_box.abs_max().max().item(),
            "C_abs_max": C_box.abs_max().max().item(),
            "h_abs_max_invariant": state["radius"],
            "invariant_holds": state["holds"],
            "invariant_induction_residual": state["induction_residual"],
            "zoh_exact": state["zoh_exact"],
            "h_abs_max_widened": h_widened.abs_max().max().item(),
            "widening_iterations": widen_iters,
            "h_abs_max_geometric": h_geom.abs_max().max().item(),
            "h_abs_max_horizon": h_horizon.abs_max().max().item(),
            "y_abs_max_invariant": y_invariant.abs_max().max().item(),
            "y_abs_max_horizon": y_horizon.abs_max().max().item(),
        },
    }


@torch.no_grad()
def certify_output_range(model: LOBMamba, tight_layernorm: bool = True) -> Interval:
    """Certified range of the scalar prediction, for any input whatsoever.

    `final_norm` bounds the head's input whatever the residual stream did.
    """
    # head[1] is the SiLU, head[2] the dropout, which is the identity at eval
    head_in, head_out = model.head[0], model.head[3]
    if tight_layernorm:
        box = iv.layernorm_linear(model.final_norm, head_in)
    else:
        box = iv.affine(iv.layernorm(model.final_norm), head_in.weight, head_in.bias)
    box = iv.silu(box)
    return iv.affine(box, head_out.weight, head_out.bias)


def certify_model(
    model: LOBMamba,
    seq_len: int,
    tight_layernorm: bool = True,
) -> dict:
    model.eval()
    with torch.no_grad():
        layers = [
            certify_block(b, seq_len=seq_len, tight_layernorm=tight_layernorm)
            for b in model.layers
        ]
        out_box = certify_output_range(model, tight_layernorm=tight_layernorm)

    all_contractive = all(l["contraction"]["contractive"] for l in layers)
    return {
        "architecture": model.architecture(),
        "seq_len": seq_len,
        "relaxation": "layernorm_linear_fused" if tight_layernorm else "layernorm_box",
        "all_layers_contractive": all_contractive,
        "output_range": {
            "lo": out_box.lo.item(),
            "hi": out_box.hi.item(),
        },
        "layers": layers,
    }


class _Envelope:
    def __init__(self) -> None:
        self.lo: dict[str, float] = {}
        self.hi: dict[str, float] = {}

    def update(self, key: str, t: torch.Tensor) -> None:
        lo, hi = t.min().item(), t.max().item()
        self.lo[key] = min(self.lo.get(key, math.inf), lo)
        self.hi[key] = max(self.hi.get(key, -math.inf), hi)

    def as_dict(self) -> dict:
        return {k: {"lo": self.lo[k], "hi": self.hi[k]} for k in sorted(self.lo)}


@torch.no_grad()
def empirical_envelope(
    model: LOBMamba,
    batches: Iterable[torch.Tensor],
    max_batches: Optional[int] = None,
) -> dict:
    """Realised ranges of every SSM quantity, to price the certified boxes against."""
    model.eval()
    envs = [_Envelope() for _ in model.layers]
    out_env = _Envelope()

    for i, x in enumerate(batches):
        if max_batches is not None and i >= max_batches:
            break
        out, traces = model(x, trace=True)
        out_env.update("output", out)
        for env, tr in zip(envs, traces):
            env.update("u", tr.u)
            env.update("delta", tr.delta)
            env.update("B", tr.B)
            env.update("C", tr.C)
            env.update("A_bar", tr.A_bar)
            env.update("B_bar", tr.B_bar)
            env.update("h", tr.h)
            env.update("y", tr.y)

    return {
        "output": out_env.as_dict(),
        "layers": [e.as_dict() for e in envs],
    }


def looseness_report(certificate: dict, envelope: dict) -> list[dict]:
    """Ratio of certified radius to realised radius, per quantity per layer."""
    rows = []
    for li, (cert_layer, env_layer) in enumerate(
        zip(certificate["layers"], envelope["layers"])
    ):
        boxes = cert_layer["boxes"]
        for key, cert_key in PRICED:
            if key not in env_layer:
                continue
            cert_r = boxes[cert_key].abs_max().max().item()
            emp_r = max(abs(env_layer[key]["lo"]), abs(env_layer[key]["hi"]))
            rows.append(
                {
                    "layer": li,
                    "quantity": key,
                    "certified_abs_max": cert_r,
                    "empirical_abs_max": emp_r,
                    "looseness": (cert_r / emp_r) if emp_r > 0 else math.inf,
                }
            )
    return rows
