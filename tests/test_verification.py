"""Soundness tests for the interval abstraction: sample inside each input box and check
the concrete output lands inside the computed output box.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mamba import LOBMamba, SelectiveSSMBlock
from src.verification import intervals as iv
from src.verification.certify import (
    certify_block,
    certify_model,
    certify_output_range,
    empirical_envelope,
)
from src.verification.intervals import Interval


def random_box(shape, scale: float = 3.0) -> Interval:
    a = torch.randn(shape) * scale
    b = torch.randn(shape) * scale
    return Interval(torch.minimum(a, b), torch.maximum(a, b))


def sample_inside(box: Interval, n: int) -> torch.Tensor:
    t = torch.rand((n, *box.lo.shape))
    return box.lo + t * (box.hi - box.lo)


@pytest.mark.parametrize(
    "fn,abstract",
    [
        (F.silu, iv.silu),
        (torch.sigmoid, iv.sigmoid),
        (F.softplus, iv.softplus),
        (torch.exp, iv.exp),
    ],
)
def test_elementwise_bounds_are_sound(fn, abstract):
    torch.manual_seed(0)
    for _ in range(50):
        box = random_box((32,))
        out = abstract(box)
        samples = fn(sample_inside(box, 400))
        assert torch.all(out.contains(samples)), f"{fn.__name__} escaped its box"


def test_silu_bound_captures_the_interior_minimum():
    box = Interval(torch.tensor([-4.0]), torch.tensor([4.0]))
    out = iv.silu(box)
    assert out.lo.item() == pytest.approx(-0.27846454, abs=1e-6)
    assert out.lo.item() < min(F.silu(box.lo).item(), F.silu(box.hi).item())


def test_interval_product_is_sound():
    torch.manual_seed(1)
    for _ in range(50):
        a, b = random_box((16,)), random_box((16,))
        prod = a * b
        sa, sb = sample_inside(a, 300), sample_inside(b, 300)
        assert torch.all(prod.contains(sa * sb))


def test_affine_bound_is_sound():
    torch.manual_seed(2)
    layer = nn.Linear(16, 8)
    box = random_box((16,))
    out = iv.affine(box, layer.weight, layer.bias)
    with torch.no_grad():
        concrete = layer(sample_inside(box, 500))
    assert torch.all(out.contains(concrete))


def test_depthwise_conv_bound_is_sound():
    torch.manual_seed(3)
    conv = nn.Conv1d(12, 12, 4, groups=12, padding=3)
    box = random_box((12,))
    out = iv.depthwise_conv1d(box, conv)

    L = 20
    x = box.lo.view(1, -1, 1) + torch.rand(64, 12, L) * (box.hi - box.lo).view(1, -1, 1)
    with torch.no_grad():
        concrete = conv(x)[:, :, :L]
    per_channel_max = concrete.amax(dim=(0, 2))
    per_channel_min = concrete.amin(dim=(0, 2))
    assert torch.all(per_channel_max <= out.hi + 1e-4)
    assert torch.all(per_channel_min >= out.lo - 1e-4)


@pytest.mark.parametrize("d", [8, 64, 256])
def test_layernorm_box_is_sound_for_arbitrary_inputs(d):
    torch.manual_seed(4)
    ln = nn.LayerNorm(d)
    with torch.no_grad():
        ln.weight.normal_(1.0, 0.4)
        ln.bias.normal_(0.0, 0.3)
    box = iv.layernorm(ln)

    for scale in (1e-4, 1.0, 1e4):
        x = torch.randn(500, d) * scale
        # one-hot rows are the extremal direction for the normaliser
        x = torch.cat([x, torch.eye(d) * 1e3], dim=0)
        with torch.no_grad():
            out = ln(x)
        assert torch.all(box.contains(out)), f"LayerNorm escaped its box at scale {scale}"


def test_layernorm_linear_is_sound_and_tighter_than_composition():
    torch.manual_seed(5)
    d = 64
    ln, lin = nn.LayerNorm(d), nn.Linear(d, 32)
    with torch.no_grad():
        ln.weight.normal_(1.0, 0.3)
        ln.bias.normal_(0.0, 0.2)

    fused = iv.layernorm_linear(ln, lin)
    composed = iv.affine(iv.layernorm(ln), lin.weight, lin.bias)

    x = torch.randn(2000, d) * torch.rand(2000, 1) * 100.0
    with torch.no_grad():
        concrete = lin(ln(x))
    assert torch.all(fused.contains(concrete))
    assert torch.all(composed.contains(concrete))
    assert fused.width().max() < composed.width().max()


def test_layernorm_linear_bound_is_attained():
    torch.manual_seed(6)
    d = 64
    ln, lin = nn.LayerNorm(d), nn.Linear(d, 8)
    with torch.no_grad():
        ln.weight.normal_(1.0, 0.3)
        ln.bias.normal_(0.0, 0.2)
    fused = iv.layernorm_linear(ln, lin)

    for k in range(8):
        v = (lin.weight[k] * ln.weight)
        v = v - v.mean()
        x = v / v.norm() * math.sqrt(d)          # maximiser over {sum z = 0, ||z||_2 = sqrt(d)}
        with torch.no_grad():
            reached = lin(ln(x.unsqueeze(0)))[0, k].item()
        assert reached == pytest.approx(fused.hi[k].item(), rel=1e-3)


def test_block_certificate_contains_realised_values():
    torch.manual_seed(7)
    block = SelectiveSSMBlock(d_model=32, d_state=8, expand=2).eval()
    cert = certify_block(block, seq_len=40)
    boxes = cert["boxes"]

    with torch.no_grad():
        _, tr = block(torch.randn(16, 40, 32) * 5.0, trace=True)

    checks = {
        "u": (tr.u, boxes["u"]),
        "delta": (tr.delta, boxes["delta"]),
        "B": (tr.B, boxes["B"]),
        "C": (tr.C, boxes["C"]),
    }
    for name, (concrete, box) in checks.items():
        assert concrete.min() >= box.lo.min() - 1e-4, f"{name} below its box"
        assert concrete.max() <= box.hi.max() + 1e-4, f"{name} above its box"

    assert tr.A_bar.max() <= boxes["A_bar"].hi.max() + 1e-6
    assert tr.h.abs().max() <= boxes["h_horizon"].abs_max().max() + 1e-4
    assert tr.y.abs().max() <= boxes["y_horizon"].abs_max().max() + 1e-4


def test_certified_output_range_holds_for_wild_inputs():
    # the final LayerNorm bounds the head's input, so the range holds with no premise on x
    torch.manual_seed(8)
    model = LOBMamba(input_dim=20).eval()
    box = certify_output_range(model)

    for scale in (1e-3, 1.0, 1e3, 1e6):
        x = torch.randn(64, 25, 20) * scale
        with torch.no_grad():
            out = model(x)
        assert torch.all(box.contains(out)), f"output escaped its range at scale {scale}"


def test_softplus_has_no_structural_contraction_factor():
    # inf softplus = 0, so sup A_bar = 1 and the geometric series diverges
    sp = SelectiveSSMBlock(d_model=32, d_state=8, dt_parametrisation="softplus").eval()
    cert = sp.contraction_certificate()
    assert cert["sup_A_bar"] == 1.0
    assert math.isinf(cert["geometric_gain"])
    assert not cert["contractive"]


def test_bounding_delta_tightens_the_in_context_state_bound_enormously():
    # behind a LayerNorm inf delta = softplus(pre_lo) > 0, so the softplus geometric bound is
    # finite, but with factor 1 - O(softplus(pre_lo)) it is far too loose to use
    torch.manual_seed(9)
    cfg = {"model": {"mamba": {"d_model": 32, "d_state": 8, "num_layers": 1}}}

    bounded = LOBMamba(20, config=cfg).eval()
    cert = certify_model(bounded, seq_len=50)
    assert cert["all_layers_contractive"]
    h_bounded = cert["layers"][0]["summary"]["h_abs_max_geometric"]
    assert math.isfinite(h_bounded)

    cfg_sp = {"model": {"mamba": {**cfg["model"]["mamba"], "dt_parametrisation": "softplus"}}}
    softplus = LOBMamba(20, config=cfg_sp).eval()
    cert_sp = certify_model(softplus, seq_len=50)
    h_softplus = cert_sp["layers"][0]["summary"]["h_abs_max_geometric"]

    assert not cert_sp["all_layers_contractive"]  # no *structural* guarantee
    assert h_softplus > 100.0 * h_bounded, "bounding delta should tighten by orders of magnitude"
    # finite-horizon bound stays finite either way, so it is the fallback
    assert math.isfinite(cert_sp["layers"][0]["summary"]["h_abs_max_horizon"])


def test_empirical_envelope_is_inside_the_certificate():
    torch.manual_seed(10)
    model = LOBMamba(input_dim=20).eval()
    cert = certify_model(model, seq_len=30)
    env = empirical_envelope(model, [torch.randn(8, 30, 20) * 3.0 for _ in range(3)])

    for layer_cert, layer_env in zip(cert["layers"], env["layers"]):
        boxes = layer_cert["boxes"]
        for key, cert_key in [
            ("u", "u"),
            ("delta", "delta"),
            ("B", "B"),
            ("C", "C"),
            ("A_bar", "A_bar"),
            ("B_bar", "B_bar"),
            ("h", "h_horizon"),
            ("y", "y_horizon"),
        ]:
            lo, hi = boxes[cert_key].lo.min().item(), boxes[cert_key].hi.max().item()
            assert layer_env[key]["lo"] >= lo - 1e-4, f"{key} below certified lo"
            assert layer_env[key]["hi"] <= hi + 1e-4, f"{key} above certified hi"


def test_state_bound_recursions_agree_in_the_limit():
    A_bar = Interval(torch.full((4,), 0.5), torch.full((4,), 0.9))
    drive = Interval(torch.full((4,), -1.0), torch.full((4,), 1.0))
    geom = iv.state_bound_geometric(A_bar, drive).abs_max()
    long = iv.state_bound_unrolled(A_bar, drive, 400).abs_max()
    assert torch.allclose(long, geom, rtol=1e-3)
    short = iv.state_bound_unrolled(A_bar, drive, 3).abs_max()
    assert torch.all(short < geom)
