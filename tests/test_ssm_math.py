"""Tests for the state-space mathematics itself, independent of any trained weights."""

from __future__ import annotations

import math

import pytest
import torch

from src.models.mamba import (
    SILU_ARGMIN,
    SILU_MIN,
    LOBMamba,
    SelectiveSSMBlock,
    zoh_phi,
)


@pytest.fixture
def block() -> SelectiveSSMBlock:
    torch.manual_seed(0)
    return SelectiveSSMBlock(d_model=16, d_state=8, expand=2).eval()


def test_zoh_phi_matches_definition_away_from_zero():
    u = torch.cat([torch.linspace(-30.0, -1e-3, 500), torch.linspace(1e-3, 5.0, 200)])
    assert torch.allclose(zoh_phi(u), torch.expm1(u) / u, atol=1e-6)


def test_zoh_phi_is_continuous_at_zero():
    assert abs(zoh_phi(torch.zeros(1)).item() - 1.0) < 1e-9
    tiny = torch.tensor([-1e-9, 1e-9, -1e-6, 1e-6])
    assert torch.all(torch.isfinite(zoh_phi(tiny)))
    assert torch.allclose(zoh_phi(tiny), torch.ones(4), atol=1e-5)


def test_zoh_phi_gradient_is_finite_at_zero():
    u = torch.zeros(1, requires_grad=True)
    zoh_phi(u).backward()
    assert torch.isfinite(u.grad).all()


def test_discrete_matrices_match_exact_zoh(block):
    x = torch.randn(3, 12, block.d_model)
    _, tr = _trace_single(block, x)

    A = tr.A
    dt = tr.delta.unsqueeze(-1)
    expected_A_bar = torch.exp(dt * A)
    # ZOH: Bbar = A^-1 (exp(dt A) - I) B. A is diagonal and strictly negative, so
    # the elementwise division is safe.
    expected_B_bar = (
        (torch.exp(dt * A) - 1.0) / A * tr.B.unsqueeze(2)
    )

    assert torch.allclose(tr.A_bar, expected_A_bar, atol=1e-6)
    assert torch.allclose(tr.B_bar, expected_B_bar, atol=1e-6)


def test_euler_shortcut_is_materially_different(block):
    x = torch.randn(2, 8, block.d_model)
    _, tr = _trace_single(block, x)
    euler = tr.delta.unsqueeze(-1) * tr.B.unsqueeze(2)
    rel = (euler - tr.B_bar).abs() / tr.B_bar.abs().clamp_min(1e-12)
    assert rel.max() > 0.01, "exact ZOH should differ measurably from Euler"


def test_A_is_strictly_negative_for_any_parameter_value(block):
    with torch.no_grad():
        block.A_log.uniform_(-20.0, 20.0)
    assert torch.all(-torch.exp(block.A_log) < 0)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3, 1e6])
def test_delta_stays_inside_its_declared_range(block, scale):
    pre = torch.randn(64, block.d_inner) * scale
    delta = block.delta_from_pre(pre)
    assert torch.all(delta >= block.dt_min)
    assert torch.all(delta <= block.dt_max)


def test_delta_endpoints_are_attained_under_saturation(block):
    # float32 sigmoid saturates to exactly 0 or 1 here, so the dt range is closed, not
    # open, and contraction_certificate has to use sup Abar = exp(-dt_min * min|A|).
    extreme = torch.tensor([[-1e4], [1e4]]).expand(2, block.d_inner)
    delta = block.delta_from_pre(extreme)
    assert delta[0].max().item() == pytest.approx(block.dt_min, rel=1e-6)
    assert delta[1].min().item() == pytest.approx(block.dt_max, rel=1e-6)


def test_delta_is_monotone_in_the_preactivation(block):
    pre = torch.linspace(-50.0, 50.0, 1000).unsqueeze(1).expand(-1, block.d_inner)
    delta = block.delta_from_pre(pre)
    assert torch.all(delta[1:] - delta[:-1] >= -1e-9)


def test_softplus_parametrisation_has_infimum_zero():
    b = SelectiveSSMBlock(d_model=16, d_state=8, dt_parametrisation="softplus").eval()
    tiny = b.delta_from_pre(torch.full((1, b.d_inner), -60.0))
    assert tiny.max().item() < 1e-20
    cert = b.contraction_certificate()
    assert cert["sup_A_bar"] == 1.0
    assert cert["geometric_gain"] == math.inf
    assert not cert["contractive"]


def test_bounded_parametrisation_is_contractive():
    b = SelectiveSSMBlock(d_model=16, d_state=8, dt_parametrisation="bounded").eval()
    cert = b.contraction_certificate()
    assert cert["contractive"]
    assert 0.0 < cert["sup_A_bar"] < 1.0
    assert math.isfinite(cert["geometric_gain"])


def test_horizon_gain_is_below_geometric_gain_and_finite_when_supremum_is_one():
    bounded = SelectiveSSMBlock(d_model=16, d_state=8).eval()
    assert bounded.horizon_gain(100) < bounded.contraction_certificate()["geometric_gain"]

    softplus = SelectiveSSMBlock(d_model=16, d_state=8, dt_parametrisation="softplus").eval()
    assert softplus.horizon_gain(100) == 100.0


def test_A_bar_lies_strictly_inside_the_unit_interval(block):
    x = torch.randn(4, 20, block.d_model) * 10.0
    _, tr = _trace_single(block, x)
    assert torch.all(tr.A_bar > 0.0)
    assert torch.all(tr.A_bar < 1.0)


def test_conv_and_scan_are_causal(block):
    torch.manual_seed(1)
    L, t = 16, 9
    x = torch.randn(1, L, block.d_model)
    x2 = x.clone()
    x2[:, t] += 5.0

    with torch.no_grad():
        y1 = block(x)
        y2 = block(x2)
    assert torch.allclose(y1[:, :t], y2[:, :t], atol=1e-6)
    assert not torch.allclose(y1[:, t], y2[:, t], atol=1e-6)


def test_sequences_in_a_batch_do_not_interact(block):
    x = torch.randn(8, 20, block.d_model)
    with torch.no_grad():
        batched = block(x)
        solo = block(x[2:3])
    assert torch.allclose(batched[2], solo[0], atol=1e-6)


def test_trace_reproduces_the_forward_pass(block):
    x = torch.randn(3, 15, block.d_model)
    with torch.no_grad():
        _, tr = block(x, trace=True)

    h = torch.zeros(x.shape[0], block.d_inner, block.d_state)
    for t in range(x.shape[1]):
        h = tr.A_bar[:, t] * h + tr.B_bar[:, t] * tr.u[:, t].unsqueeze(-1)
        y = torch.einsum("bdn,bn->bd", h, tr.C[:, t]) + block.D * tr.u[:, t]
        assert torch.allclose(y, tr.y[:, t], atol=1e-5)
        assert torch.allclose(h, tr.h[:, t], atol=1e-5)


def test_model_is_deterministic_in_eval_mode():
    torch.manual_seed(0)
    model = LOBMamba(input_dim=43).eval()
    x = torch.randn(4, 30, 43)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_model_forward_shape_and_gradients():
    torch.manual_seed(0)
    model = LOBMamba(input_dim=43)
    x = torch.randn(5, 30, 43)
    out = model(x)
    assert out.shape == (5,)

    torch.nn.functional.mse_loss(out, torch.randn(5)).backward()
    assert model.head[3].weight.grad is not None
    assert model.proj[0].weight.grad is not None
    assert model.layers[0].A_log.grad is not None
    assert model.layers[0].dt_proj.bias.grad is not None


def test_config_is_actually_honoured():
    cfg = {
        "model": {
            "mamba": {
                "d_model": 32,
                "d_state": 4,
                "expand": 3,
                "num_layers": 3,
                "dt_min": 5e-3,
                "dt_max": 5e-2,
            }
        }
    }
    model = LOBMamba(input_dim=11, config=cfg)
    arch = model.architecture()
    assert arch["d_model"] == 32
    assert arch["d_state"] == 4
    assert arch["expand"] == 3
    assert arch["d_inner"] == 96
    assert arch["num_layers"] == 3
    assert len(model.layers) == 3
    assert model.layers[0].dt_min == 5e-3
    assert model(torch.randn(2, 10, 11)).shape == (2,)


def test_conv_is_depthwise():
    block = SelectiveSSMBlock(d_model=16, d_state=8, expand=2)
    assert block.conv1d.groups == block.d_inner
    assert block.conv1d.weight.numel() == block.d_inner * block.d_conv


def test_silu_constants_are_correct():
    x = torch.linspace(-10, 10, 200001)
    y = torch.nn.functional.silu(x)
    assert abs(y.min().item() - SILU_MIN) < 1e-6
    assert abs(x[y.argmin()].item() - SILU_ARGMIN) < 1e-3


def _trace_single(block: SelectiveSSMBlock, x: torch.Tensor):
    with torch.no_grad():
        return block(x, trace=True)
