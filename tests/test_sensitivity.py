"""Soundness of the two local domains, and the ordering between local and global bounds.

Both local domains diverge on the full model at realistic perturbation sizes, which is the
finding recorded in docs/03-certificate.md. They are still required to be sound wherever they
return a finite answer, and these tests hold them to that on a small model where the numbers
stay finite.
"""

import math

import pytest
import torch

from src.models.mamba import LOBMamba
from src.verification.certify import certify_output_range
from src.verification.intervals import Interval, layernorm_input_box
from src.verification.lipschitz import Sens, layer_norm, linear, local_sensitivity, mul, silu
from src.verification.sensitivity import certified_output

SMALL = {"model": {"mamba": {"d_model": 16, "d_state": 4, "expand": 2, "num_layers": 1}}}


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(11)
    return LOBMamba(input_dim=8, config=SMALL).eval()


def sample_ball(x, eps, n):
    return [x + torch.where(torch.rand_like(x) > 0.5, eps, -eps) for _ in range(n)]


@pytest.mark.parametrize("eps", [1e-7, 1e-6, 1e-5])
def test_lipschitz_bound_is_sound(model, eps):
    x = torch.randn(2, 12, 8)
    r = local_sensitivity(model, x, eps)
    if not torch.isfinite(r["lipschitz"]).all():
        pytest.skip("bound diverged at this eps, which the domain comparison documents")
    with torch.no_grad():
        for xp in sample_ball(x, eps, 40):
            y = model(xp)
            assert (y >= r["lo"] - 1e-9).all() and (y <= r["hi"] + 1e-9).all()


@pytest.mark.parametrize("eps", [1e-7, 1e-6])
def test_interval_propagation_is_sound(model, eps):
    x = torch.randn(2, 12, 8)
    out = certified_output(model, Interval(x - eps, x + eps))
    if not torch.isfinite(out.lo).all():
        pytest.skip("bound diverged at this eps")
    with torch.no_grad():
        for xp in sample_ball(x, eps, 40):
            assert out.contains(model(xp).unsqueeze(-1), atol=1e-9).all()


def test_lipschitz_interval_brackets_the_nominal(model):
    x = torch.randn(3, 12, 8)
    r = local_sensitivity(model, x, 1e-7)
    with torch.no_grad():
        y = model(x)
    assert (r["lo"] <= y + 1e-9).all() and (r["hi"] >= y - 1e-9).all()


def test_global_bound_contains_every_local_bound(model):
    """The unconditional range holds for all inputs, so it must contain any local claim."""
    g = certify_output_range(model)
    x = torch.randn(2, 12, 8)
    r = local_sensitivity(model, x, 1e-7)
    assert (r["lo"] >= g.lo.item() - 1e-6).all()
    assert (r["hi"] <= g.hi.item() + 1e-6).all()


def test_layernorm_input_box_is_sound():
    torch.manual_seed(12)
    ln = torch.nn.LayerNorm(24)
    with torch.no_grad():
        ln.weight.normal_(1.0, 0.3)
        ln.bias.normal_(0.0, 0.2)
    for eps in (1e-5, 1e-3, 1e-1):
        x0 = torch.randn(4, 24)
        out = layernorm_input_box(Interval(x0 - eps, x0 + eps), ln)
        for _ in range(200):
            x = x0 + (torch.rand_like(x0) * 2 - 1) * eps
            with torch.no_grad():
                assert out.contains(ln(x)).all()


def test_layernorm_sensitivity_is_sound():
    torch.manual_seed(13)
    ln = torch.nn.LayerNorm(24)
    with torch.no_grad():
        ln.weight.normal_(1.0, 0.3)
        ln.bias.normal_(0.0, 0.2)
    eps = 1e-6
    x0 = torch.randn(4, 24)
    a = Sens(x0, torch.ones_like(x0))
    out = layer_norm(a, ln, eps)
    lo, hi = out.box(eps)
    for _ in range(200):
        x = x0 + (torch.rand_like(x0) * 2 - 1) * eps
        with torch.no_grad():
            y = ln(x)
        assert (y >= lo - 1e-9).all() and (y <= hi + 1e-9).all()


def test_product_rule_keeps_the_second_order_term():
    """Dropping it would make the bound first-order and unsound at finite eps."""
    eps = 0.1
    a = Sens(torch.tensor([2.0]), torch.tensor([1.0]))
    b = Sens(torch.tensor([3.0]), torch.tensor([1.0]))
    out = mul(a, b, eps)
    first_order = a.val.abs() * b.lip + b.val.abs() * a.lip
    assert out.lip.item() > first_order.item()
    worst = (2.0 + eps) * (3.0 + eps)
    assert out.val.item() + out.lip.item() * eps >= worst - 1e-9


def test_silu_derivative_sup_covers_the_interior_maximum():
    a = Sens(torch.tensor([0.0]), torch.tensor([1.0]))
    out = silu(a, 10.0)
    assert out.lip.item() == pytest.approx(1.0998393201, rel=1e-6)


def test_linear_sensitivity_matches_the_row_sum():
    torch.manual_seed(14)
    layer = torch.nn.Linear(6, 3)
    a = Sens(torch.randn(2, 6), torch.ones(2, 6))
    out = linear(a, layer.weight, layer.bias)
    assert torch.allclose(out.lip, layer.weight.abs().sum(1).expand(2, 3), atol=1e-6)
