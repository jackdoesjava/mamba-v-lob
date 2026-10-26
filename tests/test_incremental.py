"""The incremental bound: the scalar lemma, the cell theorems, and the block certificate.

The lemma and the cell theorems run in float64 on constructed sequences, with timescales down
to 1e-14 and no floor. The block certificate is checked link by link against perturbed forward
passes of both trained checkpoints in float64. The end-to-end bound is about seven orders of
magnitude above what sampling reaches, so a test of the output alone would pass whatever the
middle did; each intermediate bound is held to the realised values on its own.
"""

from __future__ import annotations

import copy
import itertools
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mamba import LOBMamba, SelectiveSSMBlock
from src.utils.repro import REPO_ROOT
from src.verification.certify import certify_block
from src.verification.incremental import (
    RULES,
    VARIANTS,
    block_boxes,
    cell_bound,
    cell_bound_common_start,
    contribution,
    decay_gap,
    decay_gap_bound,
    delta_slope,
    euler_cell_bound,
    incremental_certificate,
    layernorm_jacobian,
    layernorm_lipschitz,
    layernorm_realised,
    local_block_sensitivity,
    local_sensitivity_rule,
    local_step,
    local_step_recursive,
    log_delta_slope,
    realised_gap,
    softplus_log_slope,
    weight_free_factor,
)
from src.verification.intervals import Interval
from src.verification.lipschitz import local_sensitivity

D64 = torch.float64
SMALL = {"model": {"mamba": {"d_model": 16, "d_state": 4, "expand": 2, "num_layers": 1}}}
TWO_LAYERS = {"model": {"mamba": {"d_model": 16, "d_state": 4, "expand": 2, "num_layers": 2}}}
CHECKPOINTS = {
    "bounded": REPO_ROOT / "models/checkpoints/best_mamba.pt",
    "softplus": REPO_ROOT / "models/checkpoints/best_mamba_softplus.pt",
}


# --- the scalar lemma -------------------------------------------------------------------------


def test_decay_gap_survives_where_the_naive_difference_cancels():
    x = torch.tensor([1e-12, 1e-15, 1e-18], dtype=D64)
    s = torch.tensor(0.1, dtype=D64)
    stable = decay_gap(x, s)
    first_order = x * math.expm1(0.1)              # exact to O(x^2) at these x
    assert torch.allclose(stable, first_order, rtol=1e-9, atol=0.0)
    naive = (torch.exp(-x) - torch.exp(-x * math.exp(0.1))).abs()
    assert ((naive - first_order).abs() / first_order).max() > 1e-3


@pytest.mark.parametrize("rho", [1e-8, 1e-4, 1e-2, 0.3, 1.0, 3.0])
def test_scalar_lemma_holds_on_a_grid(rho):
    x = torch.logspace(-14, 2, 400, dtype=D64).unsqueeze(1)
    s = torch.cat([torch.linspace(-rho, rho, 201, dtype=D64),
                   torch.tensor([-rho, rho], dtype=D64)]).unsqueeze(0)
    gap = decay_gap(x, s)
    bound = decay_gap_bound(x, rho)
    assert torch.all(gap <= bound * (1.0 + 1e-12))
    # the exchanged form the local corollary uses: normalised by 1 - a~ instead of 1 - a
    exchanged = decay_gap_bound(x * torch.exp(s), rho)
    assert torch.all(gap <= exchanged * (1.0 + 1e-12))


@pytest.mark.parametrize("rho", [1e-6, 0.1, 1.0, 3.0])
def test_scalar_lemma_is_tight_as_x_vanishes(rho):
    x = torch.tensor([1e-13, 1e-11], dtype=D64)
    ratio = decay_gap(x, torch.tensor(rho, dtype=D64)) / -torch.expm1(-x)
    assert torch.allclose(ratio, torch.full_like(ratio, math.expm1(rho)), rtol=1e-8)
    # and strictly loose away from 0, where exp(-x) is far from 1
    far = torch.tensor([1.0], dtype=D64)
    assert decay_gap(far, torch.tensor(rho, dtype=D64)) < decay_gap_bound(far, rho)


def test_scalar_lemma_at_high_precision():
    mp = pytest.importorskip("mpmath")
    # the smallest gap below is 1e-50 next to a leading 1, so carry 80 digits to resolve it
    mp.mp.dps = 80
    for x, rho in itertools.product(["1e-30", "1e-12", "1e-3", "0.5", "7"], ["1e-20", "0.01", "1"]):
        x, rho = mp.mpf(x), mp.mpf(rho)
        for frac in (-1, -0.5, 0, 0.5, 1):
            s = rho * frac
            gap = abs(mp.exp(-x) - mp.exp(-x * mp.exp(s)))
            assert gap <= (mp.exp(rho) - 1) * (1 - mp.exp(-x))


# --- the cell theorems on constructed sequences -----------------------------------------------


def _scan(lam, dt, c, h0):
    """Exact ZOH in convex form, h_t = a h_{t-1} + (1 - a) c_t, with 1 - a from expm1."""
    x = lam.unsqueeze(1) * dt
    a, one_minus_a = torch.exp(-x), -torch.expm1(-x)
    h, out = h0.clone(), []
    for t in range(dt.shape[1]):
        h = a[:, t] * h + one_minus_a[:, t] * c[:, t]
        out.append(h)
    return torch.stack(out, 1)


def _random_cells(n, T, seed, rhos=(1e-6, 1e-3, 0.1, 0.5, 2.0)):
    g = torch.Generator().manual_seed(seed)
    lam = 10 ** (torch.rand(n, generator=g, dtype=D64) * 2.7 - 1.0)       # [0.1, 50]
    dt = 10 ** (torch.rand(n, T, generator=g, dtype=D64) * 15.0 - 14.0)   # [1e-14, 10]
    rho = torch.tensor(rhos, dtype=D64).repeat(n // len(rhos) + 1)[:n]
    s = (torch.rand(n, T, generator=g, dtype=D64) * 2 - 1) * rho.unsqueeze(1)
    dc = 10 ** (torch.rand(n, generator=g, dtype=D64) * 4 - 4)
    c_t = torch.rand(n, T, generator=g, dtype=D64) * 2 - 1                # |c~| <= M = 1
    c = c_t + (torch.rand(n, T, generator=g, dtype=D64) * 2 - 1) * dc.unsqueeze(1)
    return lam, dt, s, rho, dc, c, c_t, g


def test_cell_theorem_on_random_sequences():
    n, T = 2000, 300
    lam, dt, s, rho, dc, c, c_t, g = _random_cells(n, T, seed=0)
    for start in ("zero", "random"):
        if start == "zero":
            h0, ht0 = torch.zeros(n, dtype=D64), torch.zeros(n, dtype=D64)
        else:
            ht0 = torch.rand(n, generator=g, dtype=D64) * 2 - 1
            h0 = ht0 + torch.randn(n, generator=g, dtype=D64)
        e = _scan(lam, dt, c, h0) - _scan(lam, dt * torch.exp(s), c_t, ht0)
        bound = cell_bound(dc, 1.0, rho, h0 - ht0)
        assert torch.all(e.abs().amax(1) <= bound * (1 + 1e-12) + 1e-15)


def test_common_start_bound_on_random_sequences():
    """Proposition (sharp constant), part (i): from a common start the time-warp term is
    2 M tanh(rho/2), smaller than the cell theorem's by e^rho + 1."""
    n, T = 2000, 300
    lam, dt, s, rho, dc, c, c_t, g = _random_cells(n, T, seed=4, rhos=(1e-6, 1e-2, 0.3, 1.0, 4.0))
    for h0 in (torch.zeros(n, dtype=D64), torch.rand(n, generator=g, dtype=D64) * 2 - 1):
        e = _scan(lam, dt, c, h0) - _scan(lam, dt * torch.exp(s), c_t, h0)
        bound = cell_bound_common_start(dc, 1.0, rho)
        assert torch.all(e.abs().amax(1) <= bound * (1 + 1e-12) + 1e-15)
        assert torch.all(bound <= cell_bound(dc, 1.0, rho) * (1 + 1e-12))


def _greedy(rho, dc, T=1000, x0=1e-2, h0=0.0):
    """Each step picks the sign of s, c~ and c - c~ that maximises |e_t|."""
    n = rho.shape[0]
    x = torch.full((n,), x0, dtype=D64)
    h = torch.full((n,), h0, dtype=D64)
    ht, worst = h.clone(), torch.zeros(n, dtype=D64)
    for _ in range(T):
        cands = []
        for ss, cs, ds in itertools.product((-1.0, 1.0), repeat=3):
            xt = x * torch.exp(ss * rho)
            ct = torch.full((n,), cs, dtype=D64)
            cands.append((torch.exp(-x) * h - torch.expm1(-x) * (ct + ds * dc),
                          torch.exp(-xt) * ht - torch.expm1(-xt) * ct))
        hs = torch.stack([c[0] for c in cands])
        hts = torch.stack([c[1] for c in cands])
        k = (hs - hts).abs().argmax(0)
        h, ht = hs[k, torch.arange(n)], hts[k, torch.arange(n)]
        worst = torch.maximum(worst, (h - ht).abs())
    return worst


def test_cell_theorem_against_a_greedy_adversary():
    """The bound must hold, and it should not be vacuous: on a long horizon the adversary
    reaches about half of it, the factor 2 being |h~ - c~| <= 2M, which is only attained
    momentarily."""
    rho = torch.tensor([1e-3, 1e-3, 0.1, 0.1], dtype=D64)
    worst = _greedy(rho, 1e-3)
    bound = cell_bound(1e-3, 1.0, rho)
    assert torch.all(worst <= bound)
    assert torch.all(worst >= 0.4 * bound)


def test_common_start_bound_against_a_greedy_adversary():
    rho = torch.tensor([1e-3, 0.1, 1.0, 3.0], dtype=D64)
    for dc in (0.0, 1e-3):
        worst = _greedy(rho, dc)
        assert torch.all(worst <= cell_bound_common_start(dc, 1.0, rho) * (1 + 1e-12))


@pytest.mark.parametrize("rho", [0.1, 1.0, 2.0])
def test_sharp_constant_is_approached_by_chattering(rho):
    """Proposition (sharp constant), part (ii): with c = c~ alternating between M and -M and the
    timescales of the two runs swapped between lam dt = K eps and eps, the periodic gap is
    M (1 + a)(1 + b)(tanh(K eps/2) - tanh(eps/2))/(1 - a b), which tends to 2 M tanh(rho/2)."""
    K, eps = math.exp(rho), 1e-2
    a, b = math.exp(-K * eps), math.exp(-eps)
    closed = (1 + a) * (1 + b) * (math.tanh(K * eps / 2) - math.tanh(eps / 2)) / (1 - a * b)
    h = ht = 0.0
    worst = 0.0
    for _ in range(int(60 / (eps * (K + 1))) + 10):
        h, ht = a * h + (1 - a), b * ht + (1 - b)            # c = +M: nominal fast, perturbed slow
        after_plus = abs(h - ht)
        worst = max(worst, after_plus)
        h, ht = b * h - (1 - b), a * ht - (1 - a)            # c = -M: the other way round
        worst = max(worst, abs(h - ht))
    assert after_plus == pytest.approx(closed, rel=1e-9)     # the two-step orbit is reached
    sharp = float(cell_bound_common_start(0.0, 1.0, rho))
    assert worst <= sharp * (1 + 1e-12)
    assert closed >= 0.999 * sharp


def _budgets(n, T, g, rho_max):
    """Per-step budgets rho_t and realised log-ratios s_t with |s_t| <= rho_t."""
    rho_t = torch.rand(n, T, generator=g, dtype=D64) * rho_max.unsqueeze(1)
    s = rho_t * (torch.rand(n, T, generator=g, dtype=D64) * 2 - 1)
    s = torch.where(torch.rand(n, T, generator=g) > 0.7, rho_t * torch.sign(s), s)
    return rho_t, s


def test_local_corollary_on_random_sequences():
    """Both local steps, with per-step budgets up to rho = 5, where e^rho - 1 is about 150, and
    from different starts. Nothing bounds the perturbed run."""
    n, T = 2000, 300
    lam, dt, _, _, dc, c, c_t, g = _random_cells(n, T, seed=1)
    rho_max = torch.tensor([1e-6, 1e-2, 0.5, 2.0, 5.0], dtype=D64).repeat(n // 5)
    rho_t, s = _budgets(n, T, g, rho_max)
    c_t = c_t * (1 + 9 * torch.rand(n, 1, generator=g, dtype=D64))     # |c~| up to 10, no M
    c = c_t + (torch.rand(n, T, generator=g, dtype=D64) * 2 - 1) * dc.unsqueeze(1)
    ht0 = torch.randn(n, generator=g, dtype=D64)
    h0 = ht0 + torch.randn(n, generator=g, dtype=D64) * 0.1
    h = _scan(lam, dt, c, h0)
    ht = _scan(lam, dt * torch.exp(s), c_t, ht0)
    h_prev = torch.cat([h0.unsqueeze(1), h[:, :-1]], 1)
    x = lam.unsqueeze(1) * dt
    E_max = E_rec = (h0 - ht0).abs()
    for t in range(T):
        gap = (h_prev[:, t] - c[:, t]).abs()
        dct = (c[:, t] - c_t[:, t]).abs()
        E_max = local_step(E_max, torch.expm1(rho_t[:, t]), dct, gap)
        E_rec = local_step_recursive(E_rec, x[:, t], rho_t[:, t], dct, gap)
        e = (h[:, t] - ht[:, t]).abs()
        assert torch.all(e <= E_rec * (1 + 1e-12) + 1e-15)
        assert torch.all(E_rec <= E_max * (1 + 1e-12))


def test_local_recursion_is_the_exact_error_without_a_timescale_perturbation():
    """With rho = 0 and c - c~ of constant sign, the recursion reproduces |e_t| itself."""
    n, T = 500, 200
    lam, dt, _, _, dc, _, c_t, _ = _random_cells(n, T, seed=2)
    c = c_t + dc.unsqueeze(1)
    e = (_scan(lam, dt, c, torch.zeros(n, dtype=D64))
         - _scan(lam, dt, c_t, torch.zeros(n, dtype=D64)))
    x = lam.unsqueeze(1) * dt
    E = torch.zeros(n, dtype=D64)
    zero = torch.zeros(n, dtype=D64)
    for t in range(T):
        E = local_step_recursive(E, x[:, t], zero, dc, zero)
        assert torch.allclose(E, e[:, t].abs(), rtol=1e-10, atol=1e-15)


def test_local_steps_survive_overflow_and_report_no_bound_for_infinite_input():
    E = torch.tensor([0.0, 1.0, math.inf])
    rho = torch.tensor([200.0, 200.0, 1.0])                  # e^200 - 1 overflows float32
    gap = torch.tensor([0.0, 0.5, 0.5])
    dc = torch.tensor([1e-3, 1e-3, 1e-3])
    out_max = local_step(E, torch.expm1(rho), dc, gap)
    out_rec = local_step_recursive(E, torch.full_like(E, 0.1), rho, dc, gap)
    for out in (out_max, out_rec):
        assert not torch.isnan(out).any()
        assert float(out[0]) == pytest.approx(1e-3)          # 0 distance: the overflow is moot
        assert math.isinf(out[1]) and out[1] > 0             # a real overflow is no bound
        assert math.isinf(out[2]) and out[2] > 0             # and so is an infinite input


def test_additive_baseline_is_a_valid_bound_on_random_sequences():
    """The comparison is between two valid bounds: check the baseline's closed form too."""
    n, T = 1000, 300
    g = torch.Generator().manual_seed(2)
    lam = 10 ** (torch.rand(n, generator=g, dtype=D64) * 2 - 1)
    floor = 10 ** (torch.rand(n, generator=g, dtype=D64) * 6 - 7)
    dt = floor.unsqueeze(1) * 10 ** (torch.rand(n, T, generator=g, dtype=D64) * 3)
    step = floor.unsqueeze(1) * torch.rand(n, T, generator=g, dtype=D64)   # |dt - dt~|
    dt_t = dt + step * torch.where(torch.rand(n, T, generator=g) > 0.5, 1.0, -1.0).to(D64)
    dt_t = torch.maximum(dt_t, floor.unsqueeze(1))
    c_t = torch.rand(n, T, generator=g, dtype=D64) * 2 - 1
    c = c_t + (torch.rand(n, T, generator=g, dtype=D64) * 2 - 1) * 1e-3
    e = _scan(lam, dt, c, torch.zeros(n, dtype=D64)) - _scan(lam, dt_t, c_t, torch.zeros(n, dtype=D64))
    eta = lam * torch.exp(-lam * floor) * (dt - dt_t).abs().amax(1)
    x = lam * floor
    for L in (1, 10, 100, T):
        S = torch.expm1(-L * x) / torch.expm1(-x)
        assert torch.all(e[:, L - 1].abs() <= 1e-3 + 2.0 * eta * S + 1e-15)


def test_euler_variant_on_random_sequences():
    """Released Mamba's Bbar = dt B: c = g(lam dt) B u / lam with g(x) = x/(1 - e^-x)."""
    n, T = 2000, 300
    lam, dt, s, rho, dc, _, b_t, g = _random_cells(n, T, seed=3)
    b = b_t + (torch.rand(n, T, generator=g, dtype=D64) * 2 - 1) * dc.unsqueeze(1)
    dt_t = dt * torch.exp(s)

    def euler(dts, bs):
        h, out = torch.zeros(n, dtype=D64), []
        for t in range(T):
            h = torch.exp(-lam * dts[:, t]) * h + lam * dts[:, t] * bs[:, t]
            out.append(h)
        return torch.stack(out, 1)

    e = euler(dt, b) - euler(dt_t, b_t)
    x_hi = lam * torch.maximum(dt, dt_t).amax(1)
    assert torch.all(e.abs().amax(1) <= euler_cell_bound(dc, 1.0, rho, x_hi) * (1 + 1e-12))


# --- the timescale slopes ---------------------------------------------------------------------


def test_log_softplus_slope_is_in_the_unit_interval_and_decreasing():
    z = torch.linspace(-700.0, 700.0, 20001, dtype=D64, requires_grad=True)
    torch.log(F.softplus(z)).sum().backward()
    slope = softplus_log_slope(z.detach())
    assert torch.all(slope > 0) and torch.all(slope <= 1.0)
    assert torch.all(slope[1:] <= slope[:-1] + 1e-15)
    # F.softplus returns z above its threshold of 20, which moves the ratio by sigmoid(20) - 1
    assert torch.allclose(slope, z.grad, rtol=1e-8, atol=1e-300)
    # far below where both terms underflow, the supremum 1 is returned, never less
    deep = softplus_log_slope(torch.tensor([-800.0, -1e4, -40.5], dtype=D64))
    assert torch.all(deep == 1.0)


def test_bounded_log_slope_is_at_most_a_quarter_log_ratio():
    block = SelectiveSSMBlock(d_model=16, d_state=4, dt_min=1e-3, dt_max=1e-1).double()
    z = torch.linspace(-40.0, 40.0, 40001, dtype=D64, requires_grad=True)
    torch.log(block.delta_from_pre(z)).sum().backward()
    assert z.grad.abs().max() <= math.log(100.0) / 4 * (1 + 1e-12)
    box = log_delta_slope(block, Interval(z.detach().min().view(1), z.detach().max().view(1)))
    assert box.item() == pytest.approx(math.log(100.0) / 4, rel=1e-12)


@pytest.mark.parametrize("arm", ["bounded", "softplus"])
def test_delta_slope_matches_a_dense_grid(arm):
    block = SelectiveSSMBlock(d_model=16, d_state=4, dt_parametrisation=arm).double()
    for lo, hi in ((-30.0, 30.0), (-30.0, -2.0), (1.0, 6.0), (2.5, 3.0)):
        z = torch.linspace(lo, hi, 200001, dtype=D64, requires_grad=True)
        block.delta_from_pre(z).sum().backward()
        bound = delta_slope(block, Interval(torch.tensor([lo], dtype=D64),
                                            torch.tensor([hi], dtype=D64)))
        assert z.grad.max() <= bound.item() * (1 + 1e-12)
        assert z.grad.max() >= bound.item() * (1 - 1e-6)


def test_weight_free_factor_at_a_ratio_of_100():
    """sup|dt'| = 29.70 dt_min against dt_min ln(100)/4 for (log dt)', a factor of 25.80."""
    block = SelectiveSSMBlock(d_model=16, d_state=4, dt_min=1e-3, dt_max=1e-1).double()
    z = torch.linspace(-40.0, 40.0, 400001, dtype=D64, requires_grad=True)
    block.delta_from_pre(z).sum().backward()
    assert float(z.grad.max()) / 1e-3 == pytest.approx(29.70, abs=5e-3)
    on_grid = float(z.grad.max()) / (1e-3 * math.log(100) / 4)
    # the closed form is the supremum, so the grid can only approach it from below
    assert on_grid <= weight_free_factor(block) * (1 + 1e-12)
    assert weight_free_factor(block) == pytest.approx(on_grid, rel=1e-7)
    assert weight_free_factor(block) == pytest.approx(25.80, abs=5e-3)
    assert weight_free_factor(SelectiveSSMBlock(d_model=16, dt_parametrisation="softplus")) is None


# --- LayerNorm and the realised distance to the equilibrium -----------------------------------


def test_layernorm_jacobian_matches_autograd_and_its_bounds():
    torch.manual_seed(20)
    ln = nn.LayerNorm(12).double()
    with torch.no_grad():
        ln.weight.uniform_(-1.5, 1.5)
        ln.bias.normal_()
    x = torch.cat([torch.randn(5, 12, dtype=D64) * 3.0,
                   0.3 + 1e-4 * torch.randn(2, 12, dtype=D64)])       # two nearly constant rows
    J = layernorm_jacobian(ln, x).detach()
    for i in range(x.shape[0]):
        ref = torch.autograd.functional.jacobian(ln, x[i])
        torch.testing.assert_close(J[i], ref, rtol=1e-9, atol=1e-9)
    lip = layernorm_lipschitz(ln)
    norms = torch.linalg.matrix_norm(J, ord=2)
    assert torch.all(norms <= lip["l2_global_bound"] * (1 + 1e-12))
    assert lip["l2_at_constant_input"] <= lip["l2_global_bound"] * (1 + 1e-12)
    assert lip["l2_at_constant_input"] >= 0.5 * lip["l2_global_bound"]
    real = layernorm_realised(ln, x[:5], chunk=2)
    assert real["l2_max"] == pytest.approx(float(norms[:5].max()), rel=1e-12)
    assert real["l2_max"] <= real["l2_bound_max"] * (1 + 1e-12)


def test_realised_gap_matches_the_trace():
    torch.manual_seed(21)
    block = SelectiveSSMBlock(d_model=8, d_state=4).double().eval()
    xn = block.norm(torch.randn(3, 25, 8, dtype=D64) * 2.0).detach()
    bare = copy.deepcopy(block)
    bare.norm = nn.Identity()
    with torch.no_grad():
        _, tr = bare(xn, trace=True)
    lam = torch.exp(bare.A_log)
    c = tr.B.unsqueeze(2) * tr.u.unsqueeze(-1) / lam
    h_prev = torch.cat([torch.zeros_like(tr.h[:, :1]), tr.h[:, :-1]], 1)
    ref = (h_prev - c).abs().amax(dim=(0, 1))
    torch.testing.assert_close(realised_gap(block, xn), ref, rtol=1e-12, atol=1e-15)


# --- the block certificate on the trained checkpoints -----------------------------------------


def _load(arm: str) -> LOBMamba:
    path = CHECKPOINTS[arm]
    if not path.exists():
        pytest.skip(f"{path.name} not found")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(ckpt["config"])
    cfg["model"]["mamba"]["dt_parametrisation"] = arm
    model = LOBMamba(int(ckpt["architecture"]["input_dim"]), cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval()


@pytest.fixture(scope="module", params=["bounded", "softplus"])
def trained(request):
    return request.param, _load(request.param)


@pytest.fixture(scope="module")
def certified(trained):
    """Per layer: the float32 block, its certificate, and a float64 copy without the norm."""
    arm, model = trained
    out = []
    for block in model.layers:
        bare = copy.deepcopy(block).double().eval()
        bare.norm = nn.Identity()
        out.append((block, incremental_certificate(block), bare))
    return arm, out


def _internals(bare: SelectiveSSMBlock, xn: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every quantity the certificate bounds, from one float64 pass that starts at xn."""
    with torch.no_grad():
        out, tr = bare(xn, trace=True)
        z = bare.dt_proj(bare.x_proj(tr.u)[..., : bare.dt_rank])
        lam = torch.exp(bare.A_log)
        return {
            "u": tr.u,
            "gate": F.silu(bare.in_proj(xn)[..., bare.d_inner :]),
            "z": z,
            "log_delta": torch.log(tr.delta),
            "B": tr.B,
            "C": tr.C,
            "c": tr.B.unsqueeze(2) * tr.u.unsqueeze(-1) / lam,
            "h": tr.h,
            "y": tr.y,
            "out": out - xn,
        }


def _check_links(inc, nominal, perturbed, delta, arbitrary):
    """Each realised difference against its own bound, per sample, coordinate and position."""
    lip = inc.global_lipschitz if arbitrary else {
        "u": inc.L_u, "gate": inc.L_gate, "z": inc.L_z, "log_delta": inc.L_log_delta,
        "B": inc.L_B, "C": inc.L_C,
    }
    for i, d in enumerate(delta.tolist()):
        for key in ("u", "gate", "z", "log_delta", "B", "C"):
            realised = (nominal[key][i] - perturbed[key][i]).abs().amax(0)
            assert torch.all(realised <= lip[key] * d * (1 + 1e-9) + 1e-12), key
        checks = [("c", inc.forcing_slope() * d)] if not arbitrary else []
        for variant in VARIANTS:
            checks += [("h", inc.state_bound(d, arbitrary, variant)),
                       ("y", inc.y_bound(d, arbitrary, variant)),
                       ("out", inc.output_bound(d, arbitrary, variant))]
        for key, bound in checks:
            realised = (nominal[key][i] - perturbed[key][i]).abs().amax(0)
            assert torch.all(realised <= bound * (1 + 1e-9) + 1e-12), key


def test_rebuilt_boxes_agree_with_certify_block():
    model = _load("bounded")
    block = model.layers[0]
    cert = certify_block(block, seq_len=1)
    boxes = block_boxes(block)
    for key in ("u", "B", "C", "delta"):
        assert torch.equal(boxes[key].lo, cert["boxes"][key].lo), key
        assert torch.equal(boxes[key].hi, cert["boxes"][key].hi), key
    fast, slow = incremental_certificate(block), incremental_certificate(block, cert=cert)
    for field in ("M", "sup_u", "sup_B", "sup_C", "sup_y", "L_log_delta", "floor", "ceiling"):
        assert torch.equal(getattr(fast, field), getattr(slow, field)), field
    # the certified timescale box in float64 brackets the float32 one certify_block reports
    assert torch.allclose(fast.floor, cert["boxes"]["delta"].lo.double(), rtol=1e-6)
    assert torch.allclose(fast.ceiling, cert["boxes"]["delta"].hi.double(), rtol=1e-6)


def test_contribution_is_the_forward_pass_minus_the_residual(certified):
    _, layers = certified
    block, _, _ = layers[0]
    b64 = copy.deepcopy(block).double()
    x = torch.randn(3, 20, b64.d_model, dtype=D64) * 4.0
    with torch.no_grad():
        assert torch.allclose(b64(x) - x, contribution(b64, b64.norm(x)), atol=1e-12)


@pytest.mark.parametrize("layer", [0, 1])
def test_block_certificate_is_sound_for_layernorm_outputs(certified, layer):
    """Both normalised inputs are LayerNorm outputs: perturb the block's raw input."""
    _, layers = certified
    block, inc, bare = layers[layer]
    ln = copy.deepcopy(block.norm).double()
    torch.manual_seed(100 + layer)
    for x_scale, z_scale in itertools.product((0.3, 3.0, 30.0), (1e-5, 1e-3, 1e-1)):
        x = torch.randn(4, 48, block.d_model, dtype=D64) * x_scale
        x_pert = x + (torch.rand_like(x) * 2 - 1) * z_scale * x_scale
        with torch.no_grad():
            xn, xn_pert = ln(x), ln(x_pert)
        delta = (xn - xn_pert).abs().amax(dim=(1, 2))
        _check_links(inc, _internals(bare, xn), _internals(bare, xn_pert), delta, False)


@pytest.mark.parametrize("layer", [0, 1])
def test_block_certificate_is_sound_for_arbitrary_perturbations(certified, layer):
    """The perturbed normalised input is any point within delta, inside LayerNorm's range or not."""
    _, layers = certified
    block, inc, bare = layers[layer]
    ln = copy.deepcopy(block.norm).double()
    torch.manual_seed(200 + layer)
    for d in (1e-6, 1e-4, 1e-2, 1e-1):
        x = torch.randn(4, 48, block.d_model, dtype=D64) * 3.0
        with torch.no_grad():
            xn = ln(x)
        nominal = _internals(bare, xn)
        for xi in (torch.where(torch.rand_like(xn) > 0.5, d, -d), (torch.rand_like(xn) * 2 - 1) * d):
            delta = xi.abs().amax(dim=(1, 2))
            _check_links(inc, nominal, _internals(bare, xn + xi), delta, True)


def test_cell_theorems_on_trained_dynamics(certified):
    """The theorems themselves, with every constant taken from the two realised trajectories
    rather than from the certified boxes, so they are tested well inside the data-free
    looseness. Both runs start from 0, so the common-start constant applies too."""
    _, layers = certified
    for block, _, bare in layers:
        ln = copy.deepcopy(block.norm).double()
        torch.manual_seed(7)
        x = torch.randn(4, 64, block.d_model, dtype=D64) * 3.0
        with torch.no_grad():
            nominal = _internals(bare, ln(x))
            perturbed = _internals(bare, ln(x + torch.randn_like(x) * 0.05))
        rho = (nominal["log_delta"] - perturbed["log_delta"]).abs().amax(1).unsqueeze(-1)
        dc = (nominal["c"] - perturbed["c"]).abs().amax(1)
        M = torch.maximum(perturbed["h"].abs().amax(1), perturbed["c"].abs().amax(1))
        realised = (nominal["h"] - perturbed["h"]).abs().amax(1)
        assert torch.all(realised <= cell_bound(dc, M, rho) * (1 + 1e-9))
        assert torch.all(realised <= cell_bound_common_start(dc, M, rho) * (1 + 1e-9))


def test_floor_has_to_be_taken_in_float64():
    """Only the softplus floor is below float32 resolution."""
    inc = incremental_certificate(_load("softplus").layers[0])
    x = inc.lam * inc.floor.unsqueeze(1)
    assert inc.floor.min() < 1e-12
    assert torch.any(torch.exp(-x.float()) == 1.0)        # float32 would give 1 - a = 0
    gain = 1.0 / -torch.expm1(-x)
    assert torch.all(torch.isfinite(gain))
    tiny = x < 1e-9                                       # where 1/(1 - e^-x) = (1 + O(x))/x
    assert tiny.any()
    assert torch.allclose(gain[tiny] * x[tiny], torch.ones_like(x[tiny]), rtol=1e-8)


def test_additive_baseline_grows_linearly_and_its_floored_form_converges(certified):
    _, layers = certified
    _, inc, _ = layers[0]
    s100, s1000 = inc.additive_state_slope(100, floor=False), inc.additive_state_slope(1000, floor=False)
    per_step = 2.0 * inc.M * inc.lam * inc.L_delta.unsqueeze(1)
    assert torch.allclose(s1000 - s100, 900.0 * per_step, rtol=1e-9)
    limit = inc.additive_state_slope(None, floor=True)
    assert torch.all(inc.additive_state_slope(5000, floor=True) <= limit * (1 + 1e-12))
    big = torch.exp(-inc.lam * inc.floor.unsqueeze(1)) ** 1e7 < 1e-12
    assert torch.allclose(inc.additive_state_slope(10**15, floor=True)[big], limit[big], rtol=1e-9)


def test_arbitrary_reading_is_never_tighter(certified):
    _, layers = certified
    for _, inc, _ in layers:
        for variant in VARIANTS:
            for d in (1e-8, 1e-4, 1e-1):
                assert torch.all(inc.output_bound(d, True, variant)
                                 >= inc.output_bound(d, False, variant) * (1 - 1e-12))
            slope = inc.output_slope(variant)
            assert torch.allclose(inc.output_bound(1e-9, variant=variant) / 1e-9, slope, rtol=1e-6)
        for d in (1e-8, 1e-4, 1e-1):
            assert torch.all(inc.output_bound(d, variant="common_start")
                             <= inc.output_bound(d) * (1 + 1e-12))


def test_output_slope_parts_add_up(certified):
    _, layers = certified
    for _, inc, _ in layers:
        for variant in VARIANTS:
            parts = inc.output_slope_parts(variant)
            assert set(parts) == {"warp", "forcing", "C", "D", "gate"}
            torch.testing.assert_close(sum(parts.values()), inc.output_slope(variant),
                                       rtol=1e-12, atol=0.0)
        # with a common start the warp part is exactly half the cell theorem's at first order
        torch.testing.assert_close(inc.output_slope_parts("common_start")["warp"],
                                   0.5 * inc.output_slope_parts("cell")["warp"], rtol=1e-12, atol=0.0)


def test_warp_ratio_is_the_ratio_of_the_two_time_warp_terms(certified):
    _, layers = certified
    for _, inc, _ in layers:
        dc = inc.forcing_slope()
        ratio = (inc.additive_state_slope(None, floor=True) - dc) / (inc.state_slope() - dc)
        torch.testing.assert_close(inc.warp_ratio(), ratio, rtol=1e-9, atol=0.0)
        x = inc.lam * inc.floor.unsqueeze(1)
        torch.testing.assert_close(inc.warp_ratio(), inc.first_factor().unsqueeze(1) * x / torch.expm1(x),
                                   rtol=1e-9, atol=0.0)
        # the additive term decreases as the floor rises
        assert torch.all(inc.warp_ratio(inc.floor * 2.0) <= inc.warp_ratio() * (1 + 1e-12))


def test_crossover_horizon_by_brute_force(certified):
    _, layers = certified
    _, inc, _ = layers[0]
    n_star = inc.crossover_horizon()
    g = torch.Generator().manual_seed(5)
    cells = torch.randint(0, inc.lam.numel(), (40,), generator=g)
    for flat in cells.tolist():
        i, j = divmod(flat, inc.lam.shape[1])
        lam, a = float(inc.lam[i, j]), float(inc.lam[i, j] * inc.L_delta[i])
        b, hi = float(inc.L_log_delta[i]), float(inc.ceiling[i])
        n = 1
        while a * n < b * -math.expm1(-lam * hi * n):
            n += 1
        assert n_star[i, j] == n
        # the additive slope without a floor is the forcing plus a n, in units of 2 M
        additive = inc.additive_state_slope(n, floor=False)[i, j] - inc.forcing_slope()[i, j]
        assert float(additive) == pytest.approx(2.0 * float(inc.M[i, j]) * a * n, rel=1e-12)


# --- the local version as a recurrence step ---------------------------------------------------


@pytest.fixture(scope="module")
def small_model():
    torch.manual_seed(11)
    return LOBMamba(input_dim=8, config=SMALL).eval()


@pytest.mark.parametrize("eps", [1e-7, 1e-5, 1e-3])
def test_additive_rule_reproduces_the_norm_propagation_domain(small_model, eps):
    torch.manual_seed(12)
    x = torch.randn(2, 12, 8)
    old = local_sensitivity(small_model, x, eps)
    new = local_sensitivity_rule(small_model, x, eps, "additive")
    for key in ("nominal", "lipschitz", "lo", "hi"):
        torch.testing.assert_close(new[key], old[key], rtol=0.0, atol=0.0, equal_nan=True)


def test_additive_rule_reproduces_the_domain_on_the_trained_model():
    model = _load("bounded")
    torch.manual_seed(13)
    x = torch.randn(2, 24, model.input_dim)
    for eps in (1e-6, 1e-5):
        old = local_sensitivity(model, x, eps)
        new = local_sensitivity_rule(model, x, eps, "additive")
        torch.testing.assert_close(new["lipschitz"], old["lipschitz"], rtol=0.0, atol=0.0,
                                   equal_nan=True)


@pytest.mark.parametrize("rule", ["log_delta", "log_delta_recursive"])
@pytest.mark.parametrize("eps", [1e-7, 1e-6, 1e-5, 1e-3])
def test_log_delta_rules_are_sound(small_model, rule, eps):
    torch.manual_seed(14)
    x = torch.randn(2, 12, 8)
    r = local_sensitivity_rule(small_model, x, eps, rule)
    if not torch.isfinite(r["lipschitz"]).all():
        pytest.skip("no finite bound at this eps")
    with torch.no_grad():
        for _ in range(40):
            y = small_model(x + torch.where(torch.rand_like(x) > 0.5, eps, -eps))
            assert (y >= r["lo"] - 1e-9).all() and (y <= r["hi"] + 1e-9).all()


@pytest.mark.parametrize("eps", [1e-6, 1e-4, 1e-2])
def test_recursive_rule_is_never_looser_than_the_running_maximum(eps):
    torch.manual_seed(15)
    model = LOBMamba(input_dim=8, config=TWO_LAYERS).eval()
    x = torch.randn(2, 30, 8)
    a = local_sensitivity_rule(model, x, eps, "log_delta")["lipschitz"]
    b = local_sensitivity_rule(model, x, eps, "log_delta_recursive")["lipschitz"]
    assert torch.all((b <= a * (1 + 1e-5)) | ~torch.isfinite(a))
    xn = model.layers[0].norm(model.proj(x)).detach()
    a = local_block_sensitivity(model.layers[0], xn, eps, "log_delta").lip
    b = local_block_sensitivity(model.layers[0], xn, eps, "log_delta_recursive").lip
    assert torch.all(b <= a * (1 + 1e-5))


@pytest.mark.parametrize("rule", RULES)
def test_local_block_bound_is_sound_on_the_checkpoints(certified, rule):
    _, layers = certified
    block, _, bare = layers[0]
    torch.manual_seed(15)
    with torch.no_grad():
        xn = block.norm(torch.randn(2, 32, block.d_model) * 3.0)
        base = _internals(bare, xn.double())["out"]
        for eps in (1e-4, 1e-2):
            bound = local_block_sensitivity(block, xn, eps, rule)
            if not torch.isfinite(bound.lip).all():
                continue
            for _ in range(10):
                xi = torch.where(torch.rand_like(xn) > 0.5, eps, -eps).double()
                moved = (_internals(bare, xn.double() + xi)["out"] - base).abs()
                assert torch.all(moved <= bound.lip.double() * eps * (1 + 1e-4) + 1e-7)


def test_unknown_rule_and_variant_are_rejected(small_model):
    with pytest.raises(ValueError):
        local_sensitivity_rule(small_model, torch.randn(1, 4, 8), 1e-6, "geometric")
    inc = incremental_certificate(small_model.layers[0])
    with pytest.raises(ValueError):
        inc.output_slope("sharp")
