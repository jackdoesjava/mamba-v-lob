"""The released model state-spaces/mamba-130m-hf: reference forward, boxes, the incremental bound,
the local domain, and the attack and report of scripts/14_released.py.

The reference forward pass is held to transformers' slow path on real text, the certified boxes
to the realised activations at every layer, and the data-free incremental certificate to sampled
pairs of RMSNorm outputs link by link, since the end-to-end bound is far above anything sampling
reaches. The local domain, the data-free bound for points off the RMSNorm set, the scan's adjoint
and the attack are checked on a small layer built here, so they run without the checkpoint. The
report is rebuilt from docs/certification/released.json. Everything that needs the checkpoint is
skipped when its directory is absent.
"""

from __future__ import annotations

import importlib
import itertools
import json
import math

import pytest
import torch
import torch.nn as nn

from src.utils.repro import REPO_ROOT
from src.verification.incremental import (
    REPORTED_RULE,
    euler_cell_bound,
    euler_gain,
    local_step_recursive,
    timescale_offsets,
)
from src.verification.lipschitz import Sens
from src.verification.released import (
    DTYPE,
    LOCAL_RULES,
    ReleasedLayer,
    certify_layer,
    euler_additive_step,
    euler_log_delta_step,
    euler_warp,
    incremental_certificate,
    load_released,
    load_tokenizer,
    local_mixer_sensitivity,
    local_step_euler_endpoint,
    log_psi,
    log_softplus,
    mixer_trace,
    model_available,
    psi_prime,
    read_sample_text,
    realised_layer,
    reference_forward,
    released_layers,
    rms_norm,
    rmsnorm_linear_box,
    softplus_exact,
    tokenise,
)

script = importlib.import_module("scripts.14_released")

needs_model = pytest.mark.skipif(not model_available(), reason="mamba-130m-hf directory not found")


@pytest.fixture(scope="module")
def released():
    model = load_released()
    layers = released_layers(model)
    ids = tokenise(load_tokenizer(), read_sample_text())
    return model, layers, ids


@pytest.fixture(scope="module")
def certificates(released):
    _, layers, _ = released
    return [certify_layer(l) for l in layers]


# --- the discretisation, without the checkpoint ---------------------------------------------------


def test_euler_identity():
    lam = torch.logspace(-3, 6, 60, dtype=DTYPE).unsqueeze(1)
    dt = torch.logspace(-14, 3, 80, dtype=DTYPE).unsqueeze(0)
    x = lam * dt
    rebuilt = -torch.expm1(-x) * euler_gain(x) / lam
    assert torch.allclose(rebuilt, dt.expand_as(rebuilt), rtol=1e-12, atol=0.0)


def test_euler_scan_is_the_convex_form():
    g = torch.Generator().manual_seed(0)
    lam = 10 ** (torch.rand(64, generator=g, dtype=DTYPE) * 4 - 1)
    dt = 10 ** (torch.rand(64, 200, generator=g, dtype=DTYPE) * 8 - 6)
    bu = torch.randn(64, 200, generator=g, dtype=DTYPE)
    h_euler = torch.zeros(64, dtype=DTYPE)
    h_convex = torch.zeros(64, dtype=DTYPE)
    for t in range(200):
        a = torch.exp(-lam * dt[:, t])
        h_euler = a * h_euler + dt[:, t] * bu[:, t]
        c = euler_gain(lam * dt[:, t]) * bu[:, t] / lam
        h_convex = a * h_convex + (1 - a) * c
    assert torch.allclose(h_euler, h_convex, rtol=1e-10, atol=1e-300)


def test_slow_path_scan_is_exponential_euler():
    """One step from h_{-1} = 0 of transformers' torch-only scan gives dt B u, not the hold's
    (1 - exp(-lam dt)) B u / lam."""
    from transformers.models.mamba.modeling_mamba import mamba_selective_scan

    g = torch.Generator().manual_seed(1)
    d, n = 3, 2
    u = torch.randn(1, d, 1, generator=g, dtype=DTYPE)
    z = torch.randn(1, d, 1, generator=g, dtype=DTYPE)
    A = -torch.tensor([[0.5, 2.0], [1.0, 3.0], [4.0, 0.25]], dtype=DTYPE)
    B = torch.randn(1, n, 1, generator=g, dtype=DTYPE)
    C = torch.randn(1, n, 1, generator=g, dtype=DTYPE)
    y = mamba_selective_scan(u, z, A, B, C, delta_softplus=True)
    dt = softplus_exact(z)[0, :, 0].unsqueeze(1)
    Bu = B[0, :, 0].unsqueeze(0) * u[0, :, 0].unsqueeze(1)
    euler = (dt * Bu * C[0, :, 0]).sum(1)
    hold = (-torch.expm1(dt * A) * Bu / A.abs() * C[0, :, 0]).sum(1)
    assert torch.allclose(y[0, :, 0], euler, rtol=1e-6)
    assert not torch.allclose(y[0, :, 0], hold, rtol=1e-3)


def test_psi_prime_is_the_derivative_and_increases():
    x = torch.logspace(-3, 2, 400, dtype=DTYPE)
    h = 1e-5 * x.clamp(min=1.0)
    fd = (euler_gain(x + h) - euler_gain(x - h)) / (2 * h)
    assert torch.allclose(psi_prime(x), fd, rtol=1e-7)
    # the series below 1e-3 joins the closed form
    edge = torch.tensor([1e-3 * (1 - 1e-12), 1e-3], dtype=DTYPE)
    assert torch.allclose(psi_prime(edge)[0], psi_prime(edge)[1], rtol=1e-12)
    p = psi_prime(torch.logspace(-12, 3, 2000, dtype=DTYPE))
    assert torch.all(p[1:] >= p[:-1]) and p[0] > 0.5 - 1e-9 and p[-1] <= 1.0


def test_log_softplus_where_softplus_underflows():
    z = torch.tensor([-2000.0, -800.0, -40.0, -1.0, 30.0], dtype=DTYPE)
    assert torch.isfinite(log_softplus(z)).all()
    assert log_softplus(z)[0] == -2000.0
    assert torch.allclose(log_softplus(z[2:]), torch.log(softplus_exact(z[2:])), rtol=1e-12)


def test_rmsnorm_fused_bound_is_the_supremum():
    g = torch.Generator().manual_seed(2)
    d = 32
    w = torch.randn(d, generator=g, dtype=DTYPE)
    W = torch.randn(5, d, generator=g, dtype=DTYPE)
    box = rmsnorm_linear_box(w, W)
    x = torch.randn(4000, d, generator=g, dtype=DTYPE) * 10 ** torch.randn(4000, 1, generator=g, dtype=DTYPE)
    out = rms_norm(x, w, 1e-5) @ W.t()
    assert torch.all(out < box.hi) and torch.all(out > box.lo)
    # the input along W_k * w, scaled up, approaches the bound for row k
    direction = W * w
    near = (rms_norm(1e6 * direction, w, 1e-5) * W).sum(1)
    assert torch.allclose(near, box.hi, rtol=1e-9)


# --- the reference forward pass against transformers ----------------------------------------------


@needs_model
def test_reference_forward_equals_the_slow_path(released):
    model, layers, ids = released
    ids = ids[:, :96]
    with torch.no_grad():
        hf = model.backbone(input_ids=ids, output_hidden_states=True, use_cache=False)
    faithful, inputs = reference_forward(model, layers, ids, faithful=True)
    pure, _ = reference_forward(model, layers, ids)
    scale = hf.last_hidden_state.abs().max()
    # with transformers' float32 casts reproduced, the two agree to float64 round-off
    assert (faithful - hf.last_hidden_state).abs().max() <= 1e-11 * scale
    # in float64 throughout they differ by those casts only
    assert (pure - hf.last_hidden_state).abs().max() <= 1e-5 * scale
    for l in (0, 11, 23):
        blk = model.backbone.layers[l]
        # hidden_states[i] is the output of layer i, so the input of layer l is entry l - 1
        x = hf.hidden_states[l - 1] if l else inputs[0]
        assert torch.allclose(inputs[l], x, rtol=1e-6, atol=1e-6 * float(x.abs().max()))
        with torch.no_grad():
            v = blk.norm(x)
            ref = mixer_trace(layers[l], v, faithful=True).out
            out = blk.mixer(v)
        assert torch.equal(rms_norm(x, layers[l].norm_weight, layers[l].eps, faithful=True), v)
        assert (ref - out).abs().max() <= 1e-12 * out.abs().max()


@needs_model
def test_sample_text_fills_the_window(released):
    _, _, ids = released
    assert ids.shape == (1, 1024)


# --- the certificate on real text ------------------------------------------------------------------


@needs_model
def test_realised_activations_lie_in_the_certified_boxes(released, certificates):
    model, layers, ids = released
    _, inputs = reference_forward(model, layers, ids[:, :128])
    for layer, cert, x in zip(layers, certificates, inputs):
        r = realised_layer(layer, x, cert)
        assert all(r["contained"].values()), (layer.index, r["contained"])
        assert r["delta"]["certified_lo_min"] <= r["delta"]["realised_min"]
        assert r["delta"]["realised_max"] <= r["delta"]["certified_hi_max"]


@needs_model
def test_certificate_internals(certificates):
    for cert in certificates:
        # G = Psi(lam delta_hi) and M = G sup|Bu| / lam
        assert torch.allclose(cert.M, cert.G * cert.sup_Bu / cert.lam, rtol=1e-12)
        assert torch.all(cert.G >= 1.0)
        finite = torch.isfinite(cert.geometric)
        # the separated bound over the invariant is the closed-form gap, and never below 1
        assert torch.allclose(cert.geometric[finite] / cert.M[finite], cert.gap[finite], rtol=1e-9)
        assert torch.allclose(cert.log10_gap[finite], torch.log10(cert.gap[finite]), rtol=1e-9, atol=1e-9)
        assert torch.all(cert.log10_gap >= -1e-12)
        assert torch.isfinite(cert.log10_geometric).all()


# --- the incremental certificate against sampled pairs ---------------------------------------------


def _euler_equilibrium(layer, tr):
    lam = layer.lam
    dt = tr.delta.unsqueeze(-1)
    bu = tr.B.unsqueeze(2) * tr.u.unsqueeze(-1)
    return euler_gain(dt * lam) * bu / lam


@needs_model
@pytest.mark.parametrize("l", [0, 12, 23])
def test_incremental_certificate_is_never_below_sampled_pairs(released, certificates, l):
    model, layers, ids = released
    layer, cert = layers[l], certificates[l]
    inc = incremental_certificate(layer, cert)
    short = ids[:, :24]
    _, inputs = reference_forward(model, layers, short)
    x = inputs[l]
    g = torch.Generator().manual_seed(l)

    # perturbed residual streams, and one with different tokens; both normalised inputs are
    # RMSNorm outputs, which is the reading the certificate covers
    rms = x.pow(2).mean(-1, keepdim=True).sqrt()
    others = [x + s * rms * torch.randn(x.shape, generator=g, dtype=DTYPE) for s in (1e-6, 1e-3, 1e-1)]
    swapped = short.clone()
    swapped[:, ::3] = torch.randint(0, 50000, swapped[:, ::3].shape, generator=g)
    others.append(reference_forward(model, layers, swapped)[1][l])

    v = rms_norm(x, layer.norm_weight, layer.eps)
    tr = mixer_trace(layer, v, keep_states=True)
    c = _euler_equilibrium(layer, tr)
    slack = 1.0 + 1e-9
    for xt in others:
        vt = rms_norm(xt, layer.norm_weight, layer.eps)
        delta = float((v - vt).abs().max())
        tt = mixer_trace(layer, vt, keep_states=True)
        ct = _euler_equilibrium(layer, tt)

        assert torch.all((tr.u - tt.u).abs().amax((0, 1)) <= inc.L_u * delta * slack)
        assert torch.all((tr.B - tt.B).abs().amax((0, 1)) <= inc.L_B * delta * slack)
        assert torch.all((tr.C - tt.C).abs().amax((0, 1)) <= inc.L_C * delta * slack)
        assert torch.all((tr.z - tt.z).abs().amax((0, 1)) <= inc.L_z * delta * slack)
        dlog = (torch.log(tr.delta) - torch.log(tt.delta)).abs().amax((0, 1))
        assert torch.all(dlog <= inc.L_log_delta * delta * slack)
        assert torch.all((c - ct).abs().amax((0, 1)) <= inc.forcing_bound(delta) * slack)
        dh = (tr.h - tt.h).abs().amax((0, 1))
        for variant in ("cell", "common_start"):
            assert torch.all(dh <= inc.state_bound(delta, variant=variant) * slack)
            dy = (tr.y - tt.y).abs().amax((0, 1))
            assert torch.all(dy <= inc.y_bound(delta, variant=variant) * slack)
            dout = (tr.out - tt.out).abs().amax((0, 1))
            assert torch.all(dout <= inc.output_bound(delta, variant=variant) * slack)


@needs_model
def test_incremental_certificate_matches_the_euler_corollary(released, certificates):
    _, layers, _ = released
    inc = incremental_certificate(layers[3], certificates[3])
    for delta in (1e-9, 1e-6, 1e-3):
        rho = inc.L_log_delta.unsqueeze(1) * delta
        eta_B = (inc.sup_B.unsqueeze(0) * inc.L_u.unsqueeze(1)
                 + inc.sup_u.unsqueeze(1) * inc.L_B.unsqueeze(0)) * delta / inc.lam
        x_hi = inc.lam * inc.ceiling.unsqueeze(1)
        expected = euler_cell_bound(eta_B, inc.M0, rho, x_hi)
        assert torch.allclose(inc.state_bound(delta), expected, rtol=1e-12)
        assert torch.all(inc.state_bound(delta, variant="common_start") <= inc.state_bound(delta))
    # the finite-delta bound tends to the slope; the slope is the limit from above
    small = inc.output_bound(1e-12) / 1e-12
    assert torch.allclose(small, inc.output_slope(), rtol=1e-6)
    assert torch.all(inc.output_bound(1e-3) / 1e-3 >= inc.output_slope() * (1 - 1e-12))
    # the additive baseline without a floor is affine in the length
    a1, a2 = inc.additive_output_slope(10, floor=False), inc.additive_output_slope(20, floor=False)
    a0 = inc.additive_output_slope(0, floor=False)
    assert torch.allclose(a2 - a1, a1 - a0, rtol=1e-9)
    # the reading with any point within delta widens the boxes, so it is never below the other
    for delta in (1e-9, 1e-6, 1e-3):
        assert torch.all(inc.state_bound(delta, arbitrary=True) >= inc.state_bound(delta))
        assert torch.all(inc.output_bound(delta, arbitrary=True) >= inc.output_bound(delta))
    with pytest.raises(NotImplementedError):
        inc.warp_ratio()
    assert math.isinf(float(inc.additive_state_slope(None, floor=False).max()))


# --- a small layer of the released architecture, without the checkpoint ---------------------------


def _toy_layer(seed: int = 0, d_model: int = 8, d_inner: int = 6, d_state: int = 3,
               dt_rank: int = 2, d_conv: int = 4) -> ReleasedLayer:
    """Random weights in the released layout, with |A| spread over three decades so that both
    slow cells (lam dt << 1) and fast ones (lam dt >> 1) occur."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g, dtype=DTYPE)

    conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1).to(DTYPE)
    with torch.no_grad():
        conv.weight.copy_(0.5 * r(d_inner, 1, d_conv))
        conv.bias.copy_(0.1 * r(d_inner))
    conv.requires_grad_(False)
    lam = 10 ** (torch.rand(d_inner, d_state, generator=g, dtype=DTYPE) * 3 - 1.5)
    return ReleasedLayer(
        index=0, eps=1e-5, norm_weight=1.0 + 0.1 * r(d_model), W_in=0.4 * r(2 * d_inner, d_model),
        conv=conv, W_x=0.4 * r(dt_rank + 2 * d_state, d_inner), W_dt=0.5 * r(d_inner, dt_rank),
        b_dt=r(d_inner) - 1.0, A_log=torch.log(lam), D=r(d_inner), W_out=0.4 * r(d_model, d_inner),
        dt_rank=dt_rank, d_state=d_state,
    )


def _normalised(layer: ReleasedLayer, batch: int, L: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = 3.0 * torch.randn(batch, L, layer.d_model, generator=g, dtype=DTYPE)
    return rms_norm(x, layer.norm_weight, layer.eps)


def _budget(candidates: int = 16, last: int = 4) -> "script.Budget":
    return script.Budget(probes=8, candidates=candidates, last_candidates=last, steps=6,
                         restarts=4, batch=16)


def test_euler_scan_adjoint_passes_gradcheck_across_chunk_boundaries(monkeypatch):
    monkeypatch.setattr(script, "CHUNK", 3)
    g = torch.Generator().manual_seed(3)
    b, L, d, n = 2, 7, 3, 4
    delta = (torch.rand(b, L, d, generator=g, dtype=DTYPE) * 0.5 + 0.01).requires_grad_()
    B = torch.randn(b, L, n, generator=g, dtype=DTYPE).requires_grad_()
    C = torch.randn(b, L, n, generator=g, dtype=DTYPE).requires_grad_()
    u = torch.randn(b, L, d, generator=g, dtype=DTYPE).requires_grad_()
    A = -torch.rand(d, n, generator=g, dtype=DTYPE) * 3.0 - 0.1
    Dk = torch.randn(d, generator=g, dtype=DTYPE)
    assert torch.autograd.gradcheck(
        lambda dt, bb, cc, uu: script._Scan.apply(dt, bb, cc, uu, A, Dk), (delta, B, C, u))


def test_fast_mixer_matches_the_reference_and_its_gradient(monkeypatch):
    monkeypatch.setattr(script, "CHUNK", 4)
    layer = _toy_layer()
    v = _normalised(layer, 3, 11, seed=4)
    ref = mixer_trace(layer, v).out
    with torch.no_grad():
        fast = script.mixer_contribution(layer, v)
    torch.testing.assert_close(fast, ref, rtol=1e-12, atol=1e-13)
    graph = script.mixer_contribution(layer, v.clone().requires_grad_(True))
    torch.testing.assert_close(graph.detach(), ref, rtol=1e-12, atol=1e-13)
    assert torch.autograd.gradcheck(lambda w: script.mixer_contribution(layer, w),
                                    (v[:1, :5].clone().requires_grad_(True),))


@pytest.mark.parametrize("eps", [1e-4, 1e-2, 0.3])
def test_local_bound_holds_at_sampled_and_attacked_points(eps):
    layer = _toy_layer(seed=1)
    v = _normalised(layer, 1, 9, seed=5)
    base = mixer_trace(layer, v).out
    g = torch.Generator().manual_seed(6)
    corners = eps * (torch.randint(0, 2, (64, *v.shape[1:]), generator=g) * 2 - 1).to(DTYPE)
    inside = eps * (2 * torch.rand(64, *v.shape[1:], generator=g, dtype=DTYPE) - 1)
    att = script.attack_layer(layer, v, [eps], _budget(), seed=7)
    change = torch.cat([(mixer_trace(layer, v + corners).out - base).abs(),
                        (mixer_trace(layer, v + inside).out - base).abs(),
                        att["per_delta"][0]["map"].unsqueeze(0)])
    for rule in LOCAL_RULES:
        lip = local_mixer_sensitivity(layer, v, eps, rule)["lipschitz"]
        ok = torch.isfinite(lip)
        assert torch.all((change <= lip * eps * (1 + 1e-9) + 1e-14) | ~ok), rule


def test_local_domain_runs_the_nominal_pass_and_the_recursion_is_below_the_maximum():
    layer = _toy_layer(seed=2)
    v = _normalised(layer, 2, 12, seed=8)
    ref = mixer_trace(layer, v).out
    for eps in (1e-6, 1e-3, 1e-1):
        out = {rule: local_mixer_sensitivity(layer, v, eps, rule) for rule in LOCAL_RULES}
        for r in out.values():
            torch.testing.assert_close(r["nominal"], ref, rtol=1e-12, atol=1e-13)
        # E_t of the recursion is never above the running maximum, the endpoint form is never
        # above the recursion or the additive step, and the readout is monotone
        assert torch.all(out["log_delta_recursive"]["lipschitz"]
                         <= out["log_delta"]["lipschitz"] * (1 + 1e-12))
        end = out["log_delta_endpoint"]["lipschitz"]
        assert torch.isfinite(end).all()
        for rule in ("additive", "log_delta_recursive"):
            other = out[rule]["lipschitz"]
            assert torch.all((end <= other * (1 + 1e-10)) | ~torch.isfinite(other)), (eps, rule)
    with pytest.raises(ValueError):
        local_mixer_sensitivity(layer, v, 1e-3, "exact")
    assert local_mixer_sensitivity(layer, v, 1e-3)["rule"] == REPORTED_RULE == "log_delta_endpoint"
    assert LOCAL_RULES == ("additive", "log_delta", "log_delta_recursive", "log_delta_endpoint")


# --- the endpoint form for the exponential-Euler cell ---------------------------------------------


def test_log_psi_and_euler_warp_at_high_precision():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 60
    for y in (-700.0, -50.0, -1.0, -0.0101, -0.01, -1e-3, -1e-9, 0.0, 1e-9, 1e-3, 0.0099, 0.0101,
              1.0, 50.0, 800.0):
        Y = mp.mpf(y)
        exact = 0.0 if y == 0 else float(mp.log(Y / (1 - mp.exp(-Y))))
        got = float(log_psi(torch.tensor([y], dtype=DTYPE))[0])
        assert got == pytest.approx(exact, rel=1e-12, abs=1e-300), y
    for x in (1e-14, 1e-9, 1e-5, 1e-3, 0.01, 0.7, 5.0, 40.0):
        for frac in (-1.0, -0.999, -0.5, -1e-6, 1e-9, 1e-4, 0.3, 1.0, 10.0, 1e3):
            t = x * frac
            X, Tt = mp.mpf(x), mp.mpf(t)
            exact = float((mp.exp(-X) - mp.exp(-(X + Tt))) * X / (1 - mp.exp(-X)) - Tt)
            got = float(euler_warp(torch.tensor([x], dtype=DTYPE), torch.tensor([t], dtype=DTYPE))[0])
            assert got == pytest.approx(exact, rel=1e-9, abs=1e-300), (x, frac)
    # 0 at both roots, the sign pattern of the docstring, and -t at infinity
    x = torch.tensor([0.3], dtype=DTYPE)
    assert float(euler_warp(x, torch.tensor([0.0], dtype=DTYPE))) == 0.0
    assert float(euler_warp(x, -x)) == 0.0
    assert float(euler_warp(x, -x / 2)) > 0 and float(euler_warp(x, x)) < 0
    assert float(euler_warp(x, torch.tensor([math.inf], dtype=DTYPE))) == -math.inf


def _euler_cells(n, seed, x_range=(-4.0, 1.0)):
    g = torch.Generator().manual_seed(seed)

    def u(lo, hi):
        return lo + (hi - lo) * torch.rand(n, generator=g, dtype=DTYPE)

    x = 10 ** u(*x_range)
    rho = 10 ** u(-3.0, 1.0)
    dx = torch.where(torch.rand(n, generator=g) > 0.5, x * u(0.0, 3.0), torch.full_like(x, math.inf))
    E = u(0.0, 1.0) * (torch.rand(n, generator=g) > 0.3).to(DTYPE)
    b = torch.randn(n, generator=g, dtype=DTYPE)
    db = u(0.0, 0.5) * (torch.rand(n, generator=g) > 0.3).to(DTYPE)
    # states near the equilibrium Psi(x) b, where the bound is not convex in Abar~, and far
    near = torch.rand(n, generator=g) > 0.5
    h = torch.where(near, euler_gain(x) * b * (1 + 0.1 * torch.randn(n, generator=g, dtype=DTYPE)),
                    3 * torch.randn(n, generator=g, dtype=DTYPE))
    return x, rho, dx, E, h, b, db


def _one_step_errors(x, t, E, h, b, db):
    """|e_t| of explicit pairs: lam dt~ = x + t over the grid t, e_{t-1} = +-E, b - b~ = +-db."""
    worst = None
    for se, sb in itertools.product((-1.0, 1.0), repeat=2):
        ht, bt = h - se * E, b - sb * db
        e = (torch.exp(-x) * h + x * b) - (torch.exp(-(x + t)) * ht + (x + t) * bt)
        worst = e.abs() if worst is None else torch.maximum(worst, e.abs())
    return worst


def test_euler_endpoint_step_is_the_largest_one_step_error():
    """Over a dense grid of perturbed timescales in I_t and every sign of e_{t-1} and b - b~, the
    one-step error never exceeds the step, and the grid's best pair comes within its resolution
    of it, so the step is the exact one-step worst case."""
    n = 2000
    x, rho, dx, E, h, b, db = _euler_cells(n, seed=20)
    step = local_step_euler_endpoint(E, x, rho, h, b, db, dx)
    up, down = timescale_offsets(x, rho, dx)
    capped = up <= 50.0
    frac = torch.linspace(0.0, 1.0, 4001, dtype=DTYPE).unsqueeze(1)
    t = -down + frac * (up.clamp(max=50.0) + down)
    worst = _one_step_errors(x, t, E, h, b, db).amax(0)
    tol = 1e-12 * (h.abs() + b.abs() + E + db) * (1 + x + up.clamp(max=50.0))
    assert torch.all(worst <= step + tol)
    assert torch.all((worst >= step * (1 - 1e-5) - tol)[capped])


def test_euler_endpoint_step_takes_an_interior_maximum():
    """With a state well beyond its equilibrium and a wide log budget, the one-step error is
    concave in Abar~ on the fast side of Abar and peaks inside I_t, above both ends. The step
    finds the peak, at Abar~ = (|b| - db) / (|h| - E), where G' = 0."""
    x = torch.tensor([0.02, 0.02, 0.02], dtype=DTYPE)
    b = torch.tensor([-0.5, 0.5, 0.5], dtype=DTYPE)
    h = torch.tensor([-4.0, 4.0, 4.0], dtype=DTYPE)
    E = torch.tensor([0.0, 0.0, 0.1], dtype=DTYPE)
    db = torch.tensor([0.0, 0.0, 0.05], dtype=DTYPE)
    rho = torch.full_like(x, 5.4)
    step = local_step_euler_endpoint(E, x, rho, h, b, db)
    up, down = timescale_offsets(x, rho)
    t = -down + torch.linspace(0.0, 1.0, 200001, dtype=DTYPE).unsqueeze(1) * (up + down)
    err = _one_step_errors(x, t, E, h, b, db)
    best = err.amax(0)
    ends = torch.maximum(err[0], err[-1])
    assert torch.all(best > ends * 1.1)
    torch.testing.assert_close(step, best, rtol=1e-9, atol=0.0)
    s_star = torch.exp(-(x + t[err.argmax(0), torch.arange(3)]))
    torch.testing.assert_close(s_star, (b.abs() - db) / (h.abs() - E), rtol=1e-4, atol=0.0)


def test_euler_endpoint_step_on_random_sequences():
    """Two exponential-Euler runs with per-step log budgets and, in half of the cells, additive
    budgets: the endpoint form holds along the run and is never above the recursion of
    euler_log_delta_step on the same budgets."""
    n, T = 2000, 300
    g = torch.Generator().manual_seed(21)
    lam = 10 ** (torch.rand(n, generator=g, dtype=DTYPE) * 2.7 - 1.0)
    dt = 10 ** (torch.rand(n, T, generator=g, dtype=DTYPE) * 9.0 - 8.0)
    rho_max = torch.tensor([1e-6, 1e-2, 0.5, 2.0], dtype=DTYPE).repeat(n // 4)
    rho_t = torch.rand(n, T, generator=g, dtype=DTYPE) * rho_max.unsqueeze(1)
    s = rho_t * (torch.rand(n, T, generator=g, dtype=DTYPE) * 2 - 1)
    s = torch.where(torch.rand(n, T, generator=g) > 0.7, rho_t * torch.sign(s), s)
    dt_t = dt * torch.exp(s)
    beta = (dt - dt_t).abs() * (1 + 2 * torch.rand(n, T, generator=g, dtype=DTYPE))
    beta = torch.where((torch.arange(n) % 2 == 0).unsqueeze(1), beta, torch.full_like(beta, math.inf))
    db = 10 ** (torch.rand(n, generator=g, dtype=DTYPE) * 4 - 4)
    b_t = torch.randn(n, T, generator=g, dtype=DTYPE) * 3
    b = b_t + (torch.rand(n, T, generator=g, dtype=DTYPE) * 2 - 1) * db.unsqueeze(1)

    h = torch.randn(n, generator=g, dtype=DTYPE)
    ht = h + 0.01 * torch.randn(n, generator=g, dtype=DTYPE)
    E_end = E_rec = (h - ht).abs()
    for t in range(T):
        x, xt = lam * dt[:, t], lam * dt_t[:, t]
        dbt = (b[:, t] - b_t[:, t]).abs()
        E_end = local_step_euler_endpoint(E_end, x, rho_t[:, t], h, b[:, t], dbt, lam * beta[:, t])
        psi, kappa = euler_gain(x), torch.expm1(rho_t[:, t])
        dc = psi * dbt + kappa * psi * (b[:, t].abs() + dbt)
        E_rec = local_step_recursive(E_rec, x, rho_t[:, t], dc, (h - psi * b[:, t]).abs())
        h = torch.exp(-x) * h + x * b[:, t]
        ht = torch.exp(-xt) * ht + xt * b_t[:, t]
        assert torch.all((h - ht).abs() <= E_end * (1 + 1e-10) + 1e-14 * (1 + h.abs()))
        assert torch.all(E_end <= E_rec * (1 + 1e-10))


def test_euler_endpoint_step_equality_cases():
    """With no timescale perturbation the step is Abar E + lam dt db, and along a run with b - b~
    of one sign that is the error itself."""
    g = torch.Generator().manual_seed(22)
    n, T = 500, 100
    lam = 10 ** (torch.rand(n, generator=g, dtype=DTYPE) * 2 - 1)
    dt = 10 ** (torch.rand(n, T, generator=g, dtype=DTYPE) * 6 - 5)
    db = 10 ** (torch.rand(n, generator=g, dtype=DTYPE) * 3 - 3)
    b = torch.randn(n, T, generator=g, dtype=DTYPE)
    zero = torch.zeros(n, dtype=DTYPE)
    h, ht, E = zero.clone(), zero.clone(), zero.clone()
    for t in range(T):
        x = lam * dt[:, t]
        for dx in (None, zero):
            step = local_step_euler_endpoint(E, x, zero, h, b[:, t], db, dx)
            torch.testing.assert_close(step, torch.exp(-x) * E + x * db, rtol=1e-12, atol=0.0)
        # an additive budget of 0 pins the timescale whatever rho is
        pinned = local_step_euler_endpoint(E, x, torch.full_like(x, 5.0), h, b[:, t], db, zero)
        torch.testing.assert_close(pinned, step, rtol=1e-12, atol=0.0)
        E = step
        h = torch.exp(-x) * h + x * b[:, t]
        ht = torch.exp(-x) * ht + x * (b[:, t] - db)
        torch.testing.assert_close((h - ht).abs(), E, rtol=1e-9, atol=1e-15)


def test_euler_endpoint_step_without_overflow():
    """With the additive budget the step is finite for every rho; with rho alone and db > 0 it is
    not, because x~ b~ grows without bound with dt~, and with b = b~ = 0 it is again finite."""
    x = torch.tensor([0.1, 0.1, 0.1, 1e-12], dtype=DTYPE)
    rho = torch.tensor([1e4, 1e4, 1e4, 800.0], dtype=DTYPE)
    E = torch.tensor([1.0, 1.0, 1.0, 0.5], dtype=DTYPE)
    h = torch.tensor([0.3, 0.3, 0.3, -2.0], dtype=DTYPE)
    b = torch.tensor([1.0, 1.0, 0.0, 1.0], dtype=DTYPE)
    db = torch.tensor([1e-3, 1e-3, 0.0, 1e-3], dtype=DTYPE)
    dx = torch.tensor([0.05, math.inf, math.inf, 1e-6], dtype=DTYPE)
    out = local_step_euler_endpoint(E, x, rho, h, b, db, dx)
    assert not torch.isnan(out).any()
    assert math.isfinite(out[0]) and math.isinf(out[1]) and math.isfinite(out[2])
    assert math.isfinite(out[3])
    # b = 0: the perturbed state can decay to 0 at once or stay, so the step is max(|a h|+..., E)
    a = math.exp(-0.1)
    assert float(out[2]) == pytest.approx(max(1.0 + (1 - a) * 0.3, a * 0.3), rel=1e-9)
    rec = local_step_recursive(E, x, rho, db, (h - euler_gain(x) * b).abs())
    assert math.isinf(rec[0])
    inf_in = local_step_euler_endpoint(torch.full_like(E, math.inf), x, rho, h, b, db, dx)
    assert torch.isinf(inf_in).all()


def test_euler_endpoint_step_for_timescales_far_above_one_over_lam():
    """lam dt between 1e2 and 1e5, as in the released model, where the slow end of I_t lies more
    than 709 below x and exp(-x)(1 - exp(-t)) would be 0 * inf. The step stays a number, is
    finite wherever the additive budget is, covers the one-step error over a grid of I_t, and is
    never above the recursion."""
    n = 2000
    x, rho, dx, E, h, b, db = _euler_cells(n, seed=24, x_range=(2.0, 5.0))
    step = local_step_euler_endpoint(E, x, rho, h, b, db, dx)
    up, down = timescale_offsets(x, rho, dx)
    assert bool((down > 709.0).any())
    assert not torch.isnan(step).any()
    assert torch.isfinite(step[torch.isfinite(dx)]).all()
    frac = torch.linspace(0.0, 1.0, 4001, dtype=DTYPE).unsqueeze(1)
    t = -down + frac * (up.clamp(max=50.0) + down)
    worst = _one_step_errors(x, t, E, h, b, db).amax(0)
    tol = 1e-12 * (h.abs() + b.abs() + E + db) * (1 + x + up.clamp(max=50.0))
    assert torch.all(worst <= step + tol)
    psi, kappa = euler_gain(x), torch.expm1(rho)
    rec = local_step_recursive(E, x, rho, psi * db + kappa * psi * (b.abs() + db),
                               (h - psi * b).abs())
    finite = torch.isfinite(rec)
    assert torch.all(step[finite] <= rec[finite] * (1 + 1e-10))


def _euler_sens_cells(n, seed):
    g = torch.Generator().manual_seed(seed)
    d, s = 6, 4

    def r(*shape):
        return torch.randn(*shape, generator=g, dtype=DTYPE)

    A = -(10 ** (torch.rand(d, s, generator=g, dtype=DTYPE) * 3 - 1.5))
    h = Sens(r(n, d, s), torch.rand(n, d, s, generator=g, dtype=DTYPE) * 10)
    dt = Sens(10 ** (torch.rand(n, d, 1, generator=g, dtype=DTYPE) * 6 - 5),
              10 ** (torch.rand(n, d, 1, generator=g, dtype=DTYPE) * 6 - 5))
    u = Sens(r(n, d, 1), torch.rand(n, d, 1, generator=g, dtype=DTYPE) * 10)
    B = Sens(r(n, 1, s), torch.rand(n, 1, s, generator=g, dtype=DTYPE) * 10)
    log_lip = torch.where(torch.rand(n, d, 1, generator=g) > 0.2,
                          torch.rand(n, d, 1, generator=g, dtype=DTYPE) * 1e4,
                          torch.full((n, d, 1), math.inf, dtype=DTYPE))
    return h, dt, u, B, A, log_lip


@pytest.mark.parametrize("eps", [1e-7, 1e-4, 1e-2])
def test_euler_endpoint_step_is_never_above_the_additive_or_the_recursive_step(eps):
    h, dt, u, B, A, log_lip = _euler_sens_cells(400, seed=23)
    end = euler_log_delta_step(h, dt, log_lip, u, B, A, eps, endpoint=True)
    add_ = euler_additive_step(h, dt, u, B, A, eps)
    rec = euler_log_delta_step(h, dt, log_lip, u, B, A, eps, recursive=True).lip
    torch.testing.assert_close(end.val, add_.val, rtol=1e-14, atol=1e-300)
    assert torch.isfinite(end.lip).all()
    assert torch.all(end.lip <= add_.lip * (1 + 1e-10))
    finite = torch.isfinite(rec)
    assert torch.all(end.lip[finite] <= rec[finite] * (1 + 1e-10))
    assert bool((end.lip < add_.lip * (1 - 1e-6)).any())
    assert bool((end.lip[finite] < rec[finite] * (1 - 1e-6)).any())


@pytest.mark.parametrize("eps", [1e-3, 1e-1])
def test_data_free_bound_covers_points_off_the_rmsnorm_set(eps):
    layer = _toy_layer(seed=3)
    inc = incremental_certificate(layer)
    v = _normalised(layer, 1, 10, seed=9)
    base = mixer_trace(layer, v).out
    g = torch.Generator().manual_seed(10)
    xi = eps * (torch.randint(0, 2, (128, *v.shape[1:]), generator=g) * 2 - 1).to(DTYPE)
    change = (mixer_trace(layer, v + xi).out - base).abs().amax((0, 1))
    attacked = script.attack_layer(layer, v, [eps], _budget(), seed=11)["per_delta"][0]["map"].amax(0)
    bound = inc.output_bound(eps, arbitrary=True)
    assert torch.all(change <= bound) and torch.all(attacked <= bound)
    assert torch.all(bound >= inc.output_bound(eps))
    for variant in ("cell", "common_start"):
        assert torch.all(inc.state_bound(eps, arbitrary=True, variant=variant)
                         >= inc.state_bound(eps, variant=variant))


def test_attack_reaches_the_exact_first_order_optimum():
    """With every (position, coordinate) target ranked, the attack at a small delta finds the
    largest l1 norm of a row of the exact Jacobian."""
    layer = _toy_layer(seed=4)
    v = _normalised(layer, 1, 6, seed=12)
    J = torch.autograd.functional.jacobian(lambda w: script.mixer_contribution(layer, w), v)
    rows = J.reshape(6 * layer.d_model, -1).abs().sum(1)
    att = script.attack_layer(layer, v, [1e-7], _budget(6 * layer.d_model, layer.d_model), seed=13)
    assert math.isclose(att["first_order"], float(rows.max()), rel_tol=1e-9)
    assert math.isclose(att["per_delta"][0]["max"] / 1e-7, float(rows.max()), rel_tol=1e-5)


# --- the report of scripts/14_released.py ----------------------------------------------------------


def _recorded() -> dict:
    path = REPO_ROOT / "docs/certification/released.json"
    if not path.exists():
        pytest.skip("released.json not found")
    results = json.loads(path.read_text(encoding="utf-8"))
    if any(REPORTED_RULE not in pd["local"] for e in results["experiments"] for pd in e["per_delta"]):
        pytest.skip("released.json predates the endpoint rule; rerun scripts/14_released.py")
    return results


def test_report_renders_from_the_recorded_results(tmp_path):
    """The markdown twin and the figure are rebuilt from docs/certification/released.json, so a
    change to the report code that breaks on the real results fails here, and the committed twin
    must be exactly what the JSON gives."""
    results = _recorded()
    md = script.markdown(results)
    for heading in ("## The data-free certificate", "## The data-free incremental certificate",
                    "## The local certificate and the attack", "## What was run", "## Scope"):
        assert heading in md
    assert "nan |" not in md and "| nan" not in md
    assert md == (REPO_ROOT / "docs/certification/released.md").read_text(encoding="utf-8")
    script.figure(results, tmp_path / "figure.pdf")
    assert (tmp_path / "figure.pdf").stat().st_size > 0


def test_report_states_the_recorded_numbers():
    results = _recorded()
    md = script.markdown(results)
    s = results["data_free"]["summary"]
    assert script.fmt(s["out_slope_cell_max"]["max"]) in md
    assert script.fmt(results["agreement"]["pure_max_abs_diff"]) in md
    df_v, local_v = script._violations(results)
    assert ("No attacked value exceeds its certificate" in md) == (df_v + local_v == 0)
    # every attacked (layer, length, delta) has its row, and every listed attack was run
    attacked = [e for e in results["experiments"] if e["attacked"]]
    rows = [line for line in md.splitlines() if line.startswith("| ")]
    for e in attacked:
        for pd in e["per_delta"]:
            prefix = (f"| {e['layer']} | {e['seq_len']} | {script.fmt(pd['delta'])} | "
                      f"{script.fmt(pd['slope_max'])} |")
            assert any(r.startswith(prefix) for r in rows), prefix
    for L, ls in results["run"]["attack_layers_by_length"].items():
        assert f"at L = {L}" in md
        for l in ls:
            assert any(e["layer"] == l and str(e["seq_len"]) == L for e in attacked)
    assert all(b["restarts"] >= 1 for b in results["budget"].values())


def test_recorded_endpoint_rule_is_never_above_the_others():
    """The endpoint form is at most the recursion and the additive step at every step, and every
    later stage of the domain is monotone in the sensitivity it receives, so in the recorded run
    it is finite wherever either is and never above either. A failure here is a numerical defect
    in the endpoint step, such as an overflow to inf or nan."""
    results = _recorded()
    for e in results["experiments"]:
        for pd in e["per_delta"]:
            loc = pd["local"]
            end = loc[REPORTED_RULE]["lip_max"]
            for other in ("additive", "log_delta_recursive"):
                o = loc[other]["lip_max"]
                if math.isfinite(o):
                    assert math.isfinite(end), (e["layer"], e["seq_len"], pd["delta"], other)
                    assert end <= o * (1 + 1e-9), (e["layer"], e["seq_len"], pd["delta"], other)
