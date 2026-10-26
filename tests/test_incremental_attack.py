"""The attack of scripts/13_incremental_attack.py and the stage trace of the local domain.

The attack is only as good as its gradient and only as honest as its scoring, so the scan's
hand-written adjoint is checked with gradcheck, the fast pass against the reference passes, and
the attack on a block small enough that every (position, coordinate) target can be enumerated
against the exact Jacobian. Soundness is checked against both bounds the script prices the
attack with. The stage trace must reproduce local_sensitivity_rule bitwise, and its stage logs
must add up to the bound.
"""

from __future__ import annotations

import copy
import importlib
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.dataset import LOBWindows
from src.models.mamba import LOBMamba, SelectiveSSMBlock
from src.verification.incremental import (
    contribution,
    incremental_certificate,
    local_block_sensitivity,
    local_sensitivity_rule,
)

attack = importlib.import_module("scripts.13_incremental_attack")

D64 = torch.float64
TWO_LAYERS = {"model": {"mamba": {"d_model": 16, "d_state": 4, "expand": 2, "num_layers": 2}}}


def _block(arm: str, seed: int = 0) -> tuple[SelectiveSSMBlock, SelectiveSSMBlock]:
    torch.manual_seed(seed)
    block = SelectiveSSMBlock(d_model=8, d_state=4, dt_parametrisation=arm).eval()
    return block, attack.float64_copy(block)


def _model(arm: str = "bounded", seed: int = 1) -> tuple[LOBMamba, LOBMamba]:
    torch.manual_seed(seed)
    cfg = copy.deepcopy(TWO_LAYERS)
    cfg["model"]["mamba"]["dt_parametrisation"] = arm
    model = LOBMamba(input_dim=5, config=cfg).eval()
    return model, attack.float64_copy(model)


# --- the fast pass --------------------------------------------------------------------------


def test_scan_adjoint_passes_gradcheck_across_chunk_boundaries(monkeypatch):
    monkeypatch.setattr(attack, "CHUNK", 3)
    torch.manual_seed(2)
    b, L, d, n = 2, 7, 3, 4
    delta = (torch.rand(b, L, d, dtype=D64) * 0.5 + 0.01).requires_grad_()
    B = torch.randn(b, L, n, dtype=D64, requires_grad=True)
    C = torch.randn(b, L, n, dtype=D64, requires_grad=True)
    u = torch.randn(b, L, d, dtype=D64, requires_grad=True)
    A = -torch.rand(d, n, dtype=D64) * 3.0 - 0.1
    Dskip = torch.randn(d, dtype=D64)
    assert torch.autograd.gradcheck(
        lambda dt, bb, cc, uu: attack._Scan.apply(dt, bb, cc, uu, A, Dskip), (delta, B, C, u))


@pytest.mark.parametrize("arm", ["bounded", "softplus"])
def test_fast_contribution_matches_the_reference_pass(arm, monkeypatch):
    monkeypatch.setattr(attack, "CHUNK", 5)
    _, b64 = _block(arm)
    xn = b64.norm(torch.randn(3, 17, 8, dtype=D64) * 2.0)
    ref = contribution(b64, xn)
    with torch.no_grad():
        fast = attack.ssm_contribution(b64, xn)
    torch.testing.assert_close(fast, ref, rtol=1e-12, atol=1e-13)
    graph = attack.ssm_contribution(b64, xn.clone().requires_grad_(True))
    torch.testing.assert_close(graph.detach(), ref, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("arm", ["bounded", "softplus"])
def test_fast_gradient_matches_autograd_through_the_block(arm, monkeypatch):
    monkeypatch.setattr(attack, "CHUNK", 4)
    _, b64 = _block(arm, seed=3)
    bare = copy.deepcopy(b64)
    bare.norm = nn.Identity()
    xn = b64.norm(torch.randn(2, 11, 8, dtype=D64) * 2.0)
    v = torch.randn(2, 11, 8, dtype=D64)
    a = xn.clone().requires_grad_(True)
    (g_fast,) = torch.autograd.grad((attack.ssm_contribution(b64, a) * v).sum(), a)
    b = xn.clone().requires_grad_(True)
    (g_ref,) = torch.autograd.grad(((bare(b) - b) * v).sum(), b)
    torch.testing.assert_close(g_fast, g_ref, rtol=1e-9, atol=1e-12)


def test_prediction_passes_agree_with_the_model():
    model, m64 = _model()
    x = torch.randn(3, 13, 5)
    fast = attack.predict(m64, x.to(D64))
    ref = attack.predict_reference(m64, x.to(D64))
    torch.testing.assert_close(fast, ref, rtol=1e-12, atol=1e-12)
    with torch.no_grad():
        torch.testing.assert_close(ref.float(), model(x), rtol=1e-4, atol=1e-5)


# --- the block attack -----------------------------------------------------------------------


def _small_budget() -> "attack.Budget":
    # 64 candidates exceed the 6 x 8 = 48 targets, so every gradient row is taken
    return attack.Budget(probes=8, candidates=64, last_candidates=8, block_steps=4,
                         model_steps=4, batch=16)


@pytest.mark.parametrize("arm", ["bounded", "softplus"])
def test_block_attack_reaches_the_exact_first_order_optimum(arm):
    _, b64 = _block(arm, seed=4)
    xn = b64.norm(torch.randn(1, 6, 8, dtype=D64) * 2.0)
    res = attack.attack_block(b64, xn, [1e-6, 1e-2], _small_budget(), seed=0)

    jac = torch.autograd.functional.jacobian(
        lambda z: attack.ssm_contribution(b64, z.unsqueeze(0))[0], xn[0])   # (L, out, L, in)
    rows = jac.abs().sum(dim=(2, 3))
    assert res["first_order"] == pytest.approx(float(rows.max()), rel=1e-9)
    assert res["first_order_last"] == pytest.approx(float(rows[-1].max()), rel=1e-9)
    # at delta = 1e-6 the block is linear to about 1e-6, so the attack attains delta |row|_1
    smallest = res["per_delta"][0]
    assert smallest["max"] / 1e-6 >= float(rows.max()) * (1 - 1e-4)
    assert smallest["max"] / 1e-6 >= res["random_corner"] * (1 - 1e-4)


@pytest.mark.parametrize("arm", ["bounded", "softplus"])
def test_block_attack_respects_the_data_free_and_local_bounds(arm):
    block, b64 = _block(arm, seed=5)
    inc = incremental_certificate(block)
    xn = b64.norm(torch.randn(1, 6, 8, dtype=D64) * 2.0)
    deltas = [1e-6, 1e-3, 1e-1]
    res = attack.attack_block(b64, xn, deltas, _small_budget(), seed=1)
    for d, r in zip(deltas, res["per_delta"]):
        per_coordinate = r["map"].amax(0)
        assert torch.all(per_coordinate <= inc.output_bound(d, arbitrary=True) * (1 + 1e-9))
        for rule in attack.RULES:
            lip = local_block_sensitivity(b64, xn, d, rule).lip[0]
            ok = torch.isfinite(lip)
            assert torch.all((r["map"] <= lip * d * (1 + 1e-9) + 1e-14) | ~ok), (d, rule)


def test_block_attack_scores_with_the_reference_pass():
    _, b64 = _block("bounded", seed=6)
    xn = b64.norm(torch.randn(1, 6, 8, dtype=D64) * 2.0)
    res = attack.attack_block(b64, xn, [1e-2], _small_budget(), seed=2)
    r = res["per_delta"][0]
    t, k = r["argmax"]
    assert r["max"] == pytest.approx(float(r["map"][t, k]))
    assert r["last"] == pytest.approx(float(r["map"][-1].max()))
    assert r["pgd_gain"] >= 1.0


# --- the model attack -----------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["bounded", "softplus"])
def test_model_attack_is_first_order_optimal_and_sound(arm):
    model, m64 = _model(arm, seed=7)
    x = torch.randn(1, 12, 5)
    eps_list = [1e-6, 1e-3, 1e-1]
    res = attack.attack_model(m64, x.to(D64), eps_list, _small_budget())
    for row in res["per_eps"]:
        if row["eps"] <= 1e-3:
            assert row["width"] >= row["first_order_width"] * (1 - 1e-4)
        for rule in attack.RULES:
            bound = float(local_sensitivity_rule(model, x, row["eps"], rule)["width"])
            if math.isfinite(bound):
                assert row["width"] <= bound * (1 + 1e-6), (row["eps"], rule)
    # random corners at the smallest eps do no better than the gradient sign
    e = eps_list[0]
    torch.manual_seed(8)
    corners = x.to(D64) + e * torch.where(torch.rand(64, 12, 5) > 0.5, 1.0, -1.0).to(D64)
    f = attack.predict_reference(m64, corners)
    assert float(f.max() - f.min()) <= res["per_eps"][0]["width"] * (1 + 1e-4)


# --- the stage trace ------------------------------------------------------------------------


@pytest.mark.parametrize("rule", attack.RULES)
@pytest.mark.parametrize("eps", [1e-6, 1e-3, 1e-1])
def test_stage_trace_reproduces_the_domain_and_telescopes(rule, eps):
    model, _ = _model(seed=9)
    torch.manual_seed(10)
    x = torch.randn(2, 12, 5)
    tr = attack.trace_local(model, x, eps, rule)
    ref = local_sensitivity_rule(model, x, eps, rule)
    torch.testing.assert_close(tr["lipschitz"], ref["lipschitz"], rtol=0.0, atol=0.0,
                               equal_nan=True)
    final = float(ref["lipschitz"].max())
    if math.isfinite(final):
        assert tr["first_non_finite_stage"] is None
        logs = [s["log10_ratio"] for s in tr["stages"]]
        assert sum(logs) == pytest.approx(math.log10(final), abs=1e-9)
        assert sum(tr["log10_by_group"].values()) == pytest.approx(sum(logs), abs=1e-9)
        assert set(tr["log10_by_group"]) == set(attack.GROUPS)
    else:
        assert tr["first_non_finite_stage"] is not None
        assert tr["log10_by_group"] is None


def test_without_the_timescale_term_the_rules_agree():
    """With no timescale sensitivity the additive step and the recursive log-timescale step
    bound the same convex recursion of the forcing term, so the counterfactual run must come out
    the same under both, to the second-order terms they keep. The running maximum drops the
    decay of that recursion, so it can only be larger."""
    model, _ = _model(seed=11)
    torch.manual_seed(12)
    x = torch.randn(2, 12, 5)
    eps = 1e-7
    blocks = {rule: attack.trace_local(model, x, eps, rule)["blocks"] for rule in attack.RULES}
    assert blocks["additive"][0]["h_lip_max_without_timescale"] == pytest.approx(
        blocks["log_delta_recursive"][0]["h_lip_max_without_timescale"], rel=1e-4)
    assert (blocks["log_delta"][0]["h_lip_max_without_timescale"]
            >= blocks["log_delta_recursive"][0]["h_lip_max_without_timescale"] * (1 - 1e-6))
    for per_rule in blocks.values():
        for blk in per_rule:
            assert blk["h_lip_max"] >= blk["h_lip_max_without_timescale"] * (1 - 1e-6)


@pytest.mark.parametrize("eps", [1e-6, 1e-2])
def test_log_timescale_steps_never_scale_the_carried_bound_up(eps):
    model, _ = _model(seed=13)
    torch.manual_seed(14)
    x = torch.randn(2, 16, 5)
    for rule in ("log_delta", "log_delta_recursive", "log_delta_endpoint"):
        for blk in attack.trace_local(model, x, eps, rule)["blocks"]:
            assert blk["fraction_of_steps_growing"] == 0.0
            assert blk["step_multiplier_max"] <= 1.0
            assert 0.0 <= blk["fraction_of_steps_raised"] <= 1.0
    if eps <= 1e-6:
        # below 1 in the reals; at large rho, exp(-x e^-rho) rounds to 1 in float32
        for rule in ("log_delta_recursive", "log_delta_endpoint"):
            for blk in attack.trace_local(model, x, eps, rule)["blocks"]:
                assert blk["step_multiplier_max"] < 1.0
    # the endpoint rule's multiplier is the upper end of a sub-interval of the recursion's
    rec = attack.trace_local(model, x, eps, "log_delta_recursive")["blocks"]
    end = attack.trace_local(model, x, eps, "log_delta_endpoint")["blocks"]
    for r, e in zip(rec, end):
        assert e["step_multiplier_max"] <= r["step_multiplier_max"] * (1 + 1e-6)


# --- the report -----------------------------------------------------------------------------


def test_report_renders_from_the_recorded_results(tmp_path):
    """The markdown twin and the figure are rebuilt from docs/certification/incremental_attack.json,
    so a change to the report code that breaks on the real results fails here, not after an
    hour-long run."""
    import json

    from src.utils.repro import REPO_ROOT

    recorded = REPO_ROOT / "docs/certification/incremental_attack.json"
    if not recorded.exists():
        pytest.skip("incremental_attack.json not found")
    results = json.loads(recorded.read_text())
    if any(rule not in row for a in results["arms"].values()
           for row in a["local_end_to_end"]["rows"] for rule in attack.RULES):
        pytest.skip("incremental_attack.json predates the endpoint rule; rerun "
                    "scripts/13_incremental_attack.py")
    md = attack.markdown(results)
    for heading in ("## Block output against sequence length", "## The prediction against",
                    "## The local end-to-end certificate", "## Where the end-to-end bound comes from",
                    "## Scope"):
        assert heading in md
    assert "nan |" not in md.split("## The local end-to-end")[0]

    lengths = results["data"]["lengths"]
    curves = {arm: {"L": lengths, "ours": a["block"]["layers"][0]["certified"]["ours_slope_max"],
                    "no_floor": [a["block"]["layers"][0]["certified"]["additive_slope_max"][str(L)]
                                 ["no_floor"] for L in lengths],
                    "certified_floor": [a["block"]["layers"][0]["certified"]["additive_slope_max"]
                                        [str(L)]["certified_floor"] for L in lengths]}
              for arm, a in results["arms"].items()}
    attack.figure(results["arms"], lengths, tmp_path / "figure.pdf", curves)
    assert (tmp_path / "figure.pdf").stat().st_size > 0


# --- the windows ----------------------------------------------------------------------------


class _Data:
    """The part of LOBDataModule the window helpers use, on a synthetic series whose row
    index can be read back from the features."""

    def __init__(self, rows: int = 600):
        self.features = np.repeat(np.arange(rows, dtype=np.float32)[:, None], 3, axis=1)
        self.targets = np.zeros(rows, dtype=np.float32)
        self.idx_test = (350, rows)
        self.seq_len = 20
        self.horizon = 5

    def dataset(self, split: str) -> LOBWindows:
        assert split == "test"
        return LOBWindows(self.features, self.targets, *self.idx_test, self.seq_len,
                          self.horizon, purge=False)


def test_windows_end_at_the_requested_rows_inside_the_test_split():
    dm = _Data()
    ends = attack.window_ends(dm, 4, 200)
    assert ends[0] == 350 + 199 and ends[-1] == 599
    assert ends == sorted(ends)
    for L in (10, 200):
        x = attack.held_out_windows(dm, L, ends)
        assert x.shape == (4, L, 3)
        assert x[:, -1, 0].tolist() == [float(e) for e in ends]
        assert torch.all(x[:, 0, 0] >= 350)
    assert dm.seq_len == 20
    with pytest.raises(ValueError):
        attack.held_out_windows(dm, 300, ends)
