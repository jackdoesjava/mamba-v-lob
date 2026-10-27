"""The tightness search of scripts/16_cell_tightness.py: its scan, its bounds and its search.

All in float64 and fast: the scan against the plain recursion, random admissible sequences against
the theorem, a short search at one rho, and the additive baseline against eq:additive by hand.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import torch

tight = importlib.import_module("scripts.16_cell_tightness")


def test_scan_matches_loop():
    gen = torch.Generator().manual_seed(0)
    for T in (1, 2, 17, 400):
        x = torch.exp(torch.empty(5, T, dtype=torch.float64).uniform_(math.log(1e-8), math.log(20),
                                                                         generator=gen))
        c = torch.empty(5, T, dtype=torch.float64).uniform_(-3, 3, generator=gen)
        for h0 in (0.0, 0.7):
            fast = tight.final_state(x, c, h0)
            for b in range(5):
                slow = tight.scan_loop(x[b].numpy(), c[b].numpy(), h0)[-1]
                assert abs(float(fast[b]) - slow) <= 1e-12 * (1 + abs(slow))


def test_random_admissible_sequences_respect_the_theorem():
    rng = np.random.default_rng(1)
    for _ in range(300):
        T = int(rng.integers(1, 60))
        rho = float(rng.choice([0.01, 0.1, 0.5, 2.0]))
        eta = float(rng.choice([0.0, 0.1, 1.0]))
        m = float(rng.uniform(0.1, 3.0))
        x = np.exp(rng.uniform(math.log(1e-6), math.log(30), T))
        s = rng.choice([-1.0, 1.0], T) if rng.random() < 0.5 else rng.uniform(-1, 1, T)
        xt = x * np.exp(rho * s)
        ct = m * (rng.choice([-1.0, 1.0], T) if rng.random() < 0.5 else rng.uniform(-1, 1, T))
        c = ct + eta * rng.uniform(-1, 1, T)
        gap = np.abs(tight.scan_loop(x, c) - tight.scan_loop(xt, ct))
        assert gap.max() <= tight.theorem_bound(eta, m, rho) * (1 + 1e-12)
        assert gap.max() <= tight.common_start_constant(eta, m, rho) * (1 + 1e-12)


def test_short_search_is_close_but_below_the_bound():
    gen = torch.Generator().manual_seed(2)
    row = tight.search_a(T=40, rho=0.3, eta=0.0, steps=60, n_random=4, gen=gen)
    assert row["G"] <= row["B"]
    assert row["G_over_B"] > 0.3
    assert row["scan_vs_loop_abs_diff"] < 1e-12


def test_additive_baseline_matches_eq_additive_by_hand():
    # eta + 2 M lam e^{-lam dt_lo} eta_dt sum_{k=0}^{L-1} e^{-k lam dt_lo}, eta = 0.1, M = 2,
    # lam = 0.5, eta_dt = 0.01
    # no floor, L = 7: 0.1 + 2 * 2 * 0.5 * 0.01 * 7 = 0.24
    assert math.isclose(tight.additive_baseline(0.1, 2.0, 0.5, 0.01, 0.0, 7), 0.24, rel_tol=1e-14)
    # floor dt_lo = 0.2, L = 3: q = e^{-0.1}, 0.1 + 0.02 q (1 + q + q^2) = 0.1492877278...
    q = math.exp(-0.1)
    by_hand = 0.1 + 0.02 * q * (1 + q + q * q)
    assert math.isclose(by_hand, 0.1492877278, rel_tol=1e-9)
    assert math.isclose(tight.additive_baseline(0.1, 2.0, 0.5, 0.01, 0.2, 3), by_hand,
                        rel_tol=1e-14)
    # a floor of 1e-21 is no floor at any length shown
    assert math.isclose(tight.additive_baseline(0.0, 1.0, 1.0, 0.01, 1e-21, 3000),
                        tight.additive_baseline(0.0, 1.0, 1.0, 0.01, 0.0, 3000), rel_tol=1e-12)


def test_crossing_length_reaches_the_bound():
    bound = tight.theorem_bound(0.0, 1.0, 0.01)
    L = tight.crossing_length(0.0, 1.0, 1.0, 2.58e-4, 0.0, bound)
    assert math.isclose(L, bound / (2 * 2.58e-4), rel_tol=1e-14)
    L = tight.crossing_length(0.0, 1.0, 1.0, 2.58e-4, 1e-3, bound)
    assert math.isclose(tight.additive_baseline(0.0, 1.0, 1.0, 2.58e-4, 1e-3, L), bound,
                        rel_tol=1e-10)
    # saturated below the bound: no crossing
    assert tight.crossing_length(0.0, 1.0, 1.0, 1e-6, 1.0, bound) is None


def test_model_constants_follow_the_paper():
    k = tight.model_constants("bounded")
    # Z contains 0, so sup |(log dt)'| = ln(100)/4 (lem:param (ii)), and the floored ratio of
    # rem:cap is sup|dt'| / (dt_min sup|(log dt)'|), 25.8 at dt_max/dt_min = 100
    assert math.isclose(k["sup_abs_log_dt_slope"], math.log(100) / 4, rel_tol=1e-12)
    assert math.isclose(k["rho"], tight.RHO_B, rel_tol=1e-12)
    assert abs(k["sup_abs_dt_slope"] / (1e-3 * k["sup_abs_log_dt_slope"]) - 25.8) < 0.05
    s = tight.model_constants("softplus")
    assert math.isclose(s["sup_abs_log_dt_slope"], 1.0, rel_tol=1e-12)
    assert 1e-21 < s["dt_floor"] < 2e-21
