"""A forward-invariant box for the SSM state, valid at any sequence length.

Exact zero-order hold gives Bbar = (1 - Abar) B / |A| for diagonal A < 0, so the update
h_t = Abar_t h_{t-1} + (1 - Abar_t) c_t with c_t = B_t u_t / |A| is a convex combination.
Bounding c bounds h, by induction and at any length. Derivation in docs/02-invariant.md.
"""

from __future__ import annotations

import torch

from src.models.mamba import SelectiveSSMBlock, zoh_phi
from src.verification.intervals import Interval


def zoh_is_exact(A: torch.Tensor, dt_max: float, tol: float = 1e-6) -> bool:
    """Check Bbar's multiplier really is (1 - Abar)/|A| over the timescale range.

    The invariant needs the exact discretisation. Reference Mamba's Euler surrogate dt*B
    matches only as dt -> 0, so this is what stops the bound being applied to a model it
    does not hold for.
    """
    dts = torch.logspace(-8, torch.log10(torch.tensor(dt_max)).item(), 32, dtype=torch.float64)
    A64 = A.to(torch.float64)
    for dt in dts:
        dtA = dt * A64
        exact = dt * zoh_phi(dtA)
        claimed = (1.0 - torch.exp(dtA)) / A64.abs()
        if (exact - claimed).abs().max() > tol:
            return False
    return True


def equilibrium_box(A: torch.Tensor, B: Interval, u: Interval) -> Interval:
    """c = B u / |A|, the value the state is pulled towards at each step.

    A is (d_inner, d_state); B is a box over states, u a box over channels. The result is
    (d_inner, d_state), one equilibrium per channel and state.
    """
    prod = Interval(B.lo.unsqueeze(0), B.hi.unsqueeze(0)) * Interval(
        u.lo.unsqueeze(-1), u.hi.unsqueeze(-1)
    )
    inv_A = 1.0 / A.abs()
    return prod * inv_A


def invariant_box(A: torch.Tensor, B: Interval, u: Interval) -> Interval:
    """The forward-invariant box [-M, M], with M taken elementwise over (channel, state)."""
    radius = equilibrium_box(A, B, u).abs_max()
    return Interval(-radius, radius)


def induction_residual(invariant: Interval, equilibrium: Interval) -> torch.Tensor:
    """How far the one-step image escapes the box. Non-positive everywhere means invariant.

    Abar appears in both terms of a h + (1 - a) c, so this step cannot be evaluated with
    interval arithmetic. Bounding the two factors independently lets the first take sup Abar
    and the second take 1 - inf Abar, which sum to more than one, and the check then fails
    for any non-degenerate box. Keeping a as a single variable, the supremum over a in [0, 1]
    of a M + (1 - a) M_c is max(M, M_c), so the box is preserved exactly when M_c <= M.

    That is the same dependency loss that makes the geometric bound diverge, one level up.
    Checked rather than assumed.
    """
    M = invariant.abs_max()
    M_c = equilibrium.abs_max()
    return (torch.maximum(M, M_c) - M).detach()


def certify_block_state(block: SelectiveSSMBlock, u: Interval, B: Interval) -> dict:
    """Invariant state bound for one block, with the induction step checked numerically."""
    A = -torch.exp(block.A_log)
    equilibrium = equilibrium_box(A, B, u)
    invariant = invariant_box(A, B, u)

    delta = Interval(
        torch.full_like(A[:, :1], block.dt_min if block.dt_parametrisation == "bounded" else 0.0),
        torch.full_like(A[:, :1], block.dt_max if block.dt_parametrisation == "bounded" else 1e3),
    )
    A_bar = Interval(torch.exp(delta.hi * A), torch.exp(delta.lo * A))

    residual = induction_residual(invariant, equilibrium)
    return {
        "invariant": invariant,
        "equilibrium": equilibrium,
        "A_bar": A_bar,
        "radius": float(invariant.hi.max().detach()),
        "induction_residual": float(residual.max()),
        "holds": bool(residual.max() <= 1e-6),
        "zoh_exact": zoh_is_exact(A, block.dt_max),
    }
