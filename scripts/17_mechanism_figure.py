"""The idea of the cell theorem and the construction that bounds it from below, for one cell.

Left: the change of the transition e^{-x} with x = lam dt, against the input weight 1 - e^{-x}
of the step, for a fixed change of log dt (rho = 0.1) and for a fixed additive change of
lam dt (0.01). The log change shrinks with the input weight, as Lemma 2 states; the additive
change does not, and below x of about 0.01 it exceeds the whole input weight.

Right: the gap |h_t - h~_t| between the two runs of the construction of Proposition 4 at
rho = 0.5, M = 1, eta = 0 and tau = 0.05: the equilibrium of the perturbed run alternates between
M and -M, and the perturbed timescale is e^{-rho} times the nominal one on the steps with
c~ = M and e^{rho} times it on the others. The gap is drawn against the value
2 M tanh(rho/2) that the construction approaches as tau -> 0, the bound
E = 2 M (e^rho - 1) of Theorem 3 and the trivial bound 2 M. Everything is closed form or a
direct run of the recursion h_t = a_t h_{t-1} + (1 - a_t) c_t; nothing is searched or trained.

Writes docs/figures/mechanism.{pdf,png}.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

# the bounds and the figure style of the one-cell search
cell = importlib.import_module("scripts.16_cell_tightness")

FIG_DIR = Path("docs/figures")
X = np.geomspace(1e-4, 10.0, 400)  # lam dt
RHO_A = 0.1  # change of log dt in the left panel
ADD_A = 0.01  # change of lam dt in the left panel
RHO_B, TAU, STEPS, M = 0.5, 0.05, 120, 1.0  # the construction in the right panel


def construction(rho: float, tau: float, steps: int, m: float) -> np.ndarray:
    """|h_t - h~_t| for t = 0 .. steps - 1 of Proposition 4's runs with eta = 0, from 0."""
    q = math.exp(rho)
    h = ht = 0.0
    gaps = []
    for t in range(steps):
        if t % 2 == 0:  # c~ = M, nominal lam dt = q tau, perturbed tau
            a, at, c = math.exp(-q * tau), math.exp(-tau), m
        else:  # c~ = -M, nominal lam dt = tau, perturbed q tau
            a, at, c = math.exp(-tau), math.exp(-q * tau), -m
        h = a * h + (1.0 - a) * c
        ht = at * ht + (1.0 - at) * c
        gaps.append(abs(h - ht))
    return np.array(gaps)


def main() -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=cell.WIDE, layout="constrained")

    weight = -np.expm1(-X)
    log_change = np.exp(-X) - np.exp(-X * math.exp(RHO_A))
    add_change = np.exp(-X) * -math.expm1(-ADD_A)
    left.loglog(X, weight, color=cell.INK, linewidth=1.2, linestyle="--",
                label=r"input weight $1 - \bar A$")
    left.loglog(X, log_change, color=cell.OURS, linewidth=2.0,
                label=rf"change for $\log\Delta$ moved by ${RHO_A:g}$")
    left.loglog(X, add_change, color=cell.OTHER, linewidth=1.4,
                label=rf"change for $\lambda\Delta$ moved by ${ADD_A:g}$")
    left.set_xlabel(r"$\lambda\Delta$")
    left.set_ylabel(r"change of the transition $\bar A = e^{-\lambda\Delta}$")
    left.set_ylim(1e-7, 2.0)
    left.set_title("(a) one step", fontsize=9, loc="left", color=cell.INK)
    left.legend(loc="lower right", frameon=False, fontsize=6.5)

    gaps = construction(RHO_B, TAU, STEPS, M)
    t = np.arange(STEPS)
    theorem = cell.theorem_bound(0.0, M, RHO_B)
    limit = cell.common_start_constant(0.0, M, RHO_B)
    right.axhline(2.0 * M, color="0.6", linewidth=1.0, linestyle=":", label=r"trivial bound $2M$")
    right.axhline(theorem, color=cell.OURS, linewidth=2.0, label="Theorem 3")
    right.axhline(limit, color=cell.LOCAL, linewidth=1.2, linestyle="--",
                  label=r"$2M\tanh(\rho/2)$")
    right.plot(t, gaps, color=cell.INK, linewidth=0.9, marker="o", markersize=1.6,
               label=r"$|h_t - \tilde h_t|$")
    right.set_xlabel("step $t$")
    right.set_ylim(0.0, 2.15)
    right.set_xlim(0, STEPS - 1)
    right.set_title(rf"(b) Proposition 4, $\rho = {RHO_B:g}$", fontsize=9, loc="left",
                    color=cell.INK)
    # between the gap (about 0.49) and the theorem's bound (about 1.30)
    right.legend(loc="center right", bbox_to_anchor=(1.0, 0.43), frameon=False, fontsize=6.5)
    for ax in (left,):
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cell.save(fig, FIG_DIR / "mechanism")

    small = X < 1e-3
    print(f"left: log change / input weight in [{(log_change / weight).min():.4g}, "
          f"{(log_change / weight).max():.4g}], e^rho - 1 = {math.expm1(RHO_A):.4g}")
    # e^{-x}(1 - e^{-ADD_A}) = 1 - e^{-x} at x = log(2 - e^{-ADD_A})
    print(f"left: additive change exceeds the input weight below lam dt = "
          f"{math.log(2.0 - math.exp(-ADD_A)):.5f}; the changes of lam dt are equal at "
          f"{ADD_A / math.expm1(RHO_A):.4f}; at lam dt = 1e-4 it is "
          f"{(add_change / weight)[0]:.4g} times the weight ({small.sum()} points below 1e-3)")
    even = gaps[0::2]
    print(f"right: gap after the last even step {even[-1]:.5f}, limit at this tau "
          f"{2 * M * (math.exp(-TAU) - math.exp(-math.exp(RHO_B) * TAU)) / (1 - math.exp(-(math.exp(RHO_B) + 1) * TAU)):.5f}, "
          f"2M tanh(rho/2) = {limit:.5f}, E = {theorem:.5f}, max gap {gaps.max():.5f}")
    print(f"wrote {FIG_DIR / 'mechanism.pdf'}")


if __name__ == "__main__":
    main()
