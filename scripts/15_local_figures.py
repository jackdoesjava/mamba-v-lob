"""Figures of the local certificates against gradient attacks, drawn from saved artefacts.

Reads docs/certification/incremental_attack.json (scripts/13_incremental_attack.py) and
docs/certification/released.json (scripts/14_released.py), and writes

  docs/figures/sensitivity_local.{pdf,png}   block 0 of each trained arm against the length
  docs/figures/end_to_end.{pdf,png}          the chain through the whole trained network
  docs/figures/released_local.{pdf,png}      mamba-130m-hf, layer by layer

Nothing is recomputed. Every plotted value is a number already in those two files, so this
script runs in seconds and its figures are current whenever the files are. It draws only the
two local rules the paper compares, the additive rule and the endpoint rule; the running
maximum and the recursion stay in the tables of the two reports.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

CERT_DIR = Path("docs/certification")
FIG_DIR = Path("docs/figures")
DELTA = 1e-6   # the smallest perturbation, where the slopes describe the linear regime
KAPPA = 1e-6   # the network-input radius of the end-to-end chain

# the colours and type of scripts/13_incremental_attack.py and scripts/14_released.py
OURS = "#1f77b4"
OTHER = "#ff7f0e"
LOCAL = "#6f42c1"
INK = "#212529"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)
WIDE = (6.5, 3.0)
COLUMN = (3.4, 4.8)
LENGTH_STYLE = {64: dict(linestyle=":", marker="o"), 256: dict(linestyle="--", marker="s"),
                1024: dict(linestyle="-", marker="D")}


def _pos(v) -> float:
    return v if (v is not None and math.isfinite(v) and v > 0) else np.nan


def _index(values: list[float], target: float) -> int:
    return min(range(len(values)), key=lambda i: abs(math.log10(values[i]) - math.log10(target)))


def save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def sensitivity_local(attack: dict, path: Path) -> dict:
    """Block 0 of each arm at delta = 1e-6 against the length: the data-free bound of the
    theorem, the additive and endpoint local rules, and the attacks."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True, layout="constrained")
    summary = {}
    for ax, (arm, a) in zip(axes, attack["arms"].items()):
        by_length = a["block"]["layers"][0]["lengths"]
        lengths = sorted(int(L) for L in by_length)
        rows = [by_length[str(L)] for L in lengths]
        k = _index([p["delta"] for p in rows[0]["per_delta"]], DELTA)
        pd = [r["per_delta"][k] for r in rows]
        d = pd[0]["delta"]
        endpoint = [_pos(p["local"]["log_delta_endpoint"]["lip_max"]) for p in pd]
        additive = [_pos(p["local"]["additive"]["lip_max"]) for p in pd]
        attacked = [p["slope_max"] for p in pd]
        ax.loglog(lengths, [p["data_free_bound_max"] / d for p in pd], color=OURS, linewidth=2.0,
                  label="data-free, log-timescale")
        ax.loglog(lengths, additive, color=OTHER, linewidth=1.4, marker="s", markersize=4,
                  label="local, additive rule")
        ax.loglog(lengths, endpoint, color=LOCAL, linewidth=1.4, marker="D", markersize=4,
                  label="local, endpoint rule")
        ax.loglog(lengths, attacked, color=INK, linewidth=1.2, marker="o", markersize=5,
                  label="attack, worst position")
        ax.loglog(lengths, [p["last_slope_max"] for p in pd], color=INK, linewidth=1.0,
                  linestyle=":", marker="o", markersize=5, markerfacecolor="white",
                  label="attack, last position")
        ax.set_title(f"{arm} arm, block 0", fontsize=9, loc="left", color=INK)
        ax.set_xlabel("sequence length $L$")
        ax.set_xlim(lengths[0] / 1.5, lengths[-1] * 1.5)
        summary[arm] = {
            "endpoint_over_attack": [round(e / s, 2) for e, s in zip(endpoint, attacked)],
            "additive_over_attack": [round(x / s, 2) for x, s in zip(additive, attacked)],
        }
    axes[0].set_ylabel(r"$\|F(x) - F(\tilde x)\|_\infty / \delta$, $\delta = 10^{-6}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    save(fig, path)
    return summary


def end_to_end(attack: dict, path: Path) -> dict:
    """The chain from the network input to the prediction at kappa = 1e-6: the additive and
    endpoint rules against the attack, as a width over 2 kappa. Not a certificate (float32,
    no outward rounding); the paper reports it as a measurement."""
    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True, layout="constrained")
    summary = {}
    for ax, (arm, a) in zip(axes, attack["arms"].items()):
        by_length = a["model"]["lengths"]
        lengths = sorted(int(L) for L in by_length)
        j = _index([p["eps"] for p in by_length[str(lengths[0])]["per_eps"]], KAPPA)
        rows = [by_length[str(L)]["per_eps"][j] for L in lengths]
        e = rows[0]["eps"]
        additive = [r["additive"]["width_median"] / (2 * e) for r in rows]
        endpoint = [r["log_delta_endpoint"]["width_median"] / (2 * e) for r in rows]
        attacked = [r["attack_width_median"] / (2 * e) for r in rows]
        ax.loglog(lengths, [_pos(v) for v in additive], color=OTHER, linewidth=1.4, marker="s",
                  markersize=4, label="additive rule")
        ax.loglog(lengths, [_pos(v) for v in endpoint], color=LOCAL, linewidth=1.4, marker="D",
                  markersize=4, label="endpoint rule")
        ax.loglog(lengths, attacked, color=INK, linewidth=1.2, marker="o", markersize=5,
                  label="attack")
        bad = [L for L, v in zip(lengths, additive) if not math.isfinite(v)]
        good = [(L, v) for L, v in zip(lengths, additive) if math.isfinite(v)]
        if bad and good:
            L_last, v_last = good[-1]
            ax.annotate(f"additive bound infinite\nfrom L = {bad[0]}", xy=(L_last, v_last),
                        xytext=(bad[0], v_last / 100), fontsize=7.5, color=INK, ha="center",
                        va="top", arrowprops={"arrowstyle": "-", "color": OTHER, "linewidth": 0.8})
        ax.set_title(f"{arm} arm, prediction", fontsize=9, loc="left", color=INK)
        ax.set_xlabel("sequence length $L$")
        ax.set_xlim(lengths[0] / 1.5, lengths[-1] * 1.5)
        summary[arm] = {"endpoint_over_attack": [round(x / s) for x, s in zip(endpoint, attacked)]}
    axes[0].set_ylabel(r"width $/\, 2\kappa$, $\kappa = 10^{-6}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    save(fig, path)
    return summary


def released_local(rel: dict, path: Path) -> dict:
    """mamba-130m-hf at delta = 1e-6, layer by layer: the endpoint rule over the additive rule
    at every length (top), and the endpoint rule over the attack where a layer was attacked
    (bottom)."""
    deltas = rel["run"]["deltas"]
    j = _index(deltas, DELTA)
    lengths = rel["run"]["lengths"]
    exp = {(e["layer"], e["seq_len"]): e["per_delta"][j] | {"attacked": e["attacked"]}
           for e in rel["experiments"]}
    layers = sorted({e["layer"] for e in rel["experiments"]})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=COLUMN, sharex=True, layout="constrained")
    ratios, over_attack = [], []
    for L in lengths:
        r = [exp[(l, L)]["local"]["log_delta_endpoint"]["lip_max"]
             / exp[(l, L)]["local"]["additive"]["lip_max"] for l in layers]
        ratios += r
        top.plot(layers, r, color=LOCAL, linewidth=1.0, markersize=3.5, markerfacecolor="white",
                 label=f"$L = {L}$", **LENGTH_STYLE.get(L, {}))
        att = [l for l in layers if exp[(l, L)]["attacked"]]
        q = [exp[(l, L)]["local"]["log_delta_endpoint"]["lip_max"] / exp[(l, L)]["slope_max"]
             for l in att]
        over_attack += q
        # markers only: at the longest length only some layers were attacked, and a line
        # between them would suggest values that were not measured
        bottom.semilogy(att, q, color=INK, linestyle="none", markersize=4, markerfacecolor="white",
                        marker=LENGTH_STYLE.get(L, {}).get("marker", "o"))
    top.axhline(1.0, color=OTHER, linewidth=1.2, linestyle="--")
    top.text(layers[-1], 1.0, "additive rule", color=OTHER, fontsize=7.5, ha="right", va="bottom")
    top.set_ylim(min(ratios) - 0.05, 1.06)
    top.set_ylabel("endpoint / additive")
    bottom.set_ylabel("endpoint / attack")
    bottom.set_yticks([60, 100, 200, 500])
    bottom.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    bottom.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    bottom.set_xlabel("layer")
    bottom.set_xticks(list(range(0, len(layers), 4)) + [len(layers) - 1])
    handles, labels = top.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    save(fig, path)
    return {"endpoint_over_additive": [round(min(ratios), 3), round(max(ratios), 3)],
            "pairs_below_one": f"{sum(1 for x in ratios if x < 1)} of {len(ratios)}",
            "endpoint_over_attack": [round(min(over_attack), 1), round(max(over_attack), 1)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cert-dir", default=str(CERT_DIR))
    parser.add_argument("--fig-dir", default=str(FIG_DIR))
    args = parser.parse_args()
    cert, figs = Path(args.cert_dir), Path(args.fig_dir)
    figs.mkdir(parents=True, exist_ok=True)
    attack = json.loads((cert / "incremental_attack.json").read_text(encoding="utf-8"))
    rel = json.loads((cert / "released.json").read_text(encoding="utf-8"))
    print("sensitivity_local:", sensitivity_local(attack, figs / "sensitivity_local.pdf"))
    print("end_to_end:", end_to_end(attack, figs / "end_to_end.pdf"))
    print("released_local:", released_local(rel, figs / "released_local.pdf"))


if __name__ == "__main__":
    main()
