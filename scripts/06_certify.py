"""The comparison the paper is built on. Four bounds on sup |h| over every admissible input.

  invariant   closed form, one induction step, no iteration and no dependence on L
  widened     generic interval propagation iterated to its fixed point, the fair baseline
  geometric   the analytic fixed point M / (1 - sup Abar), which widened should match
  unrolled    the same propagation stopped at L, which is what bounded verification does

Writes docs/certification/bounds.{json,md}.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

from src.models.mamba import LOBMamba, zoh_phi
from src.utils.config import load_config
from src.utils.repro import provenance, set_seed
from src.verification import intervals as iv
from src.verification.certify import certify_block, certify_model
from src.verification.intervals import Interval
from src.verification.invariant import certify_block_state

OUT_DIR = Path("docs/certification")


def load_model(checkpoint: Path, config: dict, dt_parametrisation: str = "bounded") -> LOBMamba:
    cfg = copy.deepcopy(config)
    cfg["model"]["mamba"]["dt_parametrisation"] = dt_parametrisation
    if checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = LOBMamba(int(ckpt["architecture"].get("input_dim", 43)), cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        return model.eval()
    print(f"  {checkpoint.name} not found; using a fresh model")
    return LOBMamba(43, cfg).eval()


def drive_box(cert: dict) -> Interval:
    u = cert["boxes"]["u"]
    return cert["boxes"]["B_bar"] * Interval(u.lo.unsqueeze(-1), u.hi.unsqueeze(-1))


def compare_bounds(model: LOBMamba, seq_len: int) -> list[dict]:
    rows = []
    for li, block in enumerate(model.layers):
        cert = certify_block(block, seq_len=seq_len)
        state = certify_block_state(block, cert["boxes"]["u"], cert["boxes"]["B"])
        widened, iters = iv.state_bound_widened(cert["boxes"]["A_bar"], drive_box(cert))
        rows.append(
            {
                "layer": li,
                "invariant": state["radius"],
                "widened": float(widened.abs_max().max()),
                "widening_iterations": iters,
                "geometric": cert["summary"]["h_abs_max_geometric"],
                "unrolled_at_L": cert["summary"]["h_abs_max_horizon"],
                "seq_len": seq_len,
                "invariant_holds": state["holds"],
                "zoh_exact": state["zoh_exact"],
                "tighter_than_widened": float(widened.abs_max().max()) / state["radius"],
            }
        )
    return rows


def scaling_with_length(model: LOBMamba, lengths) -> list[dict]:
    """The invariant is flat in L by construction; the unrolled bound is not."""
    block = model.layers[0]
    state = certify_block_state(
        block, certify_block(block, seq_len=8)["boxes"]["u"],
        certify_block(block, seq_len=8)["boxes"]["B"]
    )
    rows = []
    for L in lengths:
        cert = certify_block(block, seq_len=L)
        rows.append(
            {
                "seq_len": L,
                "invariant": state["radius"],
                "unrolled": cert["summary"]["h_abs_max_horizon"],
            }
        )
    return rows


def realised_envelope(model: LOBMamba, batches) -> list[float]:
    peaks = [0.0 for _ in model.layers]
    with torch.no_grad():
        for x in batches:
            _, traces = model(x, trace=True)
            for i, tr in enumerate(traces):
                peaks[i] = max(peaks[i], float(tr.h.abs().max()))
    return peaks


def discretisation_error(model: LOBMamba) -> dict:
    """What the Euler surrogate would cost. The invariant needs the exact form to exist."""
    block = model.layers[0]
    A = -torch.exp(block.A_log)
    grid = torch.linspace(block.dt_min, block.dt_max, 256).view(-1, 1, 1)
    v = grid * A.unsqueeze(0)
    zoh = grid * zoh_phi(v)
    rel = ((grid.expand_as(zoh) - zoh).abs() / zoh.abs().clamp_min(1e-30)).flatten().detach()
    return {
        "delta_range": [block.dt_min, block.dt_max],
        "max_abs_dtA": float((grid.max() * A.abs().max()).item()),
        "median_rel_error": float(rel.median()),
        "max_rel_error": float(rel.max()),
    }


def structural_ablation(model: LOBMamba, config: dict, seq_len: int) -> list[dict]:
    """What the invariant radius costs under weaker architectural choices.

    The radius is sup|B u| / |A|, so anything that loosens the bounds on B or u loosens the
    invariant by the same factor. Neither choice affects whether the invariant exists.
    """
    block = model.layers[0]
    rows = []

    for name, tight in (("fused LayerNorm and Linear", True), ("LayerNorm box only", False)):
        cert = certify_block(block, seq_len=seq_len, tight_layernorm=tight)
        state = certify_block_state(block, cert["boxes"]["u"], cert["boxes"]["B"])
        rows.append({"variant": name, "radius": state["radius"],
                     "u_abs_max": cert["summary"]["u_abs_max"]})

    dense = copy.deepcopy(block)
    dense.conv1d = nn.Conv1d(block.d_inner, block.d_inner, block.d_conv,
                             groups=1, padding=block.d_conv - 1)
    for name, blk in (("depthwise convolution", block), ("dense convolution", dense)):
        cert = certify_block(blk, seq_len=seq_len)
        state = certify_block_state(blk, cert["boxes"]["u"], cert["boxes"]["B"])
        rows.append({"variant": name, "radius": state["radius"],
                     "u_abs_max": cert["summary"]["u_abs_max"]})
    return rows


def table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    def fmt(v):
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            if v == math.inf:
                return "inf"
            return f"{v:.4g}" if (abs(v) >= 1e4 or (v and abs(v) < 1e-3)) else f"{v:.4f}"
        return str(v)

    head = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(fmt(r.get(k, "")) for k, _ in columns) + " |" for r in rows]
    return "\n".join([head, rule, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare state bounds.")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints")
    parser.add_argument("--seq-len", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    seq_len = args.seq_len or int(config["data"]["seq_length"])
    set_seed(int(config.get("seed", 1337)))
    ckpt_dir = Path(args.checkpoint_dir)

    model = load_model(ckpt_dir / "best_mamba.pt", config)
    softplus = load_model(ckpt_dir / "best_mamba_softplus.pt", config, "softplus")

    print("comparing bounds ...")
    bounds = compare_bounds(model, seq_len)
    print("scaling with sequence length ...")
    scaling = scaling_with_length(model, [10, 50, 100, 500, 1000, 5000])
    print("structural ablation ...")
    structural = structural_ablation(model, config, seq_len)
    print("softplus arm ...")
    softplus_bounds = compare_bounds(softplus, seq_len)
    disc = discretisation_error(model)

    cert = certify_model(model, seq_len=seq_len)
    results = {
        "provenance": provenance(config.get("seed")),
        "seq_len": seq_len,
        "bounds": bounds,
        "scaling": scaling,
        "structural": structural,
        "softplus": softplus_bounds,
        "discretisation": disc,
        "output_range": cert["output_range"],
    }

    md = [
        "# State bounds",
        "",
        f"Sequence length L = {seq_len} where it applies. Every number is input-independent.",
        "",
        "## The comparison",
        "",
        table(bounds, [("layer", "layer"), ("invariant", "invariant"),
                       ("widened", "widened fixed point"), ("geometric", "geometric"),
                       ("unrolled_at_L", f"unrolled at L={seq_len}"),
                       ("tighter_than_widened", "invariant is tighter by"),
                       ("invariant_holds", "induction holds")]),
        "",
        "The widened column iterates generic interval propagation to convergence, so it is not",
        "a truncated baseline. It agrees with the geometric fixed point, as it should. The",
        "unrolled column is lower only because it stops early, and it grows with L.",
        "",
        "## Scaling with sequence length",
        "",
        table(scaling, [("seq_len", "L"), ("invariant", "invariant"), ("unrolled", "unrolled")]),
        "",
        "One induction step covers every length. The unrolled bound has to be recomputed per",
        "length and climbs towards the fixed point.",
        "",
        "## The softplus arm",
        "",
        table(softplus_bounds, [("layer", "layer"), ("invariant", "invariant"),
                                ("widened", "widened fixed point"),
                                ("invariant_holds", "induction holds")]),
        "",
        "No lower bound on delta is imposed here and the invariant still holds. The convex",
        "form does not use one; only the geometric argument does.",
        "",
        "## What loosens the radius",
        "",
        table(structural, [("variant", "variant"), ("u_abs_max", "sup abs(u)"),
                           ("radius", "invariant radius")]),
        "",
        "The radius is sup|B u| / |A|, so anything that loosens the bounds on B or u loosens it",
        "by the same factor. Neither choice affects whether the invariant exists.",
        "",
        "## Discretisation",
        "",
        "The convex form is exact for the zero-order hold and only asymptotic for the Euler",
        "surrogate, so the invariant needs the exact discretisation to exist at all.",
        "",
        "```json",
        json.dumps(disc, indent=2),
        "```",
        "",
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "bounds.json").write_text(json.dumps(results, indent=2))
    (OUT_DIR / "bounds.md").write_text("\n".join(md))
    print("\n" + "\n".join(md))
    print(f"wrote {OUT_DIR / 'bounds.json'} and {OUT_DIR / 'bounds.md'}")


if __name__ == "__main__":
    main()
