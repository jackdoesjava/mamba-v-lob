"""Certification ablation: delta parametrisation, conv structure, relaxation, horizon, ZOH.

Writes docs/certification/ablation.{json,md}. See docs/04-stability.md.
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
from src.verification.certify import certify_block, certify_model

OUT_DIR = Path("docs/certification")


def load_or_init(checkpoint: Path | None, config: dict, dt_param: str) -> LOBMamba:
    cfg = copy.deepcopy(config)
    cfg["model"]["mamba"]["dt_parametrisation"] = dt_param
    if checkpoint is not None and checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        arch = ckpt.get("architecture", {})
        model = LOBMamba(input_dim=int(arch.get("input_dim", 43)), config=cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        source = str(checkpoint)
    else:
        model = LOBMamba(input_dim=int(config["data"].get("input_dim", 43)), config=cfg)
        source = "randomly initialised (certificate is weight-dependent but data-free)"
    model.eval()
    model._source = source  # type: ignore[attr-defined]
    return model


def ablate_delta(config: dict, seq_len: int, ckpt_dir: Path) -> list[dict]:
    rows = []
    for dt_param, ckpt_name in (("bounded", "best_mamba.pt"), ("softplus", "best_mamba_softplus.pt")):
        set_seed(int(config.get("seed", 1337)))
        model = load_or_init(ckpt_dir / ckpt_name, config, dt_param)
        cert = certify_model(model, seq_len=seq_len)
        layer = cert["layers"][0]
        s = layer["summary"]
        rows.append(
            {
                "dt_parametrisation": dt_param,
                "source": model._source,  # type: ignore[attr-defined]
                "inf_delta": layer["contraction"]["dt_min"],
                "sup_A_bar": s["sup_A_bar"],
                "geometric_gain": layer["contraction"]["geometric_gain"],
                "horizon_gain": layer["horizon_gain"],
                "h_bound_geometric": s["h_abs_max_geometric"],
                "h_bound_horizon": s["h_abs_max_horizon"],
                "contractive": layer["contraction"]["contractive"],
                "certified_output_range": [
                    cert["output_range"]["lo"],
                    cert["output_range"]["hi"],
                ],
            }
        )
    return rows


def ablate_conv(config: dict, seq_len: int, ckpt_dir: Path) -> list[dict]:
    """Depthwise vs channel-mixing conv, each left on PyTorch's default fan-in init."""
    rows = []
    set_seed(int(config.get("seed", 1337)))
    base = load_or_init(ckpt_dir / "best_mamba.pt", config, "bounded")
    block = base.layers[0]

    dense = copy.deepcopy(block)
    dense.conv1d = nn.Conv1d(
        block.d_inner, block.d_inner, block.d_conv, groups=1, padding=block.d_conv - 1
    )

    for name, blk in (("depthwise", block), ("dense", dense)):
        cert = certify_block(blk, seq_len=seq_len)
        w = blk.conv1d.weight.abs()
        # Summing all of (in, kernel) is right for both: a depthwise weight is
        # (d_inner, 1, d_conv), so the in axis is a singleton.
        l1_gain = w.sum(dim=(1, 2)).max().item()
        rows.append(
            {
                "conv": name,
                "groups": blk.conv1d.groups,
                "conv_params": blk.conv1d.weight.numel(),
                "max_l1_gain": l1_gain,
                "u_abs_max": cert["summary"]["u_abs_max"],
                "B_abs_max": cert["summary"]["B_abs_max"],
                "h_bound_horizon": cert["summary"]["h_abs_max_horizon"],
            }
        )
    return rows


def ablate_relaxation(config: dict, seq_len: int, ckpt_dir: Path) -> list[dict]:
    rows = []
    set_seed(int(config.get("seed", 1337)))
    model = load_or_init(ckpt_dir / "best_mamba.pt", config, "bounded")
    for tight, name in ((True, "layernorm_linear_fused (exact)"), (False, "layernorm_box (sound)")):
        cert = certify_model(model, seq_len=seq_len, tight_layernorm=tight)
        s = cert["layers"][0]["summary"]
        rows.append(
            {
                "relaxation": name,
                "u_abs_max": s["u_abs_max"],
                "B_abs_max": s["B_abs_max"],
                "h_bound_horizon": s["h_abs_max_horizon"],
                "output_range": [cert["output_range"]["lo"], cert["output_range"]["hi"]],
            }
        )
    return rows


def ablate_horizon(config: dict, ckpt_dir: Path, lengths=(10, 50, 100, 500, 1000)) -> list[dict]:
    set_seed(int(config.get("seed", 1337)))
    model = load_or_init(ckpt_dir / "best_mamba.pt", config, "bounded")
    block = model.layers[0]
    s = block.contraction_certificate()["sup_A_bar"]
    rows = []
    for L in lengths:
        cert = certify_block(block, seq_len=L)
        rows.append(
            {
                "seq_len": L,
                "sup_A_bar": s,
                "horizon_gain": block.horizon_gain(L),
                "geometric_gain": 1.0 / (1.0 - s) if s < 1 else math.inf,
                "h_bound_horizon": cert["summary"]["h_abs_max_horizon"],
                "h_bound_geometric": cert["summary"]["h_abs_max_geometric"],
            }
        )
    return rows


def ablate_discretisation(config: dict, ckpt_dir: Path) -> dict:
    """Exact ZOH vs the Euler shortcut, over the certified delta range and the learned A."""
    set_seed(int(config.get("seed", 1337)))
    model = load_or_init(ckpt_dir / "best_mamba.pt", config, "bounded")
    block = model.layers[0]
    A = -torch.exp(block.A_log)

    grid = torch.linspace(block.dt_min, block.dt_max, 256).view(-1, 1, 1)
    v = grid * A.unsqueeze(0)
    zoh = grid * zoh_phi(v)                       # (exp(dt A) - 1)/A
    euler = grid.expand_as(zoh)                   # dt
    rel = ((euler - zoh).abs() / zoh.abs().clamp_min(1e-30)).flatten()

    return {
        "delta_range": [block.dt_min, block.dt_max],
        "A_range": [A.min().item(), A.max().item()],
        "max_abs_dtA": float((grid.max() * A.abs().max()).item()),
        "euler_vs_zoh_rel_error": {
            "median": float(rel.median()),
            "p99": float(rel.kthvalue(int(0.99 * rel.numel())).values),
            "max": float(rel.max()),
        },
        "quantity": "ZOH input gain (exp(dt A) - 1)/A against the Euler surrogate dt",
        "note": (
            "A is discretised exactly either way; what differs is the factor multiplying B. "
            "Euler is consistent with that exact Abar only when |delta*A| << 1. The error is "
            "a property of the delta range, so bounding delta bounds it as well."
        ),
    }


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    def fmt(v):
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            if v == math.inf:
                return "inf"
            if v != 0 and (abs(v) >= 1e4 or abs(v) < 1e-3):
                return f"{v:.3e}"
            return f"{v:.4f}"
        if isinstance(v, list):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        return str(v)

    head = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(fmt(r.get(key, "")) for key, _ in columns) + " |" for r in rows
    ]
    return "\n".join([head, rule, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the certification ablation.")
    parser.add_argument("--checkpoint-dir", default="models/checkpoints")
    parser.add_argument("--seq-len", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    seq_len = args.seq_len or int(
        config.get("verification", {}).get("seq_len", config["data"]["seq_length"])
    )
    ckpt_dir = Path(args.checkpoint_dir)

    print("1/5 delta parametrisation ...")
    delta_rows = ablate_delta(config, seq_len, ckpt_dir)
    print("2/5 convolution structure ...")
    conv_rows = ablate_conv(config, seq_len, ckpt_dir)
    print("3/5 relaxation ...")
    relax_rows = ablate_relaxation(config, seq_len, ckpt_dir)
    print("4/5 horizon ...")
    horizon_rows = ablate_horizon(config, ckpt_dir)
    print("5/5 discretisation ...")
    disc = ablate_discretisation(config, ckpt_dir)

    results = {
        "provenance": provenance(config.get("seed")),
        "seq_len": seq_len,
        "delta_parametrisation": delta_rows,
        "convolution": conv_rows,
        "relaxation": relax_rows,
        "horizon": horizon_rows,
        "discretisation": disc,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ablation.json").write_text(json.dumps(results, indent=2))

    md = [
        "# Certification ablation",
        "",
        f"Sequence length L = {seq_len}. Every number below is input-independent: it is a",
        "property of the module, established without reference to any dataset.",
        "",
        "## 1. Delta parametrisation (the main result)",
        "",
        markdown_table(
            delta_rows,
            [
                ("dt_parametrisation", "delta"),
                ("inf_delta", "inf delta"),
                ("sup_A_bar", "sup Abar"),
                ("geometric_gain", "1/(1-sup)"),
                ("horizon_gain", f"sum_{{k<{seq_len}}}"),
                ("h_bound_geometric", "|h| geometric"),
                ("h_bound_horizon", f"|h| at L={seq_len}"),
                ("contractive", "contractive"),
            ],
        ),
        "",
        "`softplus` is reference Mamba. Because softplus has infimum 0, delta can approach",
        "zero, Abar can approach 1, and no finite invariant set exists for the state. This",
        "is not conservatism in the abstraction: the supremum is genuinely attained in the",
        "limit. Bounding delta below by dt_min repairs it structurally.",
        "",
        "## 2. Convolution structure",
        "",
        markdown_table(
            conv_rows,
            [
                ("conv", "conv"),
                ("groups", "groups"),
                ("conv_params", "params"),
                ("max_l1_gain", "max L1 gain"),
                ("u_abs_max", "|u| bound"),
                ("B_abs_max", "|B| bound"),
                ("h_bound_horizon", "|h| bound"),
            ],
        ),
        "",
        "## 3. Relaxation of LayerNorm",
        "",
        markdown_table(
            relax_rows,
            [
                ("relaxation", "relaxation"),
                ("u_abs_max", "|u| bound"),
                ("B_abs_max", "|B| bound"),
                ("h_bound_horizon", "|h| bound"),
                ("output_range", "output range"),
            ],
        ),
        "",
        "The fused bound is exact for the composition: over {z : sum z = 0, ||z||_2 <= sqrt(d)}",
        "a linear functional attains ||v - mean(v)||_2 * sqrt(d), and that supremum is reached.",
        "",
        "## 4. Finite horizon vs geometric fixed point",
        "",
        markdown_table(
            horizon_rows,
            [
                ("seq_len", "L"),
                ("horizon_gain", "sum_{k<L} s^k"),
                ("geometric_gain", "1/(1-s)"),
                ("h_bound_horizon", "|h| at L"),
                ("h_bound_geometric", "|h| geometric"),
            ],
        ),
        "",
        "## 5. Discretisation",
        "",
        "```json",
        json.dumps(disc, indent=2),
        "```",
        "",
    ]
    (OUT_DIR / "ablation.md").write_text("\n".join(md))

    print("\n" + "\n".join(md[:40]))
    print(f"\nwrote {OUT_DIR / 'ablation.json'} and {OUT_DIR / 'ablation.md'}")


if __name__ == "__main__":
    main()
