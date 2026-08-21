"""Compare abstract domains on the same question: how far can the prediction move?

Three ways of answering, plus a search that lower-bounds the truth:

  attack        gradient ascent and box-vertex sampling. Whatever it reaches is reachable, so
                this is a lower bound on the true width and the yardstick for the rest.
  lipschitz     norm-based sensitivity propagation, src/verification/lipschitz.py.
  interval      box propagation through the sequence, src/verification/sensitivity.py.
  global        the unconditional certificate, src/verification/certify.py. Independent of
                eps and of the input, so it appears as a flat reference line.

The point of the script is to establish which domain is usable at which perturbation size,
and it is reported in docs/03-certificate.md. Two of the three are not usable at any size that
matters, which is the result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.dataset import LOBDataModule
from src.models.mamba import LOBMamba
from src.utils.config import load_config
from src.utils.repro import provenance, set_seed
from src.verification.certify import certify_model
from src.verification.intervals import Interval
from src.verification.lipschitz import local_sensitivity
from src.verification.sensitivity import certified_output

OUT_DIR = Path("docs/certification")


def attack_width(model: LOBMamba, x: torch.Tensor, eps: float, steps: int = 60) -> torch.Tensor:
    """Largest spread the prediction is actually shown to take over the eps ball."""
    lo = torch.full((x.shape[0],), float("inf"))
    hi = torch.full((x.shape[0],), -float("inf"))

    with torch.no_grad():
        for _ in range(steps):
            d = torch.where(torch.rand_like(x) > 0.5, eps, -eps)
            y = model(x + d)
            lo, hi = torch.minimum(lo, y), torch.maximum(hi, y)

    # a few gradient steps towards each extreme, projected back onto the ball
    for sign in (1.0, -1.0):
        d = torch.zeros_like(x, requires_grad=True)
        opt = torch.optim.Adam([d], lr=eps / 4)
        for _ in range(25):
            opt.zero_grad(set_to_none=True)
            (-sign * model(x + d).sum()).backward()
            opt.step()
            with torch.no_grad():
                d.clamp_(-eps, eps)
        with torch.no_grad():
            y = model(x + d)
            lo, hi = torch.minimum(lo, y), torch.maximum(hi, y)

    return hi - lo


@torch.no_grad()
def interval_width(model: LOBMamba, x: torch.Tensor, eps: float) -> torch.Tensor:
    out = certified_output(model, Interval(x - eps, x + eps))
    return (out.hi - out.lo).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare abstract domains on local sensitivity.")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_mamba.pt")
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--attack-steps", type=int, default=60)
    args = parser.parse_args()

    config = load_config()
    set_seed(int(config.get("seed", 1337)))

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", config)
    model = LOBMamba(ckpt["architecture"]["input_dim"], config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dm = LOBDataModule(config)
    x, _ = next(iter(dm.loader("test", shuffle=False, batch_size=max(8, args.windows))))
    x = x[: args.windows]

    cert = certify_model(model, seq_len=x.shape[1])
    global_width = cert["output_range"]["hi"] - cert["output_range"]["lo"]

    epsilons = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    rows = []
    print(f"{'eps':>8} {'attack':>12} {'lipschitz':>12} {'interval':>12} {'global':>10}")
    for eps in epsilons:
        atk = attack_width(model, x, eps, args.attack_steps).median().item()
        lip = local_sensitivity(model, x, eps)["width"].median().item()
        ivl = interval_width(model, x, eps).median().item()
        rows.append(
            {
                "eps": eps,
                "attack": atk,
                "lipschitz": lip,
                "interval": ivl,
                "global": global_width,
                "lipschitz_over_attack": lip / atk if atk > 0 else float("inf"),
                "interval_over_attack": ivl / atk if atk > 0 else float("inf"),
            }
        )
        print(f"{eps:>8.0e} {atk:>12.4e} {lip:>12.4e} {ivl:>12.4e} {global_width:>10.4f}")

    usable = [r for r in rows if r["lipschitz_over_attack"] < 10.0 or r["interval_over_attack"] < 10.0]
    print(f"\nperturbation sizes where either local domain stays within 10x of the attack: "
          f"{len(usable)} of {len(rows)}")
    print(f"unconditional certified width, independent of eps and of the input: {global_width:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "domains.json").write_text(
        json.dumps(
            {
                "provenance": provenance(config.get("seed")),
                "checkpoint": args.checkpoint,
                "windows": int(x.shape[0]),
                "seq_len": int(x.shape[1]),
                "global_certified_width": global_width,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"wrote {OUT_DIR / 'domains.json'}")


if __name__ == "__main__":
    main()
