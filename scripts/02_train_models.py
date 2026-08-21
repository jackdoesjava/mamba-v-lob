"""Train one architecture on purged, chronologically split LOB windows.

Selection criterion, step budget and the fp32 scan requirement: see README.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from src.dataset import LOBDataModule
from src.models.lstm import LSTMBaseline
from src.models.mamba import LOBMamba
from src.models.transformer import LOBTransformer
from src.utils.config import load_config
from src.utils.repro import provenance, set_seed

CHECKPOINT_DIR = Path("models/checkpoints")


def build_model(name: str, input_dim: int, config: dict) -> nn.Module:
    if name == "lstm":
        return LSTMBaseline(input_dim, config)
    if name == "mamba":
        return LOBMamba(input_dim, config)
    if name == "transformer":
        return LOBTransformer(input_dim, config)
    raise ValueError(f"unknown architecture: {name}")


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device, max_batches: int) -> dict:
    model.eval()
    losses, preds, targets = [], [], []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        pred = model(x)
        losses.append(criterion(pred, y).item())
        preds.append(pred.float().cpu())
        targets.append(y.float().cpu())

    p = torch.cat(preds).numpy()
    t = torch.cat(targets).numpy()
    return {
        "loss": float(np.mean(losses)),
        "rank_ic": float(stats.spearmanr(p, t).statistic),
        "pred_std": float(p.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one LOB sequence model.")
    parser.add_argument(
        "--model", required=True, choices=["lstm", "mamba", "transformer"]
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None, help="checkpoint name suffix")
    parser.add_argument(
        "--dt-parametrisation",
        choices=["bounded", "softplus"],
        default=None,
        help="override config; used by the certification ablation",
    )
    args = parser.parse_args()

    config = load_config()
    seed = int(config.get("seed", 1337))
    set_seed(seed)

    if args.dt_parametrisation is not None:
        config["model"]["mamba"]["dt_parametrisation"] = args.dt_parametrisation

    train_cfg = config["training"]
    max_steps = args.max_steps or int(train_cfg["max_steps"])
    eval_every = int(train_cfg["eval_every"])
    eval_batches = int(train_cfg["eval_batches"])
    patience = int(train_cfg["patience"])
    device = resolve_device(str(train_cfg.get("device", "auto")))
    # No GradScaler is built, so enabling training.amp would train against unscaled fp16
    # gradients. It is off by default because fp16 also flushes exp(delta*A) to zero.
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"

    dm = LOBDataModule(config)
    print(json.dumps(dm.split_report(), indent=2))

    generator = torch.Generator().manual_seed(seed)
    train_loader = dm.loader("train", generator=generator)
    val_loader = dm.loader("val")

    model = build_model(args.model, dm.input_dim, config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

    tag = args.tag or args.model
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"best_{tag}.pt"

    print(
        f"\n{args.model} | {n_params:,} params | device={device} | amp={use_amp} | "
        f"seed={seed} | budget={max_steps} steps of batch {dm.batch_size}"
    )
    if args.model == "mamba":
        print(f"  mamba: {model.architecture()}")

    # selection is on val rank IC, not val loss: loss alone prefers the zero predictor
    best_score = -float("inf")
    best_val_loss = float("inf")
    since_improve = 0
    history: list[dict] = []
    step = 0
    t0 = time.perf_counter()
    stop = False

    while step < max_steps and not stop:
        for x, y in train_loader:
            if step >= max_steps:
                break
            model.train()
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(train_cfg["grad_clip"])
            )
            optimizer.step()
            scheduler.step()
            step += 1

            if step % eval_every == 0 or step == max_steps:
                val = evaluate(model, val_loader, criterion, device, eval_batches)
                elapsed = time.perf_counter() - t0
                record = {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "val_loss": val["loss"],
                    "val_rank_ic": val["rank_ic"],
                    "val_pred_std": val["pred_std"],
                    "elapsed_s": round(elapsed, 1),
                }
                history.append(record)
                print(
                    f"step {step:5d}/{max_steps} | train {loss.item():.5f} | "
                    f"val {val['loss']:.5f} | val rankIC {val['rank_ic']:+.4f} | "
                    f"{elapsed / 60:.1f} min"
                )

                if val["rank_ic"] > best_score + 1e-5:
                    best_score = val["rank_ic"]
                    best_val_loss = val["loss"]
                    since_improve = 0
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "model_name": args.model,
                            "architecture": (
                                model.architecture()
                                if hasattr(model, "architecture")
                                else {"input_dim": dm.input_dim}
                            ),
                            "config": config,
                            "normalisation": dm.normalisation.to_dict(),
                            "split_report": dm.split_report(),
                            "provenance": provenance(seed),
                            "training": {
                                "step": step,
                                "max_steps": max_steps,
                                "selection_metric": "val_rank_ic",
                                "best_val_rank_ic": best_score,
                                "val_loss_at_best": best_val_loss,
                                "n_params": n_params,
                                "history": history,
                            },
                        },
                        ckpt_path,
                    )
                    print(f"  -> saved {ckpt_path} (val rank IC {best_score:+.4f})")
                else:
                    since_improve += 1
                    if since_improve >= patience:
                        print(
                            f"early stop: val rank IC has not improved in {patience} evaluations"
                        )
                        stop = True
                        break

    print(
        f"\ndone: {step} steps in {(time.perf_counter() - t0) / 60:.1f} min | "
        f"best val rank IC {best_score:+.6f} (val loss there {best_val_loss:.6f})"
    )


if __name__ == "__main__":
    main()
