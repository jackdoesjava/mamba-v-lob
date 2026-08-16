import torch
import json
from pathlib import Path

# Paths
CKPT_PATH = Path("models/checkpoints/best_mamba.pt")
EXPORT_PATH = Path("models/bounds/mamba_params.json")

def export_parameters():
    if not CKPT_PATH.exists():
        print(f"Error: Checkpoint not found at {CKPT_PATH}")
        return

    print(f"Loading checkpoint from {CKPT_PATH}...")
    # weights_only=True is safer and avoids pickle warnings
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    
    # Handle if the checkpoint is a nested dict or just the raw state_dict
    state_dict = ckpt.get("model_state_dict", ckpt)

    params_dump = {}
    print("Extracting state matrices...")

    for name, tensor in state_dict.items():
        # The Rust engine needs standard types, so convert tensors -> numpy -> nested lists
        params_dump[name] = tensor.detach().numpy().tolist()

    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(EXPORT_PATH, "w") as f:
        json.dump(params_dump, f)

    print(f"Successfully exported {len(params_dump)} parameter tensors to {EXPORT_PATH}")

if __name__ == "__main__":
    export_parameters()