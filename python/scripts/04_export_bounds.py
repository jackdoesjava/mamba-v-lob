import torch
import json
from pathlib import Path

# Paths - adjust these if your checkpoints are inside the python/ folder
CKPT_PATH = Path("models/checkpoints/best_mamba.pt")
EXPORT_PATH = Path("models/bounds/mamba_params.json")

def export_parameters():
    if not CKPT_PATH.exists():
        print(f"Error: Checkpoint not found at {CKPT_PATH}")
        return

    print(f"Loading checkpoint from {CKPT_PATH}...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    
    state_dict = ckpt.get("model_state_dict", ckpt)

    params_dump = {}
    print("Extracting state matrices...")

    for name, tensor in state_dict.items():
        # Convert tensors to nested lists for the verifier
        params_dump[name] = tensor.detach().numpy().tolist()

    # Force create the directory wherever EXPORT_PATH is pointing
    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(EXPORT_PATH, "w") as f:
        json.dump(params_dump, f)

    print(f"Successfully exported {len(params_dump)} parameter tensors to {EXPORT_PATH}")

if __name__ == "__main__":
    export_parameters()