"""Turn a trained LOBMamba into the plain-data description that reference.py can read.

Kept out of scripts/ so the round trip can be tested without invoking the CLI, and out of
reference.py so that module stays free of torch.
"""

from __future__ import annotations

import torch

from src.models.mamba import LOBMamba


def export_parameters(model: LOBMamba) -> dict:
    """Weights as nested lists, with A materialised rather than left as A_log.

    float32 round-trips exactly through Python floats and json, so nothing is lost.
    """
    params: dict = {"stem": {}, "layers": [], "head": {}}
    for name, t in model.proj.state_dict().items():
        params["stem"][f"proj.{name}"] = t.tolist()
    for name, t in model.final_norm.state_dict().items():
        params["head"][f"final_norm.{name}"] = t.tolist()
    for name, t in model.head.state_dict().items():
        params["head"][f"head.{name}"] = t.tolist()

    for block in model.layers:
        entry = {
            "A": (-torch.exp(block.A_log)).detach().tolist(),
            "A_log": block.A_log.detach().tolist(),
            "D": block.D.detach().tolist(),
            "dt_min": block.dt_min,
            "dt_max": block.dt_max,
            "dt_parametrisation": block.dt_parametrisation,
            "dt_rank": block.dt_rank,
            "d_state": block.d_state,
            "d_inner": block.d_inner,
            "d_conv": block.d_conv,
        }
        for mod_name in ("norm", "in_proj", "conv1d", "x_proj", "dt_proj", "out_proj"):
            for pname, t in getattr(block, mod_name).state_dict().items():
                entry[f"{mod_name}.{pname}"] = t.tolist()
        params["layers"].append(entry)
    return params


def minimal_artefact(model: LOBMamba) -> dict:
    """The smallest dict ReferenceModel accepts. The full export adds provenance and bounds."""
    return {"architecture": model.architecture(), "parameters": export_parameters(model)}
