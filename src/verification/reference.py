"""NumPy forward pass driven only by an exported certificate JSON.

Nothing here imports torch or reads the repository. Given
models/bounds/mamba_certificate.json it reproduces the network, so a verifier can check its
own encoding of the dynamics against a known-good implementation. scripts/04_export_bounds.py
runs this against PyTorch and refuses to write the artefact if they disagree.

    from src.verification.reference import ReferenceModel
    model = ReferenceModel.from_json("models/bounds/mamba_certificate.json")
    y = model.forward(x)                  # x is (batch, L, input_dim), already normalised
    y, trace = model.forward(x, trace=True)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # avoids overflow warnings on large |x|


def silu(x):
    return x * sigmoid(x)


def softplus(x):
    return np.logaddexp(0.0, x)


def zoh_phi(u):
    """phi(u) = expm1(u)/u, continuously extended with phi(0) = 1."""
    small = np.abs(u) < 1e-4
    safe = np.where(small, 1.0, u)
    return np.where(small, 1.0 + u / 2.0 + u * u / 6.0, np.expm1(safe) / safe)


def layer_norm(x, weight, bias, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * weight + bias


def linear(x, weight, bias=None):
    y = x @ weight.T
    return y if bias is None else y + bias


def depthwise_causal_conv1d(u, weight, bias, d_conv):
    """u is (batch, L, channels); weight is (channels, 1, d_conv).

    Left-pads by d_conv - 1 so position t reads u[t - d_conv + 1 .. t]. The PyTorch side pads
    both ends and truncates to L, which is the same thing.

    Conv1d cross-correlates rather than convolves, so the kernel is not reversed:
    out[t] = sum_k w[k] * u[t + k - d_conv + 1].
    """
    batch, L, ch = u.shape
    w = weight.reshape(ch, d_conv)
    padded = np.concatenate([np.zeros((batch, d_conv - 1, ch), u.dtype), u], axis=1)
    out = np.zeros_like(u)
    for k in range(d_conv):
        out += padded[:, k : k + L, :] * w[:, k]
    return out + bias


class ReferenceModel:
    def __init__(self, artefact: dict):
        self.artefact = artefact
        self.arch = artefact["architecture"]
        p = artefact["parameters"]
        self.stem = {k: np.asarray(v, dtype=np.float64) for k, v in p["stem"].items()}
        self.head = {k: np.asarray(v, dtype=np.float64) for k, v in p["head"].items()}
        self.layers = []
        for layer in p["layers"]:
            entry = {}
            for k, v in layer.items():
                entry[k] = np.asarray(v, dtype=np.float64) if isinstance(v, list) else v
            self.layers.append(entry)

    @classmethod
    def from_json(cls, path) -> "ReferenceModel":
        return cls(json.loads(Path(path).read_text()))

    def input_box(self):
        box = self.artefact["input_box"]
        return np.asarray(box["lo"], np.float32), np.asarray(box["hi"], np.float32)

    def check_input_box(self, x) -> None:
        lo, hi = self.input_box()
        if (x < lo - 1e-6).any() or (x > hi + 1e-6).any():
            raise ValueError("input outside the certified box; the certificate does not apply")

    def _delta(self, pre, layer):
        if layer["dt_parametrisation"] == "bounded":
            lo, hi = layer["dt_min"], layer["dt_max"]
            return np.clip(lo * (hi / lo) ** sigmoid(pre), lo, hi)
        return softplus(pre)

    def _block(self, x, layer, trace):
        L = x.shape[1]
        d_state, d_inner = int(layer["d_state"]), int(layer["d_inner"])

        xn = layer_norm(x, layer["norm.weight"], layer["norm.bias"])
        xz = linear(xn, layer["in_proj.weight"], layer["in_proj.bias"])
        u, gate = xz[..., :d_inner], xz[..., d_inner:]

        u = silu(depthwise_causal_conv1d(u, layer["conv1d.weight"], layer["conv1d.bias"],
                                         int(layer["d_conv"])))

        proj = linear(u, layer["x_proj.weight"])
        dt_rank = int(layer["dt_rank"])
        dt_pre = proj[..., :dt_rank]
        B = proj[..., dt_rank : dt_rank + d_state]
        C = proj[..., dt_rank + d_state :]
        delta = self._delta(linear(dt_pre, layer["dt_proj.weight"], layer["dt_proj.bias"]), layer)

        A, D = layer["A"], layer["D"]
        batch = x.shape[0]
        h = np.zeros((batch, d_inner, d_state))
        y = np.zeros((batch, L, d_inner))
        rec = {"delta": delta, "B": B, "C": C, "u": u, "h": [], "A_bar": [], "B_bar": []}

        for t in range(L):
            dtA = delta[:, t][:, :, None] * A[None, :, :]
            A_bar = np.exp(dtA)
            B_bar = delta[:, t][:, :, None] * zoh_phi(dtA) * B[:, t][:, None, :]
            h = A_bar * h + B_bar * u[:, t][:, :, None]
            y[:, t] = np.einsum("bdn,bn->bd", h, C[:, t]) + D * u[:, t]
            if trace:
                rec["A_bar"].append(A_bar)
                rec["B_bar"].append(B_bar)
                rec["h"].append(h.copy())

        out = linear(y * silu(gate), layer["out_proj.weight"], layer["out_proj.bias"]) + x
        if trace:
            for k in ("A_bar", "B_bar", "h"):
                rec[k] = np.stack(rec[k], axis=1)
            rec["y"] = y
        return out, rec

    def forward(self, x, trace: bool = False):
        x = np.asarray(x, dtype=np.float64)
        h = layer_norm(silu(linear(x, self.stem["proj.0.weight"], self.stem["proj.0.bias"])),
                       self.stem["proj.2.weight"], self.stem["proj.2.bias"])

        traces = []
        for layer in self.layers:
            h, rec = self._block(h, layer, trace)
            if trace:
                traces.append(rec)

        h = layer_norm(h, self.head["final_norm.weight"], self.head["final_norm.bias"])
        last = h[:, -1, :]
        out = linear(silu(linear(last, self.head["head.0.weight"], self.head["head.0.bias"])),
                     self.head["head.3.weight"], self.head["head.3.bias"])[:, 0]
        return (out, traces) if trace else out


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "models/bounds/mamba_certificate.json"
    model = ReferenceModel.from_json(path)
    lo, hi = model.input_box()
    rng = np.random.default_rng(0)
    x = lo + rng.random((4, 100, model.arch["input_dim"])).astype(np.float32) * (hi - lo)
    y = model.forward(x)
    box = model.artefact["certificate"]["output_range"]
    print(f"output {y}")
    print(f"certified range [{box['lo']:.4f}, {box['hi']:.4f}]  "
          f"inside: {bool((y >= box['lo'] - 1e-6).all() and (y <= box['hi'] + 1e-6).all())}")
