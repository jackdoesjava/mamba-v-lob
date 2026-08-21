"""Selective SSM block and the LOB regression model built on it.

Where this departs from Gu & Dao (2023), and why, is in docs/02-model.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# min_x x*sigmoid(x) and its argmin; needed to bound SiLU over an interval.
SILU_MIN = -0.27846454276107395
SILU_ARGMIN = -1.2784645427610738


def zoh_phi(u: torch.Tensor) -> torch.Tensor:
    """phi(u) = expm1(u)/u, continuously extended with phi(0) = 1.

    Taylor branch near 0 so the 0/0 limit stays finite and differentiable.
    """
    small = u.abs() < 1e-4
    u_safe = torch.where(small, torch.ones_like(u), u)
    taylor = 1.0 + u / 2.0 + u * u / 6.0
    return torch.where(small, taylor, torch.expm1(u_safe) / u_safe)


@dataclass
class SSMTrace:
    """State-space quantities realised on one forward pass.

    Everything but A is an activation, so a trace only describes the inputs that made it.
    """

    A: torch.Tensor        # (d_inner, d_state)          continuous-time, time-invariant
    delta: torch.Tensor    # (batch, L, d_inner)
    B: torch.Tensor        # (batch, L, d_state)
    C: torch.Tensor        # (batch, L, d_state)
    A_bar: torch.Tensor    # (batch, L, d_inner, d_state) discrete state matrix
    B_bar: torch.Tensor    # (batch, L, d_inner, d_state) discrete input matrix (exact ZOH)
    h: torch.Tensor        # (batch, L, d_inner, d_state) hidden state
    u: torch.Tensor        # (batch, L, d_inner)          SSM branch input
    y: torch.Tensor        # (batch, L, d_inner)          SSM output, pre-gate


class SelectiveSSMBlock(nn.Module):
    """One Mamba block. The scan is unrolled in Python, not fused, so every intermediate
    (delta, B, C, Abar, Bbar, h) stays addressable for the trace.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_parametrisation: str = "bounded",
        conv_bias: bool = True,
    ):
        super().__init__()
        if dt_parametrisation not in ("bounded", "softplus"):
            raise ValueError(f"unknown dt_parametrisation: {dt_parametrisation!r}")
        if not 0.0 < dt_min < dt_max:
            raise ValueError(f"require 0 < dt_min < dt_max, got {dt_min}, {dt_max}")

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_model / 16)
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_parametrisation = dt_parametrisation

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=True)

        # Depthwise + causal: pad d_conv-1 on both sides, then keep the first L outputs, so
        # position j reads inputs [j-d_conv+1, j] only.
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4D-Real initialisation. A_log stores log(n+1); the sign is applied in forward,
        # where A = -exp(A_log) gives A[d, n] = -(n+1), shared across channels.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=True)

        self._init_dt()

    def _init_dt(self) -> None:
        """Initialise dt_proj so delta starts log-uniform across [dt_min, dt_max]."""
        with torch.no_grad():
            # Keep the projection weights small so delta is bias-dominated at init.
            dt_init_std = self.dt_rank**-0.5
            self.dt_proj.weight.uniform_(-dt_init_std, dt_init_std)

            if self.dt_parametrisation == "bounded":
                # log-uniform delta means sigmoid(z) ~ U(0,1); stay off 0 and 1 so the
                # logit is finite and the sigmoid starts away from its flat tails.
                p = torch.empty(self.d_inner).uniform_(0.02, 0.98)
                self.dt_proj.bias.copy_(torch.log(p) - torch.log1p(-p))  # logit(p)
            else:
                dt = torch.exp(
                    torch.empty(self.d_inner).uniform_(
                        math.log(self.dt_min), math.log(self.dt_max)
                    )
                )
                # inverse softplus, computed stably: z = dt + log(-expm1(-dt))
                self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def delta_from_pre(self, pre: torch.Tensor) -> torch.Tensor:
        """Pre-activation to timescale. Both branches are monotone increasing in `pre`, so
        the verifier can bound delta from the endpoints of an interval.
        """
        if self.dt_parametrisation == "bounded":
            ratio = self.dt_max / self.dt_min
            delta = self.dt_min * ratio ** torch.sigmoid(pre)
            # In the reals this already sits inside (dt_min, dt_max). In float32 sigmoid
            # returns exactly 1.0 above pre ~ 16.64 and exactly 0.0 below pre ~ -88.72, and
            # dt_min*ratio need not round back to dt_max, so clamp to make [dt_min, dt_max]
            # hold bitwise. The upper tail is reachable in training, so this is not academic.
            return delta.clamp(min=self.dt_min, max=self.dt_max)
        return F.softplus(pre)

    def ssm_branch_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        L = x.shape[1]
        xn = self.norm(x)
        xz = self.in_proj(xn)
        u, gate = xz.chunk(2, dim=-1)

        u = u.transpose(1, 2)
        u = self.conv1d(u)[:, :, :L]      # drop the right-hand padding: this is the causality
        u = u.transpose(1, 2)
        u = F.silu(u)
        return u, gate

    def ssm_parameters(
        self, u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proj = self.x_proj(u)
        dt_pre, B, C = torch.split(
            proj, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = self.delta_from_pre(self.dt_proj(dt_pre))
        return delta, B, C

    def forward(
        self, x: torch.Tensor, trace: bool = False, detach_trace: bool = True
    ) -> torch.Tensor | tuple[torch.Tensor, SSMTrace]:
        """`detach_trace=False` keeps the trace attached to the graph, which only
        scripts/07_attack_bounds.py needs; detaching avoids holding an L-step scan graph.
        """
        batch, L, _ = x.shape
        residual = x

        u, gate = self.ssm_branch_input(x)
        delta, B, C = self.ssm_parameters(u)
        A = -torch.exp(self.A_log)                       # (d_inner, d_state), strictly < 0

        h = x.new_zeros(batch, self.d_inner, self.d_state)
        ys, tr_Abar, tr_Bbar, tr_h = [], [], [], []

        for t in range(L):
            ut = u[:, t]                                  # (batch, d_inner)
            dt = delta[:, t]                              # (batch, d_inner)
            Bt = B[:, t]                                  # (batch, d_state)
            Ct = C[:, t]                                  # (batch, d_state)

            dtA = dt.unsqueeze(-1) * A.unsqueeze(0)       # (batch, d_inner, d_state)
            A_bar = torch.exp(dtA)
            # Exact ZOH: Bbar = A^-1 (exp(dt A) - I) B = dt * phi(dt A) * B. Kept as a
            # matrix, with ut applied separately, so the export can emit the two apart.
            B_bar = dt.unsqueeze(-1) * zoh_phi(dtA) * Bt.unsqueeze(1)

            h = A_bar * h + B_bar * ut.unsqueeze(-1)
            yt = torch.einsum("bdn,bn->bd", h, Ct) + self.D * ut
            ys.append(yt)

            if trace:
                tr_Abar.append(A_bar)
                tr_Bbar.append(B_bar)
                tr_h.append(h)

        y = torch.stack(ys, dim=1)
        out = self.out_proj(y * F.silu(gate)) + residual

        if not trace:
            return out
        keep = (lambda t: t.detach()) if detach_trace else (lambda t: t)
        return out, SSMTrace(
            A=keep(A),
            delta=keep(delta),
            B=keep(B),
            C=keep(C),
            A_bar=keep(torch.stack(tr_Abar, dim=1)),
            B_bar=keep(torch.stack(tr_Bbar, dim=1)),
            h=keep(torch.stack(tr_h, dim=1)),
            u=keep(u),
            y=keep(y),
        )

    def contraction_certificate(self) -> dict[str, float]:
        """sup Abar over all inputs and the geometric gain it implies. Data-independent."""
        A_abs_min = torch.exp(self.A_log).min().item()
        if self.dt_parametrisation == "bounded":
            sup_A_bar = math.exp(-self.dt_min * A_abs_min)
        else:
            sup_A_bar = 1.0  # inf delta = 0
        gain = math.inf if sup_A_bar >= 1.0 else 1.0 / (1.0 - sup_A_bar)
        return {
            "A_abs_min": A_abs_min,
            "A_abs_max": torch.exp(self.A_log).max().item(),
            "dt_min": self.dt_min if self.dt_parametrisation == "bounded" else 0.0,
            "dt_max": self.dt_max if self.dt_parametrisation == "bounded" else math.inf,
            "sup_A_bar": sup_A_bar,
            "geometric_gain": gain,
            "contractive": sup_A_bar < 1.0,
        }

    def horizon_gain(self, seq_len: int) -> float:
        """Finite-horizon gain sum_{k<L} sup(Abar)^k; stays finite even when sup = 1."""
        s = self.contraction_certificate()["sup_A_bar"]
        if s >= 1.0:
            return float(seq_len)
        return (1.0 - s**seq_len) / (1.0 - s)


class LOBMamba(nn.Module):
    """Stacked selective SSM blocks with a scalar regression head."""

    def __init__(self, input_dim: int = 43, config: Optional[dict] = None):
        super().__init__()
        cfg = ((config or {}).get("model", {}) or {}).get("mamba", {}) or {}

        d_model = int(cfg.get("d_model", 64))
        d_state = int(cfg.get("d_state", 16))
        d_conv = int(cfg.get("d_conv", 4))
        expand = int(cfg.get("expand", 2))
        num_layers = int(cfg.get("num_layers", 2))
        dt_rank = cfg.get("dt_rank", None)
        dt_min = float(cfg.get("dt_min", 1e-3))
        dt_max = float(cfg.get("dt_max", 1e-1))
        dt_parametrisation = str(cfg.get("dt_parametrisation", "bounded"))
        head_dropout = float(cfg.get("head_dropout", 0.2))

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers

        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )
        self.layers = nn.ModuleList(
            [
                SelectiveSSMBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=int(dt_rank) if dt_rank is not None else None,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    dt_parametrisation=dt_parametrisation,
                )
                for _ in range(num_layers)
            ]
        )
        # Bounds the head's input independently of the residual stream; without it the
        # output range is only certifiable under an assumption on the input features.
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_model // 2, 1),
        )

    def architecture(self) -> dict:
        """Serialisable architecture description; goes into the export manifest."""
        b = self.layers[0]
        return {
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "d_state": b.d_state,
            "d_conv": b.d_conv,
            "expand": b.expand,
            "d_inner": b.d_inner,
            "dt_rank": b.dt_rank,
            "dt_min": b.dt_min,
            "dt_max": b.dt_max,
            "dt_parametrisation": b.dt_parametrisation,
            "num_layers": self.num_layers,
        }

    def forward(
        self, x: torch.Tensor, trace: bool = False, detach_trace: bool = True
    ) -> torch.Tensor | tuple[torch.Tensor, list[SSMTrace]]:
        x = x.to(torch.float32)
        x = self.proj(x)

        traces: list[SSMTrace] = []
        for layer in self.layers:
            if trace:
                x, tr = layer(x, trace=True, detach_trace=detach_trace)
                traces.append(tr)
            else:
                x = layer(x)

        x = self.final_norm(x)
        out = self.head(x[:, -1, :]).squeeze(-1)
        return (out, traces) if trace else out


if __name__ == "__main__":
    torch.manual_seed(0)
    model = LOBMamba(input_dim=43)
    y, traces = model(torch.randn(4, 100, 43), trace=True)
    print(f"output {tuple(y.shape)}  |  arch {model.architecture()}")
    for i, (layer, tr) in enumerate(zip(model.layers, traces)):
        cert = layer.contraction_certificate()
        print(
            f"layer {i}: delta in [{tr.delta.min():.4f}, {tr.delta.max():.4f}]  "
            f"sup Abar={cert['sup_A_bar']:.6f}  gain={cert['geometric_gain']:.2f}  "
            f"L=100 gain={layer.horizon_gain(100):.2f}"
        )
