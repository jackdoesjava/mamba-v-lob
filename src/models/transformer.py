"""Transformer baseline. Regresses the 100-tick forward log return, as the SSM does.

See docs/02-model.md for what the `causal` flag changes.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class LOBTransformer(nn.Module):
    def __init__(self, input_dim: int = 43, config: Optional[dict] = None):
        super().__init__()
        cfg = ((config or {}).get("model", {}) or {}).get("transformer", {}) or {}

        d_model = int(cfg.get("d_model", 64))
        nhead = int(cfg.get("nhead", 4))
        num_layers = int(cfg.get("num_layers", 2))
        dim_feedforward = int(cfg.get("dim_feedforward", 128))
        dropout = float(cfg.get("dropout", 0.1))
        self.causal = bool(cfg.get("causal", True))

        self.input_dim = input_dim
        self.d_model = d_model

        self.feature_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def architecture(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "causal": self.causal,
            "num_layers": len(self.transformer_encoder.layers),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.feature_projection(x)
        x = self.pos_encoder(x)
        x = self.input_dropout(x)

        # No leakage either way (the label sits past the window), but without the mask
        # this is a bidirectional encoder and shouldn't be called causal.
        mask = None
        if self.causal:
            mask = nn.Transformer.generate_square_subsequent_mask(
                x.size(1), device=x.device
            )
        x = self.transformer_encoder(x, mask=mask, is_causal=self.causal)
        return self.head(x[:, -1, :]).squeeze(-1)


if __name__ == "__main__":
    model = LOBTransformer(input_dim=43)
    print(model(torch.randn(8, 100, 43)).shape, model.architecture())
