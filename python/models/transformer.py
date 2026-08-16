import math
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
    def __init__(
        self,
        input_dim: int = 43,
        config=None,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        if config is not None:
            cfg = getattr(config, "model", config)
            if isinstance(cfg, dict):
                d_model = cfg.get("d_model", d_model)
                nhead = cfg.get("nhead", nhead)
                num_layers = cfg.get("num_layers", num_layers)
                dropout = cfg.get("dropout", dropout)
            else:
                d_model = getattr(cfg, "d_model", d_model)
                nhead = getattr(cfg, "nhead", nhead)
                num_layers = getattr(cfg, "num_layers", num_layers)
                dropout = getattr(cfg, "dropout", dropout)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.feature_projection(x)
        x = self.pos_encoder(x)
        x = self.input_dropout(x)
        x = self.transformer_encoder(x)
        return self.head(x[:, -1, :]).squeeze(-1)


if __name__ == "__main__":
    x = torch.randn(32, 100, 43)
    model = LOBTransformer(input_dim=43)
    out = model(x)
    print(f"Output shape: {out.shape}")