import torch
import torch.nn as nn
import torch.nn.functional as F

class CPUSafeMambaBlock(nn.Module):
    """
    Explicit, sequential State Space Model block. 
    Bypasses PyTorch's pscan deadlock on CPUs and explicitly exposes A, B, C matrices 
    for downstream formal verification exports.
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Mamba projections
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3)
        
        # State space discrete parameter projections
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1) # B, C, delta
        self.dt_proj = nn.Linear(1, d_model)
        
        # Continuous A matrix (fixed initialization per Mamba paper)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_batch, L_seq, _ = x.shape
        residual = x
        
        x = self.norm(x)
        
        # 1. Project and split into main branch and residual gate
        x_proj = self.in_proj(x)
        x_mamba, x_gate = x_proj.chunk(2, dim=-1)
        
        # 2. Local spatial interactions
        x_mamba = x_mamba.transpose(1, 2)
        x_mamba = F.silu(self.conv1d(x_mamba)[:, :, :L_seq])
        x_mamba = x_mamba.transpose(1, 2)
        
        # 3. Derive SSM Parameters (Delta, B, C)
        ssm_params = self.x_proj(x_mamba)
        delta, B_mat, C_mat = torch.split(ssm_params, [1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        
        A = -torch.exp(self.A_log)
        
        # 4. Sequential Scan (CPU-Safe & Mathematically identical)
        h = torch.zeros(B_batch, self.d_model, self.d_state, device=x.device)
        y_out = []
        
        for t in range(L_seq):
            xt = x_mamba[:, t, :]
            dt = delta[:, t, :]
            bt = B_mat[:, t, :]
            ct = C_mat[:, t, :]
            
            # Zero-Order Hold Discretization
            bar_A = torch.exp(A.unsqueeze(0) * dt.unsqueeze(2)) 
            bar_B = (dt * xt).unsqueeze(2) * bt.unsqueeze(1)
            
            # State Update: h(t) = A*h(t-1) + B*x(t)
            h = bar_A * h + bar_B
            
            # Output: y(t) = C*h(t) + D*x(t)
            yt = torch.sum(h * ct.unsqueeze(1), dim=-1) + self.D * xt
            y_out.append(yt)
            
        y = torch.stack(y_out, dim=1)
        
        # 5. Gating and output projection
        y = y * F.silu(x_gate)
        return self.out_proj(y) + residual


class LOBMamba(nn.Module):
    def __init__(self, input_dim: int = 43, config: dict = None):
        super().__init__()
        
        d_model = 64
        num_layers = 2
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model)
        )
        
        self.layers = nn.ModuleList([
            CPUSafeMambaBlock(d_model=d_model) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.proj(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_norm(x)
        last_step = x[:, -1, :]
        
        return self.head(last_step).squeeze(-1)

if __name__ == "__main__":
    x = torch.randn(32, 100, 43)
    model = LOBMamba(input_dim=43)
    out = model(x)
    print(f"Output shape: {out.shape}")