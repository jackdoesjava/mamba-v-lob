"""Data-free certificate for the released Mamba language model state-spaces/mamba-130m-hf.

The block this repository trains (src/models/mamba.py) uses LayerNorm and the exact zero-order
hold. The released model differs in two places, and this module certifies it as released.

Discretisation. transformers 5.15.1, src/transformers/models/mamba/modeling_mamba.py, slow
(pure PyTorch) path, which is the one that runs without the mamba_ssm and causal_conv1d
packages and without the kernels hub:

    mamba_selective_scan, lines 204-207   dt = softplus(dt + dt_proj.bias), no clamp
                          line 214        Abar = exp(A dt), A = -exp(A_log)       (line 374)
                          lines 215-216   Bbar u = dt * B * u
                          line 263        h_t = Abar_t h_{t-1} + dt_t B_t u_t,  h_{-1} = 0 (252)
                          line 266        y_t = sum_n C_{t,n} h_{t,:,n}
                          lines 270-274   y + D u, then y * SiLU(gate)

so the transition is the exact one and the input term is dt B u: the exponential-Euler step,
not the zero-order hold, whose input term would be (1 - Abar) B / |A|. Per cell, with
lam = |A| and Psi(x) = x / (1 - exp(-x)), dt = (1 - Abar) Psi(lam dt) / lam, so

    h_t = Abar_t h_{t-1} + (1 - Abar_t) c_t,     c_t = Psi(lam dt_t) B_t u_t / lam,

a convex combination with the equilibrium enlarged by Psi (paper, Corollary cor:euler).

Wiring. MambaBlock.forward, lines 521-527: residual = x; x_n = RMSNorm(x) (lines 494-499,
eps = 1e-5, no mean subtraction and no bias); out = residual + mixer(x_n). MambaMixer.forward,
lines 366-481: in_proj without bias (319, use_bias = False) -> split into x and the gate z
(396) -> depthwise causal conv1d of kernel 4 with bias, padding 3 and truncation to the
sequence length, then SiLU (causal_conv1d_fn, lines 82-101, called at 419-425) -> x_proj
without bias, split into dt (rank 48), B and C (432-436) -> dt_proj.weight @ dt (438) with
dt_proj.bias added inside the scan (439, 204-205) -> softplus -> scan -> out_proj without bias
(481). The three hub-kernel decorators fall back to the functions shown when the kernels,
mamba_ssm and causal_conv1d packages are absent (integrations/hub_kernels.py, 764-799), and the
associative-scan branch (224-246) runs only under torch.compile, so the recurrent loop at
249-268 is what executes.

Precision. The slow path casts to float32 even on a float64 model: RMSNorm computes in float32
(496-498), A = -exp(A_log.float()) (374), the dt bias, B, u in the product dt B u, and D are
rounded to float32 (439, 215-216, 465). `mixer_forward(..., faithful=True)` reproduces those
casts so the two agree to float64 round-off; the default runs everything in float64 and is the
map the certificate is about.

Premise. RMSNorm(x)_i = w_i r_i with r = x / sqrt(mean(x^2) + eps), and
||r||_2^2 = d mean(x^2) / (mean(x^2) + eps) < d. So r lies in the open ball of radius sqrt(d),
and |r_i| < sqrt(d), the coordinate form (MXNorm, arXiv:2603.13180, Lemma 1). Without the
mean subtraction there is no zero-sum constraint, so the fused bound on a row k of in_proj is
sup <W_k * w, r> = ||W_k * w||_2 sqrt(d), Cauchy-Schwarz over the ball, with no centring. Both
are suprema that no input attains. Interval propagation through the rest reuses intervals.py,
with the causal convolution's zero-padding hull.

Local certificate. `local_mixer_sensitivity` runs the norm-propagation domain of lipschitz.py
through the mixer around a realised normalised input, for any perturbation within eps in
sup-norm. Its recurrence step is the additive one of lipschitz._block with the Euler input gain dt
(`euler_additive_step`), or Corollary cor:local with the Euler equilibrium c = Psi(lam dt) B u / lam
(`euler_log_delta_step`), as the running maximum or as the recursion of the corollary's proof,
or in its endpoint form (`local_step_euler_endpoint`), the reported rule, which maximises the
exact one-step error identity over the transitions the perturbed run can take.

Everything here holds over the reals. The certificate arithmetic runs in float64.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.repro import REPO_ROOT
from src.verification import intervals as iv
from src.verification.incremental import (
    REPORTED_RULE,
    RULES,
    BlockIncrement,
    euler_gain,
    local_step,
    local_step_recursive,
    relative_factor,
    softplus_log_slope,
    timescale_offsets,
    warp_term,
)
from src.verification.intervals import Interval
from src.verification.invariant import equilibrium_box
from src.verification.lipschitz import SILU_DERIV_MAX, Sens, _silu_deriv_sup, add, linear, mul, silu

DTYPE = torch.float64

# Where the released checkpoint was downloaded to; MAMBA_130M_DIR overrides it. Nothing is
# ever fetched: every load passes local_files_only=True.
DEFAULT_MODEL_DIR = Path.home() / "hf_models" / "mamba-130m-hf"
MODEL_ID = "state-spaces/mamba-130m-hf"
SAMPLE_TEXT = REPO_ROOT / "data/text/public_domain_sample.txt"
MAX_TOKENS = 1024


def model_dir() -> Path:
    return Path(os.environ.get("MAMBA_130M_DIR", str(DEFAULT_MODEL_DIR)))


def weights_sha256(path: Path | None = None) -> str:
    """SHA-256 over the checkpoint's weight files, so a run records which weights it used."""
    p = model_dir() if path is None else Path(path)
    digest = hashlib.sha256()
    for f in sorted(p.glob("*.safetensors")) or sorted(p.glob("*.bin")):
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def model_available(path: Path | None = None) -> bool:
    p = model_dir() if path is None else Path(path)
    return (p / "config.json").exists() and (p / "model.safetensors").exists()


# --- loading ----------------------------------------------------------------------------------


def load_released(path: Path | None = None, dtype: torch.dtype = DTYPE):
    """MambaForCausalLM from the local directory, in eval mode, converted to `dtype`.

    The checkpoint is float32; dtype=float64 gives the copy the analysis runs on.
    """
    from transformers import MambaForCausalLM

    p = model_dir() if path is None else Path(path)
    model = MambaForCausalLM.from_pretrained(str(p), local_files_only=True, dtype=torch.float32)
    return model.to(dtype).eval()


def load_tokenizer(path: Path | None = None):
    from transformers import AutoTokenizer

    p = model_dir() if path is None else Path(path)
    return AutoTokenizer.from_pretrained(str(p), local_files_only=True)


def read_sample_text(path: Path = SAMPLE_TEXT) -> str:
    """The sample text without its header lines, which start with '#'."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return "\n".join(l for l in lines if not l.startswith("#")).strip() + "\n"


def tokenise(tokenizer, text: str, max_tokens: int = MAX_TOKENS) -> torch.Tensor:
    """Token ids (1, L) with L <= max_tokens. The Mamba tokenizer adds no BOS token."""
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    return ids[:, :max_tokens]


# --- one layer, as float64 tensors --------------------------------------------------------------


def _f64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(DTYPE)


@dataclass
class ReleasedLayer:
    """The weights of one MambaBlock (RMSNorm and mixer) in float64.

    W_in (2 d_inner, d_model), conv (the depthwise nn.Conv1d, weight (d_inner, 1, k)),
    W_x (dt_rank + 2 d_state, d_inner), W_dt (d_inner, dt_rank), b_dt (d_inner,),
    A_log (d_inner, d_state), D (d_inner,), W_out (d_model, d_inner).
    """

    index: int
    eps: float
    norm_weight: torch.Tensor
    W_in: torch.Tensor
    conv: nn.Conv1d
    W_x: torch.Tensor
    W_dt: torch.Tensor
    b_dt: torch.Tensor
    A_log: torch.Tensor
    D: torch.Tensor
    W_out: torch.Tensor
    dt_rank: int
    d_state: int
    residual_in_fp32: bool = False

    @staticmethod
    def from_block(block, index: int) -> "ReleasedLayer":
        m = block.mixer
        if m.in_proj.bias is not None or m.x_proj.bias is not None or m.out_proj.bias is not None:
            raise ValueError("expected in_proj, x_proj and out_proj without bias")
        if m.conv1d.groups != m.conv1d.in_channels:
            raise ValueError("expected a depthwise convolution")
        conv = nn.Conv1d(m.conv1d.in_channels, m.conv1d.out_channels, m.conv1d.kernel_size[0],
                         groups=m.conv1d.groups, bias=m.conv1d.bias is not None,
                         padding=m.conv1d.padding[0]).to(DTYPE)
        with torch.no_grad():
            conv.weight.copy_(m.conv1d.weight)
            if conv.bias is not None:
                conv.bias.copy_(m.conv1d.bias)
        conv.requires_grad_(False)
        return ReleasedLayer(
            index=index,
            eps=float(block.norm.variance_epsilon),
            norm_weight=_f64(block.norm.weight),
            W_in=_f64(m.in_proj.weight),
            conv=conv,
            W_x=_f64(m.x_proj.weight),
            W_dt=_f64(m.dt_proj.weight),
            b_dt=_f64(m.dt_proj.bias),
            A_log=_f64(m.A_log),
            D=_f64(m.D),
            W_out=_f64(m.out_proj.weight),
            dt_rank=int(m.time_step_rank),
            d_state=int(m.ssm_state_size),
            residual_in_fp32=bool(block.residual_in_fp32),
        )

    @property
    def d_model(self) -> int:
        return self.W_in.shape[1]

    @property
    def d_inner(self) -> int:
        return self.W_in.shape[0] // 2

    @property
    def d_conv(self) -> int:
        return self.conv.kernel_size[0]

    @property
    def lam(self) -> torch.Tensor:
        """|A| = exp(A_log), (d_inner, d_state)."""
        return torch.exp(self.A_log)

    def split_x(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r, n = self.dt_rank, self.d_state
        return self.W_x[:r], self.W_x[r : r + n], self.W_x[r + n :]


def released_layers(model) -> list[ReleasedLayer]:
    return [ReleasedLayer.from_block(b, i) for i, b in enumerate(model.backbone.layers)]


# --- the float64 reference forward pass ---------------------------------------------------------


def softplus_exact(z: torch.Tensor) -> torch.Tensor:
    """log(1 + e^z) without F.softplus's threshold, above which F.softplus returns z itself and so
    is not monotone: at z = 20 it is below its value just under 20 by about 2e-9. The boxes are
    certified for the real function."""
    return z.clamp(min=0.0) + torch.log1p(torch.exp(-z.abs()))


def log_softplus(z: torch.Tensor) -> torch.Tensor:
    """log softplus(z), finite where softplus(z) underflows: below z = -30 it is z to within
    e^z / 2, under 1e-13."""
    sp = softplus_exact(z)
    return torch.where(z < -30.0, z, torch.log(sp.clamp(min=torch.finfo(z.dtype).tiny)))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, faithful: bool = False) -> torch.Tensor:
    """MambaRMSNorm, w * x / sqrt(mean(x^2) + eps). faithful=True computes the normalisation in
    float32, as modeling_mamba.py does at lines 496-499."""
    if faithful:
        h = x.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
        return weight * h.to(x.dtype)
    return weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _round32(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float32).to(t.dtype)


@dataclass
class MixerTrace:
    """Intermediate quantities of one mixer pass, (batch, L, ...) unless noted.

    Per cell (d_inner, d_state), maxima over the batch and the positions: h_abs_max of |h_t|,
    bu_abs_max of |B_t u_t|, c_abs_max of the exponential-Euler equilibrium
    |Psi(lam dt_t) B_t u_t / lam|, gap_max of |h_{t-1} - c_t|. h is (batch, L, d_inner, d_state)
    when keep_states=True and None otherwise.
    """

    u_pre: torch.Tensor
    gate: torch.Tensor
    conv: torch.Tensor
    u: torch.Tensor
    z: torch.Tensor
    delta: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    y: torch.Tensor
    out: torch.Tensor
    h_abs_max: torch.Tensor
    bu_abs_max: torch.Tensor
    c_abs_max: torch.Tensor
    gap_max: torch.Tensor
    h: torch.Tensor | None = None


@torch.no_grad()
def mixer_trace(layer: ReleasedLayer, v: torch.Tensor, faithful: bool = False,
                keep_states: bool = False) -> MixerTrace:
    """The mixer on the normalised input v (batch, L, d_model), line for line as the slow path.

    faithful=False runs in float64 throughout. faithful=True also rounds to float32 where
    modeling_mamba.py casts (A, the dt bias, B and u in dt B u, D), which is what a float64 copy
    of the model computes there.
    """
    v = v.to(DTYPE)
    L = v.shape[1]
    xz = v @ layer.W_in.t()
    x, gate = xz.chunk(2, dim=-1)
    conv = F.conv1d(x.transpose(1, 2), layer.conv.weight, layer.conv.bias,
                    padding=layer.d_conv - 1, groups=layer.d_inner)[..., :L].transpose(1, 2)
    u = F.silu(conv)
    W_p, W_B, W_C = layer.split_x()
    dt_raw, B, C = u @ W_p.t(), u @ W_B.t(), u @ W_C.t()

    if faithful:
        A = -torch.exp(layer.A_log.to(torch.float32)).to(DTYPE)
        b_dt, D = _round32(layer.b_dt), _round32(layer.D)
        B_in, u_in = _round32(B), _round32(u)
    else:
        A, b_dt, D, B_in, u_in = -layer.lam, layer.b_dt, layer.D, B, u
    z = dt_raw @ layer.W_dt.t() + b_dt
    delta = F.softplus(z)
    lam = A.abs()

    batch = v.shape[0]
    h = v.new_zeros(batch, layer.d_inner, layer.d_state)
    h_abs = torch.zeros_like(lam)
    bu_abs = torch.zeros_like(lam)
    c_abs = torch.zeros_like(lam)
    gap = torch.zeros_like(lam)
    ys, hs = [], []
    for t in range(L):
        dt = delta[:, t].unsqueeze(-1)                     # (batch, d_inner, 1)
        bu = B_in[:, t].unsqueeze(1) * u_in[:, t].unsqueeze(-1)
        c = euler_gain(dt * lam) * bu / lam
        gap = torch.maximum(gap, (h - c).abs().amax(0))
        h = torch.exp(dt * A) * h + (dt * B_in[:, t].unsqueeze(1)) * u_in[:, t].unsqueeze(-1)
        ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
        h_abs = torch.maximum(h_abs, h.abs().amax(0))
        bu_abs = torch.maximum(bu_abs, bu.abs().amax(0))
        c_abs = torch.maximum(c_abs, c.abs().amax(0))
        if keep_states:
            hs.append(h)
    y = torch.stack(ys, dim=1) + u * D
    out = (y * F.silu(gate)) @ layer.W_out.t()
    return MixerTrace(
        u_pre=x, gate=gate, conv=conv, u=u, z=z, delta=delta, B=B, C=C, y=y, out=out,
        h_abs_max=h_abs, bu_abs_max=bu_abs, c_abs_max=c_abs, gap_max=gap,
        h=torch.stack(hs, dim=1) if keep_states else None,
    )


def mixer_forward(layer: ReleasedLayer, v: torch.Tensor, faithful: bool = False) -> torch.Tensor:
    """The mixer's output, out_proj(y * SiLU(gate)), before the residual."""
    return mixer_trace(layer, v, faithful=faithful).out


def block_forward(layer: ReleasedLayer, x: torch.Tensor, faithful: bool = False) -> torch.Tensor:
    """x + mixer(RMSNorm(x)), MambaBlock.forward. faithful=True also rounds the residual to float32
    when the config sets residual_in_fp32, as lines 523-524 do; the released config sets it."""
    v = rms_norm(x, layer.norm_weight, layer.eps, faithful)
    residual = _round32(x) if faithful and layer.residual_in_fp32 else x
    return residual + mixer_forward(layer, v, faithful)


@torch.no_grad()
def reference_forward(model, layers: list[ReleasedLayer], input_ids: torch.Tensor,
                      faithful: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Embedding, every block, then norm_f, in float64. Returns the final normalised hidden
    states and the residual stream at the input of each layer."""
    x = _f64(model.backbone.embeddings.weight)[input_ids]
    inputs = []
    for layer in layers:
        inputs.append(x)
        x = block_forward(layer, x, faithful)
    nf = model.backbone.norm_f
    return rms_norm(x, _f64(nf.weight), float(nf.variance_epsilon), faithful), inputs


# --- the data-free boxes ------------------------------------------------------------------------


def rmsnorm_box(weight: torch.Tensor) -> Interval:
    """|RMSNorm(x)_i| < |w_i| sqrt(d) for every x: the coordinate form of ||r||_2 < sqrt(d)."""
    reach = _f64(weight).abs() * math.sqrt(weight.shape[0])
    return Interval(-reach, reach)


def rmsnorm_linear_box(weight: torch.Tensor, W: torch.Tensor,
                       bias: torch.Tensor | None = None) -> Interval:
    """Box on W RMSNorm(x) + b for every x, fusing the two maps.

    Row k is <W_k * w, r> with ||r||_2 < sqrt(d), so Cauchy-Schwarz gives ||W_k * w||_2 sqrt(d),
    the supremum over the ball. RMSNorm subtracts no mean, so unlike layernorm_linear in
    intervals.py there is no centring.
    """
    v = _f64(W) * _f64(weight)
    reach = v.norm(dim=1) * math.sqrt(weight.shape[0])
    centre = torch.zeros_like(reach) if bias is None else _f64(bias)
    return Interval(centre - reach, centre + reach)


def _split(box: Interval, sizes: list[int]) -> list[Interval]:
    return [Interval(a, b) for a, b in zip(torch.split(box.lo, sizes, -1),
                                           torch.split(box.hi, sizes, -1))]


@torch.no_grad()
def layer_boxes(layer: ReleasedLayer, fused: bool = True) -> dict[str, Interval]:
    """Boxes on every intermediate quantity of the mixer, valid for every block input.

    fused=True bounds in_proj(RMSNorm(x)) row by row with rmsnorm_linear_box; fused=False takes
    the coordinate box and then the interval image under in_proj. The timescale pre-activation
    z = W_dt (W_p u) + b_dt is bounded through the composed matrix W_dt W_p, whose absolute value
    is entrywise at most |W_dt| |W_p|, so its box lies inside the two-step one.
    """
    if fused:
        in_box = rmsnorm_linear_box(layer.norm_weight, layer.W_in)
    else:
        in_box = iv.affine(rmsnorm_box(layer.norm_weight), layer.W_in, None)
    u_pre, gate_pre = _split(in_box, [layer.d_inner, layer.d_inner])
    conv = iv.depthwise_conv1d(u_pre, layer.conv)       # hulls the zero padding in
    u = iv.silu(conv)
    W_p, W_B, W_C = layer.split_x()
    B = iv.affine(u, W_B, None)
    C = iv.affine(u, W_C, None)
    z = iv.affine(u, layer.W_dt @ W_p, layer.b_dt)
    delta = Interval(softplus_exact(z.lo), softplus_exact(z.hi))
    return {"in": in_box, "u_pre": u_pre, "gate_pre": gate_pre, "conv": conv, "u": u,
            "B": B, "C": C, "z": z, "delta": delta}


@dataclass
class LayerCertificate:
    """The data-free certificate of one layer, in float64.

    Per channel: delta_lo, delta_hi (the timescale box), log_delta_lo (log of the floor, finite
    where the floor underflows). Per cell: lam, sup_Bu, M0 = sup|B u| / lam (the radius the exact
    hold would have), G = Psi(lam delta_hi), M = G M0 (the exponential-Euler invariant radius),
    geometric (the separated bound delta_hi sup|B u| / (1 - exp(-lam delta_lo)), inf where the
    floor underflows), gap = geometric / M, and the base-10 logarithms of both, which stay finite
    there because log(1 - exp(-x)) = log x + O(x) is taken from log lam + log delta_lo. Boxes:
    everything layer_boxes returns, plus h, y, the gate after SiLU, and out (the mixer output
    before the residual).
    """

    index: int
    boxes: dict
    lam: torch.Tensor
    delta_lo: torch.Tensor
    delta_hi: torch.Tensor
    log_delta_lo: torch.Tensor
    sup_Bu: torch.Tensor
    M0: torch.Tensor
    G: torch.Tensor
    M: torch.Tensor
    geometric: torch.Tensor
    gap: torch.Tensor
    log10_geometric: torch.Tensor
    log10_gap: torch.Tensor


def psi_prime(x: torch.Tensor) -> torch.Tensor:
    """Psi'(x) for Psi(x) = x / (1 - exp(-x)), x >= 0.

    Psi(x) = B(-x) with B(x) = x / (e^x - 1) the Bernoulli generating function, which is convex,
    so Psi' increases from 1/2 at 0 towards 1. Psi'(x) = (q - x e^-x) / q^2 with q = 1 - e^-x;
    below x = 1e-3 the numerator cancels and the series 1/2 + x/6 - x^3/180 is used instead.
    """
    x = torch.as_tensor(x, dtype=DTYPE)
    small = x < 1e-3
    xs = torch.where(small, torch.ones_like(x), x)
    q = -torch.expm1(-xs)
    exact = (q - xs * torch.exp(-xs)) / (q * q)
    return torch.where(small, 0.5 + x / 6.0 - x**3 / 180.0, exact)


@torch.no_grad()
def certify_layer(layer: ReleasedLayer, fused: bool = True) -> LayerCertificate:
    """Data-free certificate of one layer with respect to its RMSNorm output.

    The equilibrium of the exponential-Euler cell is c = Psi(lam dt) B u / lam, Psi increasing,
    so |c| <= Psi(lam delta_hi) sup|B u| / lam = M and the convex form keeps |h_t| <= M at every
    t (Corollary cor:euler). That needs the ceiling delta_hi and no floor. The separated
    argument bounds Abar by its supremum exp(-lam delta_lo) and the drive dt B u by
    delta_hi sup|B u|, which gives delta_hi sup|B u| / (1 - exp(-lam delta_lo)); its ratio to M is
    (1 - exp(-lam delta_hi)) / (1 - exp(-lam delta_lo)), the closed-form gap of the exact hold,
    since sup|B u| and Psi cancel.
    """
    b = layer_boxes(layer, fused)
    lam = layer.lam
    eq = equilibrium_box(-lam, b["B"], b["u"])            # B u / lam, (d_inner, d_state)
    M0 = eq.abs_max()
    sup_Bu = M0 * lam
    x_lo = lam * b["delta"].lo.unsqueeze(1)
    x_hi = lam * b["delta"].hi.unsqueeze(1)
    G = euler_gain(x_hi)
    M = G * M0
    one_minus_sup_abar = -torch.expm1(-x_lo)
    positive = one_minus_sup_abar > 0
    safe = torch.where(positive, one_minus_sup_abar, torch.ones_like(x_lo))
    geometric = torch.where(positive, b["delta"].hi.unsqueeze(1) * sup_Bu / safe,
                            torch.full_like(M, math.inf))
    gap = torch.where(positive, -torch.expm1(-x_hi) / safe, torch.full_like(M, math.inf))
    # log(1 - exp(-x)) = log x + log(1 - x/2 + ...), within 5e-9 of log x below x = 1e-8
    log_delta_lo = log_softplus(b["z"].lo).unsqueeze(1)
    tiny = x_lo < 1e-8
    log_q_lo = torch.where(tiny, torch.log(lam) + log_delta_lo,
                           torch.log(torch.where(tiny, torch.ones_like(safe), safe)))
    ln10 = math.log(10.0)
    log10_geometric = (torch.log(b["delta"].hi.unsqueeze(1) * sup_Bu) - log_q_lo) / ln10
    log10_gap = (torch.log(-torch.expm1(-x_hi)) - log_q_lo) / ln10

    h = Interval(-M, M)
    cy = h * Interval(b["C"].lo.unsqueeze(0), b["C"].hi.unsqueeze(0))
    y = cy.sum(dim=-1) + b["u"] * layer.D
    gate = iv.silu(b["gate_pre"])
    out = iv.affine(y * gate, layer.W_out, None)
    boxes = dict(b, h=h, y=y, gate=gate, out=out)
    return LayerCertificate(
        index=layer.index, boxes=boxes, lam=lam,
        delta_lo=b["delta"].lo, delta_hi=b["delta"].hi, log_delta_lo=log_softplus(b["z"].lo),
        sup_Bu=sup_Bu, M0=M0, G=G, M=M, geometric=geometric, gap=gap,
        log10_geometric=log10_geometric, log10_gap=log10_gap,
    )


# --- the data-free incremental certificate ------------------------------------------------------


@dataclass
class EulerIncrement(BlockIncrement):
    """Corollary cor:block for one layer of the released model, with the exponential-Euler cell.

    The perturbation is ||v_t - v~_t||_inf <= delta at every position. In the reading
    arbitrary=False, v and v~ are both RMSNorm outputs, so both runs lie in the certified boxes;
    in the reading arbitrary=True, v~ is any point within delta of the RMSNorm output v
    (state_bound gives the widening). The fields are BlockIncrement's with M the
    exponential-Euler radius G M0, and two more per cell: M0 = sup|B u| / lam and
    G = Psi(lam delta_hi).

    Per cell, by Corollary cor:euler with the slopes of the block corollary,
        rho      = L_log_delta delta,  L_log_delta = sup over the z box of d log dt / dz times L_z,
                   at most L_z because log softplus is 1-Lipschitz (Lemma lem:param)
        eta_B    = (sup|B| L_u + sup|u| L_B) delta / lam             (|B u - B~ u~| / lam)
        |c - c~| <= G eta_B + (e^rho - 1) M                           (the forcing of the cell)
        |h - h~| <= G eta_B + (e^rho - 1) M + 2 M (e^rho - 1)         variant "cell"
                 <= G eta_B + (e^rho - 1) M + 2 M tanh(rho / 2)       variant "common_start"
    The first is euler_cell_bound in incremental.py. The readout through C, D, the gate and
    out_proj is BlockIncrement.propagate, with |h~| <= M.
    """

    M0: torch.Tensor = None
    G: torch.Tensor = None

    def forcing_slope(self) -> torch.Tensor:
        """G eta_B / delta per cell; the Psi part of |c - c~| is first order in rho and is counted
        in warp_slope."""
        return self.G * super().forcing_slope()

    def warp_slope(self, variant: str = "cell") -> torch.Tensor:
        """Small-delta slope of the timescale part per cell: (1 + 2) M L_log_delta for the cell
        theorem and (1 + 1) M L_log_delta with a common start, the 1 being Psi's own change."""
        coef = {"cell": 3.0, "common_start": 2.0}
        if variant not in coef:
            raise ValueError(f"unknown variant: {variant!r}")
        return coef[variant] * self.M * self.L_log_delta.unsqueeze(1)

    def state_bound(self, delta: float, arbitrary: bool = False, variant: str = "cell") -> torch.Tensor:
        """sup_t |h_t - h~_t| per cell at perturbation size delta.

        arbitrary=False: both normalised inputs are RMSNorm outputs. arbitrary=True: the nominal
        one is an RMSNorm output and the perturbed one is any point within delta of it in
        sup-norm, which can leave the boxes. The boxes of u, B and z are then widened by the global
        constants of layer_lipschitz(layer, None) times delta, which gives a widened ceiling
        dt_hi', G' = Psi(lam dt_hi'), M0' = sup|B' u'| / lam and M' = G' M0'. With b = B u / lam,
            |c - c~| <= Psi(x) |b - b~| + |Psi(x) - Psi(x~)| |b~| <= G db + (e^rho - 1) G M0'
        where db = (sup|B| L_u + sup|u'| L_B) delta / lam with the global L_u and L_B, the nominal
        Psi(x) <= G and rho = L_log_delta delta with the global log slope 1. The perturbed run has
        |c~|, |h~| <= M', so the time-warp term of Theorem thm:cell is 2 M' (e^rho - 1), or
        2 M' tanh(rho / 2) with the common start.
        """
        if not arbitrary:
            rho = self.L_log_delta.unsqueeze(1) * delta
            forcing = self.forcing_slope() * delta + self.M * relative_factor(rho)
            return forcing + warp_term(self.M, rho, variant)
        g = self.global_lipschitz
        u_w = Interval(self.boxes["u"].lo - g["u"] * delta, self.boxes["u"].hi + g["u"] * delta)
        B_w = Interval(self.boxes["B"].lo - g["B"] * delta, self.boxes["B"].hi + g["B"] * delta)
        ceiling_w = softplus_exact(self.boxes["z"].hi + g["z"] * delta)
        M0_w = equilibrium_box(-self.lam, B_w, u_w).abs_max()
        M_w = euler_gain(self.lam * ceiling_w.unsqueeze(1)) * M0_w
        db = (self.sup_B.unsqueeze(0) * g["u"].unsqueeze(1)
              + u_w.abs_max().unsqueeze(1) * g["B"].unsqueeze(0)) * delta / self.lam
        rho = g["log_delta"].unsqueeze(1) * delta
        kappa = relative_factor(rho)
        return self.G * db + kappa * self.G * M0_w + warp_term(M_w, rho, variant)

    def forcing_bound(self, delta: float) -> torch.Tensor:
        """sup_t |c_t - c~_t| per cell for the exponential-Euler equilibria."""
        rho = self.L_log_delta.unsqueeze(1) * delta
        return self.forcing_slope() * delta + self.M * relative_factor(rho)

    def additive_state_slope(self, seq_len: int | None = None, floor: bool = True) -> torch.Tensor:
        """Per-cell slope of the additive baseline (Remark rem:additive) for the Euler cell.

        With |dt - dt~| <= L_delta delta and |a - a~| <= lam exp(-lam dt_min) |dt - dt~|, error form
        (1) gives |e_t| <= a_t |e_{t-1}| + (1 - a_t) eta + 2 M lam exp(-lam dt_min) L_delta delta,
        with eta >= |c - c~| = G eta_B + M0 lam sup Psi' L_delta delta, Psi' at lam delta_hi because
        Psi is convex. From e_{-1} = 0, |e_{L-1}| <= eta + 2 M lam exp(-lam dt_min) L_delta S_L
        delta, S_L = sum_{k<L} exp(-k lam dt_min). floor=False takes dt_min = 0 and S_L = L;
        floor=True the certified floor, with seq_len=None the limit 1/(1 - exp(-lam dt_min)),
        infinite where the floor underflows.
        """
        L_delta = self.L_delta.unsqueeze(1)
        eta = self.forcing_slope() + self.M0 * self.lam * psi_prime(self.lam * self.ceiling.unsqueeze(1)) * L_delta
        if not floor:
            if seq_len is None:
                return torch.full_like(eta, math.inf)
            return eta + 2.0 * self.M * self.lam * L_delta * float(seq_len)
        x = self.lam * self.floor.unsqueeze(1)
        positive = x > 0
        x_safe = torch.where(positive, x, torch.ones_like(x))
        if seq_len is None:
            S = torch.where(positive, 1.0 / -torch.expm1(-x_safe), torch.full_like(x, math.inf))
        else:
            S = torch.where(positive, torch.expm1(-float(seq_len) * x_safe) / torch.expm1(-x_safe),
                            torch.full_like(x, float(seq_len)))
        return eta + 2.0 * self.M * self.lam * torch.exp(-x) * L_delta * S

    def _zoh_only(self, *args, **kwargs):
        raise NotImplementedError("defined for the exact zero-order hold only")

    warp_ratio = _zoh_only
    crossover_horizon = _zoh_only
    additive_over_ours_one_step = _zoh_only


@torch.no_grad()
def layer_lipschitz(layer: ReleasedLayer, boxes: dict[str, Interval] | None) -> dict[str, torch.Tensor]:
    """Per-coordinate constants with respect to a sup-norm perturbation of the normalised input,
    as block_lipschitz in incremental.py: the convolution reads d_conv positions, each perturbed
    independently, and x_proj's dt block is composed with dt_proj before taking absolute values.
    With boxes=None the derivative suprema are global."""
    d = layer.d_inner
    L_u_pre = layer.W_in[:d].abs().sum(1)
    L_gate_pre = layer.W_in[d:].abs().sum(1)
    L_conv = layer.conv.weight.squeeze(1).abs().sum(1) * L_u_pre
    if boxes is None:
        s_u = torch.full_like(L_conv, SILU_DERIV_MAX)
        s_g = torch.full_like(L_gate_pre, SILU_DERIV_MAX)
        log_slope = torch.ones_like(L_conv)
        dt_slope = torch.ones_like(L_conv)
    else:
        s_u = _silu_deriv_sup(boxes["conv"].lo, boxes["conv"].hi)
        s_g = _silu_deriv_sup(boxes["gate_pre"].lo, boxes["gate_pre"].hi)
        log_slope = softplus_log_slope(boxes["z"].lo)     # decreasing in z, at most 1
        dt_slope = torch.sigmoid(boxes["z"].hi)           # softplus' = sigmoid, increasing
    L_u = s_u * L_conv
    W_p, W_B, W_C = layer.split_x()
    L_z = (layer.W_dt @ W_p).abs() @ L_u
    return {
        "u_pre": L_u_pre, "conv": L_conv, "u": L_u, "gate_pre": L_gate_pre,
        "gate": s_g * L_gate_pre, "B": W_B.abs() @ L_u, "C": W_C.abs() @ L_u, "z": L_z,
        "log_delta": log_slope * L_z, "delta": dt_slope * L_z,
    }


@torch.no_grad()
def incremental_certificate(layer: ReleasedLayer, cert: LayerCertificate | None = None) -> EulerIncrement:
    """The data-free incremental certificate of one layer, from the boxes of its certificate."""
    cert = certify_layer(layer) if cert is None else cert
    b = cert.boxes
    lip = layer_lipschitz(layer, b)
    return EulerIncrement(
        parametrisation="softplus",
        lam=cert.lam,
        M=cert.M,
        floor=cert.delta_lo,
        ceiling=cert.delta_hi,
        L_u=lip["u"], L_gate=lip["gate"], L_B=lip["B"], L_C=lip["C"], L_z=lip["z"],
        L_log_delta=lip["log_delta"], L_delta=lip["delta"],
        sup_u=b["u"].abs_max(), sup_B=b["B"].abs_max(), sup_C=b["C"].abs_max(),
        sup_y=b["y"].abs_max(), sup_gate=b["gate"].abs_max(),
        D_abs=layer.D.abs(), W_out_abs=layer.W_out.abs(),
        boxes={k: b[k] for k in ("u", "B", "C", "z", "gate_pre")},
        global_lipschitz=layer_lipschitz(layer, None),
        M0=cert.M0, G=cert.G,
    )


# --- realised quantities against the certificate ------------------------------------------------


def _range(t: torch.Tensor) -> dict:
    t = t.flatten()
    return {"min": float(t.min()), "median": float(t.median()), "max": float(t.max())}


@torch.no_grad()
def realised_layer(layer: ReleasedLayer, x: torch.Tensor, cert: LayerCertificate,
                   tr: MixerTrace | None = None) -> dict:
    """Realised ranges at one layer on the residual-stream input x (batch, L, d_model), against
    the certified boxes and radii. Ratios are certified over realised, per channel or cell."""
    v = rms_norm(x.to(DTYPE), layer.norm_weight, layer.eps)
    tr = mixer_trace(layer, v) if tr is None else tr
    b = cert.boxes

    def inside(t, box):
        lo, hi = box.lo, box.hi
        return bool(((t >= lo - 1e-9 * (1 + lo.abs())) & (t <= hi + 1e-9 * (1 + hi.abs()))).all())

    dims = (0, 1)
    d_lo = tr.delta.amin(dims)
    d_hi = tr.delta.amax(dims)
    contained = {
        "v": inside(v, rmsnorm_box(layer.norm_weight)),
        "u_pre": inside(tr.u_pre, b["u_pre"]),
        "gate_pre": inside(tr.gate, b["gate_pre"]),
        "conv": inside(tr.conv, b["conv"]),
        "u": inside(tr.u, b["u"]),
        "B": inside(tr.B, b["B"]),
        "C": inside(tr.C, b["C"]),
        "z": inside(tr.z, b["z"]),
        "delta": inside(tr.delta, b["delta"]),
        "y": inside(tr.y, b["y"]),
        "out": inside(tr.out, b["out"]),
        "Bu": bool((tr.bu_abs_max <= cert.sup_Bu * (1 + 1e-9)).all()),
        "c": bool((tr.c_abs_max <= cert.M * (1 + 1e-9)).all()),
        "h": bool((tr.h_abs_max <= cert.M * (1 + 1e-9)).all()),
    }
    lamdt = cert.lam * d_hi.unsqueeze(1)
    return {
        "layer": layer.index,
        "tokens": int(x.shape[1]),
        "contained": contained,
        "delta": {"realised_min": float(d_lo.min()), "realised_max": float(d_hi.max()),
                  "certified_lo_min": float(cert.delta_lo.min()),
                  "certified_log_lo_min": float(cert.log_delta_lo.min()),
                  "certified_hi_max": float(cert.delta_hi.max()),
                  "ceiling_over_realised_max": _range(cert.delta_hi / d_hi)},
        "z": {"realised_min": float(tr.z.min()), "realised_max": float(tr.z.max()),
              "certified_lo_min": float(b["z"].lo.min()), "certified_hi_max": float(b["z"].hi.max())},
        "Bu": {"realised_max": float(tr.bu_abs_max.max()), "certified_max": float(cert.sup_Bu.max()),
               "certified_over_realised": _range(cert.sup_Bu / tr.bu_abs_max)},
        "h": {"realised_max": float(tr.h_abs_max.max()), "certified_max": float(cert.M.max()),
              "certified_over_realised": _range(cert.M / tr.h_abs_max)},
        "u": {"realised_max": float(tr.u.abs().max()), "certified_max": float(b["u"].abs_max().max())},
        "G": {"realised_max": float(euler_gain(lamdt).max()), "certified_max": float(cert.G.max())},
        "out": {"realised_max": float(tr.out.abs().max()), "certified_max": float(b["out"].abs_max().max())},
        "gap_max": float(tr.gap_max.max()),
    }


# --- the local certificate along a nominal run --------------------------------------------------


def _local_conv(u: Sens, layer: ReleasedLayer) -> Sens:
    """The depthwise causal convolution on a value and its sensitivity. The left padding is exact
    zero in both runs and carries no sensitivity."""
    L = u.val.shape[1]
    w = layer.conv.weight.reshape(layer.d_inner, layer.d_conv)
    pad = layer.d_conv - 1
    zeros = u.val.new_zeros(u.val.shape[0], pad, layer.d_inner)
    v = torch.cat([zeros, u.val], 1)
    lp = torch.cat([zeros, u.lip], 1)
    cv = torch.zeros_like(u.val)
    cl = torch.zeros_like(u.lip)
    for k in range(layer.d_conv):
        cv = cv + v[:, k : k + L, :] * w[:, k]
        cl = cl + lp[:, k : k + L, :] * w[:, k].abs()
    if layer.conv.bias is not None:
        cv = cv + layer.conv.bias
    return Sens(cv, cl)


def euler_additive_step(h: Sens, dt_t: Sens, u_t: Sens, B_t: Sens, A: torch.Tensor, eps: float) -> Sens:
    """The additive recurrence step of lipschitz._block for the exponential-Euler cell.

    Abar = exp(dt A) has sensitivity |A| exp(dt_lo A) times that of dt, with
    dt_lo = max(dt - L_dt eps, 0) the smallest admissible timescale, and the input gain is dt
    itself, with the sensitivity of dt. The products keep their second-order terms (`mul`).
    """
    dtA = dt_t.val * A.unsqueeze(0)
    dtA_lo = (dt_t.val - dt_t.lip * eps).clamp(min=0.0) * A.unsqueeze(0)
    A_bar = Sens(torch.exp(dtA), A.abs() * torch.exp(dtA_lo) * dt_t.lip)
    return add(mul(A_bar, h, eps), mul(mul(dt_t, B_t, eps), u_t, eps))


def log_psi(y: torch.Tensor) -> torch.Tensor:
    """log Psi(y) = log(y / (1 - exp(-y))) for every real y, with log Psi(0) = 0.

    log Psi(y) = y/2 - log(sinh(y/2)/(y/2)). Below |y| = 1e-2 the series
    y/2 - y^2/24 + y^4/2880 - y^6/181440, whose next term is below 1e-23; elsewhere the closed
    form, and for y < 0 log Psi(y) = log|y| - |y| - log(1 - exp(-|y|)), which does not overflow.
    Psi(-x) is x / (e^x - 1) = exp(-x) Psi(x).
    """
    y = torch.as_tensor(y, dtype=DTYPE)
    small = y.abs() < 1e-2
    pos = y > 0
    yp = torch.where(pos & ~small, y, torch.ones_like(y))
    yn = torch.where(~pos & ~small, -y, torch.ones_like(y))
    above = torch.log(yp) - torch.log(-torch.expm1(-yp))
    below = torch.log(yn) - yn - torch.log(-torch.expm1(-yn))
    ys = torch.where(small, y, torch.zeros_like(y))
    series = ys / 2.0 - ys**2 / 24.0 + ys**4 / 2880.0 - ys**6 / 181440.0
    return torch.where(small, series, torch.where(pos, above, below))


def euler_warp(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """phi(t) = (a - s) Psi(x) - t, with a = exp(-x) and s = exp(-(x + t)), x >= 0, x + t >= 0.

    The coefficient of the nominal b = B u / lam in the time-warp part of one exponential-Euler
    step when lam dt~ = x + t. With 1 - exp(-t) = t / Psi(t) and exp(-x) Psi(x) = Psi(-x) it is
    t (Psi(-x)/Psi(t) - 1) = t expm1(log Psi(-x) - log Psi(t)), which keeps its relative
    precision where (a - s) Psi(x) and t cancel. It is 0 at t = 0 and at t = -x, positive
    between them and negative for t > 0, and it tends to -inf as t does.
    """
    t = torch.as_tensor(t, dtype=DTYPE)
    finite = torch.isfinite(t)
    ts = torch.where(finite, t, torch.zeros_like(t))
    out = ts * torch.expm1(log_psi(-x) - log_psi(ts))
    return torch.where(finite, out, -t)


def _euler_endpoint_value(E_prev, x, t, h_minus_c, b, db):
    """F(t) = s E + (x + t) db + |(a - s)(h - c) + b phi(t)| with s = exp(-(x + t)): the bound
    of `local_step_euler_endpoint` on |e_t| when lam dt~ = x + t."""
    s = torch.exp(-(x + t))
    drive = torch.where(db > 0, (x + t) * db, torch.zeros_like(db))
    # a - s = exp(-x)(1 - exp(-t)) for t >= 0 and s expm1(t) for t < 0. The first form alone
    # overflows to 0 * inf = nan once -t > 709, which a timescale far above 1/lam reaches.
    t_up, t_down = t.clamp(min=0.0), t.clamp(max=0.0)
    move = torch.exp(-x) * -torch.expm1(-t_up) + s * torch.expm1(t_down)   # a - s
    warp = torch.where(b != 0, b * euler_warp(x, t), torch.zeros_like(b))
    return s * E_prev + drive + (move * h_minus_c + warp).abs()


def local_step_euler_endpoint(E_prev, x, rho, h_prev, b, db, dx=None):
    """The endpoint form of Corollary cor:local for the exponential-Euler cell.

    The cell is h_t = a h_{t-1} + x b with a = exp(-x), x = lam dt and b = B u / lam, and the
    perturbed run is h~_t = s h~_{t-1} + x~ b~ with x~ = lam dt~ and s = exp(-x~). Collecting
    terms,

        e_t = s e_{t-1} + x~ (b - b~) + psi(s),    psi(s) = (a - s) h_{t-1} + (x - x~) b,

    an identity in which only the nominal h_{t-1} and b appear besides e_{t-1} and b - b~. With
    |e_{t-1}| <= E_{t-1} and |b - b~| <= db,

        |e_t| <= F(s) = s E_{t-1} + x~ db + |psi(s)|,

    and F(s) is attained by e_{t-1} and b - b~ of the sign of psi(s). The equilibrium c = Psi(x) b
    moves with dt as well, so unlike the hold's psi is not affine in s: with x~ = -log s,
    psi''(s) = -b / s^2. Split F = max(G_+, G_-) with

        G_sig(s) = s E_{t-1} - db log s + sig psi(s)
                 = s (E_{t-1} - sig h_{t-1}) + sig (a h_{t-1} + x b) + (sig b - db) log s.

    G_sig'' = -(sig b - db)/s^2. With N = sig b - db <= 0, G_sig is convex and its largest value
    over an interval is at an end. With N > 0 it is concave, G_sig' = (E_{t-1} - sig h_{t-1}) +
    N/s decreases, and its largest value is at s* = N / (sig h_{t-1} - E_{t-1}) clipped into the
    interval when sig h_{t-1} > E_{t-1}, and at the upper end otherwise. So

        E_t = max of F over {the two ends of I_t, s*_+ and s*_- clipped into I_t}

    is the largest value of F over I_t, and it bounds |e_t|. I_t is the intersection of the
    intervals given by rho_t and by dx = lam |dt - dt~| (`timescale_offsets`). Every candidate is
    an admissible s, so a candidate whose branch is convex is harmless. The candidates are
    evaluated through the offset t = x~ - x: a - s = exp(-x)(1 - exp(-t)), x - x~ = -t, and
    psi = (a - s)(h_{t-1} - c) + b phi(t) with phi from `euler_warp`, so a timescale far below
    1/lam keeps its digits.

    Properties, all over the reals. (i) On the rho interval, x~ = (1 - s) Psi(x~) with
    Psi(x~) <= e^rho Psi(x), |Psi(x) - Psi(x~)| <= (e^rho - 1) Psi(x) and
    |a - s| <= (e^rho - 1)(1 - s), and psi(s) = (a - s)(h - c) + (1 - s)(Psi(x) - Psi(x~)) b, so
    F(s) <= s E_{t-1} + (1 - s) K_t with K_t the forcing-plus-warp term of
    `euler_log_delta_step`: E_t is never larger than the recursion. (ii) With |dt - dt~| <= beta,
    the same bound F <= (a + g) E_{t-1} + g |h_{t-1}| + lam (dt + beta) db + lam beta |b|,
    g = lam exp(-lam dt_lo) beta, is at most `euler_additive_step`'s bound. (iii) With rho = 0
    and dx = 0 the step is a E_{t-1} + x db, the exact error of the forcing term. (iv) It is
    finite whenever dx is: x~ <= x + dx. With rho alone, x~ is unbounded above and so is
    x~ db, which is the perturbed input term x~ b~ growing with dt~, not an artefact.
    """
    up, down = timescale_offsets(x, rho, dx)
    c = euler_gain(x) * b
    h_minus_c = h_prev - c
    cands = [-down, up]
    for sig in (1.0, -1.0):
        N = sig * b - db
        D = sig * h_prev - E_prev
        ok = (N > 0) & (D > 0)
        # s* = N / D, so t* = -log s* - x; any admissible t is a valid candidate
        one = torch.ones_like(D)
        ts = torch.log(torch.where(ok, D, one)) - torch.log(torch.where(ok, N, one)) - x
        ts = torch.where(ok, torch.minimum(torch.maximum(ts, -down), up), -down)
        cands.append(ts)
    out = torch.stack([_euler_endpoint_value(E_prev, x, t, h_minus_c, b, db) for t in
                       torch.broadcast_tensors(*cands)]).amax(0)
    ok = torch.isfinite(torch.as_tensor(E_prev)) & torch.isfinite(torch.as_tensor(db))
    return torch.where(ok, out, torch.full_like(out, math.inf))


def euler_log_delta_step(h: Sens, dt_t: Sens, log_lip_t: torch.Tensor, u_t: Sens, B_t: Sens,
                         A: torch.Tensor, eps: float, recursive: bool = False,
                         endpoint: bool = False, additive_only: bool = False) -> Sens:
    """Corollary cor:local for the exponential-Euler cell, as a recurrence step. h.lip * eps
    carries E_t and the value is the nominal state.

    With x = lam dt, b = B u / lam and c = Psi(x) b the nominal equilibrium, the perturbed one is
    c~ = Psi(x~) b~, and
        |c - c~| <= Psi(x) |b - b~| + |Psi(x) - Psi(x~)| |b~| <= Psi(x) (db + kappa (|b| + db))
    with db >= |b - b~| from `mul`, kappa = e^rho - 1 and rho = log_lip_t eps, because
    Psi(x~)/Psi(x) lies in [e^-rho, e^rho] (Corollary cor:euler (i)). The step is then
    incremental.local_step (the running maximum) or local_step_recursive (the recursion E_t =
    K_t + a*_t (E_{t-1} - K_t)) with the realised distance |h_{t-1} - c_t| of the nominal run.
    endpoint=True applies `local_step_euler_endpoint` instead, with rho and the additive bound
    |dt - dt~| <= dt_t.lip eps that the domain carries; it overrides recursive. additive_only=True
    applies it with the additive bound alone (rho = inf), the ablation that isolates what the
    log-timescale bound adds; it overrides the other two.
    """
    lam = A.abs().unsqueeze(0)
    x = dt_t.val * lam
    psi = euler_gain(x)
    Bu = mul(B_t, u_t, eps)
    b = Bu.val / lam
    db = Bu.lip * eps / lam
    rho = log_lip_t * eps
    kappa = torch.expm1(rho)
    dc = psi * db + _times(kappa, psi * (b.abs() + db))
    gap = (h.val - psi * b).abs()
    E_prev = h.lip * eps
    if additive_only:
        E = local_step_euler_endpoint(E_prev, x, torch.full_like(rho, math.inf), h.val, b, db,
                                      dx=lam * (dt_t.lip * eps))
    elif endpoint:
        E = local_step_euler_endpoint(E_prev, x, rho, h.val, b, db, dx=lam * (dt_t.lip * eps))
    elif recursive:
        E = local_step_recursive(E_prev, x, rho, dc, gap)
    else:
        E = local_step(E_prev, kappa, dc, gap)
    E = torch.nan_to_num(E, nan=math.inf)
    val = torch.exp(-x) * h.val + dt_t.val * B_t.val * u_t.val
    return Sens(val, E / eps)


def _times(kappa: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # kappa * t with 0 where t is 0 even if kappa overflowed
    return torch.where(t > 0, kappa * t, torch.zeros_like(t))


# The comparison rules and the reported one, as in incremental.RULES. "log_delta_endpoint" is
# `local_step_euler_endpoint`, never larger than the recursion or than the additive step, and
# "additive_endpoint" is the same step with the additive interval alone.
LOCAL_RULES = RULES


@torch.no_grad()
def local_mixer_sensitivity(layer: ReleasedLayer, v: torch.Tensor, eps: float,
                            rule: str = REPORTED_RULE) -> dict:
    """Local bound on the mixer output around a nominal normalised input v (batch, L, d_model),
    for every perturbation of v within eps in sup-norm, an RMSNorm output or not.

    The norm-propagation domain of lipschitz.py through the released mixer: in_proj, the causal
    convolution, SiLU, x_proj with the timescale rows composed with dt_proj as in layer_boxes,
    softplus, the scan with the step chosen by `rule` (LOCAL_RULES), the readout C h + D u, the
    gate and out_proj. Returns the nominal output, its sensitivity (batch, L, d_model), so that
    |F(v~)_{t,k} - F(v)_{t,k}| <= lip_{t,k} eps, and per-run diagnostics of the scan.
    """
    if rule not in LOCAL_RULES:
        raise ValueError(f"unknown rule: {rule!r}")
    v = v.to(DTYPE)
    L = v.shape[1]
    d = layer.d_inner
    xz = linear(Sens(v, torch.ones_like(v)), layer.W_in, None)
    u = Sens(xz.val[..., :d], xz.lip[..., :d])
    gate = Sens(xz.val[..., d:], xz.lip[..., d:])
    u = silu(_local_conv(u, layer), eps)
    W_p, W_B, W_C = layer.split_x()
    B = linear(u, W_B, None)
    C = linear(u, W_C, None)
    z = Sens((u.val @ W_p.t()) @ layer.W_dt.t() + layer.b_dt, u.lip @ (layer.W_dt @ W_p).abs().t())
    z_lo, z_hi = z.box(eps)
    delta = Sens(F.softplus(z.val), torch.sigmoid(z_hi) * z.lip)
    log_lip = softplus_log_slope(z_lo) * z.lip

    A = -layer.lam
    batch = v.shape[0]
    zeros = v.new_zeros(batch, d, layer.d_state)
    h = Sens(zeros, zeros.clone())
    ys_v, ys_l = [], []
    E_max = 0.0
    for t in range(L):
        dt_t = Sens(delta.val[:, t].unsqueeze(-1), delta.lip[:, t].unsqueeze(-1))
        u_t = Sens(u.val[:, t].unsqueeze(-1), u.lip[:, t].unsqueeze(-1))
        B_t = Sens(B.val[:, t].unsqueeze(1), B.lip[:, t].unsqueeze(1))
        C_t = Sens(C.val[:, t].unsqueeze(1), C.lip[:, t].unsqueeze(1))
        if rule == "additive":
            h = euler_additive_step(h, dt_t, u_t, B_t, A, eps)
        else:
            h = euler_log_delta_step(h, dt_t, log_lip[:, t].unsqueeze(-1), u_t, B_t, A, eps,
                                     recursive=rule == "log_delta_recursive",
                                     endpoint=rule == "log_delta_endpoint",
                                     additive_only=rule == "additive_endpoint")
        m = h.lip.max()
        E_max = math.nan if torch.isnan(m) else max(E_max, float(m))
        cy = mul(h, C_t, eps)
        ys_v.append(cy.val.sum(-1) + layer.D * u.val[:, t])
        ys_l.append(cy.lip.sum(-1) + layer.D.abs() * u.lip[:, t])
    y = Sens(torch.stack(ys_v, 1), torch.stack(ys_l, 1))
    out = linear(mul(y, silu(gate, eps), eps), layer.W_out, None)
    return {
        "eps": eps,
        "rule": rule,
        "nominal": out.val,
        "lipschitz": out.lip,
        "diagnostics": {
            "u_lip_max": float(u.lip.max()),
            "z_lip_max": float(z.lip.max()),
            "rho_max": float(log_lip.max()) * eps,
            "delta_lip_max": float(delta.lip.max()),
            "h_lip_max": E_max,
            "y_lip_max": float(y.lip.max()),
        },
    }
