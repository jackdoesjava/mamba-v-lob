"""Interval abstract domain over the ops the selective SSM uses.

Every function must return a sound over-approximation; tests/test_verification.py checks
that by sampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mamba import SILU_ARGMIN, SILU_MIN, zoh_phi


@dataclass
class Interval:
    lo: torch.Tensor
    hi: torch.Tensor

    def __post_init__(self) -> None:
        if self.lo.shape != self.hi.shape:
            raise ValueError(f"shape mismatch: {self.lo.shape} vs {self.hi.shape}")

    @staticmethod
    def point(x: torch.Tensor) -> "Interval":
        return Interval(x.clone(), x.clone())

    @staticmethod
    def symmetric(radius: torch.Tensor) -> "Interval":
        return Interval(-radius.abs(), radius.abs())

    def __add__(self, other: "Interval | torch.Tensor") -> "Interval":
        if isinstance(other, Interval):
            return Interval(self.lo + other.lo, self.hi + other.hi)
        return Interval(self.lo + other, self.hi + other)

    def __mul__(self, other: "Interval | torch.Tensor") -> "Interval":
        if not isinstance(other, Interval):
            other = Interval.point(other)
        products = torch.stack(
            [
                self.lo * other.lo,
                self.lo * other.hi,
                self.hi * other.lo,
                self.hi * other.hi,
            ]
        )
        return Interval(products.amin(0), products.amax(0))

    def __sub__(self, other: "Interval | torch.Tensor") -> "Interval":
        if isinstance(other, Interval):
            return Interval(self.lo - other.hi, self.hi - other.lo)
        return Interval(self.lo - other, self.hi - other)

    def square(self) -> "Interval":
        lo2, hi2 = self.lo * self.lo, self.hi * self.hi
        straddles = (self.lo <= 0) & (self.hi >= 0)
        lower = torch.where(straddles, torch.zeros_like(lo2), torch.minimum(lo2, hi2))
        return Interval(lower, torch.maximum(lo2, hi2))

    def sqrt(self) -> "Interval":
        return Interval(self.lo.clamp(min=0.0).sqrt(), self.hi.clamp(min=0.0).sqrt())

    def mean(self, dim: int, keepdim: bool = False) -> "Interval":
        return Interval(self.lo.mean(dim, keepdim=keepdim), self.hi.mean(dim, keepdim=keepdim))

    def centre(self) -> torch.Tensor:
        return 0.5 * (self.lo + self.hi)

    def radius(self) -> torch.Tensor:
        return 0.5 * (self.hi - self.lo)

    def abs_max(self) -> torch.Tensor:
        return torch.maximum(self.lo.abs(), self.hi.abs())

    def width(self) -> torch.Tensor:
        return self.hi - self.lo

    def contains(self, x: torch.Tensor, atol: float = 1e-5) -> torch.Tensor:
        return (x >= self.lo - atol) & (x <= self.hi + atol)

    def hull_with_zero(self) -> "Interval":
        """Smallest interval containing this one and 0, for signals padded with exact zeros."""
        return Interval(self.lo.clamp(max=0.0), self.hi.clamp(min=0.0))

    def sum(self, dim: int) -> "Interval":
        return Interval(self.lo.sum(dim), self.hi.sum(dim))

    def to_dict(self) -> dict:
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist()}


# Monotone elementwise maps: endpoints suffice.
def monotone(box: Interval, fn) -> Interval:
    return Interval(fn(box.lo), fn(box.hi))


def sigmoid(box: Interval) -> Interval:
    return monotone(box, torch.sigmoid)


def softplus(box: Interval) -> Interval:
    return monotone(box, F.softplus)


def exp(box: Interval) -> Interval:
    return monotone(box, torch.exp)


def silu(box: Interval) -> Interval:
    """SiLU dips to SILU_MIN at x*, so the minimum can be interior to the box."""
    f_lo, f_hi = F.silu(box.lo), F.silu(box.hi)
    upper = torch.maximum(f_lo, f_hi)
    straddles = (box.lo <= SILU_ARGMIN) & (box.hi >= SILU_ARGMIN)
    lower = torch.where(
        straddles, torch.full_like(f_lo, SILU_MIN), torch.minimum(f_lo, f_hi)
    )
    return Interval(lower, upper)


def affine(box: Interval, weight: torch.Tensor, bias: torch.Tensor | None) -> Interval:
    """Interval image of `x -> W x + b`, splitting W into its positive and negative parts."""
    w_pos, w_neg = weight.clamp(min=0.0), weight.clamp(max=0.0)
    lo = w_pos @ box.lo + w_neg @ box.hi
    hi = w_pos @ box.hi + w_neg @ box.lo
    if bias is not None:
        lo, hi = lo + bias, hi + bias
    return Interval(lo, hi)


def depthwise_conv1d(box: Interval, conv: nn.Conv1d) -> Interval:
    """Interval image of a depthwise causal Conv1d, assuming the box holds at every timestep."""
    if conv.groups != conv.in_channels:
        raise ValueError("expected a depthwise conv (groups == in_channels)")
    # the leading pad taps are exactly 0, which need not lie in the box; hull it in or the
    # bound is unsound for t < d_conv - 1
    padded = box.hull_with_zero()
    w = conv.weight.squeeze(1)                       # (channels, kernel)
    w_pos, w_neg = w.clamp(min=0.0), w.clamp(max=0.0)
    lo = (w_pos * padded.lo.unsqueeze(1)).sum(1) + (w_neg * padded.hi.unsqueeze(1)).sum(1)
    hi = (w_pos * padded.hi.unsqueeze(1)).sum(1) + (w_neg * padded.lo.unsqueeze(1)).sum(1)
    if conv.bias is not None:
        lo, hi = lo + conv.bias, hi + conv.bias
    return Interval(lo, hi)


def dense_conv1d(box: Interval, conv: nn.Conv1d) -> Interval:
    """Same, for a channel-mixing Conv1d."""
    padded = box.hull_with_zero()
    w = conv.weight                                   # (out, in, kernel)
    w_pos, w_neg = w.clamp(min=0.0), w.clamp(max=0.0)
    lo = (w_pos * padded.lo.view(1, -1, 1)).sum((1, 2)) + (
        w_neg * padded.hi.view(1, -1, 1)
    ).sum((1, 2))
    hi = (w_pos * padded.hi.view(1, -1, 1)).sum((1, 2)) + (
        w_neg * padded.lo.view(1, -1, 1)
    ).sum((1, 2))
    if conv.bias is not None:
        lo, hi = lo + conv.bias, hi + conv.bias
    return Interval(lo, hi)


def layernorm(ln: nn.LayerNorm) -> Interval:
    """Box on LayerNorm's output, valid for any input."""
    d = ln.normalized_shape[0]
    # sum(z) = 0 and ||z||_2^2 <= d force |z_i| <= sqrt(d-1) by Cauchy-Schwarz on the
    # other d-1 coordinates; output is gamma*z + beta
    reach = ln.weight.abs() * math.sqrt(d - 1)
    return Interval(ln.bias - reach, ln.bias + reach)


def layernorm_linear(ln: nn.LayerNorm, linear: nn.Linear) -> Interval:
    """Box on `Linear(LayerNorm(x))`, fusing the two layers instead of bounding each."""
    d = ln.normalized_shape[0]
    v = linear.weight * ln.weight                     # (out, d)
    # sup <v, z> over {sum(z) = 0, ||z||_2 <= sqrt(d)} is ||v - mean(v)||_2 * sqrt(d),
    # and it is attained
    v_centred = v - v.mean(dim=1, keepdim=True)
    reach = v_centred.norm(dim=1) * math.sqrt(d)
    centre = linear.weight @ ln.bias
    if linear.bias is not None:
        centre = centre + linear.bias
    return Interval(centre - reach, centre + reach)


def delta_box(block, pre: Interval) -> Interval:
    """Box on delta; both parametrisations are monotone in the pre-activation."""
    out = monotone(pre, block.delta_from_pre)
    if block.dt_parametrisation == "bounded":
        # delta_from_pre already clamps; re-assert it on the box so the interval is inside
        # [dt_min, dt_max] whatever the monotone map rounded to
        out = Interval(
            out.lo.clamp(min=block.dt_min, max=block.dt_max),
            out.hi.clamp(min=block.dt_min, max=block.dt_max),
        )
    return out


def discretisation_boxes(
    A: torch.Tensor, delta: Interval
) -> tuple[Interval, Interval]:
    """Boxes on Abar = exp(dt A) and the ZOH gain g(dt) = (exp(dt A) - 1)/A.

    A is (d_inner, d_state); delta is a per-channel box broadcast over d_state.
    """
    # A < 0, dt > 0: Abar is decreasing in dt, g increasing (g' = exp(dt A) > 0), so the
    # endpoints give the box and g stays positive
    d_lo = delta.lo.unsqueeze(-1)
    d_hi = delta.hi.unsqueeze(-1)
    A_ = A.unsqueeze(0) if A.dim() == 2 and d_lo.dim() == 3 else A

    A_bar = Interval(torch.exp(d_hi * A_), torch.exp(d_lo * A_))
    g = Interval(d_lo * zoh_phi(d_lo * A_), d_hi * zoh_phi(d_hi * A_))
    return A_bar, g


def state_bound_geometric(A_bar: Interval, drive: Interval) -> Interval:
    """Fixed point of |h| <= sup(Abar) |h| + sup|drive|; infinite box when sup Abar >= 1."""
    s = A_bar.abs_max()
    m = drive.abs_max()
    with torch.no_grad():
        # torch.where evaluates both lanes, so the clamp only keeps the discarded s >= 1
        # lane finite; the selected lane returns inf explicitly
        gain = torch.where(
            s < 1.0, 1.0 / (1.0 - s).clamp(min=1e-30), torch.full_like(s, float("inf"))
        )
    radius = m * gain
    return Interval(-radius, radius)


def state_bound_unrolled(A_bar: Interval, drive: Interval, seq_len: int) -> Interval:
    """Iterate h <- Abar h + drive for L steps. Stays finite even when sup Abar = 1."""
    lo = torch.zeros_like(drive.lo)
    hi = torch.zeros_like(drive.hi)
    for _ in range(seq_len):
        # Abar > 0: the extreme of Abar*h is at (a_hi, h_hi) when h_hi >= 0, else (a_lo, h_hi)
        new_hi = torch.where(hi >= 0, A_bar.hi * hi, A_bar.lo * hi) + drive.hi
        new_lo = torch.where(lo >= 0, A_bar.lo * lo, A_bar.hi * lo) + drive.lo
        lo, hi = new_lo, new_hi
    return Interval(lo, hi)


def divide_by_positive(num: Interval, den: Interval) -> Interval:
    """num / den where den is strictly positive. Reciprocal of a positive box is monotone."""
    inv = Interval(1.0 / den.hi, 1.0 / den.lo.clamp(min=1e-30))
    return num * inv


def layernorm_input_box(box: Interval, ln: nn.LayerNorm, eps: float = 1e-5) -> Interval:
    """LayerNorm over a box around a concrete point, in centre-radius form.

    Pure interval arithmetic puts 0 in the variance, because every centred coordinate
    straddles it, and 1/sigma then reaches 1/sqrt(eps). Writing x = c + e with |e_i| <= r_i
    keeps the concrete spread of c and only gives up the perturbation: the mean shifts by at
    most mean(r), so |x_i - mean(x) - (c_i - mean(c))| <= r_i + mean(r).

    `layernorm` in this module is the complementary bound: input-independent, and the one
    the global certificate uses.
    """
    c, r = box.centre(), box.radius()
    c_centred = c - c.mean(-1, keepdim=True)
    reach = r + r.mean(-1, keepdim=True)

    lo_abs = (c_centred.abs() - reach).clamp(min=0.0)
    hi_abs = c_centred.abs() + reach
    var = Interval((lo_abs * lo_abs).mean(-1, keepdim=True), (hi_abs * hi_abs).mean(-1, keepdim=True))
    sigma = Interval(var.lo + eps, var.hi + eps).sqrt()

    z = divide_by_positive(Interval(c_centred - reach, c_centred + reach), sigma)
    return z * ln.weight + ln.bias
