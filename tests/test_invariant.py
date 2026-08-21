"""The invariant, and the discretisation identity it rests on."""

import pytest
import torch

from src.models.mamba import LOBMamba, SelectiveSSMBlock, zoh_phi
from src.verification import intervals as iv
from src.verification.certify import certify_block
from src.verification.intervals import Interval
from src.verification.invariant import (
    certify_block_state,
    equilibrium_box,
    induction_residual,
    invariant_box,
    zoh_is_exact,
)


@pytest.fixture(scope="module")
def block():
    torch.manual_seed(5)
    return SelectiveSSMBlock(d_model=16, d_state=4, expand=2).eval()


@pytest.mark.parametrize("dt", [1e-8, 1e-4, 1e-2, 0.1, 1.0, 5.0])
def test_zoh_gives_the_convex_form(dt):
    """Bbar = (1 - Abar) B / |A| exactly. Everything else follows from this."""
    A = -torch.tensor([0.5, 1.0, 7.0, 16.0], dtype=torch.float64)
    d = torch.tensor(dt, dtype=torch.float64)
    multiplier = d * zoh_phi(d * A)
    assert torch.allclose(multiplier, (1.0 - torch.exp(d * A)) / A.abs(), atol=1e-14)


def test_euler_does_not_give_the_convex_form():
    """Guards the certificate against being applied to a Euler-discretised model."""
    A = -torch.tensor([1.0, 16.0], dtype=torch.float64)
    dt = torch.tensor(0.5, dtype=torch.float64)
    euler = dt.expand_as(A)
    assert not torch.allclose(euler, (1.0 - torch.exp(dt * A)) / A.abs(), atol=1e-6)


def test_zoh_check_accepts_the_real_discretisation(block):
    assert zoh_is_exact(-torch.exp(block.A_log), block.dt_max)


def test_induction_residual_is_zero_by_construction(block):
    cert = certify_block(block, seq_len=32)
    state = certify_block_state(block, cert["boxes"]["u"], cert["boxes"]["B"])
    assert state["holds"]
    assert state["induction_residual"] == pytest.approx(0.0, abs=1e-9)


def test_invariant_contains_the_equilibrium(block):
    cert = certify_block(block, seq_len=32)
    A = -torch.exp(block.A_log)
    eq = equilibrium_box(A, cert["boxes"]["B"], cert["boxes"]["u"])
    inv = invariant_box(A, cert["boxes"]["B"], cert["boxes"]["u"])
    assert (eq.abs_max() <= inv.abs_max() + 1e-9).all()


def test_residual_grows_if_the_equilibrium_leaves_the_box():
    """A box smaller than the equilibrium is not invariant, and the check must say so."""
    eq = Interval(torch.tensor([-2.0]), torch.tensor([2.0]))
    small = Interval(torch.tensor([-1.0]), torch.tensor([1.0]))
    assert induction_residual(small, eq).item() == pytest.approx(1.0)


@pytest.mark.parametrize("seq_len", [10, 100, 1000, 4000])
def test_invariant_holds_at_lengths_it_was_not_certified_at(block, seq_len):
    """The point of an invariant: one induction step covers every length."""
    cert = certify_block(block, seq_len=32)
    state = certify_block_state(block, cert["boxes"]["u"], cert["boxes"]["B"])
    radius = state["invariant"].abs_max()

    torch.manual_seed(seq_len)
    x = torch.randn(2, seq_len, block.d_model) * 3.0
    with torch.no_grad():
        _, tr = block(x, trace=True)
    assert (tr.h.abs().amax(dim=(0, 1)) <= radius + 1e-5).all()


def test_invariant_beats_the_widened_fixed_point(block):
    """The comparison the paper makes, against a converged baseline rather than a truncated one."""
    cert = certify_block(block, seq_len=64)
    state = certify_block_state(block, cert["boxes"]["u"], cert["boxes"]["B"])
    drive = cert["boxes"]["B_bar"] * Interval(
        cert["boxes"]["u"].lo.unsqueeze(-1), cert["boxes"]["u"].hi.unsqueeze(-1)
    )
    widened, iters = iv.state_bound_widened(cert["boxes"]["A_bar"], drive)
    assert iters > 1
    assert widened.abs_max().max() > state["invariant"].abs_max().max()


def test_widening_converges_to_the_geometric_bound(block):
    """Confirms the geometric bound is the fixed point, so it is the fair baseline."""
    cert = certify_block(block, seq_len=64)
    drive = cert["boxes"]["B_bar"] * Interval(
        cert["boxes"]["u"].lo.unsqueeze(-1), cert["boxes"]["u"].hi.unsqueeze(-1)
    )
    widened, _ = iv.state_bound_widened(cert["boxes"]["A_bar"], drive)
    geometric = iv.state_bound_geometric(cert["boxes"]["A_bar"], drive)
    assert widened.abs_max().max() == pytest.approx(
        geometric.abs_max().max().item(), rel=1e-3
    )


def test_softplus_block_is_also_invariant():
    """No lower bound on delta is needed. The convex form does not use one."""
    torch.manual_seed(6)
    block = SelectiveSSMBlock(d_model=16, d_state=4, dt_parametrisation="softplus").eval()
    cert = certify_block(block, seq_len=32)
    state = certify_block_state(block, cert["boxes"]["u"], cert["boxes"]["B"])
    assert state["holds"]

    x = torch.randn(2, 500, 16) * 3.0
    with torch.no_grad():
        _, tr = block(x, trace=True)
    assert (tr.h.abs().amax(dim=(0, 1)) <= state["invariant"].abs_max() + 1e-5).all()


def test_model_certificate_reports_the_invariant():
    torch.manual_seed(7)
    model = LOBMamba(input_dim=12, config={"model": {"mamba": {
        "d_model": 16, "d_state": 4, "num_layers": 2}}}).eval()
    from src.verification.certify import certify_model

    cert = certify_model(model, seq_len=50)
    for layer in cert["layers"]:
        s = layer["summary"]
        assert s["invariant_holds"]
        assert s["zoh_exact"]
        assert s["h_abs_max_invariant"] < s["h_abs_max_widened"]


@pytest.mark.parametrize("dt_min", [1e-5, 1e-4, 1e-3, 1e-2])
def test_invariant_does_not_depend_on_the_timescale_floor(dt_min):
    """The radius is sup|B u| / |A|, which contains no dt. The geometric bound scales as 1/dt_min.

    This is what separates the two: the floor is the whole reason the geometric bound moves,
    and the invariant never looks at it.
    """
    torch.manual_seed(21)
    block = SelectiveSSMBlock(d_model=16, d_state=4, dt_min=dt_min, dt_max=0.1).eval()
    cert = certify_block(block, seq_len=32)
    state = certify_block_state(block, cert["boxes"]["u"], cert["boxes"]["B"])
    assert state["holds"]

    torch.manual_seed(21)
    reference = SelectiveSSMBlock(d_model=16, d_state=4, dt_min=1e-3, dt_max=0.1).eval()
    ref_cert = certify_block(reference, seq_len=32)
    ref_state = certify_block_state(reference, ref_cert["boxes"]["u"], ref_cert["boxes"]["B"])
    assert state["radius"] == pytest.approx(ref_state["radius"], rel=1e-6)


def test_geometric_bound_pays_for_the_floor_and_the_invariant_does_not():
    """Roughly an order of magnitude looser per decade the floor drops.

    Not exactly ten: the bound takes an elementwise maximum over (channel, state) and the
    index attaining it can move with the floor. The claim under test is that the geometric
    bound moves substantially while the invariant does not move at all.
    """
    geometric, invariant = {}, {}
    for dt_min in (1e-3, 1e-2):
        torch.manual_seed(22)
        block = SelectiveSSMBlock(d_model=16, d_state=4, dt_min=dt_min, dt_max=0.1).eval()
        cert = certify_block(block, seq_len=32)
        geometric[dt_min] = cert["summary"]["h_abs_max_geometric"]
        invariant[dt_min] = certify_block_state(
            block, cert["boxes"]["u"], cert["boxes"]["B"]
        )["radius"]

    assert geometric[1e-3] / geometric[1e-2] > 5.0
    assert invariant[1e-3] == pytest.approx(invariant[1e-2], rel=1e-6)
