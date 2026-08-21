"""The exported description must reproduce the model it was taken from."""

import numpy as np
import pytest
import torch

from src.models.mamba import LOBMamba
from src.verification.export import export_parameters, minimal_artefact
from src.verification.reference import ReferenceModel


@pytest.fixture(scope="module")
def pair():
    torch.manual_seed(3)
    model = LOBMamba(input_dim=11, config={"model": {"mamba": {
        "d_model": 16, "d_state": 4, "expand": 2, "num_layers": 2}}}).eval()
    return model, ReferenceModel(minimal_artefact(model))


def test_reference_reproduces_the_output(pair):
    model, ref = pair
    x = torch.randn(5, 24, 11)
    with torch.no_grad():
        expected = model(x).numpy()
    assert np.abs(ref.forward(x.numpy()) - expected).max() < 1e-4


def test_reference_reproduces_every_intermediate(pair):
    model, ref = pair
    x = torch.randn(3, 20, 11)
    with torch.no_grad():
        _, traces = model(x, trace=True)
    _, rec = ref.forward(x.numpy(), trace=True)

    for tr, r in zip(traces, rec):
        for name, a, b in (
            ("u", tr.u, r["u"]),
            ("delta", tr.delta, r["delta"]),
            ("B", tr.B, r["B"]),
            ("C", tr.C, r["C"]),
            ("A_bar", tr.A_bar, r["A_bar"]),
            ("B_bar", tr.B_bar, r["B_bar"]),
            ("h", tr.h, r["h"]),
            ("y", tr.y, r["y"]),
        ):
            assert np.abs(a.numpy() - b).max() < 1e-4, name


def test_reference_conv_is_causal(pair):
    """Catches a reversed kernel, which agrees at t=0 and diverges after."""
    model, ref = pair
    x = np.zeros((1, 12, 11), dtype=np.float32)
    x[0, 4] = 1.0
    y = ref.forward(x)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    assert np.abs(y - expected).max() < 1e-4


def test_softplus_variant_round_trips():
    torch.manual_seed(4)
    model = LOBMamba(input_dim=9, config={"model": {"mamba": {
        "d_model": 16, "d_state": 4, "num_layers": 1,
        "dt_parametrisation": "softplus"}}}).eval()
    ref = ReferenceModel(minimal_artefact(model))
    x = torch.randn(4, 15, 9)
    with torch.no_grad():
        expected = model(x).numpy()
    assert np.abs(ref.forward(x.numpy()) - expected).max() < 1e-4


def test_parameters_round_trip_through_json(pair):
    import json
    model, _ = pair
    params = export_parameters(model)
    restored = json.loads(json.dumps(params))
    a = np.asarray(params["layers"][0]["A"], dtype=np.float32)
    b = np.asarray(restored["layers"][0]["A"], dtype=np.float32)
    assert np.array_equal(a, b)
