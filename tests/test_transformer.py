import pytest
import torch

from src.models.transformer import LOBTransformer


def cfg(**overrides) -> dict:
    return {"model": {"transformer": {"d_model": 64, "nhead": 4, "num_layers": 2, **overrides}}}


def test_transformer_reads_its_config():
    model = LOBTransformer(input_dim=43, config=cfg(d_model=32, num_layers=3))
    assert model.input_dim == 43
    assert model.d_model == 32
    assert len(model.transformer_encoder.layers) == 3


def test_transformer_forward_shape():
    x = torch.randn(16, 50, 43)
    out = LOBTransformer(input_dim=43, config=cfg())(x)
    assert out.shape == (16,)


def test_transformer_casts_half_inputs():
    # LayerNorm on CPU rejects fp16, so the model casts on entry
    x = torch.randn(4, 10, 43, dtype=torch.float16)
    out = LOBTransformer(input_dim=43, config=cfg())(x)
    assert out.dtype == torch.float32


def test_causal_mask_is_applied_when_configured():
    torch.manual_seed(0)
    model = LOBTransformer(input_dim=8, config=cfg(dropout=0.0, causal=True)).eval()

    x = torch.randn(1, 12, 8)
    x2 = x.clone()
    x2[:, 9] += 10.0

    def hidden(inp):
        h = model.feature_projection(inp)
        h = model.pos_encoder(h)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(h.size(1))
        return model.transformer_encoder(h, mask=mask, is_causal=True)

    with torch.no_grad():
        assert torch.allclose(hidden(x)[:, :9], hidden(x2)[:, :9], atol=1e-5)


def test_bidirectional_variant_is_not_causal():
    torch.manual_seed(0)
    model = LOBTransformer(input_dim=8, config=cfg(dropout=0.0, causal=False)).eval()

    x = torch.randn(1, 12, 8)
    x2 = x.clone()
    x2[:, 9] += 10.0

    def hidden(inp):
        h = model.feature_projection(inp)
        h = model.pos_encoder(h)
        return model.transformer_encoder(h, mask=None, is_causal=False)

    with torch.no_grad():
        assert not torch.allclose(hidden(x)[:, :9], hidden(x2)[:, :9], atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_transformer_device_routing():
    model = LOBTransformer(input_dim=43, config=cfg()).cuda()
    assert model(torch.randn(8, 20, 43).cuda()).device.type == "cuda"
