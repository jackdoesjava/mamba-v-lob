import torch
import pytest
from python.models.transformer import LOBTransformer

def test_transformer_initialization():
    """Ensure the model initializes with institutional LOB feature dimensions."""
    model = LOBTransformer(input_dim=43, d_model=64, nhead=4, num_layers=2)
    assert model.input_dim == 43
    assert model.d_model == 64

def test_transformer_forward_shape():
    """Verify the forward pass outputs the correct scalar prediction shape."""
    batch_size = 16
    seq_len = 50
    features = 43
    
    x = torch.randn(batch_size, seq_len, features)
    model = LOBTransformer(input_dim=features)
    
    # Model should output a 1D tensor of shape (batch_size,) after squeezing
    out = model(x)
    assert out.shape == (batch_size,), f"Expected shape ({batch_size},), got {out.shape}"

def test_transformer_mixed_precision_dtype_safety():
    """Ensure the internal projection casts safely to avoid CPU AMP crashes."""
    batch_size, seq_len, features = 4, 10, 43
    # Simulate an FP16 input that might come from an AMP autocast context
    x_fp16 = torch.randn(batch_size, seq_len, features, dtype=torch.float16)
    model = LOBTransformer(input_dim=features)
    
    # The model's forward pass should internally cast x to float32 to prevent LayerNorm crashes
    out = model(x_fp16)
    assert out.dtype == torch.float32, "Transformer did not cast output to float32 safely."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_transformer_device_routing():
    """Ensure all internal buffers and positional encodings move to GPU correctly."""
    model = LOBTransformer(input_dim=43).cuda()
    x = torch.randn(8, 20, 43).cuda()
    out = model(x)
    assert out.device.type == 'cuda'