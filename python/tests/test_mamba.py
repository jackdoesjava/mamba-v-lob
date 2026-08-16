import torch
import pytest
from python.models.mamba import LOBMamba
from python.utils.config import load_config

@pytest.fixture(scope="module")
def mock_config():
    return load_config()

@pytest.fixture(scope="module")
def mock_input_tensor(mock_config):
    batch_size = 32
    seq_len = mock_config['data']['seq_length']
    input_dim = 47 
    return torch.randn(batch_size, seq_len, input_dim)

def test_mamba_forward_shape(mock_config, mock_input_tensor):
    model = LOBMamba(input_dim=mock_input_tensor.shape[-1], config=mock_config)
    output = model(mock_input_tensor)
    
    expected_shape = (mock_input_tensor.size(0),)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

def test_mamba_backward_pass(mock_config, mock_input_tensor):
    model = LOBMamba(input_dim=mock_input_tensor.shape[-1], config=mock_config)
    output = model(mock_input_tensor)
    
    loss = torch.nn.functional.mse_loss(output, torch.randn_like(output))
    loss.backward()
    
    # check that gradients successfully propagated through the Mamba blocks back to the input
    assert model.fc.weight.grad is not None, "Gradients failed at the final regression head."
    assert model.proj[0].weight.grad is not None, "Gradients failed to reach the initial spatial projection."

def test_mamba_batch_independence(mock_config, mock_input_tensor):
    model = LOBMamba(input_dim=mock_input_tensor.shape[-1], config=mock_config)
    model.eval()
    with torch.no_grad():
        batch_out = model(mock_input_tensor)
        # isolate the first sequence in the batch and run it solo
        single_out = model(mock_input_tensor[0:1, :, :])
        
    assert torch.allclose(batch_out[0], single_out[0], atol=1e-6), \
        "Batch prediction differs from single-sample inference. Cross-batch leakage detected."

def test_mamba_determinism(mock_config, mock_input_tensor):
    model = LOBMamba(input_dim=mock_input_tensor.shape[-1], config=mock_config)
    model.eval()
    with torch.no_grad():
        out1 = model(mock_input_tensor)
        out2 = model(mock_input_tensor)
        
    assert torch.equal(out1, out2), "Model output is non-deterministic in eval mode."