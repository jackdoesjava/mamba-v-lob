import torch
import pytest
from python.models.lstm import LSTMBaseline
from python.utils.config import load_config

@pytest.fixture(scope="module")
def mock_config():
    return load_config()

@pytest.fixture(scope="module")
def mock_input_tensor(mock_config):
    batch_size = 32
    seq_len = mock_config['data']['seq_length']
    input_dim = 46 # Updated to match your runtime dynamic check
    return torch.randn(batch_size, seq_len, input_dim)

def test_lstm_forward_shape(mock_config, mock_input_tensor):
    model = LSTMBaseline(input_dim=mock_input_tensor.shape[-1], config=mock_config)
    output = model(mock_input_tensor)
    
    expected_shape = (mock_input_tensor.size(0),)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

def test_lstm_backward_pass(mock_config, mock_input_tensor):
    model = LSTMBaseline(input_dim=mock_input_tensor.shape[-1], config=mock_config)
    output = model(mock_input_tensor)
    
    loss = torch.nn.functional.mse_loss(output, torch.randn_like(output))
    loss.backward()
    
    assert model.fc.weight.grad is not None, "Gradients failed to flow to the final layer."

def test_lstm_batch_independence(mock_config, mock_input_tensor):
    model = LSTMBaseline(input_dim=mock_input_tensor.shape[-1], config=mock_config)
    model.eval()
    
    with torch.no_grad():
        batch_out = model(mock_input_tensor)
        single_out = model(mock_input_tensor[0:1, :, :])
        
    assert torch.allclose(batch_out[0], single_out[0], atol=1e-6), \
        "Batch prediction differs from single-sample inference."

def test_lstm_determinism(mock_config, mock_input_tensor):
    model = LSTMBaseline(input_dim=mock_input_tensor.shape[-1], config=mock_config)
    model.eval()
    
    with torch.no_grad():
        out1 = model(mock_input_tensor)
        out2 = model(mock_input_tensor)
        
    assert torch.equal(out1, out2), "LSTM output is non-deterministic in eval mode."