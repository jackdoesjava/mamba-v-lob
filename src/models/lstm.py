import torch
import torch.nn as nn

class LSTMBaseline(nn.Module):
    def __init__(self, input_dim: int, config: dict):
        super().__init__()
        cfg = config['model']['lstm']
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=cfg['hidden_size'],
            num_layers=cfg['num_layers'],
            batch_first=True,
            dropout=cfg['dropout'] if cfg['num_layers'] > 1 else 0.0
        )
        
        self.fc = nn.Linear(cfg['hidden_size'], 1)
        self._init_weights()

    def _init_weights(self):
        # weight_hh is (4*hidden, hidden), so orthogonal_ gives a semi-orthogonal matrix:
        # the stacked i,f,g,o recurrent map is norm-preserving at init.
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)
                # torch packs the gate biases as i,f,g,o, so [n/4:n/2] is the forget gate;
                # 1.0 holds it open until training learns otherwise
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
                
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)