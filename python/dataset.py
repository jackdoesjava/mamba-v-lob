import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np

class LOBDataset(Dataset):
    def __init__(self, config: dict, is_train: bool = True):
        self.seq_len = config['data']['seq_length']
        
        df = pl.read_parquet(config['data']['processed_file'])
        
        # 1. Load pre-calculated ground truth targets
        targets = df.select(pl.col("target_log_return")).to_numpy().flatten().astype(np.float32)

        # 2. Drop absolute prices and target leakage
        forbidden_cols = {
            "ts_event", 
            "ts_in_delta", 
            "micro_price", 
            "mid_price",          
            "target_log_return",  
            "log_return"
        }
        feature_cols = [col for col in df.columns if col not in forbidden_cols]
        features = df.select(feature_cols).to_numpy().astype(np.float32)

        # 3. Chronological split for normalization
        split_idx = int(len(df) * config['data']['train_split'])

        # Feature normalization
        train_features = features[:split_idx]
        feat_mean = train_features.mean(axis=0, keepdims=True)
        feat_std = np.maximum(train_features.std(axis=0, keepdims=True), 1e-8)
        normalized_features = (features - feat_mean) / feat_std

        # Target normalization (Strictly using train split to avoid lookahead bias)
        train_targets = targets[:split_idx]
        target_mean = train_targets.mean()
        target_std = np.maximum(train_targets.std(), 1e-8)
        normalized_targets = (targets - target_mean) / target_std

        # 4. Assign full data splits
        if is_train:
            self.features = normalized_features[:split_idx]
            self.targets = normalized_targets[:split_idx]
        else:
            self.features = normalized_features[split_idx:]
            self.targets = normalized_targets[split_idx:]
            
        self.valid_length = len(self.features) - self.seq_len

    def __len__(self) -> int:
        return self.valid_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_window = self.features[idx : idx + self.seq_len]
        y_target = self.targets[idx + self.seq_len - 1]
        
        return torch.tensor(x_window, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)


def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    train_dataset = LOBDataset(config, is_train=True)
    test_dataset = LOBDataset(config, is_train=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=True,  
        drop_last=True,
        pin_memory=torch.cuda.is_available() 
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=False, 
        drop_last=True,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader
