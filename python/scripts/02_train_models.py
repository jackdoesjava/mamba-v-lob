import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from python.utils.config import load_config
from python.dataset import get_dataloaders, LOBDataset
from python.models.lstm import LSTMBaseline
from python.models.mamba import LOBMamba
from python.models.transformer import LOBTransformer

torch.backends.cudnn.benchmark = True

def get_model(model_name: str, input_dim: int, config: dict) -> nn.Module:
    if model_name == 'lstm':
        return LSTMBaseline(input_dim, config)
    elif model_name == 'mamba':
        return LOBMamba(input_dim, config)
    elif model_name == 'transformer':
        return LOBTransformer(input_dim, config)
    raise ValueError(f"Unknown architecture: {model_name}")

def train_epoch(model: nn.Module, dataloader, criterion, optimizer, scaler, device, steps_per_epoch=30) -> float:
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, total=steps_per_epoch, desc="Training", leave=False)
    is_cuda = device.type == 'cuda'
    
    for step, (x, y) in enumerate(progress_bar):
        if step >= steps_per_epoch:
            break
            
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=is_cuda):
            pred = model(x)
            loss = criterion(pred, y)
        
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
    return total_loss / steps_per_epoch

@torch.no_grad()
def evaluate(model: nn.Module, dataloader, criterion, device, val_steps=200) -> float:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    is_cuda = device.type == 'cuda'
    
    for step, (x, y) in enumerate(dataloader):
        if step >= val_steps:
            break
            
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=is_cuda):
            pred = model(x)
            loss = criterion(pred, y)
            
        total_loss += loss.item()
        
        all_preds.append(pred.float().cpu())
        all_targets.append(y.float().cpu())
    
    preds = torch.cat(all_preds).flatten()
    targets = torch.cat(all_targets).flatten()
    
    corr_matrix = torch.corrcoef(torch.stack([preds, targets]))
    correlation = corr_matrix[0, 1].item()
    
    print("\n--- Diagnostics ---")
    print(f"Pred Mean: {preds.mean():.8f} | Pred Std: {preds.std():.8f}")
    print(f"Target Mean: {targets.mean():.8f} | Target Std: {targets.std():.8f}")
    print(f"Correlation (R): {correlation:.4f}")
    print("-------------------\n")
    
    return total_loss / val_steps

def main():
    parser = argparse.ArgumentParser(description="Train LOB sequence models.")
    parser.add_argument('--model', type=str, required=True, choices=['lstm', 'mamba', 'transformer'])
    args = parser.parse_args()

    config = load_config()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_cuda = device.type == 'cuda'
    
    train_loader, val_loader = get_dataloaders(config)
    
    sample_dataset = LOBDataset(config, is_train=True)
    sample_x, _ = sample_dataset[0]
    input_dim = sample_x.shape[-1]
    
    print(f"Detected institutional-grade input dimension: {input_dim}")
    
    model = get_model(args.model, input_dim, config).to(device)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda' if is_cuda else 'cpu', enabled=is_cuda)
    
    epochs = config['training']['epochs']
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    save_dir = "models/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting {args.model.upper()} on {device} | Epochs: {epochs} (Virtual)")
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, steps_per_epoch=30)
        val_loss = evaluate(model, val_loader, criterion, device, val_steps=200)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = f"{save_dir}/best_{args.model}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"--> Checkpoint saved: {ckpt_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered. Model has converged.")
                break

if __name__ == "__main__":
    main()