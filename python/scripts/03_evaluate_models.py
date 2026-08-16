import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.linear_model import Ridge
import warnings

from python.utils.config import load_config
from python.dataset import get_dataloaders, LOBDataset
from python.models.lstm import LSTMBaseline
from python.models.mamba import LOBMamba
from python.models.transformer import LOBTransformer

warnings.filterwarnings('ignore')

plt.rcParams.update({
    "figure.facecolor": "#F8F9FA",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#E9ECEF",
    "axes.labelcolor": "#495057",
    "text.color": "#212529",
    "grid.color": "#F1F3F5",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
})

def calculate_metrics(preds, targets):
    step_pnl = np.sign(preds) * targets
    cum_pnl = step_pnl.cumsum()
    cum_max = np.maximum.accumulate(cum_pnl)
    drawdown = cum_max - cum_pnl
    
    pearson_ic, _ = stats.pearsonr(preds, targets)
    rank_ic, _ = stats.spearmanr(preds, targets)
    hit_rate = (np.sign(preds) == np.sign(targets)).mean()
    
    sharpe = (step_pnl.mean() / (step_pnl.std() + 1e-8)) * np.sqrt(252 * 23400)
    mse = np.mean((preds - targets)**2)
    
    return cum_pnl, drawdown.max(), pearson_ic, rank_ic, hit_rate, sharpe, mse

def main():
    print("Initializing 4-Way Architecture Evaluation...")
    config = load_config()
    device = torch.device('cpu')
    train_loader, val_loader = get_dataloaders(config)
    
    # Dynamically detect feature dimension from the current dataset
    sample_dataset = LOBDataset(config, is_train=True)
    sample_x, _ = sample_dataset[0]
    input_dim = sample_x.shape[-1]
    print(f"Detected dataset input dimension: {input_dim}")

    # 1. Train the Linear Baseline (Ridge)
    print("Fitting Ridge Regression Baseline...")
    X_train_base, y_train_base = [], []
    for i, (x, y) in enumerate(train_loader):
        if i >= 25: break
        X_train_base.append(x.view(x.shape[0], -1).numpy())
        y_train_base.append(y.numpy())
    
    baseline = Ridge(alpha=1.0)
    baseline.fit(np.concatenate(X_train_base), np.concatenate(y_train_base))
    
    # 2. Load PyTorch Checkpoints
    print("Loading Neural Checkpoints...")
    
    lstm = LSTMBaseline(input_dim, config).to(device)
    lstm.load_state_dict(torch.load("models/checkpoints/best_lstm.pt", map_location=device, weights_only=True))
    lstm.eval()
    
    transformer = LOBTransformer(input_dim, config).to(device)
    transformer.load_state_dict(torch.load("models/checkpoints/best_transformer.pt", map_location=device, weights_only=True))
    transformer.eval()
    
    mamba = LOBMamba(input_dim, config).to(device)
    mamba.load_state_dict(torch.load("models/checkpoints/best_mamba.pt", map_location=device, weights_only=True))
    mamba.eval()
    
    # 3. Out-of-Sample Inference
    print("Running Out-of-Sample Inference...")
    preds = {'Ridge': [], 'LSTM': [], 'Transformer': [], 'Mamba': [], 'Targets': []}
    
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= 60: break
            
            x_dev = x.to(device)
            preds['Ridge'].append(baseline.predict(x.view(x.shape[0], -1).numpy()))
            preds['LSTM'].append(lstm(x_dev).squeeze().cpu().numpy())
            preds['Transformer'].append(transformer(x_dev.to(torch.float32)).squeeze().cpu().numpy())
            preds['Mamba'].append(mamba(x_dev).squeeze().cpu().numpy())
            preds['Targets'].append(y.cpu().numpy())
            
    for k in preds:
        preds[k] = np.concatenate(preds[k])
        
    targets = preds['Targets']
    
    # 4. Generate Metrics
    print("Calculating Metrics...")
    results = {}
    for model_name in ['Ridge', 'LSTM', 'Transformer', 'Mamba']:
        results[model_name] = calculate_metrics(preds[model_name], targets)
        
    # 5. Plot Tear Sheet
    print("Generating Tear Sheet...")
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Model Evaluation: Limit Order Book Return Forecasting", fontsize=22, fontweight='bold', y=0.95)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.2, 1], hspace=0.25)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(results['Ridge'][0], color="#ADB5BD", linewidth=1.5, linestyle="--", label=f"Ridge (IC: {results['Ridge'][2]:.4f})")
    ax1.plot(results['LSTM'][0], color="#F59F00", linewidth=2.0, alpha=0.8, label=f"LSTM (IC: {results['LSTM'][2]:.4f})")
    ax1.plot(results['Transformer'][0], color="#748FFC", linewidth=2.0, alpha=0.8, label=f"Transformer (IC: {results['Transformer'][2]:.4f})")
    ax1.plot(results['Mamba'][0], color="#2B8A3E", linewidth=3.0, label=f"Mamba SSM (IC: {results['Mamba'][2]:.4f})")
    
    ax1.set_title("Out-of-Sample Cumulative Return", fontsize=16, fontweight='600')
    ax1.legend(loc="upper left", frameon=True, fontsize=12)
    ax1.set_ylabel("Cumulative Normalized Return", fontsize=12)
    
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    
    table_data = [
        ["Metric", "Ridge Baseline", "LSTM", "Transformer", "Mamba SSM"],
        ["Pearson IC", f"{results['Ridge'][2]:.4f}", f"{results['LSTM'][2]:.4f}", f"{results['Transformer'][2]:.4f}", f"{results['Mamba'][2]:.4f}"],
        ["Rank IC", f"{results['Ridge'][3]:.4f}", f"{results['LSTM'][3]:.4f}", f"{results['Transformer'][3]:.4f}", f"{results['Mamba'][3]:.4f}"],
        ["Hit Rate", f"{results['Ridge'][4]:.2%}", f"{results['LSTM'][4]:.2%}", f"{results['Transformer'][4]:.2%}", f"{results['Mamba'][4]:.2%}"],
        ["Pseudo-Sharpe", f"{results['Ridge'][5]:.2f}", f"{results['LSTM'][5]:.2f}", f"{results['Transformer'][5]:.2f}", f"{results['Mamba'][5]:.2f}"],
        ["Mean Squared Error", f"{results['Ridge'][6]:.4f}", f"{results['LSTM'][6]:.4f}", f"{results['Transformer'][6]:.4f}", f"{results['Mamba'][6]:.4f}"],
        ["Max Drawdown", f"{results['Ridge'][1]:.2f}", f"{results['LSTM'][1]:.2f}", f"{results['Transformer'][1]:.2f}", f"{results['Mamba'][1]:.2f}"]
    ]
    
    table = ax2.table(cellText=table_data, loc='center', cellLoc='center')
    table.set_fontsize(14)
    table.scale(1, 2.5)
    
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('#DEE2E6')
        if key[0] == 0 or key[1] == 0: 
            cell.set_text_props(weight='bold', color='#495057')
            cell.set_facecolor('#F8F9FA')

    output_file = "model_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Success! Tear sheet saved to {output_file}")

if __name__ == "__main__":
    main()