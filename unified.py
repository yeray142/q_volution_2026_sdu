import os
import torch
import optuna
import logging
import warnings
import numpy as np
import merlin as ML
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from datasets import load_dataset
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor
from torch.utils.data import DataLoader, TensorDataset

# Suppress convergence and version warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class Config:
    """Hyperparameters and architectural settings for the Hybrid Quantum Model."""
    lr: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    n_pcs: int = 8           # Number of Principal Components to retain
    lag: int = 3             # Look-back window for time-series features
    n_ensemble: int = 5      # Number of models for ensemble averaging
    window_size: int = 40    # Rolling window for PCA fitting
    shots: int = 5000        # Number of quantum circuit executions per sample

# Setup workspace and logging
SAVE_DIR = "swaptions_merlin_quantum"
os.makedirs(SAVE_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# DATA PIPELINE
# ---------------------------------------------------------

def load_data(config):
    """
    Fetches Swaptions data and applies a financial feature engineering pipeline:
    1. Robust Scaling & PCA over rolling windows.
    2. Gaussian smoothing of latent features.
    3. Lagged sequence generation for time-series forecasting.
    """
    
    ds = load_dataset("Quandela/Challenge_Swaptions", split="train")
    '''
    ds = load_dataset(
        "Quandela/Challenge_Swaptions",
        data_files="level-2_Missing_data_prediction/train_level2.csv",
        split="train"
    )
    '''

    df = pd.DataFrame(ds)
    if 'Date' in df:
        df = df.sort_values('Date').reset_index(drop=True)

    # Clean numerical data
    num_cols = df.select_dtypes(np.number).columns
    df_num = df[num_cols].fillna(method='ffill').fillna(0)

    # Define features (first N columns) and targets (last 2: Put/Call prices)
    n_feat = config.n_pcs * 2
    X_cols = num_cols[:n_feat]
    y_cols = num_cols[-2:]
    X, Y = df_num[X_cols], df_num[y_cols]

    # --- Feature Engineering: Rolling PCA ---
    # Captures the local variance structure of the swaption surface
    embeds = []
    hwin = config.window_size // 2
    for i in range(hwin, len(X) - hwin):
        win = X.iloc[i-hwin:i+hwin].values
        sc = RobustScaler()
        pc = PCA(n_components=config.n_pcs)
        # Scale window, perform PCA, and take the latest representation
        embeds.append(pc.fit_transform(sc.fit_transform(win))[-1])

    # Smooth the latent representations to reduce market noise
    Z = gaussian_filter1d(np.array(embeds), 0.8, axis=0)

    # --- Sequence Creation ---
    # Flatten lagged time steps into a single feature vector
    ns = len(Z) - config.lag
    idim = config.lag * config.n_pcs
    Xseq = np.zeros((ns, idim))
    # Align Y targets with the end of the lag sequences
    Yseq = Y.iloc[config.window_size : config.window_size + ns].values

    for i in range(ns):
        Xseq[i] = Z[i : i + config.lag].flatten()

    split = int(0.8 * ns)
    return (Xseq[:split], Yseq[:split], Xseq[split:], Yseq[split:])

# ---------------------------------------------------------
# QUANTUM MODEL ARCHITECTURE
# ---------------------------------------------------------

class MerlinQuantumNet(nn.Module):
    """
    Hybrid Quantum-Classical Architecture:
    Classical Encoder -> Quantum Variational Circuit -> Classical Decoder
    """
    def __init__(self, input_dim, shots=5000, output_dim=2):
        super().__init__()

        # Classical Pre-processor: Compresses input to quantum-ready dimensions
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(), # Tanh maps values to [-1, 1], suitable for quantum encoding
            nn.LayerNorm(16)
        )

        # Quantum Layer: A Merlin-based Variational Quantum Circuit (VQC)
        self.quantum = ML.QuantumLayer.simple(
            input_size=16,
            n_params=64,
        )

        # Classical Post-processor: Maps quantum measurements to option prices
        self.decoder = nn.Sequential(
            nn.Linear(self.quantum.output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        # Shift Tanh output [-1, 1] to [0, 1] range for quantum state preparation
        encoded = (encoded + 1) / 2
        quantum_features = self.quantum(encoded)
        return self.decoder(quantum_features)

# ---------------------------------------------------------
# TRAINING & OPTIMIZATION
# ---------------------------------------------------------

def train_quantum_model(Xtr, ytr, Xte, yte, config, hp):
    """Standard PyTorch training loop with Robust Scaling for financial stability."""
    lr = hp.get('lr', config.lr)
    bs = hp.get('batch_size', config.batch_size)

    # Scale targets as well as features for better gradient behavior
    sx = RobustScaler()
    sy = RobustScaler()
    xtr_s = sx.fit_transform(Xtr)
    xte_s = sx.transform(Xte)
    ytr_s = sy.fit_transform(ytr)
    yte_s = sy.transform(yte)

    ds = TensorDataset(torch.FloatTensor(xtr_s), torch.FloatTensor(ytr_s))
    dl = DataLoader(ds, batch_size=bs, shuffle=True)

    model = MerlinQuantumNet(Xtr.shape[1], shots=hp.get('shots', config.shots))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.MSELoss()

    model.train()
    for epoch in tqdm(range(config.epochs), desc="Training"):
        for bx, by in dl:
            opt.zero_grad()
            pred = model(bx)
            loss = crit(pred, by)
            loss.backward()
            # Gradient clipping to prevent exploding gradients in quantum layers
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        tpred_s = model(torch.FloatTensor(xte_s)).numpy()
        tpred = sy.inverse_transform(tpred_s)

    mse = mean_squared_error(yte, tpred)
    return model, sy, mse

def objective(trial):
    """Optuna objective function for hyperparameter tuning (Ensemble of 2 for speed)."""
    config = Config()
    xtr, ytr, xte, yte = load_data(config)

    hp = {
        'lr': trial.suggest_float('lr', 5e-4, 5e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'shots': trial.suggest_categorical('shots', [1000, 5000, 10000])
    }

    # Cross-validation/Stability check with 2 different seeds
    models = []
    for seed in range(2):
        torch.manual_seed(seed)
        model, _, _ = train_quantum_model(xtr, ytr, xte, yte, config, hp)
        models.append(model)

    # Evaluate ensemble performance
    sx = RobustScaler().fit(np.vstack([xtr, xte]))
    xt = torch.FloatTensor(sx.transform(xte))

    epred = np.zeros((len(xte), 2))
    for m in models:
        m.eval()
        with torch.no_grad():
            epred += m(xt).numpy()
    epred /= 2

    sy = RobustScaler().fit(ytr)
    epred = sy.inverse_transform(epred)

    return mean_squared_error(yte, epred)

# ---------------------------------------------------------
# EXECUTION & EVALUATION
# ---------------------------------------------------------

def main():
    config = Config()
    xtr, ytr, xte, yte = load_data(config)

    # 1. Hyperparameter Optimization
    logger.info("Starting Hyperparameter Optimization...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10, show_progress_bar=True)

    # 2. Final Training with Ensemble
    best_hp = study.best_params
    final_config = Config(epochs=150, n_ensemble=5, **best_hp)
    logger.info(f"Training final ensemble with params: {best_hp}")

    models, scalers, mses = [], [], []
    for i in range(final_config.n_ensemble):
        model, scaler, mse = train_quantum_model(xtr, ytr, xte, yte, final_config, best_hp)
        models.append(model)
        scalers.append(scaler)
        mses.append(mse)

    # 3. Baseline Comparisons (Quantum vs. Classical)
    xtr_f = xtr.reshape(xtr.shape[0], -1)
    xte_f = xte.reshape(xte.shape[0], -1)
    
    ridge_mse = mean_squared_error(yte, Ridge(alpha=1.0).fit(xtr_f, ytr).predict(xte_f))
    mlp_mse = mean_squared_error(yte, MLPRegressor(max_iter=1000).fit(xtr_f, ytr).predict(xte_f))

    # 4. Visualization
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    names = ['Put Option Price', 'Call Option Price']

    for i in range(2):
        ax = axes[i]
        actual = yte[:, i]
        ensemble_pred = np.zeros(len(actual))
        
        # Aggregate ensemble predictions
        sx = RobustScaler().fit(xte)
        xt = torch.FloatTensor(sx.transform(xte))
        
        for model, scaler in zip(models, scalers):
            model.eval()
            pred_scaled = model(xt).detach().numpy()[:, i]
            dummy = np.zeros((len(pred_scaled), 2))
            dummy[:, i] = pred_scaled
            pred = scaler.inverse_transform(dummy)[:, i]
            ensemble_pred += pred

        ensemble_pred /= len(models)
        
        # Plot smoothed results for visual clarity
        n_plot = min(500, len(actual))
        actual_smooth = gaussian_filter1d(actual[:n_plot], sigma=1.2)
        pred_smooth = gaussian_filter1d(ensemble_pred[:n_plot], sigma=1.2)

        ax.plot(actual_smooth, 'steelblue', linewidth=3, label='Actual', alpha=0.9)
        ax.plot(pred_smooth, 'darkorange', linewidth=3, label='Merlin Quantum', linestyle='--', alpha=0.9)
        ax.set_title(names[i], fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, "quantum_swaptions.png")
    print("Saving figure to:", out_path)  # optional debug
    plt.savefig(out_path, dpi=300, bbox_inches='tight')

    # 5. Persist the entire pipeline
    pipeline = {
        'models_state_dict': [m.state_dict() for m in models],
        'scalers_state': [s.__getstate__() for s in scalers],
        'config': vars(final_config),
        'best_params': best_hp,
        'metrics': {'quantum': np.mean(mses), 'ridge': ridge_mse, 'mlp': mlp_mse}
    }
    pipeline_path = os.path.join(SAVE_DIR, "MERLIN_QUANTUM_PIPELINE.pth")
    torch.save(pipeline, pipeline_path)
    print(f"\n💾 SAVED: {pipeline_path}")
    logger.info("Pipeline saved successfully.")

if __name__ == "__main__":
    main()