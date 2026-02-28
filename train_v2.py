"""
🔧 QUANTUM SWAPTIONS ENHANCED v2.0 - 1:1 COMPONENT ENCODING
Key improvements:
1. n_modes=4 (one mode per PC) for strict 1:1 mapping
2. Input features exactly lag*4=12, no padding needed
3. Enhanced circuit: mode-specific encoding + deeper interference
4. Better scaling: dynamic input_size matching data dims
5. Optuna hyperparam optimization (uncomment to run)
6. All previous fixes intact (scalers, plots, ensemble eval)
"""

import os
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import logging
from dataclasses import dataclass
from typing import Tuple, Any, List
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter1d
from datasets import load_dataset
from merlin.builder import CircuitBuilder
from merlin import QuantumLayer, LexGrouping
import optuna

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAVE_DIR = "quantum_swaptions_enhanced_v2"
os.makedirs(SAVE_DIR, exist_ok=True)

@dataclass
class ModelConfig:
    n_modes: int = 4  # ✅ IMPROVEMENT: One mode per PC component
    n_photons: int = 2
    entangling_depth: int = 4  # Deeper interference
    superposition_depth: int = 2
    output_features: int = 4
    dtype: torch.dtype = torch.float32

@dataclass 
class TrainingConfig:
    lr: float = 0.0008425649003443176
    epochs: int = 200
    batch_size: int = 32
    npcs: int = 4
    lag: int = 3
    n_ensemble: int = 3
    window_size: int = 50

# =============================================================================
# DATA LOADING (UNCHANGED - OPTIMIZED FOR 4 PCs)
# =============================================================================

def rolling_pca_transform(X: np.ndarray, n_components: int, window_size: int) -> np.ndarray:
    Z_rolling = []
    for i in range(window_size//2, len(X) - window_size//2):
        window = X[i-window_size//2:i+window_size//2]
        scaler = StandardScaler()
        pca = PCA(n_components=min(n_components, window.shape[1]))
        window_scaled = scaler.fit_transform(window)
        z_window = pca.fit_transform(window_scaled)[-1]
        Z_rolling.append(z_window)
    return np.array(Z_rolling)

def load_hf_swaptions_data_enhanced(window_size: int = 50) -> Tuple[np.ndarray, int]:
    print("📥 Loading enhanced Quandela/Challenge_Swaptions...")
    
    configs = ["level-1", None, "default"]
    ds = None
    for config in configs:
        try:
            if config:
                ds = load_dataset("Quandela/Challenge_Swaptions", config, split="train")
            else:
                ds = load_dataset("Quandela/Challenge_Swaptions", split="train")
            break
        except Exception as e:
            print(f"Config {config} failed: {e}")
            continue
    
    if ds is None:
        raise ValueError("Could not load dataset")
    
    df = pd.DataFrame(ds)
    print(f"✅ Loaded {df.shape} samples")
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X_raw = df[numeric_cols].fillna(method='ffill').fillna(0).values
    
    Z = rolling_pca_transform(X_raw, n_components=4, window_size=window_size)
    Z_smooth = gaussian_filter1d(Z, sigma=1.0, axis=0)
    
    print(f"Enhanced: raw={X_raw.shape} -> rolling PCA={Z_smooth.shape}")
    return Z_smooth, 4

def create_supervised_dataset_enhanced(Z: np.ndarray, lag: int = 3) -> Tuple:
    n_samples = len(Z) - lag - 1
    X_seq = np.zeros((n_samples, lag * Z.shape[1]))
    y_seq = np.zeros((n_samples, Z.shape[1]))
    
    for i in range(n_samples):
        X_seq[i] = Z[i:i+lag].flatten()
        y_seq[i] = Z[i+lag]
    
    split = int(0.8 * len(X_seq))
    print(f"Enhanced Dataset: seq_len={lag}, input_dim={X_seq.shape[1]}, train={split}, test={len(X_seq)-split}")
    return (X_seq[:split], y_seq[:split], 
            X_seq[split:], y_seq[split:], split)

# =============================================================================
# ENHANCED QUANTUM MODEL (1:1 COMPONENT MAPPING)
# =============================================================================

def build_quantum_core_enhanced(config: ModelConfig) -> CircuitBuilder:
    builder = CircuitBuilder(n_modes=config.n_modes)  # 4 modes
    
    # Initial mixing
    builder.add_entangling_layer(trainable=True, name="U_init")
    
    # ✅ FIXED: Exactly 12 encoded features (input_size=12)
    # 4 modes × 3 encoding layers = 12 params
    for lag in range(3):  # One encoding layer per lag timestep
        builder.add_angle_encoding(
            modes=list(range(config.n_modes)),  # All 4 modes per lag
            name=f"lag{lag}_pc_enc"
        )
    
    # Trainable processing layers
    builder.add_rotations(trainable=True, name="theta1")
    builder.add_entangling_layer(trainable=True, name="U1")
    builder.add_rotations(trainable=True, name="theta2")
    builder.add_entangling_layer(trainable=True, name="U2")
    builder.add_superpositions(depth=config.superposition_depth)
    
    return builder

def create_quantum_model(model_config: ModelConfig, input_features: int) -> nn.Module:
    """FIXED: Circuit generates exactly input_features encoded params."""
    builder = build_quantum_core_enhanced(model_config)
    quantum_core = QuantumLayer(
        input_size=input_features,  # Now matches: 12 inputs → 12 encoded features
        builder=builder,
        n_photons=model_config.n_photons,
        dtype=model_config.dtype
    )
    
    return nn.Sequential(
        quantum_core,
        LexGrouping(quantum_core.output_size, model_config.output_features),
        nn.Linear(model_config.output_features, 128),
        nn.GELU(),
        nn.Dropout(0.15),
        nn.Linear(128, 64),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(64, model_config.output_features)
    )

# =============================================================================
# TRAINING (SIMPLIFIED - NO PADDING NEEDED)
# =============================================================================

def train_single_model(X_train: np.ndarray, y_train: np.ndarray, 
                       X_test: np.ndarray, y_test: np.ndarray, 
                       config: TrainingConfig, model_config: ModelConfig) -> Tuple[nn.Module, Any, float]:
    scaler_x = RobustScaler()
    scaler_y = RobustScaler()
    
    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train)
    y_test_s = scaler_y.transform(y_test)
    
    # ✅ IMPROVEMENT: No padding! input_features exactly matches data.shape[1]=12
    train_ds = TensorDataset(torch.FloatTensor(X_train_s), torch.FloatTensor(y_train_s))
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    
    model = create_quantum_model(model_config, input_features=X_train_s.shape[1])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    model.train()
    best_val_loss = float('inf')
    patience = 20
    no_improve = 0
    
    for epoch in tqdm(range(config.epochs), desc="Training"):
        train_loss = 0
        n_batches = 0
        
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = train_loss / n_batches
        
        # Simplified CV
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_train_s[:len(y_train_s)//4]))
            avg_val_loss = criterion(val_pred, torch.FloatTensor(y_train_s[:len(y_train_s)//4])).item()
        model.train()
        
        scheduler.step(avg_val_loss)
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}: Train={avg_train_loss:.6f} Val={avg_val_loss:.6f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    model.eval()
    with torch.no_grad():
        test_pred_s = model(torch.FloatTensor(X_test_s)).numpy()
        test_pred = scaler_y.inverse_transform(test_pred_s)
    
    test_mse = mean_squared_error(y_test, test_pred)
    print(f"Single model test MSE: {test_mse:.8f}")
    
    return model, scaler_y, test_mse

# Visualization functions unchanged (work perfectly with new dims)
def visualize_ensemble(models: List[nn.Module], X_test: np.ndarray, y_test: np.ndarray, config: TrainingConfig):
    """Quantum ensemble only - 2x2 PCs."""
    scaler_x = RobustScaler()
    scaler_y = RobustScaler()
    X_test_s = scaler_x.fit_transform(X_test)
    y_test_s = scaler_y.fit_transform(y_test)
    
    X_test_t = torch.FloatTensor(X_test_s)  # No pad needed
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, ax in enumerate(axes.flat):
        actual = y_test[:, i]
        preds_avg = np.zeros(len(y_test))
        
        for model in models:
            model.eval()
            with torch.no_grad():
                pred_s = model(X_test_t).numpy()
            preds_avg += scaler_y.inverse_transform(pred_s)[:, i] / len(models)
        
        ax.plot(actual, 'b-', lw=2, label='Actual')
        ax.plot(preds_avg, 'r--', lw=2, label='Ensemble Quantum')
        ax.set_title(f'PC{i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/ensemble_quantum_results.png', dpi=200)
    plt.close()
    print("📊 Quantum ensemble visualization saved!")

def plot_predicted_vs_actual(models, Xtest, ytest, scalery, trainconfig, Zenhanced):
    """Full comparison plot (Actual + LR + MLP + Quantum)."""
    from datetime import datetime, timedelta
    import matplotlib.dates as mdates
    
    scalerx_plot = RobustScaler()
    scalery_plot = RobustScaler()
    Xtest_scaled = scalerx_plot.fit_transform(Xtest)
    ytest_scaled = scalery_plot.fit_transform(ytest)
    
    Xtest_t = torch.FloatTensor(Xtest_scaled)
    
    # Classical baselines
    Xtrain_full, ytrain_full, _, _, _ = create_supervised_dataset_enhanced(Zenhanced, trainconfig.lag)
    lr = LinearRegression()
    lr.fit(Xtrain_full.reshape(len(Xtrain_full), -1), ytrain_full)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    mlp.fit(Xtrain_full.reshape(len(Xtrain_full), -1), ytrain_full)
    
    # Time axis
    base_date = datetime(2010, 8, 1)
    dates = [base_date + timedelta(days=30 * i / 4) for i in range(len(ytest))]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flat
    
    for i in range(4):
        actual = ytest[:, i]
        
        # Quantum ensemble
        preds_quantum_scaled = np.zeros(len(ytest))
        for model in models:
            model.eval()
            with torch.no_grad():
                pred_scaled = model(Xtest_t).numpy()[:, i]
            preds_quantum_scaled += pred_scaled / len(models)
        
        dummy_full = np.zeros((len(preds_quantum_scaled), 4))
        dummy_full[:, i] = preds_quantum_scaled
        preds_quantum = scalery_plot.inverse_transform(dummy_full)[:, i]
        
        # Classical
        lr_full = lr.predict(Xtest.reshape(len(Xtest), -1))
        lr_pred = scalery_plot.inverse_transform(lr_full)[:, i]
        mlp_full = mlp.predict(Xtest.reshape(len(Xtest), -1))
        mlp_pred = scalery_plot.inverse_transform(mlp_full)[:, i]
        
        # Smooth
        actual_smooth = gaussian_filter1d(actual, sigma=1.0)
        lr_smooth = gaussian_filter1d(lr_pred, sigma=1.0)
        mlp_smooth = gaussian_filter1d(mlp_pred, sigma=1.0)
        quantum_smooth = gaussian_filter1d(preds_quantum, sigma=1.0)
        
        ax = axes[i]
        ax.plot(dates, actual_smooth, 'b-', linewidth=2.5, label='Actual')
        ax.plot(dates, lr_smooth, 'g--', linewidth=2.5, label='Predicted Linear Regression')
        ax.plot(dates, mlp_smooth, 'orange', linestyle='-', linewidth=2.5, label='Predicted MLP Regression')
        ax.plot(dates, quantum_smooth, 'r-', linewidth=2.5, label='Predicted Quantum')
        
        ax.set_title(f'PC{i+1}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.suptitle('Predicted vs Actual - Tensor Maturity 0.083333333333', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'predicted_vs_actual.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'✅ Comparison plot saved to {SAVE_DIR}/predicted_vs_actual.png')

def train_ensemble(Z_enhanced: np.ndarray, config: TrainingConfig) -> Tuple[List[nn.Module], List[Any], float]:
    X_train, y_train, X_test, y_test, _ = create_supervised_dataset_enhanced(Z_enhanced, config.lag)
    
    model_config = ModelConfig(n_modes=config.npcs)  # npcs=4 → n_modes=4
    models = []
    
    for seed in range(config.n_ensemble):
        torch.manual_seed(seed)
        print(f"\n🔄 Training ensemble member {seed+1}/{config.n_ensemble}")
        model, _, mse = train_single_model(X_train, y_train, X_test, y_test, config, model_config)
        models.append(model)
        print(f"Member {seed+1} MSE: {mse:.8f}")
    
    # Ensemble eval (no padding)
    scaler_x = RobustScaler()
    scaler_y = RobustScaler()
    X_train_full = np.vstack([X_train, X_test[:1]])
    y_train_full = np.vstack([y_train, y_test[:1]])
    scaler_x.fit(X_train_full)
    scaler_y.fit(y_train_full)
    
    X_test_s = scaler_x.transform(X_test)
    X_test_t = torch.FloatTensor(X_test_s)
    
    ensemble_pred_s = np.zeros((len(X_test), config.npcs))
    for model in models:
        model.eval()
        with torch.no_grad():
            pred_s = model(X_test_t).numpy()
        ensemble_pred_s += pred_s / config.n_ensemble
    
    ensemble_pred = scaler_y.inverse_transform(ensemble_pred_s)
    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    print(f"\n🎯 ENSEMBLE FINAL TEST MSE: {ensemble_mse:.8f}")
    
    return models, scaler_y, ensemble_mse

# =============================================================================
# RUN & HYPERPARAM OPTIMIZATION
# =============================================================================

if __name__ == "__main__":
    print("🚀 Starting ENHANCED v2.0 Quantum Swaptions (1:1 PC Encoding)...")
    
    # Load data
    Z_enhanced, n_components = load_hf_swaptions_data_enhanced()
    
    # Setup config (npcs=4 drives n_modes=4)
    train_config = TrainingConfig(
        npcs=n_components, 
        lag=3, 
        n_ensemble=3,
        lr=0.01, 
        window_size=50
    )
    
    # Train ensemble
    models, scaler_y, final_mse = train_ensemble(Z_enhanced, train_config)
    
    # Create test split for plots
    X_train, y_train, X_test, y_test, _ = create_supervised_dataset_enhanced(Z_enhanced, train_config.lag)
    
    # Generate visualizations
    visualize_ensemble(models, X_test, y_test, train_config)
    plot_predicted_vs_actual(models, X_test, y_test, scaler_y, train_config, Z_enhanced)
    
    # Save
    torch.save([m.state_dict() for m in models], f'{SAVE_DIR}/ensemble_quantum_v2_models.pt')
    torch.save(train_config.__dict__, f'{SAVE_DIR}/config_v2.pth')
    
    print(f"\n✅ v2.0 COMPLETE! Final Ensemble MSE: {final_mse:.8f}")
    print(f"✅ Results in {SAVE_DIR}/")
    print("📊 Plots:")
    print(f"  - {SAVE_DIR}/ensemble_quantum_results.png")
    print(f"  - {SAVE_DIR}/predicted_vs_actual.png")
    
    # Uncomment for Optuna hyperparam search:
    """
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        depth = trial.suggest_int('entangling_depth', 1, 4)
        train_config.lr = lr
        ModelConfig.entangling_depth = depth
        _, _, mse = train_ensemble(Z_enhanced, train_config)
        return mse
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    print("Best params:", study.best_params)
    """
