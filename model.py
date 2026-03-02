import os
import torch
import logging
import warnings
import numpy as np
import merlin as ML
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
from dataclasses import dataclass
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')


# =============================================================================
# Configuration container defining all experiment hyperparameters and runtime
# settings used throughout the full training and forecasting pipeline.
# =============================================================================
@dataclass
class Config:
    lr: float = 0.001
    epochs: int = 150
    batch_size: int = 32
    lag: int = 5
    n_ensemble: int = 5
    shots: int = 5000
    n_predict: int = 6


# =============================================================================
# Directory creation and logging configuration to ensure all outputs such as
# models, plots, and submission files are saved in a structured location.
# =============================================================================
SAVE_DIR = "swaptions_merlin_quantum"
os.makedirs(SAVE_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA PIPELINE
# This section is responsible for:
# 1. Loading the dataset from HuggingFace.
# 2. Sorting chronologically if needed.
# 3. Selecting numeric features only.
# 4. Scaling the data for model stability.
# 5. Creating lag-based supervised learning sequences.
# 6. Splitting into training and validation subsets.
# =============================================================================
def load_data(config):
    ds = load_dataset("Quandela/Challenge_Swaptions", split="train")
    df = pd.DataFrame(ds)

    if 'Date' in df.columns:
        df = df.sort_values('Date').reset_index(drop=True)

    num_cols = df.select_dtypes(np.number).columns.tolist()
    df_num = df[num_cols].ffill().fillna(0).values
    output_dim = df_num.shape[1]

    raw_scaler = RobustScaler().fit(df_num)
    data_scaled = raw_scaler.transform(df_num)

    Xseq, Yseq = [], []
    for i in range(config.lag, len(data_scaled)):
        Xseq.append(data_scaled[i - config.lag:i].flatten())
        Yseq.append(df_num[i])

    Xseq = np.array(Xseq)
    Yseq = np.array(Yseq)

    assert not np.isnan(Yseq).any(), "NaNs in Yseq!"

    split = int(0.8 * len(Xseq))
    return (
        Xseq[:split], Yseq[:split],
        Xseq[split:], Yseq[split:],
        data_scaled, df_num, num_cols, output_dim, raw_scaler
    )


# =============================================================================
# QUANTUM MODEL DEFINITION
# This neural network follows a hybrid architecture:
# - A classical encoder compresses high-dimensional lag features.
# - A photonic variational quantum circuit layer performs nonlinear mapping.
# - A classical decoder reconstructs predictions in original feature space.
# =============================================================================
class MerlinQuantumNet(nn.Module):

    def __init__(self, input_dim, output_dim, shots=5000):
        super().__init__()

        # Classical feature compression block
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.LayerNorm(16)
        )

        # Quantum photonic processing layer
        self.quantum = ML.QuantumLayer.simple(input_size=16, n_params=64)

        # Classical reconstruction and regression head
        self.decoder = nn.Sequential(
            nn.Linear(self.quantum.output_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        enc = self.encoder(x)

        # Rescale tanh output from [-1, 1] to [0, 1] for quantum state encoding
        enc = (enc + 1) / 2

        return self.decoder(self.quantum(enc))


# =============================================================================
# TRAINING PROCEDURE
# This function:
# - Fits robust scalers for inputs and targets.
# - Trains the hybrid quantum model using AdamW.
# - Applies gradient clipping for stability.
# - Evaluates on validation set.
# - Returns trained model, scalers, and validation MSE.
# =============================================================================
def train_model(Xtr, ytr, Xte, yte, config, output_dim):

    sx = RobustScaler()
    sy = RobustScaler()

    xtr_s = sx.fit_transform(Xtr)
    xte_s = sx.transform(Xte)
    ytr_s = sy.fit_transform(ytr)

    dl = DataLoader(
        TensorDataset(torch.FloatTensor(xtr_s), torch.FloatTensor(ytr_s)),
        batch_size=config.batch_size,
        shuffle=True
    )

    model = MerlinQuantumNet(Xtr.shape[1], output_dim, shots=config.shots)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=1e-4
    )

    crit = nn.MSELoss()

    model.train()
    for _ in tqdm(range(config.epochs), desc="Training", leave=False):
        for bx, by in dl:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()

            # Clip gradients to avoid instability from exploding updates
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()

    model.eval()
    with torch.no_grad():
        pred = sy.inverse_transform(
            model(torch.FloatTensor(xte_s)).numpy()
        )

    return model, sx, sy, mean_squared_error(yte, pred)


# =============================================================================
# AUTOREGRESSIVE FUTURE FORECASTING
# This function generates future rows sequentially by:
# - Using the last known lag window as context.
# - Predicting one step ahead.
# - Feeding the prediction back into the rolling context.
# - Repeating for n_predict steps.
# - Averaging across ensemble members.
# =============================================================================
def predict_next_6(models_scalers, data_scaled, raw_scaler, config):

    lag = config.lag
    all_preds = []

    for model, sx, sy in models_scalers:
        model.eval()

        # Initialize context with the final observed lag window
        ctx = list(data_scaled[-lag:])
        preds = []

        with torch.no_grad():
            for _ in range(config.n_predict):

                x_in = np.array(ctx[-lag:]).flatten().reshape(1, -1)
                x_in_s = sx.transform(x_in)

                pred_s = model(torch.FloatTensor(x_in_s)).numpy()
                pred_raw = sy.inverse_transform(pred_s)[0]

                preds.append(pred_raw)

                # Convert prediction back to scaled representation for next step
                ctx.append(
                    raw_scaler.transform(
                        pred_raw.reshape(1, -1)
                    )[0]
                )

        all_preds.append(np.array(preds))

    # Ensemble averaging improves robustness and reduces variance
    return np.mean(all_preds, axis=0)


# =============================================================================
# MAIN EXECUTION ENTRY POINT
# This section orchestrates:
# - Data loading
# - Ensemble training
# - Baseline comparison
# - Future prediction
# - Visualization
# - Full pipeline serialization
# =============================================================================
def main():

    config = Config()

    xtr, ytr, xte, yte, data_scaled, df_num, num_cols, output_dim, raw_scaler = \
        load_data(config)

    print(f"Dataset        : {df_num.shape[0]} rows x {output_dim} features")
    print(f"Train sequences: {xtr.shape[0]}   Val sequences: {xte.shape[0]}")
    print(f"Input dim      : {xtr.shape[1]} (lag={config.lag} x {output_dim})")

    models_scalers, mses = [], []

    # Train multiple models with different random seeds
    for i in range(config.n_ensemble):
        torch.manual_seed(i)

        model, sx, sy, mse = train_model(
            xtr, ytr, xte, yte, config, output_dim
        )

        models_scalers.append((model, sx, sy))
        mses.append(mse)

        print(f"Model {i+1}/{config.n_ensemble}  Val MSE: {mse:.6f}")

    print(f"\nEnsemble Val MSE : {np.mean(mses):.6f}")

    # Classical regression baselines for performance comparison
    ridge_mse = mean_squared_error(
        yte,
        Ridge(alpha=1.0).fit(xtr, ytr).predict(xte)
    )

    mlp_mse = mean_squared_error(
        yte,
        MLPRegressor(
            max_iter=500,
            hidden_layer_sizes=(64, 32)
        ).fit(xtr, ytr).predict(xte)
    )

    print(f"Ridge Val MSE    : {ridge_mse:.6f}")
    print(f"MLP   Val MSE    : {mlp_mse:.6f}")

    logger.info("Generating submission: 6 rows beyond dataset end...")

    predicted_6 = predict_next_6(
        models_scalers,
        data_scaled,
        raw_scaler,
        config
    )

    submission_df = pd.DataFrame(predicted_6, columns=num_cols)
    submission_df.index.name = "step"

    submission_path = os.path.join(
        SAVE_DIR,
        "submission_6_rows.csv"
    )

    submission_df.to_csv(submission_path)

    print(f"\n Submission saved : {submission_path}")
    print(f"   Shape            : {predicted_6.shape}  (6 rows × {output_dim} features)")

    # Visualization of last observed points and predicted extension
    n_show = 50
    n_cols_plot = min(6, output_dim)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx in range(n_cols_plot):
        ax = axes[idx]

        ax.plot(
            np.arange(n_show),
            df_num[-n_show:, idx],
            'steelblue',
            linewidth=2,
            label='Known data'
        )

        ax.plot(
            np.arange(n_show, n_show + 6),
            predicted_6[:, idx],
            'darkorange',
            linewidth=2,
            marker='o',
            linestyle='--',
            label='Predicted (6 rows)'
        )

        ax.axvline(n_show - 1, color='gray', linestyle=':', alpha=0.6)

        ax.set_title(f'{num_cols[idx]}', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        'Swaption Forecast: Next 6 Hidden Rows',
        fontsize=13,
        fontweight='bold'
    )

    plt.tight_layout()

    plot_path = os.path.join(
        SAVE_DIR,
        "quantum_swaptions_6rows.png"
    )

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f" Plot saved: {plot_path}")

    # Persist full training artifact for reproducibility
    torch.save({
        'models_state_dicts': [m.state_dict() for m, _, _ in models_scalers],
        'config': vars(config),
        'num_cols': num_cols,
        'output_dim': output_dim,
        'metrics': {
            'quantum_mse': float(np.mean(mses)),
            'ridge_mse': ridge_mse,
            'mlp_mse': mlp_mse
        },
        'predictions': predicted_6,
    }, os.path.join(SAVE_DIR, "MERLIN_QUANTUM_PIPELINE.pth"))

    logger.info("Pipeline saved.")


if __name__ == "__main__":
    main()
