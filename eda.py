import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from merlin.builder import CircuitBuilder
from merlin import QuantumLayer, LexGrouping
from sklearn.preprocessing import MinMaxScaler

# Create plots directory
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

print("Loading dataset...")
ds_level1 = load_dataset(
    "Quandela/Challenge_Swaptions",
    data_files="level-1_Future_prediction/train.csv",
    split="train",
)
df_level1 = pd.DataFrame(ds_level1)

# Data cleaning
print("Cleaning data...")
missing_values = df_level1.isnull().sum()
print("Missing values in each column:")
print(missing_values[missing_values > 0])

df_level1['Date'] = pd.to_datetime(df_level1['Date'], dayfirst=True)
df_level1_sorted = df_level1.sort_values('Date')

# Check missing dates
min_date = df_level1_sorted['Date'].min()
max_date = df_level1_sorted['Date'].max()
complete_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
existing_dates = set(df_level1_sorted['Date'].dt.date)
complete_dates = set(complete_date_range.date)
missing_dates_df1 = sorted(complete_dates - existing_dates)
print(f"Total missing dates: {len(missing_dates_df1)}")

# Prepare data for modeling
df = df_level1.sort_values("Date").reset_index(drop=True)
X = df.drop(columns=["Date"]).apply(pd.to_numeric, errors="coerce")

# PCA pipeline
pca_pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("pca", PCA(n_components=4, random_state=0))
])
Z = pca_pipe.fit_transform(X)
Z_df = pd.DataFrame(Z, columns=[f"PC{i+1:02d}" for i in range(Z.shape[1])])
Z_df.insert(0, "Date", df["Date"].values)

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {Z_df.shape}")
explained = pca_pipe.named_steps["pca"].explained_variance_ratio_
print(f"Explained variance (sum): {explained.sum():.4f}")

# Supervised setup (predict next-day PCs)
df_z = Z_df.sort_values("Date").reset_index(drop=True)
pc_cols = [col for col in df_z.columns if col.startswith('PC')]
X = df_z[pc_cols].iloc[:-1].values
y = df_z[pc_cols].iloc[1:].values

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Prepare yield data for inverse transform
df_raw_sorted = df_level1.sort_values("Date").reset_index(drop=True)
yield_cols = [c for c in df_raw_sorted.columns if c != "Date"]
Y_raw = df_raw_sorted[["Date"] + yield_cols].copy()
Y_raw[yield_cols] = Y_raw[yield_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill()

n_components_fit = pca_pipe.named_steps["pca"].n_components_

def inverse_to_yields(y_pred_pc, test_dates):
    """Inverse PCA + scale predictions to yield space."""
    n_test = len(y_pred_pc)
    Z_pred_full = np.zeros((n_test, n_components_fit), dtype=float)
    Z_pred_full[:, :y_pred_pc.shape[1]] = y_pred_pc
    
    Y_pred_scaled = pca_pipe.named_steps["pca"].inverse_transform(Z_pred_full)
    Y_pred = pca_pipe.named_steps["scaler"].inverse_transform(Y_pred_scaled)
    
    Y_true = Y_raw.merge(test_dates.to_frame(name="Date"), on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    n_common = min(len(Y_true), Y_pred.shape[0])
    Y_true_vals = Y_true[yield_cols].iloc[:n_common].to_numpy()
    Y_pred_vals = Y_pred[:n_common, :]
    return Y_true_vals, Y_pred_vals, n_common

# Linear Regression
print("Training Linear Regression...")
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
test_dates = df_z["Date"].iloc[split+1 : split+1 + len(y_pred_lr)].reset_index(drop=True)
Y_true_vals, Y_pred_lr_vals, n_common = inverse_to_yields(y_pred_lr, test_dates)

mse_yields_lr = mean_squared_error(Y_true_vals, Y_pred_lr_vals)
print(f"Yield-space MSE (LR): {mse_yields_lr:.6f}")

col_idx = 5
dates_plot = test_dates.iloc[:n_common]

# MLP
print("Training MLP...")
n_pcs = len(pc_cols)
hidden_dim = 1024
lr_mlp = 1e-3
epochs = 200
batch_size = 64

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

model_mlp = MLP(n_pcs, hidden_dim, n_pcs)
optimizer = torch.optim.Adam(model_mlp.parameters(), lr=lr_mlp)
criterion = nn.MSELoss()

model_mlp.train()
for epoch in tqdm(range(epochs), desc="MLP"):
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model_mlp(xb), yb)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs} — Loss: {loss.item():.6f}")

model_mlp.eval()
with torch.no_grad():
    y_pred_mlp = model_mlp(X_test_t).numpy()

Y_true_mlp, Y_pred_mlp_vals, _ = inverse_to_yields(y_pred_mlp, test_dates)
mse_pc_mlp = mean_squared_error(y_test, y_pred_mlp)
print(f"PC-space MSE (MLP): {mse_pc_mlp:.6f}")

# Train plot (LR vs MLP)
y_pred_train_lr = model_lr.predict(X_train)
train_dates = df_z["Date"].iloc[1:split+1].reset_index(drop=True)
Y_true_train, Y_pred_train_lr_vals, n_common_train = inverse_to_yields(y_pred_train_lr, train_dates)
with torch.no_grad():
    y_pred_train_mlp = model_mlp(X_train_t).numpy()
Y_pred_train_mlp_vals = inverse_to_yields(y_pred_train_mlp, train_dates)[1][:n_common_train, :]

# Quantum (Merlin)
print("Training Quantum model...")
lr_q = 0.01
n_modes = max(6, n_pcs)
print(f"n_pcs={n_pcs} -> n_modes={n_modes}")

scaler_x = MinMaxScaler(feature_range=(0, np.pi))
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)
scaler_y = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = scaler_y.fit_transform(y_train)

X_train_q_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_q_t = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_q_t = torch.tensor(X_test_scaled, dtype=torch.float32)
train_loader_q = DataLoader(TensorDataset(X_train_q_t, y_train_q_t), batch_size=batch_size, shuffle=False)

def build_quantum_model(n_modes, output_dim, n_photons=2):
    builder = CircuitBuilder(n_modes=n_modes)
    builder.add_entangling_layer(trainable=True, name="U1")
    builder.add_angle_encoding(modes=list(range(n_modes)), name="input")
    builder.add_rotations(trainable=True, name="theta1")
    builder.add_superpositions(depth=1)
    quantum_core = QuantumLayer(input_size=n_modes, builder=builder, n_photons=n_photons, dtype=torch.float32)
    model = nn.Sequential(
        quantum_core,
        LexGrouping(quantum_core.output_size, n_modes),
        nn.Linear(n_modes, 32), nn.GELU(),
        nn.Linear(32, output_dim)
    )
    return model

model_quantum = build_quantum_model(n_modes, n_pcs)
optimizer_q = torch.optim.Adam(model_quantum.parameters(), lr=lr_q)
criterion_q = nn.MSELoss()
print(f"Quantum params: {sum(p.numel() for p in model_quantum.parameters()):,}")

model_quantum.train()
for epoch in tqdm(range(epochs), desc="Quantum"):
    for xb, yb in train_loader_q:
        if xb.shape[1] < n_modes:
            xb = torch.nn.functional.pad(xb, (0, n_modes - xb.shape[1]))
        optimizer_q.zero_grad()
        loss = criterion_q(model_quantum(xb), yb)
        loss.backward()
        optimizer_q.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs} — Loss: {loss.item():.6f}")

model_quantum.eval()
with torch.no_grad():
    X_test_q_pad = X_test_q_t if X_test_q_t.shape[1] >= n_modes else torch.nn.functional.pad(X_test_q_t, (0, n_modes - X_test_q_t.shape[1]))
    y_pred_scaled_q = model_quantum(X_test_q_pad).numpy()
    y_pred_q = scaler_y.inverse_transform(y_pred_scaled_q)

Y_pred_q_vals, _, n_common_q = inverse_to_yields(y_pred_q, test_dates)
mse_pc_q = mean_squared_error(y_test, y_pred_q)
print(f"PC-space MSE (Quantum): {mse_pc_q:.6f}")

# Final plot: All models
n_common_all = min(n_common, n_common_q)
plt.figure(figsize=(12, 4))
plt.plot(dates_plot[:n_common_all], Y_true_vals[:n_common_all, col_idx], label="Actual")
plt.plot(dates_plot[:n_common_all], Y_pred_lr_vals[:n_common_all, col_idx], label="LR", linestyle="--")
plt.plot(dates_plot[:n_common_all], Y_pred_mlp_vals[:n_common_all, col_idx], label="MLP", linestyle=":")
plt.plot(dates_plot[:n_common_all], Y_pred_q_vals[:n_common_all, col_idx], label="Quantum", linestyle="-.") 
plt.title(f"All Models Comparison")
plt.xlabel("Date")
plt.ylabel("Yield")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "plot4_all_models.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"All plots saved to {PLOTS_DIR}/")
