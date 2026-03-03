import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

SAVE_DIR  = "swaptions_merlin_quantum"
os.makedirs(SAVE_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PTH_PATH  = os.path.join(SAVE_DIR, "MERLIN_QUANTUM_PIPELINE.pth")
XLSX_PATH = "test.xlsx"


def evaluate_on_test(pth_path, xlsx_path):
    import torch

    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"No checkpoint found at: {pth_path}")

    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
    logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

    # The training script already ran predict_next_6() with the correct
    # training-data context and saved the result.  Use it directly.
    predictions = np.array(checkpoint['predictions'])   # shape (6, 224)
    num_cols    = checkpoint['num_cols']
    output_dim  = checkpoint['output_dim']

    logger.info(f"Predictions shape from checkpoint: {predictions.shape}")
    logger.info(f"Loading test data from: {xlsx_path}")

    df = pd.read_excel(xlsx_path)

    if 'Date' in df.columns:
        dates = df.sort_values('Date')['Date'].values
        df    = df.sort_values('Date').reset_index(drop=True)
    else:
        dates = np.arange(len(df))

    # Align columns to training feature order
    available = [c for c in num_cols if c in df.columns]
    missing   = [c for c in num_cols if c not in df.columns]
    if missing:
        logger.warning(f"{len(missing)} training columns missing from xlsx — filling with 0.")

    df_aligned = df[available].ffill().fillna(0).copy()
    for col in missing:
        df_aligned[col] = 0.0

    actuals = df_aligned[num_cols].values   # shape (6, 224)

    if len(actuals) != len(predictions):
        raise ValueError(
            f"Row count mismatch: checkpoint has {len(predictions)} predictions "
            f"but test.xlsx has {len(actuals)} rows."
        )

    # ── Metrics ────────────────────────────────────────────────────────────────
    mse_total   = mean_squared_error(actuals, predictions)
    rmse_total  = np.sqrt(mse_total)
    mse_per_col = np.mean((actuals - predictions) ** 2, axis=0)

    metrics_df = pd.DataFrame({
        'feature': num_cols,
        'mse':     mse_per_col,
        'rmse':    np.sqrt(mse_per_col),
    }).sort_values('mse', ascending=False).reset_index(drop=True)

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    preds_df          = pd.DataFrame(predictions, columns=num_cols)
    actuals_df        = pd.DataFrame(actuals,     columns=num_cols)
    preds_df['Date']   = dates
    actuals_df['Date'] = dates
    preds_df.index.name   = 'step'
    actuals_df.index.name = 'step'

    preds_path   = os.path.join(SAVE_DIR, "test_predictions.csv")
    actuals_path = os.path.join(SAVE_DIR, "test_actuals.csv")
    metrics_path = os.path.join(SAVE_DIR, "test_metrics.csv")

    preds_df.to_csv(preds_path)
    actuals_df.to_csv(actuals_path)
    metrics_df.to_csv(metrics_path, index=False)

    # ── Console summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" TEST RESULTS  ({len(actuals)} rows × {output_dim} features)")
    print(f"{'='*60}")
    print(f" Overall MSE  : {mse_total:.6f}")
    print(f" Overall RMSE : {rmse_total:.6f}")
    print(f"\n Top-5 highest-error features:")
    print(metrics_df.head(5).to_string(index=False))
    print(f"\n Predictions : {preds_path}")
    print(f" Actuals     : {actuals_path}")
    print(f" Metrics     : {metrics_path}")
    print(f"{'='*60}")

    # ── Plot: actual vs predicted for up to 6 features ─────────────────────────
    n_cols_plot = min(6, output_dim)
    fig, axes   = plt.subplots(2, 3, figsize=(15, 8))
    axes        = axes.flatten()
    x           = np.arange(len(actuals))

    for idx in range(n_cols_plot):
        ax = axes[idx]
        ax.plot(x, actuals[:, idx],     'steelblue',  lw=2, marker='o', label='Actual')
        ax.plot(x, predictions[:, idx], 'darkorange', lw=2, marker='s', ls='--', label='Predicted')
        ax.set_title(num_cols[idx], fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Test Set: Actual vs Predicted | MSE={mse_total:.5f}',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    plot_path = os.path.join(SAVE_DIR, "test_actual_vs_predicted.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f" Plot : {plot_path}")

    return mse_total, metrics_df


if __name__ == "__main__":
    evaluate_on_test(PTH_PATH, XLSX_PATH)