"""
Refactored PCA Analysis for Swaptions Dataset.
This script performs dimensionality reduction using PCA on the Quandela/Challenge_Swaptions dataset,
focusing on columns which represent swaption volatility surface data across different maturities and tenors.
The dataset contains high-dimensional features (likely ~50-100 'Tenor_X; Maturity_Y' columns with float64 swaption prices/vols).
Goal: Find minimal principal components retaining 99% variance for lower-dimensional QML encoding.
"""

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data():
    """
    Load the swaptions dataset and extract numeric features.
    
    The Quandela/Challenge_Swaptions dataset is a tabular dataset for quantum option pricing challenges.
    Columns are named 'Tenor : N; Maturity : M' (e.g., 'Tenor : 1; Maturity : 0.0833333333333333')
    containing float64 values.
    
    Returns:
        X: numpy array of shape (n_samples, n_features)
    """
    # Load dataset (train split assumed to contain main data)
    ds = load_dataset("Quandela/Challenge_Swaptions")
    df = ds["train"].to_pandas()
    
    # Identify feature columns: all 'Tenor...' numeric columns
    X = df.select_dtypes(include=[np.number]).values
    
    print(f"Dataset shape: {X.shape}")
    return X


def standardize_data(X):
    """Standardize features to zero mean and unit variance (required for PCA)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def find_pca_components(X_scaled, variance_threshold=0.99):
    """
    Fit full PCA and determine optimal n_components for target cumulative variance.
    
    - Computes explained variance ratios.
    - Finds minimal n where cumsum(ratios) >= threshold.
    - Also computes Kaiser criterion: components with >1% individual variance.
    
    Args:
        X_scaled: standardized data
        variance_threshold: target cumulative explained variance (default 99%)
    
    Returns:
        pca_full: full PCA model
        n_opt: optimal components for threshold
        kaiser_n: Kaiser rule components
        cum_var: cumulative variance ratios
    """
    pca_full = PCA().fit(X_scaled)  # Fit on all components
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Optimal n for threshold
    n_opt = np.argmax(cum_var >= variance_threshold) + 1
    
    # Kaiser criterion: eigenvalues >1 (equiv. variance ratio >1/n_features, approx >0.01 for large dim)
    kaiser_n = np.sum(pca_full.explained_variance_ratio_ > 0.01)
    
    print(f"Optimal components for {variance_threshold*100}% fidelity: {n_opt}")
    print(f"Kaiser (>1% each): {kaiser_n}")
    print(f"First 5 ratios: {pca_full.explained_variance_ratio_[:5]}")
    
    return pca_full, n_opt, kaiser_n, cum_var


def apply_pca(X_scaled, n_components):
    """Apply PCA dimensionality reduction."""
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    
    total_var = pca.explained_variance_ratio_.sum()
    print(f"Reduced shape: {X_reduced.shape}")
    print(f"Variance explained: {total_var:.4f}")
    
    return pca, X_reduced


def compute_reconstruction_error(pca, X_scaled, X_reduced):
    """Compute MSE between original scaled data and PCA reconstruction."""
    X_recon = pca.inverse_transform(X_reduced)
    mse = np.mean((X_scaled - X_recon)**2)
    print(f"Mean squared reconstruction error: {mse:.6f}")
    return mse


def plot_pca_analysis(pca_full, cum_var, n_opt, kaiser_n, max_comps_plot=50):
    """
    Generate and save PCA visualization plots.
    
    - Scree plot: cumulative variance with 99% threshold.
    - Detailed: bar/cumulative for first 20, methods comparison up to max_comps_plot.
    
    Saves:
        pca_scree_plot.png
        enhanced_pca_analysis.png
    """
    # Plot 1: Basic scree plot
    plt.figure(figsize=(8, 5))
    plt.plot(cum_var[:n_opt*2], 'bo-')
    plt.axvline(n_opt, color='r', linestyle='--', label=f'n={n_opt} (99%)')
    plt.axhline(0.99, color='g', linestyle='--')
    plt.xlabel('Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Scree Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig('pca_scree_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Enhanced analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Detailed first 20 components
    ax1.bar(range(1, 21), pca_full.explained_variance_ratio_[:20], alpha=0.6, label='Individual')
    ax1.plot(range(1, 21), cum_var[:20], 'ro-', label='Cumulative')
    ax1.axhline(0.99, color='g', linestyle='--', label='99% threshold')
    ax1.set_xlabel('Components')
    ax1.set_ylabel('Explained Variance')
    ax1.legend()
    ax1.set_title('Detailed Scree Plot')
    
    # Right: Methods comparison
    ax2.plot(cum_var[:max_comps_plot])
    ax2.axvline(n_opt, color='r', label=f'99%: {n_opt}')
    ax2.axvline(kaiser_n, color='orange', label=f'Kaiser: {kaiser_n}')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Methods Comparison')
    
    plt.tight_layout()
    plt.savefig('enhanced_pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    """Main workflow: data prep -> PCA analysis -> plots -> metrics."""
    # Data loading and preprocessing
    X = load_and_prepare_data()
    X_scaled, scaler = standardize_data(X)
    
    # PCA analysis
    pca_full, n_opt, kaiser_n, cum_var = find_pca_components(X_scaled)
    pca_reduced, X_reduced = apply_pca(X_scaled, n_opt)
    
    # Error metric
    mse = compute_reconstruction_error(pca_reduced, X_scaled, X_reduced)
    
    # Visualizations
    plot_pca_analysis(pca_full, cum_var, n_opt, kaiser_n)
    
    print(f"Recommended dimensions: {n_opt} PCs (99% variance, MSE={mse:.2e})")
