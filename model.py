"""
Fixed PCA + Merlin Quantum Model for Swaptions.
Error fix: Force n_modes to valid integer (min 4), adjust n_photons compatibility.
Ensures GenericInterferometer receives integer mode count.
"""

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from merlin.builder import CircuitBuilder
from merlin import QuantumLayer, LexGrouping

def load_and_prepare_data():
    """Load numeric features from swaptions dataset."""
    ds = load_dataset("Quandela/Challenge_Swaptions")
    df = ds["train"].to_pandas()
    X = df.select_dtypes(include=[np.number]).values
    print(f"Dataset shape: {X.shape}")
    return X

def standardize_data(X):
    """Standardize data."""
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler

def find_pca_components(X_scaled, variance_threshold=0.99):
    """Find minimal PCs retaining target variance."""
    pca_full = PCA().fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_opt = np.argmax(cum_var >= variance_threshold) + 1
    
    # Ensure minimum modes for photonic circuit (Merlin requires >=2, recommend >=4)
    n_modes = max(n_opt, 4)
    print(f"Optimal PCs: {n_opt} -> n_modes clamped to {n_modes}")
    print(f"Variance ratios (first 5): {pca_full.explained_variance_ratio_[:5]}")
    return pca_full, n_modes  # Return clamped n_modes

def apply_pca(X_scaled, n_components):
    """Reduce dimensions."""
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)
    print(f"Reduced: {X_reduced.shape}, Variance retained: {pca.explained_variance_ratio_.sum():.4f}")
    return pca, X_reduced

def create_quantum_model(n_modes, num_output_features=5, n_photons=2):
    """
    Merlin photonic model with validated parameters.
    n_modes: PCA dimensions (clamped >=4)
    n_photons: Fixed to 2 (works reliably with small n_modes)
    """
    print(f"Building circuit: n_modes={n_modes}, n_photons={n_photons}")
    
    builder = CircuitBuilder(n_modes=n_modes)
    
    # Trainable entangling layer (requires >=2 modes)
    builder.add_entangling_layer(trainable=True, name="U1")
    
    # Data encoding on all modes
    builder.add_angle_encoding(modes=list(range(n_modes)), name="input")
    
    # Additional trainable rotations
    builder.add_rotations(trainable=True, name="theta")
    builder.add_superpositions(depth=1)
    
    # QuantumLayer with validated params
    quantum_core = QuantumLayer(
        input_size=n_modes,
        builder=builder,
        n_photons=n_photons,
        dtype=torch.float32
    )
    
    final_model = nn.Sequential(
        quantum_core,
        LexGrouping(quantum_core.output_size, num_output_features)
    )
    return final_model

if __name__ == "__main__":
    # PCA processing
    X = load_and_prepare_data()
    X_scaled, scaler = standardize_data(X)
    pca_full, n_modes = find_pca_components(X_scaled)
    pca, X_reduced = apply_pca(X_scaled, n_modes)
    
    # Create fixed model
    model = create_quantum_model(n_modes=n_modes)
    print(f"Model created successfully!")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 32
    dummy_input = torch.randn(batch_size, n_modes)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape} -> Output: {output.shape}")
    
    print("\nReady for training! Use model as torch.nn.Module.")
