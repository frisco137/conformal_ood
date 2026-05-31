import sys
import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
import umap

# Ensure parent directories are in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TransformersCanDoBayesianInference')))

import encoders
import positional_encodings
from transformer import TransformerModel
from gp_pfn_ood.pfn_hooks import PFNHookManager
from gp_pfn_sm.data_generators import spectral_mixture_kernel

# Matplotlib premium styling
plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white'
})

def generate_swept_sm_batch(batch_size, freq_max, num_points=100, device="cpu"):
    """
    Generates a batch of Spectral Mixture GP datasets where the high-frequency components
    are shifted into the range [freq_max - 2.5, freq_max].
    When freq_max = 2.5, this matches the In-Distribution (ID) range [0, 2.5] exactly.
    """
    # 1. Stratified sampling of X in [0, 1] with jitter
    bin_width = 1.0 / num_points
    grid_starts = torch.linspace(0.0, 1.0 - bin_width, num_points, device=device).view(1, num_points, 1)
    grid_starts = grid_starts.expand(batch_size, -1, -1)
    jitter = torch.rand(batch_size, num_points, 1, device=device) * bin_width * 0.6
    xs = grid_starts + jitter
    xs, _ = xs.sort(dim=1)
    
    # 2. Set up mixture components - 5 active components
    max_components = 5
    n_active = torch.ones((batch_size, 1), device=device, dtype=torch.long) * max_components
    idx = torch.arange(max_components, device=device).expand(batch_size, -1)
    mask = (idx < n_active).float()
    
    # Define shifted frequency range
    means = torch.rand(batch_size, max_components, device=device)
    if freq_max <= 2.5:
        # Standard ID distribution
        means[:, 0] = means[:, 0] * 0.2
        means[:, 1:] = means[:, 1:] * 2.5
    else:
        # Shifted OOD distribution (all 5 components in [freq_max - 2.5, freq_max])
        means = means * 2.5 + (freq_max - 2.5)
        
    scales = torch.rand(batch_size, max_components, device=device)
    if freq_max <= 2.5:
        scales[:, 0] = scales[:, 0] * 0.1 + 0.01
        scales[:, 1:] = scales[:, 1:] * 0.7 + 0.05
    else:
        scales = scales * 0.7 + 0.05
        
    weights = torch.rand(batch_size, max_components, device=device)
    weights = weights * mask 
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
    
    # 3. Compute Covariance Matrix K
    K = spectral_mixture_kernel(xs, xs, weights, means, scales)
    
    # Add noise & jitter for numerical stability
    noise_level = torch.rand(batch_size, 1, device=device) * 0.001
    K = K + (noise_level.unsqueeze(-1) ** 2 + 1e-5) * torch.eye(num_points, device=device).unsqueeze(0)
    
    # 4. Sample y ~ N(0, K)
    S, U = torch.linalg.eigh(K)
    S = torch.clamp(S, min=0.0)
    eps = torch.randn(batch_size, num_points, 1, device=device)
    ys = (U @ (S.sqrt().unsqueeze(-1) * eps)).squeeze(-1)
    
    # 5. Normalize targets
    ys = ys - ys.mean(dim=-1, keepdim=True)
    ys = ys / (ys.std(dim=-1, keepdim=True) + 1e-8)
    
    # 6. Apply random permutation
    perm = torch.randperm(num_points, device=device)
    xs = xs[:, perm, :]
    ys = ys[:, perm]
    
    return xs, ys

def compute_mahalanobis(coords, mean, cov):
    """
    Computes the Mahalanobis distance of coordinate samples from a reference distribution (mean, cov).
    """
    diff = coords - mean
    inv_cov = np.linalg.inv(cov)
    dist_sq = np.sum(diff @ inv_cov * diff, axis=1)
    return np.sqrt(dist_sq)

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device for inference: {device}")
    
    checkpoint_path = "gp_pfn_sm/checkpoints/spectral_mixture.pt"
    
    # 1. Generate Calibration Set (ID samples) and OOD reference sets for fitting
    print("Generating calibration set (1000 ID samples)...")
    x_cal, y_cal = generate_swept_sm_batch(batch_size=1000, freq_max=2.5, device="cpu")
    
    print("Generating OOD reference sets for joint PCA/UMAP fitting...")
    x_ood_ref1, y_ood_ref1 = generate_swept_sm_batch(batch_size=500, freq_max=7.5, device="cpu")
    x_ood_ref2, y_ood_ref2 = generate_swept_sm_batch(batch_size=500, freq_max=12.5, device="cpu")
    
    # Load model and hook manager
    hook_manager = PFNHookManager(checkpoint_path=checkpoint_path, device=device)
    
    # Extract calibration features from Layer 5
    print("Running PFN inference on calibration set...")
    with torch.no_grad():
        src_x = x_cal.transpose(0, 1).to(device)
        src_y = y_cal.transpose(0, 1).to(device)
        _ = hook_manager.model((src_x, src_y), single_eval_pos=100)
    layer_activations, _ = hook_manager.get_aggregated_features()
    X_cal = layer_activations[5].numpy() # (1000, 256)
    hook_manager.clear_cache()
    
    # Extract OOD reference features
    print("Running PFN inference on OOD reference sets...")
    with torch.no_grad():
        src_x1 = x_ood_ref1.transpose(0, 1).to(device)
        src_y1 = y_ood_ref1.transpose(0, 1).to(device)
        _ = hook_manager.model((src_x1, src_y1), single_eval_pos=100)
    layer_activations1, _ = hook_manager.get_aggregated_features()
    X_ood1 = layer_activations1[5].numpy()
    hook_manager.clear_cache()
    
    with torch.no_grad():
        src_x2 = x_ood_ref2.transpose(0, 1).to(device)
        src_y2 = y_ood_ref2.transpose(0, 1).to(device)
        _ = hook_manager.model((src_x2, src_y2), single_eval_pos=100)
    layer_activations2, _ = hook_manager.get_aggregated_features()
    X_ood2 = layer_activations2[5].numpy()
    hook_manager.clear_cache()
    
    # Combine into a joint fitting set
    X_fit = np.concatenate([X_cal, X_ood1, X_ood2], axis=0) # (2000, 256)
    
    # 2. Fit dimensionality reducers on joint ID + OOD features
    print("Fitting PCA and UMAP on joint ID + OOD features...")
    
    # PCA to 5D
    pca_5d = PCA(n_components=5, random_state=42)
    pca_5d.fit(X_fit)
    X_cal_pca_5d = pca_5d.transform(X_cal)
    pca_5d_mean = np.mean(X_cal_pca_5d, axis=0)
    pca_5d_cov = np.cov(X_cal_pca_5d, rowvar=False)
    
    # PCA to 2D
    pca_2d = PCA(n_components=2, random_state=42)
    pca_2d.fit(X_fit)
    X_cal_pca_2d = pca_2d.transform(X_cal)
    pca_2d_mean = np.mean(X_cal_pca_2d, axis=0)
    pca_2d_cov = np.cov(X_cal_pca_2d, rowvar=False)
    
    # UMAP to 2D
    umap_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_2d.fit(X_fit)
    X_cal_umap = umap_2d.transform(X_cal)
    umap_mean = np.mean(X_cal_umap, axis=0)
    umap_cov = np.cov(X_cal_umap, rowvar=False)
    
    # 3. Sweep the frequency parameter to evaluate OOD'ness
    freq_sweeps = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
    
    umap_mahalanobis_means = []
    umap_mahalanobis_stds = []
    pca_5d_mahalanobis_means = []
    pca_5d_mahalanobis_stds = []
    pca_2d_mahalanobis_means = []
    pca_2d_mahalanobis_stds = []
    
    print("\nStarting parametric frequency sweep...")
    for f in tqdm(freq_sweeps, desc="Sweeping Frequencies"):
        # Generate 100 test samples at this frequency limit
        x_test, y_test = generate_swept_sm_batch(batch_size=100, freq_max=f, device="cpu")
        
        # Inference
        with torch.no_grad():
            src_x = x_test.transpose(0, 1).to(device)
            src_y = y_test.transpose(0, 1).to(device)
            _ = hook_manager.model((src_x, src_y), single_eval_pos=100)
            
        layer_activations, _ = hook_manager.get_aggregated_features()
        hook_manager.clear_cache()
        
        X_test_layer = layer_activations[5].numpy() # (100, 256)
        
        # Project using PCA and UMAP
        X_test_pca_5d = pca_5d.transform(X_test_layer)
        X_test_pca_2d = pca_2d.transform(X_test_layer)
        X_test_umap = umap_2d.transform(X_test_layer)
        
        # Compute Mahalanobis distances
        d_m_umap = compute_mahalanobis(X_test_umap, umap_mean, umap_cov)
        d_m_pca_5d = compute_mahalanobis(X_test_pca_5d, pca_5d_mean, pca_5d_cov)
        d_m_pca_2d = compute_mahalanobis(X_test_pca_2d, pca_2d_mean, pca_2d_cov)
        
        umap_mahalanobis_means.append(float(np.mean(d_m_umap)))
        umap_mahalanobis_stds.append(float(np.std(d_m_umap)))
        pca_5d_mahalanobis_means.append(float(np.mean(d_m_pca_5d)))
        pca_5d_mahalanobis_stds.append(float(np.std(d_m_pca_5d)))
        pca_2d_mahalanobis_means.append(float(np.mean(d_m_pca_2d)))
        pca_2d_mahalanobis_stds.append(float(np.std(d_m_pca_2d)))
        
    hook_manager.remove_hooks()
    
    # 4. Save results to JSON
    results_dir = "gp_pfn_sm/results"
    os.makedirs(results_dir, exist_ok=True)
    summary_data = {
        "frequencies": freq_sweeps,
        "umap_2d": {
            "mean": umap_mahalanobis_means,
            "std": umap_mahalanobis_stds
        },
        "pca_5d": {
            "mean": pca_5d_mahalanobis_means,
            "std": pca_5d_mahalanobis_stds
        },
        "pca_2d": {
            "mean": pca_2d_mahalanobis_means,
            "std": pca_2d_mahalanobis_stds
        }
    }
    
    with open(os.path.join(results_dir, "ood_metric_sweep.json"), "w") as f:
        json.dump(summary_data, f, indent=4)
        
    # 5. Plotting results
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    # UMAP curve
    ax.plot(freq_sweeps, umap_mahalanobis_means, marker='o', linewidth=2.5, color='#4f46e5', label='UMAP 2D Mahalanobis Distance')
    ax.fill_between(
        freq_sweeps,
        np.array(umap_mahalanobis_means) - np.array(umap_mahalanobis_stds),
        np.array(umap_mahalanobis_means) + np.array(umap_mahalanobis_stds),
        color='#4f46e5', alpha=0.15
    )
    
    # PCA 5D curve
    ax.plot(freq_sweeps, pca_5d_mahalanobis_means, marker='s', linewidth=2.0, color='#10b981', label='PCA 5D Mahalanobis Distance')
    ax.fill_between(
        freq_sweeps,
        np.array(pca_5d_mahalanobis_means) - np.array(pca_5d_mahalanobis_stds),
        np.array(pca_5d_mahalanobis_means) + np.array(pca_5d_mahalanobis_stds),
        color='#10b981', alpha=0.15
    )

    # PCA 2D curve
    ax.plot(freq_sweeps, pca_2d_mahalanobis_means, marker='^', linewidth=2.0, color='#f59e0b', label='PCA 2D Mahalanobis Distance')
    ax.fill_between(
        freq_sweeps,
        np.array(pca_2d_mahalanobis_means) - np.array(pca_2d_mahalanobis_stds),
        np.array(pca_2d_mahalanobis_means) + np.array(pca_2d_mahalanobis_stds),
        color='#f59e0b', alpha=0.15
    )
    
    ax.axvline(x=2.5, color='#ef4444', linestyle='--', linewidth=1.5, label='In-Distribution Limit (2.5)')
    ax.set_xlabel('Maximum Frequency Limit (f_max)', fontweight='bold')
    ax.set_ylabel('OOD\'ness Metric (Mahalanobis Distance)', fontweight='bold')
    ax.set_title('OOD\'ness Metric vs. Frequency Deviation from Prior Boundary', fontweight='bold', fontsize=12, pad=12)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "ood_metric_sweep.png"), bbox_inches='tight')
    plt.close()
    
    print("\nOOD'ness metric sweep study completed successfully!")
    print(f"Results saved in: {results_dir}/")

if __name__ == "__main__":
    main()
