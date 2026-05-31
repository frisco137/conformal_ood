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

from gp_pfn_ood.pfn_hooks import PFNHookManager
from gp_pfn_sm.ood_metric_study import generate_swept_sm_batch, compute_mahalanobis

# Matplotlib premium styling
plt.rcParams.update({
    'font.size': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white'
})

def compute_cosine_distance(X, mean_vector):
    """
    Computes the cosine distance of samples in X from the reference mean vector.
    """
    norm_X = np.linalg.norm(X, axis=1)
    norm_mean = np.linalg.norm(mean_vector)
    dot_product = np.dot(X, mean_vector)
    similarity = dot_product / (norm_X * norm_mean + 1e-8)
    return 1.0 - similarity

def compute_gini(w):
    """
    Computes the Gini coefficient (sparsity metric) for the last dimension of w.
    """
    w_sorted = np.sort(w, axis=-1)
    n = w.shape[-1]
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * w_sorted, axis=-1) / (n * np.sum(w_sorted, axis=-1) + 1e-8)) - (n + 1.0) / n
    return gini

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
    
    # Joint fitting set
    X_fit = np.concatenate([X_cal, X_ood1, X_ood2], axis=0) # (2000, 256)
    
    # Fit projection spaces
    print("Fitting PCA and UMAP spaces...")
    pca_10d = PCA(n_components=10, random_state=42)
    pca_10d.fit(X_fit)
    X_cal_pca_10d = pca_10d.transform(X_cal)
    pca_10d_mean = np.mean(X_cal_pca_10d, axis=0)
    pca_10d_cov = np.cov(X_cal_pca_10d, rowvar=False)
    
    pca_5d = PCA(n_components=5, random_state=42)
    pca_5d.fit(X_fit)
    X_cal_pca_5d = pca_5d.transform(X_cal)
    pca_5d_mean = np.mean(X_cal_pca_5d, axis=0)
    pca_5d_cov = np.cov(X_cal_pca_5d, rowvar=False)
    
    umap_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_2d.fit(X_fit)
    X_cal_umap = umap_2d.transform(X_cal)
    umap_mean = np.mean(X_cal_umap, axis=0)
    umap_cov = np.cov(X_cal_umap, rowvar=False)
    
    # Raw activation space reference statistics
    X_cal_mean = np.mean(X_cal, axis=0)
    
    # 2. Parametric Sweep
    freq_sweeps = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]
    
    metrics = {
        "umap_mahalanobis": [],
        "pca_5d_mahalanobis": [],
        "pca_10d_mahalanobis": [],
        "pca_5d_euclidean": [],
        "cosine_distance": [],
        "attention_entropy": [],
        "attention_kl": [],
        "attention_gini": [],
        "prediction_mse": []
    }
    
    print("\nExecuting indicator grid search sweep...")
    for f in tqdm(freq_sweeps, desc="Frequencies"):
        # Generate 100 test samples
        x_test, y_test = generate_swept_sm_batch(batch_size=100, freq_max=f, device="cpu")
        
        # A. Representation & Attention Hooks (single_eval_pos=100)
        with torch.no_grad():
            src_x = x_test.transpose(0, 1).to(device)
            src_y = y_test.transpose(0, 1).to(device)
            _ = hook_manager.model((src_x, src_y), single_eval_pos=100)
            
        layer_activations, layer_attention = hook_manager.get_aggregated_features()
        hook_manager.clear_cache()
        
        X_test_layer = layer_activations[5].numpy() # (100, 256)
        
        # Project
        X_test_umap = umap_2d.transform(X_test_layer)
        X_test_pca_5d = pca_5d.transform(X_test_layer)
        X_test_pca_10d = pca_10d.transform(X_test_layer)
        
        # 1. Mahalanobis UMAP 2D
        d_m_umap = compute_mahalanobis(X_test_umap, umap_mean, umap_cov)
        metrics["umap_mahalanobis"].append((float(np.mean(d_m_umap)), float(np.std(d_m_umap))))
        
        # 2. Mahalanobis PCA 5D
        d_m_pca_5d = compute_mahalanobis(X_test_pca_5d, pca_5d_mean, pca_5d_cov)
        metrics["pca_5d_mahalanobis"].append((float(np.mean(d_m_pca_5d)), float(np.std(d_m_pca_5d))))
        
        # 3. Mahalanobis PCA 10D
        d_m_pca_10d = compute_mahalanobis(X_test_pca_10d, pca_10d_mean, pca_10d_cov)
        metrics["pca_10d_mahalanobis"].append((float(np.mean(d_m_pca_10d)), float(np.std(d_m_pca_10d))))
        
        # 4. Euclidean PCA 5D
        d_e_pca_5d = np.linalg.norm(X_test_pca_5d - pca_5d_mean, axis=1)
        metrics["pca_5d_euclidean"].append((float(np.mean(d_e_pca_5d)), float(np.std(d_e_pca_5d))))
        
        # 5. Cosine Distance
        d_c_raw = compute_cosine_distance(X_test_layer, X_cal_mean)
        metrics["cosine_distance"].append((float(np.mean(d_c_raw)), float(np.std(d_c_raw))))
        
        # Attention calculations (Layer 5)
        # shape: (100, 100, 100)
        attn = layer_attention[5]
        eps = 1e-12
        entropy = -torch.sum(attn * torch.log(attn + eps), dim=-1) # (100, 100)
        mean_entropy = torch.mean(entropy, dim=-1).numpy() # (100,)
        
        # 6. Shannon Attention Entropy
        metrics["attention_entropy"].append((float(np.mean(mean_entropy)), float(np.std(mean_entropy))))
        
        # 7. KL-Divergence from Uniform
        kl_div = np.log(100) - mean_entropy
        metrics["attention_kl"].append((float(np.mean(kl_div)), float(np.std(kl_div))))
        
        # 8. Attention Sparsity (Gini)
        gini_coeff = compute_gini(attn.numpy()) # (100, 100)
        mean_gini = np.mean(gini_coeff, axis=-1) # (100,)
        metrics["attention_gini"].append((float(np.mean(mean_gini)), float(np.std(mean_gini))))
        
        # B. Prediction-based Indicators (In-Context MSE at eval_pos = 50)
        with torch.no_grad():
            logits = hook_manager.model((src_x, src_y), single_eval_pos=50) # (50, 100, 1)
            targets = src_y[50:].unsqueeze(-1)
            mse_per_batch = torch.mean((logits - targets) ** 2, dim=[0, 2]).cpu().numpy() # (100,)
            
        metrics["prediction_mse"].append((float(np.mean(mse_per_batch)), float(np.std(mse_per_batch))))
        
    hook_manager.remove_hooks()
    
    # Save results to JSON
    results_dir = "gp_pfn_sm/results"
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "ood_grid_search_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # 3. Plotting the 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), dpi=300)
    axes = axes.flatten()
    
    plot_configs = [
        ("umap_mahalanobis", "Mahalanobis (UMAP 2D)", "#4f46e5"),
        ("pca_5d_mahalanobis", "Mahalanobis (PCA 5D)", "#10b981"),
        ("pca_10d_mahalanobis", "Mahalanobis (PCA 10D)", "#f59e0b"),
        ("pca_5d_euclidean", "Euclidean (PCA 5D)", "#ec4899"),
        ("cosine_distance", "Cosine Distance (Raw)", "#06b6d4"),
        ("attention_entropy", "Shannon Attention Entropy (Nats)", "#8b5cf6"),
        ("attention_kl", "Attention KL-Div from Uniform", "#ef4444"),
        ("attention_gini", "Attention Sparsity (Gini Coeff)", "#6b7280"),
        ("prediction_mse", "In-Context MSE (Prediction Error)", "#3b82f6")
    ]
    
    for idx, (key, title, color) in enumerate(plot_configs):
        ax = axes[idx]
        stats = np.array(metrics[key])
        means = stats[:, 0]
        stds = stats[:, 1]
        
        ax.plot(freq_sweeps, means, marker='o', linewidth=2.0, color=color)
        ax.fill_between(freq_sweeps, means - stds, means + stds, color=color, alpha=0.15)
        
        # Highlight ID threshold
        ax.axvline(x=2.5, color='#ef4444', linestyle='--', linewidth=1.2)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel("Max Frequency Limit (f_max)", fontsize=9)
        
    plt.suptitle("PFN OOD indicator Grid Search across Frequency Shifts", fontweight='bold', fontsize=15, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "ood_indicator_grid_search.png"), bbox_inches='tight')
    plt.close()
    
    print("\nIndicator Grid Search completed successfully!")
    print(f"Plot saved in: {results_dir}/ood_indicator_grid_search.png")

if __name__ == "__main__":
    main()
