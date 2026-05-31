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
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import umap

# Ensure parent directories are in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TransformersCanDoBayesianInference')))

from gp_pfn_ood.pfn_hooks import PFNHookManager
from gp_pfn_sm.ood_metric_study import compute_mahalanobis
from gp_pfn_sm.test_suites import (
    generate_id_low_noise,
    generate_id_high_noise,
    generate_near_ood,
    generate_far_ood,
    generate_real_world_ood
)

# Matplotlib premium styling
plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white'
})

def compute_fpr_at_95_tpr(labels, scores):
    """
    Computes False Positive Rate at 95% True Positive Rate.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # Find the threshold index where tpr >= 0.95
    idx = np.where(tpr >= 0.95)[0][0]
    return float(fpr[idx])

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device for inference: {device}")
    
    checkpoint_path = "gp_pfn_sm/checkpoints/spectral_mixture.pt"
    hook_manager = PFNHookManager(checkpoint_path=checkpoint_path, device=device)
    
    # 1. Generate Calibration and Reference fitting sets (N = 1000 each)
    print("Generating calibration (ID) and OOD reference sets for fitting...")
    x_cal, y_cal = generate_id_low_noise(batch_size=1000, device="cpu")
    x_near_ref, y_near_ref = generate_near_ood(batch_size=500, device="cpu")
    x_far_ref, y_far_ref = generate_far_ood(batch_size=500, device="cpu")
    
    # Run forward pass to extract calibration activations
    print("Extracting calibration activations...")
    with torch.no_grad():
        src_x = x_cal.transpose(0, 1).to(device)
        src_y = y_cal.transpose(0, 1).to(device)
        _ = hook_manager.model((src_x, src_y), single_eval_pos=100)
    layer_activations, _ = hook_manager.get_aggregated_features()
    X_cal = layer_activations[5].numpy()
    hook_manager.clear_cache()
    
    # Extract reference activations
    print("Extracting near-OOD and far-OOD reference activations...")
    with torch.no_grad():
        src_x = x_near_ref.transpose(0, 1).to(device)
        src_y = y_near_ref.transpose(0, 1).to(device)
        _ = hook_manager.model((src_x, src_y), single_eval_pos=100)
    layer_activations, _ = hook_manager.get_aggregated_features()
    X_near_ref = layer_activations[5].numpy()
    hook_manager.clear_cache()
    
    with torch.no_grad():
        src_x = x_far_ref.transpose(0, 1).to(device)
        src_y = y_far_ref.transpose(0, 1).to(device)
        _ = hook_manager.model((src_x, src_y), single_eval_pos=100)
    layer_activations, _ = hook_manager.get_aggregated_features()
    X_far_ref = layer_activations[5].numpy()
    hook_manager.clear_cache()
    
    # Combine into a joint fitting set
    X_fit = np.concatenate([X_cal, X_near_ref, X_far_ref], axis=0) # (2000, 256)
    
    # Fit projection models
    print("Fitting PCA and UMAP models...")
    pca_10d = PCA(n_components=10, random_state=42)
    pca_10d.fit(X_fit)
    X_cal_pca_10d = pca_10d.transform(X_cal)
    pca_10d_mean = np.mean(X_cal_pca_10d, axis=0)
    pca_10d_cov = np.cov(X_cal_pca_10d, rowvar=False)
    
    umap_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    umap_2d.fit(X_fit)
    X_cal_umap = umap_2d.transform(X_cal)
    umap_mean = np.mean(X_cal_umap, axis=0)
    umap_cov = np.cov(X_cal_umap, rowvar=False)
    
    # 2. Generate and process the 5 Test Suites (N = 1000 each)
    test_suites = {
        "Suite 1 (ID Low Noise)": generate_id_low_noise,
        "Suite 2 (ID High Noise)": generate_id_high_noise,
        "Suite 3 (Near-OOD)": generate_near_ood,
        "Suite 4 (Far-OOD)": generate_far_ood,
        "Suite 5 (Real-World OOD)": generate_real_world_ood
    }
    
    suite_data = {}
    
    print("\nProcessing test suites...")
    for suite_name, generator_fn in test_suites.items():
        print(f"Processing {suite_name}...")
        x_test, y_test = generator_fn(batch_size=1000, device="cpu")
        
        # A. Representational metrics (single_eval_pos=100)
        with torch.no_grad():
            src_x = x_test.transpose(0, 1).to(device)
            src_y = y_test.transpose(0, 1).to(device)
            _ = hook_manager.model((src_x, src_y), single_eval_pos=100)
            
        layer_activations, _ = hook_manager.get_aggregated_features()
        hook_manager.clear_cache()
        
        X_test_layer = layer_activations[5].numpy()
        X_test_umap = umap_2d.transform(X_test_layer)
        X_test_pca_10d = pca_10d.transform(X_test_layer)
        
        d_m_umap = compute_mahalanobis(X_test_umap, umap_mean, umap_cov)
        d_m_pca_10d = compute_mahalanobis(X_test_pca_10d, pca_10d_mean, pca_10d_cov)
        
        # B. In-Context Prediction error (single_eval_pos=50)
        with torch.no_grad():
            logits = hook_manager.model((src_x, src_y), single_eval_pos=50) # (50, 1000, 1)
            targets = src_y[50:].unsqueeze(-1)
            mse_per_batch = torch.mean((logits - targets) ** 2, dim=[0, 2]).cpu().numpy()
        hook_manager.clear_cache()
            
        suite_data[suite_name] = {
            "umap_coords": X_test_umap.tolist(),
            "umap_mahalanobis": d_m_umap.tolist(),
            "pca_mahalanobis": d_m_pca_10d.tolist(),
            "prediction_mse": mse_per_batch.tolist()
        }
        
    hook_manager.remove_hooks()
    
    # 3. Compute ROC & PR statistics
    # Standard Negatives: Suite 1 (ID Low Noise)
    neg_umap = np.array(suite_data["Suite 1 (ID Low Noise)"]["umap_mahalanobis"])
    neg_pca = np.array(suite_data["Suite 1 (ID Low Noise)"]["pca_mahalanobis"])
    neg_mse = np.array(suite_data["Suite 1 (ID Low Noise)"]["prediction_mse"])
    
    evaluation_results = {}
    
    pos_suites = [
        ("Suite 2 (ID High Noise)", "Robustness to Noise"),
        ("Suite 3 (Near-OOD)", "Near-OOD Detection"),
        ("Suite 4 (Far-OOD)", "Far-OOD Detection"),
        ("Suite 5 (Real-World OOD)", "Real-World OOD Detection")
    ]
    
    for suite_name, display_name in pos_suites:
        pos_umap = np.array(suite_data[suite_name]["umap_mahalanobis"])
        pos_pca = np.array(suite_data[suite_name]["pca_mahalanobis"])
        pos_mse = np.array(suite_data[suite_name]["prediction_mse"])
        
        # Debug printing
        print(f"DEBUG {display_name}: neg_umap={len(neg_umap)}, pos_umap={len(pos_umap)}, neg_pca={len(neg_pca)}, pos_pca={len(pos_pca)}, neg_mse={len(neg_mse)}, pos_mse={len(pos_mse)}")
        
        # Labels: 0 for negative, 1 for positive
        labels = np.concatenate([np.zeros_like(neg_umap), np.ones_like(pos_umap)])
        
        # UMAP
        scores_umap = np.concatenate([neg_umap, pos_umap])
        fpr_u, tpr_u, _ = roc_curve(labels, scores_umap)
        auc_roc_u = auc(fpr_u, tpr_u)
        prec_u, rec_u, _ = precision_recall_curve(labels, scores_umap)
        auc_pr_u = auc(rec_u, prec_u)
        fpr95_u = compute_fpr_at_95_tpr(labels, scores_umap)
        
        # PCA 10D
        scores_pca = np.concatenate([neg_pca, pos_pca])
        fpr_p, tpr_p, _ = roc_curve(labels, scores_pca)
        auc_roc_p = auc(fpr_p, tpr_p)
        prec_p, rec_p, _ = precision_recall_curve(labels, scores_pca)
        auc_pr_p = auc(rec_p, prec_p)
        fpr95_p = compute_fpr_at_95_tpr(labels, scores_pca)
        
        # Prediction MSE
        scores_mse = np.concatenate([neg_mse, pos_mse])
        fpr_m, tpr_m, _ = roc_curve(labels, scores_mse)
        auc_roc_m = auc(fpr_m, tpr_m)
        prec_m, rec_m, _ = precision_recall_curve(labels, scores_mse)
        auc_pr_m = auc(rec_m, prec_m)
        fpr95_m = compute_fpr_at_95_tpr(labels, scores_mse)
        
        evaluation_results[display_name] = {
            "umap_2d": {"auc_roc": float(auc_roc_u), "auc_pr": float(auc_pr_u), "fpr_at_95_tpr": float(fpr95_u)},
            "pca_10d": {"auc_roc": float(auc_roc_p), "auc_pr": float(auc_pr_p), "fpr_at_95_tpr": float(fpr95_p)},
            "prediction_mse": {"auc_roc": float(auc_roc_m), "auc_pr": float(auc_pr_m), "fpr_at_95_tpr": float(fpr95_m)}
        }
        
    results_dir = "gp_pfn_sm/results"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "formal_experiment_metrics.json"), "w") as f:
        json.dump(evaluation_results, f, indent=4)
        
    # 4. Generate Figures
    # Figure 1: Manifold Separation Scatter Plot (UMAP)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes = axes.flatten()
    
    # ID UMAP coordinates
    cal_coords = np.array(suite_data["Suite 1 (ID Low Noise)"]["umap_coords"])
    
    comparisons = [
        ("Suite 2 (ID High Noise)", "#10b981", "ID High Noise", axes[0]),
        ("Suite 3 (Near-OOD)", "#f59e0b", "Near-OOD Frequency Shift", axes[1]),
        ("Suite 4 (Far-OOD)", "#ec4899", "Far-OOD Kernel Shift", axes[2]),
        ("Suite 5 (Real-World OOD)", "#8b5cf6", "Real-World OOD", axes[3])
    ]
    
    for suite_name, color, label, ax in comparisons:
        coords = np.array(suite_data[suite_name]["umap_coords"])
        
        ax.scatter(cal_coords[:, 0], cal_coords[:, 1], color='#3b82f6', alpha=0.4, s=6, label="ID Low Noise")
        ax.scatter(coords[:, 0], coords[:, 1], color=color, alpha=0.5, s=6, label=label)
        ax.set_title(f"UMAP Space: ID vs. {label}", fontweight='bold', fontsize=11)
        ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=8)
        
    plt.suptitle("Manifold Separation in Joint-Fitted UMAP Space", fontweight='bold', fontsize=15, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig1_manifold_separation.png"), bbox_inches='tight')
    plt.close()
    
    # Figure 2: ROC Curves comparison (Grid of 2x2 for each OOD task)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes = axes.flatten()
    
    roc_tasks = [
        ("Robustness to Noise", axes[0], "ID Low Noise vs. ID High Noise (Ideal AUC ~0.5)"),
        ("Near-OOD Detection", axes[1], "ID Low Noise vs. Near-OOD"),
        ("Far-OOD Detection", axes[2], "ID Low Noise vs. Far-OOD"),
        ("Real-World OOD Detection", axes[3], "ID Low Noise vs. Real-World OOD")
    ]
    
    for idx, (display_name, ax, title) in enumerate(roc_tasks):
        # We need to recreate the curves for plotting
        suite_key = [k for k, disp in pos_suites if disp == display_name][0]
        pos_umap = np.array(suite_data[suite_key]["umap_mahalanobis"])
        pos_pca = np.array(suite_data[suite_key]["pca_mahalanobis"])
        pos_mse = np.array(suite_data[suite_key]["prediction_mse"])
        
        labels = np.concatenate([np.zeros_like(neg_umap), np.ones_like(pos_umap)])
        
        # UMAP
        fpr_u, tpr_u, _ = roc_curve(labels, np.concatenate([neg_umap, pos_umap]))
        ax.plot(fpr_u, tpr_u, color='#4f46e5', linewidth=2.0, label=f"UMAP 2D Mahalanobis (AUC = {auc(fpr_u, tpr_u):.3f})")
        
        # PCA 10D
        fpr_p, tpr_p, _ = roc_curve(labels, np.concatenate([neg_pca, pos_pca]))
        ax.plot(fpr_p, tpr_p, color='#10b981', linewidth=2.0, label=f"PCA 10D Mahalanobis (AUC = {auc(fpr_p, tpr_p):.3f})")
        
        # Prediction MSE
        fpr_m, tpr_m, _ = roc_curve(labels, np.concatenate([neg_mse, pos_mse]))
        ax.plot(fpr_m, tpr_m, color='#f59e0b', linewidth=2.0, label=f"Prediction MSE (AUC = {auc(fpr_m, tpr_m):.3f})")
        
        # Diagonal baseline
        ax.plot([0, 1], [0, 1], color='#ef4444', linestyle='--', linewidth=1.2)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xlabel("False Positive Rate", fontsize=8)
        ax.set_ylabel("True Positive Rate", fontsize=8)
        ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9, fontsize=8)
        
    plt.suptitle("OOD Detection Performance Comparison (ROC Curves)", fontweight='bold', fontsize=15, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig2_roc_curves.png"), bbox_inches='tight')
    plt.close()
    
    print("\nFormal Experiments executed successfully!")
    print(f"Results summary JSON saved to: {results_dir}/formal_experiment_metrics.json")
    print(f"Plots saved to: {results_dir}/")

if __name__ == "__main__":
    main()
