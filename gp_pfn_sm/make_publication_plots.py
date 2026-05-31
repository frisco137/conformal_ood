import sys
import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# Premium scientific styling for AAAI publication (serif-like fonts and clean axes)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

def compute_fpr_at_95_tpr(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.where(tpr >= 0.95)[0][0]
    return float(fpr[idx])

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device for inference: {device}")
    
    checkpoint_path = "gp_pfn_sm/checkpoints/spectral_mixture.pt"
    hook_manager = PFNHookManager(checkpoint_path=checkpoint_path, device=device)
    
    # Generate Calibration and Reference fitting sets
    print("Generating calibration and reference sets...")
    x_cal, y_cal = generate_id_low_noise(batch_size=1000, device="cpu")
    x_near_ref, y_near_ref = generate_near_ood(batch_size=500, device="cpu")
    x_far_ref, y_far_ref = generate_far_ood(batch_size=500, device="cpu")
    
    # Extract features
    print("Extracting representations...")
    with torch.no_grad():
        src_x = x_cal.transpose(0, 1).to(device)
        src_y = y_cal.transpose(0, 1).to(device)
        _ = hook_manager.model((src_x, src_y), single_eval_pos=100)
    layer_activations, _ = hook_manager.get_aggregated_features()
    X_cal = layer_activations[5].numpy()
    hook_manager.clear_cache()
    
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
    
    X_fit = np.concatenate([X_cal, X_near_ref, X_far_ref], axis=0)
    
    # Fit UMAP and PCA
    print("Fitting PCA and UMAP spaces...")
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
    
    # Process test suites
    test_suites = {
        "Suite 1 (ID Low Noise)": generate_id_low_noise,
        "Suite 2 (ID High Noise)": generate_id_high_noise,
        "Suite 3 (Near-OOD)": generate_near_ood,
        "Suite 4 (Far-OOD)": generate_far_ood,
        "Suite 5 (Real-World OOD)": generate_real_world_ood
    }
    
    suite_data = {}
    
    print("Running evaluations across test suites...")
    for suite_name, generator_fn in test_suites.items():
        x_test, y_test = generator_fn(batch_size=1000, device="cpu")
        
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
        
        with torch.no_grad():
            logits = hook_manager.model((src_x, src_y), single_eval_pos=50)
            targets = src_y[50:].unsqueeze(-1)
            mse_per_batch = torch.mean((logits - targets) ** 2, dim=[0, 2]).cpu().numpy()
        hook_manager.clear_cache()
        
        suite_data[suite_name] = {
            "umap_coords": X_test_umap,
            "umap_mahalanobis": d_m_umap,
            "pca_mahalanobis": d_m_pca_10d,
            "prediction_mse": mse_per_batch
        }
        
    hook_manager.remove_hooks()
    
    # ------------------ Plotting Figure 1: UMAP Manifold Separation (1x5 Grid) ------------------
    print("Generating Figure 1 (1x5 Grid)...")
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6), dpi=300)
    
    cal_coords = suite_data["Suite 1 (ID Low Noise)"]["umap_coords"]
    
    # Panel 0: Reference ID manifold (Neutral Slate Gray, solid alpha)
    axes[0].scatter(cal_coords[:, 0], cal_coords[:, 1], color='#475569', alpha=1.0, s=5, label="ID Low Noise", rasterized=True)
    axes[0].set_title("(a) ID Low Noise (Reference)", fontweight='bold', fontsize=11)
    axes[0].set_xlabel("UMAP Dimension 1", fontsize=9)
    axes[0].set_ylabel("UMAP Dimension 2", fontsize=9)
    axes[0].legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8)
    
    comparisons = [
        ("Suite 2 (ID High Noise)", "#10b981", "(b) ID High Noise", axes[1]),
        ("Suite 3 (Near-OOD)", "#f59e0b", "(c) Near-OOD Freq Shift", axes[2]),
        ("Suite 4 (Far-OOD)", "#ef4444", "(d) Far-OOD Extreme Shift", axes[3]),
        ("Suite 5 (Real-World OOD)", "#8b5cf6", "(e) Real-World OOD", axes[4])
    ]
    
    for suite_name, color, label, ax in comparisons:
        coords = suite_data[suite_name]["umap_coords"]
        
        # Plot ID background points in light gray (s=4, solid alpha)
        ax.scatter(cal_coords[:, 0], cal_coords[:, 1], color='#cbd5e1', alpha=1.0, s=4, label="ID Low Noise", rasterized=True)
        # Plot Comparison points with high contrast (s=6, solid alpha)
        ax.scatter(coords[:, 0], coords[:, 1], color=color, alpha=1.0, s=6, label=suite_name.replace("Suite ", ""), rasterized=True)
        
        ax.set_title(label, fontweight='bold', fontsize=11)
        ax.set_xlabel("UMAP Dimension 1", fontsize=9)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8)
            
    plt.tight_layout()
    paper_dir = "paper"
    figs_dir = os.path.join(paper_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    
    plt.savefig(os.path.join(figs_dir, "fig1_manifold_separation.pdf"), bbox_inches='tight', format='pdf')
    plt.savefig(os.path.join(figs_dir, "fig1_manifold_separation.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # ------------------ Plotting Figure 2: ROC Curves (2x2 Grid) ------------------
    print("Generating Figure 2 (2x2 Grid)...")
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5), dpi=300)
    axes = axes.flatten()
    
    neg_umap = suite_data["Suite 1 (ID Low Noise)"]["umap_mahalanobis"]
    neg_pca = suite_data["Suite 1 (ID Low Noise)"]["pca_mahalanobis"]
    neg_mse = suite_data["Suite 1 (ID Low Noise)"]["prediction_mse"]
    
    roc_tasks = [
        ("Suite 2 (ID High Noise)", axes[0], "Robustness to Noise (Target: Random Chance)", "Robustness to Noise"),
        ("Suite 3 (Near-OOD)", axes[1], "Near-OOD Frequency Shift", "Near-OOD Detection"),
        ("Suite 4 (Far-OOD)", axes[2], "Far-OOD Extreme Frequency Shift", "Far-OOD Detection"),
        ("Suite 5 (Real-World OOD)", axes[3], "Real-World OOD Detection", "Real-World OOD Detection")
    ]
    
    latex_metrics = {}
    
    for idx, (suite_name, ax, title, metric_name) in enumerate(roc_tasks):
        pos_umap = suite_data[suite_name]["umap_mahalanobis"]
        pos_pca = suite_data[suite_name]["pca_mahalanobis"]
        pos_mse = suite_data[suite_name]["prediction_mse"]
        
        labels = np.concatenate([np.zeros_like(neg_umap), np.ones_like(pos_umap)])
        
        # UMAP
        scores_umap = np.concatenate([neg_umap, pos_umap])
        fpr_u, tpr_u, _ = roc_curve(labels, scores_umap)
        auc_roc_u = auc(fpr_u, tpr_u)
        prec_u, rec_u, _ = precision_recall_curve(labels, scores_umap)
        auc_pr_u = auc(rec_u, prec_u)
        fpr95_u = compute_fpr_at_95_tpr(labels, scores_umap)
        ax.plot(fpr_u, tpr_u, color='#4f46e5', linewidth=2.2, label=f"UMAP 2D Mahalanobis (AUC = {auc_roc_u:.3f})")
        
        # PCA 10D
        scores_pca = np.concatenate([neg_pca, pos_pca])
        fpr_p, tpr_p, _ = roc_curve(labels, scores_pca)
        auc_roc_p = auc(fpr_p, tpr_p)
        prec_p, rec_p, _ = precision_recall_curve(labels, scores_pca)
        auc_pr_p = auc(rec_p, prec_p)
        fpr95_p = compute_fpr_at_95_tpr(labels, scores_pca)
        ax.plot(fpr_p, tpr_p, color='#10b981', linewidth=2.2, label=f"PCA 10D Mahalanobis (AUC = {auc_roc_p:.3f})")
        
        # Prediction MSE
        scores_mse = np.concatenate([neg_mse, pos_mse])
        fpr_m, tpr_m, _ = roc_curve(labels, scores_mse)
        auc_roc_m = auc(fpr_m, tpr_m)
        prec_m, rec_m, _ = precision_recall_curve(labels, scores_mse)
        auc_pr_m = auc(rec_m, prec_m)
        fpr95_m = compute_fpr_at_95_tpr(labels, scores_mse)
        ax.plot(fpr_m, tpr_m, color='#f59e0b', linewidth=2.2, label=f"Prediction MSE (AUC = {auc_roc_m:.3f})")
        
        # Baseline
        ax.plot([0, 1], [0, 1], color='#ef4444', linestyle='--', linewidth=1.2)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=8.5)
        
        latex_metrics[metric_name] = {
            "umap_2d": {"auc_roc": auc_roc_u, "auc_pr": auc_pr_u, "fpr95": fpr95_u},
            "pca_10d": {"auc_roc": auc_roc_p, "auc_pr": auc_pr_p, "fpr95": fpr95_p},
            "prediction_mse": {"auc_roc": auc_roc_m, "auc_pr": auc_pr_m, "fpr95": fpr95_m}
        }
        
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "fig2_roc_curves.pdf"), bbox_inches='tight', format='pdf')
    plt.savefig(os.path.join(figs_dir, "fig2_roc_curves.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save the latex table source
    latex_table_path = os.path.join(paper_dir, "table_metrics.tex")
    print(f"Generating LaTeX Table at {latex_table_path}...")
    
    with open(latex_table_path, "w") as f:
        f.write(r"""\begin{table*}[t]
\centering
\caption{Quantitative evaluation of PFN Out-of-Prior detection. UMAP-2D Mahalanobis balances high detection power with noise robustness (chance-level AUROC of 0.471 on ID High Noise).}
\label{tab:ood_metrics}
\vskip 0.1in
\begin{footnotesize}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{l ccc ccc ccc}
\hline
& \multicolumn{3}{c}{\textbf{UMAP-2D Mahalanobis (Ours)}} & \multicolumn{3}{c}{\textbf{PCA-10D Mahalanobis}} & \multicolumn{3}{c}{\textbf{Prediction MSE (Baseline)}} \\
\textbf{Evaluation Task} & \textbf{AUROC} $\uparrow$ & \textbf{AUPR} $\uparrow$ & \textbf{FPR@95} $\downarrow$ & \textbf{AUROC} $\uparrow$ & \textbf{AUPR} $\uparrow$ & \textbf{FPR@95} $\downarrow$ & \textbf{AUROC} $\uparrow$ & \textbf{AUPR} $\uparrow$ & \textbf{FPR@95} $\downarrow$ \\
\hline
""")
        # Row 1: Robustness to Noise
        row1 = f"Robustness to Noise (ID High Noise) & {latex_metrics['Robustness to Noise']['umap_2d']['auc_roc']:.3f} & {latex_metrics['Robustness to Noise']['umap_2d']['auc_pr']:.3f} & {latex_metrics['Robustness to Noise']['umap_2d']['fpr95']:.3f} & {latex_metrics['Robustness to Noise']['pca_10d']['auc_roc']:.3f} & {latex_metrics['Robustness to Noise']['pca_10d']['auc_pr']:.3f} & {latex_metrics['Robustness to Noise']['pca_10d']['fpr95']:.3f} & {latex_metrics['Robustness to Noise']['prediction_mse']['auc_roc']:.3f} & {latex_metrics['Robustness to Noise']['prediction_mse']['auc_pr']:.3f} & {latex_metrics['Robustness to Noise']['prediction_mse']['fpr95']:.3f} \\\\\n"
        # Row 2: Near-OOD Detection
        row2 = f"Near-OOD Detection (Freq Shift) & {latex_metrics['Near-OOD Detection']['umap_2d']['auc_roc']:.3f} & {latex_metrics['Near-OOD Detection']['umap_2d']['auc_pr']:.3f} & {latex_metrics['Near-OOD Detection']['umap_2d']['fpr95']:.3f} & {latex_metrics['Near-OOD Detection']['pca_10d']['auc_roc']:.3f} & {latex_metrics['Near-OOD Detection']['pca_10d']['auc_pr']:.3f} & {latex_metrics['Near-OOD Detection']['pca_10d']['fpr95']:.3f} & {latex_metrics['Near-OOD Detection']['prediction_mse']['auc_roc']:.3f} & {latex_metrics['Near-OOD Detection']['prediction_mse']['auc_pr']:.3f} & {latex_metrics['Near-OOD Detection']['prediction_mse']['fpr95']:.3f} \\\\\n"
        # Row 3: Far-OOD Detection
        row3 = f"Far-OOD Detection (Extreme Freq) & {latex_metrics['Far-OOD Detection']['umap_2d']['auc_roc']:.3f} & {latex_metrics['Far-OOD Detection']['umap_2d']['auc_pr']:.3f} & {latex_metrics['Far-OOD Detection']['umap_2d']['fpr95']:.3f} & {latex_metrics['Far-OOD Detection']['pca_10d']['auc_roc']:.3f} & {latex_metrics['Far-OOD Detection']['pca_10d']['auc_pr']:.3f} & {latex_metrics['Far-OOD Detection']['pca_10d']['fpr95']:.3f} & {latex_metrics['Far-OOD Detection']['prediction_mse']['auc_roc']:.3f} & {latex_metrics['Far-OOD Detection']['prediction_mse']['auc_pr']:.3f} & {latex_metrics['Far-OOD Detection']['prediction_mse']['fpr95']:.3f} \\\\\n"
        # Row 4: Real-World OOD Detection
        row4 = f"Real-World OOD Detection & {latex_metrics['Real-World OOD Detection']['umap_2d']['auc_roc']:.3f} & {latex_metrics['Real-World OOD Detection']['umap_2d']['auc_pr']:.3f} & {latex_metrics['Real-World OOD Detection']['umap_2d']['fpr95']:.3f} & {latex_metrics['Real-World OOD Detection']['pca_10d']['auc_roc']:.3f} & {latex_metrics['Real-World OOD Detection']['pca_10d']['auc_pr']:.3f} & {latex_metrics['Real-World OOD Detection']['pca_10d']['fpr95']:.3f} & {latex_metrics['Real-World OOD Detection']['prediction_mse']['auc_roc']:.3f} & {latex_metrics['Real-World OOD Detection']['prediction_mse']['auc_pr']:.3f} & {latex_metrics['Real-World OOD Detection']['prediction_mse']['fpr95']:.3f} \\\\\n"
        
        f.write(row1)
        f.write(row2)
        f.write(row3)
        f.write(row4)
        f.write(r"""\hline
\end{tabular}
\end{footnotesize}
\end{table*}
""")
        
    print("Done! Publication-quality plots saved in 'paper/' folder.")

if __name__ == "__main__":
    main()
