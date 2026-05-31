import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from sklearn.decomposition import PCA

# Add module paths
sys.path.insert(0, "/home/psquare_a6000/Desktop/conformal_ood/TransformersCanDoBayesianInference")
sys.path.append("/home/psquare_a6000/Desktop/conformal_ood")

from gp_pfn_ood.data_generators import get_id_and_ood_data
from gp_pfn_ood.pfn_hooks import PFNHookManager

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Paths configuration
    checkpoint_path = "/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_test_run/checkpoints/onefeature_gp_ls.1_pnf_4M.pt"
    folders = [
        "/home/psquare_a6000/Desktop/conformal_ood/dimensionality",
        "/home/psquare_a6000/Desktop/conformal_ood/dimnesionality"
    ]
    for f in folders:
        os.makedirs(f, exist_ok=True)

    # 1. Data Generation
    # 3,000 ID datasets and 3,000 OOD datasets
    num_samples_per_class = 3000
    x_id, y_id, x_ood, y_ood = get_id_and_ood_data(
        num_samples_per_class=num_samples_per_class,
        num_points=100,
        device="cpu"
    )

    # Combine ID and OOD
    x_all = torch.cat([x_id, x_ood], dim=0) # (6000, 100, 1)
    y_all = torch.cat([y_id, y_ood], dim=0) # (6000, 100)
    labels = np.array([0] * num_samples_per_class + [1] * num_samples_per_class)

    # 2. Extract hook activations
    hook_manager = PFNHookManager(checkpoint_path=checkpoint_path, device=device)
    batch_size = 128
    num_samples = len(x_all)
    print(f"Extracting layer activations for {num_samples} samples...")

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Inference"):
            bx = x_all[i : i + batch_size].to(device)
            by = y_all[i : i + batch_size].to(device)
            src_x = bx.transpose(0, 1)
            src_y = by.transpose(0, 1)
            _ = hook_manager.model((src_x, src_y), single_eval_pos=100)

    layer_activations, _ = hook_manager.get_aggregated_features()
    hook_manager.remove_hooks()
    
    num_layers = len(layer_activations)
    print(f"Extracted activations for {num_layers} layers.")

    # 3. Analyze Subspaces
    layer_results = []
    
    # Grid of cumulative explained variance curves
    fig_ev, axes_ev = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
    axes_ev = axes_ev.flatten()

    for l in range(num_layers):
        print(f"\nAnalyzing Layer {l}...")
        X_layer = layer_activations[l].numpy() # (6000, hidden_dim)
        
        X_id_layer = X_layer[labels == 0]     # (3000, hidden_dim)
        X_ood_layer = X_layer[labels == 1]    # (3000, hidden_dim)

        # Fit PCA separately
        pca_id = PCA()
        pca_id.fit(X_id_layer)
        cumsum_id = np.cumsum(pca_id.explained_variance_ratio_)
        d_id = int(np.argmax(cumsum_id >= 0.95) + 1)

        pca_ood = PCA()
        pca_ood.fit(X_ood_layer)
        cumsum_ood = np.cumsum(pca_ood.explained_variance_ratio_)
        d_ood = int(np.argmax(cumsum_ood >= 0.95) + 1)

        # Retrieve bases for 95% variance
        Q_id = pca_id.components_[:d_id].T   # (hidden_dim, d_id)
        Q_ood = pca_ood.components_[:d_ood].T # (hidden_dim, d_ood)

        # Calculate Subspace Overlap (normalized projection matrix squared Frobenius norm)
        # norm_F^2 (Q_id^T @ Q_ood) / min(d_id, d_ood)
        proj_matrix = Q_id.T @ Q_ood
        overlap = float(np.linalg.norm(proj_matrix, ord='fro')**2 / min(d_id, d_ood))

        # Calculate Cross-Projection Explained Variance
        # ID centered
        X_id_cent = X_id_layer - X_id_layer.mean(axis=0)
        tot_var_id = np.sum(np.var(X_id_cent, axis=0))
        proj_var_id = np.sum(np.var(X_id_cent @ Q_ood, axis=0))
        ev_id_to_ood = float(proj_var_id / (tot_var_id + 1e-8))

        # OOD centered
        X_ood_cent = X_ood_layer - X_ood_layer.mean(axis=0)
        tot_var_ood = np.sum(np.var(X_ood_cent, axis=0))
        proj_var_ood = np.sum(np.var(X_ood_cent @ Q_id, axis=0))
        ev_ood_to_id = float(proj_var_ood / (tot_var_ood + 1e-8))

        layer_results.append({
            "layer": l,
            "d_id_95": d_id,
            "d_ood_95": d_ood,
            "subspace_overlap": overlap,
            "variance_ratio_id_to_ood_subspace": ev_id_to_ood,
            "variance_ratio_ood_to_id_subspace": ev_ood_to_id
        })

        print(f"Layer {l} dimensions for 95% var: ID={d_id}, OOD={d_ood}")
        print(f"Subspace Overlap: {overlap:.4f}")
        print(f"ID variance explained by OOD subspace: {ev_id_to_ood:.4f}")
        print(f"OOD variance explained by ID subspace: {ev_ood_to_id:.4f}")

        # Plot Cumulative Variance
        ax = axes_ev[l]
        ax.plot(range(1, len(cumsum_id) + 1), cumsum_id, label='ID (GP)', color='#1f77b4', linewidth=2)
        ax.plot(range(1, len(cumsum_ood) + 1), cumsum_ood, label='OOD', color='#ff7f0e', linewidth=2)
        ax.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='95% Threshold')
        ax.axvline(d_id, color='#1f77b4', linestyle=':', alpha=0.7)
        ax.axvline(d_ood, color='#ff7f0e', linestyle=':', alpha=0.7)
        ax.set_xlim(0, 50)  # Zoom in on first few components
        ax.set_ylim(0.4, 1.02)
        ax.set_title(f"Layer {l} Cumulative Variance", fontweight='bold')
        ax.set_xlabel("PC Index")
        ax.set_ylabel("Explained Variance Ratio")
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.suptitle("PCA Explained Variance Spectrum across layers", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    for f in folders:
        plt.savefig(os.path.join(f, "explained_variance_spectra.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # Save quantitative metrics to CSV and JSON
    df_metrics = pd.DataFrame(layer_results)
    for f in folders:
        df_metrics.to_csv(os.path.join(f, "subspace_analysis.csv"), index=False)
        with open(os.path.join(f, "subspace_analysis.json"), 'w') as json_f:
            json.dump(layer_results, json_f, indent=4)

    # 4. Plot Subspace Dimensionality and Overlap Trends
    fig_trends, (ax_dim, ax_sim) = plt.subplots(2, 1, figsize=(10, 10), dpi=300, sharex=True)

    layers = range(num_layers)
    d_id_vals = [r["d_id_95"] for r in layer_results]
    d_ood_vals = [r["d_ood_95"] for r in layer_results]
    overlap_vals = [r["subspace_overlap"] for r in layer_results]
    ev_id_to_ood_vals = [r["variance_ratio_id_to_ood_subspace"] for r in layer_results]
    ev_ood_to_id_vals = [r["variance_ratio_ood_to_id_subspace"] for r in layer_results]

    # Subplot 1: Intrinsic Dimensionality
    ax_dim.plot(layers, d_id_vals, marker='o', color='#1f77b4', linewidth=2.5, label='ID (GP) Subspace Dim ($d_{95}$)')
    ax_dim.plot(layers, d_ood_vals, marker='s', color='#ff7f0e', linewidth=2.5, label='OOD Subspace Dim ($d_{95}$)')
    ax_dim.set_title("Intrinsic Subspace Dimensionality ($d_{95}$) vs. Layer Depth", fontweight='bold', fontsize=12)
    ax_dim.set_ylabel("Subspace Dimension ($d_{95}$)")
    ax_dim.grid(True, alpha=0.3)
    ax_dim.legend(loc='upper left', frameon=True)
    ax_dim.set_xticks(layers)

    # Subplot 2: Subspace Similarity / Alignment
    ax_sim.plot(layers, overlap_vals, marker='^', color='#2ca02c', linewidth=2.5, label='Subspace Overlap (Frobenius)')
    ax_sim.plot(layers, ev_id_to_ood_vals, marker='v', linestyle='--', color='#9467bd', alpha=0.8, label='ID Var explained by OOD Subspace')
    ax_sim.plot(layers, ev_ood_to_id_vals, marker='>', linestyle='--', color='#bcbd22', alpha=0.8, label='OOD Var explained by ID Subspace')
    ax_sim.set_title("Subspace Alignment and Cross-Projection Overlap vs. Layer Depth", fontweight='bold', fontsize=12)
    ax_sim.set_xlabel("Layer Depth")
    ax_sim.set_ylabel("Similarity Ratio")
    ax_sim.set_ylim(-0.05, 1.05)
    ax_sim.grid(True, alpha=0.3)
    ax_sim.legend(loc='lower left', frameon=True)

    plt.tight_layout()
    for f in folders:
        plt.savefig(os.path.join(f, "subspace_alignment_trends.png"), bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(f, "subspace_alignment_trends.pdf"), format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    print("\nSubspace analysis complete!")
    print(f"Metrics saved to {folders[0]}/subspace_analysis.csv")

if __name__ == "__main__":
    main()
