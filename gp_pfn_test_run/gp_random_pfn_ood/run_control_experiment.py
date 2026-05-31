import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

# Add module paths
sys.path.insert(0, "/home/psquare_a6000/Desktop/conformal_ood/TransformersCanDoBayesianInference")
sys.path.append("/home/psquare_a6000/Desktop/conformal_ood")

from gp_pfn_ood.data_generators import (
    generate_matern_gp_batch,
    generate_periodic_batch,
    generate_chirp_batch
)

# Premium styling
plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white'
})

class ActivationHookManager:
    """Registers forward hooks on Transformer Encoder layers to capture inputs and outputs."""
    def __init__(self, model):
        self.model = model
        self.inputs = {}
        self.outputs = {}
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        for idx, layer in enumerate(self.model.transformer_encoder.layers):
            # Capture layer inputs and outputs
            def make_hook(l_idx):
                def hook(module, inp, out):
                    # inp[0] shape: [seq_len, batch, hidden_dim]
                    # out shape: [seq_len, batch, hidden_dim]
                    self.inputs[l_idx] = inp[0].detach().cpu()
                    self.outputs[l_idx] = out.detach().cpu()
                return hook
            h = layer.register_forward_hook(make_hook(idx))
            self.hooks.append(h)
            
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

def reinitialize_weights(model):
    """Reinitializes all learnable parameters in the model to random values."""
    torch.manual_seed(999)
    print("Reinitializing model weights to random...")
    for m in model.modules():
        if hasattr(m, 'reset_parameters') and callable(getattr(m, 'reset_parameters')):
            m.reset_parameters()
        else:
            # Manually reset weights and biases if they are PyTorch parameters
            if hasattr(m, 'weight') and isinstance(m.weight, torch.nn.Parameter):
                if len(m.weight.shape) >= 2:
                    torch.nn.init.xavier_uniform_(m.weight)
                else:
                    torch.nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and isinstance(m.bias, torch.nn.Parameter):
                torch.nn.init.zeros_(m.bias)

def analyze_model(model, x_all, y_all, labels, device, batch_size=128):
    """Runs inference and extracts activation-based metrics for each layer."""
    hook_manager = ActivationHookManager(model)
    num_samples = len(x_all)
    num_layers = len(model.transformer_encoder.layers)
    
    # Store aggregated representations averaged over sequence length
    layer_acts = {l: [] for l in range(num_layers)}
    layer_inps = {l: [] for l in range(num_layers)}
    layer_outs = {l: [] for l in range(num_layers)}
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            bx = x_all[i : i + batch_size].to(device)
            by = y_all[i : i + batch_size].to(device)
            
            src_x = bx.transpose(0, 1)
            src_y = by.transpose(0, 1)
            
            _ = model((src_x, src_y), single_eval_pos=100)
            
            # Cache batch outputs
            for l in range(num_layers):
                # Shape of hook outputs: [seq_len, batch_size, hidden_dim]
                # Save full seq-level activations to compute residual updates
                layer_inps[l].append(hook_manager.inputs[l])
                layer_outs[l].append(hook_manager.outputs[l])
                # Save sequence-averaged representations for PCA and Clustering
                layer_acts[l].append(torch.mean(hook_manager.outputs[l], dim=0))
                
    hook_manager.remove_hooks()
    
    # Concatenate lists
    layer_results = []
    
    # We will also return the sequence-averaged activations of the final layer for plotting
    final_layer_acts = torch.cat(layer_acts[num_layers - 1], dim=0).numpy()
    
    for l in range(num_layers):
        # Shape: (num_samples, hidden_dim)
        X_l = torch.cat(layer_acts[l], dim=0).numpy()
        
        # Calculate silhouette score (K=2)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(X_l)
        sil = float(silhouette_score(X_l, cluster_labels))
        
        # Calculate subspace dimensionality
        X_id = X_l[labels == 0]
        X_ood = X_l[labels == 1]
        
        pca_id = PCA()
        pca_id.fit(X_id)
        d_id = int(np.argmax(np.cumsum(pca_id.explained_variance_ratio_) >= 0.95) + 1)
        
        pca_ood = PCA()
        pca_ood.fit(X_ood)
        d_ood = int(np.argmax(np.cumsum(pca_ood.explained_variance_ratio_) >= 0.95) + 1)
        
        # Subspace Overlap
        Q_id = pca_id.components_[:d_id].T
        Q_ood = pca_ood.components_[:d_ood].T
        overlap = float(np.linalg.norm(Q_id.T @ Q_ood, ord='fro')**2 / min(d_id, d_ood))
        
        # Calculate Residual Update Norm
        # We concatenate batch blocks along batch dimension (dim 1 of [seq_len, batch_size, hidden_dim])
        full_in = torch.cat(layer_inps[l], dim=1)
        full_out = torch.cat(layer_outs[l], dim=1)
        diff = full_out - full_in
        # L2 norm over hidden_dim, then average over seq and batch
        update_norm = float(torch.mean(torch.norm(diff, p=2, dim=-1)).item())
        
        layer_results.append({
            "layer": l,
            "silhouette": sil,
            "d_id_95": d_id,
            "d_ood_95": d_ood,
            "subspace_overlap": overlap,
            "residual_update": update_norm
        })
        
    return layer_results, final_layer_acts

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    checkpoint_path = "/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_test_run/checkpoints/onefeature_gp_ls.1_pnf_4M.pt"
    results_dir = "/home/psquare_a6000/Desktop/conformal_ood/gp_random_pfn_ood"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load Trained and Untrained Models
    print("Loading PFN models...")
    model_trained = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_untrained = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Patch compatibility issues for both models
    for model in [model_trained, model_untrained]:
        for m in model.modules():
            if m.__class__.__name__ == 'TransformerEncoderLayer':
                if not hasattr(m, 'norm_first'):
                    m.norm_first = False
            elif m.__class__.__name__ == 'GELU':
                if not hasattr(m, 'approximate'):
                    m.approximate = 'none'
    
    # Reinitialize untrained
    reinitialize_weights(model_untrained)
    
    model_trained.eval().to(device)
    model_untrained.eval().to(device)
    
    # 2. Generate 1000 ID and 1000 OOD samples (500 periodic, 500 chirp)
    print("Generating evaluation datasets...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    num_samples_per_class = 1000
    x_id, y_id = generate_matern_gp_batch(batch_size=num_samples_per_class, num_points=100, device="cpu")
    
    x_p, y_p = generate_periodic_batch(batch_size=num_samples_per_class // 2, num_points=100, device="cpu")
    x_c, y_c = generate_chirp_batch(batch_size=num_samples_per_class // 2, num_points=100, device="cpu")
    x_ood = torch.cat([x_p, x_c], dim=0)
    y_ood = torch.cat([y_p, y_c], dim=0)
    
    x_all = torch.cat([x_id, x_ood], dim=0)
    y_all = torch.cat([y_id, y_ood], dim=0)
    labels = np.array([0] * num_samples_per_class + [1] * num_samples_per_class)
    
    # 3. Analyze Models
    print("\nAnalyzing Trained Model...")
    trained_metrics, trained_final_acts = analyze_model(model_trained, x_all, y_all, labels, device)
    
    print("\nAnalyzing Untrained Model...")
    untrained_metrics, untrained_final_acts = analyze_model(model_untrained, x_all, y_all, labels, device)
    
    # 4. Save results to JSON and CSV
    summary_data = {
        "trained": trained_metrics,
        "untrained": untrained_metrics
    }
    
    with open(os.path.join(results_dir, "control_experiment_summary.json"), 'w') as f:
        json.dump(summary_data, f, indent=4)
        
    # Convert to CSV for easy inspection
    df_trained = pd.DataFrame(trained_metrics).add_prefix("trained_")
    df_untrained = pd.DataFrame(untrained_metrics).add_prefix("untrained_")
    df_all = pd.concat([df_trained, df_untrained], axis=1)
    df_all.to_csv(os.path.join(results_dir, "control_experiment_summary.csv"), index=False)
    
    print("\nSaved metrics to CSV and JSON.")
    
    # 5. Plot Comparison Trends
    layers = range(len(trained_metrics))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    
    # Subplot A: Silhouette Scores
    ax_sil = axes[0, 0]
    ax_sil.plot(layers, [r["silhouette"] for r in trained_metrics], marker='o', color='#1f77b4', linewidth=2.5, label='Trained PFN')
    ax_sil.plot(layers, [r["silhouette"] for r in untrained_metrics], marker='s', color='#7f7f7f', linewidth=2.5, linestyle='--', label='Untrained PFN')
    ax_sil.set_title("Cluster Separation (Silhouette Score) vs. Depth", fontweight='bold')
    ax_sil.set_xlabel("Layer Index")
    ax_sil.set_ylabel("Silhouette Score (K=2)")
    ax_sil.legend()
    
    # Subplot B: Subspace Overlap
    ax_over = axes[0, 1]
    ax_over.plot(layers, [r["subspace_overlap"] for r in trained_metrics], marker='o', color='#2ca02c', linewidth=2.5, label='Trained PFN')
    ax_over.plot(layers, [r["subspace_overlap"] for r in untrained_metrics], marker='s', color='#7f7f7f', linewidth=2.5, linestyle='--', label='Untrained PFN')
    ax_over.set_title("Subspace Overlap (Frobenius) vs. Depth", fontweight='bold')
    ax_over.set_xlabel("Layer Index")
    ax_over.set_ylabel("Overlap Ratio (95% Var)")
    ax_over.legend()
    
    # Subplot C: Intrinsic Dimensionality (ID and OOD)
    ax_dim = axes[1, 0]
    ax_dim.plot(layers, [r["d_id_95"] for r in trained_metrics], marker='o', color='#1f77b4', linewidth=2.0, label='Trained - ID')
    ax_dim.plot(layers, [r["d_ood_95"] for r in trained_metrics], marker='x', color='#ff7f0e', linewidth=2.0, label='Trained - OOD')
    ax_dim.plot(layers, [r["d_id_95"] for r in untrained_metrics], marker='s', color='#aec7e8', linewidth=1.5, linestyle=':', label='Untrained - ID')
    ax_dim.plot(layers, [r["d_ood_95"] for r in untrained_metrics], marker='d', color='#ffbb78', linewidth=1.5, linestyle=':', label='Untrained - OOD')
    ax_dim.set_title("Intrinsic Subspace Dimension ($d_{95}$) vs. Depth", fontweight='bold')
    ax_dim.set_xlabel("Layer Index")
    ax_dim.set_ylabel("Subspace Dimension")
    ax_dim.legend()
    
    # Subplot D: Residual Update Magnitude
    ax_up = axes[1, 1]
    ax_up.plot(layers, [r["residual_update"] for r in trained_metrics], marker='o', color='#d62728', linewidth=2.5, label='Trained PFN')
    ax_up.plot(layers, [r["residual_update"] for r in untrained_metrics], marker='s', color='#7f7f7f', linewidth=2.5, linestyle='--', label='Untrained PFN')
    ax_up.set_title("Residual Stream Update Norm vs. Depth", fontweight='bold')
    ax_up.set_xlabel("Layer Index")
    ax_up.set_ylabel("Mean $L_2$ Norm of Update")
    ax_up.legend()
    
    plt.suptitle("Control Experiment: Trained vs. Untrained PFN Representations", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "control_experiment_trends.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(results_dir, "control_experiment_trends.pdf"), format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 6. Plot Layer 5 Projections (Trained vs Untrained)
    print("Computing Layer 5 PCA/t-SNE projections...")
    # Trained PCA
    pca_t = PCA(n_components=2, random_state=42)
    coords_t_pca = pca_t.fit_transform(trained_final_acts)
    # Untrained PCA
    pca_u = PCA(n_components=2, random_state=42)
    coords_u_pca = pca_u.fit_transform(untrained_final_acts)
    
    # Trained t-SNE
    tsne_t = TSNE(n_components=2, random_state=42, n_jobs=-1)
    coords_t_tsne = tsne_t.fit_transform(trained_final_acts)
    # Untrained t-SNE
    tsne_u = TSNE(n_components=2, random_state=42, n_jobs=-1)
    coords_u_tsne = tsne_u.fit_transform(untrained_final_acts)
    
    fig_proj, axes_proj = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    
    # Row 1: Trained Projections
    ax = axes_proj[0, 0]
    ax.scatter(coords_t_pca[labels == 0, 0], coords_t_pca[labels == 0, 1], alpha=0.5, color='#1f77b4', s=6, label='ID (GP)')
    ax.scatter(coords_t_pca[labels == 1, 0], coords_t_pca[labels == 1, 1], alpha=0.5, color='#ff7f0e', s=6, label='OOD')
    ax.set_title("Trained PFN - PCA Projection", fontweight='bold')
    ax.legend(loc='upper right')
    
    ax = axes_proj[0, 1]
    ax.scatter(coords_t_tsne[labels == 0, 0], coords_t_tsne[labels == 0, 1], alpha=0.5, color='#1f77b4', s=6, label='ID (GP)')
    ax.scatter(coords_t_tsne[labels == 1, 0], coords_t_tsne[labels == 1, 1], alpha=0.5, color='#ff7f0e', s=6, label='OOD')
    ax.set_title("Trained PFN - t-SNE Projection", fontweight='bold')
    ax.legend(loc='upper right')
    
    # Row 2: Untrained Projections
    ax = axes_proj[1, 0]
    ax.scatter(coords_u_pca[labels == 0, 0], coords_u_pca[labels == 0, 1], alpha=0.5, color='#1f77b4', s=6, label='ID (GP)')
    ax.scatter(coords_u_pca[labels == 1, 0], coords_u_pca[labels == 1, 1], alpha=0.5, color='#ff7f0e', s=6, label='OOD')
    ax.set_title("Untrained PFN (Random) - PCA Projection", fontweight='bold')
    ax.legend(loc='upper right')
    
    ax = axes_proj[1, 1]
    ax.scatter(coords_u_tsne[labels == 0, 0], coords_u_tsne[labels == 0, 1], alpha=0.5, color='#1f77b4', s=6, label='ID (GP)')
    ax.scatter(coords_u_tsne[labels == 1, 0], coords_u_tsne[labels == 1, 1], alpha=0.5, color='#ff7f0e', s=6, label='OOD')
    ax.set_title("Untrained PFN (Random) - t-SNE Projection", fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.suptitle("Layer 5 Representation Projections: Trained vs. Untrained PFN", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "projection_comparison.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(results_dir, "projection_comparison.pdf"), format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("Saved projections plots (PNG and PDF).")
    print(f"\nControl experiment complete! All results saved in {results_dir}/")

if __name__ == "__main__":
    main()
