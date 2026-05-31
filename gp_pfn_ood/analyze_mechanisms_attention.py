import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from typing import Dict, List, Tuple, Any

# Ensure parent directory is in path to import transformer modules from the repository
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TransformersCanDoBayesianInference')))
import transformer

# Ensure we can load local modules
from data_generators import get_id_and_ood_data

plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white'
})


class PFNAttentionHookManager:
    """
    Custom hook manager specifically targeting per-head attention.
    Forces average_attn_weights=False so we get (B, Heads, Target_Len, Source_Len).
    """
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        print(f"Loading pre-trained PFN from {checkpoint_path}...")
        self.model = torch.load(checkpoint_path, map_location=device, weights_only=False)

        self.model.eval()
        self.model.to(device)
        self.attention_maps: Dict[int, List[torch.Tensor]] = {}
        self.hooks: List[Any] = []
        self._patch_and_hook()

    def _patch_and_hook(self):
        # 1. Compatibility fixes
        for m in self.model.modules():
            if m.__class__.__name__ == 'TransformerEncoderLayer':
                if not hasattr(m, 'norm_first'): m.norm_first = False
            elif m.__class__.__name__ == 'GELU':
                if not hasattr(m, 'approximate'): m.approximate = 'none'

        # 2. Force MultiheadAttention to return per-head weights
        for m in self.model.modules():
            if m.__class__.__name__ == 'MultiheadAttention':
                orig_forward = m.forward
                def make_new_forward(orig):
                    def new_forward(query, key, value, *args, **kwargs):
                        kwargs['need_weights'] = True
                        kwargs['average_attn_weights'] = False  # Important for per-head
                        return orig(query, key, value, *args, **kwargs)
                    return new_forward
                m.forward = make_new_forward(orig_forward)

        # 3. Hook just the attention maps
        for idx, layer in enumerate(self.model.transformer_encoder.layers):
            self.attention_maps[idx] = []

            def make_sa_hook(l_idx: int):
                def sa_hook(module: nn.Module, input_args: Any, output_val: Tuple[torch.Tensor, torch.Tensor]):
                    # output_val[1] shape: (Batch, Heads, Target_Seq, Source_Seq)
                    if isinstance(output_val, tuple) and len(output_val) > 1 and output_val[1] is not None:
                        self.attention_maps[l_idx].append(output_val[1].detach().cpu())
                return sa_hook

            h_sa = layer.self_attn.register_forward_hook(make_sa_hook(idx))
            self.hooks.append(h_sa)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # checkpoint_path = "gp_pfn_test_run/checkpoints/onefeature_gp_ls.1_pnf_4M.pt"
    checkpoint_path = "tabpfn_full_model.pt"
    results_dir = "gp_pfn_ood/results/mechanistic_plots_attention_tabpfn"
    os.makedirs(results_dir, exist_ok=True)

    # We need query tokens! So we generate 150 points total
    # (100 for context history, 50 for query evaluation points)
    eval_pos = 100
    total_points = 150
    num_samples_per_class = 1500  # Kept slightly smaller so UMAP runs quickly on flattened massive vectors

    print(f"Generating data with {total_points} sequence length...")
    x_id, y_id, x_ood, y_ood = get_id_and_ood_data(
        num_samples_per_class=num_samples_per_class,
        num_points=total_points,
        device="cpu"
    )

    x_all = torch.cat([x_id, x_ood], dim=0)
    y_all = torch.cat([y_id, y_ood], dim=0)
    labels = np.array([0] * num_samples_per_class + [1] * num_samples_per_class)

    hook_manager = PFNAttentionHookManager(checkpoint_path=checkpoint_path, device=device)

    batch_size = 128
    num_samples = len(x_all)

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Running Inference"):
            bx = x_all[i : i + batch_size].to(device)
            by = y_all[i : i + batch_size].to(device)

            src_x = bx.transpose(0, 1)  # (Seq, Batch, Dim)
            src_y = by.transpose(0, 1)

            # evaluate starting from point 100
            _ = hook_manager.model((src_x, src_y), single_eval_pos=eval_pos)

    # Concat batches: Maps shape will be -> (total_samples, Heads, Seq, Seq)
    # Note: Seq here is 150.
    layer_attention = {
        l: torch.cat(maps, dim=0) for l, maps in hook_manager.attention_maps.items()
    }
    hook_manager.remove_hooks()
    num_layers = len(layer_attention)

    # Tracking metrics across layers
    knn_acc_ctx_list = []
    knn_acc_qry_list = []
    sil_score_ctx_list = []
    sil_score_qry_list = []

    print("\n--- Running Dimensionality Reduction (UMAP) on Attention Sub-Matrices ---")

    for l in range(num_layers):
        print(f"Processing Attention Layer {l}...")
        attn_matrix = layer_attention[l]  # (3000, Heads, 150, 150)

        # 1. Context-to-Context Attention
        # Target = context (0:100), Source = context (0:100)
        ctx_ctx = attn_matrix[:, :, 0:eval_pos, 0:eval_pos]  # (Samples, Heads, 100, 100)

        # 2. Query-to-Context Attention
        # Target = queries (100:150), Source = context (0:100)
        qry_ctx = attn_matrix[:, :, eval_pos:, 0:eval_pos]   # (Samples, Heads, 50, 100)

        # Flatten across heads and sequences to create single vector representations per dataset
        ctx_ctx_flat = ctx_ctx.reshape(num_samples, -1).numpy()
        qry_ctx_flat = qry_ctx.reshape(num_samples, -1).numpy()

        # Reduce dimensionality with PCA, t-SNE, UMAP
        pca_ctx = PCA(n_components=2, random_state=42).fit_transform(ctx_ctx_flat)
        pca_qry = PCA(n_components=2, random_state=42).fit_transform(qry_ctx_flat)

        tsne_ctx = TSNE(n_components=2, random_state=42, n_jobs=-1).fit_transform(ctx_ctx_flat)
        tsne_qry = TSNE(n_components=2, random_state=42, n_jobs=-1).fit_transform(qry_ctx_flat)

        reducer_ctx = umap.UMAP(n_components=2, random_state=42, n_jobs=-1)
        x_umap_ctx = reducer_ctx.fit_transform(ctx_ctx_flat)

        reducer_qry = umap.UMAP(n_components=2, random_state=42, n_jobs=-1)
        x_umap_qry = reducer_qry.fit_transform(qry_ctx_flat)

        # Plotting 2 rows, 3 columns
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        methods_ctx = [("PCA", pca_ctx), ("t-SNE", tsne_ctx), ("UMAP", x_umap_ctx)]
        methods_qry = [("PCA", pca_qry), ("t-SNE", tsne_qry), ("UMAP", x_umap_qry)]

        for idx, (name, coords) in enumerate(methods_ctx):
            ax = axes[0, idx]
            ax.scatter(coords[labels == 0, 0], coords[labels == 0, 1], alpha=0.5, label='ID (GP)', color='#1f77b4', s=6)
            ax.scatter(coords[labels == 1, 0], coords[labels == 1, 1], alpha=0.5, label='OOD', color='#ff7f0e', s=6)
            ax.set_title(f"Context-to-Context ({name})", fontweight='bold')
            ax.legend(loc="upper right")

        for idx, (name, coords) in enumerate(methods_qry):
            ax = axes[1, idx]
            ax.scatter(coords[labels == 0, 0], coords[labels == 0, 1], alpha=0.5, label='ID (GP)', color='#1f77b4', s=6)
            ax.scatter(coords[labels == 1, 0], coords[labels == 1, 1], alpha=0.5, label='OOD', color='#ff7f0e', s=6)
            ax.set_title(f"Query-to-Context ({name})", fontweight='bold')
            ax.legend(loc="upper right")

        fig.suptitle(f"Transformer Layer {l} - Attention Heads Analysis (ID vs OOD)", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/layer_{l}_attention_dim_reduction.png", dpi=300)
        plt.close()

        # Compute metrics
        # KNN Accuracy on UMAP
        knn_ctx = KNeighborsClassifier(n_neighbors=5)
        knn_ctx.fit(x_umap_ctx, labels)
        knn_acc_ctx_list.append(accuracy_score(labels, knn_ctx.predict(x_umap_ctx)))

        knn_qry = KNeighborsClassifier(n_neighbors=5)
        knn_qry.fit(x_umap_qry, labels)
        knn_acc_qry_list.append(accuracy_score(labels, knn_qry.predict(x_umap_qry)))

        # Silhouette Score on raw scores
        sil_score_ctx_list.append(silhouette_score(ctx_ctx_flat, labels))
        sil_score_qry_list.append(silhouette_score(qry_ctx_flat, labels))

    # Plot KNN Accuracy Across Layers
    plt.figure(figsize=(8, 5))
    layers_range = range(num_layers)
    plt.plot(layers_range, knn_acc_ctx_list, marker='o', label='Context-to-Context', color='#1f77b4', linewidth=2.0)
    plt.plot(layers_range, knn_acc_qry_list, marker='s', label='Query-to-Context', color='#ff7f0e', linewidth=2.0)
    plt.xlabel("Layer Depth")
    plt.ylabel("KNN Accuracy (K=5)")
    plt.title("KNN Accuracy on UMAP Embeddings vs Layer Depth", fontweight='bold')
    plt.xticks(layers_range)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/knn_accuracy_layers.png", dpi=300)
    plt.close()

    # Plot Silhouette Score Across Layers
    plt.figure(figsize=(8, 5))
    plt.plot(layers_range, sil_score_ctx_list, marker='o', label='Context-to-Context', color='#1f77b4', linewidth=2.0)
    plt.plot(layers_range, sil_score_qry_list, marker='s', label='Query-to-Context', color='#ff7f0e', linewidth=2.0)
    plt.xlabel("Layer Depth")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score on Raw Embeddings vs Layer Depth", fontweight='bold')
    plt.xticks(layers_range)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/silhouette_score_layers.png", dpi=300)
    plt.close()

    print("\n--- Running KL Divergence Analysis on Query-to-Context Attention ---")

    num_heads = layer_attention[0].shape[1]
    delta_kl_matrix = np.zeros((num_layers, num_heads))

    N = eval_pos
    log_N = np.log(N)
    eps = 1e-9

    for l in range(num_layers):
        attn_matrix = layer_attention[l]
        qry_ctx = attn_matrix[:, :, eval_pos:, 0:eval_pos].numpy() # (Samples, Heads, M, N)

        # Clamp to avoid log(0)
        qry_ctx_clamped = np.clip(qry_ctx, eps, 1.0)

        # Token-level divergence: log N - H(A_q)
        # H(A_q) = - sum(A * log(A)) over context points (last dimension)
        entropy = -np.sum(qry_ctx * np.log(qry_ctx_clamped), axis=-1) # (Samples, Heads, 50)
        kl_div = log_N - entropy # (Samples, Heads, 50)

        # Mean over M query points -> (Samples, Heads)
        mean_kl_per_sample = np.mean(kl_div, axis=-1)

        # Mean over ID (In-Prior) and OOD (OoP)
        expected_kl_id = np.mean(mean_kl_per_sample[labels == 0], axis=0) # (Heads,)
        expected_kl_ood = np.mean(mean_kl_per_sample[labels == 1], axis=0) # (Heads,)

        # Delta D_KL = Expected_ID - Expected_OoP
        delta_kl = expected_kl_id - expected_kl_ood

        delta_kl_matrix[l, :] = delta_kl

    # Plot Heatmap
    plt.figure(figsize=(8, 6))
    # diverging colormap: positive=red, negative=blue. 'RdBu_r' has red for positive
    plt.imshow(delta_kl_matrix, cmap='RdBu_r', aspect='auto', origin='lower')
    cbar = plt.colorbar(label=r'$\Delta D_{KL} (In-Prior - OoP)$')
    plt.xlabel("Attention Head Index", fontweight='bold')
    plt.ylabel("Layer Depth", fontweight='bold')
    plt.title("Difference in Query-to-Context Attention Divergence", fontweight='bold')
    plt.xticks(np.arange(num_heads))
    plt.yticks(np.arange(num_layers))

    # Add text annotations on each cell
    for i in range(num_layers):
        for j in range(num_heads):
            plt.text(j, i, f"{delta_kl_matrix[i, j]:.2f}",
                     ha="center", va="center", color="black" if abs(delta_kl_matrix[i, j]) < np.max(np.abs(delta_kl_matrix))*0.5 else "white")

    plt.tight_layout()
    plt.savefig(f"{results_dir}/kl_divergence_difference_heatmap.png", dpi=300)
    plt.close()

    print(f"\nAttention matrix analysis complete! Saved inside: {results_dir}/")

if __name__ == '__main__':
    main()