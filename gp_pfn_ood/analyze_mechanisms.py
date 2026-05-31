import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
from typing import Dict, List, Any

# Import local modules
from data_generators import get_id_and_ood_data
from pfn_hooks import PFNHookManager

# Matplotlib premium styling
plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white'
})

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device for inference: {device}")

    # Paths configuration
    checkpoint_path = "gp_pfns_multi_prior/checkpoints/rbf_periodic.pt"
    results_dir = "gp_pfn_ood/results/mechanistic_plots_rbf_periodic"
    json_summary_path = "gp_pfn_ood/results/mechanistic_summary.json"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Data Generation
    # 3,000 ID datasets and 3,000 OOD datasets
    num_samples_per_class = 3000
    x_id, y_id, x_ood, y_ood = get_id_and_ood_data(
        num_samples_per_class=num_samples_per_class,
        num_points=100,
        device="cpu"  # Keep on CPU first to save GPU memory
    )

    # Combine ID and OOD into a unified collection
    x_all = torch.cat([x_id, x_ood], dim=0) # (6000, 100, 1)
    y_all = torch.cat([y_id, y_ood], dim=0) # (6000, 100)
    labels = np.array([0] * num_samples_per_class + [1] * num_samples_per_class) # 0 for ID, 1 for OOD

    # 2. Hook and Model Manager Initialization
    hook_manager = PFNHookManager(checkpoint_path=checkpoint_path, device=device)

    # 3. Inference and Feature Extraction
    batch_size = 128
    num_samples = len(x_all)
    print(f"Running PFN model inference over {num_samples} datasets in batches of {batch_size}...")

    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Inference Batches"):
            bx = x_all[i : i + batch_size].to(device)
            by = y_all[i : i + batch_size].to(device)

            # Transpose to sequence first: (seq_len, batch_size, dim) and (seq_len, batch_size)
            src_x = bx.transpose(0, 1)
            src_y = by.transpose(0, 1)

            # Run model forward pass to trigger hooks
            _ = hook_manager.model((src_x, src_y), single_eval_pos=100)

    # Retrieve and aggregate the representations from CPU cache
    print("Extracting and aggregating hooked activations...")
    layer_activations, layer_attention = hook_manager.get_aggregated_features()
    hook_manager.remove_hooks()

    num_layers = len(layer_activations)
    print(f"Aggregated feature sizes for {num_layers} layers.")

    # Data storage for exporting
    silhouette_scores: List[float] = []
    attn_entropy_id_means: List[float] = []
    attn_entropy_ood_means: List[float] = []
    intrinsic_dim_id: List[int] = []
    intrinsic_dim_ood: List[int] = []

    # ------------------ Analysis A: Dimensionality Reduction ------------------
    print("\n--- Analysis A: Running Dimensionality Reduction (PCA, t-SNE, UMAP) ---")
    for l in range(num_layers):
        print(f"Processing Layer {l}...")
        X_layer = layer_activations[l].numpy() # (6000, hidden_dim)

        # Fit PCA
        pca = PCA(n_components=2, random_state=42)
        x_pca = pca.fit_transform(X_layer)

        # Fit t-SNE
        tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
        x_tsne = tsne.fit_transform(X_layer)

        # Fit UMAP
        reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=-1)
        x_umap = reducer.fit_transform(X_layer)

        # Plot side-by-side
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        methods = [("PCA", x_pca), ("t-SNE", x_tsne), ("UMAP", x_umap)]

        for idx, (name, coords) in enumerate(methods):
            ax = axes[idx]
            ax.scatter(coords[labels == 0, 0], coords[labels == 0, 1], alpha=0.5, label='ID (GP)', color='#1f77b4', s=4)
            ax.scatter(coords[labels == 1, 0], coords[labels == 1, 1], alpha=0.5, label='OOD', color='#ff7f0e', s=4)
            ax.set_title(f"{name} Representation")
            ax.legend(loc="upper right")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

        fig.suptitle(f"Transformer Encoder Layer {l} Manifold Separation (ID vs. OOD)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/layer_{l}_dim_reduction.png", dpi=300)
        plt.close()

    # ------------------ Analysis B: Silhouette Scores ------------------
    print("\n--- Analysis B: Computing Supervised Silhouette Scores ---")
    for l in tqdm(range(num_layers), desc="Silhouette Evaluation"):
        X_layer = layer_activations[l].numpy()
        score = silhouette_score(X_layer, labels)
        silhouette_scores.append(float(score))

    # Plot Silhouette Scores
    plt.figure(figsize=(8, 5))
    plt.plot(range(num_layers), silhouette_scores, marker='o', linewidth=2.5, color='#2ca02c')
    plt.xlabel("Layer Depth")
    plt.ylabel("Supervised Silhouette Score (ID vs OOD)")
    plt.title("Supervised ID/OOD Separation vs. Transformer Layer Depth", fontweight='bold')
    plt.xticks(range(num_layers))
    plt.tight_layout()
    plt.savefig(f"{results_dir}/silhouette_scores.png", dpi=300)
    plt.close()

    # ------------------ Analysis C: Intrinsic Dimensionality ------------------
    print("\n--- Analysis C: Computing Intrinsic Dimensionality (PCA 95%) ---")
    for l in range(num_layers):
        X_layer = layer_activations[l].numpy()

        # ID Pool
        X_id = X_layer[labels == 0]
        pca_id = PCA()
        pca_id.fit(X_id)
        cumsum_id = np.cumsum(pca_id.explained_variance_ratio_)
        dim_id = int(np.argmax(cumsum_id >= 0.95) + 1)
        intrinsic_dim_id.append(dim_id)

        # OOD Pool
        X_ood = X_layer[labels == 1]
        pca_ood = PCA()
        pca_ood.fit(X_ood)
        cumsum_ood = np.cumsum(pca_ood.explained_variance_ratio_)
        dim_ood = int(np.argmax(cumsum_ood >= 0.95) + 1)
        intrinsic_dim_ood.append(dim_ood)

    # Plot Intrinsic Dimensionality
    plt.figure(figsize=(8, 5))
    plt.plot(range(num_layers), intrinsic_dim_id, marker='o', label='ID (GP)', linewidth=2.0, color='#1f77b4')
    plt.plot(range(num_layers), intrinsic_dim_ood, marker='s', label='OOD', linewidth=2.0, color='#ff7f0e')
    plt.xlabel("Layer Depth")
    plt.ylabel("Intrinsic Dimensionality (95% Var)")
    plt.title("Intrinsic Feature Dimensionality vs. Layer Depth", fontweight='bold')
    plt.xticks(range(num_layers))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/intrinsic_dimensionality.png", dpi=300)
    plt.close()

    # ------------------ Analysis D: Attention Entropy ------------------
    print("\n--- Analysis D: Computing Attention Entropy ---")
    entropy_id_stds = []
    entropy_ood_stds = []

    for l in range(num_layers):
        attn = layer_attention[l] # (6000, 100, 100)
        eps = 1e-12
        entropy = -torch.sum(attn * torch.log(attn + eps), dim=-1) # (6000, 100)
        mean_entropy = torch.mean(entropy, dim=-1).numpy() # (6000,)

        id_ent = mean_entropy[labels == 0]
        ood_ent = mean_entropy[labels == 1]

        attn_entropy_id_means.append(float(np.mean(id_ent)))
        entropy_id_stds.append(float(np.std(id_ent)))
        attn_entropy_ood_means.append(float(np.mean(ood_ent)))
        entropy_ood_stds.append(float(np.std(ood_ent)))

    # Plot Attention Entropy
    plt.figure(figsize=(8, 5))
    layers_range = range(num_layers)
    plt.plot(layers_range, attn_entropy_id_means, marker='o', label='ID (GP)', color='#1f77b4', linewidth=2.0)
    plt.fill_between(
        layers_range,
        np.array(attn_entropy_id_means) - np.array(entropy_id_stds),
        np.array(attn_entropy_id_means) + np.array(entropy_id_stds),
        color='#1f77b4', alpha=0.15
    )
    plt.plot(layers_range, attn_entropy_ood_means, marker='s', label='OOD', color='#ff7f0e', linewidth=2.0)
    plt.fill_between(
        layers_range,
        np.array(attn_entropy_ood_means) - np.array(entropy_ood_stds),
        np.array(attn_entropy_ood_means) + np.array(entropy_ood_stds),
        color='#ff7f0e', alpha=0.15
    )
    plt.xlabel("Layer Depth")
    plt.ylabel("Shannon Entropy (Nats)")
    plt.title("Shannon Attention Entropy across Layers", fontweight='bold')
    plt.xticks(layers_range)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/attention_entropy.png", dpi=300)
    plt.close()

    # ------------------ Analysis E: Residual Update Magnitude ------------------
    print("\n--- Analysis E: Computing Residual Update Magnitude ---")
    update_id_means = []
    update_ood_means = []

    for l in range(1, num_layers):
        z_prev = layer_activations[l - 1]
        z_curr = layer_activations[l]
        diff = z_curr - z_prev
        norms = torch.norm(diff, p=2, dim=-1).numpy()

        update_id_means.append(float(np.mean(norms[labels == 0])))
        update_ood_means.append(float(np.mean(norms[labels == 1])))

    # Plot Residual Update Magnitude
    plt.figure(figsize=(8, 5))
    update_layers = range(1, num_layers)
    plt.plot(update_layers, update_id_means, marker='o', label='ID (GP)', color='#1f77b4', linewidth=2.0)
    plt.plot(update_layers, update_ood_means, marker='s', label='OOD', color='#ff7f0e', linewidth=2.0)
    plt.xlabel("Layer Transition (l-1 -> l)")
    plt.ylabel("Mean L2 Norm of Update")
    plt.title("Residual Stream Update Magnitude per Layer Transition", fontweight='bold')
    plt.xticks(update_layers)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/residual_updates.png", dpi=300)
    plt.close()

    # ------------------ Step 5: Save JSON Summary ------------------
    summary_data = {
        "layers": list(range(num_layers)),
        "silhouette_scores_k2": silhouette_scores,
        "attention_entropy": {
            "id_mean": attn_entropy_id_means,
            "id_std": entropy_id_stds,
            "ood_mean": attn_entropy_ood_means,
            "ood_std": entropy_ood_stds
        },
        "intrinsic_dimensionality": {
            "id": intrinsic_dim_id,
            "ood": intrinsic_dim_ood
        },
        "residual_update_magnitude": {
            "transition_layers": list(update_layers),
            "id_mean": update_id_means,
            "ood_mean": update_ood_means
        }
    }

    with open(json_summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)

    print(f"\nMechanistic interpretability pipeline complete!")
    print(f"Generated plots saved in: {results_dir}/")
    print(f"JSON summary saved to: {json_summary_path}")

if __name__ == '__main__':
    main()
