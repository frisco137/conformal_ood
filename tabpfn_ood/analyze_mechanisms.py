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
import argparse

# Import local modules
from generate_data import get_data_by_level
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

def main(level, total_samples_per_class, batch_size):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device for inference: {device}")

    # Paths configuration
    checkpoint_path = "tabpfn_ood/checkpoints/"
    filename = "tabpfn.cpkt"

    results_dir = f"tabpfn_ood/results/mechanistic_plots_tabpfn_level_{level}"
    json_summary_path = f"tabpfn_ood/results/mechanistic_summary_level_{level}.json"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Configuration
    samples_per_class_in_batch = batch_size // level
    num_batches = total_samples_per_class // samples_per_class_in_batch
    seq_len = 1000
    single_eval_pos = int(0.9 * seq_len) # 90% context, 10% query

    # Level-based class mappings and colors
    level_names = {
        2: ["In-Prior", "Out-Prior"],
        3: ["In-Prior (No Noise)", "In-Prior (Noise)", "Out-Prior"],
        4: ["In-Prior (No Noise)", "In-Prior (Noise)", "Borderline Out-Prior", "Out-Prior"],
        5: ["In-Prior (No Noise)", "In-Prior (Noise)", "Borderline Out-Prior", "Out-Prior", "Extreme Out-Prior"]
    }
    class_names = level_names[level]
    colors = plt.cm.get_cmap("tab10").colors[:level]
    markers = ['o', 's', '^', 'D', 'v'][:level]

    # 2. Hook and Model Manager Initialization
    hook_manager = PFNHookManager(checkpoint_path=checkpoint_path, filename=filename, device=device)

    # 3. Inference and On-the-fly Feature Extraction
    print(f"Running PFN model inference generated on-the-fly in {num_batches} batches...")
    labels_list = []

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Inference Batches"):
            datasets = get_data_by_level(
                level=level,
                num_samples_per_class=samples_per_class_in_batch,
                num_points=seq_len,
                num_features=100,
                num_classes=10
            )

            x_list = [d[0] for d in datasets]
            y_list = [d[1] for d in datasets]

            # Combine classes along the batch dimension (dim 1)
            src_x = torch.cat(x_list, dim=1).to(device)
            src_y = torch.cat(y_list, dim=1).to(device)

            for c_idx in range(level):
                labels_list.extend([c_idx] * samples_per_class_in_batch)

            # Run model forward pass to trigger hooks
            _ = hook_manager.model((src_x, src_y.float()), single_eval_pos=single_eval_pos)

    labels = np.array(labels_list)

    # Retrieve and aggregate the representations from CPU cache
    print("Extracting and aggregating hooked activations...")
    layer_activations, layer_attention_ctoc, layer_attention_qtoc = hook_manager.get_aggregated_features()
    hook_manager.remove_hooks()

    num_layers = len(layer_activations)
    print(f"Aggregated feature sizes for {num_layers} layers.")

    # Data storage for exporting
    silhouette_scores_mean: List[float] = []
    silhouette_scores_std: List[float] = []
    intrinsic_dims: Dict[int, List[int]] = {c: [] for c in range(level)}
    update_means: Dict[int, List[float]] = {c: [] for c in range(level)}

    # ------------------ Analysis A: Dimensionality Reduction ------------------
    print("\n--- Analysis A: Running Dimensionality Reduction (PCA, t-SNE, UMAP) ---")
    for l in range(num_layers):
        print(f"Processing Layer {l}...")
        X_layer = layer_activations[l].numpy()

        pca = PCA(n_components=2, random_state=42)
        x_pca = pca.fit_transform(X_layer)

        tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
        x_tsne = tsne.fit_transform(X_layer)

        reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=-1)
        x_umap = reducer.fit_transform(X_layer)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        methods = [("PCA", x_pca), ("t-SNE", x_tsne), ("UMAP", x_umap)]

        for idx, (name, coords) in enumerate(methods):
            ax = axes[idx]
            for c_idx in range(level):
                ax.scatter(coords[labels == c_idx, 0], coords[labels == c_idx, 1], alpha=0.5, label=class_names[c_idx], color=colors[c_idx], s=4)
            ax.set_title(f"{name} Representation")
            ax.legend(loc="upper right")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

        fig.suptitle(f"Transformer Encoder Layer {l} Manifold Separation (Level {level})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{results_dir}/layer_{l}_dim_reduction.png", dpi=300)
        plt.close()

    # ------------------ Analysis B: Silhouette Scores ------------------
    print("\n--- Analysis B: Computing Supervised Silhouette Scores with Error Bounds ---")
    n_bootstraps = 5
    sample_size = 2000

    for l in tqdm(range(num_layers), desc="Silhouette Evaluation"):
        X_layer = layer_activations[l].numpy()

        layer_scores = []
        for _ in range(n_bootstraps):
            indices = np.random.choice(len(X_layer), size=min(sample_size, len(X_layer)), replace=False)
            X_sample = X_layer[indices]
            labels_sample = labels[indices]
            # Skip if only one class
            if len(np.unique(labels_sample)) > 1:
                score = silhouette_score(X_sample, labels_sample)
                layer_scores.append(score)

        silhouette_scores_mean.append(float(np.mean(layer_scores)) if layer_scores else 0.0)
        silhouette_scores_std.append(float(np.std(layer_scores)) if layer_scores else 0.0)

    plt.figure(figsize=(8, 5))
    layers_range = range(num_layers)
    plt.plot(layers_range, silhouette_scores_mean, marker='o', linewidth=2.5, color='#2ca02c')
    plt.fill_between(
        layers_range,
        np.array(silhouette_scores_mean) - np.array(silhouette_scores_std),
        np.array(silhouette_scores_mean) + np.array(silhouette_scores_std),
        color='#2ca02c', alpha=0.2
    )
    plt.xlabel("Layer Depth")
    plt.ylabel("Supervised Silhouette Score")
    plt.title("Supervised Separation vs. Transformer Layer Depth", fontweight='bold')
    plt.xticks(layers_range)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/silhouette_scores.png", dpi=300)
    plt.close()

    # ------------------ Analysis C: Intrinsic Dimensionality ------------------
    print("\n--- Analysis C: Computing Intrinsic Dimensionality (PCA 95%) ---")
    for l in range(num_layers):
        X_layer = layer_activations[l].numpy()
        for c_idx in range(level):
            X_class = X_layer[labels == c_idx]
            pca_class = PCA()
            pca_class.fit(X_class)
            cumsum_class = np.cumsum(pca_class.explained_variance_ratio_)
            dim_class = int(np.argmax(cumsum_class >= 0.95) + 1)
            intrinsic_dims[c_idx].append(dim_class)

    plt.figure(figsize=(8, 5))
    for c_idx in range(level):
        plt.plot(range(num_layers), intrinsic_dims[c_idx], marker=markers[c_idx], label=class_names[c_idx], linewidth=2.0, color=colors[c_idx])
    plt.xlabel("Layer Depth")
    plt.ylabel("Intrinsic Dimensionality (95% Var)")
    plt.title("Intrinsic Feature Dimensionality vs. Layer Depth", fontweight='bold')
    plt.xticks(range(num_layers))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/intrinsic_dimensionality.png", dpi=300)
    plt.close()

    # ------------------ Analysis D: Attention Entropy ------------------
    print("\n--- Analysis D: Skipped (Attention Entropy Ignored for Now) ---")
    '''
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
    '''

    # ------------------ Analysis E: Residual Update Magnitude ------------------
    print("\n--- Analysis E: Computing Residual Update Magnitude ---")
    for l in range(1, num_layers):
        z_prev = layer_activations[l - 1]
        z_curr = layer_activations[l]
        diff = z_curr - z_prev
        norms = torch.norm(diff, p=2, dim=-1).numpy()
        for c_idx in range(level):
            update_means[c_idx].append(float(np.mean(norms[labels == c_idx])))

    plt.figure(figsize=(8, 5))
    update_layers = range(1, num_layers)
    for c_idx in range(level):
        plt.plot(update_layers, update_means[c_idx], marker=markers[c_idx], label=class_names[c_idx], linewidth=2.0, color=colors[c_idx])
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
        "silhouette_scores": {
            "mean": silhouette_scores_mean,
            "std": silhouette_scores_std
        },
        "intrinsic_dimensionality": {
            class_names[c_idx]: intrinsic_dims[c_idx] for c_idx in range(level)
        },
        "residual_update_magnitude": {
            "transition_layers": list(update_layers),
            **{class_names[c_idx]: update_means[c_idx] for c_idx in range(level)}
        }
    }

    with open(json_summary_path, 'w') as f:
        json.dump(summary_data, f, indent=4)

    print("\nMechanistic interpretability pipeline complete!")
    print(f"Generated plots saved in: {results_dir}/")
    print(f"JSON summary saved to: {json_summary_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=5, choices=range(2, 6), help="Level of generated data (2-5)")
    parser.add_argument("--total_samples_per_class", type=int, default=2000, help="Total samples per class")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for generating data")
    args = parser.parse_args()

    main(args.level, args.total_samples_per_class, args.batch_size)
