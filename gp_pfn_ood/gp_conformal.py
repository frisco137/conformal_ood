import os
os.environ["OPENBLAS_NUM_THREADS"] = "1" # Limit OpenBLAS threads to avoid memory region allocation errors
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap
from typing import Tuple

# Import local modules
from data_generators import get_id_and_ood_data, generate_matern_gp_batch
from pfn_hooks import PFNHookManager

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
    checkpoint_path = "gp_pfn_test_run/checkpoints/onefeature_gp_ls.1_pnf_4M.pt"
    results_dir = "gp_pfn_ood/results/conformal"
    os.makedirs(results_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Phase 1: Search for layer with max silhouette score
    # ---------------------------------------------------------
    print("\n--- Phase 1: Search for Best Layer ---")
    search_samples_per_class = 10000
    x_id, y_id, x_ood, y_ood = get_id_and_ood_data(
        num_samples_per_class=search_samples_per_class,
        num_points=100,
        device="cpu"
    )

    x_search = torch.cat([x_id, x_ood], dim=0)
    y_search = torch.cat([y_id, y_ood], dim=0)
    labels_search = np.array([0] * search_samples_per_class + [1] * search_samples_per_class)

    hook_manager = PFNHookManager(checkpoint_path=checkpoint_path, device=device)

    batch_size = 128
    with torch.no_grad():
        for i in tqdm(range(0, len(x_search), batch_size), desc="Search Inference"):
            bx = x_search[i : i + batch_size].to(device)
            by = y_search[i : i + batch_size].to(device)
            src_x = bx.transpose(0, 1)
            src_y = by.transpose(0, 1)
            # using 100 context size like in original file
            _ = hook_manager.model((src_x, src_y), single_eval_pos=100)

    layer_activations, _ = hook_manager.get_aggregated_features()
    hook_manager.remove_hooks()

    num_layers = len(layer_activations)
    best_layer = -1
    max_silhouette = -1.0
    best_umap_model = None

    for l in range(num_layers):
        X_layer = layer_activations[l].numpy()
        score = silhouette_score(X_layer, labels_search)
        print(f"Layer {l} Silhouette Score: {score:.4f}")

        # --- Dimension Reduction Plots for search layer ---
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
            ax.scatter(coords[labels_search == 0, 0], coords[labels_search == 0, 1], alpha=0.5, label='ID (GP)', color='#1f77b4', s=4)
            ax.scatter(coords[labels_search == 1, 0], coords[labels_search == 1, 1], alpha=0.5, label='OOD', color='#ff7f0e', s=4)
            ax.set_title(f"{name} Representation")
            ax.legend(loc="upper right")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

        fig.suptitle(f"Transformer Encoder Layer {l} Manifold Separation (ID vs. OOD)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"layer_{l}_dim_reduction.png"), dpi=300)
        plt.close()

        if score > max_silhouette:
            max_silhouette = score
            best_layer = l

    print(f"=> Maximum separation at Layer {best_layer} (Score: {max_silhouette:.4f})")

    # Fit and save UMAP for the best layer
    print(f"Fitting UMAP on Layer {best_layer} representations...")
    best_umap_model = umap.UMAP(n_components=2, random_state=42, n_jobs=-1)
    best_umap_model.fit(layer_activations[best_layer].numpy())

    # ---------------------------------------------------------
    # Phase 2: Generate 10k ID samples for cluster center & dists
    # ---------------------------------------------------------
    print("\n--- Phase 2: Generating 10k Calibration Samples ---")
    n_calib = 10000

    # Depending on memory, we might need to batch data generation
    batch_size_gen = 1000
    x_calib_list = []
    y_calib_list = []
    for _ in range(n_calib // batch_size_gen):
        bx, by = generate_matern_gp_batch(batch_size_gen, num_points=100, device="cpu")
        x_calib_list.append(bx)
        y_calib_list.append(by)

    x_calib = torch.cat(x_calib_list, dim=0)
    y_calib = torch.cat(y_calib_list, dim=0)

    hook_manager = PFNHookManager(checkpoint_path=checkpoint_path, device=device)

    with torch.no_grad():
        for i in tqdm(range(0, n_calib, batch_size), desc="Calibration Inference"):
            bx = x_calib[i : i + batch_size].to(device)
            by = y_calib[i : i + batch_size].to(device)
            src_x = bx.transpose(0, 1)
            src_y = by.transpose(0, 1)
            _ = hook_manager.model((src_x, src_y), single_eval_pos=100)

    calib_activations, _ = hook_manager.get_aggregated_features()
    hook_manager.remove_hooks()

    best_layer_calib_act = calib_activations[best_layer].numpy()

    print("Transforming calibration set with UMAP...")
    calib_umap = best_umap_model.transform(best_layer_calib_act)

    # Calculate Cluster Center
    cluster_center = calib_umap.mean(axis=0)
    print(f"Cluster Center: {cluster_center}")

    # Calculate distances
    calib_dists = np.linalg.norm(calib_umap - cluster_center, axis=1)

    # ---------------------------------------------------------
    # Phase 3: Evaluate on 100 ID and 100 OOD samples
    # ---------------------------------------------------------
    print("\n--- Phase 3: Evaluating New Test Samples ---")
    n_test = 100
    x_test_id, y_test_id, x_test_ood, y_test_ood = get_id_and_ood_data(
        num_samples_per_class=n_test,
        num_points=100,
        device="cpu"
    )

    x_test = torch.cat([x_test_id, x_test_ood], dim=0)
    y_test = torch.cat([y_test_id, y_test_ood], dim=0)
    test_labels = np.array([0] * n_test + [1] * n_test) # 0 for ID, 1 for OOD

    hook_manager = PFNHookManager(checkpoint_path=checkpoint_path, device=device)

    with torch.no_grad():
        for i in tqdm(range(0, len(x_test), batch_size), desc="Test Inference"):
            bx = x_test[i : i + batch_size].to(device)
            by = y_test[i : i + batch_size].to(device)
            src_x = bx.transpose(0, 1)
            src_y = by.transpose(0, 1)
            _ = hook_manager.model((src_x, src_y), single_eval_pos=100)

    test_activations, _ = hook_manager.get_aggregated_features()
    hook_manager.remove_hooks()

    best_layer_test_act = test_activations[best_layer].numpy()
    print("Transforming test set with UMAP...")
    test_umap = best_umap_model.transform(best_layer_test_act)

    test_dists = np.linalg.norm(test_umap - cluster_center, axis=1)

    # Plot Scatter
    plt.figure(figsize=(10, 8))
    plt.scatter(calib_umap[:, 0], calib_umap[:, 1], c='blue', alpha=0.1, s=4, label='10k ID (Calib)')
    plt.scatter(test_umap[test_labels == 0, 0], test_umap[test_labels == 0, 1], c='red', alpha=0.8, s=20, label='100 ID (Test)')
    plt.scatter(test_umap[test_labels == 1, 0], test_umap[test_labels == 1, 1], c='green', alpha=0.8, s=20, label='100 OOD (Test)')
    plt.scatter(cluster_center[0], cluster_center[1], c='black', marker='X', s=100, label='Cluster Center')
    plt.legend()
    plt.title(f"UMAP Projection of Layer {best_layer} Representations")
    plt.savefig(os.path.join(results_dir, "umap_projection.png"), dpi=300)
    plt.close()

    # Calculate Percentage Guarantee
    print("Calculating Percentage Guarantees...")
    guarantees = []

    for dist in test_dists:
        num_greater = np.sum(calib_dists > dist)
        guarantee = 100.0 * (1.0 + num_greater) / (n_calib + 1.0)
        guarantees.append(guarantee)

    guarantees = np.array(guarantees)

    id_guarantees = guarantees[test_labels == 0]
    ood_guarantees = guarantees[test_labels == 1]

    avg_id_guarantee = np.mean(id_guarantees)
    avg_ood_guarantee = np.mean(ood_guarantees)

    print(f"Average Guarantee (ID Test): {avg_id_guarantee:.2f}%")
    print(f"Average Guarantee (OOD Test): {avg_ood_guarantee:.2f}%")

    # Plot Bar Chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(["ID Test", "OOD Test"], [avg_id_guarantee, avg_ood_guarantee], color=['red', 'green'])
    plt.ylabel("Average Percentage Guarantee (%)")
    plt.title("Conformal Guarantee by Dataset Type")
    plt.ylim(0, 100)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f"{height:.2f}%", ha='center', va='bottom', fontweight='bold')

    plt.savefig(os.path.join(results_dir, "conformal_guarantee.png"), dpi=300)
    plt.close()

    # Plot Candlestick Chart Based on Quantiles
    def plot_candlestick(ax, x_pos, data, color, label):
        q10, q25, q50, q75, q90 = np.percentile(data, [10, 25, 50, 75, 90])
        ax.plot([x_pos, x_pos], [q10, q90], color='black', linewidth=1.5)
        rect = plt.Rectangle((x_pos - 0.2, q25), 0.4, q75 - q25, facecolor=color, edgecolor='black', label=label)
        ax.add_patch(rect)
        ax.plot([x_pos - 0.2, x_pos + 0.2], [q50, q50], color='black', linewidth=1.5)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    plot_candlestick(ax, 1, id_guarantees, color='red', label='ID Test')
    plot_candlestick(ax, 2, ood_guarantees, color='green', label='OOD Test')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['ID Test', 'OOD Test'])
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(max(0, min(np.min(id_guarantees), np.min(ood_guarantees)) - 5), 105)
    ax.set_ylabel("Conformal Guarantee (%)")
    ax.set_title("Conformal Guarantees (Low=10th, Open=25th, Close=75th, High=90th)")

    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.savefig(os.path.join(results_dir, "conformal_guarantee_candlestick.png"), dpi=300)
    plt.close()

    print(f"Finished! Plots saved to {results_dir}")

if __name__ == "__main__":
    main()
