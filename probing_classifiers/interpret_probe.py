import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.stats import pearsonr

# Import local modules
sys.path.insert(0, "/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_ood")
sys.path.append("/home/psquare_a6000/Desktop/conformal_ood")
from data_generators import generate_matern_gp_batch, generate_periodic_batch, generate_chirp_batch
from pfn_hooks import PFNHookManager

# Premium styling
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
    print(f"Using device: {device}")

    # Paths
    checkpoint_path = "/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_test_run/checkpoints/onefeature_gp_ls.1_pnf_4M.pt"
    base_dir = "/home/psquare_a6000/Desktop/conformal_ood/probing_classifiers"
    model_path = os.path.join(base_dir, "models", "logistic_regression_layer_1.joblib")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train_probes.py first.")
        return

    # 1. Load linear probe
    print("Loading Layer 1 linear probe...")
    clf = joblib.load(model_path)
    w = clf.coef_[0]  # shape (256,)
    b = clf.intercept_[0]
    
    # 2. Generate 100 ID, 50 periodic, 50 chirp samples (200 total)
    print("Generating evaluation samples...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    x_id, y_id = generate_matern_gp_batch(batch_size=100, num_points=100, device="cpu")
    x_p, y_p = generate_periodic_batch(batch_size=50, num_points=100, device="cpu")
    x_c, y_c = generate_chirp_batch(batch_size=50, num_points=100, device="cpu")
    
    x_all = torch.cat([x_id, x_p, x_c], dim=0) # (200, 100, 1)
    y_all = torch.cat([y_id, y_p, y_c], dim=0) # (200, 100)
    
    # Category labels: 0 = ID (GP), 1 = Periodic, 2 = Chirp
    categories = np.array([0] * 100 + [1] * 50 + [2] * 50)
    
    # 3. Extract Layer 1 Activations
    hook_manager = PFNHookManager(checkpoint_path=checkpoint_path, device=device)
    hook_manager.model.eval()
    
    layer_outputs = []
    
    # Hook output of layer 1
    def hook_fn(module, inp, out):
        layer_outputs.append(out.detach().cpu())
        
    h = hook_manager.model.transformer_encoder.layers[1].register_forward_hook(hook_fn)
    
    # Run forward pass (all 200 in one batch)
    src_x = x_all.transpose(0, 1).to(device)
    src_y = y_all.transpose(0, 1).to(device)
    
    with torch.no_grad():
        _ = hook_manager.model((src_x, src_y), single_eval_pos=100)
        
    h.remove()
    
    # Layer 1 activations shape: [seq_len=100, batch_size=200, hidden_dim=256]
    z1 = layer_outputs[0]
    seq_len, batch_size, hidden_dim = z1.shape
    
    # 4. Project activations onto probe weight vector w
    # projection[t, b] = w^T * z_1(t, b)
    # z1.numpy() shape: (100, 200, 256)
    z1_np = z1.numpy()
    projections = np.zeros((seq_len, batch_size))
    for t in range(seq_len):
        for b in range(batch_size):
            projections[t, b] = np.dot(w, z1_np[t, b, :])
            
    # Calculate average projection per sample
    mean_projections = np.mean(projections, axis=0) # (200,)
    
    # 5. Compute local features of the input curves
    # For each sample, we compute:
    # - Absolute target values: |y_t|
    # - Local differences (derivative proxy): |y_t - y_{t-1}| / |x_t - x_{t-1}|
    # We sort the points by x_t to get ordered paths
    corrs_abs_y = []
    corrs_deriv = []
    
    for b in range(batch_size):
        x_b = x_all[b].numpy().flatten()
        y_b = y_all[b].numpy().flatten()
        proj_b = projections[:, b]
        
        # Sort by x to compute proper derivatives along the curve
        sort_idx = np.argsort(x_b)
        x_sorted = x_b[sort_idx]
        y_sorted = y_b[sort_idx]
        proj_sorted = proj_b[sort_idx]
        
        # Local differences
        dy = np.abs(np.diff(y_sorted))
        dx = np.abs(np.diff(x_sorted))
        # Add epsilon to dx to avoid division by zero
        deriv = dy / (dx + 1e-8)
        
        # Pearson correlations
        r_y, _ = pearsonr(proj_sorted, np.abs(y_sorted))
        corrs_abs_y.append(r_y)
        
        # Correlation with derivative (align elements to match length)
        r_d, _ = pearsonr(proj_sorted[1:], deriv)
        corrs_deriv.append(r_d)
        
    corrs_abs_y = np.array(corrs_abs_y)
    corrs_deriv = np.array(corrs_deriv)
    
    print("\n--- Pearson Correlation Coefficients between Projection and Local Curve Features ---")
    print(f"All Samples   -> Correlation with |y|: {np.nanmean(corrs_abs_y):.4f} | Correlation with |dy/dx|: {np.nanmean(corrs_deriv):.4f}")
    
    # Group by category
    cats_names = ["ID (GP)", "Periodic OOD", "Chirp OOD"]
    for cat_idx, name in enumerate(cats_names):
        idx_mask = (categories == cat_idx)
        print(f"{name:<13} -> Correlation with |y|: {np.nanmean(corrs_abs_y[idx_mask]):.4f} | Correlation with |dy/dx|: {np.nanmean(corrs_deriv[idx_mask]):.4f}")
        
    # 6. Plot Color-Coded Function Curves
    # We select 2 representative samples of each category and plot them, color-coded by projection value
    fig, axes = plt.subplots(3, 2, figsize=(14, 15), dpi=300)
    
    cat_samples = {
        0: [10, 15],  # ID (GP)
        1: [105, 110], # Periodic OOD
        2: [155, 160]  # Chirp OOD
    }
    
    for cat_idx, name in enumerate(cats_names):
        samples = cat_samples[cat_idx]
        for col_idx, b in enumerate(samples):
            ax = axes[cat_idx, col_idx]
            
            x_b = x_all[b].numpy().flatten()
            y_b = y_all[b].numpy().flatten()
            proj_b = projections[:, b]
            
            # Sort by x
            sort_idx = np.argsort(x_b)
            x_sorted = x_b[sort_idx]
            y_sorted = y_b[sort_idx]
            proj_sorted = proj_b[sort_idx]
            
            # Scatter plot color-coded by projection
            sc = ax.scatter(x_sorted, y_sorted, c=proj_sorted, cmap='coolwarm', s=45, edgecolor='black', linewidth=0.5, zorder=3, vmin=-3.0, vmax=3.0)
            ax.plot(x_sorted, y_sorted, color='gray', alpha=0.4, linewidth=1.5, zorder=2)
            
            ax.set_title(f"{name} Sample {b} (Avg Projection: {mean_projections[b]:.2f})", fontsize=11, fontweight='bold')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            
            # Add colorbar to each row
            if col_idx == 1:
                fig.colorbar(sc, ax=ax, label="Probe Projection Value ($w^T z_1$)")
                
    plt.suptitle("Layer 1 Probing Projection Visualizations (Trained PFN)\nRed = Positive (OOD Signal) | Blue = Negative (ID/GP Signal)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "probe_activation_curves.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(base_dir, "probe_activation_curves.pdf"), format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 7. Scatter Plots of Projections vs. Local Features
    fig_scat, axes_scat = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Flatten all points for plotting scatter
    proj_flat = projections.flatten()
    y_flat = np.abs(y_all.numpy().flatten())
    
    # Compute derivative flat
    deriv_flat_list = []
    proj_deriv_flat_list = []
    cat_flat_list = []
    
    for b in range(batch_size):
        x_b = x_all[b].numpy().flatten()
        y_b = y_all[b].numpy().flatten()
        proj_b = projections[:, b]
        
        sort_idx = np.argsort(x_b)
        x_sorted = x_b[sort_idx]
        y_sorted = y_b[sort_idx]
        proj_sorted = proj_b[sort_idx]
        
        dy = np.abs(np.diff(y_sorted))
        dx = np.abs(np.diff(x_sorted))
        deriv = dy / (dx + 1e-8)
        
        deriv_flat_list.extend(deriv)
        proj_deriv_flat_list.extend(proj_sorted[1:])
        cat_flat_list.extend([categories[b]] * len(deriv))
        
    deriv_flat = np.array(deriv_flat_list)
    proj_deriv_flat = np.array(proj_deriv_flat_list)
    cat_flat = np.array(cat_flat_list)
    
    # Plot A: Projection vs |y|
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    labels_scat = ['ID (GP)', 'Periodic OOD', 'Chirp OOD']
    
    ax = axes_scat[0]
    for c in range(3):
        mask_c = (categories == c)
        # Repeat mask over sequence dimension
        mask_flat = np.repeat(mask_c, seq_len)
        ax.scatter(y_flat[mask_flat], proj_flat[mask_flat], alpha=0.3, color=colors[c], label=labels_scat[c], s=10)
    ax.set_title("Probe Projection vs. Absolute Target Value $|y|$", fontweight='bold')
    ax.set_xlabel("$|y|$")
    ax.set_ylabel("Projection Value ($w^T z_1$)")
    ax.legend()
    
    # Plot B: Projection vs |dy/dx|
    ax = axes_scat[1]
    for c in range(3):
        mask_c = (cat_flat == c)
        ax.scatter(deriv_flat[mask_c], proj_deriv_flat[mask_c], alpha=0.3, color=colors[c], label=labels_scat[c], s=10)
    ax.set_xscale('log')
    ax.set_title("Probe Projection vs. Local Derivative $|dy/dx|$ (Log Scale)", fontweight='bold')
    ax.set_xlabel("Local Derivative $|dy/dx|$")
    ax.set_ylabel("Projection Value ($w^T z_1$)")
    ax.legend()
    
    plt.suptitle("Linear Probe Projection Feature Correlations", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "probe_feature_correlations.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(base_dir, "probe_feature_correlations.pdf"), format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Interpretability plots saved inside {base_dir}/")

if __name__ == "__main__":
    main()
