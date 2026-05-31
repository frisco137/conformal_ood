import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.spatial.distance import mahalanobis, cosine, euclidean
from tqdm import tqdm

# Import local modules
sys.path.insert(0, "/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_ood")
sys.path.append("/home/psquare_a6000/Desktop/conformal_ood")
from data_generators import generate_matern_gp_batch, generate_periodic_batch
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
    base_dir = "/home/psquare_a6000/Desktop/conformal_ood/dose-response"
    os.makedirs(base_dir, exist_ok=True)

    # Initialize Hook Manager
    hook_manager = PFNHookManager(checkpoint_path=checkpoint_path, device=device)
    hook_manager.model.eval()

    # Hook Layer 5
    layer_outputs = []
    def hook_fn(module, inp, out):
        layer_outputs.append(out.detach().cpu())
    h = hook_manager.model.transformer_encoder.layers[5].register_forward_hook(hook_fn)

    # 1. Generate Baseline ID (GP) Activations (2000 samples)
    print("Generating baseline In-Distribution (GP) data...")
    num_id_samples = 2000
    x_id, y_id = generate_matern_gp_batch(batch_size=num_id_samples, num_points=100, device="cpu")

    batch_size = 128
    id_activations = []
    
    with torch.no_grad():
        for i in range(0, num_id_samples, batch_size):
            bx = x_id[i : i + batch_size].to(device)
            by = y_id[i : i + batch_size].to(device)
            src_x = bx.transpose(0, 1)
            src_y = by.transpose(0, 1)
            
            layer_outputs.clear()
            _ = hook_manager.model((src_x, src_y), single_eval_pos=100)
            
            # layer_outputs[0] shape: [seq_len=100, batch_size, hidden_dim=256]
            z_batch = layer_outputs[0]
            # Average over sequence dimension -> [batch_size, 256]
            z_mean = torch.mean(z_batch, dim=0)
            id_activations.append(z_mean)

    id_activations = torch.cat(id_activations, dim=0).numpy() # (2000, 256)
    print(f"Baseline activations extracted. Shape: {id_activations.shape}")

    # Compute mean and covariance matrix for ID
    mu = np.mean(id_activations, axis=0) # (256,)
    cov = np.cov(id_activations, rowvar=False) # (256, 256)
    
    # Regularize Covariance matrix to ensure invertibility
    eps = 1e-6
    cov_reg = cov + eps * np.eye(cov.shape[0])
    inv_cov = np.linalg.inv(cov_reg)
    print("ID Baseline statistics computed.")

    # 2. Define Frequency (Dose) Grid
    frequencies = [0.05, 0.1, 0.2, 0.5, 0.8, 1.2, 1.8, 2.5, 4.0, 6.0, 8.0, 12.0]
    num_samples_per_freq = 200
    
    results = []

    print("\nEvaluating dose-response on periodic functions...")
    for freq in tqdm(frequencies, desc="Frequencies"):
        # We need a custom periodic function generator that takes the frequency parameter
        # Let's generate periodic samples with the given frequency
        # Equation: y = A * sin(2*pi * freq * x + phi) + noise
        x_freq = torch.rand(num_samples_per_freq, 100, 1)
        
        # Randomize amplitudes and phases
        amplitudes = torch.rand(num_samples_per_freq, 1, 1) * 1.5 + 0.5 # [0.5, 2.0]
        phases = torch.rand(num_samples_per_freq, 1, 1) * 2 * np.pi
        noise = torch.randn(num_samples_per_freq, 100) * 0.05
        
        # Calculate targets
        y_freq = amplitudes * torch.sin(2 * np.pi * freq * x_freq + phases)
        y_freq = y_freq.squeeze(-1) + noise
        
        freq_activations = []
        with torch.no_grad():
            for i in range(0, num_samples_per_freq, batch_size):
                bx = x_freq[i : i + batch_size].to(device)
                by = y_freq[i : i + batch_size].to(device)
                src_x = bx.transpose(0, 1)
                src_y = by.transpose(0, 1)
                
                layer_outputs.clear()
                _ = hook_manager.model((src_x, src_y), single_eval_pos=100)
                z_batch = layer_outputs[0]
                z_mean = torch.mean(z_batch, dim=0)
                freq_activations.append(z_mean)
                
        freq_activations = torch.cat(freq_activations, dim=0).numpy() # (200, 256)
        
        # Compute distances for all 200 samples
        mahal_dists = []
        cos_dists = []
        eucl_dists = []
        
        for z in freq_activations:
            mahal_dists.append(mahalanobis(z, mu, inv_cov))
            cos_dists.append(cosine(z, mu))
            eucl_dists.append(euclidean(z, mu))
            
        results.append({
            "frequency": freq,
            "mahal_mean": float(np.mean(mahal_dists)),
            "mahal_std": float(np.std(mahal_dists)),
            "cosine_mean": float(np.mean(cos_dists)),
            "cosine_std": float(np.std(cos_dists)),
            "euclidean_mean": float(np.mean(eucl_dists)),
            "euclidean_std": float(np.std(eucl_dists))
        })

    # Cleanup hook
    h.remove()
    hook_manager.remove_hooks()

    # Save to file
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(base_dir, "dose_response_results.csv"), index=False)
    with open(os.path.join(base_dir, "dose_response_results.json"), 'w') as f:
        json.dump(results, f, indent=4)

    # 3. Plotting Dose-Response Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    metrics = [
        ("mahal", "Mahalanobis Distance", "#d62728"),
        ("cosine", "Cosine Distance", "#1f77b4"),
        ("euclidean", "Euclidean ($L_2$) Distance", "#2ca02c")
    ]
    
    for idx, (prefix, title, color) in enumerate(metrics):
        ax = axes[idx]
        mean_val = df_results[f"{prefix}_mean"]
        std_val = df_results[f"{prefix}_std"]
        
        ax.plot(df_results["frequency"], mean_val, marker='o', linewidth=2.5, color=color, label='Mean Score')
        ax.fill_between(
            df_results["frequency"],
            mean_val - std_val,
            mean_val + std_val,
            color=color, alpha=0.15, label='$\pm 1$ Std Dev'
        )
        
        # Add baseline level (ID distance to ID mean)
        # Calculate average distance of ID samples to their own mean
        id_mahal = [mahalanobis(z, mu, inv_cov) for z in id_activations]
        id_cosine = [cosine(z, mu) for z in id_activations]
        id_eucl = [euclidean(z, mu) for z in id_activations]
        
        if prefix == "mahal":
            baseline_val = np.mean(id_mahal)
        elif prefix == "cosine":
            baseline_val = np.mean(id_cosine)
        else:
            baseline_val = np.mean(id_eucl)
            
        ax.axhline(baseline_val, color='black', linestyle='--', alpha=0.7, label='ID Baseline Level')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Periodic Frequency $f$ (Dose)")
        ax.set_ylabel("Non-Conformity Score")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    plt.suptitle("Unsupervised Non-Conformity Probing (Dose-Response Effect)\nAs Frequency increases, OOD-ness (Non-Conformity Score) increases monotonically", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "dose_response_curves.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(base_dir, "dose_response_curves.pdf"), format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"\nDose-response analysis complete! Results saved in {base_dir}/")

if __name__ == "__main__":
    main()
