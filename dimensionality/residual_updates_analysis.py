import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# Add module paths
sys.path.insert(0, "/home/psquare_a6000/Desktop/conformal_ood/TransformersCanDoBayesianInference")
sys.path.append("/home/psquare_a6000/Desktop/conformal_ood")

from gp_pfn_ood.data_generators import (
    generate_matern_gp_batch,
    generate_periodic_batch,
    generate_chirp_batch
)

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    checkpoint_path = "/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_test_run/checkpoints/onefeature_gp_ls.1_pnf_4M.pt"
    print(f"Loading pre-trained PFN from {checkpoint_path}...")
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Compatibility fixes
    for m in model.modules():
        if m.__class__.__name__ == 'TransformerEncoderLayer':
            if not hasattr(m, 'norm_first'):
                m.norm_first = False
        elif m.__class__.__name__ == 'GELU':
            if not hasattr(m, 'approximate'):
                m.approximate = 'none'
    model.eval()
    model.to(device)

    # Dictionary to cache activations per hook call
    layer_inputs = {}
    layer_outputs = {}

    def get_hook(layer_idx):
        def hook(module, inp, out):
            layer_inputs[layer_idx] = inp[0].detach().cpu()
            layer_outputs[layer_idx] = out.detach().cpu()
        return hook

    # Register hooks on all 6 layers
    hooks = []
    for idx, layer in enumerate(model.transformer_encoder.layers):
        h = layer.register_forward_hook(get_hook(idx))
        hooks.append(h)

    # 5 random seeds
    seeds = [42, 100, 2026, 999, 12345]
    num_layers = len(model.transformer_encoder.layers)
    
    # Storage for results across seeds
    # Shape: (5_seeds, 6_layers)
    id_updates_all = np.zeros((len(seeds), num_layers))
    ood_updates_all = np.zeros((len(seeds), num_layers))

    for seed_idx, seed in enumerate(seeds):
        print(f"\nProcessing Seed {seed} ({seed_idx + 1}/{len(seeds)})...")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 1. Generate 100 ID samples
        x_id, y_id = generate_matern_gp_batch(batch_size=100, num_points=100, device=device)
        
        # 2. Generate 100 OOD samples (50 periodic, 50 chirp)
        x_p, y_p = generate_periodic_batch(batch_size=50, num_points=100, device=device)
        x_c, y_c = generate_chirp_batch(batch_size=50, num_points=100, device=device)
        x_ood = torch.cat([x_p, x_c], dim=0)
        y_ood = torch.cat([y_p, y_c], dim=0)

        # Run ID inference
        src_x_id = x_id.transpose(0, 1)
        src_y_id = y_id.transpose(0, 1)
        with torch.no_grad():
            _ = model((src_x_id, src_y_id), single_eval_pos=100)
        
        # Calculate ID updates
        for l in range(num_layers):
            # inp, out shapes: [seq_len=100, batch_size=100, hidden_dim=256]
            inp = layer_inputs[l]
            out = layer_outputs[l]
            update = out - inp
            # L2 norm over hidden dimension, shape [100, 100]
            norms = torch.norm(update, p=2, dim=-1)
            # Average over sequence length (dim 0) and batch (dim 1)
            avg_norm = torch.mean(norms).item()
            id_updates_all[seed_idx, l] = avg_norm

        # Run OOD inference
        src_x_ood = x_ood.transpose(0, 1)
        src_y_ood = y_ood.transpose(0, 1)
        with torch.no_grad():
            _ = model((src_x_ood, src_y_ood), single_eval_pos=100)
            
        # Calculate OOD updates
        for l in range(num_layers):
            inp = layer_inputs[l]
            out = layer_outputs[l]
            update = out - inp
            norms = torch.norm(update, p=2, dim=-1)
            avg_norm = torch.mean(norms).item()
            ood_updates_all[seed_idx, l] = avg_norm

    # Remove hooks
    for h in hooks:
        h.remove()

    # Calculate mean and std dev across the 5 seeds
    id_mean = np.mean(id_updates_all, axis=0)
    id_std = np.std(id_updates_all, axis=0)
    ood_mean = np.mean(ood_updates_all, axis=0)
    ood_std = np.std(ood_updates_all, axis=0)

    # Print results
    print("\n" + "="*50)
    print(f"{'LAYER':<6} | {'ID MEAN':<10} | {'ID STD':<8} | {'OOD MEAN':<10} | {'OOD STD':<8}")
    print("="*50)
    for l in range(num_layers):
        print(f"Layer {l} | {id_mean[l]:<10.4f} | {id_std[l]:<8.4f} | {ood_mean[l]:<10.4f} | {ood_std[l]:<8.4f}")
    print("="*50)

    # Save to JSON and CSV
    results_dict = {
        "layers": list(range(num_layers)),
        "id_mean": list(id_mean),
        "id_std": list(id_std),
        "ood_mean": list(ood_mean),
        "ood_std": list(ood_std),
        "raw_id_updates": id_updates_all.tolist(),
        "raw_ood_updates": ood_updates_all.tolist()
    }

    folders = [
        "/home/psquare_a6000/Desktop/conformal_ood/dimensionality",
        "/home/psquare_a6000/Desktop/conformal_ood/dimnesionality"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        # JSON
        with open(os.path.join(folder, "residual_updates_summary.json"), 'w') as f:
            json.dump(results_dict, f, indent=4)
        # CSV
        df = pd.DataFrame({
            "Layer": range(num_layers),
            "ID_Mean": id_mean,
            "ID_Std": id_std,
            "OOD_Mean": ood_mean,
            "OOD_Std": ood_std
        })
        df.to_csv(os.path.join(folder, "residual_updates_summary.csv"), index=False)

    # Plot results
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(9, 6), dpi=300)

    layers_range = range(num_layers)
    # Plot ID
    plt.plot(layers_range, id_mean, marker='o', color='#1f77b4', linewidth=2.5, label='In-Distribution (GP)')
    plt.fill_between(layers_range, id_mean - id_std, id_mean + id_std, color='#1f77b4', alpha=0.15)

    # Plot OOD
    plt.plot(layers_range, ood_mean, marker='s', color='#ff7f0e', linewidth=2.5, label='Out-of-Distribution')
    plt.fill_between(layers_range, ood_mean - ood_std, ood_mean + ood_std, color='#ff7f0e', alpha=0.15)

    plt.title("Residual Stream Update Magnitude per Layer (5 Seeds)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Layer Index (0 to 5)", fontsize=11)
    plt.ylabel("Mean $L_2$ Norm of Layer Update", fontsize=11)
    plt.xticks(layers_range)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, fontsize=10, loc='upper left')
    plt.tight_layout()

    for folder in folders:
        plt.savefig(os.path.join(folder, "residual_updates_comparison.png"), bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(folder, "residual_updates_comparison.pdf"), format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Analysis saved inside {folders[0]}/")

if __name__ == "__main__":
    main()
