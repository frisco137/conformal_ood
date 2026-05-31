import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure the parent directory is in path to import from the repository
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gp_pfn_sm.test_suites import (
    generate_id_low_noise,
    generate_id_high_noise,
    generate_near_ood,
    generate_far_ood,
    generate_real_world_ood
)

def main():
    print("Generating data samples...")
    # Generate 3 samples for each of the 5 suites
    device = "cpu"
    
    suites = {
        "Suite 1: ID Low Noise": (generate_id_low_noise, "#3b82f6"),
        "Suite 2: ID High Noise": (generate_id_high_noise, "#64748b"),
        "Suite 3: Near-OOD": (generate_near_ood, "#f59e0b"),
        "Suite 4: Far-OOD": (generate_far_ood, "#ef4444"),
        "Suite 5: Real-World OOD": (generate_real_world_ood, "#8b5cf6")
    }
    
    fig, axes = plt.subplots(5, 3, figsize=(12, 13), dpi=300, sharex=True)
    
    for row_idx, (suite_name, (generator_fn, color)) in enumerate(suites.items()):
        print(f"Generating samples for {suite_name}...")
        xs, ys = generator_fn(batch_size=3, device=device)
        
        # xs shape: (3, 100, 1)
        # ys shape: (3, 100)
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            
            x_val = xs[col_idx, :, 0].numpy()
            y_val = ys[col_idx, :].numpy()
            
            # Sort by x for clean line plotting
            sort_idx = np.argsort(x_val)
            x_sorted = x_val[sort_idx]
            y_sorted = y_val[sort_idx]
            
            # Plot line and scatter
            ax.plot(x_sorted, y_sorted, color=color, alpha=0.4, linewidth=1.5)
            ax.scatter(x_sorted, y_sorted, color=color, s=12, alpha=0.8)
            
            # Clean layout
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.set_ylim(-3.5, 3.5)
            
            # Titles for columns and rows
            if col_idx == 1:
                ax.set_title(suite_name, fontweight="bold", fontsize=11, pad=8)
            if row_idx == 4:
                ax.set_xlabel("x", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel("y", fontsize=9)
                
    plt.tight_layout()
    plot_path = "backprop_gp_pfn/data_samples_visualization.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    print(f"Visualization saved to {plot_path}")

if __name__ == "__main__":
    main()
