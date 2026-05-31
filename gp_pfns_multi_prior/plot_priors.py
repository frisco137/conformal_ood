import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_generators import (
    generate_rbf_gp_batch, 
    generate_matern_gp_batch, 
    generate_periodic_gp_batch
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

def main():
    base_dir = "/home/psquare_a6000/Desktop/conformal_ood/gp_pfns_multi_prior"
    os.makedirs(base_dir, exist_ok=True)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    num_samples = 5
    num_points = 150 # Use slightly more points to ensure smooth curves when sorted
    
    # Generate batches
    x_rbf, y_rbf = generate_rbf_gp_batch(batch_size=num_samples, num_points=num_points, device="cpu")
    x_mat, y_mat = generate_matern_gp_batch(batch_size=num_samples, num_points=num_points, device="cpu")
    x_per, y_per = generate_periodic_gp_batch(batch_size=num_samples, num_points=num_points, device="cpu")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    
    priors_data = [
        (x_rbf, y_rbf, "Standard RBF GP Prior", "Smooth, infinitely differentiable\n$l \sim Gamma(3, 6)$"),
        (x_mat, y_mat, "Standard Matérn 5/2 GP Prior", "Moderately smooth curves\n$l \sim Gamma(3, 6)$"),
        (x_per, y_per, "Standard Periodic GP Prior", "Periodic (Exp-Sine-Squared) fluctuations\n$f \sim Uniform(0.2, 1.0)$, $l \sim Uniform(1.5, 3.0)$")
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (x, y, title, description) in enumerate(priors_data):
        ax = axes[idx]
        
        for i in range(num_samples):
            # Sort x for line plotting
            xi = x[i].flatten().numpy()
            yi = y[i].numpy()
            sort_idx = np.argsort(xi)
            
            ax.plot(xi[sort_idx], yi[sort_idx], color=colors[i], linewidth=2.0, alpha=0.85, label=f"Sample {i+1}")
            ax.scatter(xi, yi, color=colors[i], s=12, alpha=0.3, edgecolors='none')
            
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel("x")
        ax.set_ylabel("y (Normalized)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-3.0, 3.0)
        
        # Add description text box inside
        ax.text(0.5, -2.8, description, fontsize=9.5, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'), ha='center')
        
    plt.suptitle("Samples from Multi-Prior PFN Data Generators", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    plt.savefig(os.path.join(base_dir, "prior_samples.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(base_dir, "prior_samples.pdf"), format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Plot saved successfully in {base_dir}/")

if __name__ == "__main__":
    main()
