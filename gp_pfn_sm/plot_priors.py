import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_generators import generate_spectral_mixture_gp_batch

# Premium styling
plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False
})

def main():
    base_dir = "/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_sm"
    os.makedirs(base_dir, exist_ok=True)
    
    # Set seed for reproducible visualization
    torch.manual_seed(101)
    np.random.seed(101)
    
    num_samples = 6
    num_points = 200 # More points for smooth curve resolution
    
    # Generate batch from the Spectral Mixture GP prior
    xs, ys = generate_spectral_mixture_gp_batch(batch_size=num_samples, num_points=num_points, device="cpu")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), dpi=300)
    axes = axes.flatten()
    
    # Premium color palette
    colors = [
        '#4f46e5',  # Indigo
        '#06b6d4',  # Cyan
        '#10b981',  # Emerald
        '#f59e0b',  # Amber
        '#ef4444',  # Red
        '#8b5cf6'   # Purple
    ]
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Extract and sort for smooth line rendering
        x_val = xs[i].flatten().numpy()
        y_val = ys[i].numpy()
        sort_idx = np.argsort(x_val)
        
        x_sorted = x_val[sort_idx]
        y_sorted = y_val[sort_idx]
        
        # Plot continuous curve representing the function
        ax.plot(x_sorted, y_sorted, color=colors[i], linewidth=2.2, alpha=0.9, label=f"Sample {i+1}")
        
        # Scatter points (representing observed data from the function)
        ax.scatter(x_val, y_val, color=colors[i], s=15, alpha=0.25, edgecolors='none')
        
        ax.set_title(f"Spectral Mixture Sample {i+1}", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("y (Normalized)", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-3.2, 3.2)
        ax.legend(loc="upper right", frameon=True, framealpha=0.8, fontsize=9)
        
    plt.suptitle("Function Samples from Spectral Mixture GP Prior", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the plots
    png_path = os.path.join(base_dir, "sm_prior_samples.png")
    pdf_path = os.path.join(base_dir, "sm_prior_samples.pdf")
    
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Spectral Mixture prior samples plots saved successfully:")
    print(f" - PNG: {png_path}")
    print(f" - PDF: {pdf_path}")

if __name__ == "__main__":
    main()
