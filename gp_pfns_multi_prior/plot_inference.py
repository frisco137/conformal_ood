import os
import sys
import math
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add PFN repo to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TransformersCanDoBayesianInference')))

import encoders
import positional_encodings
from transformer import TransformerModel
import data_generators

# Premium styling
plt.rcParams.update({
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white'
})

def run_inference(model, device, X, y, n_train):
    """
    Runs PFN inference. 
    Inputs:
    - X: shape (num_points, 1)
    - y: shape (num_points,)
    - n_train: number of context points
    """
    model.eval()
    with torch.no_grad():
        # Shape formatting: (seq_len, batch_size, dim)
        X_train_seq = X[:n_train].unsqueeze(1) # (n_train, 1, 1)
        X_test_seq = X[n_train:].unsqueeze(1)  # (n_test, 1, 1)
        y_train_seq = y[:n_train].unsqueeze(1) # (n_train, 1)
        
        src_x = torch.cat([X_train_seq, X_test_seq], dim=0).to(device)
        src_y = y_train_seq.to(device)
        
        # Forward pass: shape (n_test, 1, 1)
        output = model((src_x, src_y), single_eval_pos=n_train)
        
        # Return predicted mean
        return output.squeeze(-1).squeeze(-1).cpu().numpy()

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running plotting inference on: {device}")
    
    # Architecture params
    emsize = 256
    nhead = 4
    nhid = 512
    nlayers = 6
    seq_len = 100 # match trained model max length
    n_train = 30  # number of context points
    
    checkpoint_dir = "/home/psquare_a6000/Desktop/conformal_ood/gp_pfns_multi_prior/checkpoints"
    model_types = ["rbf_matern", "rbf_periodic", "matern_periodic", "all_three"]
    
    # Load all models
    models = {}
    for m_type in model_types:
        checkpoint_path = os.path.join(checkpoint_dir, f"{m_type}.pt")
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found for {m_type}")
            continue
        
        encoder = encoders.Linear(1, emsize)
        y_encoder = encoders.Linear(1, emsize)
        pos_encoder = positional_encodings.PositionalEncoding(emsize, seq_len * 2)
        
        model = TransformerModel(
            encoder=encoder,
            n_out=1,
            ninp=emsize,
            nhead=nhead,
            nhid=nhid,
            nlayers=nlayers,
            dropout=0.0,
            y_encoder=y_encoder,
            pos_encoder=pos_encoder
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        models[m_type] = model
        
    if len(models) == 0:
        print("No models loaded! Exiting.")
        sys.exit(1)
        
    # Set seed for reproducible curves
    torch.manual_seed(9876)
    np.random.seed(9876)
    
    # Generate test curves (1 curve per prior, batch size = 1)
    priors = {
        "RBF GP": data_generators.generate_rbf_gp_batch,
        "Matérn 5/2 GP": data_generators.generate_matern_gp_batch,
        "Periodic GP": data_generators.generate_periodic_gp_batch
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(11, 16), dpi=300, sharex=True)
    
    # Custom color palette for the 4 models
    colors = {
        "rbf_matern": "#e53e3e",      # Red
        "rbf_periodic": "#3182ce",    # Blue
        "matern_periodic": "#38a169", # Green
        "all_three": "#805ad5"        # Purple
    }
    
    for idx, (p_name, p_fn) in enumerate(priors.items()):
        ax = axes[idx]
        
        # Generate batch of 1 sample
        x_gen, y_gen = p_fn(batch_size=1, num_points=seq_len, device="cpu")
        x_gen = x_gen.squeeze(0) # (seq_len, 1)
        y_gen = y_gen.squeeze(0) # (seq_len,)
        
        # Sort all points for plotting true underlying curve
        x_np = x_gen.numpy().flatten()
        y_np = y_gen.numpy().flatten()
        sort_all = np.argsort(x_np)
        x_all_sorted = x_np[sort_all]
        y_all_sorted = y_np[sort_all]
        
        # Context / training observations
        x_train = x_gen[:n_train]
        y_train = y_gen[:n_train]
        
        # Test observations
        x_test = x_gen[n_train:]
        y_test = y_gen[n_train:]
        x_test_np = x_test.numpy().flatten()
        sort_test = np.argsort(x_test_np)
        x_test_sorted = x_test_np[sort_test]
        
        # Plot True Function & Context
        ax.plot(x_all_sorted, y_all_sorted, color='#718096', linestyle='--', linewidth=1.5, label='True Function')
        ax.scatter(x_train.numpy().flatten(), y_train.numpy().flatten(), color='black', s=45, zorder=5, label=f'Context Data (N={n_train})')
        
        # Run inference and plot predicted mean for each model
        for m_type, model in models.items():
            pred_mean = run_inference(model, device, x_gen, y_gen, n_train)
            pred_mean_sorted = pred_mean[sort_test]
            
            ax.plot(x_test_sorted, pred_mean_sorted, color=colors[m_type], linewidth=2.5, label=f'PFN ({m_type.upper()})')
            
        ax.set_title(f"Evaluation on {p_name}", fontsize=13, fontweight='bold', color="#2d3748")
        ax.set_ylabel("y")
        ax.set_ylim(-3.5, 3.5)
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
        
    axes[-1].set_xlabel("x")
    plt.suptitle("PFN Inference Predictions on Different Gaussian Process Priors", fontsize=15, fontweight='bold', color="#1a202c", y=0.99)
    plt.tight_layout()
    
    # Save plots
    artifact_dir = "/home/psquare_a6000/.gemini/antigravity/brain/3744a7c6-ba28-4e40-86b9-acc13fb62a92/artifacts"
    os.makedirs(artifact_dir, exist_ok=True)
    
    png_path = os.path.join(artifact_dir, "multi_prior_inference_plots.png")
    pdf_path = os.path.join(artifact_dir, "multi_prior_inference_plots.pdf")
    
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved inference plots to:\n - {png_path}\n - {pdf_path}")

if __name__ == "__main__":
    main()
