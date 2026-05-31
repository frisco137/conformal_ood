import sys
import os
import math
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory and PFN repo to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TransformersCanDoBayesianInference')))

import encoders
import positional_encodings
from transformer import TransformerModel
from gp_pfn_sm.data_generators import generate_spectral_mixture_gp_batch

@torch.no_grad()
def plot_inference_30(model, device, seq_len=100, eval_pos=30, save_path="sm_inference_plots_30.png"):
    """
    Evaluates PFN inference on 3 test curves using exactly 30 context points.
    """
    model.eval()
    
    # Generate test batch (batch_size = 3)
    x_test, y_test = generate_spectral_mixture_gp_batch(batch_size=3, num_points=seq_len, device="cpu")
    
    # Transpose to sequence-first: (seq_len, batch_size, dim) and (seq_len, batch_size)
    x_input = x_test.transpose(0, 1).to(device)
    y_input = y_test.transpose(0, 1).to(device)
    
    # Run PFN forward pass
    output = model((x_input, y_input), single_eval_pos=eval_pos) # (seq_len - eval_pos, batch_size, 1)
    preds = output.squeeze(-1).cpu().numpy() # (seq_len - eval_pos, batch_size)
    
    fig, axes = plt.subplots(3, 1, figsize=(11, 14), dpi=300, sharex=True)
    
    x_np = x_test.squeeze(-1).numpy() # (batch_size, seq_len)
    y_np = y_test.numpy() # (batch_size, seq_len)
    
    # We generated X in (batch_size, seq_len, 1), so for plotting we need to transpose x_np, y_np
    # x_np is (3, 100) and y_np is (3, 100)
    for b in range(3):
        ax = axes[b]
        
        # Sort all points for plotting true underlying curve
        xb = x_np[b]
        yb = y_np[b]
        sort_all = np.argsort(xb)
        x_sorted = xb[sort_all]
        y_sorted = yb[sort_all]
        
        # Split into train/context (first eval_pos points) and test/query (remaining points)
        # Note: in train_models.py:
        # train_x = x_src[:single_eval_pos] + y_src[:single_eval_pos]
        # and we permuted xs/ys inside data_generators, so the order is random.
        x_context = xb[:eval_pos]
        y_context = yb[:eval_pos]
        
        x_query = xb[eval_pos:]
        y_query = yb[eval_pos:]
        y_pred = preds[:, b]
        
        # Sort query points for clean line plotting of predictions
        sort_query = np.argsort(x_query)
        x_query_sorted = x_query[sort_query]
        y_pred_sorted = y_pred[sort_query]
        
        ax.plot(x_sorted, y_sorted, color='#718096', linestyle='--', linewidth=1.5, label='True Function')
        ax.scatter(x_context, y_context, color='black', s=45, zorder=5, label=f'Context Data (N={eval_pos})')
        ax.plot(x_query_sorted, y_pred_sorted, color='#06b6d4', linewidth=2.5, label='PFN Prediction (SM)')
        ax.scatter(x_query, y_query, color='#e53e3e', s=25, alpha=0.4, label='True Query Points')
        
        # Compute MSE on the query points
        query_mse = np.mean((y_pred - y_query) ** 2)
        
        ax.set_title(f"Spectral Mixture Inference Sample {b+1} (Query MSE: {query_mse:.4f})", fontsize=12, fontweight='bold')
        ax.set_ylabel("y")
        ax.set_ylim(-3.5, 3.5)
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
        
    axes[-1].set_xlabel("x")
    plt.suptitle("PFN Predictions with 30 Context Points", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Inference plots saved to {save_path}")

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Initialize PFN model matching architecture
    emsize = 256
    nhead = 4
    nhid = 512
    nlayers = 6
    seq_len = 100
    
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
    
    checkpoint_path = "/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_sm/checkpoints/spectral_mixture.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    
    save_path = "/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_sm/sm_inference_plots_30.png"
    plot_inference_30(model, device, seq_len=seq_len, eval_pos=30, save_path=save_path)

if __name__ == "__main__":
    main()
