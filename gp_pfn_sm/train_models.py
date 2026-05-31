import sys
import os
import argparse
import time
import math
import torch
torch.set_num_threads(4)
import torch.nn as nn
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
from priors.utils import get_batch_to_dataloader
from utils import get_cosine_schedule_with_warmup, get_openai_lr
from gp_pfn_sm.data_generators import generate_spectral_mixture_gp_batch

def get_sm_batch_fn():
    """
    Returns a batch generation function for Spectral Mixture GP.
    Returns:
    - x_out: shape (seq_len, batch_size, 1)
    - y_out: shape (seq_len, batch_size)
    - target_out: shape (seq_len, batch_size)
    """
    def get_batch(batch_size, seq_len, num_features, device="cpu", hyperparameters=None):
        # Generate batch: x: (B, L, 1), y: (B, L)
        x, y = generate_spectral_mixture_gp_batch(batch_size=batch_size, num_points=seq_len, device=device)
        
        # Transpose to sequence-first: (L, B, 1) and (L, B)
        x_out = x.transpose(0, 1)
        y_out = y.transpose(0, 1)
        
        return x_out, y_out, y_out
        
    return get_batch

@torch.no_grad()
def evaluate_validation(model, device, batch_fn, seq_len=100, eval_pos=50, batch_size=256):
    """
    Evaluates the model on validation data with a fixed context size.
    Computes average MSE and Gaussian NLL.
    """
    model.eval()
    x_val, y_val, _ = batch_fn(batch_size=batch_size, seq_len=seq_len, num_features=1, device=device)
    
    # Predict at query index
    logits = model((x_val, y_val), single_eval_pos=eval_pos) # (seq_len - eval_pos, batch_size, 1)
    targets = y_val[eval_pos:].unsqueeze(-1)
    
    mse = torch.mean((logits - targets) ** 2).item()
    # NLL under noise variance sigma^2 = 0.01: 0.5 * ln(2 * pi * 0.01) + mse / 0.02
    nll = 0.5 * math.log(2.0 * math.pi * 0.01) + mse / 0.02
    
    return mse, nll

def plot_loss_curves(epochs, train_losses, val_losses, save_path):
    """
    Plots training and validation loss curves.
    """
    plt.rcParams.update({
        'font.size': 11,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white'
    })
    
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss (MSE)', color='#4f46e5', linewidth=2.0)
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss (MSE)', color='#ef4444', linewidth=2.0, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('PFN Spectral Mixture Training History', fontweight='bold', fontsize=13, pad=12)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to {save_path}")

@torch.no_grad()
def plot_inference_results(model, device, batch_fn, seq_len, eval_pos, save_path):
    """
    Runs inference on a few Spectral Mixture GP curves and plots predictions.
    """
    model.eval()
    x_test, y_test, _ = batch_fn(batch_size=3, seq_len=seq_len, num_features=1, device="cpu")
    
    # We want to plot the predictions
    # Shape of x_test is (seq_len, batch_size, 1), y_test is (seq_len, batch_size)
    x_input = x_test.to(device)
    y_input = y_test.to(device)
    
    # Forward pass: shape (seq_len - eval_pos, batch_size, 1)
    output = model((x_input, y_input), single_eval_pos=eval_pos)
    preds = output.squeeze(-1).cpu().numpy() # (seq_len - eval_pos, batch_size)
    
    fig, axes = plt.subplots(3, 1, figsize=(11, 14), dpi=300, sharex=True)
    
    x_np = x_test.squeeze(-1).numpy() # (seq_len, batch_size)
    y_np = y_test.numpy() # (seq_len, batch_size)
    
    for b in range(3):
        ax = axes[b]
        
        # Sort all points for plotting true underlying curve
        sort_all = np.argsort(x_np[:, b])
        x_sorted = x_np[sort_all, b]
        y_sorted = y_np[sort_all, b]
        
        # Split into train/context and test/query
        x_context = x_np[:eval_pos, b]
        y_context = y_np[:eval_pos, b]
        
        x_query = x_np[eval_pos:, b]
        y_query = y_np[eval_pos:, b]
        y_pred = preds[:, b]
        
        # Sort query points for clean line plotting of predictions
        sort_query = np.argsort(x_query)
        x_query_sorted = x_query[sort_query]
        y_pred_sorted = y_pred[sort_query]
        
        ax.plot(x_sorted, y_sorted, color='#718096', linestyle='--', linewidth=1.5, label='True Function')
        ax.scatter(x_context, y_context, color='black', s=40, zorder=5, label=f'Context Data (N={eval_pos})')
        ax.plot(x_query_sorted, y_pred_sorted, color='#8b5cf6', linewidth=2.5, label='PFN Prediction (SM)')
        ax.scatter(x_query, y_query, color='#e53e3e', s=25, alpha=0.4, label='True Query Points')
        
        ax.set_title(f"Spectral Mixture Inference Sample {b+1}", fontsize=12, fontweight='bold')
        ax.set_ylabel("y")
        ax.set_ylim(-3.5, 3.5)
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
        
    axes[-1].set_xlabel("x")
    plt.suptitle("PFN In-Context Predictions on Spectral Mixture GP Prior", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Inference plots saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Train PFN on Spectral Mixture GP prior")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (0 or 1)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="Steps per epoch")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=100, help="Sequence length (bptt)")
    parser.add_argument("--eval_pos", type=int, default=50, help="Fixed context split position")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate for training")
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"==========================================")
    print(f"Training PFN on Spectral Mixture GP Prior")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs} | Steps/Epoch: {args.steps_per_epoch}")
    print(f"Context size: {args.eval_pos} | Seq Len: {args.seq_len} | Peak LR: {args.lr}")
    print(f"==========================================")
    
    # 1. Build DataLoader
    batch_fn = get_sm_batch_fn()
    DataLoaderClass = get_batch_to_dataloader(batch_fn)
    DataLoaderClass.num_features = 1
    DataLoaderClass.num_outputs = 1
    dl = DataLoaderClass(
        num_steps=args.steps_per_epoch, 
        batch_size=args.batch_size, 
        seq_len=args.seq_len, 
        num_features=1
    )
    
    # 2. Build Model
    emsize = 256
    nhead = 4
    nhid = 512
    nlayers = 6
    
    encoder = encoders.Linear(1, emsize)
    y_encoder = encoders.Linear(1, emsize)
    pos_encoder = positional_encodings.PositionalEncoding(emsize, args.seq_len * 2)
    
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
    model.to(device)
    
    # 3. Setup optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    warmup_epochs = args.epochs // 5
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, args.epochs)
    
    criterion = nn.MSELoss(reduction='none')
    
    # Make output directory
    os.makedirs("/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_sm/checkpoints", exist_ok=True)
    os.makedirs("/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_sm/logs", exist_ok=True)
    
    best_val_mse = float("inf")
    train_losses = []
    val_losses = []
    
    # Start training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch, (data, targets) in enumerate(dl):
            # Format inputs
            inputs = tuple(e.to(device) for e in data) if isinstance(data, tuple) else data.to(device)
            targets = targets.to(device)
            
            # Forward pass with fixed eval position
            output = model(inputs, single_eval_pos=args.eval_pos)
            
            # Compute loss on query points only
            pred_targets = targets[args.eval_pos:]
            losses = criterion(output.flatten(), pred_targets.flatten())
            loss = losses.mean()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        
        train_mse = epoch_loss / args.steps_per_epoch
        train_losses.append(train_mse)
        
        # Validation
        val_mse, val_nll = evaluate_validation(
            model, device, batch_fn, seq_len=args.seq_len, eval_pos=args.eval_pos, batch_size=256
        )
        val_losses.append(val_mse)
        
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            train_nll = 0.5 * math.log(2.0 * math.pi * 0.01) + train_mse / 0.02
            print(f"Epoch {epoch:3d}/{args.epochs} | Train MSE: {train_mse:.4f} | Train NLL: {train_nll:.4f} | Val MSE: {val_mse:.4f} | Val NLL: {val_nll:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save checkpoints
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                checkpoint_path = "/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_sm/checkpoints/spectral_mixture.pt"
                torch.save(model.state_dict(), checkpoint_path)
                with open("/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_sm/logs/best_metadata.txt", "w") as f:
                    f.write(f"epoch={epoch}\nval_mse={val_mse:.6f}\nval_nll={val_nll:.6f}\n")
                    
    print(f"Training completed successfully. Best Val MSE: {best_val_mse:.6f}")
    
    # 4. Generate plots
    # Save training loss history
    history_path = "/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_sm/logs/training_history.npz"
    np.savez(history_path, train_losses=train_losses, val_losses=val_losses)
    
    # Plot training curves
    loss_plot_path = "/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_sm/training_loss.png"
    plot_loss_curves(args.epochs, train_losses, val_losses, loss_plot_path)
    
    # Load best model checkpoint for plotting inference
    model.load_state_dict(torch.load("/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_sm/checkpoints/spectral_mixture.pt", map_location=device))
    inference_plot_path = "/home/psquare_a6000/Desktop/conformal_ood/gp_pfn_sm/sm_inference_plots.png"
    plot_inference_results(model, device, batch_fn, args.seq_len, args.eval_pos, inference_plot_path)

if __name__ == "__main__":
    main()
