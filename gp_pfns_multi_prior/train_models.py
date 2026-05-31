import sys
import os
import argparse
import time
import math
import torch
torch.set_num_threads(4)
import torch.nn as nn

# Add PFN repo to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TransformersCanDoBayesianInference')))

import encoders
import positional_encodings
from transformer import TransformerModel
from priors.utils import get_batch_to_dataloader
from utils import get_cosine_schedule_with_warmup, get_openai_lr, get_weighted_single_eval_pos_sampler
import data_generators

def get_mixture_batch_fn(model_type):
    """
    Returns a batch generation function for the specified mixture type.
    All samples are transposed to (seq_len, batch_size, 1) and (seq_len, batch_size) 
    to match the expected DataLoader shapes.
    """
    if model_type == "rbf_matern":
        prior_fns = [data_generators.generate_rbf_gp_batch, data_generators.generate_matern_gp_batch]
        proportions = [0.5, 0.5]
    elif model_type == "rbf_periodic":
        prior_fns = [data_generators.generate_rbf_gp_batch, data_generators.generate_periodic_gp_batch]
        proportions = [0.5, 0.5]
    elif model_type == "matern_periodic":
        prior_fns = [data_generators.generate_matern_gp_batch, data_generators.generate_periodic_gp_batch]
        proportions = [0.5, 0.5]
    elif model_type == "all_three":
        prior_fns = [
            data_generators.generate_rbf_gp_batch, 
            data_generators.generate_matern_gp_batch, 
            data_generators.generate_periodic_gp_batch
        ]
        proportions = [1.0/3.0, 1.0/3.0, 1.0/3.0]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    def get_batch(batch_size, seq_len, num_features, device="cpu", hyperparameters=None):
        # Determine batch count per prior
        counts = []
        remaining = batch_size
        for i, prop in enumerate(proportions):
            if i == len(proportions) - 1:
                counts.append(remaining)
            else:
                count = int(batch_size * prop)
                counts.append(count)
                remaining -= count
        
        xs = []
        ys = []
        for prior_fn, count in zip(prior_fns, counts):
            if count > 0:
                # Returns X: (count, seq_len, 1), y: (count, seq_len)
                x, y = prior_fn(batch_size=count, num_points=seq_len, device=device)
                xs.append(x)
                ys.append(y)
                
        # Concatenate
        X_cat = torch.cat(xs, dim=0) # (batch_size, seq_len, 1)
        y_cat = torch.cat(ys, dim=0) # (batch_size, seq_len)
        
        # Shuffle batch elements to mix priors
        perm = torch.randperm(batch_size, device=device)
        X_shuffled = X_cat[perm]
        y_shuffled = y_cat[perm]
        
        # Transpose to (seq_len, batch_size, 1) and (seq_len, batch_size)
        x_out = X_shuffled.transpose(0, 1)
        y_out = y_shuffled.transpose(0, 1)
        
        return x_out, y_out, y_out
        
    return get_batch

@torch.no_grad()
def evaluate_validation(model, device, batch_fn, seq_len=100, batch_size=256):
    """
    Evaluates the model on validation data.
    Computes average MSE and Gaussian NLL (assuming standard noise variance sigma^2 = 0.01).
    Evaluates across multiple context lengths (20, 40, 60, 80, 99) for thoroughness.
    """
    model.eval()
    x_val, y_val, _ = batch_fn(batch_size=batch_size, seq_len=seq_len, num_features=1, device=device)
    
    eval_positions = [20, 40, 60, 80, 99]
    total_mse = 0.0
    total_nll = 0.0
    
    for eval_pos in eval_positions:
        # Predict at query index
        logits = model((x_val, y_val), single_eval_pos=eval_pos) # (seq_len - eval_pos, batch_size, 1)
        targets = y_val[eval_pos:].unsqueeze(-1)
        
        mse = torch.mean((logits - targets) ** 2).item()
        # NLL under noise variance sigma^2 = 0.01: 0.5 * ln(2 * pi * 0.01) + mse / 0.02
        nll = 0.5 * math.log(2.0 * math.pi * 0.01) + mse / 0.02
        
        total_mse += mse
        total_nll += nll
        
    return total_mse / len(eval_positions), total_nll / len(eval_positions)

def main():
    parser = argparse.ArgumentParser(description="Train PFN on multi-prior GP mixtures")
    parser.add_argument("--model_type", type=str, required=True, choices=["rbf_matern", "rbf_periodic", "matern_periodic", "all_three"])
    parser.add_argument("--gpu", type=int, required=True, help="GPU ID to use (0 or 1)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="Steps per epoch")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=100, help="Sequence length (bptt)")
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"==========================================")
    print(f"Training Model: {args.model_type}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs} | Steps/Epoch: {args.steps_per_epoch}")
    print(f"==========================================")
    
    # 1. Build DataLoader
    batch_fn = get_mixture_batch_fn(args.model_type)
    DataLoaderClass = get_batch_to_dataloader(batch_fn)
    DataLoaderClass.num_features = 1
    DataLoaderClass.num_outputs = 1
    dl = DataLoaderClass(
        num_steps=args.steps_per_epoch, 
        batch_size=args.batch_size, 
        seq_len=args.seq_len, 
        num_features=1
    )
    
    # 2. Build Model (matching 6-layer architecture used in pre-trained model)
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
    lr = get_openai_lr(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    warmup_epochs = args.epochs // 5
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_epochs, args.epochs)
    
    # Sampler for split positions
    single_eval_pos_gen = get_weighted_single_eval_pos_sampler(args.seq_len - 1)
    criterion = nn.MSELoss(reduction='none')
    
    # Make output directory
    os.makedirs("/home/psquare_a6000/Desktop/conformal_ood/gp_pfns_multi_prior/checkpoints", exist_ok=True)
    os.makedirs("/home/psquare_a6000/Desktop/conformal_ood/gp_pfns_multi_prior/logs", exist_ok=True)
    
    best_val_mse = float("inf")
    
    # Start training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch, (data, targets) in enumerate(dl):
            # Format inputs
            inputs = tuple(e.to(device) for e in data) if isinstance(data, tuple) else data.to(device)
            targets = targets.to(device)
            
            # Sample split pos
            single_eval_pos = single_eval_pos_gen()
            
            # Forward pass
            output = model(inputs, single_eval_pos=single_eval_pos)
            
            # Compute loss
            pred_targets = targets[single_eval_pos:]
            losses = criterion(output.flatten(), pred_targets.flatten())
            loss = losses.mean()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        
        # Periodic evaluation & logging
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            train_mse = epoch_loss / args.steps_per_epoch
            train_nll = 0.5 * math.log(2.0 * math.pi * 0.01) + train_mse / 0.02
            
            val_mse, val_nll = evaluate_validation(model, device, batch_fn, seq_len=args.seq_len)
            
            print(f"Epoch {epoch:3d}/{args.epochs} | Train MSE: {train_mse:.4f} | Train NLL: {train_nll:.4f} | Val MSE: {val_mse:.4f} | Val NLL: {val_nll:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save checkpoints
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                checkpoint_path = f"/home/psquare_a6000/Desktop/conformal_ood/gp_pfns_multi_prior/checkpoints/{args.model_type}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                # Keep backup of best metadata
                with open(f"/home/psquare_a6000/Desktop/conformal_ood/gp_pfns_multi_prior/logs/{args.model_type}_best.txt", "w") as f:
                    f.write(f"epoch={epoch}\nval_mse={val_mse:.6f}\nval_nll={val_nll:.6f}\n")
                    
    print(f"Training for {args.model_type} completed successfully. Best Val MSE: {best_val_mse:.6f}")

if __name__ == "__main__":
    main()
