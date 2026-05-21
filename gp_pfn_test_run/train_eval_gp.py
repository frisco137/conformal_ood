import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Add PFN repo to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TransformersCanDoBayesianInference')))

import priors
from train import train
import encoders
import positional_encodings
import utils

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # GP Hyperparameters
    hps = {'noise': 1e-4, 'outputscale': 1., 'lengthscale': .6}
    
    # DataLoader GP prior
    num_features = 1
    bptt = 100
    batch_size = 128
    steps_per_epoch = 100
    epochs = 20
    
    # Use MSE Loss for regression
    criterion = nn.MSELoss(reduction='none')
    
    # Setup hyperparameters
    extra_prior_kwargs_dict = {
        'num_features': num_features,
        'fuse_x_y': False,
        'hyperparameters': hps
    }
    
    print("Starting PFN training on GP regression prior...")
    total_loss, total_positional_losses, model = train(
        priordataloader_class=priors.fast_gp.DataLoader,
        criterion=criterion,
        encoder_generator=encoders.Linear,
        emsize=256,
        nhead=4,
        nhid=512,
        nlayers=4,
        dropout=0.0,
        warmup_epochs=epochs // 4,
        y_encoder_generator=encoders.Linear,
        pos_encoder_generator=positional_encodings.PositionalEncoding,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        bptt=bptt,
        single_eval_pos_gen=utils.get_weighted_single_eval_pos_sampler(bptt - 1),
        extra_prior_kwargs_dict=extra_prior_kwargs_dict,
        gpu_device=device,
        verbose=True
    )
    
    print("Training finished!")
    
    # Save the trained model checkpoint
    os.makedirs('parameter_free_ood/checkpoints', exist_ok=True)
    checkpoint_path = 'parameter_free_ood/checkpoints/gp_regression_pfn.pt'
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

if __name__ == '__main__':
    main()
