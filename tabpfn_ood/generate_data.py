import torch
import numpy as np
import scipy.stats as stats
import random
import math

from priors.mlp import get_batch
from priors.mlp_anti import get_batch_anti
from priors.utils import normalize_data
from priors.flexible_categorical import MulticlassRank

def sample_hyperparameters(hp_dict):
    sampled = {}
    for key, value in hp_dict.items():
        if isinstance(value, dict) and 'distribution' in value:
            dist = value['distribution']
            val = None
            if dist == 'meta_gamma':
                # Simplified meta_gamma logic based on uniform alpha & scale
                alpha = np.random.uniform(0.1, value.get('max_alpha', 1.0))
                scale = np.random.uniform(0.1, value.get('max_scale', 1.0))
                val = np.random.gamma(shape=alpha, scale=scale)
            elif dist == 'meta_beta':
                # Simplified meta_beta logic
                scale = value.get('scale', 1.0)
                min_val = value.get('min', 0.0)
                max_val = value.get('max', 2.0)
                a = np.random.uniform(0.1, max_val)
                b = np.random.uniform(0.1, max_val)
                val = min_val + np.random.beta(a, b) * (scale)
            elif dist == 'meta_trunc_norm_log_scaled':
                # Simplified meta_trunc_norm_log_scaled, scaling with logs
                min_mean = value.get('min_mean', 0.0001)
                max_mean = value.get('max_mean', 1.0)
                log_mean = np.random.uniform(math.log(min_mean), math.log(max_mean))
                mean = math.exp(log_mean)
                log_std = np.random.uniform(math.log(0.01), math.log(1.0))
                std = mean * math.exp(log_std)
                # Sample from truncated normal with lower bound 0
                val = float(stats.truncnorm((0 - mean) / std, (1000 - mean) / std, loc=mean, scale=std).rvs())
            elif dist in ['meta_choice', 'meta_choice_mixed']:
                val = random.choice(value['choice_values'])
            else:
                val = None

            if val is not None:
                if value.get('round', False):
                    val = int(np.round(val))
                if 'lower_bound' in value:
                    val = max(value['lower_bound'], val)

            sampled[key] = val
        else:
            sampled[key] = value

    return sampled

def get_base_hyperparameters():
    return {
        "mix_activations": True,
        "num_layers": {'distribution': 'meta_gamma', 'max_alpha': 2, 'max_scale': 3, 'round': True,
                       'lower_bound': 2},
        "prior_mlp_hidden_dim": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 100, 'round': True,
                                 'lower_bound': 4},
        "prior_mlp_dropout_prob": {'distribution': 'meta_beta', 'scale': 0.6, 'min': 0.1, 'max': 5.0},
        "noise_std": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': .3, 'min_mean': 0.0001, 'round': False,
                      'lower_bound': 0.0},
        "init_std": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 10.0, 'min_mean': 0.01, 'round': False,
                     'lower_bound': 0.0},
        "num_causes": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 7, 'round': True,
                       'lower_bound': 2},
        "is_causal": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "pre_sample_weights": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "y_is_effect": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "sampling": {'distribution': 'meta_choice', 'choice_values': ['normal', 'mixed']},
        "prior_mlp_activations": {'distribution': 'meta_choice_mixed', 'choice_values': [
            torch.nn.Tanh,
            torch.nn.Identity,
            torch.nn.ReLU
        ]},
        "block_wise_dropout": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "sort_features": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "in_clique": {'distribution': 'meta_choice', 'choice_values': [True, False]},
    }

def get_sampled_hyperparameters():
    hyperparameters = get_base_hyperparameters()
    sampled_hyperparameters = sample_hyperparameters(hyperparameters)

    defaults = {
        "pre_sample_causes": True,
        "prior_mlp_scale_weights_sqrt": True,
        "random_feature_rotation": False,
        "differentiable": True,
    }
    for k, v in defaults.items():
        if k not in sampled_hyperparameters:
            sampled_hyperparameters[k] = v

    if "prior_mlp_activations" in sampled_hyperparameters:
        activation_class = sampled_hyperparameters["prior_mlp_activations"]
        sampled_hyperparameters["prior_mlp_activations"] = lambda: activation_class()

    return sampled_hyperparameters

def process_batch_data(x, y, sampled_hyperparameters, num_classes):
    x, y = normalize_data(x), normalize_data(y)

    ordered_p = sampled_hyperparameters.get('output_multiclass_ordered_p', 0.0)

    class_assigner = MulticlassRank(num_classes, ordered_p=ordered_p)
    y = class_assigner(y).float()

    rotate_normalized_labels = sampled_hyperparameters.get('rotate_normalized_labels', True)

    for b in range(y.shape[1]):
        valid_labels = y[:, b] != -100
        # Compresses classes to consecutive integers
        y[valid_labels, b] = (y[valid_labels, b] > y[valid_labels, b].unique().unsqueeze(1)).sum(axis=0).unsqueeze(0).float()

        if y[valid_labels, b].numel() != 0 and rotate_normalized_labels:
            n_cls = int((y[valid_labels, b].max() + 1).item())
            random_shift = torch.randint(0, n_cls, (1,), device=y.device)
            y[valid_labels, b] = (y[valid_labels, b] + random_shift) % n_cls

    y_scm_categorical = y.long()
    return x, y_scm_categorical

def get_inprior_data(batch_size=10, seq_len=1000, num_features=100, num_classes=10, without_noise=False):
    sampled_hyperparameters = get_sampled_hyperparameters()
    if without_noise:
        sampled_hyperparameters['noise_std'] = 0.0
    x, y, y_ = get_batch(batch_size, seq_len, num_features, sampled_hyperparameters)
    x, y_categorical = process_batch_data(x, y, sampled_hyperparameters, num_classes)
    return x, y_categorical

def get_outprior_data(batch_size=10, seq_len=1000, num_features=100, num_classes=10):
    sampled_hyperparameters = get_sampled_hyperparameters()
    x, y, y_ = get_batch_anti(batch_size, seq_len, num_features, sampled_hyperparameters)
    x, y_categorical = process_batch_data(x, y, sampled_hyperparameters, num_classes)
    return x, y_categorical

def get_borderline_outprior_data(batch_size=10, seq_len=1000, num_features=100, num_classes=10):
    # Dummy function for now
    sampled_hyperparameters = get_sampled_hyperparameters()
    x, y, y_ = get_batch_anti(batch_size, seq_len, num_features, sampled_hyperparameters)
    # create some perturbation 
    x = x + 0.1 * torch.randn_like(x)
    x, y_categorical = process_batch_data(x, y, sampled_hyperparameters, num_classes)
    return x, y_categorical

def get_extremely_outprior_data(batch_size=10, seq_len=1000, num_features=100, num_classes=10):
    # Dummy function for now
    sampled_hyperparameters = get_sampled_hyperparameters()
    x, y, y_ = get_batch_anti(batch_size, seq_len, num_features, sampled_hyperparameters)
    x = x * 10.0 + 5.0 * torch.randn_like(x)
    x, y_categorical = process_batch_data(x, y, sampled_hyperparameters, num_classes)
    return x, y_categorical

def get_data_by_level(level, num_samples_per_class=1000, num_points=100, num_features=100, num_classes=10):
    """
    Returns a list of tuples (x, y) for each data category.
    level 2: [in-prior, out-prior]
    level 3: [in-prior without noise, in-prior with noise, out-prior]
    level 4: [in-prior without noise, in-prior with noise, borderline out-prior, out-prior]
    level 5: [in-prior without noise, in-prior with noise, borderline out-prior, out-prior, extremely out-prior]
    """
    assert 2 <= level <= 5, "Level must be between 2 and 5"
    
    datasets = []
    
    if level == 2:
        in_prior_x, in_prior_y = get_inprior_data(num_samples_per_class, num_points, num_features, num_classes, without_noise=False)
        out_prior_x, out_prior_y = get_outprior_data(num_samples_per_class, num_points, num_features, num_classes)
        datasets = [(in_prior_x, in_prior_y), (out_prior_x, out_prior_y)]
        
    elif level == 3:
        in_prior_no_noise_x, in_prior_no_noise_y = get_inprior_data(num_samples_per_class, num_points, num_features, num_classes, without_noise=True)
        in_prior_with_noise_x, in_prior_with_noise_y = get_inprior_data(num_samples_per_class, num_points, num_features, num_classes, without_noise=False)
        out_prior_x, out_prior_y = get_outprior_data(num_samples_per_class, num_points, num_features, num_classes)
        datasets = [(in_prior_no_noise_x, in_prior_no_noise_y), (in_prior_with_noise_x, in_prior_with_noise_y), (out_prior_x, out_prior_y)]
        
    elif level == 4:
        in_prior_no_noise_x, in_prior_no_noise_y = get_inprior_data(num_samples_per_class, num_points, num_features, num_classes, without_noise=True)
        in_prior_with_noise_x, in_prior_with_noise_y = get_inprior_data(num_samples_per_class, num_points, num_features, num_classes, without_noise=False)
        border_out_x, border_out_y = get_borderline_outprior_data(num_samples_per_class, num_points, num_features, num_classes)
        out_prior_x, out_prior_y = get_outprior_data(num_samples_per_class, num_points, num_features, num_classes)
        datasets = [(in_prior_no_noise_x, in_prior_no_noise_y), (in_prior_with_noise_x, in_prior_with_noise_y), (border_out_x, border_out_y), (out_prior_x, out_prior_y)]
        
    elif level == 5:
        in_prior_no_noise_x, in_prior_no_noise_y = get_inprior_data(num_samples_per_class, num_points, num_features, num_classes, without_noise=True)
        in_prior_with_noise_x, in_prior_with_noise_y = get_inprior_data(num_samples_per_class, num_points, num_features, num_classes, without_noise=False)
        border_out_x, border_out_y = get_borderline_outprior_data(num_samples_per_class, num_points, num_features, num_classes)
        out_prior_x, out_prior_y = get_outprior_data(num_samples_per_class, num_points, num_features, num_classes)
        extreme_out_x, extreme_out_y = get_extremely_outprior_data(num_samples_per_class, num_points, num_features, num_classes)
        datasets = [(in_prior_no_noise_x, in_prior_no_noise_y), (in_prior_with_noise_x, in_prior_with_noise_y), (border_out_x, border_out_y), (out_prior_x, out_prior_y), (extreme_out_x, extreme_out_y)]
    
    return datasets

