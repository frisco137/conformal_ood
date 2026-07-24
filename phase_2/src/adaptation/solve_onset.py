import numpy as np
import scipy.linalg as la
from phase_2.src.utils.gp_generator import sample_task

def find_solve_onset_layer(predictions_dict: dict, threshold: float = 0.01) -> int:
    """
    Finds the smallest layer index l_0 such that:
    || mu^(l_0) - mu^(11) ||_2 / || mu^(11) ||_2 < threshold
    """
    mu_final = predictions_dict[11]
    norm_final = la.norm(mu_final) + 1e-15
    
    for l in range(12):
        mu_l = predictions_dict[l]
        rel_diff = la.norm(mu_l - mu_final) / norm_final
        if rel_diff < threshold:
            return l
    return 11

def evaluate_solve_onset_scaling(model_wrapper, n_values: list, d_values: list, seed_base: int = 8000):
    """
    Evaluates solve onset layer l_0 scaling across context size n (at d=1)
    and feature dimension d (at n=100).
    """
    n_results = []
    d_results = []
    
    # 1. Scale n at fixed d=1
    for n in n_values:
        task = sample_task(n=n, d=1, kernel_type="rbf", l=1.0, seed=seed_base + n)
        art = model_wrapper.extract_task_artifacts(task["X"], task["y"], task["X_query"])
        l0 = find_solve_onset_layer(art["predictions"], threshold=0.01)
        n_results.append({
            "Varying_Parameter": "n",
            "n": n,
            "d": 1,
            "solve_onset_layer_l0": l0
        })
        
    # 2. Scale d at fixed n=100
    for d in d_values:
        task = sample_task(n=100, d=d, kernel_type="rbf", l=1.0, seed=seed_base + 1000 + d)
        art = model_wrapper.extract_task_artifacts(task["X"], task["y"], task["X_query"])
        l0 = find_solve_onset_layer(art["predictions"], threshold=0.01)
        d_results.append({
            "Varying_Parameter": "d",
            "n": 100,
            "d": d,
            "solve_onset_layer_l0": l0
        })
        
    return n_results + d_results
