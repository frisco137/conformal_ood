import numpy as np
import scipy.linalg as la
from phase_2.src.utils.gp_generator import sample_task
from phase_2.src.utils.metrics import compute_r2
from harness import solve_gcv_weights

def estimate_implied_sigma2_gcv(K_g: np.ndarray, G: np.ndarray, y: np.ndarray, mu_pred: np.ndarray):
    """
    Finds the implied regularization parameter sigma^2 that best matches
    the model's decoded predictions mu_pred = K_g (G + sigma^2 I)^-1 y.
    """
    n = G.shape[0]
    sigma2_candidates = np.logspace(-5, 1, 50)
    best_sigma2 = 0.01
    min_err = 1e9
    
    for s2 in sigma2_candidates:
        A_cand = G + s2 * np.eye(n)
        try:
            alpha_cand = la.solve(A_cand, y)
            pred_cand = np.dot(K_g, alpha_cand)
            err = np.mean((mu_pred - pred_cand)**2)
            if err < min_err:
                min_err = err
                best_sigma2 = s2
        except la.LinAlgError:
            continue
            
    return float(best_sigma2)

def evaluate_noise_adaptation(model_wrapper, sigma_true_list: list, seed_base: int = 7000):
    """
    Injects controlled label noise sigma_true into context, extracts decoded predictions,
    and measures implied GCV sigma^2 and probe decodability across layers.
    """
    results = []
    
    for idx, sigma_true in enumerate(sigma_true_list):
        sigma2_true = float(sigma_true ** 2)
        task = sample_task(n=100, d=1, kernel_type="rbf", l=1.0, sigma2=sigma2_true, seed=seed_base + idx)
        
        art = model_wrapper.extract_task_artifacts(task["X"], task["y"], task["X_query"], K_g=task["K_g"])
        
        for l in range(12):
            mu_l = art["predictions"][l]
            implied_s2 = estimate_implied_sigma2_gcv(task["K_g"], task["G"], task["y"], mu_l)
            
            results.append({
                "sigma_true": sigma_true,
                "sigma2_true": sigma2_true,
                "Layer": l,
                "implied_sigma2_gcv": implied_s2
            })
            
    return results
