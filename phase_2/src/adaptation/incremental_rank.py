import numpy as np
import scipy.linalg as la
from phase_2.src.utils.metrics import compute_effective_rank

def compute_gini_coefficient(v: np.ndarray) -> float:
    """
    Computes Gini coefficient of a 1D vector of non-negative values.
    Gini -> 1 means highly localized/concentrated; Gini -> 0 means uniform/diffuse.
    """
    v = np.asarray(v, dtype=float).flatten()
    if np.all(v == 0) or len(v) == 0:
        return 0.0
    v = np.abs(v)
    v_sorted = np.sort(v)
    n = len(v)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * v_sorted) / (n * np.sum(v_sorted))) - (n + 1.0) / n)

def evaluate_incremental_update(model_wrapper, task: dict):
    """
    Adds a single context point to a task of size n, extracts layer activation diffs \Delta H,
    and evaluates effective numerical rank and Gini spatial locality index.
    """
    X_base, y_base = task["X"], task["y"]
    X_test = task["X_query"]
    n, d = X_base.shape
    
    # 1. Clean forward pass on base context (size n)
    art_n = model_wrapper.extract_task_artifacts(X_base, y_base, X_test)
    
    # 2. Add single context point (x_{n+1}, y_{n+1})
    rng = np.random.RandomState(task["seed"] + 99)
    x_new = rng.standard_normal((1, d))
    # y_new from GP prior or sampled
    y_new = rng.standard_normal(1) * np.std(y_base)
    
    X_inc = np.vstack([X_base, x_new])
    y_inc = np.concatenate([y_base, y_new])
    
    art_inc = model_wrapper.extract_task_artifacts(X_inc, y_inc, X_test)
    
    layer_ranks = {}
    layer_ginis = {}
    
    for l in range(12):
        H_n = art_n["support_embs"][l].reshape(n, -1)         # (n, D_total)
        H_inc = art_inc["support_embs"][l][:n].reshape(n, -1) # (n, D_total)
        
        # Activation difference matrix on existing n tokens
        delta_H = H_inc - H_n                                 # (n, D_total)
        
        eff_rank = compute_effective_rank(delta_H)
        layer_ranks[l] = float(eff_rank)
        
        # Norm shift per context token i
        token_norms = la.norm(delta_H, axis=1)               # (n,)
        gini_locality = compute_gini_coefficient(token_norms)
        layer_ginis[l] = float(gini_locality)
        
    return {
        "eff_ranks": layer_ranks,
        "gini_localities": layer_ginis
    }
