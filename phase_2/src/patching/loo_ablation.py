import numpy as np
import scipy.linalg as la
from phase_2.src.utils.metrics import compute_r2

def compute_loo_influences_dual(task: dict):
    """
    Computes theoretical dual LOO influence of each context point i on query point x_*:
    Influence_i_dual = alpha_i * k(x_i, x_*)
    """
    alpha = task["alpha_star"]
    X_train = task["X"]
    X_query = task["X_query"]
    
    # Compute K(X_query, X_train)
    if "K_g" in task:
        K_g = task["K_g"]
    else:
        K_g = np.dot(X_query, X_train.T)
        
    # For a query point m, influence of support point i is alpha_i * K_g[m, i]
    # Shape: (m, n)
    dual_influence = K_g * alpha[np.newaxis, :]
    return dual_influence

def compute_loo_influences_primal(task: dict):
    """
    Computes theoretical primal LOO influence of each context point i on query point x_*:
    Influence_i_primal = x_*^T ( \hat{w} - \hat{w}_(-i) )
    """
    X = task["X"]
    y = task["y"]
    X_query = task["X_query"]
    n, d = X.shape
    m = X_query.shape[0]
    
    sigma2 = 0.01
    A_p = np.dot(X.T, X) + sigma2 * np.eye(d)
    w_full = la.solve(A_p, np.dot(X.T, y))
    
    primal_influence = np.zeros((m, n))
    
    for i in range(n):
        # Remove sample i
        X_sub = np.delete(X, i, axis=0)
        y_sub = np.delete(y, i)
        A_sub = np.dot(X_sub.T, X_sub) + sigma2 * np.eye(d)
        w_sub = la.solve(A_sub, np.dot(X_sub.T, y_sub))
        
        delta_w = w_full - w_sub
        primal_influence[:, i] = np.dot(X_query, delta_w)
        
    return primal_influence

def evaluate_loo_empirical_dependence(model_wrapper, task: dict):
    """
    Runs empirical LOO ablation on model by masking single context point tokens
    and measures prediction changes vs theoretical dual and primal influence predictions.
    """
    X_train, y_train = task["X"], task["y"]
    X_test = task["X_query"]
    n = X_train.shape[0]
    m = X_test.shape[0]
    
    # 1. Clean run
    clean_artifacts = model_wrapper.extract_task_artifacts(X_train, y_train, X_test)
    y_pred_clean = clean_artifacts["predictions"][11] # Final layer predictions
    
    empirical_delta_y = np.zeros((m, n))
    
    # Select subset of 10 context points for computational speed
    eval_pts = np.linspace(0, n-1, min(10, n), dtype=int)
    
    for i in eval_pts:
        X_train_loo = np.delete(X_train, i, axis=0)
        y_train_loo = np.delete(y_train, i)
        
        loo_artifacts = model_wrapper.extract_task_artifacts(X_train_loo, y_train_loo, X_test)
        y_pred_loo = loo_artifacts["predictions"][11]
        
        empirical_delta_y[:, i] = y_pred_clean - y_pred_loo
        
    dual_inf = compute_loo_influences_dual(task)[:, eval_pts]
    primal_inf = compute_loo_influences_primal(task)[:, eval_pts]
    emp_inf = empirical_delta_y[:, eval_pts]
    
    # Compute R^2 of empirical influence against dual vs primal predictions
    r2_dual = compute_r2(emp_inf.flatten(), dual_inf.flatten())
    r2_primal = compute_r2(emp_inf.flatten(), primal_inf.flatten())
    
    return {
        "r2_dual_influence": r2_dual,
        "r2_primal_influence": r2_primal
    }
