import numpy as np
import scipy.linalg as la
from phase_2.src.utils.metrics import compute_frobenius_distance

def run_label_perturbation_experiment(model_wrapper, task: dict):
    """
    Generates base task (X, y_A) and label-perturbed task (X, y_B).
    Runs clean pass on task A and perturbed pass on task B.
    Measures layer-wise shift in support token inner products (decoded kernel matrix)
    vs shift in decoded solution predictions \alpha.
    """
    X_train = task["X"]
    y_A = task["y"]
    X_test = task["X_query"]
    n = X_train.shape[0]
    
    # Perturb labels: add orthogonal Gaussian noise to targets y_B = y_A + delta
    rng = np.random.RandomState(task["seed"] + 100)
    y_B = y_A + rng.standard_normal(n) * np.std(y_A)
    
    # Extract clean artifacts for Task A
    artifacts_A = model_wrapper.extract_task_artifacts(X_train, y_A, X_test)
    # Extract artifacts for Task B
    artifacts_B = model_wrapper.extract_task_artifacts(X_train, y_B, X_test)
    
    layer_kernel_shifts = {}
    layer_solution_shifts = {}
    
    for l in range(12):
        H_A = artifacts_A["support_embs"][l].reshape(n, -1) # (n, D_total)
        H_B = artifacts_B["support_embs"][l].reshape(n, -1) # (n, D_total)
        
        # Inner product Gram matrices
        H_A_n = H_A / (la.norm(H_A, axis=1, keepdims=True) + 1e-10)
        H_B_n = H_B / (la.norm(H_B, axis=1, keepdims=True) + 1e-10)
        
        K_A = np.dot(H_A_n, H_A_n.T)
        K_B = np.dot(H_B_n, H_B_n.T)
        
        # Kernel shift (relative Frobenius norm)
        kernel_shift = la.norm(K_A - K_B, 'fro') / (la.norm(K_A, 'fro') + 1e-15)
        layer_kernel_shifts[l] = float(kernel_shift)
        
        # Solution shift (relative norm of predictions or alpha)
        pred_A = artifacts_A["predictions"][l]
        pred_B = artifacts_B["predictions"][l]
        sol_shift = la.norm(pred_A - pred_B) / (la.norm(pred_A) + 1e-15)
        layer_solution_shifts[l] = float(sol_shift)
        
    return {
        "kernel_shifts": layer_kernel_shifts,
        "solution_shifts": layer_solution_shifts
    }
