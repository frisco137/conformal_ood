import numpy as np
import scipy.linalg as la
import scipy.spatial.distance as dist

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, lengthscale: float = 1.0) -> np.ndarray:
    """Computes RBF kernel matrix between X1 (n1, d) and X2 (n2, d)."""
    dists = dist.cdist(X1 / lengthscale, X2 / lengthscale, metric='sqeuclidean')
    return np.exp(-0.5 * dists)

def matern32_kernel(X1: np.ndarray, X2: np.ndarray, lengthscale: float = 1.0) -> np.ndarray:
    """Computes Matern 3/2 kernel matrix between X1 (n1, d) and X2 (n2, d)."""
    dists = dist.cdist(X1 / lengthscale, X2 / lengthscale, metric='euclidean')
    sqrt3_d = np.sqrt(3.0) * dists
    return (1.0 + sqrt3_d) * np.exp(-sqrt3_d)

def sample_task(
    n: int = 100,
    d: int = 1,
    kernel_type: str = "rbf",
    l: float = 1.0,
    sigma2: float = 0.01,
    seed: int = 42,
    task_mode: str = "gp"
) -> dict:
    """
    Samples a task (GP or Linear) with support size n and feature dimension d.
    
    Returns dictionary containing:
    - X: (n, d)
    - y: (n,)
    - X_query: (m, d) with m = min(400, 4*n)
    - y_query: (m,)
    - G: (n, n) Gram matrix
    - A: (n, n) regularized system matrix G + sigma2 * I
    - alpha_star: (n,) closed-form dual solution (A^-1 y)
    - w_star: (d,) closed-form primal solution (for linear tasks)
    """
    rng = np.random.RandomState(seed)
    
    # 1. Sample inputs
    X = rng.standard_normal((n, d))
    m = min(400, 4 * n)
    X_query = rng.standard_normal((m, d))
    
    if task_mode == "linear":
        # True linear model: y = X w* + noise
        w_true = rng.standard_normal(d) / np.sqrt(d)
        noise = rng.standard_normal(n) * np.sqrt(sigma2)
        y = np.dot(X, w_true) + noise
        
        # Dual Gram matrix G = X X^T
        G = np.dot(X, X.T)
        A = G + sigma2 * np.eye(n)
        
        # Primal weight estimate \hat{w} = (X^T X + sigma2 I_d)^-1 X^T y
        w_star = la.solve(np.dot(X.T, X) + sigma2 * np.eye(d), np.dot(X.T, y))
        # Dual weight estimate \alpha = (X X^T + sigma2 I_n)^-1 y
        alpha_star = la.solve(A, y)
        
        y_query = np.dot(X_query, w_star)
        
        return {
            "X": X, "y": y, "X_query": X_query, "y_query": y_query,
            "G": G, "A": A, "alpha_star": alpha_star, "w_star": w_star,
            "task_mode": "linear", "n": n, "d": d, "seed": seed
        }
        
    else:  # GP task
        if kernel_type == "rbf":
            kernel_fn = lambda x1, x2: rbf_kernel(x1, x2, l)
        elif kernel_type == "matern32":
            kernel_fn = lambda x1, x2: matern32_kernel(x1, x2, l)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
            
        G = kernel_fn(X, X)
        A = G + sigma2 * np.eye(n)
        
        # Sample y ~ N(0, A)
        try:
            L = la.cholesky(A, lower=True)
            z = rng.standard_normal(n)
            y = np.dot(L, z)
        except la.LinAlgError:
            u, s, _ = la.svd(A)
            y = np.dot(u, np.sqrt(np.maximum(s, 1e-10)) * rng.standard_normal(n))
            
        alpha_star = la.solve(A, y)
        K_g = kernel_fn(X_query, X)
        y_query = np.dot(K_g, alpha_star)
        
        return {
            "X": X, "y": y, "X_query": X_query, "y_query": y_query,
            "G": G, "A": A, "K_g": K_g, "alpha_star": alpha_star, "w_star": None,
            "task_mode": "gp", "n": n, "d": d, "seed": seed
        }
