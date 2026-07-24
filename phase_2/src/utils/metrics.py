import numpy as np
import scipy.linalg as la

def compute_effective_rank(matrix: np.ndarray) -> float:
    """
    Computes numerical effective rank of matrix:
    rank_eff(X) = (sum s_i)^2 / sum(s_i^2)
    where s_i are the singular values.
    """
    matrix = np.asarray(matrix)
    if matrix.size == 0:
        return 0.0
    if matrix.ndim > 2:
        matrix = matrix.reshape(-1, matrix.shape[-1])
        
    # Center or take raw SVD
    s = la.svdvals(matrix)
    s_sum = np.sum(s)
    s2_sum = np.sum(s**2)
    if s2_sum < 1e-15:
        return 0.0
    return float((s_sum ** 2) / s2_sum)

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes R^2 coefficient of determination.
    Handles 1D arrays or 2D multi-target arrays (averaged R^2 across targets).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.ndim == 1:
        tss = np.sum((y_true - np.mean(y_true))**2)
        rss = np.sum((y_true - y_pred)**2)
        if tss < 1e-15:
            return 1.0 if rss < 1e-15 else 0.0
        return float(1.0 - (rss / tss))
    else:
        # Multi-target
        tss = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True))**2)
        rss = np.sum((y_true - y_pred)**2)
        if tss < 1e-15:
            return 1.0 if rss < 1e-15 else 0.0
        return float(1.0 - (rss / tss))

def compute_frobenius_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Computes Frobenius norm distance ||A - B||_F."""
    return float(la.norm(A - B, 'fro'))
