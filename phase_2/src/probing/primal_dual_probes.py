import numpy as np
import scipy.linalg as la
from phase_2.src.utils.metrics import compute_r2

def fit_ols_ridge_probe(X_feats: np.ndarray, Y_targets: np.ndarray, alpha: float = 1e-3):
    """
    Fits Ridge OLS probe mapping X_feats (M, D_in) -> Y_targets (M, D_out).
    Returns weight W (D_in, D_out) and bias b (D_out,).
    """
    X_feats = np.asarray(X_feats)
    if X_feats.ndim > 2:
        X_feats = X_feats.reshape(X_feats.shape[0], -1)
        
    M, D_in = X_feats.shape
    # Add bias column
    X_b = np.hstack([X_feats, np.ones((M, 1))])
    # Solve (X^T X + alpha I)^-1 X^T Y
    reg = alpha * np.eye(D_in + 1)
    reg[-1, -1] = 0.0 # Don't penalize bias
    
    W_b = la.solve(np.dot(X_b.T, X_b) + reg, np.dot(X_b.T, Y_targets))
    W = W_b[:-1]
    b = W_b[-1]
    return W, b

def predict_probe(X_feats: np.ndarray, W: np.ndarray, b: np.ndarray):
    X_feats = np.asarray(X_feats)
    if X_feats.ndim > 2:
        X_feats = X_feats.reshape(X_feats.shape[0], -1)
    return np.dot(X_feats, W) + b

def evaluate_layer_probes(layer_support_embs: list, targets: list):
    """
    Given a list of support embeddings per task (each task: (n, D_hidden))
    and corresponding targets per task (each task: (d,) for w_star or (n,) for alpha_star),
    fits cross-validated or pooled linear probe and returns R^2.
    """
    M = len(layer_support_embs)
    target_shape = targets[0].shape
    
    # Ensure support embs are 2D (n, D_total) for each task by preserving token count emb.shape[0]
    clean_embs = [emb.reshape(emb.shape[0], -1) for emb in layer_support_embs]
    
    if len(target_shape) == 1 and target_shape[0] == clean_embs[0].shape[0]:
        # Target is dual alpha* of shape (n,)
        # Concatenate tokens across tasks: X_all shape (M*n, D_hidden), Y_all shape (M*n,)
        X_all = np.vstack(clean_embs)       # (M*n, D_hidden)
        Y_all = np.concatenate(targets)      # (M*n,)
        
        # 5-fold cross validation for robust R^2
        indices = np.arange(len(Y_all))
        np.random.RandomState(42).shuffle(indices)
        folds = np.array_split(indices, 5)
        
        r2_list = []
        for f in range(5):
            val_idx = folds[f]
            train_idx = np.setdiff1d(indices, val_idx)
            
            W, b = fit_ols_ridge_probe(X_all[train_idx], Y_all[train_idx].reshape(-1, 1))
            pred_val = predict_probe(X_all[val_idx], W, b).flatten()
            r2_list.append(compute_r2(Y_all[val_idx], pred_val))
            
        return float(np.mean(r2_list))
        
    else:
        # Target is primal w* of shape (d,)
        # Mean-pool support embeddings for each task: X_all shape (M, D_hidden), Y_all shape (M, d)
        X_all = np.array([np.mean(emb, axis=0) for emb in clean_embs]) # (M, D_hidden)
        Y_all = np.array(targets) # (M, d)
        
        indices = np.arange(M)
        np.random.RandomState(42).shuffle(indices)
        folds = np.array_split(indices, 5)
        
        r2_list = []
        for f in range(5):
            val_idx = folds[f]
            train_idx = np.setdiff1d(indices, val_idx)
            
            W, b = fit_ols_ridge_probe(X_all[train_idx], Y_all[train_idx])
            pred_val = predict_probe(X_all[val_idx], W, b)
            r2_list.append(compute_r2(Y_all[val_idx], pred_val))
            
        return float(np.mean(r2_list))
