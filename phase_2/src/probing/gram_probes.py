import numpy as np
import scipy.linalg as la
from phase_2.src.utils.metrics import compute_r2, compute_frobenius_distance

def evaluate_gram_decodability(layer_support_embs: list, true_grams: list):
    """
    Fits linear probe to decode pairwise Gram entries K_ij from token inner products:
    \hat{K}_ij = gamma * <h_i, h_j> + beta.
    Returns decoding R^2 across tasks.
    """
    M = len(layer_support_embs)
    r2_list = []
    
    for m in range(M):
        H = layer_support_embs[m] # (n, ...)
        K_true = true_grams[m]     # (n, n)
        
        # Flatten token feature dimensions so H is (n, D_total)
        n_samples = K_true.shape[0]
        H = H.reshape(n_samples, -1)
        
        # Compute normalized inner products between tokens
        H_norm = H / (la.norm(H, axis=1, keepdims=True) + 1e-10)
        K_inner = np.dot(H_norm, H_norm.T)
        
        # Flatten upper triangle entries
        n = H.shape[0]
        triu_idx = np.triu_indices(n, k=1)
        
        y_true = K_true[triu_idx]
        x_pred = K_inner[triu_idx]
        
        # Fit OLS scalar slope & intercept
        slope, intercept = np.polyfit(x_pred, y_true, 1)
        y_pred = slope * x_pred + intercept
        
        r2_list.append(compute_r2(y_true, y_pred))
        
    return float(np.mean(r2_list))

def compute_attention_gram_alignment(item_attn_dict: dict, true_gram: np.ndarray):
    """
    Computes cosine similarity and Frobenius norm error between
    the item attention score matrix S^(l) and the row-normalized true Gram matrix K.
    """
    alignment_scores = {}
    
    # Normalize true Gram matrix row-wise so each row sums to 1 (matching softmax attention probabilities)
    n = true_gram.shape[0]
    K_norm = true_gram / (np.sum(true_gram, axis=1, keepdims=True) + 1e-15)
    
    for l, S in item_attn_dict.items():
        if S is not None and S.shape == true_gram.shape:
            # Cosine similarity between flattened matrices
            S_flat = S.flatten()
            K_flat = K_norm.flatten()
            
            denom = (la.norm(S_flat) * la.norm(K_flat)) + 1e-15
            cos_sim = float(np.dot(S_flat, K_flat) / denom)
            alignment_scores[l] = cos_sim
            
    return alignment_scores
