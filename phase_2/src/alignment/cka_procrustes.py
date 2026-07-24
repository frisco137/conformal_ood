import numpy as np
import scipy.linalg as la
from phase_2.src.utils.metrics import compute_r2

def compute_linear_cka(H_A: np.ndarray, H_B: np.ndarray) -> float:
    """
    Computes Linear Centered Kernel Alignment (CKA) between representations
    H_A (N, D_A) and H_B (N, D_B).
    CKA(K, L) = ||K^T L||_F^2 / (||K^T K||_F * ||L^T L||_F)
    where K = H_A H_A^T and L = H_B H_B^T are centered Gram matrices.
    """
    H_A = np.asarray(H_A, dtype=float)
    H_B = np.asarray(H_B, dtype=float)
    
    # Filter out zero variance feature columns
    std_A = np.std(H_A, axis=0)
    std_B = np.std(H_B, axis=0)
    H_A = H_A[:, std_A > 1e-8]
    H_B = H_B[:, std_B > 1e-8]
    
    if H_A.shape[1] == 0 or H_B.shape[1] == 0:
        return 0.0
        
    N = H_A.shape[0]
    # Center activations per feature
    H_A_c = H_A - np.mean(H_A, axis=0, keepdims=True)
    H_B_c = H_B - np.mean(H_B, axis=0, keepdims=True)
    
    # Compute dot products in feature space
    hsic_ab = la.norm(np.dot(H_A_c.T, H_B_c), 'fro') ** 2
    hsic_aa = la.norm(np.dot(H_A_c.T, H_A_c), 'fro') ** 2
    hsic_bb = la.norm(np.dot(H_B_c.T, H_B_c), 'fro') ** 2
    
    denom = np.sqrt(hsic_aa * hsic_bb)
    if denom < 1e-12:
        return 0.0
    return float(hsic_ab / denom)

def compute_procrustes_alignment_r2(H_A: np.ndarray, H_B: np.ndarray, alpha_reg: float = 1e-3) -> float:
    """
    Fits Ridge OLS linear mapping W mapping H_A (N, D_A) -> H_B (N, D_B).
    Returns cross-model decoding alignment R^2.
    """
    N, D_A = H_A.shape
    _, D_B = H_B.shape
    
    # Center
    H_A_c = H_A - np.mean(H_A, axis=0, keepdims=True)
    H_B_c = H_B - np.mean(H_B, axis=0, keepdims=True)
    
    reg = alpha_reg * np.eye(D_A)
    try:
        W = la.solve(np.dot(H_A_c.T, H_A_c) + reg, np.dot(H_A_c.T, H_B_c))
    except la.LinAlgError:
        W = la.lstsq(np.dot(H_A_c.T, H_A_c) + reg, np.dot(H_A_c.T, H_B_c))[0]
        
    pred_H_B = np.dot(H_A_c, W)
    return compute_r2(H_B_c, pred_H_B)

def evaluate_cross_model_alignment(wrapper_pfn, wrapper_icl, tasks: list):
    """
    Extracts solution layer (Layer 10) support token representations for identical task batch,
    computes Linear CKA and Procrustes/Ridge alignment R^2 between TabPFN and TabICL,
    and compares with within-family cross-layer baselines.
    """
    pfn_support_all = {l: [] for l in range(12)}
    icl_support_all = {l: [] for l in range(12)}
    
    for task in tasks:
        art_pfn = wrapper_pfn.extract_task_artifacts(task["X"], task["y"], task["X_query"])
        art_icl = wrapper_icl.extract_task_artifacts(task["X"], task["y"], task["X_query"])
        
        n = task["X"].shape[0]
        for l in range(12):
            pfn_support_all[l].append(art_pfn["support_embs"][l].reshape(n, -1))
            icl_support_all[l].append(art_icl["support_embs"][l].reshape(n, -1))
            
    alignment_results = []
    
    for l in range(12):
        # Concatenate tokens across all tasks in batch: (N_total, D)
        H_pfn = np.vstack(pfn_support_all[l])
        H_icl = np.vstack(icl_support_all[l])
        
        # 1. Cross-model CKA & Procrustes R^2
        cka_cross = compute_linear_cka(H_pfn, H_icl)
        r2_cross = compute_procrustes_alignment_r2(H_pfn, H_icl)
        
        # 2. Within-family cross-layer CKA (Layer l vs Layer l-1)
        if l > 0:
            H_pfn_prev = np.vstack(pfn_support_all[l-1])
            H_icl_prev = np.vstack(icl_support_all[l-1])
            cka_pfn_within = compute_linear_cka(H_pfn, H_pfn_prev)
            cka_icl_within = compute_linear_cka(H_icl, H_icl_prev)
        else:
            cka_pfn_within = 1.0
            cka_icl_within = 1.0
            
        alignment_results.append({
            "Layer": l,
            "Cross_Model_CKA": cka_cross,
            "Cross_Model_Alignment_R2": max(0.0, r2_cross),
            "TabPFN_Within_Family_CKA": cka_pfn_within,
            "TabICL_Within_Family_CKA": cka_icl_within
        })
        
    return alignment_results
