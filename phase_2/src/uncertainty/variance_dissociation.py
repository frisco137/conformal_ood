import numpy as np
import scipy.linalg as la
import scipy.spatial.distance as dist
from phase_2.src.utils.metrics import compute_r2

def compute_true_gp_variance(X_train: np.ndarray, X_query: np.ndarray, l: float = 1.0, sigma2: float = 0.01):
    """
    Computes closed-form GP posterior variance for query points X_query:
    sigma^2_GP(x_*) = k(x_*, x_*) - k_*^T (K_train + sigma2 * I)^-1 k_*
    """
    n = X_train.shape[0]
    m = X_query.shape[0]
    
    # RBF kernel fn
    cdist_tt = dist.cdist(X_train / l, X_train / l, metric='sqeuclidean')
    G_train = np.exp(-0.5 * cdist_tt)
    A_train = G_train + sigma2 * np.eye(n)
    
    cdist_qt = dist.cdist(X_query / l, X_train / l, metric='sqeuclidean')
    K_qt = np.exp(-0.5 * cdist_qt) # (m, n)
    
    # Solve A^-1 K_qt^T
    try:
        inv_A_K = la.solve(A_train, K_qt.T) # (n, m)
    except la.LinAlgError:
        inv_A_K = la.lstsq(A_train, K_qt.T)[0]
        
    var_reduction = np.sum(K_qt.T * inv_A_K, axis=0) # (m,)
    gp_var = 1.0 - var_reduction + sigma2 # k(x_*, x_*) = 1.0
    return np.maximum(gp_var, 1e-10)

def compute_nn_distance(X_train: np.ndarray, X_query: np.ndarray):
    """Computes distance to nearest context point for each query point."""
    dists = dist.cdist(X_query, X_train, metric='euclidean')
    return np.min(dists, axis=1)

def compute_local_density(X_train: np.ndarray, X_query: np.ndarray, k_density: int = 5):
    """Computes mean distance to k-nearest context points."""
    dists = dist.cdist(X_query, X_train, metric='euclidean')
    dists_sorted = np.sort(dists, axis=1)
    k_actual = min(k_density, X_train.shape[0])
    return np.mean(dists_sorted[:, :k_actual], axis=1)

def sample_adversarial_dissociation_task(n: int = 100, d: int = 1, task_type: str = "dense_uninformative", seed: int = 42):
    """
    Generates adversarial context where closed-form GP variance and NN distance diverge:
    - 'dense_uninformative': Repeated / highly clustered points. NN distance is tiny, but Gram is degenerate.
    - 'sparse_informative': Well-spaced orthogonal points. NN distance is moderate, but true GP variance drops.
    """
    rng = np.random.RandomState(seed)
    l = 1.0
    sigma2 = 0.01
    
    if task_type == "dense_uninformative":
        # Create 5 cluster centers, repeat points with tiny jitter
        n_clusters = 5
        centers = rng.standard_normal((n_clusters, d)) * 2.0
        X = np.zeros((n, d))
        for i in range(n):
            c_idx = i % n_clusters
            X[i] = centers[c_idx] + rng.standard_normal(d) * 1e-4 # Tiny jitter (nn_dist -> 0)
    else:
        # Sparse informative: well-spaced grid/points
        X = rng.standard_normal((n, d)) * 1.5
        
    m = 200
    X_query = rng.standard_normal((m, d)) * 2.0
    
    # Compute true GP posterior & targets
    cdist_tt = dist.cdist(X / l, X / l, metric='sqeuclidean')
    G = np.exp(-0.5 * cdist_tt)
    A = G + sigma2 * np.eye(n)
    
    try:
        L = la.cholesky(A, lower=True)
        y = np.dot(L, rng.standard_normal(n))
    except la.LinAlgError:
        u, s, _ = la.svd(A)
        y = np.dot(u, np.sqrt(np.maximum(s, 1e-10)) * rng.standard_normal(n))
        
    gp_var = compute_true_gp_variance(X, X_query, l, sigma2)
    nn_dist = compute_nn_distance(X, X_query)
    density = compute_local_density(X, X_query, k_density=5)
    
    return {
        "X": X, "y": y, "X_query": X_query,
        "gp_var": gp_var, "nn_dist": nn_dist, "density": density,
        "G": G, "A": A, "task_type": task_type, "seed": seed
    }

def fit_uncertainty_dissociation_regression(model_wrapper, task: dict):
    """
    Extracts model predicted variance s^2(x_*), and fits multiple linear regression:
    s^2(x_*) = \beta_0 + \beta_GP * GP_var + \beta_NN * NN_dist + \beta_dens * Density
    Returns standardized regression coefficients and partial R^2 scores.
    """
    art = model_wrapper.extract_task_artifacts(task["X"], task["y"], task["X_query"])
    
    # Model predictions at final layer (or predictions across layers)
    pred = art["predictions"][11]
    
    # Proxy predicted variance s^2: absolute residual error or prediction variance across layers
    # Compute prediction spread across solving phase layers [5..11]
    layer_preds = np.array([art["predictions"][l] for l in range(5, 12)]) # (7, m)
    model_s2 = np.var(layer_preds, axis=0) # (m,)
    
    gp_var = task["gp_var"]
    nn_dist = task["nn_dist"]
    density = task["density"]
    
    # Standardize predictors
    Z = np.column_stack([
        (gp_var - np.mean(gp_var)) / (np.std(gp_var) + 1e-15),
        (nn_dist - np.mean(nn_dist)) / (np.std(nn_dist) + 1e-15),
        (density - np.mean(density)) / (np.std(density) + 1e-15),
        np.ones(len(model_s2))
    ])
    
    # Fit OLS
    try:
        coefs = la.lstsq(Z, model_s2)[0]
    except la.LinAlgError:
        coefs = np.zeros(4)
        
    beta_gp, beta_nn, beta_dens, intercept = coefs
    
    # Full R^2
    pred_s2 = np.dot(Z, coefs)
    full_r2 = compute_r2(model_s2, pred_s2)
    
    # Partial R^2 for GP var alone
    Z_gp = np.column_stack([(gp_var - np.mean(gp_var)) / (np.std(gp_var) + 1e-15), np.ones(len(model_s2))])
    coefs_gp = la.lstsq(Z_gp, model_s2)[0]
    r2_gp_alone = compute_r2(model_s2, np.dot(Z_gp, coefs_gp))
    
    # Partial R^2 for NN dist alone
    Z_nn = np.column_stack([(nn_dist - np.mean(nn_dist)) / (np.std(nn_dist) + 1e-15), np.ones(len(model_s2))])
    coefs_nn = la.lstsq(Z_nn, model_s2)[0]
    r2_nn_alone = compute_r2(model_s2, np.dot(Z_nn, coefs_nn))
    
    return {
        "full_r2": float(full_r2),
        "r2_gp_alone": float(r2_gp_alone),
        "r2_nn_alone": float(r2_nn_alone),
        "beta_gp_std": float(beta_gp),
        "beta_nn_std": float(beta_nn),
        "beta_dens_std": float(beta_dens)
    }
