import numpy as np

def generate_context(n=5, d=5, sigma=0.3, lengthscale=1.0, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    dists = np.sum((X[:, None] - X[None, :])**2, axis=-1)
    K = np.exp(-dists / (2 * lengthscale**2))
    L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
    f = L @ rng.randn(n)
    y = f + sigma * rng.randn(n)
    return X, y, f
    
def generate_noisy_context(n=100, d=5, sigma=0.5, lengthscale=1.0, seed=42):
    rng = np.random.RandomState(seed)
    num_unique = n - 5
    X_unique = rng.randn(num_unique, d)
    # duplicate the first 5 to force smoothing (conflicting labels with same context)
    X = np.vstack([X_unique, X_unique[:5]])
    dists = np.sum((X[:, None] - X[None, :])**2, axis=-1)
    K = np.exp(-dists / (2 * lengthscale**2))
    L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
    f = L @ rng.randn(n)
    y = f + sigma * rng.randn(n)
    return X, y, f

def generate_audit_context(n=100, d=5, sigma=1.0, lengthscale=1.0, seed=42):
    """
    Context designed to force smoothing: substantial label noise, 
    ~10 near-duplicate x pairs with conflicting labels.
    """
    rng = np.random.RandomState(seed)
    num_unique = n - 10
    X_unique = rng.randn(num_unique, d)
    
    # Create 10 duplicate/near-duplicate pairs
    # We will just duplicate the first 10 points
    X = np.vstack([X_unique, X_unique[:10]])
    dists = np.sum((X[:, None] - X[None, :])**2, axis=-1)
    K = np.exp(-dists / (2 * lengthscale**2))
    
    # We need a small nugget for cholesky since there are exact duplicates
    L = np.linalg.cholesky(K + 1e-4 * np.eye(n))
    f = L @ rng.randn(n)
    
    # Force conflicting labels on the duplicates by using high noise
    y = f + sigma * rng.randn(n)
    return X, y, f
