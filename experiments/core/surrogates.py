import numpy as np

class ExactGP:
    def __init__(self, X, sigma=0.3, lengthscale=1.0):
        self.X = X
        self.n = len(X)
        self.sigma = sigma
        dists = np.sum((X[:, None] - X[None, :])**2, axis=-1)
        self.K = np.exp(-dists / (2 * lengthscale**2))
        self.inv = np.linalg.inv(self.K + sigma**2 * np.eye(self.n))
        self.W = self.K @ self.inv

    def predict(self, y):
        return self.W @ y

    def jacobian(self, y):
        return self.W

    def predictive_variance(self, y=None):
        """s^2_i = Var(y_new,i | y) for a fresh observation at x_i.

        Computed from the closed form Cov(f|y) = K - K(K+s2 I)^-1 K, PLUS the
        observation noise s2. Deliberately NOT computed as sigma^2 (1 + J_ii):
        the point of the A3 control is that the two channels come from
        independent code paths, so agreement is evidence rather than tautology.

        y-independent for a fixed-kernel GP; the argument is accepted for
        interface symmetry with HierarchicalGP.
        """
        post_cov = self.K - self.K @ self.inv @ self.K
        return np.diag(post_cov) + self.sigma**2

class Ridge:
    def __init__(self, X, lam=0.1):
        self.X = X
        self.n = len(X)
        self.lam = lam
        self.inv = np.linalg.inv(X.T @ X + lam * np.eye(X.shape[1]))
        self.W = X @ self.inv @ X.T

    def predict(self, y):
        return self.W @ y

    def jacobian(self, y):
        return self.W

class NadarayaWatson:
    def __init__(self, X, lengthscale=1.0):
        self.X = X
        self.n = len(X)
        dists = np.sum((X[:, None] - X[None, :])**2, axis=-1)
        self.K = np.exp(-dists / (2 * lengthscale**2))
        self.d = np.sum(self.K, axis=1)
        self.W = self.K / self.d[:, None]

    def predict(self, y):
        return self.W @ y

    def jacobian(self, y):
        return self.W

class OneNN:
    def __init__(self, X):
        self.X = X
        self.n = len(X)
        self.W = np.eye(self.n) # at context points, 1-NN is identity

    def predict(self, y):
        return self.W @ y

    def jacobian(self, y):
        return self.W

class HierarchicalGP:
    def __init__(self, X, sigma=0.3):
        self.X = X
        self.n = len(X)
        self.sigma = sigma
        self.lengthscales = np.logspace(np.log10(0.2), np.log10(5.0), 8)
        self.Ks = []
        self.invs = []
        for ls in self.lengthscales:
            dists = np.sum((X[:, None] - X[None, :])**2, axis=-1)
            K = np.exp(-dists / (2 * ls**2))
            self.Ks.append(K)
            self.invs.append(np.linalg.inv(K + self.sigma**2 * np.eye(self.n)))

    def get_w(self, y):
        log_w = []
        for K, inv in zip(self.Ks, self.invs):
            Cov = K + self.sigma**2 * np.eye(self.n)
            sign, logdet = np.linalg.slogdet(Cov)
            lw = -0.5 * (logdet + y.T @ inv @ y + self.n * np.log(2 * np.pi))
            log_w.append(lw)
        log_w = np.array(log_w)
        w = np.exp(log_w - np.max(log_w))
        return w / np.sum(w)

    def predict(self, y):
        w = self.get_w(y)
        out = np.zeros(self.n)
        for i in range(len(self.Ks)):
            W_i = self.Ks[i] @ self.invs[i]
            out += w[i] * (W_i @ y)
        return out

    def jacobian(self, y):
        w = self.get_w(y)
        J = np.zeros((self.n, self.n))
        
        v = []
        for i in range(len(self.Ks)):
            C_inv = self.invs[i]
            v.append(-C_inv @ y)
            
        v_bar = np.zeros(self.n)
        for i in range(len(self.Ks)):
            v_bar += w[i] * v[i]
            
        for i in range(len(self.Ks)):
            W_i = self.Ks[i] @ self.invs[i]
            m_i = W_i @ y
            J += w[i] * W_i
            J += w[i] * np.outer(m_i, v[i] - v_bar)
            
        return J

    def posterior_covariance(self, y):
        """Cov(f|y) = E_{theta|y}[C_theta] + Cov_{theta|y}(mu_theta), i.e. T4.

        Built from the mixture directly. Independent of jacobian().
        """
        w = self.get_w(y)
        n, m = self.n, len(self.Ks)
        EC = np.zeros((n, n))
        mus = []
        for i in range(m):
            EC += w[i] * (self.Ks[i] - self.Ks[i] @ self.invs[i] @ self.Ks[i])
            mus.append(self.Ks[i] @ self.invs[i] @ y)
        mbar = sum(w[i] * mus[i] for i in range(m))
        Cov = np.zeros((n, n))
        for i in range(m):
            d = mus[i] - mbar
            Cov += w[i] * np.outer(d, d)
        return EC + Cov

    def predictive_variance(self, y):
        """s^2_i for a fresh observation at x_i, from T4. See ExactGP note."""
        return np.diag(self.posterior_covariance(y)) + self.sigma**2

class Imitator:
    def __init__(self, X, sigma=0.3, lengthscale=1.0, epsilon=0.01):
        self.egp = ExactGP(X, sigma=sigma, lengthscale=lengthscale)
        self.epsilon = epsilon
        self.n = len(X)
        self.a = np.random.randn(self.n)
        self.a /= np.linalg.norm(self.a)
        self.b = np.random.randn(self.n)
        self.b /= np.linalg.norm(self.b)

    def predict(self, y):
        m_gp = self.egp.predict(y)
        return m_gp + self.epsilon * np.sin(np.dot(self.a, y) / self.epsilon) * self.b

    def jacobian(self, y):
        J_gp = self.egp.jacobian(y)
        return J_gp + np.cos(np.dot(self.a, y) / self.epsilon) * np.outer(self.b, self.a)
