import numpy as np

class PCA:
    """PCA from scratch using eigen-decomposition of the covariance matrix."""

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None  # (n_components, n_features)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # Covariance matrix
        cov = (Xc.T @ Xc) / (n_samples - 1)

        # Eigen-decomposition (cov is symmetric)
        eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        if self.n_components is None:
            k = n_features
        else:
            k = int(self.n_components)

        self.components_ = eigvecs[:, :k].T
        self.explained_variance_ = eigvals[:k]
        total_var = eigvals.sum()
        self.explained_variance_ratio_ = (eigvals[:k] / total_var) if total_var > 0 else np.zeros_like(eigvals[:k])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Xc = X - self.mean_
        return Xc @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=float)
        return Z @ self.components_ + self.mean_

    def reconstruction_error(self, X: np.ndarray) -> float:
        X = np.asarray(X, dtype=float)
        Z = self.transform(X)
        Xr = self.inverse_transform(Z)
        return float(np.mean((X - Xr) ** 2))
