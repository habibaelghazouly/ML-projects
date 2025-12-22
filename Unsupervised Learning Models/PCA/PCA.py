import numpy as np

class PCA:
    def __init__(self, n_components=None, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

        self.mean_ = None
        self.components_ = None           # shape (n_components, n_features)
        self.explained_variance_ = None   # eigenvalues chosen
        self.explained_variance_ratio_ = None
        self.eigenvalues_all_ = None
        self.components_all_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        n, d = X.shape

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # Covariance matrix (d x d)
        # Use (n-1) for sample covariance
        cov = (Xc.T @ Xc) / max(n - 1, 1)

        # Eigen decomposition (cov is symmetric -> use eigh)
        eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigvals

        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]  # columns are eigenvectors

        self.eigenvalues_all_ = eigvals
        self.components_all_ = eigvecs.T  # rows as components

        # Pick n_components
        if self.n_components is None:
            m = d
        else:
            m = int(self.n_components)
            m = max(1, min(m, d))

        self.components_ = self.components_all_[:m]  # (m, d)
        self.explained_variance_ = eigvals[:m]

        total_var = float(np.sum(eigvals)) if np.sum(eigvals) > 0 else 1.0
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        return self

    def transform(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        Xc = X - self.mean_
        # Project: Z = Xc * components^T
        return Xc @ self.components_.T

    def fit_transform(self, X: np.ndarray):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray):
        Z = np.asarray(Z, dtype=float)
        # Reconstruct: Xhat = Z * components + mean
        return Z @ self.components_ + self.mean_

    def reconstruction_error(self, X: np.ndarray):
        # Mean squared reconstruction error per element.
        X = np.asarray(X, dtype=float)
        Z = self.transform(X)
        Xhat = self.inverse_transform(Z)
        mse = np.mean((X - Xhat) ** 2)
        return float(mse)
