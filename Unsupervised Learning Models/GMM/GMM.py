import numpy as np


class GMM:
    def __init__(
        self,
        n_components,
        covariance_type="full",
        tol=1e-4,
        max_iter=100,
        reg_covar=1e-6,
        random_state=42,
    ):
        self.K = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        self.random_state = random_state

    # ---------- Initialization ----------
    def _initialize_parameters(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape

        if self.K > n_samples:
            raise ValueError("Number of components cannot exceed number of samples")

        # Initialize means randomly from data
        indices = np.random.choice(n_samples, self.K, replace=False)
        self.means_ = X[indices]

        # Mixing coefficients
        self.weights_ = np.ones(self.K) / self.K

        # Covariances
        if self.covariance_type == "full":
            self.covariances_ = np.array(
                [
                    np.cov(X, rowvar=False) + self.reg_covar * np.eye(n_features)
                    for _ in range(self.K)
                ]
            )
        elif self.covariance_type == "tied":
            self.covariances_ = np.cov(X, rowvar=False) + self.reg_covar * np.eye(
                n_features
            )
        elif self.covariance_type == "diag":
            self.covariances_ = np.tile(np.var(X, axis=0) + self.reg_covar, (self.K, 1))
        elif self.covariance_type == "spherical":
            self.covariances_ = np.ones(self.K) * (
                np.mean(np.var(X, axis=0)) + self.reg_covar
            )
        else:
            raise ValueError("Invalid covariance type")

    # ---------- Gaussian log-density ----------
    def _log_gaussian(self, X, k):
        n_samples, n_features = X.shape
        diff = X - self.means_[k]

        if self.covariance_type == "full":
            cov = self.covariances_[k]
            # Ensure positive definite
            cov += self.reg_covar * np.eye(n_features)
            try:
                chol = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                chol = np.linalg.cholesky(cov + self.reg_covar * np.eye(n_features))
            inv = np.linalg.inv(cov)
            log_det = 2 * np.sum(np.log(np.diag(chol)))
            return -0.5 * (
                np.sum(diff @ inv * diff, axis=1)
                + log_det
                + n_features * np.log(2 * np.pi)
            )

        elif self.covariance_type == "tied":
            cov = self.covariances_
            cov += self.reg_covar * np.eye(n_features)
            chol = np.linalg.cholesky(cov)
            inv = np.linalg.inv(cov)
            log_det = 2 * np.sum(np.log(np.diag(chol)))
            return -0.5 * (
                np.sum(diff @ inv * diff, axis=1)
                + log_det
                + n_features * np.log(2 * np.pi)
            )

        elif self.covariance_type == "diag":
            cov = self.covariances_[k] + self.reg_covar
            return -0.5 * (
                np.sum((diff**2) / cov, axis=1)
                + np.sum(np.log(cov))
                + n_features * np.log(2 * np.pi)
            )

        elif self.covariance_type == "spherical":
            var = self.covariances_[k] + self.reg_covar
            return -0.5 * (
                np.sum(diff**2, axis=1) / var
                + n_features * np.log(var)
                + n_features * np.log(2 * np.pi)
            )

    # ---------- E-step ----------
    def _e_step(self, X):
        log_prob = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            log_prob[:, k] = np.log(self.weights_[k] + 1e-16) + self._log_gaussian(
                X, k
            )  # avoid log(0)
        log_sum = np.logaddexp.reduce(log_prob, axis=1)
        responsibilities = np.exp(log_prob - log_sum[:, np.newaxis])
        return responsibilities, np.sum(log_sum)

    # ---------- M-step ----------
    def _m_step(self, X, resp):
        n_samples, n_features = X.shape
        Nk = np.sum(resp, axis=0) + 1e-16  # avoid division by zero

        # Update weights
        self.weights_ = Nk / n_samples

        # Update means
        self.means_ = (resp.T @ X) / Nk[:, np.newaxis]

        # Update covariances
        if self.covariance_type == "full":
            covariances = []
            for k in range(self.K):
                diff = X - self.means_[k]
                cov = (resp[:, k][:, None] * diff).T @ diff / Nk[k]
                cov += self.reg_covar * np.eye(n_features)
                covariances.append(cov)
            self.covariances_ = np.array(covariances)

        elif self.covariance_type == "tied":
            cov = np.zeros((n_features, n_features))
            for k in range(self.K):
                diff = X - self.means_[k]
                cov += (resp[:, k][:, None] * diff).T @ diff
            cov /= n_samples
            cov += self.reg_covar * np.eye(n_features)
            self.covariances_ = cov

        elif self.covariance_type == "diag":
            covariances = np.zeros((self.K, n_features))
            for k in range(self.K):
                diff = X - self.means_[k]
                covariances[k] = (resp[:, k][:, None] * diff**2).sum(axis=0) / Nk[
                    k
                ] + self.reg_covar
            self.covariances_ = covariances

        elif self.covariance_type == "spherical":
            covariances = np.zeros(self.K)
            for k in range(self.K):
                diff = X - self.means_[k]
                covariances[k] = (resp[:, k] * np.sum(diff**2, axis=1)).sum() / (
                    Nk[k] * n_features
                ) + self.reg_covar
            self.covariances_ = covariances

    # ---------- Training ----------
    def fit(self, X):
        self._initialize_parameters(X)
        self.log_likelihoods_ = []

        for i in range(self.max_iter):
            resp, log_likelihood = self._e_step(X)
            self._m_step(X, resp)
            self.log_likelihoods_.append(log_likelihood)

            if (
                i > 0
                and abs(self.log_likelihoods_[-1] - self.log_likelihoods_[-2])
                < self.tol
            ):
                break

        return self

    # ---------- Prediction ----------
    def predict(self, X):
        resp, _ = self._e_step(X)
        return np.argmax(resp, axis=1)

    def score(self, X):
        _, log_likelihood = self._e_step(X)
        return log_likelihood  # sum log-likelihood

    # ----------------- Number of parameters -----------------
    def _num_parameters(self):
        n_features = self.means_.shape[1]
        K = self.K
        if self.covariance_type == "full":
            return K * n_features + K * n_features * (n_features + 1) // 2 + (K - 1)
        elif self.covariance_type == "tied":
            return K * n_features + n_features * (n_features + 1) // 2 + (K - 1)
        elif self.covariance_type == "diag":
            return 2 * K * n_features + (K - 1)
        elif self.covariance_type == "spherical":
            return K * (n_features + 1) + (K - 1)
        else:
            raise ValueError("Invalid covariance type")

    # ----------------- AIC & BIC -----------------
    def aic(self, X):
        L = self.score(X)
        p = self._num_parameters()
        return 2 * p - 2 * L

    def bic(self, X):
        L = self.score(X)
        p = self._num_parameters()
        N = X.shape[0]
        return np.log(N) * p - 2 * L
