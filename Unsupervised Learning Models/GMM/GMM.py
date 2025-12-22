import numpy as np
from sklearn.cluster import KMeans


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

        # Use KMeans to initialize means
        kmeans = KMeans(n_clusters=self.K, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        self.means_ = kmeans.cluster_centers_

        # Initialize weights based on cluster sizes
        self.weights_ = np.array([np.mean(labels == k) for k in range(self.K)])

        # Initialize covariances
        if self.covariance_type == "full":
            covariances = []
            for k in range(self.K):
                cluster_points = X[labels == k]
                if cluster_points.shape[0] == 0:  # fallback if cluster empty
                    cov = np.cov(X, rowvar=False) + self.reg_covar * np.eye(n_features)
                else:
                    cov = np.cov(
                        cluster_points, rowvar=False
                    ) + self.reg_covar * np.eye(n_features)
                covariances.append(cov)
            self.covariances_ = np.array(covariances)

        elif self.covariance_type == "tied":
            cov = np.zeros((n_features, n_features))
            for k in range(self.K):
                cluster_points = X[labels == k]
                if cluster_points.shape[0] == 0:
                    cluster_points = X
                diff = cluster_points - np.mean(cluster_points, axis=0)
                cov += diff.T @ diff
            cov /= n_samples
            cov += self.reg_covar * np.eye(n_features)
            self.covariances_ = cov

        elif self.covariance_type == "diag":
            covariances = np.zeros((self.K, n_features))
            for k in range(self.K):
                cluster_points = X[labels == k]
                if cluster_points.shape[0] == 0:
                    cluster_points = X
                covariances[k] = np.var(cluster_points, axis=0) + self.reg_covar
            self.covariances_ = covariances

        elif self.covariance_type == "spherical":
            covariances = np.zeros(self.K)
            for k in range(self.K):
                cluster_points = X[labels == k]
                if cluster_points.shape[0] == 0:
                    cluster_points = X
                covariances[k] = (
                    np.mean(np.var(cluster_points, axis=0)) + self.reg_covar
                )
            self.covariances_ = covariances

    # ---------- Gaussian log-density ----------
    def _log_gaussian(self, X, k):
        n_samples, n_features = X.shape
        diff = X - self.means_[k]

        if self.covariance_type == "full":
            cov = self.covariances_[k]
            chol = np.linalg.cholesky(cov)
            inv = np.linalg.inv(cov)
            log_det = 2 * np.sum(np.log(np.diag(chol)))
            return -0.5 * (
                np.sum(diff @ inv * diff, axis=1)
                + log_det
                + n_features * np.log(2 * np.pi)
            )

        elif self.covariance_type == "tied":
            cov = self.covariances_
            chol = np.linalg.cholesky(cov)
            inv = np.linalg.inv(cov)
            log_det = 2 * np.sum(np.log(np.diag(chol)))
            return -0.5 * (
                np.sum(diff @ inv * diff, axis=1)
                + log_det
                + n_features * np.log(2 * np.pi)
            )

        elif self.covariance_type == "diag":
            cov = self.covariances_[k]
            return -0.5 * (
                np.sum((diff**2) / cov, axis=1)
                + np.sum(np.log(cov))
                + n_features * np.log(2 * np.pi)
            )

        elif self.covariance_type == "spherical":
            var = self.covariances_[k]
            return -0.5 * (
                np.sum(diff**2, axis=1) / var
                + n_features * np.log(var)
                + n_features * np.log(2 * np.pi)
            )

    # ---------- E-step ----------
    def _e_step(self, X):
        log_prob = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            log_prob[:, k] = np.log(self.weights_[k] + 1e-16) + self._log_gaussian(X, k)
        log_sum = np.logaddexp.reduce(log_prob, axis=1)
        responsibilities = np.exp(log_prob - log_sum[:, np.newaxis])
        return responsibilities, np.sum(log_sum)

    # ---------- M-step ----------
    def _m_step(self, X, resp):
        n_samples, n_features = X.shape
        Nk = np.sum(resp, axis=0) + 1e-16

        # Update weights and means
        self.weights_ = Nk / n_samples
        self.means_ = (resp.T @ X) / Nk[:, np.newaxis]

        # Update covariances
        if self.covariance_type == "full":
            covariances = []
            for k in range(self.K):
                diff = X - self.means_[k]
                cov = (diff.T @ (resp[:, k][:, None] * diff)) / Nk[k]
                cov += self.reg_covar * np.eye(n_features)
                covariances.append(cov)
            self.covariances_ = np.array(covariances)

        elif self.covariance_type == "tied":
            cov = np.zeros((n_features, n_features))
            for k in range(self.K):
                diff = X - self.means_[k]
                cov += diff.T @ (resp[:, k][:, None] * diff)
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

    # ---------- Fit ----------
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

    # ---------- Predict ----------
    def predict(self, X):
        resp, _ = self._e_step(X)
        return np.argmax(resp, axis=1)

    def score(self, X):
        _, log_likelihood = self._e_step(X)
        return log_likelihood

    # ---------- Number of parameters ----------
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

    # ---------- AIC & BIC ----------
    def aic(self, X):
        return 2 * self._num_parameters() - 2 * self.score(X)

    def bic(self, X):
        N = X.shape[0]
        return np.log(N) * self._num_parameters() - 2 * self.score(X)
