import numpy as np

def _logsumexp(a, axis=1):
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return out

def _gaussian_logpdf_full(X, mean, cov, reg=1e-6):
    d = X.shape[1]
    cov = cov + reg * np.eye(d)
    L = np.linalg.cholesky(cov)
    sol = np.linalg.solve(L, (X - mean).T)  # d x n
    quad = np.sum(sol**2, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2*np.pi) + logdet + quad)

def _gaussian_logpdf_diag(X, mean, var, reg=1e-6):
    var = var + reg
    d = X.shape[1]
    quad = np.sum(((X - mean) ** 2) / var, axis=1)
    logdet = np.sum(np.log(var))
    return -0.5 * (d * np.log(2*np.pi) + logdet + quad)

class GMM:
    """Gaussian Mixture Model with EM from scratch.
    covariance_type: 'full' | 'tied' | 'diag' | 'spherical'
    """

    def __init__(self, n_components=2, covariance_type="full", max_iter=200, tol=1e-4,
                 reg_covar=1e-6, seed=42):
        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)
        self.seed = int(seed)

        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.converged_ = False
        self.n_iter_ = 0
        self.lower_bound_ = None
        self.log_likelihood_history_ = []

    def _init_params(self, X):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        k = self.n_components

        # random means from data
        idx = rng.choice(n, size=k, replace=False)
        self.means_ = X[idx].copy()

        self.weights_ = np.full(k, 1.0 / k)

        # init covariances
        if self.covariance_type == "full":
            base = np.cov(X, rowvar=False) + self.reg_covar*np.eye(d)
            self.covariances_ = np.array([base.copy() for _ in range(k)])
        elif self.covariance_type == "tied":
            self.covariances_ = np.cov(X, rowvar=False) + self.reg_covar*np.eye(d)
        elif self.covariance_type == "diag":
            var = X.var(axis=0) + self.reg_covar
            self.covariances_ = np.array([var.copy() for _ in range(k)])
        elif self.covariance_type == "spherical":
            v = float(np.mean(X.var(axis=0)) + self.reg_covar)
            self.covariances_ = np.full(k, v)
        else:
            raise ValueError("covariance_type must be one of: full, tied, diag, spherical")

    def _estimate_log_prob(self, X):
        n, d = X.shape
        k = self.n_components
        log_prob = np.zeros((n, k))

        if self.covariance_type == "full":
            for j in range(k):
                log_prob[:, j] = _gaussian_logpdf_full(X, self.means_[j], self.covariances_[j], reg=self.reg_covar)
        elif self.covariance_type == "tied":
            for j in range(k):
                log_prob[:, j] = _gaussian_logpdf_full(X, self.means_[j], self.covariances_, reg=self.reg_covar)
        elif self.covariance_type == "diag":
            for j in range(k):
                log_prob[:, j] = _gaussian_logpdf_diag(X, self.means_[j], self.covariances_[j], reg=self.reg_covar)
        elif self.covariance_type == "spherical":
            for j in range(k):
                var = self.covariances_[j]
                log_prob[:, j] = _gaussian_logpdf_diag(X, self.means_[j], np.full(d, var), reg=self.reg_covar)
        return log_prob

    def _e_step(self, X):
        # responsibilities
        log_prob = self._estimate_log_prob(X)
        log_weights = np.log(self.weights_ + 1e-12)
        log_joint = log_prob + log_weights
        log_norm = _logsumexp(log_joint, axis=1)
        log_resp = log_joint - log_norm
        resp = np.exp(log_resp)
        ll = float(np.sum(log_norm))
        return resp, ll

    def _m_step(self, X, resp):
        n, d = X.shape
        k = self.n_components
        nk = resp.sum(axis=0) + 1e-12

        self.weights_ = nk / n
        self.means_ = (resp.T @ X) / nk[:, None]

        if self.covariance_type == "full":
            covs = []
            for j in range(k):
                Xc = X - self.means_[j]
                cov = (resp[:, j][:, None] * Xc).T @ Xc / nk[j]
                cov += self.reg_covar * np.eye(d)
                covs.append(cov)
            self.covariances_ = np.array(covs)
        elif self.covariance_type == "tied":
            cov = np.zeros((d, d))
            for j in range(k):
                Xc = X - self.means_[j]
                cov += (resp[:, j][:, None] * Xc).T @ Xc
            cov /= n
            cov += self.reg_covar * np.eye(d)
            self.covariances_ = cov
        elif self.covariance_type == "diag":
            covs = []
            for j in range(k):
                Xc = X - self.means_[j]
                var = (resp[:, j][:, None] * (Xc ** 2)).sum(axis=0) / nk[j]
                var += self.reg_covar
                covs.append(var)
            self.covariances_ = np.array(covs)
        elif self.covariance_type == "spherical":
            vars_ = np.zeros(k)
            for j in range(k):
                Xc = X - self.means_[j]
                var = (resp[:, j] * np.sum(Xc**2, axis=1)).sum() / (nk[j] * d)
                vars_[j] = var + self.reg_covar
            self.covariances_ = vars_

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self._init_params(X)

        prev_ll = None
        for it in range(1, self.max_iter + 1):
            resp, ll = self._e_step(X)
            self._m_step(X, resp)
            self.log_likelihood_history_.append(ll)

            if prev_ll is not None and abs(ll - prev_ll) <= self.tol * (abs(prev_ll) + 1e-12):
                self.converged_ = True
                self.n_iter_ = it
                break
            prev_ll = ll

        if not self.converged_:
            self.n_iter_ = self.max_iter

        self.lower_bound_ = self.log_likelihood_history_[-1] if self.log_likelihood_history_ else None
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        log_prob = self._estimate_log_prob(X)
        log_weights = np.log(self.weights_ + 1e-12)
        log_joint = log_prob + log_weights
        log_norm = _logsumexp(log_joint, axis=1)
        resp = np.exp(log_joint - log_norm)
        return resp

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score_samples(self, X):
        X = np.asarray(X, dtype=float)
        log_prob = self._estimate_log_prob(X)
        log_weights = np.log(self.weights_ + 1e-12)
        log_joint = log_prob + log_weights
        log_norm = _logsumexp(log_joint, axis=1)
        return log_norm.reshape(-1)

    def log_likelihood(self, X):
        return float(np.sum(self.score_samples(X)))

    def bic(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        k = self.n_components
        ll = self.log_likelihood(X)

        # parameter count
        if self.covariance_type == "full":
            cov_params = k * (d * (d + 1) / 2)
        elif self.covariance_type == "tied":
            cov_params = (d * (d + 1) / 2)
        elif self.covariance_type == "diag":
            cov_params = k * d
        elif self.covariance_type == "spherical":
            cov_params = k

        mean_params = k * d
        weight_params = k - 1
        p = mean_params + cov_params + weight_params
        return float(-2 * ll + p * np.log(n))

    def aic(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        k = self.n_components
        ll = self.log_likelihood(X)

        if self.covariance_type == "full":
            cov_params = k * (d * (d + 1) / 2)
        elif self.covariance_type == "tied":
            cov_params = (d * (d + 1) / 2)
        elif self.covariance_type == "diag":
            cov_params = k * d
        elif self.covariance_type == "spherical":
            cov_params = k

        mean_params = k * d
        weight_params = k - 1
        p = mean_params + cov_params + weight_params
        return float(-2 * ll + 2 * p)
