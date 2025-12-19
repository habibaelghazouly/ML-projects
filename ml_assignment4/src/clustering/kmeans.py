import numpy as np

def _pairwise_sq_dists(A, B):
    A2 = np.sum(A*A, axis=1, keepdims=True)
    B2 = np.sum(B*B, axis=1, keepdims=True).T
    D2 = A2 + B2 - 2*(A @ B.T)

    # fix floating errors + non-finite values
    D2 = np.maximum(D2, 0.0)
    D2 = np.nan_to_num(D2, nan=0.0, posinf=0.0, neginf=0.0)
    return D2


def kmeans_plusplus_init(X, k, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=float)

    # choose first center randomly
    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]

    # choose remaining centers
    d2 = _pairwise_sq_dists(X, centers[0:1]).reshape(-1)
    for i in range(1, k):
        d2 = np.nan_to_num(d2, nan=0.0, posinf=0.0, neginf=0.0)
        d2 = np.maximum(d2, 0.0)

        s = float(np.sum(d2))
        if not np.isfinite(s) or s <= 1e-12:
            idx = rng.integers(0, n)
        else:
            probs = d2 / s

            # final safety: clip and renormalize
            probs = np.maximum(probs, 0.0)
            ps = probs.sum()
            if not np.isfinite(ps) or ps <= 1e-12:
                idx = rng.integers(0, n)
            else:
                probs = probs / ps
                idx = rng.choice(n, p=probs)

        centers[i] = X[idx]
        d2_new = _pairwise_sq_dists(X, centers[i:i+1]).reshape(-1)
        d2 = np.minimum(d2, d2_new)

    return centers

def random_init(X, k, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=k, replace=False)
    return X[idx].copy()

class KMeans:
    def __init__(self, n_clusters=2, init="kmeans++", max_iter=300, tol=1e-4, seed=42):
        self.n_clusters = int(n_clusters)
        self.init = init
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed = int(seed)

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.inertia_history_ = []

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        k = self.n_clusters

        if self.init == "kmeans++":
            centers = kmeans_plusplus_init(X, k, seed=self.seed)
        elif self.init == "random":
            centers = random_init(X, k, seed=self.seed)
        else:
            raise ValueError("init must be 'kmeans++' or 'random'")

        prev_inertia = None
        for it in range(self.max_iter):
            d2 = _pairwise_sq_dists(X, centers)
            labels = np.argmin(d2, axis=1)
            inertia = float(np.sum(d2[np.arange(n), labels]))
            self.inertia_history_.append(inertia)

            # update centers
            new_centers = np.zeros_like(centers)
            for j in range(k):
                pts = X[labels == j]
                if len(pts) == 0:
                    # empty cluster: re-seed randomly
                    new_centers[j] = X[np.random.randint(0, n)]
                else:
                    new_centers[j] = pts.mean(axis=0)

            # convergence check
            shift = np.sqrt(np.sum((centers - new_centers) ** 2))
            centers = new_centers

            if prev_inertia is not None:
                if abs(prev_inertia - inertia) <= self.tol * (prev_inertia + 1e-12):
                    break
            if shift <= self.tol:
                break
            prev_inertia = inertia

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = self.inertia_history_[-1] if self.inertia_history_ else None
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        d2 = _pairwise_sq_dists(X, self.cluster_centers_)
        return np.argmin(d2, axis=1)
