import numpy as np

def _pairwise_sq_dists(X: np.ndarray, C: np.ndarray):
    X2 = np.sum(X * X, axis=1, keepdims=True)
    C2 = np.sum(C * C, axis=1, keepdims=True).T
    D2 = X2 + C2 - 2.0 * (X @ C.T)
    return np.maximum(D2, 0.0)   

class KMeans:
    def __init__(
        self,
        n_clusters: int = 2,
        init: str = "kmeans++",
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        self.n_clusters = int(n_clusters)
        self.init = init
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.inertia_history_ = []
        self.n_iter_ = 0

    def _init_random(self, X, rng):
        n = X.shape[0]
        idx = rng.choice(n, size=self.n_clusters, replace=False)
        return X[idx].copy()

    def _init_kmeanspp(self, X, rng):
        n, d = X.shape
        centers = np.empty((self.n_clusters, d), dtype=float)

        # choose first center uniformly
        i0 = int(rng.integers(0, n))
        centers[0] = X[i0]

        closest_d2 = _pairwise_sq_dists(X, centers[0:1]).reshape(-1)
        closest_d2 = np.maximum(closest_d2, 0.0)  

        for k in range(1, self.n_clusters):
            s = float(np.sum(closest_d2))

            if s <= 1e-12:
                idx = int(rng.integers(0, n))
            else:
                probs = closest_d2 / s
                probs = np.maximum(probs, 0.0)      
                probs = probs / probs.sum()        
                idx = int(rng.choice(n, p=probs))

            centers[k] = X[idx]

            d2_new = _pairwise_sq_dists(X, centers[k:k+1]).reshape(-1)
            d2_new = np.maximum(d2_new, 0.0)
            closest_d2 = np.minimum(closest_d2, d2_new)

        return centers
    
    def _assign_labels(self, X, centers):
        d2 = _pairwise_sq_dists(X, centers)  # (n,k)
        labels = np.argmin(d2, axis=1)
        inertia = float(np.sum(d2[np.arange(X.shape[0]), labels]))
        return labels, inertia

    def _update_centers(self, X, labels, rng):
        d = X.shape[1]
        centers = np.zeros((self.n_clusters, d), dtype=float)
        for k in range(self.n_clusters):
            mask = (labels == k)
            if np.any(mask):
                centers[k] = X[mask].mean(axis=0)
            else:
                # empty cluster: re-seed randomly
                centers[k] = X[int(rng.integers(0, X.shape[0]))]
        return centers

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        # init
        if self.init == "random":
            centers = self._init_random(X, rng)
        elif self.init == "kmeans++":
            centers = self._init_kmeanspp(X, rng)
        else:
            raise ValueError("init must be 'random' or 'kmeans++'")

        self.inertia_history_ = []

        prev_inertia = None
        for it in range(1, self.max_iter + 1):
            labels, inertia = self._assign_labels(X, centers)
            self.inertia_history_.append(inertia)

            new_centers = self._update_centers(X, labels, rng)

            # convergence check: centroid shift
            shift = float(np.sqrt(np.sum((centers - new_centers) ** 2)))

            # also allow inertia improvement tolerance
            if prev_inertia is not None:
                inertia_improve = prev_inertia - inertia
            else:
                inertia_improve = np.inf

            centers = new_centers
            prev_inertia = inertia
            self.n_iter_ = it

            if shift <= self.tol:
                break
            if inertia_improve >= 0 and inertia_improve <= self.tol:
                break

        # final
        labels, inertia = self._assign_labels(X, centers)
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = float(inertia)
        return self

    def fit_predict(self, X: np.ndarray):
        self.fit(X)
        return self.labels_

    def predict(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        labels, _ = self._assign_labels(X, self.cluster_centers_)
        return labels
