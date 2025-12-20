import numpy as np

def pairwise_sq_dists(X: np.ndarray, Y: np.ndarray):
    # (x - y)^2 = x^2 + y^2 - 2xy
    X2 = np.sum(X * X, axis=1, keepdims=True)  # (n,1)
    Y2 = np.sum(Y * Y, axis=1, keepdims=True).T  # (1,m)
    return X2 + Y2 - 2.0 * (X @ Y.T)


def wcss(X: np.ndarray, labels: np.ndarray, centers: np.ndarray):
    s = 0.0
    for k in range(centers.shape[0]):
        mask = (labels == k)
        if np.any(mask):
            dif = X[mask] - centers[k]
            s += float(np.sum(dif * dif))
    return s


def silhouette_score(X: np.ndarray, labels: np.ndarray):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n = X.shape[0]
    unique = np.unique(labels)
    if unique.size < 2:
        return 0.0

    # Precompute full distance matrix (euclidean)
    D2 = pairwise_sq_dists(X, X)
    D2 = np.maximum(D2, 0.0)
    D = np.sqrt(D2)

    # For each point i:
    # a(i) = avg distance to points in same cluster
    # b(i) = min avg distance to points in other clusters
    s_vals = np.zeros(n, dtype=float)

    for i in range(n):
        ci = labels[i]
        same = (labels == ci)
        same[i] = False

        if np.any(same):
            a = np.mean(D[i, same])
        else:
            a = 0.0

        b = np.inf
        for c in unique:
            if c == ci:
                continue
            other = (labels == c)
            if np.any(other):
                b = min(b, float(np.mean(D[i, other])))

        denom = max(a, b)
        if denom <= 1e-12:
            s_vals[i] = 0.0
        else:
            s_vals[i] = (b - a) / denom

    return float(np.mean(s_vals))


def gap_statistic(
    X: np.ndarray,
    clusterer_factory,
    k: int,
    B: int = 10,
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)

    # Fit on real data
    km = clusterer_factory(k, int(rng.integers(0, 1_000_000_000)))
    Wk = km.inertia_
    logWk = np.log(max(Wk, 1e-12))

    # Reference distribution: uniform within each feature bounds
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    gaps = []

    ref_logs = []
    for b in range(B):
        seedb = int(rng.integers(0, 1_000_000_000))
        rngb = np.random.default_rng(seedb)
        Xref = rngb.uniform(mins, maxs, size=X.shape)

        km_ref = clusterer_factory(k, seedb)
        ref_logs.append(np.log(max(km_ref.inertia_, 1e-12)))

    ref_logs = np.array(ref_logs)
    gap = float(np.mean(ref_logs) - logWk)
    sk = float(np.std(ref_logs, ddof=1) * np.sqrt(1.0 + 1.0 / B)) if B > 1 else 0.0
    return gap, sk


def best_k_by_gap(gaps: list, sks: list, ks: list):
    for i in range(len(ks) - 1):
        if gaps[i] >= gaps[i + 1] - sks[i + 1]:
            return ks[i]
    return ks[-1]
