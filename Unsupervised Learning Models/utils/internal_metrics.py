import numpy as np

def pairwise_sq_dists(X: np.ndarray, Y: np.ndarray):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    # (x - y)^2 = x^2 + y^2 - 2xy
    X2 = np.sum(X * X, axis=1, keepdims=True)      # (n,1)
    Y2 = np.sum(Y * Y, axis=1, keepdims=True).T    # (1,m)
    D2 = X2 + Y2 - 2.0 * (X @ Y.T)
    return np.maximum(D2, 0.0)

def wcss(X: np.ndarray, labels: np.ndarray, centers: np.ndarray):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    centers = np.asarray(centers, dtype=float)

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

    # Full distance matrix
    D2 = pairwise_sq_dists(X, X)
    D = np.sqrt(D2)

    s_vals = np.zeros(n, dtype=float)

    for i in range(n):
        ci = labels[i]

        same = (labels == ci)
        same[i] = False  # exclude itself
        a = float(np.mean(D[i, same])) if np.any(same) else 0.0

        b = np.inf
        for c in unique:
            if c == ci:
                continue
            other = (labels == c)
            if np.any(other):
                b = min(b, float(np.mean(D[i, other])))

        denom = max(a, b)
        s_vals[i] = 0.0 if denom <= 1e-12 else (b - a) / denom

    return float(np.mean(s_vals))

def gap_statistic(
    X: np.ndarray,
    clusterer_factory,
    k: int,
    B: int = 10,
    random_state: int = 42,
):
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(int(random_state))

    # --- Fit on real data ---
    seed_real = int(rng.integers(0, 1_000_000_000))
    km = clusterer_factory(int(k), seed_real)
    km.fit(X)
    Wk = float(getattr(km, "inertia_", 0.0))
    logWk = np.log(max(Wk, 1e-12))

    # Reference distribution: uniform within feature-wise bounds 
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    ref_logs = np.empty(int(B), dtype=float)
    for b in range(int(B)):
        seedb = int(rng.integers(0, 1_000_000_000))
        rngb = np.random.default_rng(seedb)

        Xref = rngb.uniform(mins, maxs, size=X.shape)

        km_ref = clusterer_factory(int(k), seedb)
        km_ref.fit(Xref)
        Wkb = float(getattr(km_ref, "inertia_", 0.0))
        ref_logs[b] = np.log(max(Wkb, 1e-12))

    gap = float(np.mean(ref_logs) - logWk)
    sk = float(np.std(ref_logs, ddof=1) * np.sqrt(1.0 + 1.0 / B)) if B > 1 else 0.0
    return gap, sk

def best_k_by_gap(gaps: list, sks: list, ks: list):
    for i in range(len(ks) - 1):
        if gaps[i] >= gaps[i + 1] - sks[i + 1]:
            return ks[i]
    return ks[-1]

def davies_bouldin_index(X, labels):
    X = np.array(X)
    labels = np.array(labels)
    clusters = np.unique(labels)
    K = len(clusters)

    # Compute cluster centroids
    centroids = np.array([X[labels == k].mean(axis=0) for k in clusters])
    
    # Compute intra-cluster distances
    S = np.zeros(K)
    for i, k in enumerate(clusters):
        cluster_points = X[labels == k]
        S[i] = np.mean(np.linalg.norm(cluster_points - centroids[i], axis=1))
    
    # Compute pairwise centroid distances
    M = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                M[i, j] = np.linalg.norm(centroids[i] - centroids[j])
    
    # Compute DBI
    R = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                R[i, j] = (S[i] + S[j]) / M[i, j]
    DB = np.mean(np.max(R, axis=1))
    return DB

def calinski_harabasz_index(X, labels):
    X = np.array(X)
    labels = np.array(labels)
    N, n_features = X.shape
    clusters = np.unique(labels)
    K = len(clusters)

    # Overall mean
    global_mean = X.mean(axis=0)

    # Between-cluster dispersion
    B = 0
    W = 0
    for k in clusters:
        cluster_points = X[labels == k]
        n_k = cluster_points.shape[0]
        centroid = cluster_points.mean(axis=0)
        B += n_k * np.sum((centroid - global_mean) ** 2)
        W += np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)

    CH = (B / (K - 1)) / (W / (N - K))
    return CH
