import numpy as np
from ..metrics.internal import silhouette_score
from ..clustering.kmeans import KMeans

def elbow_inertia(X, k_values, init="kmeans++", seed=42, max_iter=300, tol=1e-4):
    inertias = []
    iters = []
    for k in k_values:
        km = KMeans(n_clusters=k, init=init, seed=seed, max_iter=max_iter, tol=tol).fit(X)
        inertias.append(km.inertia_)
        iters.append(len(km.inertia_history_))
    return np.array(inertias, dtype=float), np.array(iters, dtype=int)

def silhouette_over_k(X, k_values, init="kmeans++", seed=42, max_iter=300, tol=1e-4):
    scores = []
    for k in k_values:
        km = KMeans(n_clusters=k, init=init, seed=seed, max_iter=max_iter, tol=tol).fit(X)
        scores.append(silhouette_score(X, km.labels_))
    return np.array(scores, dtype=float)

def gap_statistic(X, k_values, B=10, seed=42, init="kmeans++", max_iter=300, tol=1e-4):
    """Gap statistic for KMeans.

    Reference distribution: uniform sampling within feature-wise min/max box.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    n, d = X.shape

    gaps = []
    sk = []
    wk = []

    for k in k_values:
        km = KMeans(n_clusters=k, init=init, seed=seed, max_iter=max_iter, tol=tol).fit(X)
        Wk = km.inertia_
        wk.append(Wk)

        ref_logs = []
        for b in range(B):
            Xref = rng.uniform(mins, maxs, size=(n, d))
            km_ref = KMeans(n_clusters=k, init=init, seed=int(seed + 1000 + b), max_iter=max_iter, tol=tol).fit(Xref)
            ref_logs.append(np.log(km_ref.inertia_ + 1e-12))

        ref_logs = np.array(ref_logs)
        gap = float(np.mean(ref_logs) - np.log(Wk + 1e-12))
        gaps.append(gap)
        sk.append(float(np.sqrt(1 + 1/B) * np.std(ref_logs, ddof=1)))

    return np.array(gaps), np.array(sk), np.array(wk)
