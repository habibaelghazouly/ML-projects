import numpy as np

def _pairwise_distances(X):
    # Euclidean distances
    X2 = np.sum(X*X, axis=1, keepdims=True)
    D2 = X2 + X2.T - 2*(X @ X.T)
    D2[D2 < 0] = 0
    return np.sqrt(D2)

def silhouette_score(X, labels):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n = X.shape[0]
    D = _pairwise_distances(X)

    unique = np.unique(labels)
    if len(unique) == 1:
        return 0.0

    a = np.zeros(n)
    b = np.full(n, np.inf)

    for k in unique:
        idx = np.where(labels == k)[0]
        if len(idx) <= 1:
            a[idx] = 0.0
        else:
            a[idx] = (D[np.ix_(idx, idx)].sum(axis=1) - 0.0) / (len(idx) - 1)

        for k2 in unique:
            if k2 == k:
                continue
            idx2 = np.where(labels == k2)[0]
            if len(idx2) == 0:
                continue
            dist = D[np.ix_(idx, idx2)].mean(axis=1)
            b[idx] = np.minimum(b[idx], dist)

    s = (b - a) / np.maximum(a, b)
    s[np.isnan(s)] = 0.0
    return float(np.mean(s))

def davies_bouldin_index(X, labels):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    k = len(unique)
    if k <= 1:
        return 0.0

    centroids = np.array([X[labels == c].mean(axis=0) for c in unique])
    # cluster scatter (avg distance to centroid)
    S = np.zeros(k)
    for i, c in enumerate(unique):
        pts = X[labels == c]
        if len(pts) == 0:
            S[i] = 0.0
        else:
            S[i] = np.mean(np.linalg.norm(pts - centroids[i], axis=1))

    # centroid distances
    M = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    M[M == 0] = np.inf

    R = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                R[i, j] = (S[i] + S[j]) / M[i, j]
    D = np.max(R, axis=1)
    return float(np.mean(D))

def calinski_harabasz_index(X, labels):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n, d = X.shape
    unique = np.unique(labels)
    k = len(unique)
    if k <= 1:
        return 0.0

    overall_mean = X.mean(axis=0)
    # between-cluster dispersion
    B = 0.0
    # within-cluster dispersion
    W = 0.0
    for c in unique:
        pts = X[labels == c]
        ni = pts.shape[0]
        if ni == 0:
            continue
        mean_i = pts.mean(axis=0)
        B += ni * np.sum((mean_i - overall_mean) ** 2)
        W += np.sum((pts - mean_i) ** 2)

    return float((B / (k - 1)) / (W / (n - k) + 1e-12))

def wcss(X, labels, centers=None):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if centers is None:
        unique = np.unique(labels)
        centers = np.array([X[labels == c].mean(axis=0) for c in unique])
        # map cluster id -> row
        mapping = {c:i for i,c in enumerate(unique)}
        idx = np.array([mapping[c] for c in labels])
    else:
        idx = labels
    return float(np.sum((X - centers[idx]) ** 2))
