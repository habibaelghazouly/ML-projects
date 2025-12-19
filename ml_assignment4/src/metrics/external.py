import numpy as np

def confusion_matrix(y_true, y_pred, labels=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    labels = np.asarray(labels)
    m = len(labels)
    idx = {lab:i for i, lab in enumerate(labels)}
    cm = np.zeros((m, m), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm, labels

def purity_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm, _ = confusion_matrix(y_true, y_pred)
    return float(np.sum(np.max(cm, axis=0)) / np.sum(cm))

def adjusted_rand_index(y_true, y_pred):
    # ARI from scratch using contingency table
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes, class_idx = np.unique(y_true, return_inverse=True)
    clusters, cluster_idx = np.unique(y_pred, return_inverse=True)
    n_classes = len(classes)
    n_clusters = len(clusters)
    n = len(y_true)

    cont = np.zeros((n_classes, n_clusters), dtype=int)
    for i in range(n):
        cont[class_idx[i], cluster_idx[i]] += 1

    def comb2(x): return x * (x - 1) // 2

    sum_comb_c = sum(comb2(n_i) for n_i in cont.sum(axis=1))
    sum_comb_k = sum(comb2(n_j) for n_j in cont.sum(axis=0))
    sum_comb = sum(comb2(n_ij) for n_ij in cont.ravel())
    comb_n = comb2(n)

    expected = (sum_comb_c * sum_comb_k) / comb_n if comb_n > 0 else 0.0
    max_index = 0.5 * (sum_comb_c + sum_comb_k)
    denom = max_index - expected
    if denom == 0:
        return 0.0
    return float((sum_comb - expected) / denom)

def normalized_mutual_info(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes, class_idx = np.unique(y_true, return_inverse=True)
    clusters, cluster_idx = np.unique(y_pred, return_inverse=True)
    n_classes = len(classes)
    n_clusters = len(clusters)
    n = len(y_true)

    cont = np.zeros((n_classes, n_clusters), dtype=float)
    for i in range(n):
        cont[class_idx[i], cluster_idx[i]] += 1.0

    pi = cont.sum(axis=1) / n
    pj = cont.sum(axis=0) / n
    pij = cont / n

    # mutual information
    mi = 0.0
    for i in range(n_classes):
        for j in range(n_clusters):
            if pij[i, j] > 0 and pi[i] > 0 and pj[j] > 0:
                mi += pij[i, j] * np.log(pij[i, j] / (pi[i] * pj[j]))

    # entropies
    Hi = -np.sum(pi[pi > 0] * np.log(pi[pi > 0]))
    Hj = -np.sum(pj[pj > 0] * np.log(pj[pj > 0]))
    denom = (Hi + Hj) / 2.0
    if denom == 0:
        return 0.0
    return float(mi / denom)
