import numpy as np
from itertools import combinations
from collections import Counter
from math import factorial

def comb(n, k):
    if n < k or k < 0:
        return 0
    if k == 0:
        return 1
    return factorial(n) // (factorial(k) * factorial(n - k))

def adjusted_rand_index(labels_true, labels_pred):
    n = len(labels_true)
    assert n == len(labels_pred)
    
    # Count pairs in each class
    sum_comb_c = sum(comb(c, 2) for c in Counter(labels_true).values())
    sum_comb_k = sum(comb(c, 2) for c in Counter(labels_pred).values())
    
    # Contingency table
    contingency = {}
    for t, p in zip(labels_true, labels_pred):
        contingency[(t, p)] = contingency.get((t, p), 0) + 1
    
    sum_comb = sum(comb(v, 2) for v in contingency.values())
    
    # Expected index
    n_comb = comb(n, 2)
    if n_comb == 0:  # Handle edge case
        return 1.0 if sum_comb_c == sum_comb_k == 0 else 0.0
    
    prod_comb = sum_comb_c * sum_comb_k / n_comb
    max_index = (sum_comb_c + sum_comb_k) / 2
    
    # Handle division by zero
    if max_index == prod_comb:
        return 1.0
    
    ari = (sum_comb - prod_comb) / (max_index - prod_comb)
    return ari

def normalized_mutual_info(labels_true, labels_pred):
    n = len(labels_true)
    assert n == len(labels_pred)
    
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    
    # Unique classes
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    
    # Build contingency table
    contingency = np.zeros((len(classes), len(clusters)))
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            contingency[i, j] = np.sum((labels_true == c) & (labels_pred == k))
    
    # Mutual information
    MI = 0
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    
    for i in range(len(classes)):
        for j in range(len(clusters)):
            if contingency[i, j] > 0:
                MI += (contingency[i, j] / n) * np.log(
                    (contingency[i, j] * n) / (row_sums[i] * col_sums[j])
                )
    
    # Entropies (fixed: remove +1e-16 inside log, handle zeros properly)
    H_true = 0
    for rs in row_sums:
        if rs > 0:
            p = rs / n
            H_true -= p * np.log(p)
    
    H_pred = 0
    for cs in col_sums:
        if cs > 0:
            p = cs / n
            H_pred -= p * np.log(p)
    
    # Handle edge cases
    if H_true + H_pred == 0:
        return 0.0 if MI == 0 else 1.0
    
    NMI = MI / ((H_true + H_pred) / 2)
    return NMI

def purity_score(labels_true, labels_pred):
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    
    clusters = np.unique(labels_pred)
    n = len(labels_true)
    
    purity = 0
    for k in clusters:
        cluster_idx = np.where(labels_pred == k)[0]
        true_labels_in_cluster = labels_true[cluster_idx]
        most_common = Counter(true_labels_in_cluster).most_common(1)[0][1]
        purity += most_common
    
    return purity / n