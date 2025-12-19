import numpy as np
import matplotlib.pyplot as plt

def scatter_2d(Z2, labels, title="", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    Z2 = np.asarray(Z2)
    labels = np.asarray(labels)
    for c in np.unique(labels):
        idx = labels == c
        ax.scatter(Z2[idx, 0], Z2[idx, 1], s=18, alpha=0.8, label=str(c))
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(frameon=False, fontsize=8)
    return ax

def line_curve(x, y, title="", xlabel="", ylabel="", mark_x=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o")
    if mark_x is not None:
        ax.axvline(mark_x, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

def heatmap(matrix, row_labels, col_labels, title=""):
    import seaborn as sns
    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, fmt=".3f", xticklabels=col_labels, yticklabels=row_labels)
    plt.title(title)
    plt.tight_layout()
