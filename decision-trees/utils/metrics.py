from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import pandas as pd
import numpy as np


def compute_metrics(y_true, y_pred, class_names=None):
    """
    Computes classification metrics for each class and overall accuracy.

    Args:
        y_true: True labels (array-like)
        y_pred: Predicted labels (array-like)
        class_names: List of class names in the order of label indices (e.g., ["malignant", "benign"])

    Returns:
        metrics_dict: Dictionary containing accuracy, per-class metrics, and confusion matrix
    """
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(y_true)))]

    # Overall accuracy
    acc = accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # Prepare dictionary
    metrics_dict = {
        "accuracy": acc,
        "precision": dict(zip(class_names, precision)),
        "recall": dict(zip(class_names, recall)),
        "f1": dict(zip(class_names, f1)),
        "confusion_matrix": cm_df,
    }

    return metrics_dict
