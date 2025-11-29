from utils.metrics import compute_metrics
from tree.DecisionTree import DecisionTree
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def hyperparameter_tuning(
    X_train, y_train, X_val, y_val,
    max_depth_values, min_samples_split_values
):
    results = []
    models = []

    for max_depth in max_depth_values:
        for min_samples_split in min_samples_split_values:

            tree = DecisionTree(
                max_depth=max_depth,
                min_samples_split=min_samples_split
            )
            tree.fit(X_train, y_train)

            train_preds = tree.predict(X_train)
            val_preds = tree.predict(X_val)

            metrics_train = compute_metrics(y_train, train_preds)
            metrics_val = compute_metrics(y_val, val_preds)

            results.append({
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "train_acc": metrics_train["accuracy"],
                "val_acc": metrics_val["accuracy"],
            })

            models.append({
                "tree": tree,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "train_metrics": metrics_train,
                "val_metrics": metrics_val,
            })

    return pd.DataFrame(results), models
