from utils.metrics import compute_metrics
from tree.DecisionTree import DecisionTree
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def evaluate_tree(tree, X_test, y_test, class_names=None):
    """
    Evaluates a trained tree on the test set and prints metrics with confusion matrix.

    Args:
        tree: Trained DecisionTree object
        X_test: Test features
        y_test: Test labels
        class_names: Optional list of class names for labeling

    Returns:
        metrics_dict: Dictionary containing accuracy, per-class metrics, and confusion matrix
    """
    # Predictions
    y_pred = tree.predict(X_test)

    # Compute metrics
    metrics_dict = compute_metrics(y_test, y_pred, class_names=class_names)

    # Print summary
    print("Test Set Performance\n")
    print(f"Overall Accuracy: {metrics_dict['accuracy']:.4f}\n")

    per_class_df = pd.DataFrame(
        {
            "Precision": metrics_dict["precision"],
            "Recall": metrics_dict["recall"],
            "F1-score": metrics_dict["f1"],
        }
    )
    print("Per-class Metrics:")
    print(per_class_df)

    # Confusion matrix
    cm = metrics_dict["confusion_matrix"]
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    return metrics_dict