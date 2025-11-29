from utils.metrics import compute_metrics
from tree.DecisionTree import DecisionTree
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def compute_feature_importance(tree, feature_names=None):
    """
    Computes feature importance for a decision tree based on information gain.
    """
    importance = {}

    def traverse(node):
        if node.is_leaf():
            return
        # Accumulate gain for the feature
        importance[node.feature_index] = (
            importance.get(node.feature_index, 0) + node.information_gain
        )
        traverse(node.left)
        traverse(node.right)

    traverse(tree.root)

    # Convert to DataFrame
    if feature_names is not None:
        df = pd.DataFrame(
            {
                "Feature": [feature_names[i] for i in importance.keys()],
                "Importance": list(importance.values()),
            }
        ).sort_values("Importance", ascending=False)
    else:
        df = pd.DataFrame(
            {
                "Feature": list(importance.keys()),
                "Importance": list(importance.values()),
            }
        ).sort_values("Importance", ascending=False)

    print(df)

    return df
