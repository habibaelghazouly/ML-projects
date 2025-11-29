from utils.information_gain import information_gain
from utils.plot_tree import plot_tree
from .Node import Node
import numpy as np


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        max_depth: maximum depth of the tree
        min_samples_split: minimum samples required to split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None
        best_left_mask = None
        best_right_mask = None

        for feature_index in range(n_features):
            values = X[:, feature_index]
            sorted_vals = np.unique(values)
            # Compute potential thresholds
            # These are midpoints between consecutive unique values
            # Example: [1, 2, 4] → thresholds = [1.5, 3]
            thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2

            for t in thresholds:
                left_mask = values <= t
                right_mask = values > t

                y_left = y[left_mask]
                y_right = y[right_mask]

                gain = information_gain(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = t
                    best_left_mask = left_mask
                    best_right_mask = right_mask

        return best_feature, best_threshold, best_gain, best_left_mask, best_right_mask

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape

        # Stopping conditions
        # 1. All labels same → leaf
        if len(np.unique(y)) == 1:
            return Node(value=y[0])

        # 2. Max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=self._majority_class(y))

        # 3. No enough samples to split
        if num_samples < self.min_samples_split:
            return Node(value=self._majority_class(y))

        # Find best split
        feature_index, threshold, gain, left_mask, right_mask = self._best_split(X, y)

        # No improvement → leaf
        if gain <= 0 or feature_index is None:
            return Node(value=self._majority_class(y))

        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=left_child,
            right=right_child,
            information_gain=gain,
        )

    def _majority_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def predict_one(self, x):
        node = self.root
        while not node.is_leaf():
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        return np.array([self.predict_one(sample) for sample in X])

    def plot(self, feature_names=None, class_names=None, filename="tree_output"):
        dot = plot_tree(self.root, feature_names=feature_names, class_names=class_names)
        dot.render(filename, format="png", view=False)  # saves and opens PNG file
