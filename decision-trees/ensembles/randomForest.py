from tree.DecisionTree import DecisionTree
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_trees=5, max_depth=5 , min_samples_split=2 , max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.features_subsets = []

    def make_tree(self):
        return DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)

    def bootstrap_sample(self, x, y):
        n_samples = x.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True) # replacement
        return x[indices], y[indices]
    
    def fit(self, x, y):
        self.trees = []
        self.features_subsets = []

        n_feats = x.shape[1]

        for _ in range(self.n_trees):

            # sampling features (affects rows/samples)
            x_sample, y_sample = self.bootstrap_sample(x, y)

            # feature subset (affects columns/features)
            if self.max_features is None:
                features_indices = np.arange(n_feats)
            else:
                features_indices = np.random.choice(n_feats, self.max_features, replace=False)
            
            self.features_subsets.append(features_indices)

            tree = self.make_tree()
            tree.fit(x_sample[:, features_indices], y_sample)
            self.trees.append(tree)
            
    def predict(self, x):
        tree_preds =[]

        for tree , indx in zip(self.trees , self.features_subsets):
            preds = tree.predict(x[:, indx])
            tree_preds.append(preds)    

        tree_preds = np.array(tree_preds)  # shape: (n_trees, n_samples)
       
       # majority voting
        majority_preds = []
        for sample_preds in tree_preds.T:  # iterate over samples
            vals, counts = np.unique(sample_preds, return_counts=True)
            majority_pred = vals[np.argmax(counts)]
            majority_preds.append(majority_pred)
            
        return np.array(majority_preds)
    