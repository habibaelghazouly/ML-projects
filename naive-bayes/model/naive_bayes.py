from collections import defaultdict
import numpy as np

class NaiveBayes:
    def __init__(self, alpha=1.0):
       
        # smoothing parameter
        self.alpha = alpha

        # prior and likelihood 
        self.class_priors = {}
        self.feature_likelihoods = {}

        self.feature_counts = {} # count each feature value appears per class
        self.feature_value_counts = {} # total counts of feature values per class
        self.classes = None
        self.class_count = defaultdict(int)

    def train(self , x , y):
        n_samples, n_features = x.shape
        self.classes, class_counts = np.unique(y, return_counts=True)
        k = len(self.classes)

        # Calculate class priors 
        for c , cnt in zip(self.classes, class_counts):
            self.class_priors[c] = (cnt + self.alpha) / (n_samples + k * self.alpha)
            self.class_count[c] = cnt

        # Calculate count of feature values per class
        for f in range(n_features):
            values = np.unique(x[:, f])
            self.feature_counts[f] = {}
            self.feature_value_counts[f] = len(values)

            # calculate counts per class for feature f value v => needed in the likelihood
            for c in self.classes:
                counts  = defaultdict(int)

                for v in values:
                    counts[v] = np.sum((x[:, f] == v) & (y == c))
                self.feature_counts[f][c] = counts


    def feature_likelihood(self,f, v , c):
          
        count_c = self.feature_counts[f][c].get(v, 0)

        likelihood = (count_c + self.alpha) / (self.class_count[c] + self.feature_value_counts[f] * self.alpha)
        return likelihood
        
    def predict_prob(self, x):
        n_samples, n_features = x.shape
        prob = np.zeros((n_samples, len(self.classes)))

        for i, row in enumerate(x):
            class_probs = []
            for c in self.classes:
                p = self.class_priors[c]

                for f in range(n_features):
                    p *= self.feature_likelihood(f, row[f], c)
            
                class_probs.append(p)
        
            prob[i, :] = class_probs

        return prob
        
    def predict(self, x):
            prob = self.predict_prob(x)
            indx = np.argmax(prob, axis=1)
            return self.classes[indx]

