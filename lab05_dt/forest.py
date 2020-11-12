import numpy as np
from math import sqrt

from sklearn.tree import DecisionTreeClassifier


class Forest:
    def __init__(self, n_trees, **kwargs):
        self.n_trees = n_trees
        self.trees = []
        self.subfeatures = []
        self.args = kwargs

    def fit(self, x, y):
        self.trees = []
        n = x.shape[0]
        m = x.shape[1]
        fs = int(sqrt(m) + 1)

        for i in range(self.n_trees):
            items = np.random.randint(0, n, n)
            features = np.random.choice(m, fs, replace=False)
            X, Y = x[items, :][:, features], y[items]
            self.subfeatures.append(features)
            self.trees.append(DecisionTreeClassifier(**self.args))
            self.trees[-1].fit(X, Y)

    def predict(self, x):
        results = np.vstack([
            tree.predict(x[:, self.subfeatures[i]]) for i, tree in enumerate(self.trees)
        ])
        return np.apply_along_axis(lambda r: np.argmax(np.bincount(r)), 0, results)
