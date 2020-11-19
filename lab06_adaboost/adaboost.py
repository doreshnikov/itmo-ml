import numpy as np
from math import log, exp
from copy import deepcopy

from utils.metrics import accuracy
from lab06_adaboost.utils.data import bootstrap

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


class AdaBoostClassifier:
    def __init__(self, n_trees, tree_params):
        self.n_trees = n_trees
        self.tree_params = tree_params
        self.trees = []

    def fit_one(self, x, y, weights):
        return GridSearchCV(
            DecisionTreeClassifier(),
            self.tree_params,
            n_jobs=-1
        ).fit(x, y, sample_weight=weights)

    def fit_one2(self, x, y, weights):
        return GridSearchCV(
            DecisionTreeClassifier(),
            self.tree_params,
            n_jobs=-1
        ).fit(x, y, sample_weight=weights)

    def predict(self, x, limit=None):
        iteration = 0
        n = x.shape[0]
        class_scores = np.zeros((n, 2))
        for tree, score in self.trees:
            y_pred = tree.predict(x)
            for i in range(n):
                class_scores[i][y_pred[i]] += score
            iteration += 1
            if limit is not None and iteration == limit:
                break
        return (class_scores[:, 0] < class_scores[:, 1]).astype(int)

    def fit(self, x, y, score_steps=True, do_bootstrap=False):
        self.trees = []
        n, m = x.shape
        X, Y = deepcopy(x), deepcopy(y)
        scores = []
        weights = np.full(n, 1 / n)

        for _ in range(self.n_trees):
            stump = self.fit_one(X, Y, weights)
            y_pred = stump.predict(X)
            error_weight = max((weights * (y_pred != Y)).sum(), 1e-6)

            score = 1 / 2 * log((1 - error_weight) / error_weight)
            self.trees.append((stump, score))
            weights *= np.where(y_pred != Y, exp(score), exp(-score))
            weights /= weights.sum()

            if do_bootstrap:
                X, Y = bootstrap(X, Y, weights)
                weights = np.full(n, 1 / n)
            if score_steps:
                scores.append(accuracy(y, self.predict(x)))

        return np.array(scores)
