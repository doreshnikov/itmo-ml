import numpy as np
import pandas as pd
from utils.utils import input_ints


def k_splits(X, y, splits=10, index=False):
    indices = [[] for _ in range(splits)]
    for i in range(X.shape[0]):
        indices[i % splits].append(i)

    return indices if index else [(X[indices[i]].flatten(), y[indices[i]]) for i in range(splits)]


def stratified_k_splits(X, y, splits=10, index=False):
    order = sorted(range(X.shape[0]), key=lambda i: y[i])
    indices = [[] for _ in range(splits)]
    for i in range(X.shape[0]):
        indices[i % splits].append(order[i])

    return indices if index else [(X[indices[i]].flatten(), y[indices[i]]) for i in range(splits)]


def normalize(X, columns=None):
    x = X.copy(deep=True)
    if columns is None:
        columns = x.columns
    for c in columns:
        std = np.std(x.loc[:, c])
        if std != 0:
            x.at[:, c] = (x.loc[:, c] - np.mean(x.loc[:, c])) / std
    return x


def one_hot_encode(X, columns=None):
    x = X.copy(deep=True)
    if columns is None:
        columns = x.columns
    for c in columns:
        dummies = pd.get_dummies(x[c])
        dummies.columns = list(map(lambda v: f'{c}@{v}', dummies.columns))
        x = x.join(dummies).drop(c, axis=1)
    return x


def k_fold_validation(X, y, model, grader, folds=10, stratified=True):
    splits = (k_splits, stratified_k_splits)[stratified](X, y, splits=folds, index=True)
    scores = []
    for i, test_indices in enumerate(splits):
        train_indices = np.hstack(splits[:i] + splits[i + 1:])
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        y_pred = model.clone_unfit().fit(X_train, y_train).predict(X_test)
        scores.append(grader(y_test, y_pred))
    return scores


def one_out_validation(X, y, model, grader):
    predictions = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        X_train, y_train = np.vstack((X[:i], X[i + 1:])), np.hstack((y[:i], y[i + 1:]))

        y_pred = model.clone_unfit().fit(X_train, y_train).predict(np.asarray([X[i]]))
        predictions[i] = round(y_pred[0])
    return grader(y, predictions)


if __name__ == '__main__':
    n, m, k = input_ints()
    data = np.arange(1, n + 1).reshape((n, 1))
    classes = np.asarray(input_ints())

    splits = stratified_k_splits(data, classes, splits=k)
    for group in splits:
        print(len(group[0]), *sorted(group[0]))
