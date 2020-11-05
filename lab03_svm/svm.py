import numpy as np
from utils.metrics import accuracy
from lab03_svm.utils.kernels import apply_kernel


def predict(x, y, alpha, w0, kernel, xnew):
    xs = apply_kernel(x, xnew, kernel)
    return np.sign(alpha * np.multiply(np.mat(y).T, xs) - w0)


def score(x, y, alpha, w0, kernel, xnew, ynew):
    # return accuracy(y, np.sign(w.dot(x.T) - w0))
    return accuracy(ynew, predict(x, y, alpha, w0, kernel, xnew))


def score_folds(x, y, method, kernel, C, folds=5, **kwargs):
    x_parts, y_parts = np.array_split(x, folds), np.array_split(y, folds)

    scores = []
    for i in range(folds):
        xx, yy = np.vstack(x_parts[:i] + x_parts[i + 1:]), np.hstack(y_parts[:i] + y_parts[i + 1:])
        alpha, w0 = method(xx, yy, C, kernel, **kwargs)
        scores.append(score(xx, yy, alpha, w0, kernel, x_parts[i], y_parts[i]))

    return np.min(scores), np.mean(scores)


def solve_svm(data, method, Cs, kernels, **kwargs):
    y = data['class'].to_numpy()
    x = data.drop(['class'], axis=1).to_numpy()

    scores = []
    count, i = len(Cs) * len(kernels), 0
    for C in Cs:
        for name, kernel in kernels.items():
            scores.append((C, name, score_folds(x, y, method, kernel, C, **kwargs)))
            i += 1
            if i % 5 == 0:
                print(f'-- finished {i} of {count}')
    return sorted(scores, key=lambda t: t[2][1], reverse=True)
