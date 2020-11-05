import numpy as np


def gradient(x, y_square, alpha):
    return alpha * y_square * x * x.T - 1


def cut(alpha, C):
    alpha = np.min(alpha, C)
    alpha = np.max(alpha, 0)
    return alpha


def solve(iterations, x, y, C, lmbd=lambda _: 1e-5):
    alpha = np.zeros(x.shape[0])
    y_m = np.mat(y)
    y_square = y_m.T.dot(y_m)

    for i in range(iterations):
        grad = gradient(x, y_square, alpha)
        alpha -= lmbd(i) * grad
        reg = (alpha * y_m.T) / y.dot(y)
        alpha -= reg * y
        cut(alpha, C)

    w = (alpha * y).reshape((x.shape[0], 1)) * x
    w = w.sum(axis=0)
    w0 = np.mean(w.dot(x.T) - y)
    return w, w0
