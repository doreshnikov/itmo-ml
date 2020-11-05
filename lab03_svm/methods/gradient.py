import numpy as np
from lab03_svm.utils.kernels import apply_kernel


def gradient(xd, y_square, alpha):
    return alpha * np.multiply(y_square, xd) - 1
    # return alpha * np.multiply(y_square, x * x.T) - 1


def cut(alpha, C):
    alpha = np.minimum(alpha, C)
    alpha = np.maximum(alpha, 0)
    return alpha


def solve(x, y, C, kernel, iterations, lmbd=lambda _: 1e-5):
    alpha = np.mat(np.zeros(x.shape[0], dtype=float))
    y_m = np.mat(y)
    x_m = np.mat(x)
    y_square = y_m.T.dot(y_m)
    xd = apply_kernel(x, x, kernel)

    for i in range(iterations):
        grad = gradient(xd, y_square, alpha)
        alpha -= lmbd(i) * grad
        reg = alpha * y_m.T / (y_m * y_m.T).sum()
        alpha -= reg * y_m
        cut(alpha, C)

    # w = np.mat(np.zeros((1, x_m.shape[1])))
    # for j, row in enumerate(x_m):
    #     w += alpha.item(j) * y_m.item(j) * row
    w0 = (np.multiply(alpha, y_m) * xd - y_m).mean()
    return alpha, w0
