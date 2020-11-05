import numpy as np


def apply_kernel(x, y, kernel):
    xd = np.zeros((x.shape[0], y.shape[0]))
    for i, row1 in enumerate(x):
        for j, row2 in enumerate(y):
            xd[i][j] = kernel(row1, row2)
    return np.mat(xd)


class Kernel:
    @staticmethod
    def linear():
        def f(x, y):
            return np.sum(x * y)

        return f

    @staticmethod
    def polynomial(d):
        def f(x, y):
            return (1 + np.sum(x * y)) ** d

        return f

    @staticmethod
    def gauss(beta):
        def f(x, y):
            diff = x - y
            return np.power(np.e, -beta * (np.sum(diff * diff)))

        return f
