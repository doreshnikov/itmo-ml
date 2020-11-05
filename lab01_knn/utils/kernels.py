from math import *
import numpy as np


class Distance:
    @staticmethod
    def minkowski(row1, row2, p=3):
        return np.linalg.norm(row1 - row2, ord=p)

    @staticmethod
    def manhattan(row1, row2):
        """same as minkowski(..., p=1)"""
        return np.linalg.norm(row1 - row2, ord=1)

    @staticmethod
    def euclidean(row1, row2):
        """same as minkowski(..., p=2)"""
        return np.linalg.norm(row1 - row2, ord=2)

    @staticmethod
    def chebyshev(row1, row2):
        return np.max(np.abs(row1 - row2))

    @staticmethod
    def get(name):
        return getattr(Distance, name)


def limited_kernel(f):
    def call(x):
        return 0 if abs(x) >= 1 else f(x)

    return call


def infinite_kernel(f):
    def call(x):
        return 0 if x == float('+inf') else f(x)

    return call


class Kernel:
    @staticmethod
    @limited_kernel
    def uniform(x):
        return 1 / 2

    @staticmethod
    @limited_kernel
    def triangular(x):
        return 1 - abs(x)

    @staticmethod
    @limited_kernel
    def epanechnikov(x):
        return 3 / 4 * (1 - x * x)

    @staticmethod
    @limited_kernel
    def quartic(x):
        return 15 / 16 * pow(1 - x * x, 2)

    @staticmethod
    @limited_kernel
    def triweight(x):
        return 35 / 32 * pow(1 - x * x, 3)

    @staticmethod
    @limited_kernel
    def tricube(x):
        return 70 / 81 * pow(1 - pow(abs(x), 3), 3)

    @staticmethod
    @infinite_kernel
    def gaussian(x):
        return pow(e, -x * x / 2) / sqrt(2 * pi)

    @staticmethod
    @limited_kernel
    def cosine(x):
        return pi / 4 * cos(pi * x / 2)

    @staticmethod
    @infinite_kernel
    def logistic(x):
        return 1 / (2 + pow(e, x) + pow(e, -x))

    @staticmethod
    @infinite_kernel
    def sigmoid(x):
        return 2 / pi / (pow(e, x) + pow(e, -x))

    @staticmethod
    def get(name):
        return getattr(Kernel, name)
