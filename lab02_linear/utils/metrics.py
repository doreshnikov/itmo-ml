import math
import numpy as np


def smart_div(a, b):
    return a if b == 0 else a / b


__smart_div = np.vectorize(smart_div)


def prediction(alpha, x):
    return x.dot(alpha)


def nrmse(y_pred, y):
    mse = np.mean((y_pred - y) ** 2)
    return smart_div(math.sqrt(mse), np.ptp(y))


def smape(y_pred, y, percent=False):
    score = np.mean(__smart_div(
        np.abs(y_pred - y),
        np.abs(y_pred) + np.abs(y)
    ))
    return score if not percent else score * 100
