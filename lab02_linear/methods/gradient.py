import math
import numpy as np

from typing import Union, Generator, Any
from lab02_linear.common.graders import smart_div, prediction, smape
from lab02_linear.common.data import normalize


def sign(a):
    return 1 if a >= 0 else -1


__sign = np.vectorize(sign)
__smart_div = np.vectorize(smart_div)


def smape_gradient(x, y_pred, y):
    diff = y_pred - y
    scale = np.abs(y) + np.abs(y_pred)
    multiplier = __sign(diff) - __smart_div(np.abs(diff) * __sign(y_pred), scale)
    return __smart_div(multiplier, scale).dot(x) / y.size


def solve(
        iterations, x, y,
        do_normalize=False, fill='zero', lmbd=None, reg=0,
        sequential=False
):
    __fills = {
        'smart': lambda: __smart_div(np.mean(x, axis=0), np.max(np.abs(x), axis=0)) * \
                         smart_div(np.max(np.abs(y)), np.mean(y)),
        'zero': lambda: np.zeros(x[0].size),
        'uniform': lambda: np.random.uniform(-0.001, 0.001, x[0].size)
    }
    alpha = __fills[fill]()

    if do_normalize:
        xx, yy, scalex, scaley = normalize(x, y)
    else:
        xx, yy = x, y
        scalex, scaley = 1, 1
    yy_pred = prediction(alpha, xx)

    if lmbd is None:
        def lmbd(step):
            return 0.8 / math.pow(step + 1, 0.3)

    for i in range(iterations):
        if sequential:
            yield alpha / scalex * scaley, smape(yy_pred, yy)
        gradient = smape_gradient(xx, yy_pred, yy)

        l = lmbd(i)
        alpha *= 1 - reg * l
        alpha -= gradient * l
        yy_pred = prediction(alpha, xx)

    yield alpha / scalex * scaley, smape(yy_pred, yy)


def memoized_solve(*args, **kwargs):
    best_score = 1
    best_alpha = None

    for alpha, score in solve(*args, **kwargs):
        if score < best_score:
            best_score = score
            best_alpha = alpha

    return best_alpha, best_score
