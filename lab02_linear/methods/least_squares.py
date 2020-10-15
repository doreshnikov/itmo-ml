import math
import numpy as np

from lab02_linear.common.graders import prediction, smape
from lab02_linear.common.data import normalize


def solve(
        x, y,
        do_normalize=False, ridge=0
):
    if do_normalize:
        xx, yy, scalex, scaley = normalize(x, y)
    else:
        xx, yy = x, y
        scalex, scaley = 1, 1

    xxT = xx.T
    xplus = np.linalg.inv(xxT.dot(xx) + ridge * np.eye(xx[0].size)).dot(xxT)

    alpha = xplus.dot(y) / scalex * scaley
    y_pred = prediction(alpha, x)
    err = math.pow(np.linalg.norm(y_pred - y), 2)
    return alpha, smape(y_pred, y), err


def solve_svd(
        x, y,
        do_normalize=False, ridge=0
):
    if do_normalize:
        xx, yy, scalex, scaley = normalize(x, y)
    else:
        xx, yy = x, y
        scalex, scaley = 1, 1

    r = np.linalg.matrix_rank(xx)
    u, sigma, vt = np.linalg.svd(xx, full_matrices=False)

    diag = np.zeros(xx.shape[1])
    diag[:r] = 1 / sigma[:r]
    dplus = np.diag(diag + ridge)
    xplus = vt.T.dot(dplus).dot(u.T)

    alpha = xplus.dot(yy) / scalex * scaley
    y_pred = prediction(alpha, x)
    err = math.pow(np.linalg.norm(y_pred - y), 2)
    return alpha, smape(y_pred, y), err
