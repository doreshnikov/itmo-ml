import math


def mean(values):
    return sum(values) / len(values)


def smart_division(a, b):
    return a / b if b != 0 else (0 if a == 0 else float('+inf'))


def prediction(coef, x):
    n, m = len(x), len(coef)
    return [
        sum([coef[j] * x[i][j] for j in range(m)])
        for i in range(n)
    ]


def nrmse(y_pred, y):
    n = len(y)
    return smart_division(
        mean([math.pow(y_pred[i] - y[i], 2) for i in range(n)]),
        max(y) - min(y)
    )


def smape(y_pred, y, percent=False):
    n = len(y)
    score = mean(
        [smart_division(
            abs(y_pred[i] - y[i]),
            abs(y_pred[i]) + abs(y[i])
        ) for i in range(n)])
    return score if not percent else score * 100
