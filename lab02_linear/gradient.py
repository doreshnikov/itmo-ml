import math

from lab02_linear.common.graders import prediction, smape


def eqsign(a, b):
    if a == 0:
        return 1
    return 1 if a == 0 or a > 0 and b >= 0 else -1


def smape_gradient(coef, x, y_pred, y, labels=None):
    n, m = len(x), len(coef)
    if labels is None:
        labels = range(m)
    grad = dict()
    for idx in labels:
        grad[idx] = 0
        for i in range(n):
            diff = y_pred[i] - y[i]
            asum = abs(y[i]) + abs(y_pred[i])
            if asum != 0:
                multiplier = asum * eqsign(diff, x[i][idx]) - abs(diff) * eqsign(y_pred[i], x[i][idx])
                grad[idx] += multiplier * x[i][idx] / math.pow(asum, 2)
    return grad


if __name__ == '__main__':
    n, m = map(int, input().split())
    x, y = [], []
    for i in range(n):
        obj = list(map(int, input().split()))
        y.append(obj.pop())
        obj.append(1)
        x.append(obj)

    coef = [0 for _ in range(m + 1)]
    y_pred = prediction(coef, x)

    for i in range(10):
        grad = smape_gradient(coef, x, y_pred, y)
        for j in range(m + 1):
            coef[j] -= grad[j] / n
        y_pred = prediction(coef, x)

    print(*coef)