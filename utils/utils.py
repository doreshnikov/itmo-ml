from math import log


def weighted_average(values, weights=None):
    if weights is None:
        return sum(values) / len(values)
    total = sum(weights)
    return smart_div(sum([values[i] * weights[i] for i in range(len(values))]), total)


def count_uniform(values):
    count = dict()
    for v in values:
        if v not in count:
            count[v] = 0
        count[v] += 1
    return count


def argmax(values):
    return max(values.items(), key=lambda item: item[1])[0]


def input_ints():
    return list(map(int, input().split()))


def smart_div(a, b):
    return a / b if b != 0 else 0


def smart_log(x):
    return log(x) if x > 0 else float('-inf')
