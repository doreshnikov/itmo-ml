import math
import numpy as np
import sys
from copy import copy
from random import uniform

from lab02_linear.utils.metrics import smape, prediction


def linear_force_downhill(max_temp):
    def flag(old_score, new_score, temp):
        return uniform(0, 1) < math.pow(math.e, (old_score - new_score) * max_temp / temp)

    return flag


def random_vector(n, l, r):
    return [uniform(l, r) for _ in range(n)]


def simulate_annealing(
        # blackbox,
        whitebox,
        data,
        time,
        temperature=None,
        mutation=lambda x, temp: uniform(-temp, temp),
        force_downhill=None,
        sequential=False
):
    m = len(data)
    x, y = whitebox[0], whitebox[1]
    if temperature is None:
        temperature = lambda t: time - t
    if force_downhill is None:
        force_downhill = linear_force_downhill(temperature(1))

    # score = blackbox(data)
    y_pred = prediction(data, x)
    score = smape(y_pred, y)

    for t in range(1, time + 1):
        if sequential:
            yield data, score

        temp = temperature(time)
        for i in range(m):
            if uniform(0, m) > 1:
                continue

            # values = copy(data)
            # values[i] = mutation(values[i], temp)
            # new_score = blackbox(values)
            new_y_pred = copy(y_pred)
            delta = mutation(data[i], temp)
            for j in range(len(x)):
                new_y_pred[j] += delta * x[j][i]
            new_score = smape(new_y_pred, y)

            if new_score < score or force_downhill(score, new_score, temp):
                # data[i] = values[i]
                data[i] += delta
                y_pred = new_y_pred
                score = new_score

    yield data, score


def solve(iterations, x, y, sequential=False):
    n, m = len(x), len(x[0])
    return simulate_annealing(
        whitebox=(x, y),
        data=np.zeros(m),
        time=iterations,
        temperature=lambda t: iterations / t * max(1., math.log(iterations - t + 1)),
        sequential=sequential
    )
