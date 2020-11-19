import numpy as np


def split_data(data_set):
    return data_set.drop(columns=['class']).to_numpy(), data_set['class'].to_numpy()


def bootstrap(x, y, weights):
    n = x.shape[0]
    indices = np.random.choice(n, size=n, replace=True, p=weights)
    return x[indices], y[indices]
