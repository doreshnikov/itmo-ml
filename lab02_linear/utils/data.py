import sys
import numpy as np


def read(file):
    n = int(file.readline().strip())
    x, y = [], []
    for i in range(n):
        point = list(map(int, file.readline().strip().split()))
        y.append(point.pop())
        point.append(1)
        x.append(point)
    return n, np.asarray(x, dtype=np.float), np.asarray(y, dtype=np.float)


def normalize(x, y):
    scalex = np.max(np.abs(x), axis=0)
    scaley = np.max(np.abs(y))
    return x / scalex, y / scaley, scalex, scaley
