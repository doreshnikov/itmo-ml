import numpy as np
from lab01_knn.utils.graders import Distance, Kernel
from utils.utils import weighted_average, count_uniform, argmax, input_ints, smart_div


class KNNClassifier:
    def __init__(self, k=1, distance='euclidean'):
        self.k = k
        self.distance = Distance.get(distance) if isinstance(distance, str) else distance
        self._Xy = None

    def clone_unfit(self):
        return KNNClassifier(self.k, self.distance)

    def fit(self, X, y):
        assert len(X) == len(y)
        self._Xy = np.asarray([(X[i], y[i]) for i in range(len(X))])
        return self

    def __len__(self):
        return len(self._Xy) if self._Xy is not None else None

    def _sorted(self, row):
        return sorted(self._Xy, key=lambda row2: self.distance(row, row2[0]))

    def predict(self, X):
        assert self._Xy is not None
        y = []
        for row in X:
            neighbors = self._sorted(row)
            count = dict()
            for i in range(self.k):
                cls = neighbors[i][1]
                if cls not in count:
                    count[cls] = 0
                count[cls] += 1
            y.append(argmax(count))
        return y


class KNNRegressionClassifier(KNNClassifier):
    def __init__(self, k=1, distance='euclidean', kernel='uniform', window='variable', h=None):
        super().__init__(k, distance)
        self.kernel = Kernel.get(kernel) if isinstance(kernel, str) else kernel
        assert window == 'variable' or window == 'fixed'
        self.window = window
        self.h = h

    def clone_unfit(self):
        return KNNRegressionClassifier(self.k, self.distance, self.kernel, self.window, self.h)

    def predict(self, X, policy='argmax'):
        assert self._Xy is not None
        assert policy == 'mean' or policy == 'argmax'
        y = []
        for row in X:
            weight = dict()
            neighbors = self._Xy if self.window == 'fixed' else self._sorted(row)
            h = self.h if self.window == 'fixed' else \
                self.distance(row, neighbors[self.k][0]) if self.k < len(self) else float('+inf')

            if h == 0:
                coincident = list(filter(lambda row2: self.distance(row, row2[0]) == 0, neighbors))
                if len(coincident) > 0:
                    neighbors = coincident
                weight = count_uniform([neighbors[i][1] for i in range(len(neighbors))])
            else:
                for row2 in neighbors:
                    cls = row2[1]
                    if cls not in weight.keys():
                        weight[cls] = 0
                    weight[cls] += self.kernel(self.distance(row, row2[0]) / h)
                if sum(weight.values()) == 0:
                    weight = count_uniform([neighbors[i][1] for i in range(len(self))])

            if policy == 'mean':
                y.append(weighted_average(list(weight.keys()), weights=list(weight.values())))
            else:
                y.append(argmax(weight))
        return y


if __name__ == '__main__':
    n, m = input_ints()
    Xy = [input_ints() for _ in range(n)]
    X, y = [Xy[i][:-1] for i in range(n)], [Xy[i][-1] for i in range(n)]

    q = input_ints()
    distance, kernel, window = input(), input(), input()
    hk = int(input())

    knn = KNNRegressionClassifier(hk, distance, kernel, window, hk)
    print('%.9f' % knn.fit(X, y).predict([q], policy='mean')[0])
