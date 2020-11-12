import numpy as np
from math import log

from utils.utils import smart_log, smart_div


class BayesianClassifier:
    def __init__(self, k, alpha, lambdas):
        self.k = k
        self.alpha = alpha
        self.lambdas = lambdas
        self.all_words = set()
        self.word_count = [dict() for _ in range(k)]
        self.class_count = [0 for _ in range(k)]

    def fit(self, data):
        for c in range(self.k):
            self.class_count[c] += len(data[c])
            for words in data[c]:
                for word in words:
                    self.all_words.add(word)
                    if word in self.word_count[c].keys():
                        self.word_count[c][word] += 1
                    else:
                        self.word_count[c][word] = 1
        return self

    def log_prob_x(self, c, word, include):
        numerator = self.word_count[c].get(word, 0) + self.alpha
        denominator = self.class_count[c] + 2 * self.alpha
        if not include:
            numerator = denominator - numerator
        return smart_log(numerator) - smart_log(denominator)

    def all_log_prob_x(self, word, include):
        return np.array([self.log_prob_x(c, word, include) for c in range(self.k)])

    @staticmethod
    def normalize_logs(result):
        tmp = np.exp(result - np.mean(result))
        total = np.sum(tmp)
        return np.vectorize(smart_div)(tmp, total)

    def predict(self, data, verbose=False, compute_probs=True):
        if verbose:
            print('Total values:', len(data))
        result = np.log(self.lambdas) + np.log(self.class_count) - log(sum(self.class_count))
        result = np.tile(result, (len(data), 1))
        for i, words in enumerate(data):
            for word in self.all_words:
                log_probs = self.all_log_prob_x(word, word in words)
                result[i] += log_probs
            if compute_probs:
                result[i] = BayesianClassifier.normalize_logs(result[i])
            if verbose and (i == 0 or i % 10 == 9):
                print(f'- {i + 1} processed')
        return result


if __name__ == '__main__':
    data = [
        [{'ant', 'emu'}, {'ant', 'dog', 'bird'}],
        [{'dog', 'fish', 'dog'}],
        [{'bird', 'emu', 'ant'}]
    ]
    test_data = [
        {'emu'},
        {'emu', 'dog', 'fish'},
        {'fish', 'emu', 'ant', 'cat'},
        {'emu', 'cat'}, {'cat'}
    ]
    cls = BayesianClassifier(3, 1, [1, 1, 1])
    cls.fit(data)
    print(cls.predict(test_data))