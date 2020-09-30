import numpy as np
from utils.utils import weighted_mean, input_ints, smart_div


def __stats(cm):
    return np.sum(cm, axis=1), np.sum(cm, axis=0)


__smart_div = np.vectorize(smart_div, otypes=[float])


def __f1_score(prec, rec):
    return __smart_div(2 * prec * rec, prec + rec)


def micro_f1_score(cm):
    y_true, y_pred = __stats(cm)
    prec = __smart_div(cm.diagonal(), y_pred)
    rec = __smart_div(cm.diagonal(), y_true)
    return weighted_mean(__f1_score(prec, rec), weights=y_true)


def macro_f1_score(cm):
    y_true, y_pred = __stats(cm)
    m_prec = weighted_mean(__smart_div(cm.diagonal(), y_pred), weights=y_true)
    m_rec = np.sum(cm.diagonal()) / np.sum(y_true)
    return __f1_score(m_prec, m_rec)


def confusion_matrix(y_true, y_pred, classes=None):
    if classes is None:
        classes = max(y_true) + 1
    cm = np.zeros((classes, classes))
    for i in range(len(y_true)):
        cm[y_pred[i]][y_true[i]] += 1
    return cm


if __name__ == '__main__':
    k = int(input())
    cm = np.asarray([input_ints() for _ in range(k)])
    print('%.9f' % macro_f1_score(cm))
    print('%.9f' % micro_f1_score(cm))
