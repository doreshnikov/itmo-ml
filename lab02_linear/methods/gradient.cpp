#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <random>
#include <algorithm>
#include <fstream>
#include <ctime>

typedef double value;
typedef std::vector<value> row;

const value MAX_LOSS = 1.;
const unsigned int STEPS = 200;
#ifdef LOCAL
const double TIME_LIMIT = 1.47;
#else
const double TIME_LIMIT = 0.97;
#endif

unsigned int n, m;
row alpha, y;
std::vector<row> x;
row shifts, scales;

std::random_device device; // NOLINT(cert-err58-cpp)
std::mt19937_64 mersenne_engine{device()}; // NOLINT(cert-err58-cpp)

void fill_random_inplace(row::iterator const &begin, row::iterator const &end, value from, value to) {
    std::uniform_real_distribution<value> dist(from, to);
    for (auto it = begin; it != end; it++) {
        *it = dist(mersenne_engine);
    }
}

void predict_inplace(row &pred) {
    pred.assign(n, 0);
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < m; j++) {
            pred[i] += alpha[j] * x[i][j];
        }
    }
}

inline void absmax_scales() {
    shifts.resize(m + 1);
    scales.resize(m + 1);

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < m; j++) {
            value v = std::abs(x[i][j]);
            if (v > scales[j]) {
                scales[j] = v;
            }
            shifts[j] += x[i][j];
        }
        value v = std::abs(y[i]);
        if (v > scales[m]) {
            scales[m] = v;
        }
        shifts[m] += y[i];
    }

    for (unsigned int i = 0; i <= m; i++) {
        shifts[i] /= n;
    }
}

inline void normalize() {
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < m; j++) {
//            x[i][j] = (x[i][j] - shifts[j]) / scales[j];
            x[i][j] /= scales[j];
        }
//        y[i] = (y[i] - shifts[m]) / scales[m];
        y[i] /= scales[m];
    }
}

row predict() {
    row pred;
    predict_inplace(pred);
    return pred;
}

inline value divide(value a, value b) {
    return b == 0 ? a : a / b;
}

value nrmse(row const &pred) {
    value total_loss = 0;
    value maxy = std::numeric_limits<int>::min(), miny = std::numeric_limits<int>::max();
    for (unsigned int i = 0; i < n; i++) {
        value diff = pred[i] - y[i];
        total_loss += diff * diff;
        if (y[i] > maxy) {
            maxy = y[i];
        }
        if (y[i] < miny) {
            miny = y[i];
        }
    }
    return divide(total_loss / n, maxy - miny);
}

value smape(row const &pred) {
    value total_loss = 0;
    for (unsigned int i = 0; i < n; i++) {
        total_loss += divide(
                std::abs(pred[i] - y[i]),
                std::abs(pred[i]) + std::abs(y[i])
        );
    }
    return total_loss / n;
}

inline int sign(value t) {
    return t >= 0 ? 1 : -1;
}

row smape_gradient(row const &pred, bool normalized = false) {
    row gradient(m);
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < m; j++) {
            value diff = pred[i] - y[i];
            value scale = std::abs(pred[i]) + std::abs(y[i]);
            if (scale != 0) {
                value comp = sign(diff) - std::abs(diff) * sign(pred[i]) / scale;
                gradient[j] += comp * x[i][j] / scale;
            }
        }
    }
    if (normalized) {
        for (unsigned int i = 0; i < m; i++) {
            gradient[i] /= n;
        }
    }
    return gradient;
}

inline value lambda(unsigned int time) {
    if (m < 3) return 1. / std::log2(time + 2);
//    return 0.001 * std::log2(time + 2) / std::pow(time + 1, 0.1);
//    return 0.05 / std::log2(time + 2);
//    return std::log(time + 2) / std::pow(time + 1, 0.4);
    return std::max(1., 100. - time) * .0001 / std::pow(time + 1, 0.01);
//    return 0.01 / std::pow(time + 1, 0.02);
//    return .175 / std::pow(time + 1, 0.1);
//    return 0.001;
}

int main() {

#ifdef FNAME
    std::freopen("lab02_linear/resources/LR-CF/" FNAME, "r", stdin);
//    std::fstream::sync_with_stdio(false);
#endif
//    std::ios_base::sync_with_stdio(false);
//    std::cin.tie(nullptr);
//    std::cout.tie(nullptr);
#ifdef FNAME
    scanf("%d\n%d\n", &m, &n);
#else
    scanf("%d %d\n", &n, &m);
#endif

    x.assign(n, row(m));
    y.reserve(n);

    int tmp;
    for (auto &item : x) {
        for (auto &val : item) {
            scanf("%d", &tmp);
            val = tmp;
        }
        scanf("%d", &tmp);
        y.push_back(tmp);
        item.push_back(1);
    }
    m++;
    absmax_scales();
    value max_shift = 0;
    for (unsigned int i = 0; i < m; i++) {
        value shift = std::abs(shifts[i]);
        if (shift > max_shift) {
            max_shift = shift;
        }
    }
    normalize();

    alpha.resize(m);
    for (unsigned int i = 0; i < m; i++) {
        alpha[i] = divide(shifts[m] * scales[i], scales[m] * shifts[i]) / m / 10;
    }
    row pred = predict();

    clock_t base = clock(), measure = 0;
    unsigned int step = 0;
    value min_score = MAX_LOSS;
    row min_res(m);
#ifndef UNTIMED
    while (clock() + measure < TIME_LIMIT * CLOCKS_PER_SEC) {
#else
    while (step < STEPS) {
#endif
        row gradient = smape_gradient(pred, true);
        value l = lambda(step);
        bool changed = false;

        row opt_res(m);
        value scale = 0.025, opt_score = MAX_LOSS;
        for (int retry = -1; retry < 5; retry++) {
            for (unsigned int i = 0; i < m; i++) {
                if (gradient[i] != 0) {
                    changed = true;
                    alpha[i] -= scale * l * gradient[i];
                }
            }
            scale *= 4;

            if (!changed) {
                for (unsigned int i = 0; i < m; i++) {
                    alpha[i] = divide(shifts[m] * scales[i], scales[m] * shifts[i]) / m / 10;
                }
                break;
            }

            predict_inplace(pred);
            value score = smape(pred);
            if (score < opt_score) {
                opt_score = score;
                for (unsigned int i = 0; i < m; i++) {
                    opt_res[i] = alpha[i];
                }
            }
        }

        step++;
        measure = (clock() - base) / step;
        if (opt_score < min_score) {
            min_score = opt_score;
            for (unsigned int i = 0; i < m; i++) {
                min_res[i] = opt_res[i];
            }
        }
    }

#ifndef LOCAL
    for (unsigned int i = 0; i < m; i++) {
        value coef = min_res[i] * scales[m] / scales[i];
//        value coef = alpha[i];
        printf("%.9f\n", coef);
    }
#else
    alpha = min_res;
    printf("%u 0-%ld-%ld\n", step, base, clock());
    printf("%.9f / %0.9f\n", smape(predict()), min_score);
#endif

}