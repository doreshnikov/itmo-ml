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
const double TIME_LIMIT = 0.97;

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
    scales.resize(m + 1);
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < m; j++) {
            value v = std::abs(x[i][j]);
            if (v > scales[j]) {
                scales[j] = v;
            }
        }
        value v = std::abs(y[i]);
        if (v > scales[m]) {
            scales[m] = v;
        }
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

inline int relative_sign(value t, value against) {
    if (against == 0) return 1;
    if (t == 0) return 0;
    return against > 0 && t > 0 ? 1 : -1;
}

row smape_gradient(row const &pred, bool normalized = false) {
    row gradient(m);
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < m; j++) {
            value diff = pred[i] - y[i];
            value scale = std::abs(pred[i]) + std::abs(y[i]);
            if (scale != 0) {
                value comp = relative_sign(x[i][j], diff) - std::abs(diff) * relative_sign(x[i][j], pred[i]) / scale;
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
    return 0.000000001 * std::log2(m) / std::log2(time + 2);
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
    scanf("%d %d\n", &m, &n);
#else
    scanf("%d %d\n", &n, &m);
#endif

    x.assign(n, row(m + 1));
    y.resize(n);
    std::vector<std::vector<double>> xx(n, std::vector<double>(m + 1));

    printf("%d\n", clock());
    int tmp;
    for (auto &item : xx) {
        for (auto &val : item) {
            std::cin >> tmp;
            val = tmp;
        }
    }
    printf("%d\n", clock());
    for (unsigned int i = 0; i < n; i++) {
        y[i] = x[i][m];
        x[i][m] = 1;
    }
    printf("%d\n", clock());
    m++;
    absmax_scales();
    normalize();

    alpha.resize(m);
    fill_random_inplace(alpha.begin(), alpha.end(), -1., 1.);
    row pred = predict();
    clock_t base = clock(), measure = 0;
    unsigned int step = 0;

#ifndef UNTIMED
    while (clock() + measure < TIME_LIMIT * CLOCKS_PER_SEC) {
#else
    while (step < STEPS) {
#endif
        row gradient = smape_gradient(pred, true);
        value l = lambda(step);
        bool changed = false;
        for (unsigned int i = 0; i < m; i++) {
            if (gradient[i] != 0) {
                changed = true;
                alpha[i] -= l * gradient[i];
            }
        }

        if (!changed) {
            fill_random_inplace(alpha.begin(), alpha.end(), -1, 1);
        }
        predict_inplace(pred);
        step++;
        measure = (clock() - base) / step;
    }

    for (unsigned int i = 0; i < m; i++) {
        value coef = alpha[i] * scales[m] / scales[i];
        printf("%.9f ", coef);
    }
#ifdef LOCAL
    printf("\n%u 0-%ld-%ld\n", step, base, clock());
    printf("%.9f\n", smape(predict()));
#endif


}