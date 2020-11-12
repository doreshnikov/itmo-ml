#include <iostream>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <cmath>

using namespace std;

struct message {
    unsigned int c = 0;
    unordered_set<string> words;

    message() = default;
};

istream &operator>>(istream &in, message &m) {
    unsigned int t;
    in >> t;
    string s;
    for (unsigned int i = 0; i < t; i++) {
        in >> s;
        m.words.insert(s);
    }
    return in;
}

long double prob_x(unsigned int wc, unsigned int cc, unsigned int alpha) {
    return static_cast<long double>(wc + alpha) / (cc + alpha * 2);
}

int main() {

    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    unsigned int k, alpha;
    cin >> k;
    vector<unsigned int> lambda(k);
    for (unsigned int i = 0; i < k; i++) {
        cin >> lambda[i];
    }
    cin >> alpha;

    vector<unsigned int> class_count(k);
    vector<unordered_map<string, unsigned int>> word_count(k);
    unordered_set<string> all_words;

    unsigned int n;
    cin >> n;
    vector<message> train(n);
    for (unsigned int i = 0; i < n; i++) {
        cin >> train[i].c >> train[i];
        class_count[--train[i].c]++;
        for (string const &word: train[i].words) {
            word_count[train[i].c][word]++;
            all_words.insert(word);
        }
    }

    unsigned int m;
    cin >> m;
    vector<message> test(m);
    cout.precision(20);
    for (unsigned int i = 0; i < m; i++) {
        cin >> test[i];
        vector<long double> probs(k);
        long double denominator = 0;

        for (unsigned int c = 0; c < k; c++) {
            probs[c] = static_cast<long double>(lambda[c]) * class_count[c] / n;
            for (string const &word : all_words) {
                long double prob = prob_x(word_count[c][word], class_count[c], alpha);
                if (test[i].words.count(word) == 0) {
                    prob = 1 - prob;
                }
                probs[c] *= prob;
            }
            denominator += probs[c];
        }

        for (unsigned int c = 0; c < k; c++) {
            cout << fixed << static_cast<double>(probs[c] / denominator) << ' ';
        }
        cout << '\n';
    }

}