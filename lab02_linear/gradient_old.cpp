#include <vector>
#include <ctime>
#include <cstdio>
#include <iostream>

using namespace std;

int main() {

    freopen("lab02_linear/resources/LR-CF/" FNAME, "r", stdin);
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    int m, n;
    cin >> m >> n;
//    cin >> n >> m;
    vector<vector<double>> xx(n, vector<double>(m + 1));

    for (auto &item : xx) {
        for (auto &val : item) {
            int tmp;
            cin >> tmp;
            val = tmp;
        }
    }

    cout << (double) clock() / CLOCKS_PER_SEC;

    if (n == 2) {
        cout << "";
    }

}