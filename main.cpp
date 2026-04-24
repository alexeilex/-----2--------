
//g++ main.cpp -o prog -no-M_PIe
//7 вариант
//h=0.001
//D=[0, 1] x [0, 1]
//задаем eps, получаем погрешность. 
#include <bits/stdc++.h>
#include <chrono>
using namespace std;
struct SparseMatrix {
    int n;
    vector<vector<pair<int, double>>> rows;

    // CRS
    vector<int> rowPtr, colIdx;
    vector<double> val;
    bool finalized = false;

    SparseMatrix(int n_ = 0) : n(n_), rows(n_) {}

    void add(int i, int j, double v) {
        if (fabs(v) < 1e-15) return;
        rows[i].push_back({j, v});
    }

    void finalize() {
        rowPtr.assign(n + 1, 0);
        colIdx.clear();
        val.clear();

        for (int i = 0; i < n; ++i) {
            auto &r = rows[i];
            sort(r.begin(), r.end(),
                 [](const auto &a, const auto &b) { return a.first < b.first; });

            for (size_t k = 0; k < r.size(); ) {
                int j = r[k].first;
                double s = 0.0;
                while (k < r.size() && r[k].first == j) {//дубликаты бом бом
                    s += r[k].second;
                    ++k;
                }
                if (fabs(s) > 1e-15) {
                    colIdx.push_back(j);
                    val.push_back(s);
                }
            }
            rowPtr[i + 1] = (int)colIdx.size(); 
        }

        rows.clear();
        rows.shrink_to_fit();//чисти память, вилкой чисти
        finalized = true;
    }

    vector<double> matvec(const vector<double>& x) const {//это умножение на вектор просто имба
        vector<double> y(n, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int k = rowPtr[i]; k < rowPtr[i + 1]; ++k) {
                y[i] += val[k] * x[colIdx[k]];
            }
        }
        return y;
    }

    double infNorm() const {
        double mx = 0.0;
        for (int i = 0; i < n; ++i) {
            double s = 0.0;
            for (int k = rowPtr[i]; k < rowPtr[i + 1]; ++k) {
                s += fabs(val[k]);
            }
            mx = max(mx, s);
        }
        return mx;
    }
};
void compare_with_theory(double a, double b,
                         double lx, double rx,
                         double ly, double ry,
                         int nx, int ny,
                         double lambda_max_num,
                         double lambda_min_num)
{
    double hx = (rx - lx) / (nx + 1);
    double hy = (ry - ly) / (ny + 1);

    // более точные формулы
    double lambda_max_th =
        4.0 * a / (hx * hx) * pow(cos(M_PI / (2.0 * (nx + 1))), 2) +
        4.0 * b / (hy * hy) * pow(cos(M_PI / (2.0 * (ny + 1))), 2);

    double lambda_min_th =
        4.0 * a / (hx * hx) * pow(sin(M_PI / (2.0 * (nx + 1))), 2) +
        4.0 * b / (hy * hy) * pow(sin(M_PI / (2.0 * (ny + 1))), 2);

    cout << "\n--- Theory comparison ---\n";

    cout << "lambda_max (num) = " << lambda_max_num << "\n";
    cout << "lambda_max (theory) = " << lambda_max_th << "\n";
    cout << "rel error = "
         << fabs(lambda_max_num - lambda_max_th) / fabs(lambda_max_th)
         << "\n\n";

    cout << "lambda_min (num) = " << lambda_min_num << "\n";
    cout << "lambda_min (theory) = " << lambda_min_th << "\n";
    cout << "rel error = "
         << fabs(lambda_min_num - lambda_min_th) / fabs(lambda_min_th)
         << "\n";
}
static double dotp(const vector<double>& a, const vector<double>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

static double norm2(const vector<double>& a) {
    return sqrt(dotp(a, a));
}

static void normalize(vector<double>& x) {
    double n = norm2(x);
    if (n == 0.0) return;
    for (double &v : x) v /= n;
}

static double rayleigh(const SparseMatrix& A, const vector<double>& x) {
    vector<double> y = A.matvec(x);
    return dotp(x, y) / dotp(x, x);
}

// Наибольшее собственное значение
double power_max(const SparseMatrix& A, int maxIter = 200, double tol = 1e-5) {
    int n = A.n;
    vector<double> x(n, 1.0);
    normalize(x);

    double lambda_prev = 0.0, lambda = 0.0;

    for (int it = 0; it < maxIter; ++it) {
        cout << "PowerMax Iteration: " << it << endl;
        vector<double> y = A.matvec(x);
        double ny = norm2(y);
        if (ny == 0.0) return 0.0;

        for (int i = 0; i < n; ++i) x[i] = y[i] / ny;

        lambda = rayleigh(A, x);
        double pogr=fabs(lambda - lambda_prev) / (fabs(lambda) + 1e-15);
        cout << "pogr is " << pogr << endl;
        if (it > 0 && pogr < tol)
            break;

        lambda_prev = lambda;
    }
    return lambda;
}

// Наименьшее собственное значение через сдвиг beta I - A
double power_min_shifted(const SparseMatrix& A,
                         double lambda_max,
                         int maxIter = 2000,
                         double tol = 1e-8)
{
    int n = A.n;
    double beta = 2.0 * lambda_max;   // сдвиг
    vector<double> x(n, 1.0);
    normalize(x);

    double lambdaB_prev = 0.0;
    double lambdaB = 0.0;

    for (int it = 0; it < maxIter; ++it) {
        cout << "PowerMin Iteration: " << it << endl;

        // Bx = beta*x - A*x
        vector<double> Ax = A.matvec(x);
        vector<double> y(n);
        for (int i = 0; i < n; ++i)
            y[i] = beta * x[i] - Ax[i];

        double ny = norm2(y);
        if (ny == 0.0) return 0.0;

        // новый вектор
        for (int i = 0; i < n; ++i)
            x[i] = y[i] / ny;

        // теперь считаем Rayleigh уже для НОВОГО x
        Ax = A.matvec(x);
        lambdaB = beta - dotp(x, Ax);   // x нормирован, значит x^T B x = beta - x^T A x

        double pogr = fabs(lambdaB - lambdaB_prev) / (fabs(lambdaB) + 1e-15);
        cout << "pogr is " << pogr << endl;

        lambdaB_prev = lambdaB;

        //if (it > 0 && pogr < tol)
          //  break;
    }

    return beta - lambdaB;  // λ_min = β - λ_B
}

// Точное решение
double exact_u(double x, double y) {
    return sin(y * y) + x * x * y;
}

// Правая часть для exact_u
double source_f(double a, double b, double x, double y) {
    (void)x; // здесь f от x не зависит
    return -2.0 * a * y - 2.0 * b * cos(y * y) + 4.0 * b * y * y * sin(y * y);
}

struct System {
    SparseMatrix A;
    vector<double> F;
};

System buildSystem(double a, double b,
                   double lx, double rx,
                   double ly, double ry,
                   int nx, int ny)
{
    // nx, ny — число внутренних узлов
    int N = nx * ny;
    SparseMatrix A(N);
    vector<double> F(N, 0.0);

    double hx = (rx - lx) / (nx + 1);
    double hy = (ry - ly) / (ny + 1);

    double ax = a / (hx * hx);
    double by = b / (hy * hy);
    double diag = 2.0 * ax + 2.0 * by;

    auto id = [nx](int i, int j) {
        // i = 1..nx, j = 1..ny
        return (j - 1) * nx + (i - 1);
    };

    for (int j = 1; j <= ny; ++j) {
        double y = ly + j * hy;
        for (int i = 1; i <= nx; ++i) {
            double x = lx + i * hx;
            int p = id(i, j);

            // diagonal
            A.add(p, p, diag);

            // source
            F[p] = source_f(a, b, x, y);

            // left/right in x
            if (i > 1) A.add(p, id(i - 1, j), -ax);
            else       F[p] += ax * exact_u(lx, y);

            if (i < nx) A.add(p, id(i + 1, j), -ax);
            else        F[p] += ax * exact_u(rx, y);

            // down/up in y
            if (j > 1) A.add(p, id(i, j - 1), -by);
            else       F[p] += by * exact_u(x, ly);

            if (j < ny) A.add(p, id(i, j + 1), -by);
            else        F[p] += by * exact_u(x, ry);
        }
    }

    A.finalize();
    return {A, F};
}

int main() {

    auto t1 = chrono::steady_clock::now();
    // Коэффициенты из задачи
    const double a = 1.1;
    const double b = 0.8;

    // Область
    const double lx = 0.0, rx = 1.0;
    const double ly = 0.0, ry = 1.0;

    // Число внутренних узлов
    const int nx = 999;
    const int ny = 999;

    System sys = buildSystem(a, b, lx, rx, ly, ry, nx, ny);

    double lambda_max = power_max(sys.A);
    double lambda_min = power_min_shifted(sys.A,lambda_max);

    auto t2 = chrono::steady_clock::now();
    chrono::duration<double> elapsed = t2 - t1;

    cout.setf(std::ios::fixed);
    cout << setprecision(15);

    cout << "Matrix size N = " << sys.A.n << "\n";
    cout << "||A||_inf     = " << sys.A.infNorm() << "\n";
    cout << "lambda_max    = " << lambda_max << "\n";
    cout << "lambda_min    = " << lambda_min << "\n";
    cout << "time          = " << elapsed.count() << " sec\n";
    compare_with_theory(a, b, lx, rx, ly, ry,
                    nx, ny,
                    lambda_max, lambda_min);
    return 0;
}