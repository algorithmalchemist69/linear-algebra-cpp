#include "linalg.h"
#include <sstream>
#include <cassert>
using namespace std;

// ── Constructors ──────────────────────────────────────────────────────────────
Matrix::Matrix(int m_, int n_, double val)
    : m(m_), n(n_), A(m_, vector<double>(n_, val)) {}

Matrix::Matrix(const vector<vector<double>>& data)
    : m((int)data.size()), n(data.empty() ? 0 : (int)data[0].size()), A(data) {}

// ── Static helpers ────────────────────────────────────────────────────────────
Matrix Matrix::eye(int n) {
    Matrix I(n, n, 0.0);
    for (int i = 0; i < n; i++) I.A[i][i] = 1.0;
    return I;
}
Matrix Matrix::zeros(int rows, int cols) { return Matrix(rows, cols, 0.0); }

Matrix Matrix::fromInput(int rows, int cols) {
    Matrix M(rows, cols);
    cout << "Enter " << rows << "x" << cols << " matrix (row by row):\n";
    for (int i = 0; i < rows; i++) {
        cout << "  Row " << i+1 << ": ";
        for (int j = 0; j < cols; j++) cin >> M.A[i][j];
    }
    return M;
}

// ── Basic Operations ──────────────────────────────────────────────────────────
Matrix Matrix::operator+(const Matrix& B) const {
    if (m != B.m || n != B.n) throw runtime_error("+: size mismatch");
    Matrix C(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            C.A[i][j] = A[i][j] + B.A[i][j];
    return C;
}
Matrix Matrix::operator-(const Matrix& B) const {
    if (m != B.m || n != B.n) throw runtime_error("-: size mismatch");
    Matrix C(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            C.A[i][j] = A[i][j] - B.A[i][j];
    return C;
}
Matrix Matrix::operator*(const Matrix& B) const {
    if (n != B.m) throw runtime_error("*: size mismatch");
    Matrix C(m, B.n, 0.0);
    for (int i = 0; i < m; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < B.n; j++)
                C.A[i][j] += A[i][k] * B.A[k][j];
    return C;
}
Matrix Matrix::operator*(double s) const {
    Matrix C(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            C.A[i][j] = A[i][j] * s;
    return C;
}
Matrix Matrix::T() const {
    Matrix C(n, m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            C.A[j][i] = A[i][j];
    return C;
}
bool Matrix::approxEqual(const Matrix& B) const {
    if (m != B.m || n != B.n) return false;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (abs(A[i][j] - B.A[i][j]) > 1e-6) return false;
    return true;
}

// ── Print ─────────────────────────────────────────────────────────────────────
void Matrix::print(const string& label, int prec) const {
    if (!label.empty()) cout << label << ":\n";
    for (int i = 0; i < m; i++) {
        cout << "  [";
        for (int j = 0; j < n; j++) {
            double v = (abs(A[i][j]) < EPS) ? 0.0 : A[i][j];
            cout << setw(prec+6) << fixed << setprecision(prec) << v;
        }
        cout << " ]\n";
    }
    if (label.empty() && m == 0) cout << "  (empty matrix)\n";
}

// ── RREF ──────────────────────────────────────────────────────────────────────
RREFResult Matrix::rref() const {
    Matrix R = *this;
    vector<int> pivot_cols;
    int cur_row = 0;

    for (int col = 0; col < n && cur_row < m; col++) {
        int piv = -1;
        double best = EPS;
        for (int r = cur_row; r < m; r++) {
            if (abs(R.A[r][col]) > best) { best = abs(R.A[r][col]); piv = r; }
        }
        if (piv == -1) continue;

        swap(R.A[cur_row], R.A[piv]);

        double sc = R.A[cur_row][col];
        for (int j = 0; j < n; j++) R.A[cur_row][j] /= sc;

        for (int r = 0; r < m; r++) {
            if (r == cur_row) continue;
            double f = R.A[r][col];
            if (abs(f) < EPS) continue;
            for (int j = 0; j < n; j++) R.A[r][j] -= f * R.A[cur_row][j];
        }
        pivot_cols.push_back(col);
        cur_row++;
    }
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (abs(R.A[i][j]) < EPS) R.A[i][j] = 0.0;

    return {R, pivot_cols, (int)pivot_cols.size()};
}

// ── LUP Decomposition  (PA = LU) ──────────────────────────────────────────────
tuple<Matrix,Matrix,Matrix> Matrix::LUP() const {
    if (!isSquare()) throw runtime_error("LUP requires square matrix");
    int nn = m;
    Matrix L = eye(nn), U = *this, P = eye(nn);

    for (int k = 0; k < nn; k++) {
        int piv = k;
        double best = abs(U.A[k][k]);
        for (int i = k+1; i < nn; i++)
            if (abs(U.A[i][k]) > best) { best = abs(U.A[i][k]); piv = i; }

        if (piv != k) {
            swap(U.A[k], U.A[piv]);
            swap(P.A[k], P.A[piv]);
            for (int j = 0; j < k; j++) swap(L.A[k][j], L.A[piv][j]);
        }
        if (abs(U.A[k][k]) < EPS) continue;

        for (int i = k+1; i < nn; i++) {
            L.A[i][k] = U.A[i][k] / U.A[k][k];
            for (int j = k; j < nn; j++)
                U.A[i][j] -= L.A[i][k] * U.A[k][j];
        }
    }
    return {L, U, P};
}

// ── QR Decomposition (Modified Gram-Schmidt) ──────────────────────────────────
pair<Matrix,Matrix> Matrix::QR() const {
    int rows = m, cols = n;
    Matrix Q = zeros(rows, cols);
    Matrix R = zeros(cols, cols);
    Matrix U = *this;

    for (int j = 0; j < cols; j++) {
        double norm_sq = 0.0;
        for (int i = 0; i < rows; i++) norm_sq += U.A[i][j] * U.A[i][j];
        double norm = sqrt(norm_sq);

        if (norm < EPS) continue;

        R.A[j][j] = norm;
        for (int i = 0; i < rows; i++) Q.A[i][j] = U.A[i][j] / norm;

        for (int k = j+1; k < cols; k++) {
            double dot = 0.0;
            for (int i = 0; i < rows; i++) dot += Q.A[i][j] * U.A[i][k];
            R.A[j][k] = dot;
            for (int i = 0; i < rows; i++) U.A[i][k] -= dot * Q.A[i][j];
        }
    }
    return {Q, R};
}

// ── Determinant ───────────────────────────────────────────────────────────────
double Matrix::det() const {
    if (!isSquare()) throw runtime_error("det requires square matrix");
    int nn = m;
    Matrix U = *this;
    int swaps = 0;

    for (int k = 0; k < nn; k++) {
        int piv = k;
        double best = abs(U.A[k][k]);
        for (int i = k+1; i < nn; i++)
            if (abs(U.A[i][k]) > best) { best = abs(U.A[i][k]); piv = i; }
        if (piv != k) { swap(U.A[k], U.A[piv]); swaps++; }
        if (abs(U.A[k][k]) < EPS) return 0.0;

        for (int i = k+1; i < nn; i++) {
            double f = U.A[i][k] / U.A[k][k];
            for (int j = k; j < nn; j++) U.A[i][j] -= f * U.A[k][j];
        }
    }
    double d = (swaps % 2 == 0) ? 1.0 : -1.0;
    for (int i = 0; i < nn; i++) d *= U.A[i][i];
    return d;
}

int Matrix::rank() const { return rref().rank; }

// ── Properties ────────────────────────────────────────────────────────────────
bool Matrix::isSymmetric() const {
    if (!isSquare()) return false;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (abs(A[i][j] - A[j][i]) > EPS) return false;
    return true;
}
bool Matrix::isOrthogonal() const {
    if (!isSquare()) return false;
    return (T() * (*this)).approxEqual(eye(n));
}
bool Matrix::isSingular() const {
    if (!isSquare()) return true;
    return abs(det()) < EPS;
}
bool Matrix::isPositiveDefinite() const {
    if (!isSymmetric()) return false;
    auto er = _eigenSymmetric();
    for (auto v : er.real_vals) if (v <= EPS) return false;
    return true;
}
bool Matrix::isPosSemiDef() const {
    if (!isSymmetric()) return false;
    auto er = _eigenSymmetric();
    for (auto v : er.real_vals) if (v < -EPS) return false;
    return true;
}

// ── Inverse (Gauss-Jordan on augmented matrix) ────────────────────────────────
Matrix Matrix::inv() const {
    if (!isSquare())  throw runtime_error("inv: not square");
    if (isSingular()) throw runtime_error("inv: singular matrix");
    int nn = m;
    Matrix aug(nn, 2*nn, 0.0);
    for (int i = 0; i < nn; i++) {
        for (int j = 0; j < nn; j++) aug.A[i][j] = A[i][j];
        aug.A[i][nn+i] = 1.0;
    }
    int cur = 0;
    for (int col = 0; col < nn && cur < nn; col++) {
        int piv = -1; double best = EPS;
        for (int r = cur; r < nn; r++)
            if (abs(aug.A[r][col]) > best) { best = abs(aug.A[r][col]); piv = r; }
        if (piv == -1) throw runtime_error("inv: singular");
        swap(aug.A[cur], aug.A[piv]);
        double sc = aug.A[cur][col];
        for (int j = 0; j < 2*nn; j++) aug.A[cur][j] /= sc;
        for (int r = 0; r < nn; r++) {
            if (r == cur) continue;
            double f = aug.A[r][col];
            for (int j = 0; j < 2*nn; j++) aug.A[r][j] -= f * aug.A[cur][j];
        }
        cur++;
    }
    Matrix inv_mat(nn, nn);
    for (int i = 0; i < nn; i++)
        for (int j = 0; j < nn; j++)
            inv_mat.A[i][j] = (abs(aug.A[i][nn+j]) < EPS) ? 0.0 : aug.A[i][nn+j];
    return inv_mat;
}

// ── Gram-Schmidt (orthonormalize columns) ─────────────────────────────────────
Matrix Matrix::gramSchmidt() const {
    vector<vector<double>> basis;
    int ncols = 0;
    Matrix Q(m, n, 0.0);

    for (int j = 0; j < n; j++) {
        vector<double> v(m);
        for (int i = 0; i < m; i++) v[i] = A[i][j];

        for (auto& u : basis) {
            double dot = 0.0;
            for (int i = 0; i < m; i++) dot += u[i] * v[i];
            for (int i = 0; i < m; i++) v[i] -= dot * u[i];
        }
        double norm = 0.0;
        for (int i = 0; i < m; i++) norm += v[i]*v[i];
        norm = sqrt(norm);
        if (norm < EPS) continue;

        for (int i = 0; i < m; i++) v[i] /= norm;
        basis.push_back(v);
        for (int i = 0; i < m; i++) Q.A[i][ncols] = v[i];
        ncols++;
    }
    Matrix result(m, ncols);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < ncols; j++)
            result.A[i][j] = Q.A[i][j];
    return result;
}

// ── Eigenvalues: Symmetric (unshifted QR — reliable, no near-zero column bug) ─
EigenResult Matrix::_eigenSymmetric() const {
    int nn = m;
    Matrix T_ = *this;
    Matrix V  = eye(nn);

    for (int iter = 0; iter < 12000 * nn; iter++) {
        double off = 0.0;
        for (int i = 0; i < nn; i++)
            for (int j = 0; j < nn; j++)
                if (i != j) off += T_.A[i][j] * T_.A[i][j];
        if (sqrt(off) < EPS * nn) break;

        auto [Q, R] = T_.QR();
        T_ = R * Q;
        V  = V * Q;
    }
    vector<double> vals(nn), ivals(nn, 0.0);
    for (int i = 0; i < nn; i++) vals[i] = T_.A[i][i];
    return {vals, ivals, V};
}

// ── Eigenvalues: General (Faddeev-LeVerrier char-poly + Durand-Kerner roots) ──
// Handles complex eigenvalues correctly; eigenvectors via null-space for real λ.
EigenResult Matrix::_eigenGeneral() const {
    int nn = m;
    using C = complex<double>;

    // ── Faddeev-LeVerrier: characteristic polynomial coefficients ─────────────
    // p(λ) = λⁿ + c[n-1]λⁿ⁻¹ + ... + c[0]
    vector<double> coeff(nn + 1);
    coeff[nn] = 1.0;
    Matrix B = eye(nn);
    for (int k = 1; k <= nn; k++) {
        B = (*this) * B;          // C_k = A * B_{k-1}
        double tr = 0.0;
        for (int i = 0; i < nn; i++) tr += B.A[i][i];
        coeff[nn - k] = -tr / k;
        if (k < nn)
            for (int i = 0; i < nn; i++) B.A[i][i] += coeff[nn - k];
    }

    // ── Durand-Kerner simultaneous root finder ────────────────────────────────
    auto poly_eval = [&](C z) -> C {
        C r(coeff[nn]);
        for (int k = nn - 1; k >= 0; k--) r = r * z + C(coeff[k]);
        return r;
    };

    // Initial circle radius: slightly larger than max |root| (Cauchy bound)
    double cauchy = 0.0;
    for (int k = 0; k < nn; k++)
        cauchy = max(cauchy, abs(coeff[k] / coeff[nn]));
    double r0 = 1.0 + cauchy;

    vector<C> roots(nn);
    for (int k = 0; k < nn; k++) {
        double ang = 2.0 * M_PI * k / nn + 0.17;  // offset avoids exact symmetry
        roots[k] = C(r0 * cos(ang), r0 * sin(ang));
    }

    for (int iter = 0; iter < 1000 * nn; iter++) {
        double max_chg = 0.0;
        vector<C> nr(nn);
        for (int i = 0; i < nn; i++) {
            C denom(1.0);
            for (int j = 0; j < nn; j++)
                if (j != i) denom *= (roots[i] - roots[j]);
            if (abs(denom) < 1e-15) { nr[i] = roots[i]; continue; }
            C corr = poly_eval(roots[i]) / denom;
            nr[i] = roots[i] - corr;
            max_chg = max(max_chg, abs(corr));
        }
        roots = nr;
        if (max_chg < 1e-12) break;
    }

    // Sort: real-only eigenvalues first (descending real part), then complex pairs
    sort(roots.begin(), roots.end(), [](const C& a, const C& b) {
        bool ar = abs(a.imag()) < 1e-7, br = abs(b.imag()) < 1e-7;
        if (ar && !br) return true;
        if (!ar && br) return false;
        return a.real() > b.real();
    });

    // Clean up: imaginary part is "zero" if |imag| < 1e-4 * max(1, |real|)
    // This handles numerical noise for multiple real eigenvalues.
    vector<double> real_vals(nn), imag_vals(nn);
    for (int i = 0; i < nn; i++) {
        double re = roots[i].real(), im = roots[i].imag();
        bool real_eig = abs(im) < max(1e-6, 1e-4 * abs(re));
        real_vals[i] = (abs(re) < 1e-9) ? 0.0 : re;
        imag_vals[i] = real_eig ? 0.0 : im;
    }

    // ── Eigenvectors via null-space for real eigenvalues ──────────────────────
    Matrix V_ = eye(nn);  // fallback identity
    for (int k = 0; k < nn; k++) {
        if (abs(imag_vals[k]) > 1e-7) continue;  // skip complex eigenvalues
        double lam = real_vals[k];
        Matrix A_lam = *this;
        for (int i = 0; i < nn; i++) A_lam.A[i][i] -= lam;
        Matrix ns = A_lam.nullSpace();
        if (ns.n > 0)
            for (int i = 0; i < nn; i++) V_.A[i][k] = ns.A[i][0];
    }
    return {real_vals, imag_vals, V_};
}

EigenResult Matrix::eigen() const {
    if (!isSquare()) throw runtime_error("eigen: not square");
    if (isSymmetric()) return _eigenSymmetric();
    return _eigenGeneral();
}

// ── SVD  A = U Σ V^T ─────────────────────────────────────────────────────────
// Uses eigendecomposition of A^T A (V, Σ²), then computes U = AV/σ
tuple<Matrix,Matrix,Matrix> Matrix::SVD() const {
    Matrix AtA = T() * (*this);
    auto er = AtA._eigenSymmetric();
    int rr = min(m, n);

    // Sort singular values descending
    vector<pair<double,int>> sp;
    for (int i = 0; i < n; i++) sp.push_back({er.real_vals[i], i});
    sort(sp.begin(), sp.end(), [](auto& a, auto& b){ return a.first > b.first; });

    Matrix V(n, n, 0.0);
    vector<double> sigmas(n);
    for (int j = 0; j < n; j++) {
        int idx = sp[j].second;
        for (int i = 0; i < n; i++) V.A[i][j] = er.vecs.A[i][idx];
        sigmas[j] = sqrt(max(0.0, sp[j].first));
    }

    Matrix S = zeros(m, n);
    for (int i = 0; i < rr; i++) S.A[i][i] = sigmas[i];

    // Compute U columns for nonzero singular values
    Matrix U(m, m, 0.0);
    int rank_A = 0;
    for (int j = 0; j < rr; j++) {
        if (sigmas[j] > EPS) {
            for (int i = 0; i < m; i++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) s += A[i][k] * V.A[k][j];
                U.A[i][rank_A] = s / sigmas[j];
            }
            rank_A++;
        }
    }

    // Complete U with orthonormal basis for null(A^T)
    vector<vector<double>> u_basis;
    for (int j = 0; j < rank_A; j++) {
        vector<double> col(m);
        for (int i = 0; i < m; i++) col[i] = U.A[i][j];
        u_basis.push_back(col);
    }
    int cur_col = rank_A;
    for (int k = 0; k < m && cur_col < m; k++) {
        vector<double> e(m, 0.0); e[k] = 1.0;
        for (auto& u : u_basis) {
            double dot = 0.0;
            for (int i = 0; i < m; i++) dot += u[i] * e[i];
            for (int i = 0; i < m; i++) e[i] -= dot * u[i];
        }
        double norm = 0.0;
        for (int i = 0; i < m; i++) norm += e[i]*e[i];
        norm = sqrt(norm);
        if (norm < EPS) continue;
        for (int i = 0; i < m; i++) e[i] /= norm;
        u_basis.push_back(e);
        for (int i = 0; i < m; i++) U.A[i][cur_col] = e[i];
        cur_col++;
    }
    return {U, S, V};
}

// ── Pseudoinverse  A^+ = V Σ^+ U^T ──────────────────────────────────────────
Matrix Matrix::pinv() const {
    auto [U, S, V] = SVD();
    Matrix Sp = zeros(n, m);
    for (int i = 0; i < min(m,n); i++)
        if (S.A[i][i] > EPS) Sp.A[i][i] = 1.0 / S.A[i][i];
    return V * Sp * U.T();
}

// ── Fundamental Subspaces ─────────────────────────────────────────────────────
Matrix Matrix::nullSpace() const {
    auto [R, pivot_cols, rank_] = rref();
    int num_free = n - rank_;
    if (num_free == 0) return Matrix(n, 0);

    vector<bool> is_pivot(n, false);
    for (int c : pivot_cols) is_pivot[c] = true;
    vector<int> free_cols;
    for (int j = 0; j < n; j++) if (!is_pivot[j]) free_cols.push_back(j);

    Matrix N(n, num_free, 0.0);
    for (int fc = 0; fc < num_free; fc++) {
        int fc_col = free_cols[fc];
        N.A[fc_col][fc] = 1.0;
        for (int pr = 0; pr < rank_; pr++)
            N.A[pivot_cols[pr]][fc] = -R.A[pr][fc_col];
    }
    return N;
}

Matrix Matrix::columnSpace() const {
    auto [R, pivot_cols, rank_] = rref();
    Matrix C(m, rank_);
    for (int j = 0; j < rank_; j++)
        for (int i = 0; i < m; i++)
            C.A[i][j] = A[i][pivot_cols[j]];
    return C;
}

// Row space: returned as columns of an n×rank matrix
Matrix Matrix::rowSpace() const {
    auto [R, pivot_cols, rank_] = rref();
    Matrix RS(n, rank_);
    for (int i = 0; i < rank_; i++)
        for (int j = 0; j < n; j++)
            RS.A[j][i] = R.A[i][j];
    return RS;
}

Matrix Matrix::leftNullSpace() const { return T().nullSpace(); }

Matrix Matrix::orthNullSpace()     const { auto N=nullSpace();     return (N.n==0)?N:N.gramSchmidt(); }
Matrix Matrix::orthColumnSpace()   const { auto C=columnSpace();   return (C.n==0)?C:C.gramSchmidt(); }
Matrix Matrix::orthRowSpace()      const { auto R=rowSpace();      return (R.n==0)?R:R.gramSchmidt(); }
Matrix Matrix::orthLeftNullSpace() const { auto L=leftNullSpace(); return (L.n==0)?L:L.gramSchmidt(); }

// ── Projections ───────────────────────────────────────────────────────────────
Matrix Matrix::projOntoColSpace()  const { return (*this) * pinv(); }
Matrix Matrix::projOntoRowSpace()  const { return T().projOntoColSpace().T(); }
Matrix Matrix::projOntoNullSpace() const { return eye(m) - projOntoColSpace(); }

// ── Solve Ax = b (minimum-norm least-squares) ─────────────────────────────────
Matrix Matrix::solve(const Matrix& b) const {
    if (b.n != 1 || b.m != m) throw runtime_error("solve: dimension mismatch");
    Matrix x = pinv() * b;
    for (int i = 0; i < x.m; i++)
        for (int j = 0; j < x.n; j++)
            if (abs(x.A[i][j]) < EPS) x.A[i][j] = 0.0;
    return x;
}

// ── Jordan Form ───────────────────────────────────────────────────────────────
JordanInfo Matrix::jordanForm() const {
    if (!isSquare()) throw runtime_error("jordanForm: not square");
    int nn = m;
    auto er = eigen();

    vector<double> uniq_r, uniq_i;
    vector<int> alg_m, geo_m, num_b;
    vector<bool> seen(nn, false);

    for (int i = 0; i < nn; i++) {
        if (seen[i]) continue;
        seen[i] = true;
        int cnt = 1;
        double scale_i = max(1.0, abs(er.real_vals[i]));
        for (int j = i+1; j < nn; j++) {
            if (!seen[j] &&
                abs(er.real_vals[j]-er.real_vals[i]) < 1e-3 * scale_i &&
                abs(er.imag_vals[j]-er.imag_vals[i]) < 1e-3 * scale_i) {
                seen[j] = true; cnt++;
            }
        }
        uniq_r.push_back(er.real_vals[i]);
        uniq_i.push_back(er.imag_vals[i]);
        alg_m.push_back(cnt);

        // Geometric multiplicity = dim null(A - λI)
        if (abs(er.imag_vals[i]) < EPS) {
            Matrix AlamI = *this;
            for (int k = 0; k < nn; k++) AlamI.A[k][k] -= er.real_vals[i];
            int gm = max(1, nn - AlamI.rank());
            geo_m.push_back(gm);
            num_b.push_back(gm);
        } else {
            geo_m.push_back(1);
            num_b.push_back(1);
        }
    }

    // Build Jordan form matrix
    Matrix J = zeros(nn, nn);
    int pos = 0;
    for (int k = 0; k < (int)uniq_r.size() && pos < nn; k++) {
        int am = alg_m[k], nb = num_b[k];
        double lam = uniq_r[k];
        int base = am / nb, extra = am % nb;
        for (int b = 0; b < nb && pos < nn; b++) {
            int bsz = base + (b < extra ? 1 : 0);
            for (int r = 0; r < bsz; r++) {
                J.A[pos+r][pos+r] = lam;
                if (r < bsz-1) J.A[pos+r][pos+r+1] = 1.0;
            }
            pos += bsz;
        }
    }
    return {uniq_r, uniq_i, alg_m, geo_m, num_b, J};
}

// ── Change of Basis ───────────────────────────────────────────────────────────
Matrix Matrix::changeOfBasis(const Matrix& P) const {
    if (!isSquare() || P.m != m || P.n != n)
        throw runtime_error("changeOfBasis: dimension mismatch");
    return P.inv() * (*this) * P;
}

bool Matrix::isSimilarTo(const Matrix& B) const {
    if (!isSquare() || m != B.m || n != B.n) return false;
    // Same eigenvalues (multiset) + same determinant + same trace
    auto sort_ev = [](vector<double> r, vector<double> im) {
        vector<pair<double,double>> v;
        for (int i = 0; i < (int)r.size(); i++) v.push_back({r[i], im[i]});
        sort(v.begin(), v.end());
        return v;
    };
    auto ea = eigen(), eb = B.eigen();
    if (ea.real_vals.size() != eb.real_vals.size()) return false;
    auto sa = sort_ev(ea.real_vals, ea.imag_vals);
    auto sb = sort_ev(eb.real_vals, eb.imag_vals);
    for (int i = 0; i < (int)sa.size(); i++) {
        if (abs(sa[i].first  - sb[i].first)  > 1e-5) return false;
        if (abs(sa[i].second - sb[i].second) > 1e-5) return false;
    }
    return (abs(det() - B.det()) < 1e-5);
}
