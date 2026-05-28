#include "linalg.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
using namespace std;

static void sep(const string& title = "") {
    cout << "\n" << string(62, '=') << "\n";
    if (!title.empty())
        cout << "  " << title << "\n" << string(62, '-') << "\n";
}

static void printSubspace(const string& name, const Matrix& S, int dim) {
    cout << name << "  [dim = " << dim << "]";
    if (S.n == 0) { cout << "  →  {0}  (trivial)\n"; return; }
    cout << "  (basis as columns):\n";
    S.print();
}

int main() {
    cout << "\n";
    cout << "╔══════════════════════════════════════════════════╗\n";
    cout << "║      Linear Algebra Analyzer  —  C++ Edition     ║\n";
    cout << "╚══════════════════════════════════════════════════╝\n";

    int m, n;
    cout << "\nEnter matrix dimensions (rows cols): ";
    cin >> m >> n;
    Matrix A = Matrix::fromInput(m, n);

    sep("INPUT MATRIX A  (" + to_string(m) + "×" + to_string(n) + ")");
    A.print("A");

    // ── RREF, Rank, Pivots ────────────────────────────────────────────────────
    sep("RREF  ·  RANK  ·  PIVOTS");
    auto rres = A.rref();
    rres.R.print("RREF(A)");
    cout << "\n  Rank    r  = " << rres.rank << "\n";
    cout << "  Nullity n-r= " << (n - rres.rank) << "\n";
    cout << "  Pivot columns (0-indexed): ";
    for (int c : rres.pivot_cols) cout << c << " ";
    cout << "\n  Pivot positions (row, col): ";
    for (int i = 0; i < (int)rres.pivot_cols.size(); i++)
        cout << "(" << i << "," << rres.pivot_cols[i] << ") ";
    cout << "\n";

    // ── Fundamental Subspaces ─────────────────────────────────────────────────
    sep("FOUR FUNDAMENTAL SUBSPACES");
    int r = rres.rank;
    printSubspace("N(A)   Null Space      ", A.nullSpace(),     n - r);  cout << "\n";
    printSubspace("C(A)   Column Space    ", A.columnSpace(),   r);      cout << "\n";
    printSubspace("C(Aᵀ)  Row Space       ", A.rowSpace(),      r);      cout << "\n";
    printSubspace("N(Aᵀ)  Left Null Space ", A.leftNullSpace(), m - r);

    sep("ORTHONORMAL BASES  (via Gram-Schmidt)");
    printSubspace("Orth N(A)  ", A.orthNullSpace(),     n - r);  cout << "\n";
    printSubspace("Orth C(A)  ", A.orthColumnSpace(),   r);      cout << "\n";
    printSubspace("Orth C(Aᵀ) ", A.orthRowSpace(),      r);      cout << "\n";
    printSubspace("Orth N(Aᵀ) ", A.orthLeftNullSpace(), m - r);

    // ── QR Decomposition ──────────────────────────────────────────────────────
    if (m >= n) {
        sep("QR DECOMPOSITION  (A = QR, Modified Gram-Schmidt)");
        auto [Q, R] = A.QR();
        Q.print("Q  (orthonormal columns)");  cout << "\n";
        R.print("R  (upper triangular)");
    }

    // ── SVD ───────────────────────────────────────────────────────────────────
    sep("SINGULAR VALUE DECOMPOSITION  (A = U Σ Vᵀ)");
    auto [U_s, S_s, V_s] = A.SVD();
    U_s.print("U");  cout << "\n";
    S_s.print("Σ (Sigma)");  cout << "\n";
    V_s.print("V");
    cout << "\n  Singular values: ";
    for (int i = 0; i < min(m,n); i++)
        if (S_s.A[i][i] > Matrix::EPS) cout << fixed << setprecision(5) << S_s.A[i][i] << "  ";
    cout << "\n";

    // ── Projection Matrices ───────────────────────────────────────────────────
    sep("PROJECTION MATRICES");
    Matrix Pc = A.projOntoColSpace();
    Matrix Pn = A.projOntoNullSpace();
    Pc.print("P_col  (onto column space,  P = A A⁺)");   cout << "\n";
    Pn.print("P_null (onto null space,    I − P_col)");
    if (!A.isSquare()) {
        cout << "\n";
        A.projOntoRowSpace().print("P_row  (onto row space, Aᵀ(AAᵀ)⁻¹A)");
    }

    // ── Square-matrix section ─────────────────────────────────────────────────
    if (A.isSquare()) {
        // LU
        sep("LUP DECOMPOSITION  (PA = LU)");
        auto [L, U_lu, P_lu] = A.LUP();
        L.print("L"); cout << "\n";
        U_lu.print("U"); cout << "\n";
        P_lu.print("P  (permutation)");

        // Properties
        sep("PROPERTIES");
        auto yn = [](bool b) { return b ? "YES" : "NO"; };
        cout << "  Symmetric:            " << yn(A.isSymmetric())        << "\n";
        cout << "  Orthogonal (Q^TQ=I):  " << yn(A.isOrthogonal())       << "\n";
        cout << "  Singular (det=0):     " << yn(A.isSingular())         << "\n";
        cout << "  Positive Definite:    " << yn(A.isPositiveDefinite()) << "\n";
        cout << "  Pos. Semi-Definite:   " << yn(A.isPosSemiDef())       << "\n";

        // Determinant
        sep("DETERMINANT");
        cout << "  det(A) = " << fixed << setprecision(8) << A.det() << "\n";

        // Inverse / Pseudoinverse
        if (!A.isSingular()) {
            sep("INVERSE  A⁻¹");
            A.inv().print("A⁻¹");
        } else {
            sep("MOORE-PENROSE PSEUDOINVERSE  (A is singular)");
            A.pinv().print("A⁺");
        }

        // Eigenvalues & Eigenvectors
        sep("EIGENVALUES  &  EIGENVECTORS");
        auto er = A.eigen();
        int ne = (int)er.real_vals.size();
        cout << "  Eigenvalues:\n";
        for (int i = 0; i < ne; i++) {
            cout << "    λ" << i+1 << " = " << fixed << setprecision(6) << er.real_vals[i];
            if (abs(er.imag_vals[i]) > 1e-6)
                cout << " + " << er.imag_vals[i] << "i  (complex pair)";
            cout << "\n";
        }
        cout << "\n  Eigenvectors (columns, real part):\n";
        er.vecs.print();

        // Jordan Form
        sep("JORDAN FORM ANALYSIS");
        auto ji = A.jordanForm();
        int nev = (int)ji.eigenvalues.size();
        cout << "  " << left
             << setw(14) << "Eigenvalue"
             << setw(8)  << "Alg.M"
             << setw(8)  << "Geo.M"
             << setw(14) << "#Jordan Blocks"
             << "Diagonalizable?\n";
        cout << "  " << string(52, '-') << "\n";
        for (int i = 0; i < nev; i++) {
            cout << "  " << left;
            ostringstream os;
            os << fixed << setprecision(4) << ji.eigenvalues[i];
            if (abs(ji.imag_eigenvalues[i]) > 1e-6)
                os << "±" << abs(ji.imag_eigenvalues[i]) << "i";
            cout << setw(14) << os.str()
                 << setw(8)  << ji.alg_mult[i]
                 << setw(8)  << ji.geo_mult[i]
                 << setw(14) << ji.num_blocks[i]
                 << (ji.alg_mult[i] == ji.geo_mult[i] ? "YES" : "NO")
                 << "\n";
        }
        cout << "\n  Jordan Form J:\n";
        ji.J.print();

        // Similar matrices
        sep("SIMILAR MATRICES  &  CHANGE OF BASIS");
        cout << "  B ~ A  iff  B = P⁻¹AP  for some invertible P.\n";
        cout << "  Equivalent: same Jordan form  (same eigenvalue multiset + block sizes).\n";

        // Diagonalization demo (symmetric or when all alg=geo mult)
        bool diagonalizable = true;
        for (int i = 0; i < nev; i++)
            if (ji.alg_mult[i] != ji.geo_mult[i]) { diagonalizable = false; break; }

        if (diagonalizable && n <= 5) {
            Matrix P_eig = er.vecs;
            bool P_ok = !P_eig.isSingular();
            if (P_ok) {
                cout << "\n  A is diagonalizable.  Using eigenvectors as basis:\n";
                try {
                    Matrix D = A.changeOfBasis(P_eig);
                    D.print("  P⁻¹AP  (≈ diagonal Λ)");
                } catch (...) {
                    cout << "  (could not diagonalize — likely repeated eigenvalues)\n";
                }
            }
        } else if (!diagonalizable) {
            cout << "\n  A is NOT diagonalizable (defective eigenvalue exists).\n";
            cout << "  Use Jordan form J above as the canonical similar matrix.\n";
        }
    } else {
        // Non-square: show pseudoinverse
        sep("MOORE-PENROSE PSEUDOINVERSE  A⁺");
        A.pinv().print("A⁺");
    }

    // ── Solve Ax = b ──────────────────────────────────────────────────────────
    sep("SOLVE  Ax = b");
    cout << "Enter column vector b  (" << m << " entries): ";
    Matrix b_vec(m, 1);
    for (int i = 0; i < m; i++) cin >> b_vec.A[i][0];

    Matrix x_sol = A.solve(b_vec);
    x_sol.print("x  (minimum-norm least-squares solution)");

    Matrix residual = A * x_sol - b_vec;
    double res = 0.0;
    for (int i = 0; i < m; i++) res += residual.A[i][0] * residual.A[i][0];
    res = sqrt(res);
    cout << "\n  Residual  ‖Ax − b‖ = " << fixed << setprecision(8) << res << "\n";
    if (res > 1e-6)
        cout << "  (b ∉ column space → x is the least-squares solution)\n";
    else
        cout << "  (exact solution found)\n";

    cout << "\n" << string(62, '=') << "\n";
    cout << "  Analysis complete.\n\n";
    return 0;
}
