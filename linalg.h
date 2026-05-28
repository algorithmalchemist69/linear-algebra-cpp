#pragma once
#include <vector>
#include <string>
#include <utility>
#include <tuple>
#include <complex>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <sstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Forward-declare result types so Matrix can use them as return types
struct RREFResult;
struct EigenResult;
struct JordanInfo;

class Matrix {
public:
    int m, n;
    std::vector<std::vector<double>> A;
    static constexpr double EPS = 1e-9;

    Matrix(int m = 0, int n = 0, double val = 0.0);
    Matrix(const std::vector<std::vector<double>>& data);

    double& operator()(int i, int j)       { return A[i][j]; }
    double  operator()(int i, int j) const { return A[i][j]; }

    Matrix operator+(const Matrix& B) const;
    Matrix operator-(const Matrix& B) const;
    Matrix operator*(const Matrix& B) const;
    Matrix operator*(double s) const;
    Matrix T() const;
    bool   approxEqual(const Matrix& B) const;

    void print(const std::string& label = "", int prec = 4) const;

    // ── Decompositions ──────────────────────────────────────────────
    std::tuple<Matrix,Matrix,Matrix> LUP() const;  // PA = LU
    std::pair<Matrix,Matrix>         QR()  const;  // A = QR
    std::tuple<Matrix,Matrix,Matrix> SVD() const;  // A = U Σ V^T

    // ── RREF & Pivots ───────────────────────────────────────────────
    RREFResult rref() const;

    // ── Properties ─────────────────────────────────────────────────
    double det()  const;
    int    rank() const;
    Matrix inv()  const;
    Matrix pinv() const;

    bool isSquare()           const { return m == n; }
    bool isSymmetric()        const;
    bool isOrthogonal()       const;
    bool isSingular()         const;
    bool isPositiveDefinite() const;
    bool isPosSemiDef()       const;

    // ── Eigenvalues / Vectors ───────────────────────────────────────
    EigenResult eigen() const;

    // ── Fundamental Subspaces ───────────────────────────────────────
    Matrix nullSpace()     const;
    Matrix columnSpace()   const;
    Matrix rowSpace()      const;
    Matrix leftNullSpace() const;

    Matrix orthNullSpace()     const;
    Matrix orthColumnSpace()   const;
    Matrix orthRowSpace()      const;
    Matrix orthLeftNullSpace() const;

    // ── Projections ─────────────────────────────────────────────────
    Matrix projOntoColSpace()  const;
    Matrix projOntoRowSpace()  const;
    Matrix projOntoNullSpace() const;

    // ── Solve ───────────────────────────────────────────────────────
    Matrix solve(const Matrix& b) const;

    // ── Jordan Form ─────────────────────────────────────────────────
    JordanInfo jordanForm() const;

    // ── Change of Basis ─────────────────────────────────────────────
    Matrix changeOfBasis(const Matrix& P) const;
    bool   isSimilarTo(const Matrix& B)   const;

    // ── Gram-Schmidt ────────────────────────────────────────────────
    Matrix gramSchmidt() const;

    // ── Static helpers ──────────────────────────────────────────────
    static Matrix eye(int n);
    static Matrix zeros(int rows, int cols);
    static Matrix fromInput(int rows, int cols);

private:
    EigenResult _eigenSymmetric() const;
    EigenResult _eigenGeneral()   const;
};

// ── Result structs (defined after Matrix is complete) ────────────────────────

struct RREFResult {
    Matrix           R;
    std::vector<int> pivot_cols;
    int              rank;
};

struct EigenResult {
    std::vector<double> real_vals;
    std::vector<double> imag_vals;
    Matrix              vecs;      // columns = eigenvectors
};

struct JordanInfo {
    std::vector<double> eigenvalues;
    std::vector<double> imag_eigenvalues;
    std::vector<int>    alg_mult;
    std::vector<int>    geo_mult;
    std::vector<int>    num_blocks;
    Matrix              J;
};
