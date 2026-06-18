# Linear Algebra Analyzer in C++

A self-contained C++17 command-line tool for analyzing real matrices. The
program reads a matrix `A`, prints core linear algebra facts about it, and then
solves `Ax = b` using the Moore-Penrose pseudoinverse.

The project is intended as an educational implementation of the main topics in
a first linear algebra course, with numerical methods written directly in C++
instead of delegated to a library.

## Features

- Basic matrix operations: addition, subtraction, multiplication, transpose,
  scalar multiplication, and approximate equality
- RREF with pivot columns, pivot positions, rank, and nullity
- Four fundamental subspaces:
  - null space
  - column space
  - row space
  - left null space
- Orthonormal bases via Gram-Schmidt
- QR decomposition using modified Gram-Schmidt
- Singular value decomposition using eigendecomposition of `A^T A`
- Moore-Penrose pseudoinverse
- Projection matrices
- Minimum-norm least-squares solver for `Ax = b`
- Square-matrix analysis:
  - LUP decomposition with partial pivoting
  - determinant
  - inverse when nonsingular
  - eigenvalues and real eigenvectors
  - positive definite and positive semidefinite checks
  - Jordan form analysis
  - diagonalization and change-of-basis checks when applicable

## Repository Layout

| File | Purpose |
| --- | --- |
| `main.cpp` | Interactive CLI driver that reads input and prints the full analysis |
| `linalg.h` | `Matrix` class API and result structs |
| `linalg.cpp` | Matrix operations, decompositions, subspaces, eigenvalue logic, and solvers |
| `linear_algebra_doc.pdf` | Theory-focused project documentation |
| `linear_algebra_doc.tex` | LaTeX source for the theory documentation |
| `code_walkthrough.pdf` | Code walkthrough and implementation notes |
| `code_walkthrough.tex` | LaTeX source for the code walkthrough |

## Requirements

- A C++17 compiler

No external C++ linear algebra libraries are required.

## Build

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Wno-unused-variable -o linalg main.cpp linalg.cpp
```

This builds the executable:

```bash
./linalg
```

To remove the compiled binary:

```bash
rm -f linalg
```

## Run

Start the analyzer:

```bash
./linalg
```

The program asks for:

1. Matrix dimensions: `rows cols`
2. Matrix entries, row by row
3. A right-hand-side vector `b` with `rows` entries

Example input:

```text
3 3
4 2 2
2 3 1
2 1 3
1
2
3
```

That input analyzes

```text
A = [ 4 2 2 ]
    [ 2 3 1 ]
    [ 2 1 3 ]

b = [ 1 ]
    [ 2 ]
    [ 3 ]
```

You can also pipe input directly:

```bash
printf "3 3\n4 2 2\n2 3 1\n2 1 3\n1\n2\n3\n" | ./linalg
```

For this example, the solver reports the exact solution:

```text
x = [ -0.7500 ]
    [  0.7500 ]
    [  1.2500 ]
```

with residual `||Ax - b|| = 0`.

## Using the Matrix Class in Your Own Code

You can use `linalg.h` and `linalg.cpp` as a small matrix-analysis library.

```cpp
#include "linalg.h"
#include <iostream>

int main() {
    Matrix A({
        {1, 2},
        {3, 4}
    });

    auto rref = A.rref();
    std::cout << "rank = " << rref.rank << "\n";

    Matrix b({
        {5},
        {6}
    });

    Matrix x = A.solve(b);
    x.print("x");
}
```

Compile it with:

```bash
g++ -std=c++17 -O2 -Wall -Wextra your_file.cpp linalg.cpp -o your_program
```

## Numerical Methods

| Operation | Method used |
| --- | --- |
| RREF, rank, pivots | Gauss-Jordan elimination with partial pivoting |
| LUP | Gaussian elimination with row pivoting |
| QR | Modified Gram-Schmidt |
| SVD | Eigendecomposition of `A^T A`, then `u_i = A v_i / sigma_i` |
| Pseudoinverse | `A^+ = V Sigma^+ U^T` from the SVD |
| Symmetric eigenvalues | Unshifted QR iteration |
| General eigenvalues | Faddeev-LeVerrier characteristic polynomial plus Durand-Kerner roots |
| Real eigenvectors | Null space of `A - lambda I` via RREF |
| Jordan form | Algebraic and geometric multiplicity comparison |

## Notes and Limitations

- All computations use `double` precision.
- The global zero tolerance is `1e-9`.
- Complex eigenvalues are reported, but complex eigenvectors are not computed.
- The SVD is based on `A^T A`, which is simple and readable but less stable for
  ill-conditioned matrices than a production Golub-Reinsch or LAPACK-style SVD.
- Rank decisions depend on the fixed tolerance, so nearly rank-deficient
  matrices may be sensitive to scaling.
- The implementation stores dense matrices only.
- Large matrices are not the target use case; the algorithms are mainly cubic
  and intended for small to medium educational examples.
- `projOntoNullSpace()` currently computes `I_m - A A^+`, the orthogonal
  complement of the column-space projection. The domain null-space projector
  would be `I_n - A^+ A`.

## Documentation

For a deeper explanation, see:

- `linear_algebra_doc.pdf` for the mathematical theory and numerical methods
- `code_walkthrough.pdf` for implementation details and function-by-function
  source notes

## License

No license file is currently included in this repository.

## Author

Sarvesh S — IIT Madras (ED23B042)
