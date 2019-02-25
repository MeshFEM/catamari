![](./images/rainbow_lhr34.png)
**catamari** is a C++14, header-only implementations of sequential and
DAG-scheduled, real and complex, supernodal sparse-direct Cholesky, LDL^T, and
LDL^H factorizations. It similarly contains sequential and DAG-scheduled,
dense and sparse-direct, real and complex, Determinantal Point Process sampling
through modified LDL^H factorizations.

### Dependencies
The only strict dependency for manually including the headers in your project
is:

* [quotient](https://gitlab.com/hodge_star/quotient): A C++14 header-only,
MPL-licensed, implementation of the (Approximate) Minimum Degree reordering
method.

But, if you would like to make use of the project's build system, the only
strict dependency is:

* [meson](http://mesonbuild.com): "Meson is an open source build system meant
to be both extremely fast, and, even more importantly, as user friendly as
possible." 

Meson will automatically install [quotient](https://gitlab.com/hodge_star/quotient), [Catch2](https://github.com/catchorg/Catch2) (a header-only C++
unit-testing library), and [specify](https://gitlab.com/hodge_star/specify)
(a C++14 header-only, command-line argument processor).

Further, it is strongly recommended that one have optimized implementations of
the Basic Linear Algebra Subprograms (BLAS) and the Linear Algebra PACKage
(LAPACK), such as [OpenBLAS](https://www.openblas.net),
[BLIS](https://github.com/flame/blis), or a proprietary alternative such as
[Intel MKL](https://software.intel.com/en-us/mkl).

### Example usage

Usage through the `catamari::CoordinateMatrix` template class is fairly
straight-forward:
```c++
#include "catamari.hpp"

[...]

// Build a real or complex symmetric input matrix.
//
// Alternatively, one could use
// catamari::CoordinateMatrix<Field>::FromMatrixMarket to read the matrix from
// a Matrix Market file (e.g., from the Davis sparse matrix collection). But
// keep in mind that one often needs to enforce explicit symmetry.
catamari::CoordinateMatrix<double> matrix;
matrix.Resize(num_rows, num_rows);
matrix.ReserveEntryAdditions(num_entries_upper_bound);
// Queue updates of entries in the sparse matrix using commands of the form:
//   matrix.QueueEdgeAddition(row, column, value);
matrix.FlushEntryQueues();

// Factor the matrix.
catamari::LDLControl ldl_control;
catamari::LDLFactorization<double> ldl_factorization;
const catamari::LDLResult result =
    ldl_factorization.Factor(matrix, ldl_control);

// Solve a linear system using the factorization.
catamari::BlasMatrix<double> right_hand_sides;
right_hand_sides.height = num_rows;
right_hand_sides.width = num_right_hand_sides;
right_hand_sides.leading_dim = num_rows;
right_hand_sides.data = /* pointer to an input buffer. */;
// The (i, j) entry of the right-hand side can easily be read or modified, e.g.:
//   right_hand_sides(i, j) = 1.;
ldl_factorization.Solve(&right_hand_sides);
```

### Running the unit tests
[meson](http://mesonbuild.com) defaults to debug builds. One might start by
building the project via:
```
mkdir build-debug/
meson build-debug
cd build-debug
ninja
ninja test
```

A release version can be built via:
```
mkdir build-release/
meson build-release --build-type=release
cd build-release
ninja
ninja test
```

OpenMP task parallelism will be used by default if support for OpenMP was
detected; shared-memory parallelism can be disabled with the
`-Ddisable-openmp=true` configuration option.

### Running the example drivers
One can factor Matrix Market examples from the Davis sparse matrix collection
via `examples/factor_matrix_market.cc`, sample a Determinantal Point Process
via `examples/dpp_matrix_market.cc`, or factor 2D or 3D Helmholtz Finite
Element Method discretizations (with Perfectly Matched Layer boundary
conditions) using `examples/helmholtz_2d_pml.cc` and
`examples/helmholtz_3d_pml.cc`. An example plane from running the 3D Helmholtz
solve using 120 x 120 x 120 trilinear hexahedral elements with a converging
lens model spanning 14 wavelengths is shown below:
![](./images/helmholtz_3d_lens_14w.png)

### License
`catamari` is distributed under the
[Mozilla Public License, v. 2.0](https://www.mozilla.org/media/MPL/2.0/index.815ca599c9df.txt).
