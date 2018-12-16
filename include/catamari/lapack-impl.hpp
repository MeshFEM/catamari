/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LAPACK_IMPL_H_
#define CATAMARI_LAPACK_IMPL_H_

#include <cmath>

#include "catamari/blas.hpp"
#include "catamari/lapack.hpp"

#ifdef CATAMARI_HAVE_LAPACK
extern "C" {

#define LAPACK_SYMBOL(name) name##_

void LAPACK_SYMBOL(spotrf)(const char* uplo, const BlasInt* n, float* matrix,
                           const BlasInt* leading_dim, BlasInt* info);

void LAPACK_SYMBOL(dpotrf)(const char* uplo, const BlasInt* n, double* matrix,
                           const BlasInt* leading_dim, BlasInt* info);
}
#endif  // ifdef CATAMARI_HAVE_LAPACK

namespace catamari {

template <class Field>
Int LowerUnblockedCholeskyFactorization(BlasMatrix<Field>* matrix) {
  const Int height = matrix->height;
  for (Int i = 0; i < height; ++i) {
    const Field delta = matrix->Entry(i, i);
    // TODO(Jack Poulson): Enforce 'delta' being real-valued.
    if (delta <= Field{0}) {
      return i;
    }

    // TODO(Jack Poulson): Switch to a custom square-root function so that
    // more general datatypes can be supported.
    const Field delta_sqrt = std::sqrt(delta);
    matrix->Entry(i, i) = delta_sqrt;

    // Solve for the remainder of the i'th column of L.
    for (Int k = i + 1; k < height; ++k) {
      matrix->Entry(k, i) /= delta_sqrt;
    }

    // Perform the Hermitian rank-one update.
    // TODO(Jack Poulson): Replace with a call to HER.
    for (Int j = i + 1; j < height; ++j) {
      const Field eta = Conjugate(matrix->Entry(j, i));
      for (Int k = j; k < height; ++k) {
        const Field& lambda_left = matrix->Entry(k, i);
        matrix->Entry(k, j) -= lambda_left * eta;
      }
    }
  }
  return height;
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline Int LowerUnblockedCholeskyFactorization(BlasMatrix<float>* matrix) {
  const Int height = matrix->height;
  const Int leading_dim = matrix->leading_dim;
  for (Int i = 0; i < height; ++i) {
    const float delta = matrix->Entry(i, i);
    if (delta <= 0) {
      return i;
    }

    const float delta_sqrt = std::sqrt(delta);
    matrix->Entry(i, i) = delta_sqrt;

    // Solve for the remainder of the i'th column of L.
    for (Int k = i + 1; k < height; ++k) {
      matrix->Entry(k, i) /= delta_sqrt;
    }

    // Perform the Hermitian rank-one update.
    {
      const BlasInt rem_height_blas = height - (i + 1);
      const char lower = 'L';
      const float alpha = -1;
      const BlasInt unit_inc_blas = 1;
      const BlasInt leading_dim_blas = leading_dim;
      BLAS_SYMBOL(ssyr)
      (&lower, &rem_height_blas, &alpha, matrix->Pointer(i + 1, i),
       &unit_inc_blas, matrix->Pointer(i + 1, i + 1), &leading_dim_blas);
    }
  }
  return height;
}

template <>
inline Int LowerUnblockedCholeskyFactorization(BlasMatrix<double>* matrix) {
  const Int height = matrix->height;
  const Int leading_dim = matrix->leading_dim;
  for (Int i = 0; i < height; ++i) {
    const double delta = matrix->Entry(i, i);
    if (delta <= 0) {
      return i;
    }

    const double delta_sqrt = std::sqrt(delta);
    matrix->Entry(i, i) = delta_sqrt;

    // Solve for the remainder of the i'th column of L.
    for (Int k = i + 1; k < height; ++k) {
      matrix->Entry(k, i) /= delta_sqrt;
    }

    // Perform the Hermitian rank-one update.
    {
      const BlasInt rem_height_blas = height - (i + 1);
      const char lower = 'L';
      const double alpha = -1;
      const BlasInt unit_inc_blas = 1;
      const BlasInt leading_dim_blas = leading_dim;
      BLAS_SYMBOL(dsyr)
      (&lower, &rem_height_blas, &alpha, matrix->Pointer(i + 1, i),
       &unit_inc_blas, matrix->Pointer(i + 1, i + 1), &leading_dim_blas);
    }
  }
  return height;
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
Int LowerBlockedCholeskyFactorization(BlasMatrix<Field>* matrix,
                                      Int blocksize) {
  const Int height = matrix->height;
  const Int leading_dim = matrix->leading_dim;
  for (Int i = 0; i < height; i += blocksize) {
    const Int bsize = std::min(height - i, blocksize);

    ConstBlasMatrix<Field> diagonal_block;
    diagonal_block.height = bsize;
    diagonal_block.width = bsize;
    diagonal_block.leading_dim = leading_dim;
    diagonal_block.data = matrix->Pointer(i, i);

    BlasMatrix<Field> subdiagonal;
    subdiagonal.height = height - (i + bsize);
    subdiagonal.width = bsize;
    subdiagonal.leading_dim = leading_dim;
    subdiagonal.data = matrix->Pointer(i + bsize, i);

    BlasMatrix<Field> submatrix;
    submatrix.height = height - (i + bsize);
    submatrix.width = height - (i + bsize);
    submatrix.leading_dim = leading_dim;
    submatrix.data = matrix->Pointer(i + bsize, i + bsize);

    // Overwrite the diagonal block with its Cholesky factor.
    const Int num_diag_pivots =
        LowerUnblockedCholeskyFactorization(diagonal_block);
    if (num_diag_pivots < bsize) {
      return i + num_diag_pivots;
    }

    // Solve for the remainder of the block column of L.
    RightLowerAdjointTriangularSolves(diagonal_block, &subdiagonal);

    // Perform the Hermitian rank-bsize update.
    LowerNormalHermitianOuterProduct(Field{-1}, subdiagonal, Field{1},
                                     &submatrix);
  }
  return height;
}

template <class Field>
Int LowerCholeskyFactorization(BlasMatrix<Field>* matrix) {
  const Int blocksize = 64;
  return LowerBlockedCholeskyFactorization(matrix, blocksize);
}

#ifdef CATAMARI_HAVE_LAPACK
template <>
inline Int LowerCholeskyFactorization(BlasMatrix<float>* matrix) {
  const char uplo = 'L';
  const BlasInt height_blas = matrix->height;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BlasInt info;
  LAPACK_SYMBOL(spotrf)
  (&uplo, &height_blas, matrix->data, &leading_dim_blas, &info);
  if (info < 0) {
    std::cerr << "Argument " << -info << " had an illegal value." << std::endl;
    return 0;
  } else if (info == 0) {
    return matrix->height;
  } else {
    return info;
  }
}

template <>
inline Int LowerCholeskyFactorization(BlasMatrix<double>* matrix) {
  const char uplo = 'L';
  const BlasInt height_blas = matrix->height;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BlasInt info;
  LAPACK_SYMBOL(dpotrf)
  (&uplo, &height_blas, matrix->data, &leading_dim_blas, &info);
  if (info < 0) {
    std::cerr << "Argument " << -info << " had an illegal value." << std::endl;
    return 0;
  } else if (info == 0) {
    return matrix->height;
  } else {
    return info;
  }
}
#endif  // ifdef CATAMARI_HAVE_LAPACK

// TODO(Jack Poulson): Extend with optimized versions of this routine.
template <class Field>
Int LowerLDLAdjointFactorization(BlasMatrix<Field>* matrix) {
  const Int height = matrix->height;
  for (Int i = 0; i < height; ++i) {
    const Field& delta = matrix->Entry(i, i);
    // TODO(Jack Poulson): Enforce 'delta' being real-valued.
    if (delta == Field{0}) {
      return i;
    }

    // Solve for the remainder of the i'th column of L.
    for (Int k = i + 1; k < height; ++k) {
      matrix->Entry(k, i) /= delta;
    }

    // Perform the rank-one update.
    for (Int j = i + 1; j < height; ++j) {
      const Field eta = delta * Conjugate(matrix->Entry(j, i));
      for (Int k = j; k < height; ++k) {
        const Field& lambda_left = matrix->Entry(k, i);
        matrix->Entry(k, j) -= lambda_left * eta;
      }
    }
  }
  return height;
}

}  // namespace catamari

#endif  // ifndef CATAMARI_LAPACK_IMPL_H_
