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
Int LowerUnblockedCholeskyFactorization(
    Int height, Field* matrix, Int leading_dim) {
  for (Int i = 0; i < height; ++i) {
    const Field delta = matrix[i + i * leading_dim];
    // TODO(Jack Poulson): Enforce 'delta' being real-valued.
    if (delta <= Field{0}) {
      return i;
    }

    // TODO(Jack Poulson): Switch to a custom square-root function so that
    // more general datatypes can be supported.
    const Field delta_sqrt = std::sqrt(delta);
    matrix[i + i * leading_dim] = delta_sqrt;

    // Solve for the remainder of the i'th column of L.
    for (Int k = i + 1; k < height; ++k) {
      matrix[k + i * leading_dim] /= delta_sqrt;
    }

    // Perform the Hermitian rank-one update.
    // TODO(Jack Poulson): Replace with a call to HER.
    for (Int j = i + 1; j < height; ++j) {
      const Field eta = Conjugate(matrix[j + i * leading_dim]);
      for (Int k = j; k < height; ++k) {
        const Field& lambda_left = matrix[k + i * leading_dim];
        matrix[k + j * height] -= lambda_left * eta;
      }
    }
  }
  return height;
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline Int LowerUnblockedCholeskyFactorization(Int height, float* matrix,
                                               Int leading_dim) {
  for (Int i = 0; i < height; ++i) {
    const float delta = matrix[i + i * leading_dim];
    if (delta <= 0) {
      return i;
    }

    const float delta_sqrt = std::sqrt(delta);
    matrix[i + i * leading_dim] = delta_sqrt;

    // Solve for the remainder of the i'th column of L.
    for (Int k = i + 1; k < height; ++k) {
      matrix[k + i * leading_dim] /= delta_sqrt;
    }

    // Perform the Hermitian rank-one update.
    {
      const BlasInt rem_height_blas = height - (i + 1);
      const char lower = 'L';
      const float alpha = -1;
      const BlasInt unit_inc_blas = 1;
      const BlasInt leading_dim_blas = leading_dim;
      BLAS_SYMBOL(ssyr)
      (&lower, &rem_height_blas, &alpha, &matrix[(i + 1) + i * leading_dim],
       &unit_inc_blas, &matrix[(i + 1) + (i + 1) * leading_dim],
       &leading_dim_blas);
    }
  }
  return height;
}

template <>
inline Int LowerUnblockedCholeskyFactorization(Int height, double* matrix,
                                               Int leading_dim) {
  for (Int i = 0; i < height; ++i) {
    const double delta = matrix[i + i * leading_dim];
    if (delta <= 0) {
      return i;
    }

    const double delta_sqrt = std::sqrt(delta);
    matrix[i + i * leading_dim] = delta_sqrt;

    // Solve for the remainder of the i'th column of L.
    for (Int k = i + 1; k < height; ++k) {
      matrix[k + i * leading_dim] /= delta_sqrt;
    }

    // Perform the Hermitian rank-one update.
    {
      const BlasInt rem_height_blas = height - (i + 1);
      const char lower = 'L';
      const double alpha = -1;
      const BlasInt unit_inc_blas = 1;
      const BlasInt leading_dim_blas = leading_dim;
      BLAS_SYMBOL(dsyr)
      (&lower, &rem_height_blas, &alpha, &matrix[(i + 1) + i * leading_dim],
       &unit_inc_blas, &matrix[(i + 1) + (i + 1) * leading_dim],
       &leading_dim_blas);
    }
  }
  return height;
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
Int LowerBlockedCholeskyFactorization(
    Int height, Field* matrix, Int leading_dim, Int blocksize) {
  for (Int i = 0; i < height; i += blocksize) {
    const Int bsize = std::min(height - i, blocksize);

    // Overwrite the diagonal block with its Cholesky factor.
    const Int num_diag_pivots = LowerUnblockedCholeskyFactorization(
        bsize, &matrix[i + i * leading_dim], leading_dim);
    if (num_diag_pivots < bsize) {
      return i + num_diag_pivots;
    }

    // Solve for the remainder of the block column of L.
    RightLowerAdjointTriangularSolves(
        height - (i + bsize), bsize, &matrix[i + i * leading_dim], leading_dim,
        &matrix[(i + bsize) + i * leading_dim], leading_dim);

    // Perform the Hermitian rank-bsize update.
    LowerNormalHermitianOuterProduct(
        height - (i + bsize), bsize, Field{-1},
        &matrix[(i + bsize) + i * leading_dim], leading_dim, Field{1},
        &matrix[(i + bsize) + (i + bsize) * leading_dim], leading_dim);
  }
  return height;
}

template <class Field>
Int LowerCholeskyFactorization(Int height, Field* matrix, Int leading_dim) {
  const Int blocksize = 64;
  return LowerBlockedCholeskyFactorization(
      height, matrix, leading_dim, blocksize);
}

#ifdef CATAMARI_HAVE_LAPACK
template <>
inline Int LowerCholeskyFactorization(Int height, float* matrix,
                                      Int leading_dim) {
  const char uplo = 'L';
  const BlasInt height_blas = height;
  const BlasInt leading_dim_blas = leading_dim;
  BlasInt info;
  LAPACK_SYMBOL(spotrf)(&uplo, &height_blas, matrix, &leading_dim_blas, &info);
  if (info < 0) {
    std::cerr << "Argument " << -info << " had an illegal value." << std::endl;
    return 0;
  } else if (info == 0) {
    return height;
  } else {
    return info;
  }
}

template <>
inline Int LowerCholeskyFactorization(Int height, double* matrix,
                                      Int leading_dim) {
  const char uplo = 'L';
  const BlasInt height_blas = height;
  const BlasInt leading_dim_blas = leading_dim;
  BlasInt info;
  LAPACK_SYMBOL(dpotrf)(&uplo, &height_blas, matrix, &leading_dim_blas, &info);
  if (info < 0) {
    std::cerr << "Argument " << -info << " had an illegal value." << std::endl;
    return 0;
  } else if (info == 0) {
    return height;
  } else {
    return info;
  }
}
#endif  // ifdef CATAMARI_HAVE_LAPACK

// TODO(Jack Poulson): Extend with optimized versions of this routine.
template <class Field>
Int LowerLDLAdjointFactorization(Int height, Field* matrix, Int leading_dim) {
  for (Int i = 0; i < height; ++i) {
    const Field& delta = matrix[i + i * leading_dim];
    // TODO(Jack Poulson): Enforce 'delta' being real-valued.
    if (delta == Field{0}) {
      return i;
    }

    // Solve for the remainder of the i'th column of L.
    for (Int k = i + 1; k < height; ++k) {
      matrix[k + i * leading_dim] /= delta;
    }

    // Perform the rank-one update.
    for (Int j = i + 1; j < height; ++j) {
      const Field eta = delta * Conjugate(matrix[j + i * leading_dim]);
      for (Int k = j; k < height; ++k) {
        const Field& lambda_left = matrix[k + i * leading_dim];
        matrix[k + j * height] -= lambda_left * eta;
      }
    }
  }
  return height;
}

}  // namespace catamari

#endif  // ifndef CATAMARI_LAPACK_IMPL_H_
