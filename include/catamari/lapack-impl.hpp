/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LAPACK_IMPL_H_
#define CATAMARI_LAPACK_IMPL_H_

#include "catamari/blas.hpp"
#include "catamari/lapack.hpp"

namespace catamari {

// TODO(Jack Poulson): Overload with calls to POTRF.
template <class Field>
Int LowerCholeskyFactorization(Int height, Field* A, Int leading_dim) {
  for (Int i = 0; i < height; ++i) {
    const Field delta = A[i + i * leading_dim];
    // TODO(Jack Poulson): Enforce 'delta' being real-valued.
    if (delta <= Field{0}) {
      return i;
    }

    // TODO(Jack Poulson): Switch to a custom square-root function so that
    // more general datatypes can be supported.
    const Field delta_sqrt = std::sqrt(delta);
    A[i + i * leading_dim] = delta_sqrt;

    // Solve for the remainder of the i'th column of L.
    for (Int k = i + 1; k < height; ++k) {
      A[k + i * leading_dim] /= delta_sqrt;
    }

    // Perform the Hermitian rank-one update.
    // TODO(Jack Poulson): Replace with a call to HER.
    for (Int j = i + 1; j < height; ++j) {
      const Field eta = Conjugate(A[j + i * leading_dim]);
      for (Int k = j; k < height; ++k) {
        const Field& lambda_left = A[k + i * leading_dim];
        A[k + j * height] -= lambda_left * eta;
      }
    }
  }
  return height;
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline Int LowerCholeskyFactorization(Int height, float* A, Int leading_dim) {
  for (Int i = 0; i < height; ++i) {
    const float delta = A[i + i * leading_dim];
    if (delta <= 0) {
      return i;
    }

    const float delta_sqrt = std::sqrt(delta);
    A[i + i * leading_dim] = delta_sqrt;

    // Solve for the remainder of the i'th column of L.
    for (Int k = i + 1; k < height; ++k) {
      A[k + i * leading_dim] /= delta_sqrt;
    }

    // Perform the Hermitian rank-one update.
    {
      const BlasInt rem_height_blas = height - (i + 1);
      const char lower = 'L';
      const float alpha = -1;
      const BlasInt unit_inc_blas = 1;
      const BlasInt leading_dim_blas = leading_dim;
      BLAS_SYMBOL(ssyr)(
          &lower, &rem_height_blas, &alpha, &A[(i + 1) + i * leading_dim],
          &unit_inc_blas, &A[(i + 1) + (i + 1) * leading_dim],
          &leading_dim_blas);
    }
  }
  return height;
}

template <>
inline Int LowerCholeskyFactorization(Int height, double* A, Int leading_dim) {
  for (Int i = 0; i < height; ++i) {
    const double delta = A[i + i * leading_dim];
    if (delta <= 0) {
      return i;
    }

    const double delta_sqrt = std::sqrt(delta);
    A[i + i * leading_dim] = delta_sqrt;

    // Solve for the remainder of the i'th column of L.
    for (Int k = i + 1; k < height; ++k) {
      A[k + i * leading_dim] /= delta_sqrt;
    }

    // Perform the Hermitian rank-one update.
    {
      const BlasInt rem_height_blas = height - (i + 1);
      const char lower = 'L';
      const double alpha = -1;
      const BlasInt unit_inc_blas = 1;
      const BlasInt leading_dim_blas = leading_dim;
      BLAS_SYMBOL(dsyr)(
          &lower, &rem_height_blas, &alpha, &A[(i + 1) + i * leading_dim],
          &unit_inc_blas, &A[(i + 1) + (i + 1) * leading_dim],
          &leading_dim_blas);
    }
  }
  return height;
}
#endif  // ifdef CATAMARI_HAVE_BLAS

// TODO(Jack Poulson): Extend with optimized versions of this routine.
template <class Field>
Int LowerLDLAdjointFactorization(Int height, Field* A, Int leading_dim) {
  for (Int i = 0; i < height; ++i) {
    const Field& delta = A[i + i * leading_dim];
    // TODO(Jack Poulson): Enforce 'delta' being real-valued.
    if (delta == Field{0}) {
      return i;
    }

    // Solve for the remainder of the i'th column of L.
    for (Int k = i + 1; k < height; ++k) {
      A[k + i * leading_dim] /= delta;
    }

    // Perform the rank-one update.
    for (Int j = i + 1; j < height; ++j) {
      const Field eta = delta * Conjugate(A[j + i * leading_dim]);
      for (Int k = j; k < height; ++k) {
        const Field& lambda_left = A[k + i * leading_dim];
        A[k + j * height] -= lambda_left * eta;
      }
    }
  }
  return height;
}

}  // namespace catamari

#endif  // ifndef CATAMARI_LAPACK_IMPL_H_
