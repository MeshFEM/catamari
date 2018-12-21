/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_FACTORIZATIONS_IMPL_H_
#define CATAMARI_DENSE_FACTORIZATIONS_IMPL_H_

#include <cmath>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/lapack.hpp"

#include "catamari/dense_factorizations.hpp"

namespace catamari {

template <class Field>
Int LowerUnblockedCholeskyFactorization(BlasMatrix<Field>* matrix) {
  typedef ComplexBase<Field> Real;
  const Int height = matrix->height;
  for (Int i = 0; i < height; ++i) {
    const Real delta = RealPart(matrix->Entry(i, i));
    // TODO(Jack Poulson): Enforce 'delta' being real-valued.
    if (delta <= Real{0}) {
      return i;
    }

    // TODO(Jack Poulson): Switch to a custom square-root function so that
    // more general datatypes can be supported.
    const Real delta_sqrt = std::sqrt(delta);
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
  typedef ComplexBase<Field> Real;
  const Int height = matrix->height;
  const Int leading_dim = matrix->leading_dim;
  for (Int i = 0; i < height; i += blocksize) {
    const Int bsize = std::min(height - i, blocksize);

    // Overwrite the diagonal block with its Cholesky factor.
    BlasMatrix<Field> diagonal_block;
    diagonal_block.height = bsize;
    diagonal_block.width = bsize;
    diagonal_block.leading_dim = leading_dim;
    diagonal_block.data = matrix->Pointer(i, i);
    const Int num_diag_pivots =
        LowerUnblockedCholeskyFactorization(&diagonal_block);
    if (num_diag_pivots < bsize) {
      return i + num_diag_pivots;
    }
    if (height == i + bsize) {
      break;
    }

    // Solve for the remainder of the block column of L.
    BlasMatrix<Field> subdiagonal;
    subdiagonal.height = height - (i + bsize);
    subdiagonal.width = bsize;
    subdiagonal.leading_dim = leading_dim;
    subdiagonal.data = matrix->Pointer(i + bsize, i);
    const ConstBlasMatrix<Field> const_diagonal_block = diagonal_block;
    RightLowerAdjointTriangularSolves(const_diagonal_block, &subdiagonal);

    // Perform the Hermitian rank-bsize update.
    BlasMatrix<Field> submatrix;
    submatrix.height = height - (i + bsize);
    submatrix.width = height - (i + bsize);
    submatrix.leading_dim = leading_dim;
    submatrix.data = matrix->Pointer(i + bsize, i + bsize);
    const ConstBlasMatrix<Field> const_subdiagonal = subdiagonal;
    LowerNormalHermitianOuterProduct(Real{-1}, const_subdiagonal, Real{1},
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

template <class Field>
Int LowerUnblockedLDLAdjointFactorization(BlasMatrix<Field>* matrix) {
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

template <class Field>
Int LowerBlockedLDLAdjointFactorization(BlasMatrix<Field>* matrix,
                                        Int blocksize) {
  const Int height = matrix->height;
  const Int leading_dim = matrix->leading_dim;

  std::vector<Field> buffer(std::max(height - blocksize, Int(0)) * blocksize);
  BlasMatrix<Field> factor;
  factor.data = buffer.data();

  for (Int i = 0; i < height; i += blocksize) {
    const Int bsize = std::min(height - i, blocksize);

    // Overwrite the diagonal block with its LDL' factorization.
    BlasMatrix<Field> diagonal_block;
    diagonal_block.height = bsize;
    diagonal_block.width = bsize;
    diagonal_block.leading_dim = leading_dim;
    diagonal_block.data = matrix->Pointer(i, i);
    const Int num_diag_pivots =
        LowerUnblockedLDLAdjointFactorization(&diagonal_block);
    if (num_diag_pivots < bsize) {
      return i + num_diag_pivots;
    }
    if (height == i + bsize) {
      break;
    }

    // Solve for the remainder of the block column of L.
    BlasMatrix<Field> subdiagonal;
    subdiagonal.height = height - (i + bsize);
    subdiagonal.width = bsize;
    subdiagonal.leading_dim = leading_dim;
    subdiagonal.data = matrix->Pointer(i + bsize, i);
    const ConstBlasMatrix<Field> const_diagonal_block = diagonal_block;
    RightLowerAdjointUnitTriangularSolves(const_diagonal_block, &subdiagonal);

    // Copy the conjugate of the current factor.
    factor.height = subdiagonal.height;
    factor.width = subdiagonal.width;
    factor.leading_dim = subdiagonal.height;
    for (Int j = 0; j < subdiagonal.width; ++j) {
      for (Int k = 0; k < subdiagonal.height; ++k) {
        factor(k, j) = Conjugate(subdiagonal(k, j));
      }
    }

    // Solve against the diagonal.
    for (Int j = 0; j < subdiagonal.width; ++j) {
      const ComplexBase<Field> delta = RealPart(const_diagonal_block(j, j));
      for (Int k = 0; k < subdiagonal.height; ++k) {
        subdiagonal(k, j) /= delta;
      }
    }

    // Perform the Hermitian rank-bsize update.
    const ConstBlasMatrix<Field> const_subdiagonal = subdiagonal;
    const ConstBlasMatrix<Field> const_factor = factor;
    BlasMatrix<Field> submatrix;
    submatrix.height = height - (i + bsize);
    submatrix.width = height - (i + bsize);
    submatrix.leading_dim = leading_dim;
    submatrix.data = matrix->Pointer(i + bsize, i + bsize);
    MatrixMultiplyLowerNormalTranspose(Field{-1}, const_subdiagonal,
                                       const_factor, Field{1}, &submatrix);
  }
  return height;
}

template <class Field>
Int LowerLDLAdjointFactorization(BlasMatrix<Field>* matrix) {
  const Int blocksize = 64;
  return LowerBlockedLDLAdjointFactorization(matrix, blocksize);
}

template <class Field>
Int LowerUnblockedLDLTransposeFactorization(BlasMatrix<Field>* matrix) {
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
      const Field eta = delta * matrix->Entry(j, i);
      for (Int k = j; k < height; ++k) {
        const Field& lambda_left = matrix->Entry(k, i);
        matrix->Entry(k, j) -= lambda_left * eta;
      }
    }
  }
  return height;
}

template <class Field>
Int LowerBlockedLDLTransposeFactorization(BlasMatrix<Field>* matrix,
                                          Int blocksize) {
  const Int height = matrix->height;
  const Int leading_dim = matrix->leading_dim;

  std::vector<Field> buffer(std::max(height - blocksize, Int(0)) * blocksize);
  BlasMatrix<Field> factor;
  factor.data = buffer.data();

  for (Int i = 0; i < height; i += blocksize) {
    const Int bsize = std::min(height - i, blocksize);

    // Overwrite the diagonal block with its LDL' factorization.
    BlasMatrix<Field> diagonal_block;
    diagonal_block.height = bsize;
    diagonal_block.width = bsize;
    diagonal_block.leading_dim = leading_dim;
    diagonal_block.data = matrix->Pointer(i, i);
    const Int num_diag_pivots =
        LowerUnblockedLDLTransposeFactorization(&diagonal_block);
    if (num_diag_pivots < bsize) {
      return i + num_diag_pivots;
    }
    if (height == i + bsize) {
      break;
    }

    // Solve for the remainder of the block column of L.
    BlasMatrix<Field> subdiagonal;
    subdiagonal.height = height - (i + bsize);
    subdiagonal.width = bsize;
    subdiagonal.leading_dim = leading_dim;
    subdiagonal.data = matrix->Pointer(i + bsize, i);
    const ConstBlasMatrix<Field> const_diagonal_block = diagonal_block;
    RightLowerTransposeUnitTriangularSolves(const_diagonal_block, &subdiagonal);

    // Copy the current factor.
    factor.height = subdiagonal.height;
    factor.width = subdiagonal.width;
    factor.leading_dim = subdiagonal.height;
    for (Int j = 0; j < subdiagonal.width; ++j) {
      for (Int k = 0; k < subdiagonal.height; ++k) {
        factor(k, j) = subdiagonal(k, j);
      }
    }

    // Solve against the diagonal.
    for (Int j = 0; j < subdiagonal.width; ++j) {
      const Field delta = const_diagonal_block(j, j);
      for (Int k = 0; k < subdiagonal.height; ++k) {
        subdiagonal(k, j) /= delta;
      }
    }

    // Perform the symmetric rank-bsize update.
    const ConstBlasMatrix<Field> const_subdiagonal = subdiagonal;
    const ConstBlasMatrix<Field> const_factor = factor;
    BlasMatrix<Field> submatrix;
    submatrix.height = height - (i + bsize);
    submatrix.width = height - (i + bsize);
    submatrix.leading_dim = leading_dim;
    submatrix.data = matrix->Pointer(i + bsize, i + bsize);
    MatrixMultiplyLowerNormalTranspose(Field{-1}, const_subdiagonal,
                                       const_factor, Field{1}, &submatrix);
  }
  return height;
}

template <class Field>
Int LowerLDLTransposeFactorization(BlasMatrix<Field>* matrix) {
  const Int blocksize = 64;
  return LowerBlockedLDLTransposeFactorization(matrix, blocksize);
}

template <class Field>
std::vector<Int> LowerUnblockedFactorAndSampleDPP(
    bool maximum_likelihood, BlasMatrix<Field>* matrix, std::mt19937* generator,
    std::uniform_real_distribution<ComplexBase<Field>>* uniform_dist) {
  typedef ComplexBase<Field> Real;
  const Int height = matrix->height;
  std::vector<Int> sample;
  sample.reserve(height);

  for (Int i = 0; i < height; ++i) {
    Real delta = RealPart(matrix->Entry(i, i));
    CATAMARI_ASSERT(delta >= Real{0} && delta <= Real{1},
                    "Diagonal value was outside of [0, 1].");
    const bool keep_index = maximum_likelihood
                                ? delta >= Real(1) / Real(2)
                                : (*uniform_dist)(*generator) <= delta;
    if (keep_index) {
      sample.push_back(i);
    } else {
      delta -= Real{1};
      matrix->Entry(i, i) = delta;
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
  return sample;
}

template <class Field>
std::vector<Int> LowerBlockedFactorAndSampleDPP(
    bool maximum_likelihood, BlasMatrix<Field>* matrix, std::mt19937* generator,
    std::uniform_real_distribution<ComplexBase<Field>>* uniform_dist,
    Int blocksize) {
  const Int height = matrix->height;
  const Int leading_dim = matrix->leading_dim;

  std::vector<Int> sample;
  sample.reserve(height);

  std::vector<Field> buffer(std::max(height - blocksize, Int(0)) * blocksize);
  BlasMatrix<Field> factor;
  factor.data = buffer.data();

  for (Int i = 0; i < height; i += blocksize) {
    const Int bsize = std::min(height - i, blocksize);

    // Overwrite the diagonal block with its LDL' factorization.
    BlasMatrix<Field> diagonal_block;
    diagonal_block.height = bsize;
    diagonal_block.width = bsize;
    diagonal_block.leading_dim = leading_dim;
    diagonal_block.data = matrix->Pointer(i, i);
    std::vector<Int> block_sample = LowerUnblockedFactorAndSampleDPP(
        maximum_likelihood, &diagonal_block, generator, uniform_dist);
    for (const Int& index : block_sample) {
      sample.push_back(i + index);
    }
    if (height == i + bsize) {
      break;
    }

    // Solve for the remainder of the block column of L.
    BlasMatrix<Field> subdiagonal;
    subdiagonal.height = height - (i + bsize);
    subdiagonal.width = bsize;
    subdiagonal.leading_dim = leading_dim;
    subdiagonal.data = matrix->Pointer(i + bsize, i);
    const ConstBlasMatrix<Field> const_diagonal_block = diagonal_block;
    RightLowerAdjointUnitTriangularSolves(const_diagonal_block, &subdiagonal);

    // Copy the conjugate of the current factor.
    factor.height = subdiagonal.height;
    factor.width = subdiagonal.width;
    factor.leading_dim = subdiagonal.height;
    for (Int j = 0; j < subdiagonal.width; ++j) {
      for (Int k = 0; k < subdiagonal.height; ++k) {
        factor(k, j) = Conjugate(subdiagonal(k, j));
      }
    }

    // Solve against the diagonal.
    for (Int j = 0; j < subdiagonal.width; ++j) {
      const ComplexBase<Field> delta = RealPart(const_diagonal_block(j, j));
      for (Int k = 0; k < subdiagonal.height; ++k) {
        subdiagonal(k, j) /= delta;
      }
    }

    // Perform the Hermitian rank-bsize update.
    const ConstBlasMatrix<Field> const_subdiagonal = subdiagonal;
    const ConstBlasMatrix<Field> const_factor = factor;
    BlasMatrix<Field> submatrix;
    submatrix.height = height - (i + bsize);
    submatrix.width = height - (i + bsize);
    submatrix.leading_dim = leading_dim;
    submatrix.data = matrix->Pointer(i + bsize, i + bsize);
    MatrixMultiplyLowerNormalTranspose(Field{-1}, const_subdiagonal,
                                       const_factor, Field{1}, &submatrix);
  }
  return sample;
}

template <class Field>
std::vector<Int> LowerFactorAndSampleDPP(
    bool maximum_likelihood, BlasMatrix<Field>* matrix, std::mt19937* generator,
    std::uniform_real_distribution<ComplexBase<Field>>* uniform_dist) {
  const Int blocksize = 64;
  return LowerBlockedFactorAndSampleDPP(maximum_likelihood, matrix, generator,
                                        uniform_dist, blocksize);
}

}  // namespace catamari

#endif  // ifndef CATAMARI_DENSE_FACTORIZATIONS_IMPL_H_
