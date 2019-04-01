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
Int LowerUnblockedCholeskyFactorization(BlasMatrixView<Field>* matrix) {
  typedef ComplexBase<Field> Real;
  const Int height = matrix->height;
  for (Int i = 0; i < height; ++i) {
    const Real delta = RealPart(matrix->Entry(i, i));
    matrix->Entry(i, i) = delta;
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
inline Int LowerUnblockedCholeskyFactorization(BlasMatrixView<float>* matrix) {
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
inline Int LowerUnblockedCholeskyFactorization(BlasMatrixView<double>* matrix) {
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
Int LowerBlockedCholeskyFactorization(Int block_size,
                                      BlasMatrixView<Field>* matrix) {
  typedef ComplexBase<Field> Real;
  const Int height = matrix->height;
  for (Int i = 0; i < height; i += block_size) {
    const Int bsize = std::min(height - i, block_size);

    // Overwrite the diagonal block with its Cholesky factor.
    BlasMatrixView<Field> diagonal_block =
        matrix->Submatrix(i, i, bsize, bsize);
    const Int num_diag_pivots =
        LowerUnblockedCholeskyFactorization(&diagonal_block);
    if (num_diag_pivots < bsize) {
      return i + num_diag_pivots;
    }
    if (height == i + bsize) {
      break;
    }

    // Solve for the remainder of the block column of L.
    BlasMatrixView<Field> subdiagonal =
        matrix->Submatrix(i + bsize, i, height - (i + bsize), bsize);
    RightLowerAdjointTriangularSolves(diagonal_block.ToConst(), &subdiagonal);

    // Perform the Hermitian rank-bsize update.
    BlasMatrixView<Field> submatrix = matrix->Submatrix(
        i + bsize, i + bsize, height - (i + bsize), height - (i + bsize));
    LowerNormalHermitianOuterProduct(Real{-1}, subdiagonal.ToConst(), Real{1},
                                     &submatrix);
  }
  return height;
}

template <class Field>
Int LowerCholeskyFactorization(Int block_size, BlasMatrixView<Field>* matrix) {
  return LowerBlockedCholeskyFactorization(block_size, matrix);
}

#ifdef CATAMARI_HAVE_LAPACK
template <>
inline Int LowerCholeskyFactorization(Int block_size,
                                      BlasMatrixView<float>* matrix) {
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
inline Int LowerCholeskyFactorization(Int block_size,
                                      BlasMatrixView<double>* matrix) {
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

template <>
inline Int LowerCholeskyFactorization(Int block_size,
                                      BlasMatrixView<Complex<float>>* matrix) {
  const char uplo = 'L';
  const BlasInt height_blas = matrix->height;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BlasInt info;
  LAPACK_SYMBOL(cpotrf)
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
inline Int LowerCholeskyFactorization(Int block_size,
                                      BlasMatrixView<Complex<double>>* matrix) {
  const char uplo = 'L';
  const BlasInt height_blas = matrix->height;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BlasInt info;
  LAPACK_SYMBOL(zpotrf)
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
Int LowerUnblockedLDLAdjointFactorization(BlasMatrixView<Field>* matrix) {
  typedef ComplexBase<Field> Real;
  const Int height = matrix->height;
  for (Int i = 0; i < height; ++i) {
    const Real delta = RealPart(matrix->Entry(i, i));
    matrix->Entry(i, i) = delta;
    if (delta == Real{0}) {
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
Int LowerBlockedLDLAdjointFactorization(Int block_size,
                                        BlasMatrixView<Field>* matrix) {
  const Int height = matrix->height;

  Buffer<Field> buffer(std::max(height - block_size, Int(0)) * block_size);
  BlasMatrixView<Field> factor;
  factor.data = buffer.Data();

  for (Int i = 0; i < height; i += block_size) {
    const Int bsize = std::min(height - i, block_size);

    // Overwrite the diagonal block with its LDL' factorization.
    BlasMatrixView<Field> diagonal_block =
        matrix->Submatrix(i, i, bsize, bsize);
    const Int num_diag_pivots =
        LowerUnblockedLDLAdjointFactorization(&diagonal_block);
    if (num_diag_pivots < bsize) {
      return i + num_diag_pivots;
    }
    if (height == i + bsize) {
      break;
    }

    // Solve for the remainder of the block column of L.
    BlasMatrixView<Field> subdiagonal =
        matrix->Submatrix(i + bsize, i, height - (i + bsize), bsize);
    RightLowerAdjointUnitTriangularSolves(diagonal_block.ToConst(),
                                          &subdiagonal);

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
      const ComplexBase<Field> delta = RealPart(diagonal_block(j, j));
      for (Int k = 0; k < subdiagonal.height; ++k) {
        subdiagonal(k, j) /= delta;
      }
    }

    // Perform the Hermitian rank-bsize update.
    BlasMatrixView<Field> submatrix = matrix->Submatrix(
        i + bsize, i + bsize, height - (i + bsize), height - (i + bsize));
    MatrixMultiplyLowerNormalTranspose(Field{-1}, subdiagonal.ToConst(),
                                       factor.ToConst(), Field{1}, &submatrix);
  }
  return height;
}

template <class Field>
Int LowerLDLAdjointFactorization(Int block_size,
                                 BlasMatrixView<Field>* matrix) {
  return LowerBlockedLDLAdjointFactorization(block_size, matrix);
}

template <class Field>
Int LowerUnblockedLDLTransposeFactorization(BlasMatrixView<Field>* matrix) {
  const Int height = matrix->height;
  for (Int i = 0; i < height; ++i) {
    const Field& delta = matrix->Entry(i, i);
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
Int LowerBlockedLDLTransposeFactorization(Int block_size,
                                          BlasMatrixView<Field>* matrix) {
  const Int height = matrix->height;

  Buffer<Field> buffer(std::max(height - block_size, Int(0)) * block_size);
  BlasMatrixView<Field> factor;
  factor.data = buffer.Data();

  for (Int i = 0; i < height; i += block_size) {
    const Int bsize = std::min(height - i, block_size);

    // Overwrite the diagonal block with its LDL' factorization.
    BlasMatrixView<Field> diagonal_block =
        matrix->Submatrix(i, i, bsize, bsize);
    const Int num_diag_pivots =
        LowerUnblockedLDLTransposeFactorization(&diagonal_block);
    if (num_diag_pivots < bsize) {
      return i + num_diag_pivots;
    }
    if (height == i + bsize) {
      break;
    }

    // Solve for the remainder of the block column of L.
    BlasMatrixView<Field> subdiagonal =
        matrix->Submatrix(i + bsize, i, height - (i + bsize), bsize);
    RightLowerTransposeUnitTriangularSolves(diagonal_block.ToConst(),
                                            &subdiagonal);

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
      const Field delta = diagonal_block(j, j);
      for (Int k = 0; k < subdiagonal.height; ++k) {
        subdiagonal(k, j) /= delta;
      }
    }

    // Perform the symmetric rank-bsize update.
    BlasMatrixView<Field> submatrix = matrix->Submatrix(
        i + bsize, i + bsize, height - (i + bsize), height - (i + bsize));
    MatrixMultiplyLowerNormalTranspose(Field{-1}, subdiagonal.ToConst(),
                                       factor.ToConst(), Field{1}, &submatrix);
  }
  return height;
}

template <class Field>
Int LowerLDLTransposeFactorization(Int block_size,
                                   BlasMatrixView<Field>* matrix) {
  return LowerBlockedLDLTransposeFactorization(block_size, matrix);
}

template <class Field>
std::vector<Int> LowerUnblockedFactorAndSampleDPP(bool maximum_likelihood,
                                                  BlasMatrixView<Field>* matrix,
                                                  std::mt19937* generator) {
  typedef ComplexBase<Field> Real;
  const Int height = matrix->height;
  std::vector<Int> sample;
  sample.reserve(height);

  std::uniform_real_distribution<Real> uniform_dist{Real{0}, Real{1}};

#ifdef CATAMARI_DEBUG
  const Real tolerance = 10 * std::numeric_limits<Real>::epsilon();
#endif  // ifdef CATAMARI_DEBUG

  for (Int i = 0; i < height; ++i) {
    Real delta = RealPart(matrix->Entry(i, i));
    CATAMARI_ASSERT(
        delta >= -tolerance && delta <= Real{1} + tolerance,
        "Diagonal value was outside of [0, 1]: " + std::to_string(delta));
    const bool keep_index = maximum_likelihood
                                ? delta >= Real(1) / Real(2)
                                : uniform_dist(*generator) <= delta;
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
std::vector<Int> LowerBlockedFactorAndSampleDPP(Int block_size,
                                                bool maximum_likelihood,
                                                BlasMatrixView<Field>* matrix,
                                                std::mt19937* generator) {
  const Int height = matrix->height;

  std::vector<Int> sample;
  sample.reserve(height);

  Buffer<Field> buffer(std::max(height - block_size, Int(0)) * block_size);
  BlasMatrixView<Field> factor;
  factor.data = buffer.Data();

  for (Int i = 0; i < height; i += block_size) {
    const Int bsize = std::min(height - i, block_size);

    // Overwrite the diagonal block with its LDL' factorization.
    BlasMatrixView<Field> diagonal_block =
        matrix->Submatrix(i, i, bsize, bsize);
    std::vector<Int> block_sample = LowerUnblockedFactorAndSampleDPP(
        maximum_likelihood, &diagonal_block, generator);
    for (const Int& index : block_sample) {
      sample.push_back(i + index);
    }
    if (height == i + bsize) {
      break;
    }

    // Solve for the remainder of the block column of L.
    BlasMatrixView<Field> subdiagonal =
        matrix->Submatrix(i + bsize, i, height - (i + bsize), bsize);
    RightLowerAdjointUnitTriangularSolves(diagonal_block.ToConst(),
                                          &subdiagonal);

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
      const ComplexBase<Field> delta = RealPart(diagonal_block(j, j));
      for (Int k = 0; k < subdiagonal.height; ++k) {
        subdiagonal(k, j) /= delta;
      }
    }

    // Perform the Hermitian rank-bsize update.
    BlasMatrixView<Field> submatrix = matrix->Submatrix(
        i + bsize, i + bsize, height - (i + bsize), height - (i + bsize));
    MatrixMultiplyLowerNormalTranspose(Field{-1}, subdiagonal.ToConst(),
                                       factor.ToConst(), Field{1}, &submatrix);
  }
  return sample;
}

template <class Field>
std::vector<Int> LowerFactorAndSampleDPP(Int block_size,
                                         bool maximum_likelihood,
                                         BlasMatrixView<Field>* matrix,
                                         std::mt19937* generator) {
  return LowerBlockedFactorAndSampleDPP(block_size, maximum_likelihood, matrix,
                                        generator);
}

template <class Field>
std::vector<Int> LowerUnblockedFactorAndSampleNonsymmetricDPP(
    bool maximum_likelihood, BlasMatrixView<Field>* matrix,
    std::mt19937* generator) {
  typedef ComplexBase<Field> Real;
  const Int height = matrix->height;
  CATAMARI_ASSERT(height == matrix->width, "Can only sample square kernels.");
  std::vector<Int> sample;
  sample.reserve(height);

  std::uniform_real_distribution<Real> uniform_dist{Real{0}, Real{1}};

#ifdef CATAMARI_DEBUG
  const Real tolerance = 10 * std::numeric_limits<Real>::epsilon();
#endif  // ifdef CATAMARI_DEBUG

  for (Int i = 0; i < height; ++i) {
    Real delta = RealPart(matrix->Entry(i, i));
    CATAMARI_ASSERT(
        delta >= -tolerance && delta <= Real{1} + tolerance,
        "Diagonal value was outside of [0, 1]: " + std::to_string(delta));
    CATAMARI_ASSERT(std::abs(ImagPart(matrix->Entry(i, i))) <= tolerance,
                    "Imaginary part of diagonal was " +
                        std::to_string(ImagPart(matrix->Entry(i, i))));
    const bool keep_index = maximum_likelihood
                                ? delta >= Real(1) / Real(2)
                                : uniform_dist(*generator) <= delta;
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
      const Field eta = matrix->Entry(i, j);
      for (Int k = i + 1; k < height; ++k) {
        const Field gamma = matrix->Entry(k, i);
        matrix->Entry(k, j) -= gamma * eta;
      }
    }
  }
  return sample;
}

template <class Field>
std::vector<Int> LowerBlockedFactorAndSampleNonsymmetricDPP(
    Int block_size, bool maximum_likelihood, BlasMatrixView<Field>* matrix,
    std::mt19937* generator) {
  const Int height = matrix->height;

  std::vector<Int> sample;
  sample.reserve(height);

  for (Int i = 0; i < height; i += block_size) {
    const Int bsize = std::min(height - i, block_size);

    // Overwrite the diagonal block with its LDL' factorization.
    BlasMatrixView<Field> diagonal_block =
        matrix->Submatrix(i, i, bsize, bsize);
    std::vector<Int> block_sample =
        LowerUnblockedFactorAndSampleNonsymmetricDPP(
            maximum_likelihood, &diagonal_block, generator);
    for (const Int& index : block_sample) {
      sample.push_back(i + index);
    }
    if (height == i + bsize) {
      break;
    }

    // Solve for the remainder of the block column of L.
    BlasMatrixView<Field> subdiagonal =
        matrix->Submatrix(i + bsize, i, height - (i + bsize), bsize);
    RightUpperTriangularSolves(diagonal_block.ToConst(), &subdiagonal);

    // Solve for the remainder of the block row of U.
    BlasMatrixView<Field> superdiagonal =
        matrix->Submatrix(i, i + bsize, bsize, height - (i + bsize));
    LeftLowerUnitTriangularSolves(diagonal_block.ToConst(), &superdiagonal);

    // Perform the rank-bsize update.
    BlasMatrixView<Field> submatrix = matrix->Submatrix(
        i + bsize, i + bsize, height - (i + bsize), height - (i + bsize));
    MatrixMultiplyNormalNormal(Field{-1}, subdiagonal.ToConst(),
                               superdiagonal.ToConst(), Field{1}, &submatrix);
  }
  return sample;
}

template <class Field>
std::vector<Int> LowerFactorAndSampleNonsymmetricDPP(
    Int block_size, bool maximum_likelihood, BlasMatrixView<Field>* matrix,
    std::mt19937* generator) {
  return LowerBlockedFactorAndSampleNonsymmetricDPP(
      block_size, maximum_likelihood, matrix, generator);
}

}  // namespace catamari

#include "catamari/dense_factorizations/openmp-impl.hpp"

#endif  // ifndef CATAMARI_DENSE_FACTORIZATIONS_IMPL_H_
