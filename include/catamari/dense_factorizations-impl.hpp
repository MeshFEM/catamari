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
Int LowerBlockedCholeskyFactorization(Int block_size,
                                      BlasMatrix<Field>* matrix) {
  typedef ComplexBase<Field> Real;
  const Int height = matrix->height;
  for (Int i = 0; i < height; i += block_size) {
    const Int bsize = std::min(height - i, block_size);

    // Overwrite the diagonal block with its Cholesky factor.
    BlasMatrix<Field> diagonal_block = matrix->Submatrix(i, i, bsize, bsize);
    const Int num_diag_pivots =
        LowerUnblockedCholeskyFactorization(&diagonal_block);
    if (num_diag_pivots < bsize) {
      return i + num_diag_pivots;
    }
    if (height == i + bsize) {
      break;
    }

    // Solve for the remainder of the block column of L.
    BlasMatrix<Field> subdiagonal =
        matrix->Submatrix(i + bsize, i, height - (i + bsize), bsize);
    RightLowerAdjointTriangularSolves(diagonal_block.ToConst(), &subdiagonal);

    // Perform the Hermitian rank-bsize update.
    BlasMatrix<Field> submatrix = matrix->Submatrix(
        i + bsize, i + bsize, height - (i + bsize), height - (i + bsize));
    LowerNormalHermitianOuterProduct(Real{-1}, subdiagonal.ToConst(), Real{1},
                                     &submatrix);
  }
  return height;
}

template <class Field>
Int LowerCholeskyFactorization(Int block_size, BlasMatrix<Field>* matrix) {
  return LowerBlockedCholeskyFactorization(block_size, matrix);
}

#ifdef CATAMARI_HAVE_LAPACK
template <>
inline Int LowerCholeskyFactorization(Int block_size,
                                      BlasMatrix<float>* matrix) {
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
                                      BlasMatrix<double>* matrix) {
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
                                      BlasMatrix<Complex<float>>* matrix) {
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
                                      BlasMatrix<Complex<double>>* matrix) {
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

#ifdef _OPENMP
template <class Field>
Int MultithreadedLowerCholeskyFactorization(Int tile_size, Int block_size,
                                            BlasMatrix<Field>* matrix) {
  typedef ComplexBase<Field> Real;
  const Int height = matrix->height;

  // For use in tracking dependencies.
  Field* const matrix_data CATAMARI_UNUSED = matrix->data;
  const Int leading_dim CATAMARI_UNUSED = matrix->leading_dim;

  Int num_pivots = 0;
  for (Int i = 0; i < height; i += tile_size) {
    const Int tsize = std::min(height - i, tile_size);

    // Overwrite the diagonal block with its Cholesky factor.
    BlasMatrix<Field> diagonal_block = matrix->Submatrix(i, i, tsize, tsize);
    bool failed_pivot = false;
    #pragma omp taskgroup
    #pragma omp task default(none) firstprivate(block_size, diagonal_block) \
        shared(num_pivots, failed_pivot)                                    \
        depend(inout: matrix_data[i + i * leading_dim])
    {
      const Int num_diag_pivots =
          LowerCholeskyFactorization(block_size, &diagonal_block);
      num_pivots += num_diag_pivots;
      if (num_diag_pivots < diagonal_block.height) {
        failed_pivot = true;
      }
    }
    if (failed_pivot || height == i + tsize) {
      break;
    }
    const ConstBlasMatrix<Field> const_diagonal_block = diagonal_block;

    // Solve for the remainder of the block column of L.
    for (Int i_sub = i + tsize; i_sub < height; i_sub += tile_size) {
      #pragma omp task default(none)                                  \
          firstprivate(i, i_sub, matrix, const_diagonal_block, tsize) \
          depend(inout: matrix_data[i_sub + i * leading_dim])
      {
        const Int tsize_solve = std::min(height - i_sub, tsize);
        BlasMatrix<Field> subdiagonal_block =
            matrix->Submatrix(i_sub, i, tsize_solve, tsize);
        RightLowerAdjointTriangularSolves(const_diagonal_block,
                                          &subdiagonal_block);
      }
    }

    // Perform the Hermitian rank-bsize update.
    for (Int j_sub = i + tsize; j_sub < height; j_sub += tile_size) {
      #pragma omp task default(none)                       \
          firstprivate(i, j_sub, matrix, tsize)            \
          depend(in: matrix_data[j_sub + i * leading_dim]) \
          depend(inout: matrix_data[j_sub + j_sub * leading_dim])
      {
        const Int column_tsize = std::min(height - j_sub, tsize);
        const ConstBlasMatrix<Field> column_block =
            matrix->Submatrix(j_sub, i, column_tsize, tsize).ToConst();
        BlasMatrix<Field> update_block =
            matrix->Submatrix(j_sub, j_sub, column_tsize, column_tsize);
        LowerNormalHermitianOuterProduct(Real{-1}, column_block, Real{1},
                                         &update_block);
      }

      for (Int i_sub = j_sub + tsize; i_sub < height; i_sub += tile_size) {
        #pragma omp task default(none)                     \
          firstprivate(i, i_sub, j_sub, matrix, tsize)     \
          depend(in: matrix_data[i_sub + i * leading_dim], \
              matrix_data[j_sub + i * leading_dim])        \
          depend(inout: matrix_data[i_sub + j_sub * leading_dim])
        {
          const Int row_tsize = std::min(height - i_sub, tsize);
          const Int column_tsize = std::min(height - j_sub, tsize);
          const ConstBlasMatrix<Field> row_block =
              matrix->Submatrix(i_sub, i, row_tsize, tsize).ToConst();
          const ConstBlasMatrix<Field> column_block =
              matrix->Submatrix(j_sub, i, column_tsize, tsize).ToConst();
          BlasMatrix<Field> update_block =
              matrix->Submatrix(i_sub, j_sub, row_tsize, column_tsize);
          MatrixMultiplyNormalAdjoint(Field{-1}, row_block, column_block,
                                      Field{1}, &update_block);
        }
      }
    }
  }
  return num_pivots;
}
#endif  // ifdef _OPENMP

template <class Field>
Int LowerUnblockedLDLAdjointFactorization(BlasMatrix<Field>* matrix) {
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
                                        BlasMatrix<Field>* matrix) {
  const Int height = matrix->height;

  Buffer<Field> buffer(std::max(height - block_size, Int(0)) * block_size);
  BlasMatrix<Field> factor;
  factor.data = buffer.Data();

  for (Int i = 0; i < height; i += block_size) {
    const Int bsize = std::min(height - i, block_size);

    // Overwrite the diagonal block with its LDL' factorization.
    BlasMatrix<Field> diagonal_block = matrix->Submatrix(i, i, bsize, bsize);
    const Int num_diag_pivots =
        LowerUnblockedLDLAdjointFactorization(&diagonal_block);
    if (num_diag_pivots < bsize) {
      return i + num_diag_pivots;
    }
    if (height == i + bsize) {
      break;
    }

    // Solve for the remainder of the block column of L.
    BlasMatrix<Field> subdiagonal =
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
    BlasMatrix<Field> submatrix = matrix->Submatrix(
        i + bsize, i + bsize, height - (i + bsize), height - (i + bsize));
    MatrixMultiplyLowerNormalTranspose(Field{-1}, subdiagonal.ToConst(),
                                       factor.ToConst(), Field{1}, &submatrix);
  }
  return height;
}

template <class Field>
Int LowerLDLAdjointFactorization(Int block_size, BlasMatrix<Field>* matrix) {
  return LowerBlockedLDLAdjointFactorization(block_size, matrix);
}

#ifdef _OPENMP
template <class Field>
Int MultithreadedLowerLDLAdjointFactorization(Int tile_size, Int block_size,
                                              BlasMatrix<Field>* matrix,
                                              Buffer<Field>* buffer) {
  const Int height = matrix->height;
  if (buffer->Size() < height * height) {
    buffer->Resize(height * height);
  }

  BlasMatrix<Field> factor;
  factor.height = height;
  factor.width = height;
  factor.leading_dim = height;
  factor.data = buffer->Data();

  // For use in tracking dependencies.
  const Int leading_dim CATAMARI_UNUSED = matrix->leading_dim;
  Field* const matrix_data CATAMARI_UNUSED = matrix->data;

  Int num_pivots = 0;
  for (Int i = 0; i < height; i += tile_size) {
    const Int tsize = std::min(height - i, tile_size);

    // Overwrite the diagonal block with its LDL' factorization.
    BlasMatrix<Field> diagonal_block = matrix->Submatrix(i, i, tsize, tsize);
    bool failed_pivot = false;
    #pragma omp taskgroup
    #pragma omp task default(none)               \
        firstprivate(block_size, diagonal_block) \
        shared(num_pivots, failed_pivot) \
        depend(inout: matrix_data[i + i * leading_dim])
    {
      const Int num_diag_pivots =
          LowerLDLAdjointFactorization(block_size, &diagonal_block);
      num_pivots += num_diag_pivots;
      if (num_diag_pivots < diagonal_block.height) {
        failed_pivot = true;
      }
    }
    if (failed_pivot || height == i + tsize) {
      break;
    }
    const ConstBlasMatrix<Field> const_diagonal_block = diagonal_block;

    // Solve for the remainder of the block column of L and then simultaneously
    // copy its conjugate into 'factor' and the solve against the diagonal.
    for (Int i_sub = i + tsize; i_sub < height; i_sub += tile_size) {
      #pragma omp task default(none)                          \
          firstprivate(i, i_sub, const_diagonal_block, tsize) \
          shared(matrix, factor)                              \
          depend(inout: matrix_data[i_sub + i * leading_dim])
      {
        // Solve agains the unit lower-triangle of the diagonal block.
        const Int tsize_solve = std::min(height - i_sub, tsize);
        BlasMatrix<Field> subdiagonal_block =
            matrix->Submatrix(i_sub, i, tsize_solve, tsize);
        RightLowerAdjointUnitTriangularSolves(const_diagonal_block,
                                              &subdiagonal_block);

        // Copy the conjugate into 'factor' and solve against the diagonal.
        BlasMatrix<Field> factor_block =
            factor.Submatrix(i_sub, i, tsize_solve, tsize);
        for (Int j = 0; j < subdiagonal_block.width; ++j) {
          const ComplexBase<Field> delta = RealPart(const_diagonal_block(j, j));
          for (Int k = 0; k < subdiagonal_block.height; ++k) {
            factor_block(k, j) = Conjugate(subdiagonal_block(k, j));
            subdiagonal_block(k, j) /= delta;
          }
        }
      }
    }

    // Perform the Hermitian rank-bsize update.
    for (Int j_sub = i + tsize; j_sub < height; j_sub += tile_size) {
      #pragma omp task default(none)                       \
          firstprivate(i, j_sub, tsize)                    \
          shared(matrix, factor)                           \
          depend(in: matrix_data[j_sub + i * leading_dim]) \
          depend(inout: matrix_data[j_sub + j_sub * leading_dim])
      {
        const Int column_tsize = std::min(height - j_sub, tsize);
        const ConstBlasMatrix<Field> row_block =
            matrix->Submatrix(j_sub, i, column_tsize, tsize).ToConst();
        const ConstBlasMatrix<Field> column_block =
            factor.Submatrix(j_sub, i, column_tsize, tsize);
        BlasMatrix<Field> update_block =
            matrix->Submatrix(j_sub, j_sub, column_tsize, column_tsize);
        MatrixMultiplyLowerNormalTranspose(Field{-1}, row_block, column_block,
                                           Field{1}, &update_block);
      }

      for (Int i_sub = j_sub + tsize; i_sub < height; i_sub += tile_size) {
        #pragma omp task default(none)                     \
          firstprivate(i, i_sub, j_sub, tsize)             \
          shared(matrix, factor)                           \
          depend(in: matrix_data[i_sub + i * leading_dim], \
              matrix_data[j_sub + i * leading_dim])        \
          depend(inout: matrix_data[i_sub + j_sub * leading_dim])
        {
          const Int row_tsize = std::min(height - i_sub, tsize);
          const Int column_tsize = std::min(height - j_sub, tsize);
          const ConstBlasMatrix<Field> row_block =
              matrix->Submatrix(i_sub, i, row_tsize, tsize).ToConst();
          const ConstBlasMatrix<Field> column_block =
              factor.Submatrix(j_sub, i, column_tsize, tsize);
          BlasMatrix<Field> update_block =
              matrix->Submatrix(i_sub, j_sub, row_tsize, column_tsize);
          MatrixMultiplyNormalTranspose(Field{-1}, row_block, column_block,
                                        Field{1}, &update_block);
        }
      }
    }
  }
  return num_pivots;
}
#endif  // ifdef _OPENMP

template <class Field>
Int LowerUnblockedLDLTransposeFactorization(BlasMatrix<Field>* matrix) {
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
                                          BlasMatrix<Field>* matrix) {
  const Int height = matrix->height;

  Buffer<Field> buffer(std::max(height - block_size, Int(0)) * block_size);
  BlasMatrix<Field> factor;
  factor.data = buffer.Data();

  for (Int i = 0; i < height; i += block_size) {
    const Int bsize = std::min(height - i, block_size);

    // Overwrite the diagonal block with its LDL' factorization.
    BlasMatrix<Field> diagonal_block = matrix->Submatrix(i, i, bsize, bsize);
    const Int num_diag_pivots =
        LowerUnblockedLDLTransposeFactorization(&diagonal_block);
    if (num_diag_pivots < bsize) {
      return i + num_diag_pivots;
    }
    if (height == i + bsize) {
      break;
    }

    // Solve for the remainder of the block column of L.
    BlasMatrix<Field> subdiagonal =
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
    BlasMatrix<Field> submatrix = matrix->Submatrix(
        i + bsize, i + bsize, height - (i + bsize), height - (i + bsize));
    MatrixMultiplyLowerNormalTranspose(Field{-1}, subdiagonal.ToConst(),
                                       factor.ToConst(), Field{1}, &submatrix);
  }
  return height;
}

template <class Field>
Int LowerLDLTransposeFactorization(Int block_size, BlasMatrix<Field>* matrix) {
  return LowerBlockedLDLTransposeFactorization(block_size, matrix);
}

#ifdef _OPENMP
template <class Field>
Int MultithreadedLowerLDLTransposeFactorization(Int tile_size, Int block_size,
                                                BlasMatrix<Field>* matrix,
                                                Buffer<Field>* buffer) {
  const Int height = matrix->height;
  if (buffer->Size() < height * height) {
    buffer->Resize(height * height);
  }

  BlasMatrix<Field> factor;
  factor.height = height;
  factor.width = height;
  factor.leading_dim = height;
  factor.data = buffer->Data();

  // For use in tracking dependencies.
  const Int leading_dim CATAMARI_UNUSED = matrix->leading_dim;
  Field* const matrix_data CATAMARI_UNUSED = matrix->data;

  Int num_pivots = 0;
  for (Int i = 0; i < height; i += tile_size) {
    const Int tsize = std::min(height - i, tile_size);

    // Overwrite the diagonal block with its LDL^T factorization.
    BlasMatrix<Field> diagonal_block = matrix->Submatrix(i, i, tsize, tsize);
    bool failed_pivot = false;
    #pragma omp taskgroup
    #pragma omp task default(none)               \
        firstprivate(block_size, diagonal_block) \
        shared(num_pivots, failed_pivot)         \
        depend(inout: matrix_data[i + i * leading_dim])
    {
      const Int num_diag_pivots =
          LowerLDLTransposeFactorization(block_size, &diagonal_block);
      num_pivots += num_diag_pivots;
      if (num_diag_pivots < diagonal_block.height) {
        failed_pivot = true;
      }
    }
    if (failed_pivot || height == i + tsize) {
      break;
    }
    const ConstBlasMatrix<Field> const_diagonal_block = diagonal_block;

    // Solve for the remainder of the block column of L and then simultaneously
    // copy its conjugate into 'factor' and the solve against the diagonal.
    for (Int i_sub = i + tsize; i_sub < height; i_sub += tile_size) {
      #pragma omp task default(none)                          \
          firstprivate(i, i_sub, const_diagonal_block, tsize) \
          shared(matrix, factor)                              \
          depend(inout: matrix_data[i_sub + i * leading_dim])
      {
        // Solve agains the unit lower-triangle of the diagonal block.
        const Int tsize_solve = std::min(height - i_sub, tsize);
        BlasMatrix<Field> subdiagonal_block =
            matrix->Submatrix(i_sub, i, tsize_solve, tsize);
        RightLowerTransposeUnitTriangularSolves(const_diagonal_block,
                                                &subdiagonal_block);

        // Copy into 'factor' and solve against the diagonal.
        BlasMatrix<Field> factor_block =
            factor.Submatrix(i_sub, i, tsize_solve, tsize);
        for (Int j = 0; j < subdiagonal_block.width; ++j) {
          const Field delta = const_diagonal_block(j, j);
          for (Int k = 0; k < subdiagonal_block.height; ++k) {
            factor_block(k, j) = subdiagonal_block(k, j);
            subdiagonal_block(k, j) /= delta;
          }
        }
      }
    }

    // Perform the Hermitian rank-tsize update.
    for (Int j_sub = i + tsize; j_sub < height; j_sub += tile_size) {
      #pragma omp task default(none)                       \
          firstprivate(i, j_sub, tsize)                    \
          shared(matrix, factor)                           \
          depend(in: matrix_data[j_sub + i * leading_dim]) \
          depend(inout: matrix_data[j_sub + j_sub * leading_dim])
      {
        const Int column_tsize = std::min(height - j_sub, tsize);
        const ConstBlasMatrix<Field> row_block =
            matrix->Submatrix(j_sub, i, column_tsize, tsize).ToConst();
        const ConstBlasMatrix<Field> column_block =
            factor.Submatrix(j_sub, i, column_tsize, tsize);
        BlasMatrix<Field> update_block =
            matrix->Submatrix(j_sub, j_sub, column_tsize, column_tsize);
        MatrixMultiplyLowerNormalTranspose(Field{-1}, row_block, column_block,
                                           Field{1}, &update_block);
      }

      for (Int i_sub = j_sub + tsize; i_sub < height; i_sub += tile_size) {
        #pragma omp task default(none)                     \
          firstprivate(i, i_sub, j_sub, tsize)             \
          shared(matrix, factor)                           \
          depend(in: matrix_data[i_sub + i * leading_dim], \
              matrix_data[j_sub + i * leading_dim])        \
          depend(inout: matrix_data[i_sub + j_sub * leading_dim])
        {
          const Int row_tsize = std::min(height - i_sub, tsize);
          const Int column_tsize = std::min(height - j_sub, tsize);
          const ConstBlasMatrix<Field> row_block =
              matrix->Submatrix(i_sub, i, row_tsize, tsize).ToConst();
          const ConstBlasMatrix<Field> column_block =
              factor.Submatrix(j_sub, i, column_tsize, tsize);
          BlasMatrix<Field> update_block =
              matrix->Submatrix(i_sub, j_sub, row_tsize, column_tsize);
          MatrixMultiplyNormalTranspose(Field{-1}, row_block, column_block,
                                        Field{1}, &update_block);
        }
      }
    }
  }
  return num_pivots;
}
#endif  // ifdef _OPENMP

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
    Int block_size, bool maximum_likelihood, BlasMatrix<Field>* matrix,
    std::mt19937* generator,
    std::uniform_real_distribution<ComplexBase<Field>>* uniform_dist) {
  const Int height = matrix->height;

  std::vector<Int> sample;
  sample.reserve(height);

  Buffer<Field> buffer(std::max(height - block_size, Int(0)) * block_size);
  BlasMatrix<Field> factor;
  factor.data = buffer.Data();

  for (Int i = 0; i < height; i += block_size) {
    const Int bsize = std::min(height - i, block_size);

    // Overwrite the diagonal block with its LDL' factorization.
    BlasMatrix<Field> diagonal_block = matrix->Submatrix(i, i, bsize, bsize);
    std::vector<Int> block_sample = LowerUnblockedFactorAndSampleDPP(
        maximum_likelihood, &diagonal_block, generator, uniform_dist);
    for (const Int& index : block_sample) {
      sample.push_back(i + index);
    }
    if (height == i + bsize) {
      break;
    }

    // Solve for the remainder of the block column of L.
    BlasMatrix<Field> subdiagonal =
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
    BlasMatrix<Field> submatrix = matrix->Submatrix(
        i + bsize, i + bsize, height - (i + bsize), height - (i + bsize));
    MatrixMultiplyLowerNormalTranspose(Field{-1}, subdiagonal.ToConst(),
                                       factor.ToConst(), Field{1}, &submatrix);
  }
  return sample;
}

template <class Field>
std::vector<Int> LowerFactorAndSampleDPP(
    Int block_size, bool maximum_likelihood, BlasMatrix<Field>* matrix,
    std::mt19937* generator,
    std::uniform_real_distribution<ComplexBase<Field>>* uniform_dist) {
  return LowerBlockedFactorAndSampleDPP(block_size, maximum_likelihood, matrix,
                                        generator, uniform_dist);
}

#ifdef _OPENMP
template <class Field>
std::vector<Int> MultithreadedLowerBlockedFactorAndSampleDPP(
    Int tile_size, Int block_size, bool maximum_likelihood,
    BlasMatrix<Field>* matrix, std::mt19937* generator,
    std::uniform_real_distribution<ComplexBase<Field>>* uniform_dist,
    Buffer<Field>* buffer) {
  const Int height = matrix->height;
  if (buffer->Size() < height * height) {
    buffer->Resize(height * height);
  }

  std::vector<Int> sample;
  sample.reserve(height);

  BlasMatrix<Field> factor;
  factor.height = height;
  factor.width = height;
  factor.leading_dim = height;
  factor.data = buffer->Data();

  // For use in tracking dependencies.
  const Int leading_dim CATAMARI_UNUSED = matrix->leading_dim;
  Field* const matrix_data CATAMARI_UNUSED = matrix->data;

  std::vector<Int> block_sample;

  std::vector<Int>* sample_ptr = &sample;
  std::vector<Int>* block_sample_ptr = &block_sample;

  for (Int i = 0; i < height; i += tile_size) {
    const Int tsize = std::min(height - i, tile_size);

    // Overwrite the diagonal block with its LDL' factorization.
    BlasMatrix<Field> diagonal_block = matrix->Submatrix(i, i, tsize, tsize);
    #pragma omp taskgroup
    #pragma omp task default(none)                                      \
        firstprivate(block_size, i, diagonal_block, maximum_likelihood, \
            generator, uniform_dist, sample_ptr, block_sample_ptr)      \
        depend(inout: matrix_data[i + i * leading_dim])
    {
      *block_sample_ptr = LowerBlockedFactorAndSampleDPP(
          block_size, maximum_likelihood, &diagonal_block, generator,
          uniform_dist);
      for (const Int& index : *block_sample_ptr) {
        sample_ptr->push_back(i + index);
      }
    }
    if (height == i + tsize) {
      break;
    }
    const ConstBlasMatrix<Field> const_diagonal_block = diagonal_block;

    // Solve for the remainder of the block column of L and then simultaneously
    // copy its conjugate into 'factor' and the solve against the diagonal.
    for (Int i_sub = i + tsize; i_sub < height; i_sub += tile_size) {
      #pragma omp task default(none)                          \
          firstprivate(i, i_sub, const_diagonal_block, tsize) \
          shared(matrix, factor)                              \
          depend(inout: matrix_data[i_sub + i * leading_dim])
      {
        // Solve agains the unit lower-triangle of the diagonal block.
        const Int tsize_solve = std::min(height - i_sub, tsize);
        BlasMatrix<Field> subdiagonal_block =
            matrix->Submatrix(i_sub, i, tsize_solve, tsize);
        RightLowerAdjointUnitTriangularSolves(const_diagonal_block,
                                              &subdiagonal_block);

        // Copy the conjugate into 'factor' and solve against the diagonal.
        BlasMatrix<Field> factor_block =
            factor.Submatrix(i_sub, i, tsize_solve, tsize);
        for (Int j = 0; j < subdiagonal_block.width; ++j) {
          const ComplexBase<Field> delta = RealPart(const_diagonal_block(j, j));
          for (Int k = 0; k < subdiagonal_block.height; ++k) {
            factor_block(k, j) = Conjugate(subdiagonal_block(k, j));
            subdiagonal_block(k, j) /= delta;
          }
        }
      }
    }

    // Perform the Hermitian rank-tsize update.
    for (Int j_sub = i + tsize; j_sub < height; j_sub += tile_size) {
      #pragma omp task default(none)                       \
          firstprivate(i, j_sub, tsize)                    \
          shared(matrix, factor)                           \
          depend(in: matrix_data[j_sub + i * leading_dim]) \
          depend(inout: matrix_data[j_sub + j_sub * leading_dim])
      {
        const Int column_tsize = std::min(height - j_sub, tsize);
        const ConstBlasMatrix<Field> row_block =
            matrix->Submatrix(j_sub, i, column_tsize, tsize).ToConst();
        const ConstBlasMatrix<Field> column_block =
            factor.Submatrix(j_sub, i, column_tsize, tsize);
        BlasMatrix<Field> update_block =
            matrix->Submatrix(j_sub, j_sub, column_tsize, column_tsize);
        MatrixMultiplyLowerNormalTranspose(Field{-1}, row_block, column_block,
                                           Field{1}, &update_block);
      }

      for (Int i_sub = j_sub + tsize; i_sub < height; i_sub += tile_size) {
        #pragma omp task default(none)                     \
          firstprivate(i, i_sub, j_sub, tsize)             \
          shared(matrix, factor)                           \
          depend(in: matrix_data[i_sub + i * leading_dim], \
              matrix_data[j_sub + i * leading_dim])        \
          depend(inout: matrix_data[i_sub + j_sub * leading_dim])
        {
          const Int row_tsize = std::min(height - i_sub, tsize);
          const Int column_tsize = std::min(height - j_sub, tsize);
          const ConstBlasMatrix<Field> row_block =
              matrix->Submatrix(i_sub, i, row_tsize, tsize).ToConst();
          const ConstBlasMatrix<Field> column_block =
              factor.Submatrix(j_sub, i, column_tsize, tsize);
          BlasMatrix<Field> update_block =
              matrix->Submatrix(i_sub, j_sub, row_tsize, column_tsize);
          MatrixMultiplyNormalTranspose(Field{-1}, row_block, column_block,
                                        Field{1}, &update_block);
        }
      }
    }
  }
  return sample;
}

template <class Field>
std::vector<Int> MultithreadedLowerFactorAndSampleDPP(
    Int tile_size, Int block_size, bool maximum_likelihood,
    BlasMatrix<Field>* matrix, std::mt19937* generator,
    std::uniform_real_distribution<ComplexBase<Field>>* uniform_dist,
    Buffer<Field>* buffer) {
  return MultithreadedLowerBlockedFactorAndSampleDPP(
      tile_size, block_size, maximum_likelihood, matrix, generator,
      uniform_dist, buffer);
}
#endif  // ifdef _OPENMP

}  // namespace catamari

#endif  // ifndef CATAMARI_DENSE_FACTORIZATIONS_IMPL_H_
