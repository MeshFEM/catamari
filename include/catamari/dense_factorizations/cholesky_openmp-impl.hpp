/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_FACTORIZATIONS_CHOLESKY_OPENMP_IMPL_H_
#define CATAMARI_DENSE_FACTORIZATIONS_CHOLESKY_OPENMP_IMPL_H_
#ifdef CATAMARI_OPENMP

#include <cmath>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/lapack.hpp"

#include "catamari/dense_factorizations.hpp"

namespace catamari {

template <class Field>
Int OpenMPLowerCholeskyFactorization(Int tile_size, Int block_size,
                                     BlasMatrixView<Field>* matrix) {
  typedef ComplexBase<Field> Real;
  const Int height = matrix->height;

  // For use in tracking dependencies.
  Field* const matrix_data CATAMARI_UNUSED = matrix->data;
  const Int leading_dim CATAMARI_UNUSED = matrix->leading_dim;

  Int num_pivots = 0;
  #pragma omp taskgroup
  for (Int i = 0; i < height; i += tile_size) {
    const Int tsize = std::min(height - i, tile_size);

    // Overwrite the diagonal block with its Cholesky factor.
    BlasMatrixView<Field> diagonal_block =
        matrix->Submatrix(i, i, tsize, tsize);
    bool failed_pivot = false;
    #pragma omp taskgroup
    #pragma omp task default(none)               \
        firstprivate(block_size, diagonal_block) \
        shared(num_pivots, failed_pivot)         \
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
    const ConstBlasMatrixView<Field> const_diagonal_block = diagonal_block;

    // Solve for the remainder of the block column of L.
    for (Int i_sub = i + tsize; i_sub < height; i_sub += tile_size) {
      #pragma omp task default(none)                          \
          firstprivate(height, i, i_sub, matrix, leading_dim, \
              const_diagonal_block, tsize)                    \
          depend(in: matrix_data[i + i * leading_dim])        \
          depend(inout: matrix_data[i_sub + i * leading_dim])
      {
        const Int tsize_solve = std::min(height - i_sub, tsize);
        BlasMatrixView<Field> subdiagonal_block =
            matrix->Submatrix(i_sub, i, tsize_solve, tsize);
        RightLowerAdjointTriangularSolves(const_diagonal_block,
                                          &subdiagonal_block);
      }
    }

    // Perform the Hermitian rank-bsize update.
    for (Int j_sub = i + tsize; j_sub < height; j_sub += tile_size) {
      #pragma omp task default(none)                       \
          firstprivate(height, i, j_sub, matrix, tsize)    \
          depend(in: matrix_data[j_sub + i * leading_dim]) \
          depend(inout: matrix_data[j_sub + j_sub * leading_dim])
      {
        const Int column_tsize = std::min(height - j_sub, tsize);
        const ConstBlasMatrixView<Field> column_block =
            matrix->Submatrix(j_sub, i, column_tsize, tsize).ToConst();
        BlasMatrixView<Field> update_block =
            matrix->Submatrix(j_sub, j_sub, column_tsize, column_tsize);
        LowerNormalHermitianOuterProduct(Real{-1}, column_block, Real{1},
                                         &update_block);
      }

      for (Int i_sub = j_sub + tsize; i_sub < height; i_sub += tile_size) {
        #pragma omp task default(none)                         \
          firstprivate(height, i, i_sub, j_sub, matrix, tsize) \
          depend(in: matrix_data[i_sub + i * leading_dim])     \
          depend(in: matrix_data[j_sub + i * leading_dim])     \
          depend(inout: matrix_data[i_sub + j_sub * leading_dim])
        {
          const Int row_tsize = std::min(height - i_sub, tsize);
          const Int column_tsize = std::min(height - j_sub, tsize);
          const ConstBlasMatrixView<Field> row_block =
              matrix->Submatrix(i_sub, i, row_tsize, tsize).ToConst();
          const ConstBlasMatrixView<Field> column_block =
              matrix->Submatrix(j_sub, i, column_tsize, tsize).ToConst();
          BlasMatrixView<Field> update_block =
              matrix->Submatrix(i_sub, j_sub, row_tsize, column_tsize);
          MatrixMultiplyNormalAdjoint(Field{-1}, row_block, column_block,
                                      Field{1}, &update_block);
        }
      }
    }
  }

  return num_pivots;
}

template <class Field>
Int OpenMPDynamicallyRegularizedLowerCholeskyFactorization(
    Int tile_size, Int block_size,
    const DynamicRegularizationParams<Field>& dynamic_reg_params,
    BlasMatrixView<Field>* matrix,
    std::vector<std::pair<Int, ComplexBase<Field>>>* dynamic_regularization) {
  typedef ComplexBase<Field> Real;
  const Int height = matrix->height;

  // For use in tracking dependencies.
  Field* const matrix_data CATAMARI_UNUSED = matrix->data;
  const Int leading_dim CATAMARI_UNUSED = matrix->leading_dim;

  DynamicRegularizationParams<Field> subparams = dynamic_reg_params;

  Int num_pivots = 0;
  #pragma omp taskgroup
  for (Int i = 0; i < height; i += tile_size) {
    const Int tsize = std::min(height - i, tile_size);

    // Overwrite the diagonal block with its Cholesky factor.
    BlasMatrixView<Field> diagonal_block =
        matrix->Submatrix(i, i, tsize, tsize);
    bool failed_pivot = false;
    subparams.offset = dynamic_reg_params.offset + i;
    #pragma omp taskgroup
    #pragma omp task default(none)                                       \
        firstprivate(block_size, diagonal_block, dynamic_regularization) \
        shared(num_pivots, failed_pivot, subparams)                      \
        depend(inout: matrix_data[i + i * leading_dim])
    {
      const Int num_diag_pivots =
          DynamicallyRegularizedLowerCholeskyFactorization(
              block_size, subparams, &diagonal_block, dynamic_regularization);
      num_pivots += num_diag_pivots;
      if (num_diag_pivots < diagonal_block.height) {
        failed_pivot = true;
      }
    }
    if (failed_pivot || height == i + tsize) {
      break;
    }
    const ConstBlasMatrixView<Field> const_diagonal_block = diagonal_block;

    // Solve for the remainder of the block column of L.
    for (Int i_sub = i + tsize; i_sub < height; i_sub += tile_size) {
      #pragma omp task default(none)                          \
          firstprivate(height, i, i_sub, matrix, leading_dim, \
              const_diagonal_block, tsize)                    \
          depend(in: matrix_data[i + i * leading_dim])        \
          depend(inout: matrix_data[i_sub + i * leading_dim])
      {
        const Int tsize_solve = std::min(height - i_sub, tsize);
        BlasMatrixView<Field> subdiagonal_block =
            matrix->Submatrix(i_sub, i, tsize_solve, tsize);
        RightLowerAdjointTriangularSolves(const_diagonal_block,
                                          &subdiagonal_block);
      }
    }

    // Perform the Hermitian rank-bsize update.
    for (Int j_sub = i + tsize; j_sub < height; j_sub += tile_size) {
      #pragma omp task default(none)                       \
          firstprivate(height, i, j_sub, matrix, tsize)    \
          depend(in: matrix_data[j_sub + i * leading_dim]) \
          depend(inout: matrix_data[j_sub + j_sub * leading_dim])
      {
        const Int column_tsize = std::min(height - j_sub, tsize);
        const ConstBlasMatrixView<Field> column_block =
            matrix->Submatrix(j_sub, i, column_tsize, tsize).ToConst();
        BlasMatrixView<Field> update_block =
            matrix->Submatrix(j_sub, j_sub, column_tsize, column_tsize);
        LowerNormalHermitianOuterProduct(Real{-1}, column_block, Real{1},
                                         &update_block);
      }

      for (Int i_sub = j_sub + tsize; i_sub < height; i_sub += tile_size) {
        #pragma omp task default(none)                         \
          firstprivate(height, i, i_sub, j_sub, matrix, tsize) \
          depend(in: matrix_data[i_sub + i * leading_dim])     \
          depend(in: matrix_data[j_sub + i * leading_dim])     \
          depend(inout: matrix_data[i_sub + j_sub * leading_dim])
        {
          const Int row_tsize = std::min(height - i_sub, tsize);
          const Int column_tsize = std::min(height - j_sub, tsize);
          const ConstBlasMatrixView<Field> row_block =
              matrix->Submatrix(i_sub, i, row_tsize, tsize).ToConst();
          const ConstBlasMatrixView<Field> column_block =
              matrix->Submatrix(j_sub, i, column_tsize, tsize).ToConst();
          BlasMatrixView<Field> update_block =
              matrix->Submatrix(i_sub, j_sub, row_tsize, column_tsize);
          MatrixMultiplyNormalAdjoint(Field{-1}, row_block, column_block,
                                      Field{1}, &update_block);
        }
      }
    }
  }

  return num_pivots;
}

}  // namespace catamari

#endif  // ifdef CATAMARI_OPENMP
#endif  // ifndef CATAMARI_DENSE_FACTORIZATIONS_CHOLESKY_OPENMP_IMPL_H_
