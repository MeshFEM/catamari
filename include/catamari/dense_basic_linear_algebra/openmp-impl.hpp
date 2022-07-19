/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_BASIC_LINEAR_ALGEBRA_OPENMP_IMPL_H_
#define CATAMARI_DENSE_BASIC_LINEAR_ALGEBRA_OPENMP_IMPL_H_
#ifdef CATAMARI_OPENMP

#include "catamari/blas.hpp"
#include "catamari/macros.hpp"

#include "catamari/dense_basic_linear_algebra.hpp"

namespace catamari {

template <class Field>
void OpenMPMatrixMultiplyLowerNormalNormal(
    Int tile_size, const Field& alpha,
    const ConstBlasMatrixView<Field>& left_matrix,
    const ConstBlasMatrixView<Field>& right_matrix, const Field& beta,
    BlasMatrixView<Field>* output_matrix) {
  const Int height = output_matrix->height;
  const Int rank = left_matrix.width;
  const Field alpha_copy = alpha;
  const Field beta_copy = beta;
  const ConstBlasMatrixView<Field> left_matrix_copy = left_matrix;
  const ConstBlasMatrixView<Field> right_matrix_copy = right_matrix;
  CATAMARI_ASSERT(left_matrix.width == right_matrix.height,
                  "Contraction dimensions do not match.");

  #pragma omp taskgroup
  for (Int j = 0; j < height; j += tile_size) {
    #pragma omp task default(none)                                \
       firstprivate(tile_size, j, height, rank, left_matrix_copy, \
           right_matrix_copy, output_matrix, alpha_copy, beta_copy)
    {
      const Int tsize = std::min(height - j, tile_size);
      const ConstBlasMatrixView<Field> row_block =
          left_matrix_copy.Submatrix(j, 0, tsize, rank);
      const ConstBlasMatrixView<Field> column_block =
          right_matrix_copy.Submatrix(0, j, rank, tsize);
      BlasMatrixView<Field> output_block =
          output_matrix->Submatrix(j, j, tsize, tsize);
      MatrixMultiplyLowerNormalNormal(alpha_copy, row_block, column_block,
                                      beta_copy, &output_block);
    }

    for (Int i = j + tile_size; i < height; i += tile_size) {
      #pragma omp task default(none)                                    \
          firstprivate(tile_size, i, j, height, rank, left_matrix_copy, \
              right_matrix_copy, output_matrix, alpha_copy, beta_copy)
      {
        const Int row_tsize = std::min(height - i, tile_size);
        const Int column_tsize = std::min(height - j, tile_size);
        const ConstBlasMatrixView<Field> row_block =
            left_matrix_copy.Submatrix(i, 0, row_tsize, rank);
        const ConstBlasMatrixView<Field> column_block =
            right_matrix_copy.Submatrix(0, j, rank, column_tsize);
        BlasMatrixView<Field> output_block =
            output_matrix->Submatrix(i, j, row_tsize, column_tsize);
        MatrixMultiplyNormalNormal(alpha_copy, row_block, column_block,
                                   beta_copy, &output_block);
      }
    }
  }
}

template <class Field>
void OpenMPLowerNormalHermitianOuterProduct(
    Int tile_size, const ComplexBase<Field>& alpha,
    const ConstBlasMatrixView<Field>& left_matrix,
    const ComplexBase<Field>& beta, BlasMatrixView<Field>* output_matrix) {
  typedef ComplexBase<Field> Real;
  const Int height = output_matrix->height;
  const Int rank = left_matrix.width;
  const Real alpha_copy = alpha;
  const Real beta_copy = beta;
  const ConstBlasMatrixView<Field> left_matrix_copy = left_matrix;

  #pragma omp taskgroup
  for (Int j = 0; j < height; j += tile_size) {
    #pragma omp task default(none)                                                \
        firstprivate(tile_size, j, height, rank, left_matrix_copy, output_matrix, \
            alpha_copy, beta_copy)
    {
      const Int tsize = std::min(height - j, tile_size);
      const ConstBlasMatrixView<Field> column_block = left_matrix_copy.Submatrix(j, 0, tsize, rank);
      BlasMatrixView<Field> output_block            =   output_matrix->Submatrix(j, j, tsize, tsize);
      LowerNormalHermitianOuterProduct(alpha_copy, column_block, beta_copy, &output_block);
    }

    for (Int i = j + tile_size; i < height; i += tile_size) {
      #pragma omp task default(none)                                                   \
          firstprivate(tile_size, i, j, height, rank, left_matrix_copy, output_matrix, \
              alpha_copy, beta_copy)
      {
        const Int   row_tsize  = std::min(height - i, tile_size);
        const Int column_tsize = std::min(height - j, tile_size);
        const ConstBlasMatrixView<Field> row_block    = left_matrix_copy.Submatrix(i, 0,    row_tsize, rank);
        const ConstBlasMatrixView<Field> column_block = left_matrix_copy.Submatrix(j, 0, column_tsize, rank);
        BlasMatrixView<Field>            output_block = output_matrix->  Submatrix(i, j, row_tsize, column_tsize);
        MatrixMultiplyNormalAdjoint(Field{alpha_copy}, row_block, column_block, Field{beta_copy}, &output_block);
      }
    }
  }
}

}  // namespace catamari

#endif  // ifdef CATAMARI_OPENMP
#endif  // ifndef CATAMARI_DENSE_BASIC_LINEAR_ALGEBRA_OPENMP_IMPL_H_
