/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_DPP_NONHERMITIAN_DPP_OPENMP_IMPL_H_
#define CATAMARI_DENSE_DPP_NONHERMITIAN_DPP_OPENMP_IMPL_H_
#ifdef CATAMARI_OPENMP

#include <cmath>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/lapack.hpp"

#include "catamari/dense_dpp.hpp"

namespace catamari {

template <class Field>
std::vector<Int> OpenMPBlockedSampleNonHermitianDPP(
    Int tile_size, Int block_size, bool maximum_likelihood,
    BlasMatrixView<Field>* matrix, std::mt19937* generator) {
  const Int height = matrix->height;

  std::vector<Int> sample;
  sample.reserve(height);

  // For use in tracking dependencies.
  const Int leading_dim CATAMARI_UNUSED = matrix->leading_dim;
  Field* const matrix_data CATAMARI_UNUSED = matrix->data;

  std::vector<Int> block_sample;

  std::vector<Int>* sample_ptr = &sample;
  std::vector<Int>* block_sample_ptr = &block_sample;

  #pragma omp taskgroup
  for (Int i = 0; i < height; i += tile_size) {
    const Int tsize = std::min(height - i, tile_size);

    // Overwrite the diagonal block with its LDL' factorization.
    BlasMatrixView<Field> diagonal_block =
        matrix->Submatrix(i, i, tsize, tsize);
    #pragma omp taskgroup
    #pragma omp task default(none)                                      \
        firstprivate(block_size, i, diagonal_block, maximum_likelihood, \
            generator, sample_ptr, block_sample_ptr)                    \
        depend(inout: matrix_data[i + i * leading_dim])
    {
      *block_sample_ptr = BlockedSampleNonHermitianDPP(
          block_size, maximum_likelihood, &diagonal_block, generator);
      for (const Int& index : *block_sample_ptr) {
        sample_ptr->push_back(i + index);
      }
    }
    if (height == i + tsize) {
      break;
    }
    const ConstBlasMatrixView<Field> const_diagonal_block = diagonal_block;

    // Solve for the remainder of the block column of L.
    for (Int i_sub = i + tsize; i_sub < height; i_sub += tile_size) {
      #pragma omp task default(none)                                  \
          firstprivate(i, i_sub, matrix, const_diagonal_block, tsize) \
          depend(inout: matrix_data[i_sub + i * leading_dim])
      {
        // Solve agains the upper-triangle of the diagonal block.
        const Int tsize_solve = std::min(height - i_sub, tsize);
        BlasMatrixView<Field> subdiagonal_block =
            matrix->Submatrix(i_sub, i, tsize_solve, tsize);
        RightUpperTriangularSolves(const_diagonal_block, &subdiagonal_block);
      }
    }

    // Solve for the remainder of the block row of U.
    for (Int i_sub = i + tsize; i_sub < height; i_sub += tile_size) {
      #pragma omp task default(none)                                  \
          firstprivate(i, i_sub, matrix, const_diagonal_block, tsize) \
          depend(inout: matrix_data[i + i_sub * leading_dim])
      {
        // Solve agains the unit lower-triangle of the diagonal block.
        const Int tsize_solve = std::min(height - i_sub, tsize);
        BlasMatrixView<Field> superdiagonal_block =
            matrix->Submatrix(i, i_sub, tsize, tsize_solve);
        LeftLowerUnitTriangularSolves(const_diagonal_block,
                                      &superdiagonal_block);
      }
    }

    // Perform the rank-tsize update.
    for (Int j_sub = i + tsize; j_sub < height; j_sub += tile_size) {
      for (Int i_sub = i + tsize; i_sub < height; i_sub += tile_size) {
        #pragma omp task default(none)                     \
          firstprivate(i, i_sub, j_sub, matrix, tsize)     \
          depend(in: matrix_data[i_sub + i * leading_dim], \
              matrix_data[i + j_sub * leading_dim])        \
          depend(inout: matrix_data[i_sub + j_sub * leading_dim])
        {
          const Int row_tsize = std::min(height - i_sub, tsize);
          const Int column_tsize = std::min(height - j_sub, tsize);
          const ConstBlasMatrixView<Field> row_block =
              matrix->Submatrix(i_sub, i, row_tsize, tsize).ToConst();
          const ConstBlasMatrixView<Field> column_block =
              matrix->Submatrix(i, j_sub, tsize, column_tsize);
          BlasMatrixView<Field> update_block =
              matrix->Submatrix(i_sub, j_sub, row_tsize, column_tsize);
          MatrixMultiplyNormalNormal(Field{-1}, row_block, column_block,
                                     Field{1}, &update_block);
        }
      }
    }
  }

  return sample;
}

template <class Field>
std::vector<Int> OpenMPSampleNonHermitianDPP(Int tile_size, Int block_size,
                                             bool maximum_likelihood,
                                             BlasMatrixView<Field>* matrix,
                                             std::mt19937* generator) {
  return OpenMPBlockedSampleNonHermitianDPP(
      tile_size, block_size, maximum_likelihood, matrix, generator);
}

}  // namespace catamari

#endif  // ifdef CATAMARI_OPENMP
#endif  // ifndef CATAMARI_DENSE_DPP_NONHERMITIAN_DPP_OPENMP_IMPL_H_
