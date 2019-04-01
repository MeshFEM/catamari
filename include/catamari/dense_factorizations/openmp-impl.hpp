/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_FACTORIZATIONS_OPENMP_IMPL_H_
#define CATAMARI_DENSE_FACTORIZATIONS_OPENMP_IMPL_H_
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
Int OpenMPLowerLDLAdjointFactorization(Int tile_size, Int block_size,
                                       BlasMatrixView<Field>* matrix,
                                       Buffer<Field>* buffer) {
  const Int height = matrix->height;
  if (buffer->Size() < static_cast<std::size_t>(height * height)) {
    buffer->Resize(height * height);
  }

  BlasMatrixView<Field> factor;
  factor.height = height;
  factor.width = height;
  factor.leading_dim = height;
  factor.data = buffer->Data();

  // For use in tracking dependencies.
  const Int leading_dim CATAMARI_UNUSED = matrix->leading_dim;
  Field* const matrix_data CATAMARI_UNUSED = matrix->data;

  Int num_pivots = 0;
  #pragma omp taskgroup
  for (Int i = 0; i < height; i += tile_size) {
    const Int tsize = std::min(height - i, tile_size);

    // Overwrite the diagonal block with its LDL' factorization.
    BlasMatrixView<Field> diagonal_block =
        matrix->Submatrix(i, i, tsize, tsize);
    bool failed_pivot = false;
    #pragma omp taskgroup
    #pragma omp task default(none) firstprivate(block_size, diagonal_block) \
        shared(num_pivots, failed_pivot)                                    \
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
    const ConstBlasMatrixView<Field> const_diagonal_block = diagonal_block;

    // Solve for the remainder of the block column of L and then simultaneously
    // copy its conjugate into 'factor' and the solve against the diagonal.
    for (Int i_sub = i + tsize; i_sub < height; i_sub += tile_size) {
      #pragma omp task default(none)                                  \
          firstprivate(i, i_sub, matrix, const_diagonal_block, tsize) \
          shared(factor)                                              \
          depend(inout: matrix_data[i_sub + i * leading_dim])
      {
        // Solve agains the unit lower-triangle of the diagonal block.
        const Int tsize_solve = std::min(height - i_sub, tsize);
        BlasMatrixView<Field> subdiagonal_block =
            matrix->Submatrix(i_sub, i, tsize_solve, tsize);
        RightLowerAdjointUnitTriangularSolves(const_diagonal_block,
                                              &subdiagonal_block);

        // Copy the conjugate into 'factor' and solve against the diagonal.
        BlasMatrixView<Field> factor_block =
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
      #pragma omp task default(none) firstprivate(i, j_sub, matrix, tsize) \
          shared(factor)                                                   \
          depend(in: matrix_data[j_sub + i * leading_dim])                 \
          depend(inout: matrix_data[j_sub + j_sub * leading_dim])
      {
        const Int column_tsize = std::min(height - j_sub, tsize);
        const ConstBlasMatrixView<Field> row_block =
            matrix->Submatrix(j_sub, i, column_tsize, tsize).ToConst();
        const ConstBlasMatrixView<Field> column_block =
            factor.Submatrix(j_sub, i, column_tsize, tsize);
        BlasMatrixView<Field> update_block =
            matrix->Submatrix(j_sub, j_sub, column_tsize, column_tsize);
        MatrixMultiplyLowerNormalTranspose(Field{-1}, row_block, column_block,
                                           Field{1}, &update_block);
      }

      for (Int i_sub = j_sub + tsize; i_sub < height; i_sub += tile_size) {
        #pragma omp task default(none)                     \
          firstprivate(i, i_sub, j_sub, matrix, tsize)     \
          shared(factor)                                   \
          depend(in: matrix_data[i_sub + i * leading_dim], \
              matrix_data[j_sub + i * leading_dim])        \
          depend(inout: matrix_data[i_sub + j_sub * leading_dim])
        {
          const Int row_tsize = std::min(height - i_sub, tsize);
          const Int column_tsize = std::min(height - j_sub, tsize);
          const ConstBlasMatrixView<Field> row_block =
              matrix->Submatrix(i_sub, i, row_tsize, tsize).ToConst();
          const ConstBlasMatrixView<Field> column_block =
              factor.Submatrix(j_sub, i, column_tsize, tsize);
          BlasMatrixView<Field> update_block =
              matrix->Submatrix(i_sub, j_sub, row_tsize, column_tsize);
          MatrixMultiplyNormalTranspose(Field{-1}, row_block, column_block,
                                        Field{1}, &update_block);
        }
      }
    }
  }

  return num_pivots;
}

template <class Field>
Int OpenMPLowerLDLTransposeFactorization(Int tile_size, Int block_size,
                                         BlasMatrixView<Field>* matrix,
                                         Buffer<Field>* buffer) {
  const Int height = matrix->height;
  if (buffer->Size() < static_cast<std::size_t>(height * height)) {
    buffer->Resize(height * height);
  }

  BlasMatrixView<Field> factor;
  factor.height = height;
  factor.width = height;
  factor.leading_dim = height;
  factor.data = buffer->Data();

  BlasMatrixView<Field>* factor_ptr = &factor;

  // For use in tracking dependencies.
  const Int leading_dim CATAMARI_UNUSED = matrix->leading_dim;
  Field* const matrix_data CATAMARI_UNUSED = matrix->data;

  Int num_pivots = 0;
  #pragma omp taskgroup
  for (Int i = 0; i < height; i += tile_size) {
    const Int tsize = std::min(height - i, tile_size);

    // Overwrite the diagonal block with its LDL^T factorization.
    BlasMatrixView<Field> diagonal_block =
        matrix->Submatrix(i, i, tsize, tsize);
    bool failed_pivot = false;
    #pragma omp taskgroup
    #pragma omp task default(none) firstprivate(block_size, diagonal_block) \
        shared(num_pivots, failed_pivot)                                    \
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
    const ConstBlasMatrixView<Field> const_diagonal_block = diagonal_block;

    // Solve for the remainder of the block column of L and then simultaneously
    // copy its conjugate into 'factor' and the solve against the diagonal.
    for (Int i_sub = i + tsize; i_sub < height; i_sub += tile_size) {
      #pragma omp task default(none)                                  \
          firstprivate(i, i_sub, matrix, const_diagonal_block, tsize, \
              factor_ptr)                                             \
          depend(inout: matrix_data[i_sub + i * leading_dim])
      {
        // Solve agains the unit lower-triangle of the diagonal block.
        const Int tsize_solve = std::min(height - i_sub, tsize);
        BlasMatrixView<Field> subdiagonal_block =
            matrix->Submatrix(i_sub, i, tsize_solve, tsize);
        RightLowerTransposeUnitTriangularSolves(const_diagonal_block,
                                                &subdiagonal_block);

        // Copy into 'factor' and solve against the diagonal.
        BlasMatrixView<Field> factor_block =
            factor_ptr->Submatrix(i_sub, i, tsize_solve, tsize);
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
      #pragma omp task default(none)                        \
          firstprivate(i, j_sub, matrix, tsize, factor_ptr) \
          depend(in: matrix_data[j_sub + i * leading_dim])  \
          depend(inout: matrix_data[j_sub + j_sub * leading_dim])
      {
        const Int column_tsize = std::min(height - j_sub, tsize);
        const ConstBlasMatrixView<Field> row_block =
            matrix->Submatrix(j_sub, i, column_tsize, tsize).ToConst();
        const ConstBlasMatrixView<Field> column_block =
            factor_ptr->Submatrix(j_sub, i, column_tsize, tsize);
        BlasMatrixView<Field> update_block =
            matrix->Submatrix(j_sub, j_sub, column_tsize, column_tsize);
        MatrixMultiplyLowerNormalTranspose(Field{-1}, row_block, column_block,
                                           Field{1}, &update_block);
      }

      for (Int i_sub = j_sub + tsize; i_sub < height; i_sub += tile_size) {
        #pragma omp task default(none)                             \
          firstprivate(i, i_sub, j_sub, matrix, tsize, factor_ptr) \
          depend(in: matrix_data[i_sub + i * leading_dim],         \
              matrix_data[j_sub + i * leading_dim])                \
          depend(inout: matrix_data[i_sub + j_sub * leading_dim])
        {
          const Int row_tsize = std::min(height - i_sub, tsize);
          const Int column_tsize = std::min(height - j_sub, tsize);
          const ConstBlasMatrixView<Field> row_block =
              matrix->Submatrix(i_sub, i, row_tsize, tsize).ToConst();
          const ConstBlasMatrixView<Field> column_block =
              factor_ptr->Submatrix(j_sub, i, column_tsize, tsize);
          BlasMatrixView<Field> update_block =
              matrix->Submatrix(i_sub, j_sub, row_tsize, column_tsize);
          MatrixMultiplyNormalTranspose(Field{-1}, row_block, column_block,
                                        Field{1}, &update_block);
        }
      }
    }
  }

  return num_pivots;
}

template <class Field>
std::vector<Int> OpenMPLowerBlockedFactorAndSampleDPP(
    Int tile_size, Int block_size, bool maximum_likelihood,
    BlasMatrixView<Field>* matrix, std::mt19937* generator,
    Buffer<Field>* buffer) {
  const Int height = matrix->height;
  if (buffer->Size() < static_cast<std::size_t>(height * height)) {
    buffer->Resize(height * height);
  }

  std::vector<Int> sample;
  sample.reserve(height);

  BlasMatrixView<Field> factor;
  factor.height = height;
  factor.width = height;
  factor.leading_dim = height;
  factor.data = buffer->Data();

  BlasMatrixView<Field>* factor_ptr = &factor;

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
      *block_sample_ptr = LowerBlockedFactorAndSampleDPP(
          block_size, maximum_likelihood, &diagonal_block, generator);
      for (const Int& index : *block_sample_ptr) {
        sample_ptr->push_back(i + index);
      }
    }
    if (height == i + tsize) {
      break;
    }
    const ConstBlasMatrixView<Field> const_diagonal_block = diagonal_block;

    // Solve for the remainder of the block column of L and then simultaneously
    // copy its conjugate into 'factor' and the solve against the diagonal.
    for (Int i_sub = i + tsize; i_sub < height; i_sub += tile_size) {
      #pragma omp task default(none)                                  \
          firstprivate(i, i_sub, matrix, const_diagonal_block, tsize, \
              factor_ptr)                                             \
          depend(inout: matrix_data[i_sub + i * leading_dim])
      {
        // Solve agains the unit lower-triangle of the diagonal block.
        const Int tsize_solve = std::min(height - i_sub, tsize);
        BlasMatrixView<Field> subdiagonal_block =
            matrix->Submatrix(i_sub, i, tsize_solve, tsize);
        RightLowerAdjointUnitTriangularSolves(const_diagonal_block,
                                              &subdiagonal_block);

        // Copy the conjugate into 'factor' and solve against the diagonal.
        BlasMatrixView<Field> factor_block =
            factor_ptr->Submatrix(i_sub, i, tsize_solve, tsize);
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
      #pragma omp task default(none)                        \
          firstprivate(i, j_sub, matrix, tsize, factor_ptr) \
          depend(in: matrix_data[j_sub + i * leading_dim])  \
          depend(inout: matrix_data[j_sub + j_sub * leading_dim])
      {
        const Int column_tsize = std::min(height - j_sub, tsize);
        const ConstBlasMatrixView<Field> row_block =
            matrix->Submatrix(j_sub, i, column_tsize, tsize).ToConst();
        const ConstBlasMatrixView<Field> column_block =
            factor_ptr->Submatrix(j_sub, i, column_tsize, tsize);
        BlasMatrixView<Field> update_block =
            matrix->Submatrix(j_sub, j_sub, column_tsize, column_tsize);
        MatrixMultiplyLowerNormalTranspose(Field{-1}, row_block, column_block,
                                           Field{1}, &update_block);
      }

      for (Int i_sub = j_sub + tsize; i_sub < height; i_sub += tile_size) {
        #pragma omp task default(none)                             \
          firstprivate(i, i_sub, j_sub, matrix, tsize, factor_ptr) \
          depend(in: matrix_data[i_sub + i * leading_dim],         \
              matrix_data[j_sub + i * leading_dim])                \
          depend(inout: matrix_data[i_sub + j_sub * leading_dim])
        {
          const Int row_tsize = std::min(height - i_sub, tsize);
          const Int column_tsize = std::min(height - j_sub, tsize);
          const ConstBlasMatrixView<Field> row_block =
              matrix->Submatrix(i_sub, i, row_tsize, tsize).ToConst();
          const ConstBlasMatrixView<Field> column_block =
              factor_ptr->Submatrix(j_sub, i, column_tsize, tsize);
          BlasMatrixView<Field> update_block =
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
std::vector<Int> OpenMPLowerFactorAndSampleDPP(Int tile_size, Int block_size,
                                               bool maximum_likelihood,
                                               BlasMatrixView<Field>* matrix,
                                               std::mt19937* generator,
                                               Buffer<Field>* buffer) {
  return OpenMPLowerBlockedFactorAndSampleDPP(
      tile_size, block_size, maximum_likelihood, matrix, generator, buffer);
}

template <class Field>
std::vector<Int> OpenMPLowerBlockedFactorAndSampleNonsymmetricDPP(
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
      *block_sample_ptr = LowerBlockedFactorAndSampleNonsymmetricDPP(
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
std::vector<Int> OpenMPLowerFactorAndSampleNonsymmetricDPP(
    Int tile_size, Int block_size, bool maximum_likelihood,
    BlasMatrixView<Field>* matrix, std::mt19937* generator) {
  return OpenMPLowerBlockedFactorAndSampleNonsymmetricDPP(
      tile_size, block_size, maximum_likelihood, matrix, generator);
}

}  // namespace catamari

#endif  // ifdef CATAMARI_OPENMP
#endif  // ifndef CATAMARI_DENSE_FACTORIZATIONS_OPENMP_IMPL_H_
