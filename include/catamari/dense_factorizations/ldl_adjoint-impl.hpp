/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_FACTORIZATIONS_LDL_ADJOINT_IMPL_H_
#define CATAMARI_DENSE_FACTORIZATIONS_LDL_ADJOINT_IMPL_H_

#include <cmath>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/lapack.hpp"

#include "catamari/dense_factorizations.hpp"

namespace catamari {

template <class Field>
Int UnblockedLDLAdjointFactorization(BlasMatrixView<Field>* matrix) {
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
Int UnblockedDynamicallyRegularizedLDLAdjointFactorization(
    const DynamicRegularizationParams<Field>& dynamic_reg_params,
    BlasMatrixView<Field>* matrix,
    std::vector<std::pair<Int, ComplexBase<Field>>>* dynamic_regularization) {
  typedef ComplexBase<Field> Real;
  const Int height = matrix->height;
  const Int offset = dynamic_reg_params.offset;
  const Buffer<bool>& signatures = *dynamic_reg_params.signatures;

  for (Int i = 0; i < height; ++i) {
    Real delta = RealPart(matrix->Entry(i, i));
    if (signatures[i + offset]) {
      // Handle a positive pivot.
      if (delta < dynamic_reg_params.positive_threshold) {
        const Real regularization =
            dynamic_reg_params.positive_threshold - delta;
        dynamic_regularization->emplace_back(offset + i, regularization);
        delta = dynamic_reg_params.positive_threshold;
      }
    } else {
      // Handle a negative pivot.
      if (delta > -dynamic_reg_params.negative_threshold) {
        const Real regularization =
            dynamic_reg_params.negative_threshold - (-delta);
        dynamic_regularization->emplace_back(offset + i, -regularization);
        delta = -dynamic_reg_params.negative_threshold;
      }
    }

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
Int BlockedLDLAdjointFactorization(Int block_size,
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
        UnblockedLDLAdjointFactorization(&diagonal_block);
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

    // Copy the conjugate of the current factor and divide by the diagonal.
    factor.height = subdiagonal.height;
    factor.width = subdiagonal.width;
    factor.leading_dim = subdiagonal.height;
    for (Int j = 0; j < subdiagonal.width; ++j) {
      const ComplexBase<Field> delta = RealPart(diagonal_block(j, j));
      for (Int k = 0; k < subdiagonal.height; ++k) {
        factor(k, j) = Conjugate(subdiagonal(k, j));
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
Int BlockedDynamicallyRegularizedLDLAdjointFactorization(
    Int block_size,
    const DynamicRegularizationParams<Field>& dynamic_reg_params,
    BlasMatrixView<Field>* matrix,
    std::vector<std::pair<Int, ComplexBase<Field>>>* dynamic_regularization) {
  const Int height = matrix->height;

  Buffer<Field> buffer(std::max(height - block_size, Int(0)) * block_size);
  BlasMatrixView<Field> factor;
  factor.data = buffer.Data();

  DynamicRegularizationParams<Field> subparams = dynamic_reg_params;

  for (Int i = 0; i < height; i += block_size) {
    const Int bsize = std::min(height - i, block_size);
    subparams.offset = dynamic_reg_params.offset + i;

    // Overwrite the diagonal block with its LDL' factorization.
    BlasMatrixView<Field> diagonal_block =
        matrix->Submatrix(i, i, bsize, bsize);
    const Int num_diag_pivots =
        UnblockedDynamicallyRegularizedLDLAdjointFactorization(
            subparams, &diagonal_block, dynamic_regularization);
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

    // Copy the conjugate of the current factor and divide by the diagonal.
    factor.height = subdiagonal.height;
    factor.width = subdiagonal.width;
    factor.leading_dim = subdiagonal.height;
    for (Int j = 0; j < subdiagonal.width; ++j) {
      const ComplexBase<Field> delta = RealPart(diagonal_block(j, j));
      for (Int k = 0; k < subdiagonal.height; ++k) {
        factor(k, j) = Conjugate(subdiagonal(k, j));
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
Int LDLAdjointFactorization(Int block_size, BlasMatrixView<Field>* matrix) {
  return BlockedLDLAdjointFactorization(block_size, matrix);
}

template <class Field>
Int DynamicallyRegularizedLDLAdjointFactorization(
    Int block_size,
    const DynamicRegularizationParams<Field>& dynamic_reg_params,
    BlasMatrixView<Field>* matrix,
    std::vector<std::pair<Int, ComplexBase<Field>>>* dynamic_regularization) {
  return BlockedDynamicallyRegularizedLDLAdjointFactorization(
      block_size, dynamic_reg_params, matrix, dynamic_regularization);
}

}  // namespace catamari

#endif  // ifndef CATAMARI_DENSE_FACTORIZATIONS_LDL_ADJOINT_IMPL_H_
