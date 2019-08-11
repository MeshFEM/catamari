/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_FACTORIZATIONS_PIVOTED_LDL_ADJOINT_IMPL_H_
#define CATAMARI_DENSE_FACTORIZATIONS_PIVOTED_LDL_ADJOINT_IMPL_H_

#include <cmath>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/lapack.hpp"

#include "catamari/dense_factorizations.hpp"

// For the RowSwap, ColumnSwap, and LowerHermitianSwap subroutines.
// TODO(Jack Poulson): Expose these routines as utilities.
#include "catamari/dense_dpp.hpp"

namespace catamari {

namespace dense_pivoted_ldl {

// Return the index of the maximum absolute value on the remaining diagonal.
template <class Field>
Int PanelPivotSelection(Int rel_index, const ConstBlasMatrixView<Field>& panel,
                        const Buffer<ComplexBase<Field>>& diagonal) {
  typedef ComplexBase<Field> Real;
  const Int diag_length = diagonal.Size();
  Int max_index = 0;
  Real max_abs = -1;
  for (Int i = rel_index; i < diag_length; ++i) {
    const Real abs_value = std::abs(diagonal[i]);
    if (abs_value > max_abs) {
      max_index = i;
      max_abs = abs_value;
    }
  }
  return max_index;
}

template <class Field>
Int PanelPivotedLDLAdjointFactorization(Int panel_offset, Int panel_width,
                                        BlasMatrixView<Int>* permutation,
                                        BlasMatrixView<Field>* matrix,
                                        Buffer<Field>* panel_row,
                                        Buffer<ComplexBase<Field>>* diagonal) {
  typedef ComplexBase<Field> Real;
  const Int panel_height = matrix->height - panel_offset;
  Buffer<Real>& d = *diagonal;

  const ConstBlasMatrixView<Field> panel =
      matrix->Submatrix(panel_offset, panel_offset, panel_height, panel_width);

  // Ensure there is enough space to store a row of the panel.
  panel_row->Resize(panel_width);

  // Store the diagonal of the remaining submatrix.
  d.Resize(panel_height);
  for (Int rel_index = 0; rel_index < panel_height; ++rel_index) {
    const Int index = panel_offset + rel_index;
    d[rel_index] = RealPart(matrix->Entry(index, index));
  }

  Int num_pivots = 0;
  for (Int rel_index = 0; rel_index < panel_width; ++rel_index) {
    const Int index = panel_offset + rel_index;

    const Int rel_pivot_index = PanelPivotSelection(rel_index, panel, d);
    const Int pivot_index = panel_offset + rel_pivot_index;
    if (std::abs(d[rel_pivot_index]) == Real(0)) {
      break;
    }
    ++num_pivots;

    // Perform a Hermitian swap of indices 'index' and 'pivot_index'.
    LowerHermitianSwap(index, pivot_index, matrix);
    std::swap(permutation->Entry(index), permutation->Entry(pivot_index));
    std::swap(d[rel_index], d[rel_pivot_index]);

    // matrix(index+1:end, index) -=
    //     matrix(index+1:end, panel_offset:index) *
    //     (diag(d(panel_offset:index)) * matrix(index, panel_offset:index)')
    for (Int j_rel = 0; j_rel < rel_index; ++j_rel) {
      (*panel_row)[j_rel] = Conjugate(panel(rel_index, j_rel)) * d[j_rel];
    }
    const ConstBlasMatrixView<Field> panel_bottom_left = panel.Submatrix(
        rel_index + 1, 0, panel_height - (rel_index + 1), rel_index);
    MatrixVectorProduct(Field{-1}, panel_bottom_left, panel_row->Data(),
                        matrix->Pointer(index + 1, index));

    // Form the new column:
    //   matrix(index+1:end, index) /= alpha11
    // and update the remainder of the diagonal.
    const Real alpha11 = d[rel_index];
    matrix->Entry(index, index) = alpha11;
    const Real alpha11_inv = Real(1) / alpha11;
    CATAMARI_ASSERT(alpha11_inv == alpha11_inv,
                    "NaN at pivot " + std::to_string(index));
    for (Int i = index + 1; i < matrix->height; ++i) {
      const Int i_rel = i - panel_offset;
      Field& entry = matrix->Entry(i, index);
      d[i_rel] -= RealPart(entry * alpha11_inv * Conjugate(entry));
      entry *= alpha11_inv;
      CATAMARI_ASSERT(entry == entry, "Writing NaN into matrix(" +
                                          std::to_string(i) + ", " +
                                          std::to_string(index) + ").");
    }
  }

  return num_pivots;
}

}  // namespace dense_pivoted_ldl

template <class Field>
Int BlockedPivotedLDLAdjointFactorization(Int block_size,
                                          BlasMatrixView<Field>* matrix,
                                          BlasMatrixView<Int>* permutation) {
  typedef ComplexBase<Field> Real;
  const Int height = matrix->height;
  block_size = std::min(block_size, height);

  Buffer<Field> buffer(std::max(height - block_size, Int(0)) * block_size);
  BlasMatrixView<Field> factor;
  factor.data = buffer.Data();

  // For storing the complex conjugate of a single row of the panel to allow
  // for a direct GEMV call.
  Buffer<Field> panel_row(block_size);

  // Initialize the original indices.
  CATAMARI_ASSERT(permutation->height == height,
                  "Incorrect permutation height.");
  CATAMARI_ASSERT(permutation->width == 1, "Incorrect permutation width.");
  for (Int i = 0; i < height; ++i) {
    permutation->Entry(i) = i;
  }

  // For storing the diagonal of the Schur complement.
  Buffer<Real> diagonal(height);

  Int num_pivots = 0;
  for (Int i = 0; i < height; i += block_size) {
    const Int bsize = std::min(height - i, block_size);
    diagonal.Resize(height - i);

    const Int num_panel_pivots =
        dense_pivoted_ldl::PanelPivotedLDLAdjointFactorization(
            i, bsize, permutation, matrix, &panel_row, &diagonal);
    num_pivots += num_panel_pivots;
    if (num_panel_pivots != bsize) {
      break;
    }
    if (i + bsize == height) {
      break;
    }

    const Int size_left = height - (i + bsize);
    BlasMatrixView<Field> panel_lower =
        matrix->Submatrix(i + bsize, i, size_left, bsize);
    BlasMatrixView<Field> matrix_bottom_right =
        matrix->Submatrix(i + bsize, i + bsize, size_left, size_left);

    // Copy the conjugate of the current factor and multiply by the diagonal.
    factor.height = panel_lower.height;
    factor.width = panel_lower.width;
    factor.leading_dim = panel_lower.height;
    for (Int j = 0; j < panel_lower.width; ++j) {
      const Real delta = diagonal[j];
      for (Int k = 0; k < panel_lower.height; ++k) {
        factor(k, j) = Conjugate(panel_lower(k, j)) * delta;
      }
    }

    MatrixMultiplyLowerNormalTranspose(Field{-1}, panel_lower.ToConst(),
                                       factor.ToConst(), Field{1},
                                       &matrix_bottom_right);
  }

  return num_pivots;
}

template <class Field>
Int PivotedLDLAdjointFactorization(Int block_size,
                                   BlasMatrixView<Field>* matrix,
                                   BlasMatrixView<Int>* permutation) {
  return BlockedPivotedLDLAdjointFactorization(block_size, matrix, permutation);
}

}  // namespace catamari

#endif  // ifndef CATAMARI_DENSE_FACTORIZATIONS_PIVOTED_LDL_ADJOINT_IMPL_H_
