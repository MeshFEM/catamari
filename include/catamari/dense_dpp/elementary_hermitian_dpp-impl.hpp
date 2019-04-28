/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_DPP_ELEMENTARY_HERMITIAN_DPP_IMPL_H_
#define CATAMARI_DENSE_DPP_ELEMENTARY_HERMITIAN_DPP_IMPL_H_

#include <cmath>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/lapack.hpp"

#include "catamari/dense_dpp.hpp"

namespace catamari {

template <class Field>
void RowSwap(Int index0, Int index1, BlasMatrixView<Field>* matrix) {
  const Int width = matrix->width;
  for (Int j = 0; j < width; ++j) {
    std::swap(matrix->Entry(index0, j), matrix->Entry(index1, j));
  }
}

template <class Field>
void ColumnSwap(Int index0, Int index1, BlasMatrixView<Field>* matrix) {
  const Int height = matrix->height;
  for (Int i = 0; i < height; ++i) {
    std::swap(matrix->Entry(i, index0), matrix->Entry(i, index1));
  }
}

template <class Field>
void LowerHermitianSwap(Int index0, Int index1, BlasMatrixView<Field>* matrix) {
  typedef ComplexBase<Field> Real;
  const Int matrix_size = matrix->height;

  // Ensure that index0 < index1.
  if (index0 > index1) {
    std::swap(index0, index1);
  }
  if (index0 == index1) {
    return;
  }

  // Bottom swap
  const Int bottom_size = matrix_size - (index1 + 1);
  if (bottom_size > 0) {
    BlasMatrixView<Field> matrix_bottom =
        matrix->Submatrix(index1 + 1, 0, bottom_size, matrix_size);
    ColumnSwap(index0, index1, &matrix_bottom);
  }

  // Inner swap
  const Int inner_size = index1 - (index0 + 1);
  if (index0 + 1 < index1) {
    BlasMatrixView<Field> matrix_inner0 =
        matrix->Submatrix(index0 + 1, index0, inner_size, 1);
    BlasMatrixView<Field> matrix_inner1 =
        matrix->Submatrix(index1, index0 + 1, 1, inner_size);

    Field temp;
    for (Int j = 0; j < inner_size; ++j) {
      temp = matrix_inner0(j, 0);
      matrix_inner0(j, 0) = Conjugate(matrix_inner1(0, j));
      matrix_inner1(0, j) = Conjugate(temp);
    }
  }

  // Corner swap
  matrix->Entry(index1, index0) = Conjugate(matrix->Entry(index1, index0));

  // Diagonal swap
  {
    const Real temp = RealPart(matrix->Entry(index1, index1));
    matrix->Entry(index1, index1) = RealPart(matrix->Entry(index0, index0));
    matrix->Entry(index0, index0) = temp;
  }

  // Left swap
  if (index0 > 0) {
    BlasMatrixView<Field> matrix_left =
        matrix->Submatrix(0, 0, matrix_size, index0);
    RowSwap(index0, index1, &matrix_left);
  }
}

namespace elem_herm_dpp {

template <class Field>
Int PanelPivotSelection(Int rel_index, Int rank, bool maximum_likelihood,
                        const ConstBlasMatrixView<Field>& panel,
                        const Buffer<ComplexBase<Field>>& diagonal,
                        std::mt19937* generator) {
  typedef ComplexBase<Field> Real;
  const Int diag_length = diagonal.Size();

  // Sample from the distribution (or pick its maximum).
  if (maximum_likelihood) {
    // Return the index of the maximum value.
    Int max_index = 0;
    Real max_value = -1;
    for (Int i = rel_index; i < diag_length; ++i) {
      if (diagonal[i] > max_value) {
        max_index = i;
        max_value = diagonal[i];
      }
    }
    return max_index;
  } else {
    // Draw a uniform random number in [0, 1].
    std::uniform_real_distribution<Real> uniform_dist{Real{0}, Real{1}};
    const Real target_cdf_value = uniform_dist(*generator);

    // The diagonal sum is guaranteed to be `rank - rel_index`.
    const Real diagonal_sum = rank - rel_index;

    Int sample_index = rel_index;
    for (Real cdf_value = 0; cdf_value < target_cdf_value;) {
      CATAMARI_ASSERT(sample_index < diag_length, "Exceeded panel size");
      cdf_value += diagonal[sample_index] / diagonal_sum;
      if (cdf_value >= target_cdf_value) {
        break;
      }
      ++sample_index;
    }

    return sample_index;
  }
}

template <class Field>
void PanelSampleElementaryLowerHermitianDPP(
    Int panel_offset, Int panel_width, Int rank, bool maximum_likelihood,
    Buffer<Int>* indices, BlasMatrixView<Field>* matrix,
    Buffer<Field>* panel_row, Buffer<ComplexBase<Field>>* diagonal,
    std::mt19937* generator, std::vector<Int>* sample) {
  typedef ComplexBase<Field> Real;
  const Int panel_height = matrix->height - panel_offset;
  const Int rank_remaining = rank - panel_offset;
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

  for (Int rel_index = 0; rel_index < panel_width; ++rel_index) {
    const Int index = panel_offset + rel_index;
    const Int rel_pivot_index = PanelPivotSelection(
        rel_index, rank_remaining, maximum_likelihood, panel, d, generator);
    const Int pivot_index = panel_offset + rel_pivot_index;
    sample->push_back((*indices)[pivot_index]);

    // Perform a Hermitian swap of indices 'index' and 'pivot_index'.
    LowerHermitianSwap(index, pivot_index, matrix);
    std::swap((*indices)[index], (*indices)[pivot_index]);
    std::swap(d[rel_index], d[rel_pivot_index]);

    // matrix(index+1:end, index) -=
    //     matrix(index+1:end, panel_offset:index)
    //     matrix(index, panel_offset:index)'
    for (Int j_rel = 0; j_rel < rel_index; ++j_rel) {
      (*panel_row)[j_rel] = Conjugate(panel(rel_index, j_rel));
    }
    const ConstBlasMatrixView<Field> panel_bottom_left = panel.Submatrix(
        rel_index + 1, 0, panel_height - (rel_index + 1), rel_index);
    MatrixVectorProduct(Field{-1}, panel_bottom_left, panel_row->Data(),
                        matrix->Pointer(index + 1, index));

    // Form the new column:
    //   matrix(index+1:end, index) /= sqrt(alpha11)
    // and update the remainder of the diagonal.
    const Real alpha11_sqrt = std::sqrt(d[rel_index]);
    matrix->Entry(index, index) = alpha11_sqrt;
    const Real alpha11_inv_sqrt = Real(1) / alpha11_sqrt;
    CATAMARI_ASSERT(alpha11_inv_sqrt == alpha11_inv_sqrt,
                    "NaN at pivot " + std::to_string(index));
    for (Int i = index + 1; i < matrix->height; ++i) {
      Field& entry = matrix->Entry(i, index);
      entry *= alpha11_inv_sqrt;
      CATAMARI_ASSERT(entry == entry, "Writing NaN into matrix(" +
                                          std::to_string(i) + ", " +
                                          std::to_string(index) + ").");

      const Int i_rel = i - panel_offset;
      d[i_rel] -= RealPart(entry * Conjugate(entry));
      d[i_rel] = std::max(d[i_rel], Real(0));
    }
  }
}

}  // namespace elem_herm_dpp

template <class Field>
std::vector<Int> BlockedSampleElementaryLowerHermitianDPP(
    Int block_size, Int rank, bool maximum_likelihood,
    BlasMatrixView<Field>* matrix, std::mt19937* generator) {
  typedef ComplexBase<Field> Real;
  const Int height = matrix->height;
  rank = std::min(height, rank);
  block_size = std::min(block_size, rank);

  std::vector<Int> sample;
  sample.reserve(height);

  // For storing the complex conjugate of a single row of the panel to allow
  // for a direct GEMV call.
  Buffer<Field> panel_row(block_size);

  // Initialize the original indices.
  Buffer<Int> indices(height);
  for (Int i = 0; i < height; ++i) {
    indices[i] = i;
  }

  // For storing the diagonal of the Schur complement.
  Buffer<Real> diagonal(height);

  for (Int i = 0; i < height; i += block_size) {
    const Int bsize = std::min(rank - i, block_size);
    diagonal.Resize(height - i);

    elem_herm_dpp::PanelSampleElementaryLowerHermitianDPP(
        i, bsize, rank, maximum_likelihood, &indices, matrix, &panel_row,
        &diagonal, generator, &sample);
    if (rank == i + bsize) {
      break;
    }

    const Int size_left = height - (i + bsize);
    const ConstBlasMatrixView<Field> panel_lower =
        matrix->Submatrix(i + bsize, i, size_left, bsize);
    BlasMatrixView<Field> matrix_bottom_right =
        matrix->Submatrix(i + bsize, i + bsize, size_left, size_left);

    LowerNormalHermitianOuterProduct(Real{-1}, panel_lower, Real{1},
                                     &matrix_bottom_right);
  }

  return sample;
}

template <class Field>
std::vector<Int> SampleElementaryLowerHermitianDPP(
    Int block_size, Int rank, bool maximum_likelihood,
    BlasMatrixView<Field>* matrix, std::mt19937* generator) {
  return BlockedSampleElementaryLowerHermitianDPP(
      block_size, rank, maximum_likelihood, matrix, generator);
}

}  // namespace catamari

#endif  // ifndef CATAMARI_DENSE_DPP_ELEMENTARY_HERMITIAN_DPP_IMPL_H_
