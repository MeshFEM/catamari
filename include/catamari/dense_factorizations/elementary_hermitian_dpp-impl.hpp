/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_FACTORIZATIONS_ELEMENTARY_HERMITIAN_DPP_IMPL_H_
#define CATAMARI_DENSE_FACTORIZATIONS_ELEMENTARY_HERMITIAN_DPP_IMPL_H_

#include <cmath>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/lapack.hpp"

// DO_NOT_SUBMIT
#include "quotient/timer.hpp"

#include "catamari/dense_factorizations.hpp"

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
Int PanelPivotSelection(Int panel_offset, bool maximum_likelihood,
                        const ConstBlasMatrixView<Field>& panel,
                        Buffer<ComplexBase<Field>>* diagonal,
                        std::mt19937* generator) {
  typedef ComplexBase<Field> Real;

  const Int diag_length = diagonal->Size();
  if (panel_offset > 0) {
    // Update the diagonal using the newest column of the panel.
    for (Int i = panel_offset; i < diag_length; ++i) {
      const Field& entry = panel(i, panel_offset - 1);
      CATAMARI_ASSERT(entry == entry,
                      "NaN at index (" + std::to_string(i) + ", " +
                          std::to_string(panel_offset - 1) + ") of panel.");
      (*diagonal)[i] -= RealPart(entry * Conjugate(entry));
      (*diagonal)[i] = std::max((*diagonal)[i], Real(0));
    }
  } else {
    for (Int i = panel_offset; i < diag_length; ++i) {
      (*diagonal)[i] = std::max((*diagonal)[i], Real(0));
    }
  }

  // Sample from the distribution (or pick its maximum).
  if (maximum_likelihood) {
    // Return the index of the maximum value.
    Int max_index = 0;
    Real max_value = -1;
    for (Int i = panel_offset; i < diag_length; ++i) {
      if ((*diagonal)[i] > max_value) {
        max_index = i;
        max_value = (*diagonal)[i];
      }
    }
    return max_index;
  } else {
    // Draw a uniform random number in [0, 1].
    std::uniform_real_distribution<Real> uniform_dist{Real{0}, Real{1}};
    const Real target_cdf_value = uniform_dist(*generator);

    // Compute the sum of the diagonal to normalize the PDF.
    // NOTE: We could compute this value analytically, as it should be the
    // number of remaining indices to sample.
    Real diagonal_sum = 0;
    for (Int i = panel_offset; i < diag_length; ++i) {
      diagonal_sum += (*diagonal)[i];
    }

    Int sample_index = panel_offset;
    for (Real cdf_value = 0; cdf_value < target_cdf_value;) {
      CATAMARI_ASSERT(sample_index < diag_length, "Exceeded panel size");
      cdf_value += (*diagonal)[sample_index] / diagonal_sum;
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
    Int panel_offset, bool maximum_likelihood, Buffer<Int>* indices,
    BlasMatrixView<Field>* matrix, BlasMatrixView<Field>* panel,
    Buffer<ComplexBase<Field>>* diagonal, Buffer<Field>* panel_row,
    std::mt19937* generator, std::vector<Int>* sample) {
  typedef ComplexBase<Field> Real;
  const Int panel_height = panel->height;
  const Int panel_width = panel->width;

  // Store the diagonal of the remaining submatrix.
  diagonal->Resize(panel_height);
  for (Int rel_index = 0; rel_index < panel_height; ++rel_index) {
    const Int index = panel_offset + rel_index;
    (*diagonal)[rel_index] = RealPart(matrix->Entry(index, index));
  }

  panel_row->Resize(panel_width);

  for (Int rel_index = 0; rel_index < panel_width; ++rel_index) {
    const Int index = panel_offset + rel_index;
    const Int rel_pivot_index = PanelPivotSelection(
        rel_index, maximum_likelihood, panel->ToConst(), diagonal, generator);
    const Int pivot_index = panel_offset + rel_pivot_index;
    sample->push_back((*indices)[pivot_index]);

    // Perform a Hermitian swap of indices 'index' and 'pivot_index'.
    LowerHermitianSwap(index, pivot_index, matrix);
    std::swap((*indices)[index], (*indices)[pivot_index]);
    std::swap((*diagonal)[rel_index], (*diagonal)[rel_pivot_index]);
    RowSwap(rel_index, rel_pivot_index, panel);

    // matrix(index:end, index) -=
    //     panel(rel_index:end, 0:rel_index) panel(rel_index, 0:rel_index)'
    for (Int j_rel = 0; j_rel < rel_index; ++j_rel) {
      (*panel_row)[j_rel] = Conjugate(panel->Entry(rel_index, j_rel));
    }
    const ConstBlasMatrixView<Field> panel_bottom_left =
        panel->Submatrix(rel_index, 0, panel_height - rel_index, rel_index);
    MatrixVectorProduct(Field{-1}, panel_bottom_left, panel_row->Data(),
                        matrix->Pointer(index, index));

    // matrix(index+1:end, index:index+1) /= sqrt(alpha11)
    const Real alpha11_sqrt = std::sqrt(RealPart(matrix->Entry(index, index)));
    matrix->Entry(index, index) = alpha11_sqrt;
    const Real alpha11_inv_sqrt = Real(1) / alpha11_sqrt;
    CATAMARI_ASSERT(alpha11_inv_sqrt == alpha11_inv_sqrt,
                    "NaN at pivot " + std::to_string(index));
    for (Int i_rel = rel_index + 1; i_rel < panel_height; ++i_rel) {
      const Int i = panel_offset + i_rel;
      matrix->Entry(i, index) *= alpha11_inv_sqrt;
      panel->Entry(i_rel, rel_index) = matrix->Entry(i, index);
      CATAMARI_ASSERT(matrix->Entry(i, index) == matrix->Entry(i, index),
                      "Writing NaN into matrix(" + std::to_string(i) + ", " +
                          std::to_string(index) + ").");
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

  // Set up a buffer for the largest possible panel.
  Buffer<Field> panel_buffer(height * block_size);
  BlasMatrixView<Field> panel;
  panel.data = panel_buffer.Data();

  // Initialize the original indices.
  Buffer<Int> indices(height);
  for (Int i = 0; i < height; ++i) {
    indices[i] = i;
  }

  Buffer<Real> diagonal(height);
  Buffer<Field> panel_row(block_size);

  quotient::Timer panel_timer, outer_prod_timer;

  for (Int i = 0; i < height; i += block_size) {
    const Int bsize = std::min(rank - i, block_size);
    panel.height = height - i;
    panel.width = bsize;
    panel.leading_dim = panel.height;
    diagonal.Resize(panel.height);

    panel_timer.Start();
    elem_herm_dpp::PanelSampleElementaryLowerHermitianDPP(
        i, maximum_likelihood, &indices, matrix, &panel, &diagonal, &panel_row,
        generator, &sample);
    panel_timer.Stop();
    if (rank == i + bsize) {
      break;
    }

    const Int size_left = height - (i + bsize);
    const ConstBlasMatrixView<Field> panel_lower =
        panel.Submatrix(bsize, 0, size_left, bsize);
    BlasMatrixView<Field> matrix_bottom_right =
        matrix->Submatrix(i + bsize, i + bsize, size_left, size_left);

    outer_prod_timer.Start();
    LowerNormalHermitianOuterProduct(Real{-1}, panel_lower, Real{1},
                                     &matrix_bottom_right);
    outer_prod_timer.Stop();
  }

  std::cout << "panel:      " << panel_timer.TotalSeconds() << " seconds.\n"
            << "outer-prod: " << outer_prod_timer.TotalSeconds() << " seconds."
            << std::endl;

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

#endif  // ifndef CATAMARI_DENSE_FACTORIZATIONS_ELEMENTARY_HERMITIAN_DPP_IMPL_H_
