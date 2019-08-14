/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_EQUILIBRATE_SYMMETRIC_MATRIX_IMPL_H_
#define CATAMARI_EQUILIBRATE_SYMMETRIC_MATRIX_IMPL_H_

#include <limits>

#include "catamari/equilibrate_symmetric_matrix.hpp"

namespace catamari {

template <typename Field>
void EquilibrateSymmetricMatrix(CoordinateMatrix<Field>* matrix,
                                BlasMatrix<ComplexBase<Field>>* scaling,
                                bool verbose) {
  typedef ComplexBase<Field> Real;
  const Int num_rows = matrix->NumRows();
  CATAMARI_ASSERT(num_rows == matrix->NumColumns(), "Expected square matrix.");

  scaling->Resize(num_rows, 1, Real(1));

  // Compute the maximum and minimum absolute value in each column.
  Buffer<Real> column_max_abs(num_rows, Real(0));
  Buffer<Real> new_scaling(num_rows);
  for (const auto& entry : matrix->Entries()) {
    const Int j = entry.column;
    const Real value_abs = std::abs(entry.value);
    column_max_abs[j] = std::max(column_max_abs[j], value_abs);

  }
  for (Int i = 0; i < num_rows; ++i) {
    new_scaling[i] = std::sqrt(column_max_abs[i]);
  }

  // Perform the rescaling iterations.
  const Int kNumPasses = 1;
  for (Int pass = 0; pass < kNumPasses; ++pass) {
    if (verbose) {
      // Find the largest and smallest absolute values.
      const Real max_abs =
          *std::max_element(column_max_abs.begin(), column_max_abs.end());
      std::cout << "Pass " << pass << ":\n"
                << "  max_abs: " << max_abs << std::endl;
    }

    // Rescale the problem using the row geometric means and update the
    // minimum and maximum absolute values.
    column_max_abs.Resize(num_rows, Real(0));
    for (auto& entry : matrix->Entries()) {
      const Int i = entry.row;
      const Int j = entry.column;
      entry.value /= new_scaling[i] * new_scaling[j];
      const Real abs_value = std::abs(entry.value);
      column_max_abs[j] = std::max(column_max_abs[j], abs_value);
    }
    for (Int j = 0; j < num_rows; ++j) {
      scaling->Entry(j) *= new_scaling[j];
    }

    if (pass != kNumPasses - 1) {
      for (Int i = 0; i < num_rows; ++i) {
        new_scaling[i] = std::sqrt(column_max_abs[i]);
      }
    }
  }

  if (verbose) {
    const Real max_abs =
        *std::max_element(column_max_abs.begin(), column_max_abs.end());
    std::cout << "Final results:\n"
              << "  max_abs: " << max_abs << std::endl;
  }
}

}  // namespace catamari

#endif  // ifndef CATAMARI_EQUILIBRATE_SYMMETRIC_MATRIX_IMPL_H_
