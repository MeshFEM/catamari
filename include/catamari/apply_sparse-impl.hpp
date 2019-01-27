/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_APPLY_SPARSE_IMPL_H_
#define CATAMARI_APPLY_SPARSE_IMPL_H_

#include "catamari/apply_sparse.hpp"
#include "catamari/integers.hpp"
#include "catamari/macros.hpp"

namespace catamari {

template <class Field>
void ApplySparse(const Field& alpha,
                 const CoordinateMatrix<Field>& sparse_matrix,
                 const ConstBlasMatrix<Field>& input_matrix, const Field& beta,
                 BlasMatrix<Field>* result) {
  const Int num_rows = sparse_matrix.NumRows();
  const Int num_rhs = input_matrix.width;
  CATAMARI_ASSERT(input_matrix.height == sparse_matrix.NumColumns(),
                  "input_matrix was the incorrect height.");
  CATAMARI_ASSERT(result->height == num_rows,
                  "result was the incorrect height.");
  CATAMARI_ASSERT(input_matrix.width == result->width,
                  "result was the incorrect width.");

  // vec1 += alpha matrix vec0 + beta vec1
  const Buffer<MatrixEntry<Field>>& entries = sparse_matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    for (Int j = 0; j < num_rhs; ++j) {
      result->Entry(row, j) *= beta;
    }

    const Int row_beg = sparse_matrix.RowEntryOffset(row);
    const Int row_end = sparse_matrix.RowEntryOffset(row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      CATAMARI_ASSERT(entry.row == row, "Invalid entry row index.");
      for (Int j = 0; j < num_rhs; ++j) {
        result->Entry(row, j) +=
            alpha * entry.value * input_matrix(entry.column, j);
      }
    }
  }
}

}  // namespace catamari

#endif  // ifndef CATAMARI_APPLY_SPARSE_IMPL_H_
