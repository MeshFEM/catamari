/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_MATRIX_VECTOR_PRODUCT_IMPL_H_
#define CATAMARI_MATRIX_VECTOR_PRODUCT_IMPL_H_

#include "catamari/integers.hpp"
#include "catamari/macros.hpp"
#include "catamari/matrix_vector_product.hpp"

namespace catamari {

template <class Field>
void MatrixVectorProduct(const Field& alpha,
                         const CoordinateMatrix<Field>& matrix,
                         const std::vector<Field>& vec0, const Field& beta,
                         std::vector<Field>* vec1) {
  const Int num_rows = matrix.NumRows();
  CATAMARI_ASSERT(static_cast<Int>(vec0.size()) == matrix.NumColumns(),
                  "vec0 was an incorrect size.");
  CATAMARI_ASSERT(static_cast<Int>(vec1->size()) == num_rows,
                  "vec1 was an incorrect size.");

  // vec1 += alpha matrix vec0 + beta vec1
  const std::vector<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    (*vec1)[row] *= beta;
    const Int row_beg = matrix.RowEntryOffset(row);
    const Int row_end = matrix.RowEntryOffset(row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      CATAMARI_ASSERT(entry.row == row, "Invalid entry row index.");
      // vec1(row) += alpha matrix(row, column) vec0(column).
      (*vec1)[row] += alpha * entry.value * vec0[entry.column];
    }
  }
}

}  // namespace catamari

#endif  // ifndef CATAMARI_MATRIX_VECTOR_PRODUCT_IMPL_H_
