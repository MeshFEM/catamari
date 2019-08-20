/*
 * Copyright (c) 2018-2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SCALAR_FACTORIZATION_COMMON_IMPL_H_
#define CATAMARI_SPARSE_LDL_SCALAR_FACTORIZATION_COMMON_IMPL_H_

#include <cmath>

#include "catamari/index_utils.hpp"
#include "catamari/sparse_ldl/scalar/scalar_utils.hpp"
#include "quotient/io_utils.hpp"

#include "catamari/sparse_ldl/scalar/factorization.hpp"

namespace catamari {
namespace scalar_ldl {

template <class Field>
void Factorization<Field>::FillNonzeros(const CoordinateMatrix<Field>& matrix)
    CATAMARI_NOEXCEPT {
  LowerStructure& lower_structure = lower_factor.structure;
  const Int num_rows = matrix.NumRows();
  const Int num_entries = lower_structure.indices.Size();
  const bool have_permutation = !ordering.permutation.Empty();

  lower_factor.values.Resize(num_entries, Field{0});
  diagonal_factor.values.Resize(num_rows, Field{0});

  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int orig_row =
        have_permutation ? ordering.inverse_permutation[row] : row;
    const Int row_beg = matrix.RowEntryOffset(orig_row);
    const Int row_end = matrix.RowEntryOffset(orig_row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      Int column =
          have_permutation ? ordering.permutation[entry.column] : entry.column;

      if (column == row) {
        diagonal_factor.values[column] = entry.value;
      }
      if (column >= row) {
        if (have_permutation) {
          continue;
        } else {
          break;
        }
      }

      Int* iter = std::lower_bound(lower_structure.ColumnBeg(column),
                                   lower_structure.ColumnEnd(column), row);
      CATAMARI_ASSERT(iter != lower_structure.ColumnEnd(column),
                      "Exceeded column indices.");
      CATAMARI_ASSERT(*iter == row, "Did not find index.");
      const Int structure_index =
          std::distance(lower_structure.indices.Data(), iter);
      lower_factor.values[structure_index] = entry.value;
    }
  }
}

template <class Field>
SparseLDLResult<Field> Factorization<Field>::Factor(
    const CoordinateMatrix<Field>& matrix,
    const SymmetricOrdering& manual_ordering,
    const Control<Field>& control_value) {
  ordering = manual_ordering;
  control = control_value;
  if (control.algorithm == kLeftLookingLDL) {
    LeftLookingSetup(matrix);
    return LeftLooking(matrix);
  } else {
    UpLookingSetup(matrix);
    return UpLooking(matrix);
  }
}

template <class Field>
SparseLDLResult<Field> Factorization<Field>::RefactorWithFixedSparsityPattern(
    const CoordinateMatrix<Field>& matrix) {
  if (control.algorithm == kLeftLookingLDL) {
    FillNonzeros(matrix);
    return LeftLooking(matrix);
  } else {
    return UpLooking(matrix);
  }
}

template <class Field>
Int Factorization<Field>::NumRows() const {
  return diagonal_factor.values.Size();
}

template <class Field>
const Buffer<Int>& Factorization<Field>::Permutation() const {
  return ordering.permutation;
}

template <class Field>
const Buffer<Int>& Factorization<Field>::InversePermutation() const {
  return ordering.inverse_permutation;
}

}  // namespace scalar_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_SPARSE_LDL_SCALAR_FACTORIZATION_COMMON_IMPL_H_
