/*
 * Copyright (c) 2018-2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SYMMETRIC_ORDERING_IMPL_H_
#define CATAMARI_SYMMETRIC_ORDERING_IMPL_H_

#include "catamari/symmetric_ordering.hpp"

namespace catamari {

inline void AssemblyForest::FillFromParents() {
  const Int num_indices = parents.Size();

  // Compute the number of children (initially stored in 'child_offsets') of
  // each vertex. Along the way, count the number of trees in the forest.
  Int num_roots = 0;
  child_offsets.Resize(num_indices + 1, 0);
  for (Int index = 0; index < num_indices; ++index) {
    const Int parent = parents[index];
    if (parent >= 0) {
      ++child_offsets[parent];
    } else {
      ++num_roots;
    }
  }

  // Compute the child offsets using an in-place scan.
  Int num_total_children = 0;
  for (Int index = 0; index < num_indices; ++index) {
    const Int num_children = child_offsets[index];
    child_offsets[index] = num_total_children;
    num_total_children += num_children;
  }
  child_offsets[num_indices] = num_total_children;

  // Pack the children into the 'children' buffer.
  children.Resize(num_total_children);
  roots.Resize(num_roots);
  Int counter = 0;
  Buffer<Int> offsets_copy = child_offsets;
  for (Int index = 0; index < num_indices; ++index) {
    const Int parent = parents[index];
    if (parent >= 0) {
      children[offsets_copy[parent]++] = index;
    } else {
      roots[counter++] = index;
    }
  }
}

inline Int AssemblyForest::NumChildren(Int index) const {
  return child_offsets[index + 1] - child_offsets[index];
}

template <class Field>
void PermuteMatrix(const CoordinateMatrix<Field>& matrix,
                   const SymmetricOrdering& ordering,
                   CoordinateMatrix<Field>* reordered_matrix) {
  const Int num_rows = matrix.NumRows();
  reordered_matrix->Resize(num_rows, num_rows);

  // Compute the offsets for the reordered matrix.
  const auto& row_entry_offsets = matrix.RowEntryOffsets();
  auto& reordered_row_entry_offsets = reordered_matrix->RowEntryOffsets();
  {
    reordered_row_entry_offsets.Resize(num_rows + 1);
    Int offset = 0;
    for (Int i = 0; i < num_rows; ++i) {
      reordered_row_entry_offsets[i] = offset;
      const Int i_old = ordering.inverse_permutation[i];
      const Int row_count =
          row_entry_offsets[i_old + 1] - row_entry_offsets[i_old];
      offset += row_count;
    }
    reordered_row_entry_offsets[num_rows] = offset;
  }

  // Fill in the rows of the reordered matrix, with each row initially
  // arbitrarily ordered, then sorted.
  const auto& entries = matrix.Entries();
  auto& reordered_entries = reordered_matrix->Entries();
  reordered_entries.Resize(entries.Size());
  {
    Int row_beg = 0;
    for (Int row = 0; row < num_rows; ++row) {
      const Int orig_row = ordering.inverse_permutation[row];
      const Int orig_row_beg = row_entry_offsets[orig_row];
      const Int orig_row_end = row_entry_offsets[orig_row + 1];
      const Int row_size = orig_row_end - orig_row_beg;
      const Int row_end = row_beg + row_size;
      for (Int index = 0; index < row_size; ++index) {
        const MatrixEntry<Field>& entry = entries[orig_row_beg + index];
        const Int column = ordering.permutation[entry.column];
        reordered_entries[row_beg + index] =
            MatrixEntry<Field>{row, column, entry.value};
      }
      std::sort(&reordered_entries[row_beg], &reordered_entries[row_end]);
      row_beg = row_end;
    }
  }
}

}  // namespace catamari

#endif  // ifndef CATAMARI_SYMMETRIC_ORDERING_IMPL_H_
