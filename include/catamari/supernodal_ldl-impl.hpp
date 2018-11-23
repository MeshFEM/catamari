/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SUPERNODAL_LDL_IMPL_H_
#define CATAMARI_SUPERNODAL_LDL_IMPL_H_

#include "catamari/supernodal_ldl.hpp"

namespace catamari {

namespace supernodal_ldl {

// Fills 'supernode_offsets' with a length 'num_supernodes + 1' array whose
// i'th index is the sum of the supernode sizes whose indices are less than i.
inline void SupernodeOffsets(const std::vector<Int>& supernode_sizes,
                             std::vector<Int>* supernode_offsets) {
  const Int num_supernodes = supernode_sizes.size();
  supernode_offsets->resize(num_supernodes + 1);

  Int offset = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    (*supernode_offsets)[supernode] = offset;
    offset += supernode_sizes[supernode];
  }
  (*supernode_offsets)[num_supernodes] = offset;
}

// Fills 'supernode_container' with a length 'num_rows' array whose i'th index
// is the index of the supernode containing column 'i'.
inline void SupernodeContainers(Int num_rows,
                                const std::vector<Int>& supernode_offsets,
                                std::vector<Int>* supernode_container) {
  supernode_container->resize(num_rows);

  Int supernode = 0;
  for (Int column = 0; column < num_rows; ++column) {
    if (column == supernode_offsets[supernode + 1]) {
      ++supernode;
    }
    CATAMARI_ASSERT(column >= supernode_offsets[supernode] &&
                        column < supernode_offsets[supernode + 1],
                    "Column was not in the marked supernode.");
    (*supernode_container)[column] = supernode;
  }
  CATAMARI_ASSERT(supernode == supernode_offsets.size() - 2,
                  "Did not end on the last supernode.");
}

// Computes the elimination forest (via the 'parents' array) and sizes of the
// structures of a supernodal LDL' factorization.
template <class Field>
void EliminationForestAndStructureSizes(
    const CoordinateMatrix<Field>& matrix,
    const std::vector<Int>& supernode_sizes,
    const std::vector<Int>& supernode_offsets,
    const std::vector<Int>& supernode_container, std::vector<Int>* parents,
    std::vector<Int>* degrees) {
  const Int num_rows = matrix.NumRows();
  const Int num_supernodes = supernode_sizes.size();

  // Initialize the last member of each supernode as unset, but other supernode
  // members as having the member to their right as their parent.
  {
    parents->resize(num_rows, -1);
    Int supernode = 0;
    for (Int column = 0; column < num_rows; ++column) {
      if (column == supernode_offsets[supernode + 1]) {
        ++supernode;
      }
      if (column != supernode_offsets[supernode + 1] - 1) {
        (*parents)[column] = column + 1;
      }
    }
  }

  // A data structure for marking whether or not an index is in the pattern
  // of the active row of the lower-triangular factor.
  std::vector<Int> pattern_flags(num_rows);

  // Initialize the number of entries that will be stored into each supernode
  degrees->clear();
  degrees->resize(num_supernodes, 0);

  const std::vector<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int supernode = supernode_container[row];
    pattern_flags[row] = row;

    const Int row_beg = matrix.RowEntryOffset(row);
    const Int row_end = matrix.RowEntryOffset(row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      Int column = entry.column;
      Int column_supernode = supernode_container[column];

      // We are traversing the strictly lower triangle and know that the
      // indices are sorted.
      if (column_supernode >= supernode) {
        break;
      }

      // Look for new entries in the pattern by walking up to the root of this
      // subtree of the elimination forest from index 'column'. Any unset
      // parent pointers can be filled in during the traversal, as the current
      // row index would then be the parent.
      while (pattern_flags[column] != row) {
        // Mark index 'column' as in the pattern of row 'row'.
        pattern_flags[column] = row;

        // Since the pattern of each column in a supernode is the same, we can
        // compute a supernode's pattern size by only incrementing for
        // principal supernode members.
        const Int column_supernode = supernode_container[column];
        const Int column_supernode_beg = supernode_offsets[column_supernode];
        if (column == column_supernode_beg) {
          ++(*degrees)[column_supernode];
        }

        if ((*parents)[column] == -1) {
          // This is the first occurrence of 'column' in a row pattern.
          // This can only occur for the last member of a supernode.
          (*parents)[column] = row;
        }

        // Move up to the parent in this subtree of the elimination forest.
        // Moving to the parent will increase the index (but remain bounded
        // from above by 'row').
        column = (*parents)[column];
      }
    }
  }
}

// Fills in the structure indices for the lower factor.
template <class Field>
void FillStructureIndices(const CoordinateMatrix<Field>& matrix,
                          const std::vector<Int>& parents,
                          SupernodalLDLFactorization<Field>* factorization) {
  const Int num_rows = matrix.NumRows();
  const Int num_supernodes = factorization->supernode_sizes.size();
  SupernodalLowerFactor<Field>& lower_factor = factorization->lower_factor;
  SupernodalDiagonalFactor<Field>& diagonal_factor =
      factorization->diagonal_factor;

  // A data structure for marking whether or not an index is in the pattern
  // of the active row of the lower-triangular factor.
  std::vector<Int> pattern_flags(num_rows);

  // A set of pointers for keeping track of where to insert supernode pattern
  // indices.
  std::vector<Int> supernode_ptrs(num_supernodes);

  // Fill in the structure indices.
  const std::vector<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int supernode = factorization->supernode_container[row];
    pattern_flags[row] = row;
    supernode_ptrs[supernode] = lower_factor.supernode_index_offsets[supernode];

    const Int row_beg = matrix.RowEntryOffset(row);
    const Int row_end = matrix.RowEntryOffset(row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      const Int column = entry.column;
      const Int column_supernode = factorization->supernode_container[column];

      // We are traversing the strictly lower triangle and know that the
      // indices are sorted.
      if (column_supernode >= supernode) {
        break;
      }

      // Look for new entries in the pattern by walking up to the root of this
      // subtree of the elimination forest from index 'column'. Any unset
      // parent pointers can be filled in during the traversal, as the current
      // index would then be the parent.
      while (pattern_flags[column] != row) {
        // Mark index 'column' as in the pattern of this row.
        pattern_flags[column] = row;

        // Since the pattern of each column in a supernode is the same, we can
        // compute a supernode's pattern size by only incrementing for
        // principal supernode members.
        const Int column_supernode = factorization->supernode_container[column];
        const Int column_supernode_beg =
            factorization->supernode_offsets[column_supernode];
        if (column == column_supernode_beg) {
          lower_factor.indices[supernode_ptrs[column_supernode]++] = row;
        }

        // Move up to the parent in this subtree of the elimination forest.
        // Moving to the parent will increase the index (but remain bounded
        // from above by 'row').
        column = parents[column];
      }
    }
  }
}

// Fill in the nonzeros from the original sparse matrix.
template <class Field>
void FillNonzeros(const CoordinateMatrix<Field>& matrix,
                  SupernodalLDLFactorization<Field>* factorization) {
  const Int num_rows = matrix.NumRows();
  SupernodalLowerFactor<Field>& lower_factor = factorization->lower_factor;
  SupernodalDiagonalFactor<Field>& diagonal_factor =
      factorization->diagonal_factor;

  const std::vector<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int supernode = factorization->supernode_container[row];
    const Int supernode_beg = factorization->supernode_offsets[supernode];
    const Int supernode_size = factorization->supernode_sizes[supernode];

    const Int row_beg = matrix.RowEntryOffset(row);
    const Int row_end = matrix.RowEntryOffset(row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      const Int column = entry.column;
      const Int column_supernode = factorization->supernode_container[column];

      if (column_supernode == supernode) {
        // Insert the value into the diagonal block.
        const Int diag_supernode_offset =
            diagonal_factor.supernode_value_offsets[supernode];
        const Int rel_row = row - supernode_beg;
        const Int rel_column = column - supernode_beg;
        const Int rel_index = rel_row + rel_column * supernode_size;
        diagonal_factor.values[diag_supernode_offset + rel_index] = entry.value;
      }

      if (column_supernode >= supernode) {
        break;
      }

      // Compute the relative index of 'row' in this column supernode.
      const Int column_index_beg =
          lower_factor.supernode_index_offsets[column_supernode];
      const Int column_index_end =
          lower_factor.supernode_index_offsets[column_supernode + 1];
      Int* iter =
          std::lower_bound(lower_factor.indices.data() + column_index_beg,
                           lower_factor.indices.data() + column_index_end, row);
      CATAMARI_ASSERT(iter != lower_factor.indices.data() + column_index_end,
                      "Exceeded column indices.");
      CATAMARI_ASSERT(*iter == row, "Did not find index.");
      const Int index_index = std::distance(lower_factor.indices.data(), iter);
      const Int rel_index_index = index_index - column_index_beg;

      // Convert the relative row index into the corresponding value index
      // (for column 'column').
      const Int column_value_beg =
          lower_factor.supernode_value_offsets[column_supernode];
      const Int column_supernode_size =
          factorization->supernode_sizes[column_supernode];
      const Int column_supernode_beg =
          factorization->supernode_offsets[column_supernode];
      const Int rel_column = column - column_supernode_beg;
      const Int value_index =
          column_value_beg * column_supernode_size + rel_column;
      lower_factor.values[value_index] = entry.value;
    }
  }
}

template <class Field>
void InitializeLeftLookingFactors(
    const CoordinateMatrix<Field>& matrix, const std::vector<Int>& parents,
    std::vector<Int>* degrees,
    SupernodalLDLFactorization<Field>* factorization) {
  LowerFactor<Field>& lower_factor = factorization->lower_factor;
  DiagonalFactor<Field>& diagonal_factor = factorization->diagonal_factor;
  const Int num_rows = matrix.NumRows();
  const Int num_supernodes = factorization->supernode_sizes.size();

  // Set up the column offsets and allocate space (initializing the values of
  // the unit-lower and diagonal and all zeros).
  lower_factor.supernode_index_offsets.resize(num_supernodes + 1);
  lower_factor.supernode_value_offsets.resize(num_supernodes + 1);
  Int num_entries = 0;
  Int num_supernode_entries = 0;
  Int num_diagonal_entries = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    lower_factor.supernode_index_offsets[supernode] = num_supernode_entries;
    lower_factor.supernode_value_offsets[supernode] = num_entries;
    diagonal_factor.supernode_value_offsets[supernode] = num_diagonal_entries;

    const Int struct_size = (*degrees)[supernode];
    const Int supernode_size = factorization->supernode_sizes[supernode];
    num_supernode_entries += struct_size;
    num_entries += struct_size * supernode_size;
    num_diagonal_entries += supernode_size * supernode_size;
  }
  lower_factor.supernode_index_offsets[num_supernodes] = num_supernode_entries;
  lower_factor.supernode_value_offsets[num_supernodes] = num_entries;
  diagonal_factor.supernode_value_offsets[num_supernodes] =
      num_diagonal_entries;
  lower_factor.indices.resize(num_supernode_entries);
  lower_factor.values.resize(num_entries, Field{0});
  diagonal_factor.values.resize(num_diagonal_entries, Field{0});

  FillStructureIndices(matrix, parents, factorization);
  FillNonzeros(matrix, factorization);
}

template <class Field>
void LeftLookingSetup(const CoordinateMatrix<Field>& matrix,
                      const std::vector<Int>& supernode_sizes,
                      std::vector<Int>* parents,
                      SupernodalLDLFactorization<Field>* factorization) {
  std::vector<Int> degrees;
  factorization->supernode_sizes = supernode_sizes;
  SupernodeOffsets(supernode_sizes, &factorization->supernode_offsets);
  SupernodeContainers(supernode_sizes.size(), factorization->supernode_offsets,
                      &factorization->supernode_container);

  EliminationForestAndStructureSizes(
      matrix, factorization->supernode_sizes, factorization->supernode_offsets,
      factorization->supernode_container, parents, &degrees);
  InitializeLeftLookingFactors(matrix, *parents, &degrees, factorization);
}

}  // namespace supernodal_ldl

}  // namespace catamari

#endif  // ifndef CATAMARI_SUPERNODAL_LDL_IMPL_H_
