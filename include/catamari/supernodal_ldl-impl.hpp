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
    CATAMARI_ASSERT(supernode_sizes[supernode], "Supernode was empty.");
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
  CATAMARI_ASSERT(supernode == static_cast<Int>(supernode_offsets.size()) - 2,
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
    const Int main_supernode = supernode_container[row];
    pattern_flags[row] = row;

    const Int row_beg = matrix.RowEntryOffset(row);
    const Int row_end = matrix.RowEntryOffset(row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      Int descendant = entry.column;
      Int descendant_supernode = supernode_container[descendant];

      // We are traversing the strictly lower triangle and know that the
      // indices are sorted.
      if (descendant_supernode >= main_supernode) {
        break;
      }

      // Look for new entries in the pattern by walking up to the root of this
      // subtree of the elimination forest from index 'descendant'. Any unset
      // parent pointers can be filled in during the traversal, as the current
      // row index would then be the parent.
      while (pattern_flags[descendant] != row) {
        // Mark index 'column' as in the pattern of row 'row'.
        pattern_flags[descendant] = row;

        // Since the pattern of each column in a supernode is the same, we can
        // compute a supernode's pattern size by only incrementing for
        // principal supernode members.
        descendant_supernode = supernode_container[descendant];
        const Int descendant_supernode_beg =
            supernode_offsets[descendant_supernode];
        if (descendant == descendant_supernode_beg) {
          ++(*degrees)[descendant_supernode];
        }

        if ((*parents)[descendant] == -1) {
          // This is the first occurrence of 'descendant' in a row pattern.
          // This can only occur for the last member of a supernode.
          (*parents)[descendant] = row;
        }

        // Move up to the parent in this subtree of the elimination forest.
        // Moving to the parent will increase the index (but remain bounded
        // from above by 'row').
        descendant = (*parents)[descendant];
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

  // A data structure for marking whether or not a supernode is in the pattern
  // of the active row of the lower-triangular factor.
  std::vector<Int> pattern_flags(num_supernodes);

  // A set of pointers for keeping track of where to insert supernode pattern
  // indices.
  std::vector<Int> supernode_ptrs(num_supernodes);

  // Fill in the structure indices.
  const std::vector<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int main_supernode = factorization->supernode_container[row];
    pattern_flags[main_supernode] = row;
    supernode_ptrs[main_supernode] = lower_factor.index_offsets[main_supernode];

    const Int row_beg = matrix.RowEntryOffset(row);
    const Int row_end = matrix.RowEntryOffset(row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      Int descendant = entry.column;
      Int descendant_supernode = factorization->supernode_container[descendant];

      // We are traversing the strictly lower triangle and know that the
      // indices are sorted.
      if (descendant_supernode >= main_supernode) {
        break;
      }

      // Move to the root of the descendant supernode.
      descendant = factorization->supernode_offsets[descendant_supernode] +
                   factorization->supernode_sizes[descendant_supernode] - 1;

      // Look for new entries in the pattern by walking up to the root of this
      // subtree of the elimination forest.
      while (pattern_flags[descendant_supernode] != row) {
        // Mark supernode 'descendant_supernode' as in the pattern of this row.
        pattern_flags[descendant_supernode] = row;
        lower_factor.indices[supernode_ptrs[descendant_supernode]++] = row;

        // Move up to the parent in this subtree of the elimination forest.
        // Moving to the parent will increase the index (but remain bounded
        // from above by 'row').
        descendant = parents[descendant];
        descendant_supernode = factorization->supernode_container[descendant];
        descendant = factorization->supernode_offsets[descendant_supernode] +
                     factorization->supernode_sizes[descendant_supernode] - 1;
      }
    }
  }
}

// Fills in the sizes of the supernodal intersections.
template <class Field>
void FillIntersections(SupernodalLDLFactorization<Field>* factorization) {
  const Int num_supernodes = factorization->supernode_sizes.size();
  SupernodalLowerFactor<Field>& lower_factor = factorization->lower_factor;

  // Compute the supernode offsets.
  Int num_supernode_intersections = 0;
  lower_factor.intersection_size_offsets.resize(num_supernodes + 1);
  for (Int column_supernode = 0; column_supernode < num_supernodes;
       ++column_supernode) {
    lower_factor.intersection_size_offsets[column_supernode] =
        num_supernode_intersections;
    Int last_supernode = -1;

    const Int index_beg = factorization->index_offsets[column_supernode];
    const Int index_end = factorization->index_offsets[column_supernode + 1];
    for (Int index = index_beg; index < index_end; ++index) {
      const Int row = lower_factor.indices[index];
      const Int supernode = factorization->supernode_container[row];
      if (supernode != last_supernode) {
        last_supernode = supernode;
        ++num_supernode_intersections;
      }
    }
  }
  lower_factor.intersection_size_offsets[num_supernodes] =
      num_supernode_intersections;

  // Fill the supernode intersection sizes.
  lower_factor.intersection_sizes.resize(num_supernode_intersections);
  Int num_intersections = 0;
  num_supernode_intersections = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    Int last_supernode = -1;
    Int intersection_size = 0;

    const Int index_beg = factorization->index_offsets[supernode];
    const Int index_end = factorization->index_offsets[supernode + 1];
    for (Int index = index_beg; index < index_end; ++index) {
      const Int row = lower_factor.indices[index];
      const Int row_supernode = factorization->supernode_container[row];
      if (row_supernode != last_supernode) {
        if (last_supernode != -1) {
          // Close out the supernodal intersection.
          lower_factor.intersection_sizes[num_supernode_intersections++] =
              intersection_size;
        }
        last_supernode = row_supernode;
        intersection_size = 0;
      }
    }
    if (last_supernode != -1) {
      // Close out the last intersection count for this column supernode.
      lower_factor.intersection_sizes[num_supernode_intersections++] =
          intersection_size;
    }
  }
  CATAMARI_ASSERT(num_supernode_intersections ==
                      static_cast<Int>(lower_factor.intersection_sizes.size()),
                  "Incorrect number of supernode intersections");
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
            diagonal_factor.value_offsets[supernode];
        const Int rel_row = row - supernode_beg;
        const Int rel_column = column - supernode_beg;
        const Int rel_index = rel_row + rel_column * supernode_size;
        diagonal_factor.values[diag_supernode_offset + rel_index] = entry.value;
      }

      if (column_supernode >= supernode) {
        break;
      }

      // Compute the relative index of 'row' in this column supernode.
      const Int column_index_beg = lower_factor.index_offsets[column_supernode];
      const Int column_index_end =
          lower_factor.index_offsets[column_supernode + 1];
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
      const Int column_value_beg = lower_factor.value_offsets[column_supernode];
      const Int column_supernode_size =
          factorization->supernode_sizes[column_supernode];
      const Int column_supernode_beg =
          factorization->supernode_offsets[column_supernode];
      const Int rel_column = column - column_supernode_beg;
      const Int value_index = column_value_beg +
                              rel_index_index * column_supernode_size +
                              rel_column;
      lower_factor.values[value_index] = entry.value;
    }
  }
}

template <class Field>
void InitializeLeftLookingFactors(
    const CoordinateMatrix<Field>& matrix, const std::vector<Int>& parents,
    std::vector<Int>* degrees,
    SupernodalLDLFactorization<Field>* factorization) {
  SupernodalLowerFactor<Field>& lower_factor = factorization->lower_factor;
  SupernodalDiagonalFactor<Field>& diagonal_factor =
      factorization->diagonal_factor;
  const Int num_supernodes = factorization->supernode_sizes.size();
  CATAMARI_ASSERT(static_cast<Int>(degrees->size()) == num_supernodes,
                  "Invalid degrees size.");

  // Set up the column offsets and allocate space (initializing the values of
  // the unit-lower and diagonal and all zeros).
  lower_factor.index_offsets.resize(num_supernodes + 1);
  lower_factor.value_offsets.resize(num_supernodes + 1);
  diagonal_factor.value_offsets.resize(num_supernodes + 1);
  Int num_entries = 0;
  Int num_supernode_entries = 0;
  Int num_diagonal_entries = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    lower_factor.index_offsets[supernode] = num_supernode_entries;
    lower_factor.value_offsets[supernode] = num_entries;
    diagonal_factor.value_offsets[supernode] = num_diagonal_entries;

    const Int struct_size = (*degrees)[supernode];
    const Int supernode_size = factorization->supernode_sizes[supernode];
    num_supernode_entries += struct_size;
    num_entries += struct_size * supernode_size;
    num_diagonal_entries += supernode_size * supernode_size;
  }
  lower_factor.index_offsets[num_supernodes] = num_supernode_entries;
  lower_factor.value_offsets[num_supernodes] = num_entries;
  diagonal_factor.value_offsets[num_supernodes] = num_diagonal_entries;
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
  factorization->supernode_sizes = supernode_sizes;
  SupernodeOffsets(supernode_sizes, &factorization->supernode_offsets);
  SupernodeContainers(matrix.NumRows(), factorization->supernode_offsets,
                      &factorization->supernode_container);

  std::vector<Int> degrees;
  EliminationForestAndStructureSizes(
      matrix, factorization->supernode_sizes, factorization->supernode_offsets,
      factorization->supernode_container, parents, &degrees);
  InitializeLeftLookingFactors(matrix, *parents, &degrees, factorization);
}

// Computes the supernodal nonzero pattern of L(row, :) in
// row_structure[0 : num_packed - 1].
template <class Field>
Int ComputeRowPattern(const CoordinateMatrix<Field>& matrix,
                      const std::vector<Int>& supernode_sizes,
                      const std::vector<Int>& supernode_offsets,
                      const std::vector<Int>& supernode_container,
                      const std::vector<Int>& parents, Int main_supernode,
                      Int* pattern_flags, Int* row_structure) {
  const Int row = supernode_offsets[main_supernode];
  const Int row_beg = matrix.RowEntryOffset(row);
  const Int row_end = matrix.RowEntryOffset(row + 1);
  const std::vector<MatrixEntry<Field>>& entries = matrix.Entries();
  Int num_packed = 0;
  for (Int index = row_beg; index < row_end; ++index) {
    const MatrixEntry<Field>& entry = entries[index];
    Int descendant = entry.column;
    Int descendant_supernode = supernode_container[descendant];
    if (descendant_supernode >= main_supernode) {
      break;
    }

    // Move to the root of the descendant supernode.
    descendant = supernode_offsets[descendant_supernode] +
                 supernode_sizes[descendant_supernode] - 1;

    // Walk up to the root of the current subtree of the elimination
    // forest, stopping if we encounter a member already marked as in the
    // row pattern.
    while (pattern_flags[descendant] != main_supernode) {
      // Place 'descendant_supernode' into the pattern of this supernode.
      row_structure[num_packed++] = descendant_supernode;
      pattern_flags[descendant_supernode] = main_supernode;

      // Move up to the parent in this subtree of the elimination forest.
      descendant = parents[descendant];
      descendant_supernode = supernode_container[descendant];
      descendant = supernode_offsets[descendant_supernode] +
                   supernode_sizes[descendant_supernode] - 1;
    }
  }
  return num_packed;
}

// Store the scaled adjoint update matrix, Z(d, m) = D(d, d) L(m, d)'
// contiguously in column-major form with minimal leading dimension.
//
// The matrix will be packed into the first
// descendant_supernode_size x descendant_main_intersection_size entries.
template <class Field>
void FormScaledAdjoint(const SupernodalLDLFactorization<Field>& factorization,
                       Int descendant_supernode,
                       Int descendant_main_intersection_size,
                       Int descendant_main_value_beg,
                       Field* scaled_adjoint_buffer) {
  const SupernodalLowerFactor<Field>& lower_factor = factorization.lower_factor;
  const SupernodalDiagonalFactor<Field>& diagonal_factor =
      factorization.diagonal_factor;
  const Int descendant_supernode_size =
      factorization.supernode_sizes[descendant_supernode];

  const Field* descendant_diag_block =
      &diagonal_factor
           .values[diagonal_factor.value_offsets[descendant_supernode]];
  const Field* descendant_main_block =
      &lower_factor.values[descendant_main_value_beg];
  for (Int i = 0; i < descendant_supernode_size; ++i) {
    const Field& delta =
        descendant_diag_block[i + i * descendant_supernode_size];
    for (Int j_rel = 0; j_rel < descendant_main_intersection_size; ++j_rel) {
      // Recall that lower_insersect_block is row-major.
      const Field& lambda_right_conj = Conjugate(
          descendant_main_block[i + j_rel * descendant_supernode_size]);
      scaled_adjoint_buffer[i + j_rel * descendant_supernode_size] =
          delta * lambda_right_conj;
    }
  }
}

// L(m, m) -= L(m, d) * (D(d, d) * L(m, d)')
//          = L(m, d) * Z(:, m).
//
// The update is to the main_supernode_size x main_supernode_size dense
// diagonal block, L(m, m), with the densified L(m, d) matrix being
// descendant_main_intersection_size x descendant_supernode_size, and Z(:, m)
// being descendant_supernode_size x descendant_main_intersection_size.
template <class Field>
void UpdateDiagonalBlock(Int main_supernode, Int descendant_supernode,
                         Int descendant_main_intersection_size,
                         Int descendant_main_index_beg,
                         Int descendant_main_value_beg,
                         const Field* scaled_adjoint_buffer,
                         SupernodalLDLFactorization<Field>* factorization,
                         Field* update_buffer) {
  const SupernodalLowerFactor<Field>& lower_factor =
      factorization->lower_factor;
  SupernodalDiagonalFactor<Field>& diagonal_factor =
      factorization->diagonal_factor;

  const Int main_supernode_size =
      factorization->supernode_sizes[main_supernode];
  const Int descendant_supernode_size =
      factorization->supernode_sizes[descendant_supernode];

  const bool inplace_update =
      descendant_main_intersection_size == main_supernode_size;

  Field* main_diag_block =
      &diagonal_factor.values[diagonal_factor.value_offsets[main_supernode]];
  Field* update_block = inplace_update ? main_diag_block : update_buffer;

  // TODO(Jack Poulson): Switch to a level 3 BLAS implementation after
  // verifying correctness.
  const Field* descendant_main_block =
      &lower_factor.values[descendant_main_value_beg];
  for (Int j = 0; j < descendant_main_intersection_size; ++j) {
    const Field* right_col =
        &scaled_adjoint_buffer[j * descendant_supernode_size];
    for (Int i = j; i < descendant_main_intersection_size; ++i) {
      const Field* left_row =
          &descendant_main_block[i * descendant_supernode_size];
      // L(m_beg + i, m_beg + j) -=
      //     L(m_beg + i, d) * (D(d, d) * conj(L(m_beg + j, d))) =
      //     L(m_beg + i, d) * Z(:, j).
      for (Int k = 0; k < descendant_supernode_size; ++k) {
        const Field& lambda_left = left_row[k];
        const Field& eta = right_col[k];
        update_block[i + j * descendant_main_intersection_size] -=
            lambda_left * eta;
      }
    }
  }

  if (!inplace_update) {
    // Apply the out-of-place update and zero the buffer.
    const Int main_supernode_beg =
        factorization->supernode_offsets[main_supernode];
    const Int* descendant_main_indices =
        &lower_factor.indices[descendant_main_index_beg];
    for (Int j = 0; j < descendant_main_intersection_size; ++j) {
      const Int j_rel = descendant_main_indices[j] - main_supernode_beg;
      for (Int i = j; i < descendant_main_intersection_size; ++i) {
        const Int i_rel = descendant_main_indices[i] - main_supernode_beg;
        main_diag_block[i_rel + j_rel * main_supernode_size] +=
            update_buffer[i + j * descendant_main_intersection_size];
        update_buffer[i + j * descendant_main_intersection_size] = 0;
      }
    }
  }
}

// L(a, m) -= L(a, d) * (D(d, d) * L(m, d)')
//          = L(a, d) * Z(:, m).
//
// The update is to the main_active_intersection_size x main_supernode_size
// densified matrix L(a, m) using the
// descendant_active_intersection_size x descendant_supernode_size densified
// L(a, d) and thd escendant_supernode_size x descendant_main_intersection_size
// Z(:, m).
template <class Field>
void UpdateSubdiagonalBlock(
    Int main_supernode, Int descendant_supernode,
    Int descendant_main_intersection_size, Int descendant_main_index_beg,
    Int descendant_active_intersection_size, Int descendant_active_index_beg,
    Int descendant_active_value_beg, const Field* scaled_adjoint_buffer,
    SupernodalLDLFactorization<Field>* factorization,
    Int* main_active_intersection_size_beg, Int* main_active_index_beg,
    Int* main_active_value_beg, Field* update_buffer) {
  SupernodalLowerFactor<Field>& lower_factor = factorization->lower_factor;

  const Int main_supernode_size =
      factorization->supernode_sizes[main_supernode];
  const Int descendant_supernode_size =
      factorization->supernode_sizes[descendant_supernode];

  // Move the pointers for the main supernode down to the active supernode of
  // the descendant column block.
  const Int active_supernode =
      factorization->supernode_container
          [lower_factor.indices[descendant_active_index_beg]];
  Int main_active_intersection_size =
      lower_factor.intersection_sizes[*main_active_intersection_size_beg];
  while (
      factorization
          ->supernode_container[lower_factor.indices[*main_active_index_beg]] <
      active_supernode) {
    *main_active_index_beg += main_active_intersection_size;
    *main_active_value_beg +=
        main_active_intersection_size * main_supernode_size;
    ++*main_active_intersection_size_beg;

    main_active_intersection_size =
        lower_factor.intersection_sizes[*main_active_intersection_size_beg];
  }
  CATAMARI_ASSERT(factorization->supernode_container
                          [lower_factor.indices[*main_active_index_beg]] ==
                      active_supernode,
                  "Did not find active supernode.");

  const bool inplace_update =
      main_active_intersection_size == descendant_active_intersection_size &&
      main_supernode_size == descendant_main_intersection_size;

  Field* main_active_block = &lower_factor.values[*main_active_value_beg];
  Field* update_block = inplace_update ? main_active_block : update_buffer;

  const Field* descendant_active_block =
      &lower_factor.values[descendant_active_value_beg];

  for (Int i = 0; i < descendant_active_intersection_size; ++i) {
    const Field* left_row =
        &descendant_active_block[i * descendant_supernode_size];
    for (Int j = 0; j < descendant_main_intersection_size; ++j) {
      const Field* right_col =
          &scaled_adjoint_buffer[j * descendant_supernode_size];
      // L(a_beg + i, m_beg + j) -=
      //     L(a_beg + i, d) * (D(d, d) * conj(L(m_beg + j, d))) =
      //     L(a_beg + i, d) * Z(:, j).
      for (Int k = 0; k < descendant_supernode_size; ++k) {
        const Field& lambda_left = left_row[k];
        const Field& eta = right_col[k];
        update_block[i + j * descendant_active_intersection_size] -=
            lambda_left * eta;
      }
    }
  }

  if (!inplace_update) {
    const Int main_supernode_beg =
        factorization->supernode_offsets[main_supernode];
    const Int active_supernode_beg =
        factorization->supernode_offsets[active_supernode];
    const Int* descendant_main_indices =
        &lower_factor.indices[descendant_main_index_beg];
    const Int* descendant_active_indices =
        &lower_factor.indices[descendant_active_index_beg];
    for (Int i = 0; i < descendant_active_intersection_size; ++i) {
      const Int i_rel = descendant_active_indices[i] - active_supernode_beg;
      for (Int j = 0; j < descendant_main_intersection_size; ++j) {
        const Int j_rel = descendant_main_indices[j] - main_supernode_beg;
        main_active_block[i_rel + j_rel * main_active_intersection_size] +=
            update_buffer[i + j * descendant_active_intersection_size];
        update_buffer[i + j * descendant_active_intersection_size] = 0;
      }
    }
  }
}

// Perform an in-place LDL' factorization of the supernodal diagonal block.
template <class Field>
Int FactorDiagonalBlock(Int supernode,
                        SupernodalLDLFactorization<Field>* factorization) {
  const Int supernode_size = factorization->supernode_sizes[supernode];

  const Int diag_value_beg =
      factorization->diagonal_factor.value_offsets[supernode];
  Field* diag_block = &factorization->diagonal_factor.values[diag_value_beg];

  // TODO(Jack Poulson): Use a level-3 BLAS version of this routine.
  for (Int i = 0; i < supernode_size; ++i) {
    const Field& delta = diag_block[i + i * supernode_size];
    const Field delta_conj = Conjugate(delta);
    if (delta == Field{0}) {
      return i;
    }

    // Solve for the remainder of the i'th column of L.
    for (Int k = i + 1; k < supernode_size; ++k) {
      diag_block[k + i * supernode_size] /= delta_conj;
    }

    // Perform the rank-one update.
    for (Int j = i + 1; j < supernode_size; ++j) {
      const Field eta = delta * Conjugate(diag_block[j + i * supernode_size]);
      for (Int k = j; k < supernode_size; ++k) {
        const Field& lambda_left = diag_block[k + i * supernode_size];
        diag_block[k + j * supernode_size] -= lambda_left * eta;
      }
    }
  }

  return supernode_size;
}

// L(KNext:n, K) /= D(K, K) L(K, K)'.
template <class Field>
void SolveAgainstDiagonalBlock(
    Int supernode, SupernodalLDLFactorization<Field>* factorization) {
  SupernodalLowerFactor<Field>& lower_factor = factorization->lower_factor;
  const SupernodalDiagonalFactor<Field>& diagonal_factor =
      factorization->diagonal_factor;
  const Int supernode_size = factorization->supernode_sizes[supernode];

  const Int diag_value_beg = diagonal_factor.value_offsets[supernode];
  const Field* diag_block = &diagonal_factor.values[diag_value_beg];

  const Int index_beg = lower_factor.index_offsets[supernode];
  const Int value_beg = lower_factor.value_offsets[supernode];
  const Int degree = lower_factor.index_offsets[supernode + 1] - index_beg;

  // TODO(Jack Poulson): Switch to a level-3 BLAS implementation.
  Field* lower_block = &lower_factor.values[value_beg];
  for (Int i = 0; i < degree; ++i) {
    Field* lower_row = &lower_block[i * supernode_size];

    // Solve against the unit-lower matrix L(K, K)', which is equivalent to
    // solving     conj(L(K, K)) x = y.
    //
    // Interleave with a subsequent solve against D(K, K).
    for (Int j = 0; j < supernode_size; ++j) {
      const Field* l_column = &diag_block[j * supernode_size];
      for (Int k = j + 1; k < supernode_size; ++k) {
        lower_row[k] -= Conjugate(l_column[k]) * lower_row[j];
      }
      lower_row[j] /= l_column[j];
    }
  }
}

template <class Field>
Int LeftLooking(const CoordinateMatrix<Field>& matrix,
                const std::vector<Int>& supernode_sizes,
                SupernodalLDLFactorization<Field>* factorization) {
  const Int num_supernodes = supernode_sizes.size();
  SupernodalLowerFactor<Field>& lower_factor = factorization->lower_factor;

  std::vector<Int> parents;
  LeftLookingSetup(matrix, supernode_sizes, &parents, factorization);

  // Set up a buffer for supernodal updates.
  Int max_supernode_size = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    max_supernode_size =
        std::max(max_supernode_size, factorization->supernode_sizes[supernode]);
  }
  std::vector<Field> scaled_adjoint_buffer(
      (max_supernode_size - 1) * (max_supernode_size - 1), Field{0});
  std::vector<Field> update_buffer(
      (max_supernode_size - 1) * (max_supernode_size - 1), Field{0});

  // A data structure for marking whether or not a supernode is in the pattern
  // of the active row of the lower-triangular factor.
  std::vector<Int> pattern_flags(num_supernodes);

  // An integer workspace for storing the supernodes in the current row
  // pattern.
  std::vector<Int> row_structure(num_supernodes);

  // Since we will sequentially access each of the entries in each block column
  // of  L during the updates of a supernode, we can avoid the need for binary
  // search by maintaining a separate counter for each supernode.
  std::vector<Int> intersection_ptrs(num_supernodes);
  std::vector<Int> index_ptrs(num_supernodes);
  std::vector<Int> value_ptrs(num_supernodes);

  Int num_pivots = 0;
  for (Int main_supernode = 0; main_supernode < num_supernodes;
       ++main_supernode) {
    const Int main_supernode_size =
        factorization->supernode_sizes[main_supernode];

    pattern_flags[main_supernode] = main_supernode;

    intersection_ptrs[main_supernode] =
        lower_factor.intersection_size_offsets[main_supernode];
    index_ptrs[main_supernode] = lower_factor.index_offsets[main_supernode];
    value_ptrs[main_supernode] = lower_factor.value_offsets[main_supernode];

    // Compute the supernodal row pattern.
    const Int num_packed = ComputeRowPattern(
        matrix, factorization->supernode_sizes,
        factorization->supernode_offsets, factorization->supernode_container,
        parents, main_supernode, pattern_flags.data(), row_structure.data());

    // for J = find(L(K, :))
    //   L(K:n, K) -= L(K:n, J) * (D(J, J) * L(K, J)')
    for (Int index = 0; index < num_packed; ++index) {
      const Int descendant_supernode = row_structure[index];
      CATAMARI_ASSERT(descendant_supernode < main_supernode,
                      "Looking into upper triangle.");
      const Int descendant_supernode_size =
          factorization->supernode_sizes[descendant_supernode];

      const Int descendant_main_intersection_size_beg =
          intersection_ptrs[descendant_supernode];
      const Int descendant_main_index_beg = index_ptrs[descendant_supernode];
      const Int descendant_main_value_beg = value_ptrs[descendant_supernode];
      const Int descendant_main_intersection_size =
          lower_factor
              .intersection_sizes[descendant_main_intersection_size_beg];

      FormScaledAdjoint(*factorization, descendant_supernode,
                        descendant_main_intersection_size,
                        descendant_main_value_beg,
                        scaled_adjoint_buffer.data());

      UpdateDiagonalBlock(main_supernode, descendant_supernode,
                          descendant_main_intersection_size,
                          descendant_main_index_beg, descendant_main_value_beg,
                          scaled_adjoint_buffer.data(), factorization,
                          update_buffer.data());
      intersection_ptrs[descendant_supernode]++;
      index_ptrs[descendant_supernode] += descendant_main_intersection_size;
      value_ptrs[descendant_supernode] +=
          descendant_main_intersection_size * descendant_supernode_size;

      // L(KNext:n, K) -= L(KNext:n, J) * (D(J, J) * L(K, J)')
      //                = L(KNext:n, J) * Z(J, K).
      Int descendant_active_intersection_size_beg =
          intersection_ptrs[descendant_supernode];
      Int descendant_active_index_beg = index_ptrs[descendant_supernode];
      Int descendant_active_value_beg = value_ptrs[descendant_supernode];
      Int main_active_intersection_size_beg =
          lower_factor.intersection_size_offsets[main_supernode];
      Int main_active_index_beg = lower_factor.index_offsets[main_supernode];
      Int main_active_value_beg = lower_factor.value_offsets[main_supernode];
      const Int descendant_index_guard =
          lower_factor.index_offsets[descendant_supernode + 1];
      while (descendant_active_index_beg != descendant_index_guard) {
        const Int descendant_active_intersection_size =
            lower_factor
                .intersection_sizes[descendant_active_intersection_size_beg];
        UpdateSubdiagonalBlock(
            main_supernode, descendant_supernode,
            descendant_main_intersection_size, descendant_main_index_beg,
            descendant_active_intersection_size, descendant_active_index_beg,
            descendant_active_value_beg, scaled_adjoint_buffer.data(),
            factorization, &main_active_intersection_size_beg,
            &main_active_index_beg, &main_active_value_beg,
            update_buffer.data());

        ++descendant_active_intersection_size_beg;
        descendant_active_index_beg += descendant_active_intersection_size;
        descendant_active_value_beg +=
            descendant_active_intersection_size * descendant_supernode_size;
      }
    }

    const Int num_supernode_pivots =
        FactorDiagonalBlock(main_supernode, factorization);
    num_pivots += num_supernode_pivots;
    if (num_supernode_pivots < main_supernode_size) {
      return num_pivots;
    }

    SolveAgainstDiagonalBlock(main_supernode, factorization);
  }

  return num_pivots;
}

}  // namespace supernodal_ldl

template <class Field>
Int LDL(const CoordinateMatrix<Field>& matrix,
        const std::vector<Int>& supernode_sizes, LDLAlgorithm algorithm,
        SupernodalLDLFactorization<Field>* factorization) {
  return supernodal_ldl::LeftLooking(matrix, supernode_sizes, factorization);
}

template <class Field>
void LDLSolve(const SupernodalLDLFactorization<Field>& factorization,
              std::vector<Field>* vector) {
  UnitLowerTriangularSolve(factorization, vector);
  DiagonalSolve(factorization, vector);
  UnitLowerAdjointTriangularSolve(factorization, vector);
}

template <class Field>
void UnitLowerTriangularSolve(
    const SupernodalLDLFactorization<Field>& factorization,
    std::vector<Field>* vector) {
  const Int num_supernodes = factorization.supernode_sizes.size();
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int supernode_size = factorization.supernode_sizes[supernode];
    const Int supernode_beg = factorization.supernode_offsets[supernode];

    // Handle the diagonal-block portion of the supernode.
    const Int diag_value_beg =
        factorization.diagonal_factor.value_offsets[supernode];
    const Field* diag_block =
        &factorization.diagonal_factor.values[diag_value_beg];
    for (Int j = 0; j < supernode_size; ++j) {
      const Int column = supernode_beg + j;
      for (Int i = j + 1; i < supernode_size; ++i) {
        const Int row = supernode_beg + i;
        const Field& value = diag_block[i + j * supernode_size];
        (*vector)[row] -= value * (*vector)[column];
      }
    }

    // Handle the below-diagonal portion of this supernode.
    const Int index_beg = factorization.lower_factor.index_offsets[supernode];
    const Int degree =
        factorization.lower_factor.index_offsets[supernode + 1] - index_beg;
    const Int value_beg = factorization.lower_factor.value_offsets[supernode];
    for (Int i = 0; i < degree; ++i) {
      const Int row = factorization.lower_factor.indices[index_beg + i];
      const Field* values =
          &factorization.lower_factor.values[value_beg + i * supernode_size];
      for (Int j = 0; j < supernode_size; ++j) {
        const Field& value = values[j];
        const Int column = supernode_beg + j;
        (*vector)[row] -= value * (*vector)[column];
      }
    }
  }
}

template <class Field>
void DiagonalSolve(const SupernodalLDLFactorization<Field>& factorization,
                   std::vector<Field>* vector) {
  const Int num_supernodes = factorization.supernode_sizes.size();
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int supernode_size = factorization.supernode_sizes[supernode];
    const Int supernode_beg = factorization.supernode_offsets[supernode];

    // Handle the diagonal-block portion of the supernode.
    const Int diag_value_beg =
        factorization.diagonal_factor.value_offsets[supernode];
    const Field* diag_block =
        &factorization.diagonal_factor.values[diag_value_beg];
    for (Int j = 0; j < supernode_size; ++j) {
      const Int column = supernode_beg + j;
      const Field& value = diag_block[j + j * supernode_size];
      (*vector)[column] /= value;
    }
  }
}

template <class Field>
void UnitLowerAdjointTriangularSolve(
    const SupernodalLDLFactorization<Field>& factorization,
    std::vector<Field>* vector) {
  const Int num_supernodes = factorization.supernode_sizes.size();
  for (Int supernode = num_supernodes - 1; supernode >= 0; --supernode) {
    const Int supernode_size = factorization.supernode_sizes[supernode];
    const Int supernode_beg = factorization.supernode_offsets[supernode];

    // Handle the diagonal-block portion of the supernode.
    const Int diag_value_beg =
        factorization.diagonal_factor.value_offsets[supernode];
    const Field* diag_block =
        &factorization.diagonal_factor.values[diag_value_beg];
    for (Int j = supernode_size - 1; j >= 0; --j) {
      const Int column = supernode_beg + j;
      for (Int i = j + 1; i < supernode_size; ++i) {
        const Int row = supernode_beg + i;
        const Field& value = diag_block[i + j * supernode_size];
        (*vector)[column] -= Conjugate(value) * (*vector)[row];
      }
    }

    // Handle the below-diagonal portion of this supernode.
    const Int index_beg = factorization.lower_factor.index_offsets[supernode];
    const Int degree =
        factorization.lower_factor.index_offsets[supernode + 1] - index_beg;
    const Int value_beg = factorization.lower_factor.value_offsets[supernode];
    for (Int i = 0; i < degree; ++i) {
      const Int row = factorization.lower_factor.indices[index_beg + i];
      const Field* values =
          &factorization.lower_factor.values[value_beg + i * supernode_size];
      for (Int j = supernode_size - 1; j >= 0; --j) {
        const Field& value = values[j];
        const Int column = supernode_beg + j;
        (*vector)[column] -= Conjugate(value) * (*vector)[row];
      }
    }
  }
}

}  // namespace catamari

#endif  // ifndef CATAMARI_SUPERNODAL_LDL_IMPL_H_
