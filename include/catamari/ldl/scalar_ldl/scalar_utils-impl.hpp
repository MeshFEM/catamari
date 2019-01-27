/*
 * Copyright (c) 2018-2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SCALAR_LDL_SCALAR_UTILS_IMPL_H_
#define CATAMARI_LDL_SCALAR_LDL_SCALAR_UTILS_IMPL_H_

#include "catamari/ldl/scalar_ldl/scalar_utils.hpp"

namespace catamari {
namespace scalar_ldl {

template <class Field>
void EliminationForestAndDegrees(const CoordinateMatrix<Field>& matrix,
                                 const SymmetricOrdering& ordering,
                                 Buffer<Int>* parents, Buffer<Int>* degrees) {
  const Int num_rows = matrix.NumRows();

  // Initialize all of the parent indices as unset.
  parents->Resize(num_rows, -1);

  // A data structure for marking whether or not an index is in the pattern
  // of the active row of the lower-triangular factor.
  Buffer<Int> pattern_flags(num_rows);

  // Initialize the number of subdiagonal entries that will be stored into
  // each column.
  degrees->Resize(num_rows, 0);

  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  const bool have_permutation = !ordering.permutation.Empty();
  for (Int row = 0; row < num_rows; ++row) {
    pattern_flags[row] = row;

    const Int orig_row =
        have_permutation ? ordering.inverse_permutation[row] : row;
    const Int row_beg = matrix.RowEntryOffset(orig_row);
    const Int row_end = matrix.RowEntryOffset(orig_row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      Int column =
          have_permutation ? ordering.permutation[entry.column] : entry.column;

      // We are traversing the strictly lower triangle and know that the
      // indices are sorted.
      if (column >= row) {
        if (have_permutation) {
          continue;
        } else {
          break;
        }
      }

      // Look for new entries in the pattern by walking up to the root of this
      // subtree of the elimination forest from index 'column'. Any unset
      // parent pointers can be filled in during the traversal, as the current
      // row index would then be the parent.
      while (pattern_flags[column] != row) {
        // Mark index 'column' as in the pattern of row 'row'.
        pattern_flags[column] = row;
        ++(*degrees)[column];

        if ((*parents)[column] == -1) {
          // This is the first occurrence of 'column' in a row pattern.
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

#ifdef _OPENMP
template <class Field>
void MultithreadedEliminationForestAndDegreesRecursion(
    const CoordinateMatrix<Field>& matrix, const SymmetricOrdering& ordering,
    Int root, Buffer<Int>* parents, Buffer<Int>* degrees,
    Buffer<Int>* pattern_flags) {
  const Int child_beg = ordering.assembly_forest.child_offsets[root];
  const Int child_end = ordering.assembly_forest.child_offsets[root + 1];
  #pragma omp taskgroup
  for (Int index = child_beg; index < child_end; ++index) {
    const Int child = ordering.assembly_forest.children[index];

    #pragma omp task default(none) firstprivate(child) \
        shared(matrix, ordering, parents, degrees, pattern_flags)
    MultithreadedEliminationForestAndDegreesRecursion(
        matrix, ordering, child, parents, degrees, pattern_flags);
  }

  const bool have_permutation = !ordering.permutation.Empty();
  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  const Int supernode_size = ordering.supernode_sizes[root];
  const Int supernode_offset = ordering.supernode_offsets[root];
  for (Int index = 0; index < supernode_size; ++index) {
    const Int row = supernode_offset + index;
    (*pattern_flags)[row] = row;

    const Int orig_row =
        have_permutation ? ordering.inverse_permutation[row] : row;
    const Int row_beg = matrix.RowEntryOffset(orig_row);
    const Int row_end = matrix.RowEntryOffset(orig_row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      Int column =
          have_permutation ? ordering.permutation[entry.column] : entry.column;

      // We are traversing the strictly lower triangle and know that the
      // indices are sorted.
      if (column >= row) {
        if (have_permutation) {
          continue;
        } else {
          break;
        }
      }

      // Look for new entries in the pattern by walking up to the root of this
      // subtree of the elimination forest from index 'column'. Any unset
      // parent pointers can be filled in during the traversal, as the current
      // row index would then be the parent.
      while ((*pattern_flags)[column] != row) {
        // Mark index 'column' as in the pattern of row 'row'.
        (*pattern_flags)[column] = row;
        ++(*degrees)[column];

        if ((*parents)[column] == -1) {
          // This is the first occurrence of 'column' in a row pattern.
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

template <class Field>
void MultithreadedEliminationForestAndDegrees(
    const CoordinateMatrix<Field>& matrix, const SymmetricOrdering& ordering,
    Buffer<Int>* parents, Buffer<Int>* degrees) {
  const Int num_rows = matrix.NumRows();

  // Initialize all of the parent indices as unset.
  parents->Resize(num_rows, -1);

  // A data structure for marking whether or not an index is in the pattern
  // of the active row of the lower-triangular factor.
  Buffer<Int> pattern_flags(num_rows);

  // Initialize the number of subdiagonal entries that will be stored into
  // each column.
  degrees->Resize(num_rows, 0);

  #pragma omp taskgroup
  for (const Int root : ordering.assembly_forest.roots) {
    #pragma omp task default(none) firstprivate(root) \
        shared(matrix, ordering, parents, degrees, pattern_flags)
    MultithreadedEliminationForestAndDegreesRecursion(
        matrix, ordering, root, parents, degrees, &pattern_flags);
  }
}
#endif  // ifdef _OPENMP

template <class Field>
Int ComputeRowPattern(const CoordinateMatrix<Field>& matrix,
                      const SymmetricOrdering& ordering,
                      const Buffer<Int>& parents, Int row, Int* pattern_flags,
                      Int* row_structure) {
  const bool have_permutation = !ordering.permutation.Empty();

  const Int orig_row =
      have_permutation ? ordering.inverse_permutation[row] : row;
  const Int row_beg = matrix.RowEntryOffset(orig_row);
  const Int row_end = matrix.RowEntryOffset(orig_row + 1);
  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  Int num_packed = 0;
  for (Int index = row_beg; index < row_end; ++index) {
    const MatrixEntry<Field>& entry = entries[index];
    Int column =
        have_permutation ? ordering.permutation[entry.column] : entry.column;

    if (column >= row) {
      if (have_permutation) {
        continue;
      } else {
        break;
      }
    }

    // Walk up to the root of the current subtree of the elimination
    // forest, stopping if we encounter a member already marked as in the
    // row pattern.
    while (pattern_flags[column] != row) {
      // Place 'column' into the pattern of row 'row'.
      row_structure[num_packed++] = column;
      pattern_flags[column] = row;

      // Move up to the parent in this subtree of the elimination forest.
      column = parents[column];
    }
  }
  return num_packed;
}

template <class Field>
Int ComputeTopologicalRowPatternAndScatterNonzeros(
    const CoordinateMatrix<Field>& matrix, const SymmetricOrdering& ordering,
    const Buffer<Int>& parents, Int row, Int* pattern_flags, Int* row_structure,
    Field* row_workspace) {
  const bool have_permutation = !ordering.permutation.Empty();
  Int start = matrix.NumRows();

  const Int orig_row =
      have_permutation ? ordering.inverse_permutation[row] : row;
  const Int row_beg = matrix.RowEntryOffset(orig_row);
  const Int row_end = matrix.RowEntryOffset(orig_row + 1);
  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int index = row_beg; index < row_end; ++index) {
    const MatrixEntry<Field>& entry = entries[index];
    Int column =
        have_permutation ? ordering.permutation[entry.column] : entry.column;

    if (column > row) {
      if (have_permutation) {
        continue;
      } else {
        break;
      }
    }

    // Scatter matrix(row, column) into row_workspace.
    row_workspace[column] = entry.value;

    // Walk up to the root of the current subtree of the elimination
    // forest, stopping if we encounter a member already marked as in the
    // row pattern.
    Int num_packed = 0;
    while (pattern_flags[column] != row) {
      // Place 'column' into the pattern of row 'row'.
      row_structure[num_packed++] = column;
      pattern_flags[column] = row;

      // Move up to the parent in this subtree of the elimination forest.
      column = parents[column];
    }

    // Pack this ancestral sequence into the back of the pattern.
    while (num_packed > 0) {
      row_structure[--start] = row_structure[--num_packed];
    }
  }
  return start;
}

template <class Field>
void FillStructureIndices(const CoordinateMatrix<Field>& matrix,
                          const SymmetricOrdering& ordering,
                          const AssemblyForest& forest,
                          const Buffer<Int>& degrees,
                          LowerStructure* lower_structure) {
  const Int num_rows = matrix.NumRows();
  const bool have_permutation = !ordering.permutation.Empty();

  // Set up the column offsets and allocate space (initializing the values of
  // the unit-lower and diagonal and all zeros).
  OffsetScan(degrees, &lower_structure->column_offsets);
  lower_structure->indices.Resize(lower_structure->column_offsets.Back());

  // A data structure for marking whether or not an index is in the pattern
  // of the active row of the lower-triangular factor.
  Buffer<Int> pattern_flags(num_rows);

  // A set of pointers for keeping track of where to insert column pattern
  // indices.
  Buffer<Int> column_ptrs(num_rows);

  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    pattern_flags[row] = row;
    column_ptrs[row] = lower_structure->column_offsets[row];

    const Int orig_row =
        have_permutation ? ordering.inverse_permutation[row] : row;
    const Int row_beg = matrix.RowEntryOffset(orig_row);
    const Int row_end = matrix.RowEntryOffset(orig_row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      Int column =
          have_permutation ? ordering.permutation[entry.column] : entry.column;

      // We are traversing the strictly lower triangle and know that the
      // indices are sorted.
      if (column >= row) {
        if (have_permutation) {
          continue;
        } else {
          break;
        }
      }

      // Look for new entries in the pattern by walking up to the root of this
      // subtree of the elimination forest from index 'column'.
      while (pattern_flags[column] != row) {
        // Mark index 'column' as in the pattern of row 'row'.
        pattern_flags[column] = row;
        lower_structure->indices[column_ptrs[column]++] = row;

        // Move up to the parent in this subtree of the elimination forest.
        // Moving to the parent will increase the index (but remain bounded
        // from above by 'row').
        column = forest.parents[column];
      }
    }
  }
}

#ifdef _OPENMP
template <class Field>
void MultithreadedFillStructureIndicesRecursion(
    const CoordinateMatrix<Field>& matrix, const SymmetricOrdering& ordering,
    const AssemblyForest& scalar_forest, Int root,
    LowerStructure* lower_structure,
    Buffer<Buffer<Int>>* private_pattern_flags) {
  const Int child_beg = ordering.assembly_forest.child_offsets[root];
  const Int child_end = ordering.assembly_forest.child_offsets[root + 1];
  #pragma omp taskgroup
  for (Int index = child_beg; index < child_end; ++index) {
    const Int child = ordering.assembly_forest.children[index];

    #pragma omp task default(none) firstprivate(child)           \
        shared(matrix, ordering, scalar_forest, lower_structure, \
            private_pattern_flags)
    MultithreadedFillStructureIndicesRecursion(matrix, ordering, scalar_forest,
                                               child, lower_structure,
                                               private_pattern_flags);
  }

  const int thread = omp_get_thread_num();
  Buffer<Int>& pattern_flags = (*private_pattern_flags)[thread];

  const Int supernode_size = ordering.supernode_sizes[root];
  const Int supernode_offset = ordering.supernode_offsets[root];
  const bool have_permutation = !ordering.permutation.Empty();
  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int column = supernode_offset;
       column < supernode_offset + supernode_size; ++column) {
    // Form this node's structure by unioning that of its direct children
    // (removing portions that intersect this supernode).
    const Int scalar_child_beg = scalar_forest.child_offsets[column];
    const Int scalar_child_end = scalar_forest.child_offsets[column + 1];
    const Int struct_beg = lower_structure->column_offsets[column];
    const Int struct_end = lower_structure->column_offsets[column + 1];
    Int struct_ptr = struct_beg;
    for (Int child_index = scalar_child_beg; child_index < scalar_child_end;
         ++child_index) {
      const Int child = scalar_forest.children[child_index];
      const Int child_struct_beg = lower_structure->column_offsets[child];
      const Int child_struct_end = lower_structure->column_offsets[child + 1];
      for (Int child_struct_ptr = child_struct_beg;
           child_struct_ptr != child_struct_end; ++child_struct_ptr) {
        const Int row = lower_structure->indices[child_struct_ptr];
        if (row == column) {
          continue;
        }

        if (pattern_flags[row] != column) {
          CATAMARI_ASSERT(row > column, "row was < column.");
          pattern_flags[row] = column;
          lower_structure->indices[struct_ptr++] = row;
        }
      }
    }

    // Incorporate this column's structure.
    const Int orig_column =
        have_permutation ? ordering.inverse_permutation[column] : column;
    const Int column_beg = matrix.RowEntryOffset(orig_column);
    const Int column_end = matrix.RowEntryOffset(orig_column + 1);
    for (Int index = column_beg; index < column_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      const Int row =
          have_permutation ? ordering.permutation[entry.column] : entry.column;
      if (row <= column) {
        continue;
      }

      if (pattern_flags[row] != column) {
        pattern_flags[row] = column;
        lower_structure->indices[struct_ptr++] = row;
      }
    }

    CATAMARI_ASSERT(struct_ptr <= struct_end, "Stored too many indices.");
    CATAMARI_ASSERT(struct_ptr >= struct_end, "Stored too few indices.");

    // TODO(Jack Poulson): Incorporate a parallel sort?
    std::sort(&lower_structure->indices[struct_beg],
              &lower_structure->indices[struct_end]);
  }
}

template <class Field>
void MultithreadedFillStructureIndices(const CoordinateMatrix<Field>& matrix,
                                       const SymmetricOrdering& ordering,
                                       const AssemblyForest& forest,
                                       const Buffer<Int>& degrees,
                                       LowerStructure* lower_structure) {
  const Int num_rows = matrix.NumRows();

  // Set up the column offsets and allocate space (initializing the values of
  // the unit-lower and diagonal and all zeros).
  OffsetScan(degrees, &lower_structure->column_offsets);
  lower_structure->indices.Resize(lower_structure->column_offsets.Back());

  // A data structure for marking whether or not a node is in the pattern of
  // the active row of the lower-triangular factor. Each thread potentially
  // needs its own since different subtrees can have intersecting structure.
  const int max_threads = omp_get_max_threads();
  Buffer<Buffer<Int>> private_pattern_flags(max_threads);
  for (int thread = 0; thread < max_threads; ++thread) {
    private_pattern_flags[thread].Resize(num_rows, -1);
  }

  #pragma omp taskgroup
  for (const Int root : ordering.assembly_forest.roots) {
    #pragma omp task default(none) firstprivate(root)     \
        shared(matrix, ordering, forest, lower_structure, \
            private_pattern_flags)
    MultithreadedFillStructureIndicesRecursion(matrix, ordering, forest, root,
                                               lower_structure,
                                               &private_pattern_flags);
  }
}
#endif  // ifdef _OPENMP

}  // namespace scalar_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_LDL_SCALAR_LDL_SCALAR_UTILS_IMPL_H_
