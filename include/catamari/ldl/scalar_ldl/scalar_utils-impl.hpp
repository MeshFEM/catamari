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
    Int root, bool keep_structures, Buffer<Int>* parents, Buffer<Int>* degrees,
    Buffer<std::vector<Int>>* children_list, Buffer<Buffer<Int>>* structures,
    Buffer<Buffer<Int>>* private_pattern_flags,
    Buffer<Buffer<Int>>* private_tmp_structures) {
  const Int order_child_beg = ordering.assembly_forest.child_offsets[root];
  const Int order_child_end = ordering.assembly_forest.child_offsets[root + 1];
  #pragma omp taskgroup
  for (Int index = order_child_beg; index < order_child_end; ++index) {
    const Int child = ordering.assembly_forest.children[index];

    #pragma omp task default(none) firstprivate(child, keep_structures)       \
        shared(matrix, ordering, parents, degrees, children_list, structures, \
        private_pattern_flags, private_tmp_structures)
    MultithreadedEliminationForestAndDegreesRecursion(
        matrix, ordering, child, keep_structures, parents, degrees,
        children_list, structures, private_pattern_flags,
        private_tmp_structures);
  }

  const int thread = omp_get_thread_num();
  Buffer<Int>& pattern_flags = (*private_pattern_flags)[thread];
  Buffer<Int>& tmp_structure = (*private_tmp_structures)[thread];

  const Int supernode_size = ordering.supernode_sizes[root];
  const Int supernode_offset = ordering.supernode_offsets[root];
  const bool have_permutation = !ordering.permutation.Empty();
  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int column = supernode_offset;
       column < supernode_offset + supernode_size; ++column) {
    const Int orig_column =
        have_permutation ? ordering.inverse_permutation[column] : column;
    const Int column_beg = matrix.RowEntryOffset(orig_column);
    const Int column_end = matrix.RowEntryOffset(orig_column + 1);

    Buffer<Int>& structure = (*structures)[column];
    std::vector<Int>& children = (*children_list)[column];
    const Int num_children = children.size();

    Int degree = 0;
    Int parent = -1;
    if (num_children == 0) {
      // Incorporate this column's structure.
      for (Int index = column_beg; index < column_end; ++index) {
        const MatrixEntry<Field>& entry = entries[index];
        const Int row = have_permutation ? ordering.permutation[entry.column]
                                         : entry.column;
        if (row > column) {
          if (!degree || row < parent) {
            parent = row;
          }
          tmp_structure[degree++] = row;
        }
      }
    } else {
      // Form this node's structure by unioning that of its direct children
      // (removing portions that intersect this column).
      //
      // We specially handle the first child to avoid unnecessary reads from
      // 'pattern_flags'.
      if (num_children >= 1) {
        const Int child = children[0];
        const Buffer<Int>& child_struct = (*structures)[child];
        for (const Int& row : child_struct) {
          if (row != column) {
            CATAMARI_ASSERT(row > column, "row was < column.");
            pattern_flags[row] = column;
            if (!degree || row < parent) {
              parent = row;
            }
            tmp_structure[degree++] = row;
          }
        }
      }
      for (Int child_index = 1; child_index < num_children; ++child_index) {
        const Int child = children[child_index];
        const Buffer<Int>& child_struct = (*structures)[child];
        for (const Int& row : child_struct) {
          if (row != column && pattern_flags[row] != column) {
            CATAMARI_ASSERT(row > column, "row was < column.");
            pattern_flags[row] = column;
            if (!degree || row < parent) {
              parent = row;
            }
            tmp_structure[degree++] = row;
          }
        }
      }

      // Incorporate this column's structure.
      for (Int index = column_beg; index < column_end; ++index) {
        const MatrixEntry<Field>& entry = entries[index];
        const Int row = have_permutation ? ordering.permutation[entry.column]
                                         : entry.column;
        if (row > column && pattern_flags[row] != column) {
          if (!degree || row < parent) {
            parent = row;
          }
          tmp_structure[degree++] = row;
        }
      }
    }
    structure.Resize(degree);
    std::copy(tmp_structure.begin(), tmp_structure.begin() + degree,
              structure.begin());

    (*parents)[column] = parent;
    (*degrees)[column] = degree;
    if (parent >= 0) {
      (*children_list)[parent].push_back(column);
    }

    // Free the resources of this subtree.
    if (!keep_structures) {
      for (const Int& child : children) {
        (*structures)[child].Clear();
      }
    }
    children.clear();
    children.shrink_to_fit();
  }
}

template <class Field>
void MultithreadedEliminationForestAndDegrees(
    const CoordinateMatrix<Field>& matrix, const SymmetricOrdering& ordering,
    Buffer<Int>* parents, Buffer<Int>* degrees) {
  const Int num_rows = matrix.NumRows();
  parents->Resize(num_rows);
  degrees->Resize(num_rows);

  // TODO(Jack Poulson): Make the children reservation configurable.
  const Int children_reservation = 4;
  Buffer<std::vector<Int>> children_list(num_rows);
  for (Int row = 0; row < num_rows; ++row) {
    children_list[row].reserve(children_reservation);
  }

  Buffer<Buffer<Int>> structures(num_rows);

  // A data structure for marking whether or not a node is in the pattern of
  // the active row of the lower-triangular factor. Each thread potentially
  // needs its own since different subtrees can have intersecting structure.
  const int max_threads = omp_get_max_threads();
  Buffer<Buffer<Int>> private_pattern_flags(max_threads);
  for (int thread = 0; thread < max_threads; ++thread) {
    private_pattern_flags[thread].Resize(num_rows, -1);
  }

  Buffer<Buffer<Int>> private_tmp_structures(max_threads);
  for (int thread = 0; thread < max_threads; ++thread) {
    private_tmp_structures[thread].Resize(num_rows - 1);
  }

  const bool keep_structures = true;

  #pragma omp taskgroup
  for (const Int root : ordering.assembly_forest.roots) {
    #pragma omp task default(none) firstprivate(root, keep_structures)        \
        shared(matrix, ordering, parents, degrees, children_list, structures, \
            private_pattern_flags, private_tmp_structures)
    MultithreadedEliminationForestAndDegreesRecursion(
        matrix, ordering, root, keep_structures, parents, degrees,
        &children_list, &structures, &private_pattern_flags,
        &private_tmp_structures);
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
    column_ptrs[row] = lower_structure->ColumnOffset(row);

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
  const Int order_child_beg = ordering.assembly_forest.child_offsets[root];
  const Int order_child_end = ordering.assembly_forest.child_offsets[root + 1];
  #pragma omp taskgroup
  for (Int index = order_child_beg; index < order_child_end; ++index) {
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
    const Int orig_column =
        have_permutation ? ordering.inverse_permutation[column] : column;
    const Int column_beg = matrix.RowEntryOffset(orig_column);
    const Int column_end = matrix.RowEntryOffset(orig_column + 1);

    Int* struct_ptr = lower_structure->ColumnBeg(column);
    const Int child_beg = scalar_forest.child_offsets[column];
    const Int child_end = scalar_forest.child_offsets[column + 1];
    const Int num_children = child_end - child_beg;

    if (num_children == 0) {
      // Incorporate this column's structure.
      for (Int index = column_beg; index < column_end; ++index) {
        const MatrixEntry<Field>& entry = entries[index];
        const Int row = have_permutation ? ordering.permutation[entry.column]
                                         : entry.column;
        if (row > column) {
          *(struct_ptr++) = row;
        }
      }
    } else {
      // Form this node's structure by unioning that of its direct children
      // (removing portions that intersect this column).
      //
      // We specially handle the first child to avoid unnecessary reads from
      // 'pattern_flags'.
      if (num_children >= 1) {
        const Int child_index = child_beg;
        const Int child = scalar_forest.children[child_index];
        const Int* child_struct_beg = lower_structure->ColumnBeg(child);
        const Int* child_struct_end = lower_structure->ColumnEnd(child);
        for (const Int* child_struct_ptr = child_struct_beg;
             child_struct_ptr != child_struct_end; ++child_struct_ptr) {
          const Int& row = *child_struct_ptr;
          if (row != column) {
            CATAMARI_ASSERT(row > column, "row was < column.");
            pattern_flags[row] = column;
            *(struct_ptr++) = row;
          }
        }
      }
      for (Int child_index = child_beg + 1; child_index < child_end;
           ++child_index) {
        const Int child = scalar_forest.children[child_index];
        const Int* child_struct_beg = lower_structure->ColumnBeg(child);
        const Int* child_struct_end = lower_structure->ColumnEnd(child);
        for (const Int* child_struct_ptr = child_struct_beg;
             child_struct_ptr != child_struct_end; ++child_struct_ptr) {
          const Int& row = *child_struct_ptr;
          if (row != column && pattern_flags[row] != column) {
            CATAMARI_ASSERT(row > column, "row was < column.");
            pattern_flags[row] = column;
            *(struct_ptr++) = row;
          }
        }
      }

      // Incorporate this column's structure.
      for (Int index = column_beg; index < column_end; ++index) {
        const MatrixEntry<Field>& entry = entries[index];
        const Int row = have_permutation ? ordering.permutation[entry.column]
                                         : entry.column;
        if (row > column && pattern_flags[row] != column) {
          *(struct_ptr++) = row;
        }
      }
    }
  }
}

template <class Field>
void MultithreadedFillStructureIndices(const CoordinateMatrix<Field>& matrix,
                                       const SymmetricOrdering& ordering,
                                       AssemblyForest* forest,
                                       LowerStructure* lower_structure) {
  const Int num_rows = matrix.NumRows();

  // TODO(Jack Poulson): Make this configurable.
  const bool preallocate = true;

  // A data structure for marking whether or not a node is in the pattern of
  // the active row of the lower-triangular factor. Each thread potentially
  // needs its own since different subtrees can have intersecting structure.
  const int max_threads = omp_get_max_threads();
  Buffer<Buffer<Int>> private_pattern_flags(max_threads);
  for (int thread = 0; thread < max_threads; ++thread) {
    private_pattern_flags[thread].Resize(num_rows, -1);
  }

  Buffer<Int> degrees;
  Buffer<Buffer<Int>> structures(num_rows);
  {
    Buffer<Int>* parents = &forest->parents;
    parents->Resize(num_rows);
    degrees.Resize(num_rows);

    // TODO(Jack Poulson): Make the children reservation configurable.
    const Int children_reservation = 4;
    Buffer<std::vector<Int>> children_list(num_rows);
    for (Int row = 0; row < num_rows; ++row) {
      children_list[row].reserve(children_reservation);
    }

    Buffer<Buffer<Int>> private_tmp_structures(max_threads);
    for (int thread = 0; thread < max_threads; ++thread) {
      private_tmp_structures[thread].Resize(num_rows - 1);
    }

    const bool keep_structures = !preallocate;

    #pragma omp taskgroup
    for (const Int root : ordering.assembly_forest.roots) {
      #pragma omp task default(none) firstprivate(root, keep_structures) \
          shared(matrix, ordering, parents, degrees, children_list,      \
              structures, private_pattern_flags, private_tmp_structures)
      MultithreadedEliminationForestAndDegreesRecursion(
          matrix, ordering, root, keep_structures, parents, &degrees,
          &children_list, &structures, &private_pattern_flags,
          &private_tmp_structures);
    }
  }
  forest->FillFromParents();

  // Set up the column offsets and allocate space (initializing the values of
  // the unit-lower and diagonal and all zeros).
  OffsetScan(degrees, &lower_structure->column_offsets);
  lower_structure->indices.Resize(lower_structure->column_offsets.Back());

  if (preallocate) {
    // Reset the private pattern flags.
    for (int thread = 0; thread < max_threads; ++thread) {
      private_pattern_flags[thread].Resize(num_rows, -1);
    }

    #pragma omp taskgroup
    for (const Int root : ordering.assembly_forest.roots) {
      #pragma omp task default(none) firstprivate(root)     \
          shared(matrix, ordering, forest, lower_structure, \
              private_pattern_flags)
      MultithreadedFillStructureIndicesRecursion(matrix, ordering, *forest,
                                                 root, lower_structure,
                                                 &private_pattern_flags);
    }

    // Sort the structures.
    const Int grain_size = 500;  // TODO(Jack Poulson): Make this configurable.
    #pragma omp taskgroup
    for (Int j = 0; j < num_rows; j += grain_size) {
      #pragma omp task default(none) firstprivate(j, grain_size) \
          shared(lower_structure)
      {
        const Int column_end = std::min(num_rows, j + grain_size);
        for (Int column = j; column < column_end; ++column) {
          std::sort(lower_structure->ColumnBeg(column),
                    lower_structure->ColumnEnd(column));
        }
      }
    }
  } else {
    // Fill and sort the structures.
    const Int grain_size = 500;  // TODO(Jack Poulson): Make this configurable.
    #pragma omp taskgroup
    for (Int j = 0; j < num_rows; j += grain_size) {
      #pragma omp task default(none) firstprivate(j, grain_size) \
          shared(structures, lower_structure)
      {
        const Int column_end = std::min(num_rows, j + grain_size);
        for (Int column = j; column < column_end; ++column) {
          std::copy(structures[column].begin(), structures[column].end(),
                    lower_structure->ColumnBeg(column));
          std::sort(lower_structure->ColumnBeg(column),
                    lower_structure->ColumnEnd(column));
        }
      }
    }
  }
}
#endif  // ifdef _OPENMP

}  // namespace scalar_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_LDL_SCALAR_LDL_SCALAR_UTILS_IMPL_H_
