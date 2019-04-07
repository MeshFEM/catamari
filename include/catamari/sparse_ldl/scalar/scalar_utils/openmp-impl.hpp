/*
 * Copyright (c) 2018-2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SCALAR_SCALAR_UTILS_OPENMP_IMPL_H_
#define CATAMARI_SPARSE_LDL_SCALAR_SCALAR_UTILS_OPENMP_IMPL_H_
#ifdef CATAMARI_OPENMP

#include "catamari/sparse_ldl/scalar/scalar_utils.hpp"

namespace catamari {
namespace scalar_ldl {

template <class Field>
void OpenMPEliminationForestAndDegreesRecursion(
    const CoordinateMatrix<Field>& matrix, const SymmetricOrdering& ordering,
    Int root, bool keep_structures, Buffer<Int>* parents, Buffer<Int>* degrees,
    Buffer<Buffer<std::vector<Int>>>* private_children_lists,
    Buffer<Buffer<Int>>* structures, Buffer<Buffer<Int>>* private_pattern_flags,
    Buffer<Buffer<Int>>* private_tmp_structures) {
  const Int order_child_beg = ordering.assembly_forest.child_offsets[root];
  const Int order_child_end = ordering.assembly_forest.child_offsets[root + 1];
  #pragma omp taskgroup
  for (Int index = order_child_beg; index < order_child_end; ++index) {
    const Int child = ordering.assembly_forest.children[index];

    #pragma omp task default(none) firstprivate(child, keep_structures)    \
        shared(matrix, ordering, parents, degrees, private_children_lists, \
        structures, private_pattern_flags, private_tmp_structures)
    OpenMPEliminationForestAndDegreesRecursion(
        matrix, ordering, child, keep_structures, parents, degrees,
        private_children_lists, structures, private_pattern_flags,
        private_tmp_structures);
  }

  const int thread = omp_get_thread_num();
  Buffer<std::vector<Int>>& children_list = (*private_children_lists)[thread];
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

    // Merge all threads' children into this thread's list.
    std::vector<Int>& children = children_list[column];
    {
      std::size_t num_total_children = 0;
      for (std::size_t t = 0; t < private_children_lists->Size(); ++t) {
        num_total_children += (*private_children_lists)[t][column].size();
      }

      const std::size_t num_local_children = children.size();
      if (num_total_children > num_local_children) {
        children.resize(num_total_children);
        std::size_t offset = num_local_children;
        for (std::size_t t = 0; t < private_children_lists->Size(); ++t) {
          if (t == std::size_t(thread)) {
            continue;
          }
          std::vector<Int>& other_children =
              (*private_children_lists)[t][column];
          std::copy(other_children.begin(), other_children.end(),
                    children.begin() + offset);
          offset += other_children.size();
          other_children.clear();
          other_children.shrink_to_fit();
        }
        CATAMARI_ASSERT(offset == num_total_children, "Invalid child offset");
      }
    }
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
            CATAMARI_ASSERT(row > column, "row was < column (EFaDR).");
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
            CATAMARI_ASSERT(row > column, "row was < column (EFaDR).");
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
      children_list[parent].push_back(column);
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
void OpenMPEliminationForestAndDegrees(const CoordinateMatrix<Field>& matrix,
                                       const SymmetricOrdering& ordering,
                                       Buffer<Int>* parents,
                                       Buffer<Int>* degrees) {
  const Int num_rows = matrix.NumRows();
  const int max_threads = omp_get_max_threads();
  parents->Resize(num_rows);
  degrees->Resize(num_rows);

  Buffer<Buffer<std::vector<Int>>> private_children_lists(max_threads);

  // A data structure for marking whether or not a node is in the pattern of
  // the active row of the lower-triangular factor. Each thread potentially
  // needs its own since different subtrees can have intersecting structure.
  Buffer<Buffer<Int>> private_pattern_flags(max_threads);

  Buffer<Buffer<Int>> private_tmp_structures(max_threads);

  #pragma omp taskgroup
  for (int t = 0; t < max_threads; ++t) {
    #pragma omp task default(none) firstprivate(t, num_rows)  \
        shared(private_children_lists, private_pattern_flags, \
            private_tmp_structures)
    {
      private_children_lists[t].Resize(num_rows);
      private_pattern_flags[t].Resize(num_rows, -1);
      private_tmp_structures[t].Resize(num_rows - 1);
    }
  }

  const bool keep_structures = false;
  Buffer<Buffer<Int>> structures(num_rows);

  #pragma omp taskgroup
  for (const Int root : ordering.assembly_forest.roots) {
    #pragma omp task default(none) firstprivate(root, keep_structures)     \
        shared(matrix, ordering, parents, degrees, private_children_lists, \
            structures, private_pattern_flags, private_tmp_structures)
    OpenMPEliminationForestAndDegreesRecursion(
        matrix, ordering, root, keep_structures, parents, degrees,
        &private_children_lists, &structures, &private_pattern_flags,
        &private_tmp_structures);
  }
}

template <class Field>
void OpenMPFillStructureIndicesRecursion(
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
    OpenMPFillStructureIndicesRecursion(matrix, ordering, scalar_forest, child,
                                        lower_structure, private_pattern_flags);
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
      CATAMARI_ASSERT(
          struct_ptr <= lower_structure->ColumnEnd(column),
          "struct_ptr was past the end of the column (no children)");
      CATAMARI_ASSERT(
          struct_ptr >= lower_structure->ColumnEnd(column),
          "struct_ptr was before the end of the column (no children)");
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
            CATAMARI_ASSERT(row > column, "row was < column (FSIR first).");
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
            CATAMARI_ASSERT(row > column, "row was < column (FSIR later).");
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

      CATAMARI_ASSERT(
          struct_ptr <= lower_structure->ColumnEnd(column),
          "struct_ptr was past the end of the column (multi children)");
      CATAMARI_ASSERT(
          struct_ptr >= lower_structure->ColumnEnd(column),
          "struct_ptr was before the end of the column (multi children)");
    }
  }
}

template <class Field>
void OpenMPFillStructureIndices(const CoordinateMatrix<Field>& matrix,
                                const SymmetricOrdering& ordering,
                                AssemblyForest* forest,
                                LowerStructure* lower_structure,
                                bool preallocate, int sort_grain_size) {
  const Int num_rows = matrix.NumRows();
  const int max_threads = omp_get_max_threads();

  // TODO(Jack Poulson): Add support for using the degrees computed from
  // a MinimumDegree reordering to avoid the initial degree computation.

  // A data structure for marking whether or not a node is in the pattern of
  // the active row of the lower-triangular factor. Each thread potentially
  // needs its own since different subtrees can have intersecting structure.
  Buffer<Buffer<Int>> private_pattern_flags(max_threads);

  // Build the elimination forest and degrees.
  Buffer<Int> degrees;
  Buffer<Buffer<Int>> structures(num_rows);
  {
    Buffer<Int>* parents = &forest->parents;
    parents->Resize(num_rows);
    degrees.Resize(num_rows);

    Buffer<Buffer<std::vector<Int>>> private_children_lists(max_threads);
    Buffer<Buffer<Int>> private_tmp_structures(max_threads);

    const bool keep_structures = !preallocate;

    // Allocate/initialize the needed buffers in parallel.
    #pragma omp taskgroup
    for (int t = 0; t < max_threads; ++t) {
      #pragma omp task default(none) firstprivate(t, num_rows)  \
          shared(private_pattern_flags, private_children_lists, \
              private_tmp_structures)
      {
        private_pattern_flags[t].Resize(num_rows, -1);
        private_children_lists[t].Resize(num_rows);
        private_tmp_structures[t].Resize(num_rows - 1);
      }
    }

    // Build the up-links and degrees of the elimination forest.
    #pragma omp taskgroup
    for (const Int root : ordering.assembly_forest.roots) {
      #pragma omp task default(none) firstprivate(root, keep_structures)     \
          shared(matrix, ordering, parents, degrees, private_children_lists, \
              structures, private_pattern_flags, private_tmp_structures)
      OpenMPEliminationForestAndDegreesRecursion(
          matrix, ordering, root, keep_structures, parents, &degrees,
          &private_children_lists, &structures, &private_pattern_flags,
          &private_tmp_structures);
    }
  }
  forest->FillFromParents();

  // Set up the column offsets and allocate space (initializing the values of
  // the unit-lower and diagonal and all zeros).
  OffsetScan(degrees, &lower_structure->column_offsets);
  lower_structure->indices.Resize(lower_structure->column_offsets.Back());

  if (preallocate) {
    // Re-initialize each thread's pattern flags.
    #pragma omp taskgroup
    for (int t = 0; t < max_threads; ++t) {
      #pragma omp task default(none) firstprivate(t, num_rows) \
          shared(private_pattern_flags)
      private_pattern_flags[t].Resize(num_rows, -1);
    }

    // Fill in the (unsorted) structure indices.
    #pragma omp taskgroup
    for (const Int root : ordering.assembly_forest.roots) {
      #pragma omp task default(none) firstprivate(root)     \
          shared(matrix, ordering, forest, lower_structure, \
              private_pattern_flags)
      OpenMPFillStructureIndicesRecursion(matrix, ordering, *forest, root,
                                          lower_structure,
                                          &private_pattern_flags);
    }

    // Sort the structures.
    #pragma omp taskgroup
    for (Int j = 0; j < num_rows; j += sort_grain_size) {
      #pragma omp task default(none) firstprivate(j, sort_grain_size) \
          shared(lower_structure)
      {
        const Int column_end = std::min(num_rows, j + sort_grain_size);
        for (Int column = j; column < column_end; ++column) {
          std::sort(lower_structure->ColumnBeg(column),
                    lower_structure->ColumnEnd(column));
        }
      }
    }
  } else {
    // Fill and sort the structures.
    #pragma omp taskgroup
    for (Int j = 0; j < num_rows; j += sort_grain_size) {
      #pragma omp task default(none) firstprivate(j, sort_grain_size) \
          shared(structures, lower_structure)
      {
        const Int column_end = std::min(num_rows, j + sort_grain_size);
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

}  // namespace scalar_ldl
}  // namespace catamari

#endif  // ifdef CATAMARI_OPENMP
#endif  // ifndef CATAMARI_SPARSE_LDL_SCALAR_SCALAR_UTILS_OPENMP_IMPL_H_
