/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_SUPERNODE_UTILS_OPENMP_IMPL_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_SUPERNODE_UTILS_OPENMP_IMPL_H_
#ifdef CATAMARI_OPENMP

#include "catamari/sparse_ldl/supernodal/supernode_utils.hpp"

#include "quotient/index_utils.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
void OpenMPFillStructureIndicesRecursion(
    const CoordinateMatrix<Field>& matrix, const SymmetricOrdering& ordering,
    const Buffer<Int>& supernode_member_to_index, Int root,
    LowerFactor<Field>* lower_factor,
    Buffer<Buffer<Int>>* private_pattern_flags) {
  const Int child_beg = ordering.assembly_forest.child_offsets[root];
  const Int child_end = ordering.assembly_forest.child_offsets[root + 1];
  #pragma omp taskgroup
  for (Int child_index = child_beg; child_index < child_end; ++child_index) {
    const Int child = ordering.assembly_forest.children[child_index];
    #pragma omp task default(none) firstprivate(child)                    \
        shared(matrix, ordering, supernode_member_to_index, lower_factor, \
            private_pattern_flags)
    OpenMPFillStructureIndicesRecursion(matrix, ordering,
                                        supernode_member_to_index, child,
                                        lower_factor, private_pattern_flags);
  }

  const int thread = omp_get_thread_num();
  Buffer<Int>& pattern_flags = (*private_pattern_flags)[thread];

  // Form this node's structure by unioning that of its direct children
  // (removing portions that intersect this supernode).
  Int* struct_ptr = lower_factor->StructureBeg(root);
  for (Int child_index = child_beg; child_index < child_end; ++child_index) {
    const Int child = ordering.assembly_forest.children[child_index];
    const Int* child_struct_beg = lower_factor->StructureBeg(child);
    const Int* child_struct_end = lower_factor->StructureEnd(child);
    for (const Int* child_struct_ptr = child_struct_beg;
         child_struct_ptr != child_struct_end; ++child_struct_ptr) {
      const Int row = *child_struct_ptr;
      const Int row_supernode = supernode_member_to_index[row];
      if (row_supernode == root) {
        continue;
      }

      if (pattern_flags[row] != root) {
        CATAMARI_ASSERT(row_supernode > root, "row supernode was < root.");
        pattern_flags[row] = root;
        *struct_ptr = row;
        ++struct_ptr;
      }
    }
  }

  // Incorporate this supernode's structure.
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
    for (Int index = column_beg; index < column_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      const Int row =
          have_permutation ? ordering.permutation[entry.column] : entry.column;
      const Int row_supernode = supernode_member_to_index[row];
      if (row_supernode <= root) {
        continue;
      }

      if (pattern_flags[row] != root) {
        pattern_flags[row] = root;
        *struct_ptr = row;
        ++struct_ptr;
      }
    }
  }
  CATAMARI_ASSERT(struct_ptr <= lower_factor->StructureEnd(root),
                  "Stored too many indices.");
  CATAMARI_ASSERT(struct_ptr >= lower_factor->StructureEnd(root),
                  "Stored too few indices.");
}

template <class Field>
void OpenMPFillStructureIndices(Int sort_grain_size,
                                const CoordinateMatrix<Field>& matrix,
                                const SymmetricOrdering& ordering,
                                const Buffer<Int>& supernode_member_to_index,
                                LowerFactor<Field>* lower_factor) {
  const Int num_rows = matrix.NumRows();
  const int max_threads = omp_get_max_threads();

  // A data structure for marking whether or not a node is in the pattern of
  // the active row of the lower-triangular factor. Each thread potentially
  // needs its own since different subtrees can have intersecting structure.
  Buffer<Buffer<Int>> private_pattern_flags(max_threads);

  #pragma omp taskgroup
  for (int t = 0; t < max_threads; ++t) {
    #pragma omp task default(none) firstprivate(t, num_rows) \
        shared(private_pattern_flags)
    private_pattern_flags[t].Resize(num_rows, -1);
  }

  #pragma omp taskgroup
  for (const Int root : ordering.assembly_forest.roots) {
    #pragma omp task default(none) firstprivate(root)                     \
        shared(matrix, ordering, supernode_member_to_index, lower_factor, \
            private_pattern_flags)
    OpenMPFillStructureIndicesRecursion(matrix, ordering,
                                        supernode_member_to_index, root,
                                        lower_factor, &private_pattern_flags);
  }

  // Sort the structures.
  const Int num_supernodes = ordering.supernode_sizes.Size();
  #pragma omp taskgroup
  for (Int s = 0; s < num_supernodes; s += sort_grain_size) {
    #pragma omp task default(none) \
        firstprivate(sort_grain_size, s, num_supernodes) shared(lower_factor)
    {
      const Int supernode_end = std::min(num_supernodes, s + sort_grain_size);
      for (Int supernode = s; supernode < supernode_end; ++supernode) {
        std::sort(lower_factor->StructureBeg(supernode),
                  lower_factor->StructureEnd(supernode));
      }
    }
  }
}

template <class Field>
void OpenMPFillNonzerosRecursion(const CoordinateMatrix<Field>& matrix,
                                 const SymmetricOrdering& ordering,
                                 const Buffer<Int>& supernode_member_to_index,
                                 Int root, LowerFactor<Field>* lower_factor,
                                 DiagonalFactor<Field>* diagonal_factor) {
  const Int child_beg = ordering.assembly_forest.child_offsets[root];
  const Int child_end = ordering.assembly_forest.child_offsets[root + 1];
  #pragma omp taskgroup
  for (Int child_index = child_beg; child_index < child_end; ++child_index) {
    const Int child = ordering.assembly_forest.children[child_index];
    #pragma omp task default(none) firstprivate(child)                    \
        shared(matrix, ordering, supernode_member_to_index, lower_factor, \
            diagonal_factor)
    OpenMPFillNonzerosRecursion(matrix, ordering, supernode_member_to_index,
                                child, lower_factor, diagonal_factor);
  }

  const bool have_permutation = !ordering.permutation.Empty();
  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  const Int supernode_size = ordering.supernode_sizes[root];
  const Int supernode_offset = ordering.supernode_offsets[root];
  for (Int index = 0; index < supernode_size; ++index) {
    const Int row = supernode_offset + index;
    const Int orig_row =
        have_permutation ? ordering.inverse_permutation[row] : row;
    const Int row_beg = matrix.RowEntryOffset(orig_row);
    const Int row_end = matrix.RowEntryOffset(orig_row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      const Int column =
          have_permutation ? ordering.permutation[entry.column] : entry.column;
      const Int column_supernode = supernode_member_to_index[column];

      if (column_supernode == root) {
        // Insert the value into the diagonal block.
        const Int rel_row = row - supernode_offset;
        const Int rel_column = column - supernode_offset;
        diagonal_factor->blocks[root](rel_row, rel_column) = entry.value;
        continue;
      }

      if (column_supernode > root) {
        if (have_permutation) {
          continue;
        } else {
          break;
        }
      }

      // Insert the value into the subdiagonal block.
      const Int* column_index_beg =
          lower_factor->StructureBeg(column_supernode);
      const Int* column_index_end =
          lower_factor->StructureEnd(column_supernode);
      const Int* iter =
          std::lower_bound(column_index_beg, column_index_end, row);
      CATAMARI_ASSERT(iter != column_index_end, "Exceeded column indices.");
      CATAMARI_ASSERT(*iter == row, "Entry (" + std::to_string(row) + ", " +
                                        std::to_string(column) +
                                        ") wasn't in the structure.");
      const Int rel_row = std::distance(column_index_beg, iter);
      const Int rel_column =
          column - ordering.supernode_offsets[column_supernode];
      lower_factor->blocks[column_supernode](rel_row, rel_column) = entry.value;
    }
  }
}

template <class Field>
void OpenMPFillNonzeros(const CoordinateMatrix<Field>& matrix,
                        const SymmetricOrdering& ordering,
                        const Buffer<Int>& supernode_member_to_index,
                        LowerFactor<Field>* lower_factor,
                        DiagonalFactor<Field>* diagonal_factor) {
  #pragma omp taskgroup
  for (const Int root : ordering.assembly_forest.roots) {
    #pragma omp task default(none) firstprivate(root)                     \
        shared(matrix, ordering, supernode_member_to_index, lower_factor, \
            diagonal_factor)
    OpenMPFillNonzerosRecursion(matrix, ordering, supernode_member_to_index,
                                root, lower_factor, diagonal_factor);
  }
}

template <class Field>
void OpenMPFillZerosRecursion(const SymmetricOrdering& ordering, Int root,
                              LowerFactor<Field>* lower_factor,
                              DiagonalFactor<Field>* diagonal_factor) {
  const Int child_beg = ordering.assembly_forest.child_offsets[root];
  const Int child_end = ordering.assembly_forest.child_offsets[root + 1];
  #pragma omp taskgroup
  for (Int child_index = child_beg; child_index < child_end; ++child_index) {
    const Int child = ordering.assembly_forest.children[child_index];
    #pragma omp task default(none) firstprivate(child) \
        shared(ordering, lower_factor, diagonal_factor)
    OpenMPFillZerosRecursion(ordering, child, lower_factor, diagonal_factor);
  }

  BlasMatrixView<Field>& diagonal_block = diagonal_factor->blocks[root];
  std::fill(
      diagonal_block.data,
      diagonal_block.data + diagonal_block.leading_dim * diagonal_block.width,
      Field{0});

  BlasMatrixView<Field>& lower_block = lower_factor->blocks[root];
  std::fill(lower_block.data,
            lower_block.data + lower_block.leading_dim * lower_block.width,
            Field{0});
}

template <class Field>
void OpenMPFillZeros(const SymmetricOrdering& ordering,
                     LowerFactor<Field>* lower_factor,
                     DiagonalFactor<Field>* diagonal_factor) {
  #pragma omp taskgroup
  for (const Int root : ordering.assembly_forest.roots) {
    #pragma omp task default(none) firstprivate(root) \
        shared(ordering, lower_factor, diagonal_factor)
    OpenMPFillZerosRecursion(ordering, root, lower_factor, diagonal_factor);
  }
}

template <class Field>
void OpenMPFormScaledTranspose(Int tile_size,
                               SymmetricFactorizationType factorization_type,
                               const ConstBlasMatrixView<Field>& diagonal_block,
                               const ConstBlasMatrixView<Field>& matrix,
                               BlasMatrixView<Field>* scaled_transpose) {
  const ConstBlasMatrixView<Field> diagonal_block_copy = diagonal_block;
  const ConstBlasMatrixView<Field> matrix_copy = matrix;

  if (factorization_type == kCholeskyFactorization) {
    for (Int i_beg = 0; i_beg < matrix.height; i_beg += tile_size) {
      #pragma omp task default(none) \
          firstprivate(i_beg, matrix_copy, tile_size, scaled_transpose)
      {
        const Int i_end = std::min(matrix_copy.height, i_beg + tile_size);
        for (Int i = i_beg; i < i_end; ++i) {
          for (Int j = 0; j < matrix_copy.width; ++j) {
            scaled_transpose->Entry(j, i) = Conjugate(matrix_copy(i, j));
          }
        }
      }
    }
  } else if (factorization_type == kLDLAdjointFactorization) {
    for (Int i_beg = 0; i_beg < matrix.height; i_beg += tile_size) {
      #pragma omp task default(none)                                       \
          firstprivate(i_beg, matrix_copy, diagonal_block_copy, tile_size, \
              scaled_transpose)
      {
        const Int i_end = std::min(matrix_copy.height, i_beg + tile_size);
        for (Int i = i_beg; i < i_end; ++i) {
          for (Int j = 0; j < matrix_copy.width; ++j) {
            scaled_transpose->Entry(j, i) =
                diagonal_block_copy(j, j) * Conjugate(matrix_copy(i, j));
          }
        }
      }
    }
  } else {
    for (Int i_beg = 0; i_beg < matrix.height; i_beg += tile_size) {
      #pragma omp task default(none)                                       \
          firstprivate(i_beg, matrix_copy, diagonal_block_copy, tile_size, \
              scaled_transpose)
      {
        const Int i_end = std::min(matrix_copy.height, i_beg + tile_size);
        for (Int i = i_beg; i < i_end; ++i) {
          for (Int j = 0; j < matrix_copy.width; ++j) {
            scaled_transpose->Entry(j, i) =
                diagonal_block_copy(j, j) * matrix_copy(i, j);
          }
        }
      }
    }
  }
}

template <class Field>
void OpenMPMergeChildSchurComplements(
    Int merge_grain_size, Int supernode, const SymmetricOrdering& ordering,
    LowerFactor<Field>* lower_factor, DiagonalFactor<Field>* diagonal_factor,
    RightLookingSharedState<Field>* shared_state) {
  const Int child_beg = ordering.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;
  BlasMatrixView<Field> lower_block = lower_factor->blocks[supernode];
  BlasMatrixView<Field> diagonal_block = diagonal_factor->blocks[supernode];
  BlasMatrixView<Field> schur_complement =
      shared_state->schur_complements[supernode];

  const Int supernode_size = ordering.supernode_sizes[supernode];
  const Int supernode_start = ordering.supernode_offsets[supernode];
  const Int* main_indices = lower_factor->StructureBeg(supernode);
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering.assembly_forest.children[child_beg + child_index];
    const Int* child_indices = lower_factor->StructureBeg(child);
    Buffer<Field>& child_schur_complement_buffer =
        shared_state->schur_complement_buffers[child];
    BlasMatrixView<Field> child_schur_complement =
        shared_state->schur_complements[child];
    const Int child_degree = child_schur_complement.height;

    // Fill the mapping from the child structure into the parent front.
    Int num_child_diag_indices = 0;
    Buffer<Int> child_rel_indices(child_degree);
    {
      Int i_rel = supernode_size;
      for (Int i = 0; i < child_degree; ++i) {
        const Int row = child_indices[i];
        if (row < supernode_start + supernode_size) {
          child_rel_indices[i] = row - supernode_start;
          ++num_child_diag_indices;
        } else {
          while (main_indices[i_rel - supernode_size] != row) {
            ++i_rel;
            CATAMARI_ASSERT(i_rel < supernode_size + schur_complement.height,
                            "Relative index is out-of-bounds.");
          }
          child_rel_indices[i] = i_rel;
        }
      }
    }
    const Int* child_rel_indices_ptr = child_rel_indices.Data();

    // Add the child Schur complement into this supernode's front.
    #pragma omp taskgroup
    for (Int j_beg = 0; j_beg < child_degree; j_beg += merge_grain_size) {
      #pragma omp task default(none)                                           \
          firstprivate(j_beg, child_degree, diagonal_block, lower_block,       \
              child_schur_complement, schur_complement, child_rel_indices_ptr, \
              num_child_diag_indices, supernode_size, merge_grain_size)
      {
        const Int j_end = std::min(child_degree, j_beg + merge_grain_size);
        for (Int j = j_beg; j < j_end; ++j) {
          const Int j_rel = child_rel_indices_ptr[j];
          if (j < num_child_diag_indices) {
            // Contribute into the upper-left diagonal block of the front.
            for (Int i = j; i < num_child_diag_indices; ++i) {
              const Int i_rel = child_rel_indices_ptr[i];
              diagonal_block(i_rel, j_rel) += child_schur_complement(i, j);
            }

            // Contribute into the lower-left block of the front.
            for (Int i = num_child_diag_indices; i < child_degree; ++i) {
              const Int i_rel = child_rel_indices_ptr[i];
              lower_block(i_rel - supernode_size, j_rel) +=
                  child_schur_complement(i, j);
            }
          } else {
            // Contribute into the bottom-right block of the front.
            for (Int i = j; i < child_degree; ++i) {
              const Int i_rel = child_rel_indices_ptr[i];
              schur_complement(i_rel - supernode_size,
                               j_rel - supernode_size) +=
                  child_schur_complement(i, j);
            }
          }
        }
      }
    }

    shared_state->schur_complements[child].height = 0;
    shared_state->schur_complements[child].width = 0;
    shared_state->schur_complements[child].data = nullptr;
    child_schur_complement_buffer.Clear();
  }
}

template <class Field>
Int OpenMPFactorDiagonalBlock(Int tile_size, Int block_size,
                              SymmetricFactorizationType factorization_type,
                              BlasMatrixView<Field>* diagonal_block,
                              Buffer<Field>* buffer) {
  Int num_pivots;
  if (factorization_type == kCholeskyFactorization) {
    num_pivots =
        OpenMPLowerCholeskyFactorization(tile_size, block_size, diagonal_block);
  } else if (factorization_type == kLDLAdjointFactorization) {
    num_pivots = OpenMPLDLAdjointFactorization(tile_size, block_size,
                                               diagonal_block, buffer);
  } else {
    num_pivots = OpenMPLDLTransposeFactorization(tile_size, block_size,
                                                 diagonal_block, buffer);
  }
  return num_pivots;
}

template <class Field>
void OpenMPSolveAgainstDiagonalBlock(
    Int tile_size, SymmetricFactorizationType factorization_type,
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* lower_matrix) {
  if (!lower_matrix->height) {
    return;
  }

  const ConstBlasMatrixView<Field> triangular_matrix_copy = triangular_matrix;
  BlasMatrixView<Field> lower_matrix_copy = *lower_matrix;

  const Int height = lower_matrix->height;
  const Int width = lower_matrix->width;
  if (factorization_type == kCholeskyFactorization) {
    // Solve against the lower-triangular matrix L(K, K)' from the right.
    for (Int i = 0; i < height; i += tile_size) {
      #pragma omp task default(none)                                        \
          firstprivate(i, height, width, tile_size, triangular_matrix_copy, \
              lower_matrix_copy)
      {
        const Int tsize = std::min(height - i, tile_size);
        BlasMatrixView<Field> lower_block =
            lower_matrix_copy.Submatrix(i, 0, tsize, width);
        RightLowerAdjointTriangularSolves(triangular_matrix_copy, &lower_block);
      }
    }
  } else if (factorization_type == kLDLAdjointFactorization) {
    // Solve against D(K, K) L(K, K)' from the right.
    for (Int i = 0; i < height; i += tile_size) {
      #pragma omp task default(none)                                        \
          firstprivate(i, height, width, tile_size, triangular_matrix_copy, \
          lower_matrix_copy)
      {
        const Int tsize = std::min(height - i, tile_size);
        BlasMatrixView<Field> lower_block =
            lower_matrix_copy.Submatrix(i, 0, tsize, width);
        RightDiagonalTimesLowerAdjointUnitTriangularSolves(
            triangular_matrix_copy, &lower_block);
      }
    }
  } else {
    // Solve against D(K, K) L(K, K)^T from the right.
    for (Int i = 0; i < height; i += tile_size) {
      #pragma omp task default(none)                                        \
          firstprivate(i, height, width, tile_size, triangular_matrix_copy, \
              lower_matrix_copy)
      {
        const Int tsize = std::min(height - i, tile_size);
        BlasMatrixView<Field> lower_block =
            lower_matrix_copy.Submatrix(i, 0, tsize, width);
        RightDiagonalTimesLowerTransposeUnitTriangularSolves(
            triangular_matrix_copy, &lower_block);
      }
    }
  }
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifdef CATAMARI_OPENMP
#endif  // ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_SUPERNODE_UTILS_OPENMP_IMPL_H_
