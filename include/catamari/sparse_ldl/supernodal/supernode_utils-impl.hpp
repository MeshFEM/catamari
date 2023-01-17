/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_SUPERNODE_UTILS_IMPL_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_SUPERNODE_UTILS_IMPL_H_

#include "catamari/dense_factorizations/cholesky-impl.hpp"
#include "catamari/sparse_ldl/supernodal/supernode_utils.hpp"

#include "quotient/index_utils.hpp"

#define VECTORIZE_MERGE_SCHUR_COMPLEMENTS 0 // Appears to be a pessimization :(

namespace catamari {
namespace supernodal_ldl {

inline void MemberToIndex(Int num_rows, const Buffer<Int>& supernode_starts,
                          Buffer<Int>* member_to_index) {
  member_to_index->Resize(num_rows);

  Int supernode = 0;
  for (Int column = 0; column < num_rows; ++column) {
    if (column == supernode_starts[supernode + 1]) {
      ++supernode;
    }
    CATAMARI_ASSERT(column >= supernode_starts[supernode] &&
                        column < supernode_starts[supernode + 1],
                    "Column was not in the marked supernode.");
    (*member_to_index)[column] = supernode;
  }
  CATAMARI_ASSERT(supernode == Int(supernode_starts.Size()) - 2,
                  "Ended on supernode " + std::to_string(supernode) +
                      " instead of " +
                      std::to_string(supernode_starts.Size() - 2));
}

inline void ConvertFromScalarToSupernodalEliminationForest(
    Int num_supernodes, const Buffer<Int>& parents,
    const Buffer<Int>& member_to_index, Buffer<Int>* supernode_parents) {
  const Int num_rows = parents.Size();
  supernode_parents->Resize(num_supernodes);
  for (Int row = 0; row < num_rows; ++row) {
    const Int supernode = member_to_index[row];
    const Int parent = parents[row];
    if (parent == -1) {
      (*supernode_parents)[supernode] = -1;
    } else {
      const Int parent_supernode = member_to_index[parent];
      if (parent_supernode != supernode) {
        (*supernode_parents)[supernode] = parent_supernode;
      }
    }
  }
}

template <class Field>
bool ValidFundamentalSupernodes(const CoordinateMatrix<Field>& matrix,
                                const SymmetricOrdering& ordering,
                                const Buffer<Int>& supernode_sizes) {
  const Int num_rows = matrix.NumRows();
  for (UInt index = 0; index < supernode_sizes.Size(); ++index) {
    CATAMARI_ASSERT(supernode_sizes[index] > 0,
                    "Supernode " + std::to_string(index) + " had length " +
                        std::to_string(supernode_sizes[index]));
  }
  {
    Int size_sum = 0;
    for (const Int& supernode_size : supernode_sizes) {
      size_sum += supernode_size;
    }
    CATAMARI_ASSERT(size_sum == num_rows,
                    "Supernode sizes summed to " + std::to_string(size_sum) +
                        " instead of " + std::to_string(num_rows));
  }

  Buffer<Int> parents, degrees;
  scalar_ldl::EliminationForestAndDegrees(matrix, ordering, &parents, &degrees);

  Buffer<Int> supernode_starts, member_to_index;
  OffsetScan(supernode_sizes, &supernode_starts);
  MemberToIndex(num_rows, supernode_starts, &member_to_index);

  Buffer<Int> row_structure(num_rows);
  Buffer<Int> pattern_flags(num_rows);

  bool valid = true;
  for (Int row = 0; row < num_rows; ++row) {
    const Int row_supernode = member_to_index[row];

    pattern_flags[row] = row;
    const Int num_packed = scalar_ldl::ComputeRowPattern(
        matrix, ordering, parents, row, pattern_flags.Data(),
        row_structure.Data());
    std::sort(row_structure.Data(), row_structure.Data() + num_packed);

    // TODO(Jack Poulson): Extend the tests to ensure that the diagonal blocks
    // of the supernodes are dense.

    // Check that the pattern of this row intersects entire supernodes that
    // are not the current row's supernode.
    Int index = 0;
    while (index < num_packed) {
      const Int column = row_structure[index];
      const Int supernode = member_to_index[column];
      if (supernode == row_supernode) {
        break;
      }

      const Int supernode_start = supernode_starts[supernode];
      const Int supernode_size = supernode_sizes[supernode];
      if (num_packed < index + supernode_size) {
        std::cerr << "Did not pack enough indices to hold supernode "
                  << supernode << " of size " << supernode_size << " in row "
                  << row << std::endl;
        return false;
      }
      for (Int j = 0; j < supernode_size; ++j) {
        if (row_structure[index++] != supernode_start + j) {
          std::cerr << "Missed column " << supernode_start + j << " in row "
                    << row << " and supernode " << supernode_start << ":"
                    << supernode_start + supernode_size << std::endl;
          valid = false;
        }
      }
    }
  }
  return valid;
}

inline void FormFundamentalSupernodes(const Buffer<Int>& scalar_parents,
                                      const Buffer<Int>& scalar_degrees,
                                      Buffer<Int>* supernode_sizes) {
  const Int num_rows = scalar_degrees.Size();
  supernode_sizes->Clear();
  if (!num_rows) {
    return;
  }

  // Count the number of parents of each supernode. We do not assume that the
  // list of children has been constructed already.
  Buffer<Int> num_children(num_rows, 0);
  for (Int column = 0; column < num_rows; ++column) {
    const Int parent = scalar_parents[column];
    if (parent != -1) {
      ++num_children[parent];
    }
  }

  // Rather than explicitly traversing the structure, we can use its metadata
  // and the properties of the assembly tree to determine the fundamental
  // supernode structure.
  //
  // Proposition. Column j is contained in the same fundamental supernode as
  // column j - 1 if and only if column j - 1 is the sole child of column j
  // and the degree of column j is one less than that of column j - 1.
  //
  // Proof. Suppose column j is in the same fundamental supernode. Then, by
  // definition, the properties hold.
  //   Now suppose the properties hold. Column j - 1 being a child implies that
  // its structure, minus node j, is a subset of the structure of column j.
  // Thus, degree[j] >= degree[j - 1] - 1, and equality implies that
  //
  //   struct[j - 1] \ {j} = struct[j].
  //
  // That column j - 1 is the only child of column j is part of the definition
  // of a fundamental supernode. See:
  //
  //   Alex Pothen and Sivan Toledo,
  //   "Elimination Structures in Scientific Computing", CRC Press, 2001.
  //
  // Consider the example lower sparsity pattern:
  //
  //   | x     |
  //   |   x   |,
  //   | x x x |
  //
  // where the maximal contiguous cliques {0} and {1, 2}, but the fundamental
  // supernodes as {0}, {1}, and {2}.
  Int num_supernodes = 0;
  supernode_sizes->Resize(num_rows);
  (*supernode_sizes)[num_supernodes++] = 1;
  for (Int column = 1; column < num_rows; ++column) {
    const bool is_parent = scalar_parents[column - 1] == column;
    const bool one_child = num_children[column] == 1;
    const bool matching_degrees =
        scalar_degrees[column] == scalar_degrees[column - 1] - 1;
    if (is_parent && one_child && matching_degrees) {
      // Include this column in the active supernode.
      ++(*supernode_sizes)[num_supernodes - 1];
    } else {
      // Begin a new supernode at this column.
      (*supernode_sizes)[num_supernodes++] = 1;
    }
  }
  supernode_sizes->Resize(num_supernodes);
}

inline MergableStatus MergableSupernode(
    Int child_size, Int child_degree, Int parent_size, Int parent_degree,
    Int num_child_zeros, Int num_parent_zeros,
    const SupernodalRelaxationControl& control) {
  MergableStatus status;

  // Since the child structure is contained in the union of the parent structure
  // and parent supernode, we know how many rows of explicit zeros will be
  // introduced underneath the diagonal block of the child supernode via the
  // merger.
  const Int num_new_zeros =
      (parent_size + parent_degree - child_degree) * child_size;
  if (num_new_zeros == 0) {
    status.mergable = true;
    return status;
  }

  const Int num_old_zeros = num_child_zeros + num_parent_zeros;
  const Int num_zeros = num_new_zeros + num_old_zeros;
  status.num_merged_zeros = num_zeros;

  const Int combined_size = child_size + parent_size;
  const Int num_expanded_entries =
      (combined_size * (combined_size + 1)) / 2 + parent_degree * combined_size;
  CATAMARI_ASSERT(
      num_expanded_entries > num_zeros,
      "Number of expanded entries was <= the number of computed zeros.");

  for (const std::pair<Int, float>& cutoff : control.cutoff_pairs) {
    const Int num_zeros_cutoff = num_expanded_entries * cutoff.second;
    if (cutoff.first >= combined_size && num_zeros_cutoff >= num_zeros) {
      status.mergable = true;
      return status;
    }
  }

  status.mergable = false;
  return status;
}

inline void MergeChildren(Int parent, const Buffer<Int>& /* orig_supernode_starts */,
                          const Buffer<Int>& /* orig_supernode_sizes */,
                          const Buffer<Int>& orig_supernode_degrees,
                          const Buffer<Int>& child_list_heads,
                          const Buffer<Int>& child_lists,
                          const SupernodalRelaxationControl& control,
                          Buffer<Int>* supernode_sizes, Buffer<Int>* num_zeros,
                          Buffer<Int>* last_merged_child,
                          Buffer<Int>* merge_parents) {
  // The following loop can execute at most 'num_children' times.
  while (true) {
    // Compute the index of a maximal mergable child (if it exists).
    Int merging_child = -1;
    Int num_new_zeros = 0;
    Int num_merged_zeros = 0;
    Int largest_mergable_size = 0;
    for (Int child = child_list_heads[parent]; child != -1;
         child = child_lists[child]) {
      if ((*merge_parents)[child] != -1) {
        continue;
      }

      const Int child_size = (*supernode_sizes)[child];
      if (child_size < largest_mergable_size) {
        continue;
      }
      const Int child_degree = orig_supernode_degrees[child];

      const Int parent_size = (*supernode_sizes)[parent];
      const Int parent_degree = orig_supernode_degrees[parent];

      const Int num_child_zeros = (*num_zeros)[child];
      const Int num_parent_zeros = (*num_zeros)[parent];

      const MergableStatus status = MergableSupernode(
          child_size, child_degree, parent_size, parent_degree, num_child_zeros,
          num_parent_zeros, control);
      if (!status.mergable) {
        continue;
      }
      const Int num_proposed_new_zeros =
          status.num_merged_zeros - (num_child_zeros + num_parent_zeros);
      if (child_size > largest_mergable_size ||
          num_proposed_new_zeros < num_new_zeros) {
        merging_child = child;
        num_new_zeros = num_proposed_new_zeros;
        num_merged_zeros = status.num_merged_zeros;
        largest_mergable_size = child_size;
      }
    }

    // Skip this supernode if no children can be merged into it.
    if (merging_child == -1) {
      break;
    }

    // Absorb the child size into the parent.
    (*supernode_sizes)[parent] += (*supernode_sizes)[merging_child];
    (*supernode_sizes)[merging_child] = 0;

    // Update the number of explicit zeros in the merged supernode.
    (*num_zeros)[parent] = num_merged_zeros;

    // Mark the child as merged.
    //
    // TODO(Jack Poulson): Consider following a similar strategy as quotient
    // and using SYMMETRIC_INDEX to pack a parent into the negative indices.

    // Build the new uplink from the child in the assembly tree. We connect up
    // to the last merged child of the parent since the parent start location
    // moves to the merged child's supernodal index.
    if ((*last_merged_child)[parent] == -1) {
      (*merge_parents)[merging_child] = parent;
    } else {
      (*merge_parents)[merging_child] = (*last_merged_child)[parent];
    }

    // Build the new downlink from the parent in the assembly tree.
    if ((*last_merged_child)[merging_child] == -1) {
      (*last_merged_child)[parent] = merging_child;
    } else {
      (*last_merged_child)[parent] = (*last_merged_child)[merging_child];
    }
  }
}

inline void RelaxedParentsAndDegrees(const Buffer<Int>& parents,
                                     const Buffer<Int>& degrees,
                                     const Buffer<Int>& merge_parents,
                                     Buffer<Int>* original_to_relaxed,
                                     Buffer<Int>* relaxed_parents,
                                     Buffer<Int>* relaxed_degrees) {
  // Count the number of remaining supernodes and construct a map from the
  // original to relaxed indices.
  const Int num_supernodes = merge_parents.Size();
  original_to_relaxed->Resize(num_supernodes);
  Int num_relaxed_supernodes = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    if (merge_parents[supernode] != -1) {
      continue;
    }
    (*original_to_relaxed)[supernode] = num_relaxed_supernodes++;
  }

  // Fill in the parents for the relaxed supernodal elimination forest.
  // Simultaneously, fill in the degrees of the relaxed supernodes (which are
  // the same as the degrees of the parents of each merge tree).
  relaxed_parents->Resize(num_relaxed_supernodes);
  relaxed_degrees->Resize(num_relaxed_supernodes);
  Int relaxed_offset = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    if (merge_parents[supernode] != -1) {
      continue;
    }
    (*relaxed_degrees)[relaxed_offset] = degrees[supernode];

    Int parent = parents[supernode];
    if (parent == -1) {
      // This is a root node, so mark it as such in the relaxation.
      (*relaxed_parents)[relaxed_offset++] = -1;
      continue;
    }

    // Compute the root of the merge sequence of our parent.
    while (merge_parents[parent] != -1) {
      parent = merge_parents[parent];
    }

    // Convert the root of the merge sequence of our parent from the original
    // to the relaxed indexing.
    (*relaxed_parents)[relaxed_offset++] = (*original_to_relaxed)[parent];
  }
}

// Fill the inverse permutation, the supernode sizes, and the supernode
// offsets.
inline void RelaxationSizesOffsetsAndInversePermutation(
    Int num_relaxed_supernodes, const Buffer<Int>& orig_sizes,
    const Buffer<Int>& orig_offsets, const Buffer<Int>& last_merged_children,
    const Buffer<Int>& merge_parents, const Buffer<Int>& original_to_relaxed,
    Buffer<Int>* relaxed_sizes, Buffer<Int>* relaxed_offsets,
    Buffer<Int>* relaxed_supernode_member_to_index,
    Buffer<Int>* relaxation_inverse_permutation) {
  const Int num_supernodes = orig_sizes.Size();
  const Int num_rows = orig_offsets[num_supernodes];

  relaxed_sizes->Resize(num_relaxed_supernodes);
  relaxed_offsets->Resize(num_relaxed_supernodes + 1);
  relaxed_supernode_member_to_index->Resize(num_rows);
  relaxation_inverse_permutation->Resize(num_rows);

  Int pack_offset = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    if (merge_parents[supernode] != -1) {
      continue;
    }
    const Int relaxed_supernode = original_to_relaxed[supernode];

    // Get the leaf of the merge sequence to pack.
    Int leaf_of_merge = supernode;
    if (last_merged_children[supernode] != -1) {
      leaf_of_merge = last_merged_children[supernode];
    }

    // Pack the merge sequence and count its total size.
    (*relaxed_offsets)[relaxed_supernode] = pack_offset;
    Int supernode_size = 0;
    Int* supernode_index =
        relaxed_supernode_member_to_index->Data() + pack_offset;
    Int* supernode_inverse =
        relaxation_inverse_permutation->Data() + pack_offset;
    Int supernode_to_pack = leaf_of_merge;
    while (supernode_to_pack != -1) {
      const Int start = orig_offsets[supernode_to_pack];
      const Int size = orig_sizes[supernode_to_pack];
      for (Int j = 0; j < size; ++j) {
        supernode_index[supernode_size] = relaxed_supernode;
        supernode_inverse[supernode_size++] = start + j;
      }
      supernode_to_pack = merge_parents[supernode_to_pack];
    }
    (*relaxed_sizes)[relaxed_supernode] = supernode_size;
    pack_offset += supernode_size;
  }
  CATAMARI_ASSERT(num_rows == pack_offset, "Did not pack num_rows indices.");
  (*relaxed_offsets)[num_relaxed_supernodes] = num_rows;
}

inline void RelaxSupernodes(const SymmetricOrdering& orig_ordering,
                            const Buffer<Int>& orig_supernode_degrees,
                            const SupernodalRelaxationControl& control,
                            SymmetricOrdering* relaxed_ordering,
                            Buffer<Int>* relaxed_supernode_degrees,
                            Buffer<Int>* relaxed_supernode_member_to_index) {
  const Int num_rows = orig_ordering.supernode_offsets.Back();
  const Int num_supernodes = orig_ordering.supernode_sizes.Size();

  // Construct the children in the elimination forest.
  Buffer<Int> child_list_heads(num_supernodes, -1);
  Buffer<Int> child_lists(num_supernodes);
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int parent = orig_ordering.assembly_forest.parents[supernode];
    if (parent != -1) {
      // Insert 'supernode' in the child list of its parent.
      child_lists[supernode] = child_list_heads[parent];
      child_list_heads[parent] = supernode;
    }
  }

  Buffer<Int> last_merged_children(num_supernodes, -1);
  Buffer<Int> merge_parents(num_supernodes, -1);
  {
    // Initialize the sizes of the merged supernodes, using the original
    // indexing and absorbing child sizes into the parents.
    Buffer<Int> supernode_sizes = orig_ordering.supernode_sizes;

    // Initialize the number of explicit zeros stored in each original
    // supernode.
    Buffer<Int> num_zeros(num_supernodes, 0);

    for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
      MergeChildren(supernode, orig_ordering.supernode_offsets,
                    orig_ordering.supernode_sizes, orig_supernode_degrees,
                    child_list_heads, child_lists, control, &supernode_sizes,
                    &num_zeros, &last_merged_children, &merge_parents);
    }
  }

  // Fill in the parents for the relaxed supernodal elimination forest.
  // Simultaneously, fill in the degrees of the relaxed supernodes (which are
  // the same as the degrees of the parents of each merge tree).
  Buffer<Int> original_to_relaxed;
  RelaxedParentsAndDegrees(
      orig_ordering.assembly_forest.parents, orig_supernode_degrees,
      merge_parents, &original_to_relaxed,
      &relaxed_ordering->assembly_forest.parents, relaxed_supernode_degrees);

  // Fill the inverse permutation, the supernode sizes, and the supernode
  // offsets.
  Buffer<Int> relaxation_inverse_permutation;
  RelaxationSizesOffsetsAndInversePermutation(
      relaxed_supernode_degrees->Size(), orig_ordering.supernode_sizes,
      orig_ordering.supernode_offsets, last_merged_children, merge_parents,
      original_to_relaxed, &relaxed_ordering->supernode_sizes,
      &relaxed_ordering->supernode_offsets, relaxed_supernode_member_to_index,
      &relaxation_inverse_permutation);

  // Compose the relaxation permutation with the original permutation.
  if (relaxed_ordering->permutation.Empty()) {
    relaxed_ordering->inverse_permutation = relaxation_inverse_permutation;
    InvertPermutation(relaxed_ordering->inverse_permutation,
                      &relaxed_ordering->permutation);
  } else {
    // Compute the inverse of relaxation_permutation with
    // relaxed_ordering->permutation, which is the composition of their inverses
    // in the opposite order.
    Buffer<Int> inverse_perm_copy = relaxed_ordering->inverse_permutation;
    for (Int row = 0; row < num_rows; ++row) {
      relaxed_ordering->inverse_permutation[row] =
          inverse_perm_copy[relaxation_inverse_permutation[row]];
    }

    // Invert the composed permutation.
    InvertPermutation(relaxed_ordering->inverse_permutation,
                      &relaxed_ordering->permutation);
  }
  relaxed_ordering->assembly_forest.FillFromParents();
}

template <class Field>
void FillStructureIndices(const CoordinateMatrix<Field>& matrix,
                          const SymmetricOrdering& ordering,
                          const Buffer<Int>& supernode_member_to_index,
                          LowerFactor<Field>* lower_factor) {
  const Int num_supernodes = ordering.supernode_sizes.Size();
  const bool have_permutation = !ordering.permutation.Empty();
  const Buffer<Int>& parents = ordering.assembly_forest.parents;

  // A data structure for marking whether or not a supernode is in the pattern
  // of the active row of the lower-triangular factor.
  Buffer<Int> pattern_flags(num_supernodes);

  // A set of pointers for keeping track of where to insert supernode pattern
  // indices.
  Buffer<Int*> supernode_ptrs(num_supernodes);
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    supernode_ptrs[supernode] = lower_factor->StructureBeg(supernode);
  }

  // Fill in the structure indices.
  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int supernode_offset = ordering.supernode_offsets[supernode];
    const Int supernode_size = ordering.supernode_sizes[supernode];
    const Int supernode_end = supernode_offset + supernode_size;

    for (Int row = supernode_offset; row < supernode_end; ++row) {
      pattern_flags[supernode] = row;

      const Int orig_row =
          have_permutation ? ordering.inverse_permutation[row] : row;
      const Int row_beg = matrix.RowEntryOffset(orig_row);
      const Int row_end = matrix.RowEntryOffset(orig_row + 1);
      for (Int index = row_beg; index < row_end; ++index) {
        const MatrixEntry<Field>& entry = entries[index];
        const Int column = have_permutation ? ordering.permutation[entry.column]
                                            : entry.column;

        // Skip this entry if it is not to the left of the diagonal block.
        if (column >= supernode_offset) {
          if (have_permutation) {
            continue;
          } else {
            break;
          }
        }

        for (Int ancestor_supernode = supernode_member_to_index[column];
             pattern_flags[ancestor_supernode] < row;
             ancestor_supernode = parents[ancestor_supernode]) {
          *supernode_ptrs[ancestor_supernode] = row;
          ++supernode_ptrs[ancestor_supernode];
          pattern_flags[ancestor_supernode] = row;
        }
      }
    }
  }

#ifdef CATAMARI_DEBUG
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int* index_beg = lower_factor->StructureBeg(supernode);
    const Int* index_end = lower_factor->StructureEnd(supernode);
    CATAMARI_ASSERT(supernode_ptrs[supernode] == index_end,
                    "Supernode pointers did not match index offsets.");
    bool sorted = true;
    Int last_row = -1;
    for (const Int* row_ptr = index_beg; row_ptr != index_end; ++row_ptr) {
      const Int row = *row_ptr;
      if (row <= last_row) {
        sorted = false;
        break;
      }
      last_row = row;
    }

    if (!sorted) {
      std::cerr << "Supernode " << supernode << " did not have sorted indices."
                << std::endl;
      for (const Int* row_ptr = index_beg; row_ptr != index_end; ++row_ptr) {
        std::cout << *row_ptr << " ";
      }
      std::cout << std::endl;
    }
  }
#endif  // ifdef CATAMARI_DEBUG
}

template <class Field>
void FillSubtreeWorkEstimates(Int root, const AssemblyForest& supernode_forest,
                              const LowerFactor<Field>& lower_factor,
                              Buffer<double>* work_estimates) {
  const Int child_beg = supernode_forest.child_offsets[root];
  const Int child_end = supernode_forest.child_offsets[root + 1];
  const Int num_children = child_end - child_beg;

  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child = supernode_forest.children[child_beg + child_index];
    FillSubtreeWorkEstimates(child, supernode_forest, lower_factor,
                             work_estimates);
    (*work_estimates)[root] += (*work_estimates)[child];
  }

  const ConstBlasMatrixView<Field>& lower_block =
      lower_factor.blocks[root].ToConst();
  const Int supernode_size = lower_block.width;
  const Int degree = lower_block.height;
  (*work_estimates)[root] += std::pow(1. * supernode_size, 3.) / 3;
  (*work_estimates)[root] += std::pow(1. * degree, 2.) * supernode_size;
}

template <class Field>
void FillNonzeros(const CoordinateMatrix<Field>& matrix,
                  const SymmetricOrdering& ordering,
                  const Buffer<Int>& supernode_member_to_index,
                  LowerFactor<Field>* lower_factor,
                  DiagonalFactor<Field>* diagonal_factor) {
  const Int num_rows = matrix.NumRows();
  const bool have_permutation = !ordering.permutation.Empty();

  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int supernode = supernode_member_to_index[row];
    const Int supernode_start = ordering.supernode_offsets[supernode];

    const Int orig_row =
        have_permutation ? ordering.inverse_permutation[row] : row;
    const Int row_beg = matrix.RowEntryOffset(orig_row);
    const Int row_end = matrix.RowEntryOffset(orig_row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      const Int column =
          have_permutation ? ordering.permutation[entry.column] : entry.column;
      const Int column_supernode = supernode_member_to_index[column];

      if (column_supernode == supernode) {
        // Insert the value into the diagonal block.
        const Int rel_row = row - supernode_start;
        const Int rel_column = column - supernode_start;
        diagonal_factor->blocks[supernode](rel_row, rel_column) = entry.value;
        continue;
      }

      if (column_supernode > supernode) {
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
void FillZeros(const SymmetricOrdering& ordering,
               LowerFactor<Field>* lower_factor,
               DiagonalFactor<Field>* diagonal_factor) {
  const Int num_supernodes = diagonal_factor->blocks.Size();
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    BlasMatrixView<Field>& diagonal_block = diagonal_factor->blocks[supernode];
    std::fill(
        diagonal_block.data,
        diagonal_block.data + diagonal_block.leading_dim * diagonal_block.width,
        Field{0});

    BlasMatrixView<Field>& lower_block = lower_factor->blocks[supernode];
    std::fill(lower_block.data,
              lower_block.data + lower_block.leading_dim * lower_block.width,
              Field{0});
  }
}

template <class Field>
void FormScaledTranspose(SymmetricFactorizationType factorization_type,
                         const ConstBlasMatrixView<Field>& diagonal_block,
                         const ConstBlasMatrixView<Field>& matrix,
                         BlasMatrixView<Field>* scaled_transpose) {
  if (factorization_type == kCholeskyFactorization) {
    for (Int j = 0; j < matrix.width; ++j) {
      for (Int i = 0; i < matrix.height; ++i) {
        scaled_transpose->Entry(j, i) = Conjugate(matrix(i, j));
      }
    }
  } else if (factorization_type == kLDLAdjointFactorization) {
    for (Int j = 0; j < matrix.width; ++j) {
      const Field& delta = diagonal_block(j, j);
      for (Int i = 0; i < matrix.height; ++i) {
        scaled_transpose->Entry(j, i) = delta * Conjugate(matrix(i, j));
      }
    }
  } else {
    for (Int j = 0; j < matrix.width; ++j) {
      const Field& delta = diagonal_block(j, j);
      for (Int i = 0; i < matrix.height; ++i) {
        scaled_transpose->Entry(j, i) = delta * matrix(i, j);
      }
    }
  }
}

template <class Field>
void UpdateDiagonalBlock(
    SymmetricFactorizationType factorization_type,
    const Buffer<Int>& supernode_starts, const LowerFactor<Field>& lower_factor,
    Int main_supernode, Int descendant_supernode, Int descendant_main_rel_row,
    const ConstBlasMatrixView<Field>& descendant_main_matrix,
    const ConstBlasMatrixView<Field>& scaled_transpose,
    BlasMatrixView<Field>* main_diag_block,
    BlasMatrixView<Field>* workspace_matrix) {
  typedef ComplexBase<Field> Real;
  const Int main_supernode_size = main_diag_block->height;
  const Int descendant_main_intersect_size = descendant_main_matrix.height;

  const bool inplace_update =
      descendant_main_intersect_size == main_supernode_size;
  BlasMatrixView<Field>* accumulation_block =
      inplace_update ? main_diag_block : workspace_matrix;

  if (factorization_type == kCholeskyFactorization) {
    LowerNormalHermitianOuterProduct(Real{-1}, descendant_main_matrix, Real{1},
                                     accumulation_block);
  } else {
    MatrixMultiplyLowerNormalNormal(Field{-1}, descendant_main_matrix,
                                    scaled_transpose, Field{1},
                                    accumulation_block);
  }

  if (!inplace_update) {
    // Apply the out-of-place update and zero the buffer.
    const Int main_supernode_start = supernode_starts[main_supernode];
    const Int* descendant_main_indices =
        lower_factor.StructureBeg(descendant_supernode) +
        descendant_main_rel_row;

    for (Int j = 0; j < descendant_main_intersect_size; ++j) {
      const Int column = descendant_main_indices[j];
      const Int j_rel = column - main_supernode_start;
      for (Int i = j; i < descendant_main_intersect_size; ++i) {
        const Int row = descendant_main_indices[i];
        const Int i_rel = row - main_supernode_start;
        main_diag_block->Entry(i_rel, j_rel) += workspace_matrix->Entry(i, j);
        workspace_matrix->Entry(i, j) = 0;
      }
    }
  }
}

template <class Field>
void UpdateSubdiagonalBlock(
    SymmetricFactorizationType factorization_type, Int main_supernode,
    Int descendant_supernode, Int main_active_rel_row,
    Int descendant_main_rel_row,
    const ConstBlasMatrixView<Field>& descendant_main_matrix,
    Int descendant_active_rel_row, const Buffer<Int>& supernode_starts,
    const Buffer<Int>& supernode_member_to_index,
    const ConstBlasMatrixView<Field>& scaled_transpose,
    const ConstBlasMatrixView<Field>& descendant_active_matrix,
    const LowerFactor<Field>& lower_factor,
    BlasMatrixView<Field>* main_active_block,
    BlasMatrixView<Field>* workspace_matrix) {
  const Int main_supernode_size = lower_factor.blocks[main_supernode].width;
  const Int main_active_intersect_size = main_active_block->height;
  const Int descendant_main_intersect_size = descendant_main_matrix.height;
  const Int descendant_active_intersect_size = descendant_active_matrix.height;
  const bool inplace_update =
      main_active_intersect_size == descendant_active_intersect_size &&
      main_supernode_size == descendant_main_intersect_size;

  BlasMatrixView<Field>* accumulation_matrix =
      inplace_update ? main_active_block : workspace_matrix;
  if (factorization_type == kCholeskyFactorization) {
    MatrixMultiplyNormalAdjoint(Field{-1}, descendant_active_matrix,
                                descendant_main_matrix, Field{1},
                                accumulation_matrix);
  } else {
    MatrixMultiplyNormalNormal(Field{-1}, descendant_active_matrix,
                               scaled_transpose, Field{1}, accumulation_matrix);
  }

  if (!inplace_update) {
    const Int main_supernode_start = supernode_starts[main_supernode];

    const Int* main_indices = lower_factor.StructureBeg(main_supernode);
    const Int* main_active_indices = main_indices + main_active_rel_row;

    const Int* descendant_indices =
        lower_factor.StructureBeg(descendant_supernode);
    const Int* descendant_main_indices =
        descendant_indices + descendant_main_rel_row;
    const Int* descendant_active_indices =
        descendant_indices + descendant_active_rel_row;

    CATAMARI_ASSERT(
        workspace_matrix->height == descendant_active_intersect_size,
        "workspace_matrix.height != descendant_active_intersect_size");
    CATAMARI_ASSERT(workspace_matrix->width == descendant_main_intersect_size,
                    "workspace_matrix.width != descendant_main_intersect_size");

    Int i_rel = 0;
    for (Int i = 0; i < descendant_active_intersect_size; ++i) {
      const Int row = descendant_active_indices[i];

      // Both the main and descendant supernodal intersections can be sparse,
      // and different. Thus, we must scan through the intersection indices to
      // find the appropriate relative index here.
      //
      // Scan downwards in the main active indices until we find the equivalent
      // row from the descendant active indices.
      while (main_active_indices[i_rel] != row) {
        ++i_rel;
      }
      CATAMARI_ASSERT(i_rel < main_active_block->height,
                      "i_rel out-of-bounds for main_active_block.");

      for (Int j = 0; j < descendant_main_intersect_size; ++j) {
        const Int column = descendant_main_indices[j];
        const Int j_rel = column - main_supernode_start;
        CATAMARI_ASSERT(j_rel < main_active_block->width,
                        "j_rel out-of-bounds for main_active_block.");

        main_active_block->Entry(i_rel, j_rel) += workspace_matrix->Entry(i, j);
        workspace_matrix->Entry(i, j) = 0;
      }
    }
  }
}

template <class Field>
void MergeChildSchurComplement(Int supernode, Int child,
                               const SymmetricOrdering& ordering,
                               const LowerFactor<Field> *lower_factor,
                               const BlasMatrixView<Field> &child_schur_complement,
                               BlasMatrixView<Field> lower_block,
                               BlasMatrixView<Field> diagonal_block,
                               BlasMatrixView<Field> schur_complement,
                               bool freshShurComplement) {
    const Int child_degree = child_schur_complement.height;

#if 1
    auto &ncdi   = const_cast<Buffer<Int> &>(ordering.assembly_forest.num_child_diag_indices);
    auto &cri    = const_cast<Buffer<Buffer<Int>> &>(ordering.assembly_forest.child_rel_indices);
    auto &cri_rl = const_cast<Buffer<Buffer<Int>> &>(ordering.assembly_forest.child_rel_indices_run_len);
    Int &num_child_diag_indices = ncdi[child];
    auto &child_rel_indices     = cri[child];
    auto &child_rel_indices_rl  = cri_rl[child];
#else
    Int num_child_diag_indices = 0;
    Buffer<Int> child_rel_indices;
#endif
    const Int supernode_size = ordering.supernode_sizes[supernode];
    const Int supernode_start = ordering.supernode_offsets[supernode];
    const Int supernode_end = supernode_start + supernode_size;
    if (child_rel_indices.Size() == 0) {
        const Int* parent_indices = lower_factor->StructureBeg(supernode);
        // Fill the mapping from the child structure into the parent front.
        child_rel_indices.Resize(child_degree);
        num_child_diag_indices = 0;
        {
          const Int* child_indices = lower_factor->StructureBeg(child);
          Int i_rel = 0;
          for (Int i = 0; i < child_degree; ++i) {
            const Int row = child_indices[i];
            if (row < supernode_end) {
              child_rel_indices[i] = row - supernode_start;
              ++num_child_diag_indices;
            } else {
              while (parent_indices[i_rel] != row) {
                ++i_rel;
                CATAMARI_ASSERT(i_rel < schur_complement.height, "Relative index is out-of-bounds.");
              }
              child_rel_indices[i] = i_rel;
            }
          }
       }
#if VECTORIZE_MERGE_SCHUR_COMPLEMENTS
       // Calculate the run lengths to assist vectorization...
       child_rel_indices_rl.Resize(child_degree, 1);
       for (Int i = 0; i < child_degree; /* incremented inside */ ) {
           Int rl = 1;
           while ((i + rl) < child_degree && child_rel_indices[i + rl] == child_rel_indices[i] + rl)
               ++rl;
           while (rl > 0) { child_rel_indices_rl[i++] = rl--; } // write all (partial) run lengths
       }
#endif
    }

    using  VecMap = Eigen::Map<Eigen::Matrix<Field, Eigen::Dynamic, 1>, Eigen::Unaligned>;
    using CVecMap = Eigen::Map<const Eigen::Matrix<Field, Eigen::Dynamic, 1>, Eigen::Unaligned>;

    // Add the child Schur complement into this supernode's front.
    for (Int j = 0; j < num_child_diag_indices; ++j) {
        const Int j_rel = child_rel_indices[j];
        const Field* child_column = child_schur_complement.Pointer(0, j);

        // Contribute into the upper-left diagonal block of the front.
        Field* diag_column = diagonal_block.Pointer(0, j_rel);
        for (Int i = j; i < num_child_diag_indices; ++i) {
            const Int i_rel = child_rel_indices[i];
            diag_column[i_rel] += child_column[i];
        }

        // Contribute into the lower-left block of the front.
        Field* lower_column = lower_block.Pointer(0, j_rel);
#if VECTORIZE_MERGE_SCHUR_COMPLEMENTS
        for (Int i = num_child_diag_indices; i < child_degree; i += child_rel_indices_rl[i]) {
            int len = child_rel_indices_rl[i];
            Field * __restrict__ ptr = lower_column + child_rel_indices[i];
            const Field *__restrict__ iptr = child_column + i;
            while (len-- > 0) {
                *ptr++ += *iptr++;
            }
        }
        // for (Int i = num_child_diag_indices; i < child_degree; i += child_rel_indices_rl[i]) {
        //     VecMap(lower_column + child_rel_indices[i], child_rel_indices_rl[i])
        //             += CVecMap(child_column + i, child_rel_indices_rl[i]);
        // }
#else
        for (Int i = num_child_diag_indices; i < child_degree; ++i) {
            const Int i_rel = child_rel_indices[i];
            lower_column[i_rel] += child_column[i];
        }
#endif
    }
    if (freshShurComplement) {
        // Clear and contribute into the bottom-right block of the front.
        eigenMap(schur_complement).setZero();
        for (Int j = num_child_diag_indices; j < child_degree; ++j) {
            const Field* child_column = child_schur_complement.Pointer(0, j);
            Field* schur_column = schur_complement.Pointer(0, child_rel_indices[j]);
#if 0
            for (Int i = j; i < child_degree; i += child_rel_indices_rl[i]) {
                VecMap(schur_column + child_rel_indices[i], child_rel_indices_rl[i])
                        = CVecMap(child_column + i, child_rel_indices_rl[i]);
            }
#else
            for (Int i = j; i < child_degree; ++i)
                schur_column[child_rel_indices[i]] = child_column[i];
#endif
        }
    } 
    else {
        // Contribute into the bottom-right block of the front.
        for (Int j = num_child_diag_indices; j < child_degree; ++j) {
            const Field* child_column = child_schur_complement.Pointer(0, j);
            Field* schur_column = schur_complement.Pointer(0, child_rel_indices[j]);
#if 0
            for (Int i = j; i < child_degree; i += child_rel_indices_rl[i]) {
                VecMap(schur_column + child_rel_indices[i], child_rel_indices_rl[i])
                        += CVecMap(child_column + i, child_rel_indices_rl[i]);
            }
#else
            for (Int i = j; i < child_degree; ++i)
                schur_column[child_rel_indices[i]] += child_column[i];
#endif
        }
    }
}

template <class Field>
void MergeChildSchurComplements(Int supernode,
                                const SymmetricOrdering& ordering,
                                LowerFactor<Field>* lower_factor,
                                DiagonalFactor<Field>* diagonal_factor,
                                RightLookingSharedState<Field>* shared_state) {
  const Int child_beg = ordering.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering.assembly_forest.child_offsets[supernode + 1];

  // Output destination buffers
  BlasMatrixView<Field> lower_block      = lower_factor->blocks[supernode];
  BlasMatrixView<Field> diagonal_block   = diagonal_factor->blocks[supernode];
  BlasMatrixView<Field> schur_complement = shared_state->schur_complements[supernode];

  for (Int child_index = child_beg; child_index < child_end; ++child_index) {
      const Int child = ordering.assembly_forest.children[child_index];
      // Input schur complement
      const BlasMatrixView<Field> &child_schur_complement = shared_state->schur_complements[child];
      MergeChildSchurComplement(supernode, child, ordering, lower_factor,
                                child_schur_complement, lower_block, diagonal_block,
                                schur_complement, /* freshShurComplement = */ child_index == child_beg);
  }
}

template <class Field>
Int FactorDiagonalBlock(
    Int block_size, SymmetricFactorizationType factorization_type,
    const DynamicRegularizationParams<Field>& dynamic_reg_params,
    BlasMatrixView<Field>* diagonal_block,
    std::vector<std::pair<Int, ComplexBase<Field>>>* dynamic_regularization) {
  Int num_pivots;
  if (factorization_type == kCholeskyFactorization) {
    if (dynamic_reg_params.enabled) {
      num_pivots = DynamicallyRegularizedLowerCholeskyFactorization(
          block_size, dynamic_reg_params, diagonal_block,
          dynamic_regularization);
    } else {
      num_pivots = LowerCholeskyFactorizationDynamicBLASDispatch(block_size, diagonal_block);
    }
  } else if (factorization_type == kLDLAdjointFactorization) {
    if (dynamic_reg_params.enabled) {
      num_pivots = DynamicallyRegularizedLDLAdjointFactorization(
          block_size, dynamic_reg_params, diagonal_block,
          dynamic_regularization);
    } else {
      num_pivots = LDLAdjointFactorization(block_size, diagonal_block);
    }
  } else {
    num_pivots = LDLTransposeFactorization(block_size, diagonal_block);
  }
  return num_pivots;
}

template <class Field>
Int PivotedFactorDiagonalBlock(Int block_size,
                               SymmetricFactorizationType factorization_type,
                               BlasMatrixView<Field>* diagonal_block,
                               BlasMatrixView<Int>* permutation) {
  CATAMARI_ASSERT(factorization_type == kLDLAdjointFactorization,
                  "Pivoting is currently only supported for LDL^H fact'ns.");
  Int num_pivots =
      PivotedLDLAdjointFactorization(block_size, diagonal_block, permutation);
  return num_pivots;
}

template <class Field>
void SolveAgainstDiagonalBlock(
    SymmetricFactorizationType factorization_type,
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* lower_matrix) {
  if (!lower_matrix->height) {
    return;
  }
  if (factorization_type == kCholeskyFactorization) {
    // Solve against the lower-triangular matrix L(K, K)' from the right.
    RightLowerAdjointTriangularSolves(triangular_matrix, lower_matrix);
  } else if (factorization_type == kLDLAdjointFactorization) {
    // Solve against D(K, K) L(K, K)' from the right.
    RightDiagonalTimesLowerAdjointUnitTriangularSolves(triangular_matrix,
                                                       lower_matrix);
  } else {
    // Solve against D(K, K) L(K, K)^T from the right.
    RightDiagonalTimesLowerTransposeUnitTriangularSolves(triangular_matrix,
                                                         lower_matrix);
  }
}

}  // namespace supernodal_ldl
}  // namespace catamari

#include "catamari/sparse_ldl/supernodal/supernode_utils/openmp-impl.hpp"

#endif  // ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_SUPERNODE_UTILS_IMPL_H_
