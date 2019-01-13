/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SUPERNODAL_LDL_SUPERNODE_UTILS_IMPL_H_
#define CATAMARI_LDL_SUPERNODAL_LDL_SUPERNODE_UTILS_IMPL_H_

#include "catamari/ldl/supernodal_ldl/supernode_utils.hpp"

namespace catamari {
namespace supernodal_ldl {

inline void MemberToIndex(Int num_rows,
                          const std::vector<Int>& supernode_starts,
                          std::vector<Int>* member_to_index) {
  member_to_index->resize(num_rows);

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
  CATAMARI_ASSERT(supernode == static_cast<Int>(supernode_starts.size()) - 2,
                  "Did not end on the last supernode.");
}

inline void EliminationForestFromParents(const std::vector<Int>& parents,
                                         std::vector<Int>* children,
                                         std::vector<Int>* child_offsets) {
  const Int num_indices = parents.size();

  std::vector<Int> num_children(num_indices, 0);
  for (Int index = 0; index < num_indices; ++index) {
    const Int parent = parents[index];
    if (parent >= 0) {
      ++num_children[parent];
    }
  }

  OffsetScan(num_children, child_offsets);

  children->resize(num_indices);
  auto offsets_copy = *child_offsets;
  for (Int index = 0; index < num_indices; ++index) {
    const Int parent = parents[index];
    if (parent >= 0) {
      (*children)[offsets_copy[parent]++] = index;
    }
  }
}

inline void EliminationForestAndRootsFromParents(
    const std::vector<Int>& parents, std::vector<Int>* children,
    std::vector<Int>* child_offsets, std::vector<Int>* roots) {
  const Int num_indices = parents.size();

  // Compute the number of children (initially stored in 'child_offsets') of
  // each vertex. Along the way, count the number of trees in the forest.
  Int num_roots = 0;
  child_offsets->clear();
  child_offsets->resize(num_indices + 1, 0);
  for (Int index = 0; index < num_indices; ++index) {
    const Int parent = parents[index];
    if (parent >= 0) {
      ++(*child_offsets)[parent];
    } else {
      ++num_roots;
    }
  }

  // Compute the child offsets using an in-place scan.
  Int num_total_children = 0;
  for (Int index = 0; index < num_indices; ++index) {
    const Int num_children = (*child_offsets)[index];
    (*child_offsets)[index] = num_total_children;
    num_total_children += num_children;
  }
  (*child_offsets)[num_indices] = num_total_children;

  // Pack the children into the 'children' buffer.
  children->resize(num_total_children);
  roots->reserve(num_roots);
  std::vector<Int> offsets_copy = *child_offsets;
  for (Int index = 0; index < num_indices; ++index) {
    const Int parent = parents[index];
    if (parent >= 0) {
      (*children)[offsets_copy[parent]++] = index;
    } else {
      roots->push_back(index);
    }
  }
}

inline void ConvertFromScalarToSupernodalEliminationForest(
    Int num_supernodes, const std::vector<Int>& parents,
    const std::vector<Int>& member_to_index,
    std::vector<Int>* supernode_parents) {
  const Int num_rows = parents.size();
  supernode_parents->resize(num_supernodes, -1);
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
                                const std::vector<Int>& permutation,
                                const std::vector<Int>& inverse_permutation,
                                const std::vector<Int>& supernode_sizes) {
  const Int num_rows = matrix.NumRows();

  std::vector<Int> parents, degrees;
  scalar_ldl::EliminationForestAndDegrees(
      matrix, permutation, inverse_permutation, &parents, &degrees);

  std::vector<Int> supernode_starts, member_to_index;
  OffsetScan(supernode_sizes, &supernode_starts);
  MemberToIndex(num_rows, supernode_starts, &member_to_index);

  std::vector<Int> row_structure(num_rows);
  std::vector<Int> pattern_flags(num_rows);

  bool valid = true;
  for (Int row = 0; row < num_rows; ++row) {
    const Int row_supernode = member_to_index[row];

    pattern_flags[row] = row;
    const Int num_packed = scalar_ldl::ComputeRowPattern(
        matrix, permutation, inverse_permutation, parents, row,
        pattern_flags.data(), row_structure.data());
    std::sort(row_structure.data(), row_structure.data() + num_packed);

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

// TODO(Jack Poulson): Provide a multithreaded version which splits the columns
// and patches together any overlapping supernodes.
template <class Field>
void FormFundamentalSupernodes(const CoordinateMatrix<Field>& matrix,
                               const std::vector<Int>& permutation,
                               const std::vector<Int>& inverse_permutation,
                               const std::vector<Int>& parents,
                               const std::vector<Int>& degrees,
                               std::vector<Int>* supernode_sizes,
                               scalar_ldl::LowerStructure* scalar_structure) {
  const Int num_rows = matrix.NumRows();

  // We will only fill the indices and offsets of the factorization.
  scalar_ldl::FillStructureIndices(matrix, permutation, inverse_permutation,
                                   parents, degrees, scalar_structure);

  supernode_sizes->clear();
  if (!num_rows) {
    return;
  }

  // We will iterate down each column structure to determine the supernodes.
  std::vector<Int> column_ptrs = scalar_structure->column_offsets;

  Int supernode_start = 0;
  supernode_sizes->reserve(num_rows);
  supernode_sizes->push_back(1);
  for (Int column = 1; column < num_rows; ++column) {
    // Ensure that the diagonal block would be fully-connected. Due to the
    // loop invariant, each column pointer in the current supernode would need
    // to be pointing to index 'column'.
    bool dense_diagonal_block = true;
    for (Int j = supernode_start; j < column; ++j) {
      const Int index = column_ptrs[j];
      const Int next_column_beg = scalar_structure->column_offsets[j + 1];
      if (index < next_column_beg &&
          scalar_structure->indices[index] == column) {
        ++column_ptrs[j];
      } else {
        dense_diagonal_block = false;
        break;
      }
    }
    if (!dense_diagonal_block) {
      // This column begins a new supernode.
      supernode_start = column;
      supernode_sizes->push_back(1);
      continue;
    }

    // Test if the structure of this supernode matches that of the previous
    // column (with all indices up to this column removed). We first test that
    // the set sizes are equal and then test the individual entries.

    // Test that the set sizes match.
    const Int column_beg = column_ptrs[column];
    const Int structure_size = column_ptrs[column + 1] - column_beg;
    const Int prev_column_ptr = column_ptrs[column - 1];
    const Int prev_remaining_structure_size = column_beg - prev_column_ptr;
    if (structure_size != prev_remaining_structure_size) {
      // This column begins a new supernode.
      supernode_start = column;
      supernode_sizes->push_back(1);
      continue;
    }

    // Test that the individual entries match.
    bool equal_structures = true;
    for (Int i = 0; i < structure_size; ++i) {
      const Int row = scalar_structure->indices[column_beg + i];
      const Int prev_row = scalar_structure->indices[prev_column_ptr + i];
      if (prev_row != row) {
        equal_structures = false;
        break;
      }
    }
    if (!equal_structures) {
      // This column begins a new supernode.
      supernode_start = column;
      supernode_sizes->push_back(1);
      continue;
    }

    // All tests passed, so we may extend the current supernode to incorporate
    // this column.
    ++supernode_sizes->back();
  }

#ifdef CATAMARI_DEBUG
  if (!ValidFundamentalSupernodes(matrix, permutation, inverse_permutation,
                                  *supernode_sizes)) {
    std::cerr << "Invalid fundamental supernodes." << std::endl;
    return;
  }
#endif
}

inline MergableStatus MergableSupernode(
    Int child_tail, Int parent_tail, Int child_size, Int parent_size,
    Int num_child_explicit_zeros, Int num_parent_explicit_zeros,
    const std::vector<Int>& orig_member_to_index,
    const scalar_ldl::LowerStructure& scalar_structure,
    const SupernodalRelaxationControl& control) {
  const Int parent = orig_member_to_index[parent_tail];
  const Int child_structure_beg = scalar_structure.column_offsets[child_tail];
  const Int child_structure_end =
      scalar_structure.column_offsets[child_tail + 1];
  const Int parent_structure_beg = scalar_structure.column_offsets[parent_tail];
  const Int parent_structure_end =
      scalar_structure.column_offsets[parent_tail + 1];
  const Int child_structure_size = child_structure_end - child_structure_beg;
  const Int parent_structure_size = parent_structure_end - parent_structure_beg;

  MergableStatus status;

  // Count the number of intersections of the child's structure with the
  // parent supernode (and then leave the child structure pointer after the
  // parent supernode.
  //
  // We know that any intersections of the child with the parent occur at the
  // beginning of the child's structure.
  Int num_child_parent_intersections = 0;
  {
    Int child_structure_ptr = child_structure_beg;
    while (child_structure_ptr < child_structure_end) {
      const Int row = scalar_structure.indices[child_structure_ptr];
      const Int orig_row_supernode = orig_member_to_index[row];
      CATAMARI_ASSERT(orig_row_supernode >= parent,
                      "There was an intersection before the parent.");
      if (orig_row_supernode == parent) {
        ++child_structure_ptr;
        ++num_child_parent_intersections;
      } else {
        break;
      }
    }
  }
  const Int num_missing_parent_intersections =
      parent_size - num_child_parent_intersections;

  // Since the structure of L21 contains the structure of L20, we need only
  // compare the sizes of the structure of the parent supernode to the
  // remaining structure size to know how many indices will need to be
  // introduced into L20.
  const Int remaining_child_structure_size =
      child_structure_size - num_child_parent_intersections;
  const Int num_missing_structure_indices =
      parent_structure_size - remaining_child_structure_size;

  const Int num_new_zeros =
      (num_missing_parent_intersections + num_missing_structure_indices) *
      child_size;
  const Int num_old_zeros =
      num_child_explicit_zeros + num_parent_explicit_zeros;
  const Int num_zeros = num_new_zeros + num_old_zeros;
  status.num_merged_zeros = num_zeros;

  // Check if the merge would meet the absolute merge criterion.
  if (num_zeros <= control.allowable_supernode_zeros) {
    status.mergable = true;
    return status;
  }

  // Check if the merge would meet the relative merge criterion.
  const Int num_expanded_entries =
      /* num_nonzeros(L00) */
      (child_size + (child_size + 1)) / 2 +
      /* num_nonzeros(L10) */
      parent_size * child_size +
      /* num_nonzeros(L20) */
      remaining_child_structure_size * child_size +
      /* num_nonzeros(L11) */
      (parent_size + (parent_size + 1)) / 2 +
      /* num_nonzeros(L21) */
      parent_structure_size * parent_size;
  if (num_zeros <=
      control.allowable_supernode_zero_ratio * num_expanded_entries) {
    status.mergable = true;
    return status;
  }

  status.mergable = false;
  return status;
}

inline void MergeChildren(
    Int parent, const std::vector<Int>& orig_supernode_starts,
    const std::vector<Int>& orig_supernode_sizes,
    const std::vector<Int>& orig_member_to_index,
    const std::vector<Int>& children, const std::vector<Int>& child_offsets,
    const scalar_ldl::LowerStructure& scalar_structure,
    const SupernodalRelaxationControl& control,
    std::vector<Int>* supernode_sizes, std::vector<Int>* num_explicit_zeros,
    std::vector<Int>* last_merged_child, std::vector<Int>* merge_parents) {
  const Int child_beg = child_offsets[parent];
  const Int num_children = child_offsets[parent + 1] - child_beg;

  // TODO(Jack Poulson): Reserve a default size for these arrays.
  std::vector<Int> mergable_children;
  std::vector<Int> num_merged_zeros;

  // The following loop can execute at most 'num_children' times.
  while (true) {
    // Compute the list of child indices that can be merged.
    for (Int child_index = 0; child_index < num_children; ++child_index) {
      const Int child = children[child_beg + child_index];
      if ((*merge_parents)[child] != -1) {
        continue;
      }

      const Int child_tail =
          orig_supernode_starts[child] + orig_supernode_sizes[child] - 1;
      const Int parent_tail =
          orig_supernode_starts[parent] + orig_supernode_sizes[parent] - 1;

      const Int child_size = (*supernode_sizes)[child];
      const Int parent_size = (*supernode_sizes)[parent];

      const Int num_child_explicit_zeros = (*num_explicit_zeros)[child];
      const Int num_parent_explicit_zeros = (*num_explicit_zeros)[parent];

      const MergableStatus status =
          MergableSupernode(child_tail, parent_tail, child_size, parent_size,
                            num_child_explicit_zeros, num_parent_explicit_zeros,
                            orig_member_to_index, scalar_structure, control);
      if (status.mergable) {
        mergable_children.push_back(child_index);
        num_merged_zeros.push_back(status.num_merged_zeros);
      }
    }

    // Skip this supernode if no children can be merged into it.
    if (mergable_children.empty()) {
      break;
    }
    const Int first_mergable_child = children[child_beg + mergable_children[0]];

    // Select the largest mergable supernode.
    Int merging_index = 0;
    Int largest_mergable_size = (*supernode_sizes)[first_mergable_child];
    for (std::size_t mergable_index = 1;
         mergable_index < mergable_children.size(); ++mergable_index) {
      const Int child_index = mergable_children[mergable_index];
      const Int child = children[child_beg + child_index];
      const Int child_size = (*supernode_sizes)[child];
      if (child_size > largest_mergable_size) {
        merging_index = mergable_index;
        largest_mergable_size = child_size;
      }
    }
    const Int child_index = mergable_children[merging_index];
    const Int child = children[child_beg + child_index];

    // Absorb the child size into the parent.
    (*supernode_sizes)[parent] += (*supernode_sizes)[child];
    (*supernode_sizes)[child] = 0;

    // Update the number of explicit zeros in the merged supernode.
    (*num_explicit_zeros)[parent] = num_merged_zeros[merging_index];

    // Mark the child as merged.
    //
    // TODO(Jack Poulson): Consider following a similar strategy as quotient
    // and using SYMMETRIC_INDEX to pack a parent into the negative indices.

    if ((*last_merged_child)[parent] == -1) {
      (*merge_parents)[child] = parent;
    } else {
      (*merge_parents)[child] = (*last_merged_child)[parent];
    }

    if ((*last_merged_child)[child] == -1) {
      (*last_merged_child)[parent] = child;
    } else {
      (*last_merged_child)[parent] = (*last_merged_child)[child];
    }

    // Clear the mergable children information since it is now stale.
    mergable_children.clear();
    num_merged_zeros.clear();
  }
}

inline void RelaxSupernodes(
    const std::vector<Int>& orig_parents,
    const std::vector<Int>& orig_supernode_sizes,
    const std::vector<Int>& orig_supernode_starts,
    const std::vector<Int>& orig_supernode_parents,
    const std::vector<Int>& orig_supernode_degrees,
    const std::vector<Int>& orig_member_to_index,
    const scalar_ldl::LowerStructure& scalar_structure,
    const SupernodalRelaxationControl& control,
    std::vector<Int>* relaxed_permutation,
    std::vector<Int>* relaxed_inverse_permutation,
    std::vector<Int>* relaxed_parents,
    std::vector<Int>* relaxed_supernode_parents,
    std::vector<Int>* relaxed_supernode_degrees,
    std::vector<Int>* relaxed_supernode_sizes,
    std::vector<Int>* relaxed_supernode_starts,
    std::vector<Int>* relaxed_supernode_member_to_index) {
  const Int num_rows = orig_supernode_starts.back();
  const Int num_supernodes = orig_supernode_sizes.size();

  // Construct the down-links for the elimination forest.
  std::vector<Int> children;
  std::vector<Int> child_offsets;
  EliminationForestFromParents(orig_supernode_parents, &children,
                               &child_offsets);

  // Initialize the sizes of the merged supernodes, using the original indexing
  // and absorbing child sizes into the parents.
  std::vector<Int> supernode_sizes = orig_supernode_sizes;

  // Initialize the number of explicit zeros stored in each original supernode.
  std::vector<Int> num_explicit_zeros(num_supernodes, 0);

  std::vector<Int> last_merged_children(num_supernodes, -1);
  std::vector<Int> merge_parents(num_supernodes, -1);
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    MergeChildren(supernode, orig_supernode_starts, orig_supernode_sizes,
                  orig_member_to_index, children, child_offsets,
                  scalar_structure, control, &supernode_sizes,
                  &num_explicit_zeros, &last_merged_children, &merge_parents);
  }

  // Count the number of remaining supernodes and construct a map from the
  // original to relaxed indices.
  std::vector<Int> original_to_relaxed(num_supernodes, -1);
  Int num_relaxed_supernodes = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    if (merge_parents[supernode] != -1) {
      continue;
    }
    original_to_relaxed[supernode] = num_relaxed_supernodes++;
  }

  // Fill in the parents for the relaxed supernodal elimination forest.
  // Simultaneously, fill in the degrees of the relaxed supernodes (which are
  // the same as the degrees of the parents of each merge tree).
  relaxed_supernode_parents->resize(num_relaxed_supernodes);
  relaxed_supernode_degrees->resize(num_relaxed_supernodes);
  Int relaxed_offset = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    if (merge_parents[supernode] != -1) {
      continue;
    }
    (*relaxed_supernode_degrees)[relaxed_offset] =
        orig_supernode_degrees[supernode];

    Int parent = orig_supernode_parents[supernode];
    if (parent == -1) {
      // This is a root node, so mark it as such in the relaxation.
      (*relaxed_supernode_parents)[relaxed_offset++] = -1;
      continue;
    }

    // Compute the root of the merge sequence of our parent.
    while (merge_parents[parent] != -1) {
      parent = merge_parents[parent];
    }

    // Convert the root of the merge sequence of our parent from the original
    // to the relaxed indexing.
    const Int relaxed_parent = original_to_relaxed[parent];
    (*relaxed_supernode_parents)[relaxed_offset++] = relaxed_parent;
  }

  // Fill the inverse permutation, the supernode sizes, and the supernode
  // offsets.
  std::vector<Int> relaxation_inverse_permutation(num_rows);
  {
    relaxed_supernode_sizes->resize(num_relaxed_supernodes);
    relaxed_supernode_starts->resize(num_relaxed_supernodes + 1);
    relaxed_supernode_member_to_index->resize(num_rows);
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
      (*relaxed_supernode_starts)[relaxed_supernode] = pack_offset;
      Int supernode_size = 0;
      Int supernode_to_pack = leaf_of_merge;
      while (true) {
        const Int start = orig_supernode_starts[supernode_to_pack];
        const Int size = orig_supernode_sizes[supernode_to_pack];
        for (Int j = 0; j < size; ++j) {
          (*relaxed_supernode_member_to_index)[pack_offset] = relaxed_supernode;
          relaxation_inverse_permutation[pack_offset++] = start + j;
        }
        supernode_size += size;

        if (merge_parents[supernode_to_pack] == -1) {
          break;
        }
        supernode_to_pack = merge_parents[supernode_to_pack];
      }
      (*relaxed_supernode_sizes)[relaxed_supernode] = supernode_size;
    }
    CATAMARI_ASSERT(num_rows == pack_offset, "Did not pack num_rows indices.");
    (*relaxed_supernode_starts)[num_relaxed_supernodes] = num_rows;
  }

  // Compute the relaxation permutation.
  std::vector<Int> relaxation_permutation(num_rows);
  for (Int row = 0; row < num_rows; ++row) {
    relaxation_permutation[relaxation_inverse_permutation[row]] = row;
  }

  // Permute the 'orig_parents' array to form the 'parents' array.
  relaxed_parents->resize(num_rows);
  for (Int row = 0; row < num_rows; ++row) {
    const Int orig_parent = orig_parents[row];
    if (orig_parent == -1) {
      (*relaxed_parents)[relaxation_permutation[row]] = -1;
    } else {
      (*relaxed_parents)[relaxation_permutation[row]] =
          relaxation_permutation[orig_parent];
    }
  }

  // Compose the relaxation permutation with the original permutation.
  if (relaxed_permutation->empty()) {
    *relaxed_permutation = relaxation_permutation;
    *relaxed_inverse_permutation = relaxation_inverse_permutation;
  } else {
    // Compuse the relaxation and original permutations.
    std::vector<Int> perm_copy = *relaxed_permutation;
    for (Int row = 0; row < num_rows; ++row) {
      (*relaxed_permutation)[row] = relaxation_permutation[perm_copy[row]];
    }

    // Invert the composed permutation.
    InvertPermutation(*relaxed_permutation, relaxed_inverse_permutation);
  }
}

template <class Field>
void SupernodalDegrees(const CoordinateMatrix<Field>& matrix,
                       const std::vector<Int>& permutation,
                       const std::vector<Int>& inverse_permutation,
                       const std::vector<Int>& supernode_sizes,
                       const std::vector<Int>& supernode_starts,
                       const std::vector<Int>& member_to_index,
                       const std::vector<Int>& parents,
                       std::vector<Int>* supernode_degrees) {
  const Int num_rows = matrix.NumRows();
  const Int num_supernodes = supernode_sizes.size();
  const bool have_permutation = !permutation.empty();

  // A data structure for marking whether or not an index is in the pattern
  // of the active row of the lower-triangular factor.
  std::vector<Int> pattern_flags(num_rows);

  // A data structure for marking whether or not a supernode is in the pattern
  // of the active row of the lower-triangular factor.
  std::vector<Int> supernode_pattern_flags(num_supernodes);

  // Initialize the number of entries that will be stored into each supernode
  supernode_degrees->clear();
  supernode_degrees->resize(num_supernodes, 0);

  const std::vector<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int main_supernode = member_to_index[row];
    pattern_flags[row] = row;
    supernode_pattern_flags[main_supernode] = row;

    const Int orig_row = have_permutation ? inverse_permutation[row] : row;
    const Int row_beg = matrix.RowEntryOffset(orig_row);
    const Int row_end = matrix.RowEntryOffset(orig_row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      Int descendant =
          have_permutation ? permutation[entry.column] : entry.column;
      Int descendant_supernode = member_to_index[descendant];

      // We are traversing the strictly lower triangle and know that the
      // indices are sorted.
      if (descendant >= row) {
        if (have_permutation) {
          continue;
        } else {
          break;
        }
      }

      // Look for new entries in the pattern by walking up to the root of this
      // subtree of the elimination forest from index 'descendant'. Any unset
      // parent pointers can be filled in during the traversal, as the current
      // row index would then be the parent.
      while (pattern_flags[descendant] != row) {
        // Mark index 'descendant' as in the pattern of row 'row'.
        pattern_flags[descendant] = row;

        descendant_supernode = member_to_index[descendant];
        CATAMARI_ASSERT(descendant_supernode <= main_supernode,
                        "Descendant supernode was larger than main supernode.");
        if (descendant_supernode < main_supernode) {
          if (supernode_pattern_flags[descendant_supernode] != row) {
            supernode_pattern_flags[descendant_supernode] = row;
            ++(*supernode_degrees)[descendant_supernode];
          }
        }

        // Move up to the parent in this subtree of the elimination forest.
        // Moving to the parent will increase the index (but remain bounded
        // from above by 'row').
        descendant = parents[descendant];
      }
    }
  }
}

template <class Field>
void FillStructureIndices(const CoordinateMatrix<Field>& matrix,
                          const std::vector<Int>& permutation,
                          const std::vector<Int>& inverse_permutation,
                          const std::vector<Int>& parents,
                          const std::vector<Int>& supernode_sizes,
                          const std::vector<Int>& supernode_member_to_index,
                          LowerFactor<Field>* lower_factor,
                          Int* max_descendant_entries) {
  const Int num_rows = matrix.NumRows();
  const Int num_supernodes = supernode_sizes.size();
  const bool have_permutation = !permutation.empty();

  // A data structure for marking whether or not a node is in the pattern of
  // the active row of the lower-triangular factor.
  std::vector<Int> pattern_flags(num_rows);

  // A data structure for marking whether or not a supernode is in the pattern
  // of the active row of the lower-triangular factor.
  std::vector<Int> supernode_pattern_flags(num_supernodes);

  // A set of pointers for keeping track of where to insert supernode pattern
  // indices.
  std::vector<Int*> supernode_ptrs(num_supernodes);
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    supernode_ptrs[supernode] = lower_factor->Structure(supernode);
  }

  // Fill in the structure indices.
  const std::vector<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int main_supernode = supernode_member_to_index[row];
    pattern_flags[row] = row;
    supernode_pattern_flags[main_supernode] = row;

    const Int orig_row = have_permutation ? inverse_permutation[row] : row;
    const Int row_beg = matrix.RowEntryOffset(orig_row);
    const Int row_end = matrix.RowEntryOffset(orig_row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      Int descendant =
          have_permutation ? permutation[entry.column] : entry.column;
      Int descendant_supernode = supernode_member_to_index[descendant];

      if (descendant >= row) {
        if (have_permutation) {
          continue;
        } else {
          // We are traversing the strictly lower triangle and know that the
          // indices are sorted.
          break;
        }
      }

      // Look for new entries in the pattern by walking up to the root of this
      // subtree of the elimination forest.
      while (pattern_flags[descendant] != row) {
        // Mark 'descendant' as in the pattern of this row.
        pattern_flags[descendant] = row;

        descendant_supernode = supernode_member_to_index[descendant];
        CATAMARI_ASSERT(descendant_supernode <= main_supernode,
                        "Descendant supernode was larger than main supernode.");
        if (descendant_supernode == main_supernode) {
          break;
        }

        if (supernode_pattern_flags[descendant_supernode] != row) {
          supernode_pattern_flags[descendant_supernode] = row;
          CATAMARI_ASSERT(
              descendant_supernode < main_supernode,
              "Descendant supernode was as large as main supernode.");
          CATAMARI_ASSERT(
              supernode_ptrs[descendant_supernode] >=
                      lower_factor->Structure(descendant_supernode) &&
                  supernode_ptrs[descendant_supernode] <
                      lower_factor->Structure(descendant_supernode + 1),
              "Left supernode's indices.");
          *supernode_ptrs[descendant_supernode] = row;
          ++supernode_ptrs[descendant_supernode];
        }

        // Move up to the parent in this subtree of the elimination forest.
        // Moving to the parent will increase the index (but remain bounded
        // from above by 'row').
        descendant = parents[descendant];
      }
    }
  }

#ifdef CATAMARI_DEBUG
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int* index_beg = lower_factor->Structure(supernode);
    const Int* index_end = lower_factor->Structure(supernode + 1);
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
#endif

  lower_factor->FillIntersectionSizes(
      supernode_sizes, supernode_member_to_index, max_descendant_entries);
}

template <class Field>
void FillSubtreeWorkEstimates(Int root,
                              const std::vector<Int>& supernode_children,
                              const std::vector<Int>& supernode_child_offsets,
                              const LowerFactor<Field>& lower_factor,
                              std::vector<double>* work_estimates) {
  const Int child_beg = supernode_child_offsets[root];
  const Int child_end = supernode_child_offsets[root + 1];
  const Int num_children = child_end - child_beg;

  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child = supernode_children[child_beg + child_index];
    FillSubtreeWorkEstimates(child, supernode_children, supernode_child_offsets,
                             lower_factor, work_estimates);
    (*work_estimates)[root] += (*work_estimates)[child];
  }

  const ConstBlasMatrix<Field>& lower_block =
      lower_factor.blocks[root].ToConst();
  const Int supernode_size = lower_block.width;
  const Int degree = lower_block.height;
  (*work_estimates)[root] += std::pow(1. * supernode_size, 3.) / 3;
  (*work_estimates)[root] += std::pow(1. * degree, 2.) * supernode_size;
}

template <class Field>
void FillNonzeros(const CoordinateMatrix<Field>& matrix,
                  const std::vector<Int>& permutation,
                  const std::vector<Int>& inverse_permutation,
                  const std::vector<Int>& supernode_starts,
                  const std::vector<Int>& supernode_sizes,
                  const std::vector<Int>& supernode_member_to_index,
                  LowerFactor<Field>* lower_factor,
                  DiagonalFactor<Field>* diagonal_factor) {
  const Int num_rows = matrix.NumRows();
  const bool have_permutation = !permutation.empty();

  const std::vector<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int supernode = supernode_member_to_index[row];
    const Int supernode_start = supernode_starts[supernode];

    const Int orig_row = have_permutation ? inverse_permutation[row] : row;
    const Int row_beg = matrix.RowEntryOffset(orig_row);
    const Int row_end = matrix.RowEntryOffset(orig_row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      const Int column =
          have_permutation ? permutation[entry.column] : entry.column;
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
      const Int* column_index_beg = lower_factor->Structure(column_supernode);
      const Int* column_index_end =
          lower_factor->Structure(column_supernode + 1);
      const Int* iter =
          std::lower_bound(column_index_beg, column_index_end, row);
      CATAMARI_ASSERT(iter != column_index_end, "Exceeded column indices.");
      CATAMARI_ASSERT(*iter == row, "Entry (" + std::to_string(row) + ", " +
                                        std::to_string(column) +
                                        ") wasn't in the structure.");
      const Int rel_row = std::distance(column_index_beg, iter);
      const Int rel_column = column - supernode_starts[column_supernode];
      lower_factor->blocks[column_supernode](rel_row, rel_column) = entry.value;
    }
  }
}

template <class Field>
Int ComputeRowPattern(const CoordinateMatrix<Field>& matrix,
                      const std::vector<Int>& permutation,
                      const std::vector<Int>& inverse_permutation,
                      const std::vector<Int>& supernode_sizes,
                      const std::vector<Int>& supernode_starts,
                      const std::vector<Int>& member_to_index,
                      const std::vector<Int>& supernode_parents,
                      Int main_supernode, Int* pattern_flags,
                      Int* row_structure) {
  Int num_packed = 0;
  const bool have_permutation = !permutation.empty();
  const std::vector<MatrixEntry<Field>>& entries = matrix.Entries();

  // Take the union of the row patterns of each row in the supernode.
  const Int main_supernode_size = supernode_sizes[main_supernode];
  const Int main_supernode_start = supernode_starts[main_supernode];
  for (Int row = main_supernode_start;
       row < main_supernode_start + main_supernode_size; ++row) {
    const Int orig_row = have_permutation ? inverse_permutation[row] : row;
    const Int row_beg = matrix.RowEntryOffset(orig_row);
    const Int row_end = matrix.RowEntryOffset(orig_row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      const Int descendant =
          have_permutation ? permutation[entry.column] : entry.column;

      Int descendant_supernode = member_to_index[descendant];
      if (descendant_supernode >= main_supernode) {
        if (have_permutation) {
          continue;
        } else {
          break;
        }
      }

      // Walk up to the root of the current subtree of the elimination
      // forest, stopping if we encounter a member already marked as in the
      // row pattern.
      while (pattern_flags[descendant_supernode] != main_supernode) {
        // Place 'descendant_supernode' into the pattern of this supernode.
        row_structure[num_packed++] = descendant_supernode;
        pattern_flags[descendant_supernode] = main_supernode;

        descendant_supernode = supernode_parents[descendant_supernode];
      }
    }
  }

  return num_packed;
}

template <class Field>
void FormScaledTranspose(SymmetricFactorizationType factorization_type,
                         const ConstBlasMatrix<Field>& diagonal_block,
                         const ConstBlasMatrix<Field>& matrix,
                         BlasMatrix<Field>* scaled_transpose) {
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
void UpdateDiagonalBlock(SymmetricFactorizationType factorization_type,
                         const std::vector<Int>& supernode_starts,
                         const LowerFactor<Field>& lower_factor,
                         Int main_supernode, Int descendant_supernode,
                         Int descendant_main_rel_row,
                         const ConstBlasMatrix<Field>& descendant_main_matrix,
                         const ConstBlasMatrix<Field>& scaled_transpose,
                         BlasMatrix<Field>* main_diag_block,
                         BlasMatrix<Field>* workspace_matrix) {
  typedef ComplexBase<Field> Real;
  const Int main_supernode_size = main_diag_block->height;
  const Int descendant_main_intersect_size = scaled_transpose.width;

  const bool inplace_update =
      descendant_main_intersect_size == main_supernode_size;
  BlasMatrix<Field>* accumulation_block =
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
        lower_factor.Structure(descendant_supernode) + descendant_main_rel_row;

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
    Int main_supernode, Int descendant_supernode, Int main_active_rel_row,
    Int descendant_main_rel_row, Int descendant_active_rel_row,
    const std::vector<Int>& supernode_starts,
    const std::vector<Int>& supernode_member_to_index,
    const ConstBlasMatrix<Field>& scaled_transpose,
    const ConstBlasMatrix<Field>& descendant_active_matrix,
    const LowerFactor<Field>& lower_factor,
    BlasMatrix<Field>* main_active_block, BlasMatrix<Field>* workspace_matrix) {
  const Int main_supernode_size = lower_factor.blocks[main_supernode].width;
  const Int main_active_intersect_size = main_active_block->height;
  const Int descendant_main_intersect_size = scaled_transpose.width;
  const Int descendant_active_intersect_size = descendant_active_matrix.height;
  const bool inplace_update =
      main_active_intersect_size == descendant_active_intersect_size &&
      main_supernode_size == descendant_main_intersect_size;

  BlasMatrix<Field>* accumulation_matrix =
      inplace_update ? main_active_block : workspace_matrix;
  MatrixMultiplyNormalNormal(Field{-1}, descendant_active_matrix,
                             scaled_transpose, Field{1}, accumulation_matrix);

  if (!inplace_update) {
    const Int main_supernode_start = supernode_starts[main_supernode];

    const Int* main_indices = lower_factor.Structure(main_supernode);
    const Int* main_active_indices = main_indices + main_active_rel_row;

    const Int* descendant_indices =
        lower_factor.Structure(descendant_supernode);
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
void SeekForMainActiveRelativeRow(
    Int main_supernode, Int descendant_supernode, Int descendant_active_rel_row,
    const std::vector<Int>& supernode_member_to_index,
    const LowerFactor<Field>& lower_factor, Int* main_active_rel_row,
    const Int** main_active_intersect_sizes) {
  const Int* main_indices = lower_factor.Structure(main_supernode);
  const Int* descendant_indices = lower_factor.Structure(descendant_supernode);
  const Int descendant_active_supernode_start =
      descendant_indices[descendant_active_rel_row];
  const Int active_supernode =
      supernode_member_to_index[descendant_active_supernode_start];
  CATAMARI_ASSERT(active_supernode > main_supernode,
                  "Active supernode " + std::to_string(active_supernode) +
                      " was <= the main supernode " +
                      std::to_string(main_supernode));

  Int main_active_intersect_size = **main_active_intersect_sizes;
  Int main_active_first_row = main_indices[*main_active_rel_row];
  while (supernode_member_to_index[main_active_first_row] < active_supernode) {
    *main_active_rel_row += main_active_intersect_size;
    ++*main_active_intersect_sizes;

    main_active_first_row = main_indices[*main_active_rel_row];
    main_active_intersect_size = **main_active_intersect_sizes;
  }
#ifdef CATAMARI_DEBUG
  const Int main_active_supernode =
      supernode_member_to_index[main_active_first_row];
  CATAMARI_ASSERT(main_active_supernode == active_supernode,
                  "Did not find active supernode.");
#endif
}

template <class Field>
Int FactorDiagonalBlock(SymmetricFactorizationType factorization_type,
                        BlasMatrix<Field>* diagonal_block) {
  Int num_pivots;
  if (factorization_type == kCholeskyFactorization) {
    num_pivots = LowerCholeskyFactorization(diagonal_block);
  } else if (factorization_type == kLDLAdjointFactorization) {
    num_pivots = LowerLDLAdjointFactorization(diagonal_block);
  } else {
    num_pivots = LowerLDLTransposeFactorization(diagonal_block);
  }
  return num_pivots;
}

#ifdef _OPENMP
template <class Field>
Int MultithreadedFactorDiagonalBlock(
    Int tile_size, SymmetricFactorizationType factorization_type,
    BlasMatrix<Field>* diagonal_block, std::vector<Field>* buffer) {
  Int num_pivots;
  if (factorization_type == kCholeskyFactorization) {
    num_pivots =
        MultithreadedLowerCholeskyFactorization(tile_size, diagonal_block);
  } else if (factorization_type == kLDLAdjointFactorization) {
    num_pivots = MultithreadedLowerLDLAdjointFactorization(
        tile_size, diagonal_block, buffer);
  } else {
    num_pivots = MultithreadedLowerLDLTransposeFactorization(
        tile_size, diagonal_block, buffer);
  }
  return num_pivots;
}
#endif  // ifdef _OPENMP

template <class Field>
void SolveAgainstDiagonalBlock(SymmetricFactorizationType factorization_type,
                               const ConstBlasMatrix<Field>& triangular_matrix,
                               BlasMatrix<Field>* lower_matrix) {
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

#ifdef _OPENMP
template <class Field>
void MultithreadedSolveAgainstDiagonalBlock(
    Int tile_size, SymmetricFactorizationType factorization_type,
    const ConstBlasMatrix<Field>& triangular_matrix,
    BlasMatrix<Field>* lower_matrix) {
  if (!lower_matrix->height) {
    return;
  }

  const ConstBlasMatrix<Field> triangular_matrix_copy = triangular_matrix;
  BlasMatrix<Field> lower_matrix_copy = *lower_matrix;

  const Int height = lower_matrix->height;
  const Int width = lower_matrix->width;
  if (factorization_type == kCholeskyFactorization) {
    // Solve against the lower-triangular matrix L(K, K)' from the right.
    for (Int i = 0; i < height; i += tile_size) {
      #pragma omp task default(none) \
          firstprivate(i, height, width, tile_size, triangular_matrix_copy, \
              lower_matrix_copy)
      {
        const Int tsize = std::min(height - i, tile_size);
        BlasMatrix<Field> lower_block =
            lower_matrix_copy.Submatrix(i, 0, tsize, width);
        RightLowerAdjointTriangularSolves(triangular_matrix_copy, &lower_block);
      }
    }
  } else if (factorization_type == kLDLAdjointFactorization) {
    // Solve against D(K, K) L(K, K)' from the right.
    for (Int i = 0; i < height; i += tile_size) {
      #pragma omp task default(none) \
          firstprivate(i, height, width, tile_size, triangular_matrix_copy, \
          lower_matrix_copy)
      {
        const Int tsize = std::min(height - i, tile_size);
        BlasMatrix<Field> lower_block =
            lower_matrix_copy.Submatrix(i, 0, tsize, width);
        RightLowerAdjointUnitTriangularSolves(triangular_matrix_copy,
                                              &lower_block);
      }
    }
  } else {
    // Solve against D(K, K) L(K, K)^T from the right.
    for (Int i = 0; i < height; i += tile_size) {
      #pragma omp task default(none) \
          firstprivate(i, height, width, tile_size, triangular_matrix_copy, \
              lower_matrix_copy)
      {
        const Int tsize = std::min(height - i, tile_size);
        BlasMatrix<Field> lower_block =
            lower_matrix_copy.Submatrix(i, 0, tsize, width);
        RightDiagonalTimesLowerTransposeUnitTriangularSolves(
            triangular_matrix_copy, &lower_block);
      }
    }
  }
}
#endif  // ifdef _OPENMP

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_LDL_SUPERNODAL_LDL_SUPERNODE_UTILS_IMPL_H_
