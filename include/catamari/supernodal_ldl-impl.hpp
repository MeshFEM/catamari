/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SUPERNODAL_LDL_IMPL_H_
#define CATAMARI_SUPERNODAL_LDL_IMPL_H_

#include "catamari/blas.hpp"
#include "catamari/lapack.hpp"
#include "catamari/supernodal_ldl.hpp"

namespace catamari {

namespace supernodal_ldl {

// Fills 'member_to_index' with a length 'num_rows' array whose i'th index
// is the index of the supernode containing column 'i'.
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

// Builds an elimination forest over the supernodes from an elimination forest
// over the nodes.
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

// Initialize the last member of each supernode as unset, but other supernode
// members as having the member to their right as their parent.
inline void InitializeEliminationForest(
    const std::vector<Int>& supernode_starts,
    const std::vector<Int>& member_to_index, std::vector<Int>* parents) {
  const Int num_rows = member_to_index.size();
  parents->resize(num_rows, -1);
  for (Int column = 0; column < num_rows; ++column) {
    const Int supernode = member_to_index[column];
    if (column != supernode_starts[supernode + 1] - 1) {
      (*parents)[column] = column + 1;
    }
  }
}

// Computes the sizes of the structures of a supernodal LDL' factorization.
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

// Fills in the structure indices for the lower factor.
//
// The 'parents' array is in the original ordering of the matrix; each index
// points to the parent in the original elimination tree.
template <class Field>
void FillStructureIndices(const CoordinateMatrix<Field>& matrix,
                          const std::vector<Int>& parents,
                          SupernodalLDLFactorization<Field>* factorization) {
  const Int num_rows = matrix.NumRows();
  const Int num_supernodes = factorization->supernode_sizes.size();
  const std::vector<Int>& perm = factorization->permutation;
  const std::vector<Int>& inverse_perm = factorization->inverse_permutation;
  const bool have_permutation = !perm.empty();
  SupernodalLowerFactor<Field>& lower_factor = factorization->lower_factor;

  // A data structure for marking whether or not a node is in the pattern of
  // the active row of the lower-triangular factor.
  std::vector<Int> pattern_flags(num_rows);

  // A data structure for marking whether or not a supernode is in the pattern
  // of the active row of the lower-triangular factor.
  std::vector<Int> supernode_pattern_flags(num_supernodes);

  // A set of pointers for keeping track of where to insert supernode pattern
  // indices.
  std::vector<Int> supernode_ptrs(num_supernodes);
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    supernode_ptrs[supernode] = lower_factor.index_offsets[supernode];
  }

  // Fill in the structure indices.
  const std::vector<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int main_supernode = factorization->supernode_member_to_index[row];
    pattern_flags[row] = row;
    supernode_pattern_flags[main_supernode] = row;

    const Int orig_row = have_permutation ? inverse_perm[row] : row;
    const Int row_beg = matrix.RowEntryOffset(orig_row);
    const Int row_end = matrix.RowEntryOffset(orig_row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      Int descendant = have_permutation ? perm[entry.column] : entry.column;
      Int descendant_supernode =
          factorization->supernode_member_to_index[descendant];

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

        descendant_supernode =
            factorization->supernode_member_to_index[descendant];
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
                      lower_factor.index_offsets[descendant_supernode] &&
                  supernode_ptrs[descendant_supernode] <
                      lower_factor.index_offsets[descendant_supernode + 1],
              "Left supernode's indices.");
          lower_factor.indices[supernode_ptrs[descendant_supernode]++] = row;
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
    const Int index_beg = lower_factor.index_offsets[supernode];
    const Int index_end = lower_factor.index_offsets[supernode + 1];
    CATAMARI_ASSERT(supernode_ptrs[supernode] == index_end,
                    "Supernode pointers did not match index offsets.");
    bool sorted = true;
    Int last_row = -1;
    for (Int index = index_beg; index < index_end; ++index) {
      const Int row = lower_factor.indices[index];
      if (row <= last_row) {
        sorted = false;
        break;
      }
      last_row = row;
    }

    if (!sorted) {
      std::cerr << "Supernode " << supernode << " did not have sorted indices."
                << std::endl;
      for (Int index = index_beg; index < index_end; ++index) {
        const Int row = lower_factor.indices[index];
        std::cout << row << " ";
      }
      std::cout << std::endl;
    }
  }
#endif
}

// Fills in the sizes of the supernodal intersections.
template <class Field>
void FillIntersections(SupernodalLDLFactorization<Field>* factorization) {
  const Int num_supernodes = factorization->supernode_sizes.size();
  SupernodalLowerFactor<Field>& lower_factor = factorization->lower_factor;

  // Compute the supernode offsets.
  Int num_supernode_intersects = 0;
  lower_factor.intersect_size_offsets.resize(num_supernodes + 1);
  for (Int column_supernode = 0; column_supernode < num_supernodes;
       ++column_supernode) {
    lower_factor.intersect_size_offsets[column_supernode] =
        num_supernode_intersects;
    Int last_supernode = -1;

    const Int index_beg = lower_factor.index_offsets[column_supernode];
    const Int index_end = lower_factor.index_offsets[column_supernode + 1];
    for (Int index = index_beg; index < index_end; ++index) {
      const Int row = lower_factor.indices[index];
      const Int supernode = factorization->supernode_member_to_index[row];
      if (supernode != last_supernode) {
        last_supernode = supernode;
        ++num_supernode_intersects;
      }
    }
  }
  lower_factor.intersect_size_offsets[num_supernodes] =
      num_supernode_intersects;

  // Fill the supernode intersection sizes.
  lower_factor.intersect_sizes.resize(num_supernode_intersects);
  num_supernode_intersects = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    Int last_supernode = -1;
    Int intersect_size = 0;

    const Int index_beg = lower_factor.index_offsets[supernode];
    const Int index_end = lower_factor.index_offsets[supernode + 1];
    for (Int index = index_beg; index < index_end; ++index) {
      const Int row = lower_factor.indices[index];
      const Int row_supernode = factorization->supernode_member_to_index[row];
      if (row_supernode != last_supernode) {
        if (last_supernode != -1) {
          // Close out the supernodal intersection.
          lower_factor.intersect_sizes[num_supernode_intersects++] =
              intersect_size;
        }
        last_supernode = row_supernode;
        intersect_size = 0;
      }
      ++intersect_size;
    }
    if (last_supernode != -1) {
      // Close out the last intersection count for this column supernode.
      lower_factor.intersect_sizes[num_supernode_intersects++] = intersect_size;
    }
  }
  CATAMARI_ASSERT(num_supernode_intersects ==
                      static_cast<Int>(lower_factor.intersect_sizes.size()),
                  "Incorrect number of supernode intersections");
}

// Fill in the nonzeros from the original sparse matrix.
template <class Field>
void FillNonzeros(const CoordinateMatrix<Field>& matrix,
                  SupernodalLDLFactorization<Field>* factorization) {
  const Int num_rows = matrix.NumRows();
  const std::vector<Int>& perm = factorization->permutation;
  const std::vector<Int>& inverse_perm = factorization->inverse_permutation;
  const bool have_permutation = !perm.empty();
  SupernodalLowerFactor<Field>& lower_factor = factorization->lower_factor;
  SupernodalDiagonalFactor<Field>& diagonal_factor =
      factorization->diagonal_factor;

  const std::vector<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int supernode = factorization->supernode_member_to_index[row];
    const Int supernode_start = factorization->supernode_starts[supernode];
    const Int supernode_size = factorization->supernode_sizes[supernode];

    const Int orig_row = have_permutation ? inverse_perm[row] : row;
    const Int row_beg = matrix.RowEntryOffset(orig_row);
    const Int row_end = matrix.RowEntryOffset(orig_row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      const Int column = have_permutation ? perm[entry.column] : entry.column;
      const Int column_supernode =
          factorization->supernode_member_to_index[column];

      if (column_supernode == supernode) {
        // Insert the value into the diagonal block.
        const Int diag_supernode_start =
            diagonal_factor.value_offsets[supernode];
        Field* diag_block = &diagonal_factor.values[diag_supernode_start];

        const Int rel_row = row - supernode_start;
        const Int rel_column = column - supernode_start;
        const Int rel_index = rel_row + rel_column * supernode_size;
        diag_block[rel_index] = entry.value;
        continue;
      }

      if (column_supernode > supernode) {
        if (have_permutation) {
          continue;
        } else {
          break;
        }
      }

      // Compute the relative index of 'row' in this column supernode.
      Int* column_index_beg =
          &lower_factor.indices[lower_factor.index_offsets[column_supernode]];
      Int* column_index_end =
          &lower_factor
               .indices[lower_factor.index_offsets[column_supernode + 1]];
      Int* iter = std::lower_bound(column_index_beg, column_index_end, row);
      CATAMARI_ASSERT(iter != column_index_end, "Exceeded column indices.");
      CATAMARI_ASSERT(*iter == row, "Did not find index.");
      const Int rel_index_index = std::distance(column_index_beg, iter);
#ifdef CATAMARI_DEBUG
      if (*iter != row) {
        const Int column_supernode_start =
            factorization->supernode_starts[column_supernode];
        const Int column_supernode_size =
            factorization->supernode_sizes[column_supernode];
        std::cerr << "row=" << row << ", column=" << column
                  << ", column_supernode_start=" << column_supernode_start
                  << ", column_supernode_size=" << column_supernode_size
                  << ", rel_index_index=" << rel_index_index;
        if (rel_index_index > 0) {
          std::cerr << ", *(iter - 1)=" << *(iter - 1) << ", *iter=" << *iter
                    << ", *(iter + 1)=" << *(iter + 1) << std::endl;
        } else {
          std::cerr << ", *iter=" << *iter << std::endl;
        }
      }
#endif

      // Convert the relative row index into the corresponding value index
      // (for column 'column').
      const Int column_value_beg = lower_factor.value_offsets[column_supernode];
      const Int column_supernode_size =
          factorization->supernode_sizes[column_supernode];
      const Int column_supernode_start =
          factorization->supernode_starts[column_supernode];
      const Int rel_column = column - column_supernode_start;
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
    const std::vector<Int>& supernode_degrees,
    SupernodalLDLFactorization<Field>* factorization) {
  SupernodalLowerFactor<Field>& lower_factor = factorization->lower_factor;
  SupernodalDiagonalFactor<Field>& diagonal_factor =
      factorization->diagonal_factor;
  const Int num_supernodes = factorization->supernode_sizes.size();
  CATAMARI_ASSERT(static_cast<Int>(supernode_degrees.size()) == num_supernodes,
                  "Invalid supernode degrees size.");

  // Set up the column offsets and allocate space (initializing the values of
  // the unit-lower and diagonal and all zeros).
  lower_factor.index_offsets.resize(num_supernodes + 1);
  lower_factor.value_offsets.resize(num_supernodes + 1);
  diagonal_factor.value_offsets.resize(num_supernodes + 1);
  factorization->largest_degree = 0;
  Int degree_sum = 0;
  Int num_entries = 0;
  Int num_diagonal_entries = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    lower_factor.index_offsets[supernode] = degree_sum;
    lower_factor.value_offsets[supernode] = num_entries;
    diagonal_factor.value_offsets[supernode] = num_diagonal_entries;

    const Int degree = supernode_degrees[supernode];
    const Int supernode_size = factorization->supernode_sizes[supernode];
    degree_sum += degree;
    num_entries += degree * supernode_size;
    num_diagonal_entries += supernode_size * supernode_size;
    factorization->largest_degree =
        std::max(factorization->largest_degree, degree);
  }
  lower_factor.index_offsets[num_supernodes] = degree_sum;
  lower_factor.value_offsets[num_supernodes] = num_entries;
  diagonal_factor.value_offsets[num_supernodes] = num_diagonal_entries;

  lower_factor.indices.resize(degree_sum);
  lower_factor.values.resize(num_entries, Field{0});
  diagonal_factor.values.resize(num_diagonal_entries, Field{0});

  FillStructureIndices(matrix, parents, factorization);
  FillNonzeros(matrix, factorization);
  FillIntersections(factorization);
}

// Computes the supernodal nonzero pattern of L(row, :) in
// row_structure[0 : num_packed - 1].
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

// Store the scaled adjoint update matrix, Z(d, m) = D(d, d) L(m, d)'
// contiguously in column-major form with minimal leading dimension.
//
// The matrix will be packed into the first
// descendant_supernode_size x descendant_main_intersect_size entries.
template <class Field>
void FormScaledAdjoint(const SupernodalLDLFactorization<Field>& factorization,
                       Int descendant_supernode,
                       Int descendant_main_intersect_size,
                       Int descendant_main_value_beg,
                       Field* scaled_adjoint_buffer) {
  const SupernodalLowerFactor<Field>& lower_factor = factorization.lower_factor;
  const SupernodalDiagonalFactor<Field>& diagonal_factor =
      factorization.diagonal_factor;
  const Int descendant_supernode_size =
      factorization.supernode_sizes[descendant_supernode];
  const bool is_cholesky = factorization.is_cholesky;

  const Field* descendant_main_block =
      &lower_factor.values[descendant_main_value_beg];
  if (is_cholesky) {
    for (Int i = 0; i < descendant_supernode_size; ++i) {
      for (Int j_rel = 0; j_rel < descendant_main_intersect_size; ++j_rel) {
        // Recall that lower_insersect_block is row-major.
        const Field& lambda_right_conj = Conjugate(
            descendant_main_block[i + j_rel * descendant_supernode_size]);
        scaled_adjoint_buffer[i + j_rel * descendant_supernode_size] =
            lambda_right_conj;
      }
    }
  } else {
    const Field* descendant_diag_block =
        &diagonal_factor
             .values[diagonal_factor.value_offsets[descendant_supernode]];
    for (Int i = 0; i < descendant_supernode_size; ++i) {
      const Field& delta =
          descendant_diag_block[i + i * descendant_supernode_size];
      for (Int j_rel = 0; j_rel < descendant_main_intersect_size; ++j_rel) {
        // Recall that lower_insersect_block is row-major.
        const Field& lambda_right_conj = Conjugate(
            descendant_main_block[i + j_rel * descendant_supernode_size]);
        scaled_adjoint_buffer[i + j_rel * descendant_supernode_size] =
            delta * lambda_right_conj;
      }
    }
  }
}

// L(m, m) -= L(m, d) * (D(d, d) * L(m, d)')
//          = L(m, d) * Z(:, m).
//
// The update is to the main_supernode_size x main_supernode_size dense
// diagonal block, L(m, m), with the densified L(m, d) matrix being
// descendant_main_intersect_size x descendant_supernode_size, and Z(:, m)
// being descendant_supernode_size x descendant_main_intersect_size.
template <class Field>
void UpdateDiagonalBlock(Int main_supernode, Int descendant_supernode,
                         Int descendant_main_intersect_size,
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
      descendant_main_intersect_size == main_supernode_size;

  const Field* descendant_main_block =
      &lower_factor.values[descendant_main_value_beg];

  Field* main_diag_block =
      &diagonal_factor.values[diagonal_factor.value_offsets[main_supernode]];
  Field* update_block = inplace_update ? main_diag_block : update_buffer;

  ConstBlasMatrix<Field> descendant_main_matrix;
  descendant_main_matrix.height = descendant_supernode_size;
  descendant_main_matrix.width = descendant_main_intersect_size;
  descendant_main_matrix.leading_dim = descendant_supernode_size;
  descendant_main_matrix.data = descendant_main_block;

  ConstBlasMatrix<Field> scaled_adjoint;
  scaled_adjoint.height = descendant_supernode_size;
  scaled_adjoint.width = descendant_main_intersect_size;
  scaled_adjoint.leading_dim = descendant_supernode_size;
  scaled_adjoint.data = scaled_adjoint_buffer;

  BlasMatrix<Field> update_matrix;
  update_matrix.height = descendant_main_intersect_size;
  update_matrix.width = descendant_main_intersect_size;
  update_matrix.leading_dim = descendant_main_intersect_size;
  update_matrix.data = update_block;

  if (factorization->is_cholesky) {
    LowerTransposeHermitianOuterProduct(Field{-1}, descendant_main_matrix,
                                        Field{1}, &update_matrix);
  } else {
    MatrixMultiplyLowerTransposeNormal(Field{-1}, descendant_main_matrix,
                                       scaled_adjoint, Field{1},
                                       &update_matrix);
  }

  if (!inplace_update) {
    // Apply the out-of-place update and zero the buffer.
    const Int main_supernode_start =
        factorization->supernode_starts[main_supernode];
    const Int* descendant_main_indices =
        &lower_factor.indices[descendant_main_index_beg];
    for (Int j = 0; j < descendant_main_intersect_size; ++j) {
      const Int column = descendant_main_indices[j];
      const Int j_rel = column - main_supernode_start;
      Field* main_diag_col = &main_diag_block[j_rel * main_supernode_size];
      Field* update_col = &update_buffer[j * descendant_main_intersect_size];

      for (Int i = j; i < descendant_main_intersect_size; ++i) {
        const Int row = descendant_main_indices[i];
        const Int i_rel = row - main_supernode_start;
        main_diag_col[i_rel] += update_col[i];
        update_col[i] = 0;
      }
    }
  }
}

// L(a, m) -= L(a, d) * (D(d, d) * L(m, d)')
//          = L(a, d) * Z(:, m).
//
// The update is to the main_active_intersect_size x main_supernode_size
// densified matrix L(a, m) using the
// descendant_active_intersect_size x descendant_supernode_size densified
// L(a, d) and thd escendant_supernode_size x descendant_main_intersect_size
// Z(:, m).
template <class Field>
void UpdateSubdiagonalBlock(
    Int main_supernode, Int descendant_supernode,
    Int descendant_main_intersect_size, Int descendant_main_index_beg,
    Int descendant_active_intersect_size, Int descendant_active_index_beg,
    Int descendant_active_value_beg, const Field* scaled_adjoint_buffer,
    SupernodalLDLFactorization<Field>* factorization,
    Int* main_active_intersect_size_beg, Int* main_active_index_beg,
    Int* main_active_value_beg, Field* update_buffer) {
  SupernodalLowerFactor<Field>& lower_factor = factorization->lower_factor;

  const Int main_supernode_size =
      factorization->supernode_sizes[main_supernode];
  const Int descendant_supernode_size =
      factorization->supernode_sizes[descendant_supernode];

  // Move the pointers for the main supernode down to the active supernode of
  // the descendant column block.
  const Int descendant_active_supernode_start =
      lower_factor.indices[descendant_active_index_beg];
  const Int active_supernode =
      factorization
          ->supernode_member_to_index[descendant_active_supernode_start];
  CATAMARI_ASSERT(active_supernode > main_supernode,
                  "Active supernode was <= the main supernode in update.");

  Int main_active_intersect_size =
      lower_factor.intersect_sizes[*main_active_intersect_size_beg];
  while (factorization->supernode_member_to_index
             [lower_factor.indices[*main_active_index_beg]] <
         active_supernode) {
    *main_active_index_beg += main_active_intersect_size;
    *main_active_value_beg += main_active_intersect_size * main_supernode_size;
    ++*main_active_intersect_size_beg;

    main_active_intersect_size =
        lower_factor.intersect_sizes[*main_active_intersect_size_beg];
  }
#ifdef CATAMARI_DEBUG
  const Int main_active_supernode_start =
      lower_factor.indices[*main_active_index_beg];
  const Int main_active_supernode =
      factorization->supernode_member_to_index[main_active_supernode_start];
  if (main_active_supernode != active_supernode) {
    std::cerr << "main_supernode=" << main_supernode
              << ", descendant_supernode=" << descendant_supernode
              << ", active_supernode=" << active_supernode
              << ", main_active_supernode=" << main_active_supernode
              << std::endl;
  }
  CATAMARI_ASSERT(main_active_supernode == active_supernode,
                  "Did not find active supernode.");
#endif

  const bool inplace_update =
      main_active_intersect_size == descendant_active_intersect_size &&
      main_supernode_size == descendant_main_intersect_size;

  Field* main_active_block = &lower_factor.values[*main_active_value_beg];

  ConstBlasMatrix<Field> scaled_adjoint;
  scaled_adjoint.height = descendant_supernode_size;
  scaled_adjoint.width = descendant_main_intersect_size;
  scaled_adjoint.leading_dim = descendant_supernode_size;
  scaled_adjoint.data = scaled_adjoint_buffer;

  ConstBlasMatrix<Field> descendant_active_matrix;
  descendant_active_matrix.height = descendant_supernode_size;
  descendant_active_matrix.width = descendant_active_intersect_size;
  descendant_active_matrix.leading_dim = descendant_supernode_size;
  descendant_active_matrix.data =
      &lower_factor.values[descendant_active_value_beg];

  BlasMatrix<Field> update_matrix;
  update_matrix.height = descendant_main_intersect_size;
  update_matrix.width = descendant_active_intersect_size;
  update_matrix.leading_dim = descendant_main_intersect_size;
  update_matrix.data = inplace_update ? main_active_block : update_buffer;

  MatrixMultiplyTransposeNormal(Field{-1}, scaled_adjoint,
                                descendant_active_matrix, Field{1},
                                &update_matrix);

  if (!inplace_update) {
    Int main_active_index_offset = 0;
    const Int* main_active_indices =
        &lower_factor.indices[*main_active_index_beg];
    const Int main_supernode_start =
        factorization->supernode_starts[main_supernode];
    const Int* descendant_main_indices =
        &lower_factor.indices[descendant_main_index_beg];
    const Int* descendant_active_indices =
        &lower_factor.indices[descendant_active_index_beg];
    for (Int i = 0; i < descendant_active_intersect_size; ++i) {
      const Int row = descendant_active_indices[i];

      // Both the main and descendant supernodal intersections can be sparse,
      // and different. Thus, we must scan through the intersection indices to
      // find the appropriate relative index here.
      //
      // Scan downwards in the main active indices until we find the equivalent
      // row from the descendant active indices.
      while (main_active_indices[main_active_index_offset] != row) {
        ++main_active_index_offset;
      }

      Field* main_active_row =
          &main_active_block[main_active_index_offset * main_supernode_size];
      Field* update_row = &update_buffer[i * descendant_main_intersect_size];

      for (Int j = 0; j < descendant_main_intersect_size; ++j) {
        const Int column = descendant_main_indices[j];
        const Int j_rel = column - main_supernode_start;
        Field& main_active_entry = main_active_row[j_rel];
        Field& update_entry = update_row[j];

        main_active_entry += update_entry;
        update_entry = 0;
      }
    }
  }
}

// Perform an in-place LDL' factorization of the supernodal diagonal block.
template <class Field>
Int FactorDiagonalBlock(Int supernode,
                        SupernodalLDLFactorization<Field>* factorization) {
  const Int supernode_size = factorization->supernode_sizes[supernode];

  BlasMatrix<Field> diagonal_block;
  diagonal_block.height = supernode_size;
  diagonal_block.width = supernode_size;
  diagonal_block.leading_dim = supernode_size;
  diagonal_block.data =
      &factorization->diagonal_factor
           .values[factorization->diagonal_factor.value_offsets[supernode]];

  Int num_pivots;
  if (factorization->is_cholesky) {
    num_pivots = LowerCholeskyFactorization(&diagonal_block);
  } else {
    num_pivots = LowerLDLAdjointFactorization(&diagonal_block);
  }

  return num_pivots;
}

// L(KNext:n, K) /= D(K, K) L(K, K)'.
template <class Field>
void SolveAgainstDiagonalBlock(
    Int supernode, SupernodalLDLFactorization<Field>* factorization) {
  SupernodalLowerFactor<Field>& lower_factor = factorization->lower_factor;
  const SupernodalDiagonalFactor<Field>& diagonal_factor =
      factorization->diagonal_factor;
  const Int supernode_size = factorization->supernode_sizes[supernode];

  const Int index_beg = lower_factor.index_offsets[supernode];
  const Int value_beg = lower_factor.value_offsets[supernode];
  const Int degree = lower_factor.index_offsets[supernode + 1] - index_beg;

  ConstBlasMatrix<Field> triangular_matrix;
  triangular_matrix.height = supernode_size;
  triangular_matrix.width = supernode_size;
  triangular_matrix.leading_dim = supernode_size;
  triangular_matrix.data =
      &diagonal_factor.values[diagonal_factor.value_offsets[supernode]];

  BlasMatrix<Field> lower_matrix;
  lower_matrix.height = supernode_size;
  lower_matrix.width = degree;
  lower_matrix.leading_dim = supernode_size;
  lower_matrix.data = &lower_factor.values[value_beg];

  if (factorization->is_cholesky) {
    // Solve against the lower-triangular matrix L(K, K)' from the right using
    // row-major storage of the input matrix, which is equivalent to solving
    // conj(L(K, K)) X = Y using column-major storage.
    LeftLowerConjugateTriangularSolves(triangular_matrix, &lower_matrix);
  } else {
    // Solve against the D(K, K) times the unit-lower matrix L(K, K)' from the
    // right using row-major storage of the input matrix, which is equivalent to
    // solving D(K, K) conj(L(K, K)) X = Y using column-major storage.
    DiagonalTimesLeftLowerConjugateUnitTriangularSolves(triangular_matrix,
                                                        &lower_matrix);
  }
}

// Checks that a valid set of supernodes has been provided by explicitly
// computing each row pattern and ensuring that each intersects entire
// supernodes.
template <class Field>
bool ValidFundamentalSupernodes(const CoordinateMatrix<Field>& matrix,
                                const std::vector<Int>& permutation,
                                const std::vector<Int>& inverse_permutation,
                                const std::vector<Int>& supernode_sizes) {
  const Int num_rows = matrix.NumRows();

  std::vector<Int> parents, degrees;
  ldl::EliminationForestAndDegrees(matrix, permutation, inverse_permutation,
                                   &parents, &degrees);

  std::vector<Int> supernode_starts, member_to_index;
  ldl::OffsetScan(supernode_sizes, &supernode_starts);
  MemberToIndex(num_rows, supernode_starts, &member_to_index);

  std::vector<Int> row_structure(num_rows);
  std::vector<Int> pattern_flags(num_rows);

  bool valid = true;
  for (Int row = 0; row < num_rows; ++row) {
    const Int row_supernode = member_to_index[row];

    pattern_flags[row] = row;
    const Int num_packed = ldl::ComputeRowPattern(
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

// Compute an unrelaxed supernodal partition using the existing ordering.
// We require that supernodes have dense diagonal blocks and equal structures
// below the diagonal block.
template <class Field>
void FormFundamentalSupernodes(const CoordinateMatrix<Field>& matrix,
                               const std::vector<Int>& permutation,
                               const std::vector<Int>& inverse_permutation,
                               const std::vector<Int>& parents,
                               const std::vector<Int>& degrees,
                               std::vector<Int>* supernode_sizes,
                               ScalarLowerStructure* scalar_structure) {
  const Int num_rows = matrix.NumRows();

  // We will only fill the indices and offsets of the factorization.
  ldl::FillStructureIndices(matrix, permutation, inverse_permutation, parents,
                            degrees, scalar_structure);

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

// A data structure for representing whether or not a child supernode can be
// merged with its parent and, if so, how many explicit zeros would be in the
// combined supernode's block column.
struct MergableStatus {
  // Whether or not the supernode can be merged.
  bool mergable;

  // How many explicit zeros would be stored in the merged supernode.
  Int num_merged_zeros;
};

// Returns whether or not the child supernode can be merged into its parent by
// counting the number of explicit zeros that would be introduced by the
// merge.
//
// Consider the possibility of merging a child supernode '0' with parent
// supernode '1', which would result in an expanded supernode of the form:
//
//    -------------
//   | L00  |      |
//   |------|------|
//   | L10  | L11  |
//   ---------------
//   |      |      |
//   | L20  | L21  |
//   |      |      |
//    -------------
//
// Because it is assumed that supernode 1 is the parent of supernode 0,
// the only places explicit nonzeros can be introduced are in:
//   * L10: if supernode 0 was not fully-connected to supernode 1.
//   * L20: if supernode 0's structure (with supernode 1 removed) does not
//     contain every member of the structure of supernode 1.
//
// Counting the number of explicit zeros that would be introduced is thus
// simply a matter of counting these mismatches.
//
// The reason that downstream explicit zeros would not be introduced is the
// same as the reason explicit zeros are not introduced into L21; supernode
// 1 is the parent of supernode 0.
//
inline MergableStatus MergableSupernode(
    Int child_tail, Int parent_tail, Int child_size, Int parent_size,
    Int num_child_explicit_zeros, Int num_parent_explicit_zeros,
    const std::vector<Int>& orig_member_to_index,
    const ScalarLowerStructure& scalar_structure,
    const SupernodalLDLControl& control) {
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
    const ScalarLowerStructure& scalar_structure,
    const SupernodalLDLControl& control, std::vector<Int>* supernode_sizes,
    std::vector<Int>* num_explicit_zeros, std::vector<Int>* last_merged_child,
    std::vector<Int>* merge_parents) {
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

// Builds a packed set of child links for an elimination forest given the
// parent links (with root nodes having their parent set to -1).
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

  ldl::OffsetScan(num_children, child_offsets);

  children->resize(num_indices);
  auto offsets_copy = *child_offsets;
  for (Int index = 0; index < num_indices; ++index) {
    const Int parent = parents[index];
    if (parent >= 0) {
      (*children)[offsets_copy[parent]++] = index;
    }
  }
}

// Walk up the tree in the original postordering, merging supernodes as we
// progress.
template <class Field>
void RelaxSupernodes(const std::vector<Int>& orig_parents,
                     const std::vector<Int>& orig_supernode_sizes,
                     const std::vector<Int>& orig_supernode_starts,
                     const std::vector<Int>& orig_supernode_parents,
                     const std::vector<Int>& orig_supernode_degrees,
                     const std::vector<Int>& orig_member_to_index,
                     const ScalarLowerStructure& scalar_structure,
                     const SupernodalLDLControl& control,
                     std::vector<Int>* relaxed_parents,
                     std::vector<Int>* relaxed_supernode_parents,
                     std::vector<Int>* relaxed_supernode_degrees,
                     SupernodalLDLFactorization<Field>* factorization) {
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
    factorization->supernode_sizes.resize(num_relaxed_supernodes);
    factorization->supernode_starts.resize(num_relaxed_supernodes + 1);
    factorization->supernode_member_to_index.resize(num_rows);
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
      factorization->supernode_starts[relaxed_supernode] = pack_offset;
      Int supernode_size = 0;
      Int supernode_to_pack = leaf_of_merge;
      while (true) {
        const Int start = orig_supernode_starts[supernode_to_pack];
        const Int size = orig_supernode_sizes[supernode_to_pack];
        for (Int j = 0; j < size; ++j) {
          factorization->supernode_member_to_index[pack_offset] =
              relaxed_supernode;
          relaxation_inverse_permutation[pack_offset++] = start + j;
        }
        supernode_size += size;

        if (merge_parents[supernode_to_pack] == -1) {
          break;
        }
        supernode_to_pack = merge_parents[supernode_to_pack];
      }
      factorization->supernode_sizes[relaxed_supernode] = supernode_size;
    }
    CATAMARI_ASSERT(num_rows == pack_offset, "Did not pack num_rows indices.");
    factorization->supernode_starts[num_relaxed_supernodes] = num_rows;
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
  if (factorization->permutation.empty()) {
    factorization->permutation = relaxation_permutation;
    factorization->inverse_permutation = relaxation_inverse_permutation;
  } else {
    std::vector<Int> perm_copy = factorization->permutation;
    for (Int row = 0; row < num_rows; ++row) {
      factorization->permutation[row] = relaxation_permutation[perm_copy[row]];
    }

    // Invert the composed permutation.
    for (Int row = 0; row < num_rows; ++row) {
      factorization->inverse_permutation[factorization->permutation[row]] = row;
    }
  }
}

// Form the (possibly relaxed) supernodes for the factorization.
template <class Field>
void FormSupernodes(const CoordinateMatrix<Field>& matrix,
                    const SupernodalLDLControl& control,
                    std::vector<Int>* parents,
                    std::vector<Int>* supernode_degrees,
                    std::vector<Int>* supernode_parents,
                    SupernodalLDLFactorization<Field>* factorization) {
  // Compute the non-supernodal elimination tree using the original ordering.
  std::vector<Int> orig_parents, orig_degrees;
  ldl::EliminationForestAndDegrees(matrix, factorization->permutation,
                                   factorization->inverse_permutation,
                                   &orig_parents, &orig_degrees);

  // Greedily compute a supernodal partition using the original ordering.
  std::vector<Int> orig_supernode_sizes;
  ScalarLowerStructure scalar_structure;
  FormFundamentalSupernodes(
      matrix, factorization->permutation, factorization->inverse_permutation,
      orig_parents, orig_degrees, &orig_supernode_sizes, &scalar_structure);

#ifdef CATAMARI_DEBUG
  {
    Int supernode_size_sum = 0;
    for (const Int& supernode_size : orig_supernode_sizes) {
      supernode_size_sum += supernode_size;
    }
    CATAMARI_ASSERT(supernode_size_sum == matrix.NumRows(),
                    "Supernodes did not sum to the matrix size.");
  }
#endif

  std::vector<Int> orig_supernode_starts;
  ldl::OffsetScan(orig_supernode_sizes, &orig_supernode_starts);

  std::vector<Int> orig_member_to_index;
  MemberToIndex(matrix.NumRows(), orig_supernode_starts, &orig_member_to_index);

  std::vector<Int> orig_supernode_degrees;
  SupernodalDegrees(matrix, factorization->permutation,
                    factorization->inverse_permutation, orig_supernode_sizes,
                    orig_supernode_starts, orig_member_to_index, orig_parents,
                    &orig_supernode_degrees);

  const Int num_orig_supernodes = orig_supernode_sizes.size();
  std::vector<Int> orig_supernode_parents;
  ConvertFromScalarToSupernodalEliminationForest(
      num_orig_supernodes, orig_parents, orig_member_to_index,
      &orig_supernode_parents);

  if (control.relax_supernodes) {
    RelaxSupernodes(orig_parents, orig_supernode_sizes, orig_supernode_starts,
                    orig_supernode_parents, orig_supernode_degrees,
                    orig_member_to_index, scalar_structure, control, parents,
                    supernode_parents, supernode_degrees, factorization);
  } else {
    *parents = orig_parents;
    *supernode_degrees = orig_supernode_degrees;
    *supernode_parents = orig_supernode_parents;
    factorization->supernode_sizes = orig_supernode_sizes;
    factorization->supernode_starts = orig_supernode_starts;
    factorization->supernode_member_to_index = orig_member_to_index;
  }
}

template <class Field>
LDLResult LeftLooking(const CoordinateMatrix<Field>& matrix,
                      const SupernodalLDLControl& control,
                      SupernodalLDLFactorization<Field>* factorization) {
  std::vector<Int> parents;
  std::vector<Int> supernode_degrees;
  std::vector<Int> supernode_parents;
  FormSupernodes(matrix, control, &parents, &supernode_degrees,
                 &supernode_parents, factorization);

  const Int num_supernodes = factorization->supernode_sizes.size();
  SupernodalLowerFactor<Field>& lower_factor = factorization->lower_factor;

  InitializeLeftLookingFactors(matrix, parents, supernode_degrees,
                               factorization);

  // Set up a buffer for supernodal updates.
  Int max_supernode_size = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    max_supernode_size =
        std::max(max_supernode_size, factorization->supernode_sizes[supernode]);
  }
  std::vector<Field> scaled_adjoint_buffer(
      max_supernode_size * max_supernode_size, Field{0});
  std::vector<Field> update_buffer(
      max_supernode_size * (max_supernode_size - 1), Field{0});

  // A data structure for marking whether or not a supernode is in the pattern
  // of the active row of the lower-triangular factor.
  std::vector<Int> pattern_flags(num_supernodes);

  // An integer workspace for storing the supernodes in the current row
  // pattern.
  std::vector<Int> row_structure(num_supernodes);

  // Since we will sequentially access each of the entries in each block column
  // of  L during the updates of a supernode, we can avoid the need for binary
  // search by maintaining a separate counter for each supernode.
  std::vector<Int> intersect_ptrs(num_supernodes);
  std::vector<Int> index_ptrs(num_supernodes);
  std::vector<Int> value_ptrs(num_supernodes);

  LDLResult result;
  for (Int main_supernode = 0; main_supernode < num_supernodes;
       ++main_supernode) {
    const Int main_supernode_size =
        factorization->supernode_sizes[main_supernode];

    pattern_flags[main_supernode] = main_supernode;

    intersect_ptrs[main_supernode] =
        lower_factor.intersect_size_offsets[main_supernode];
    index_ptrs[main_supernode] = lower_factor.index_offsets[main_supernode];
    value_ptrs[main_supernode] = lower_factor.value_offsets[main_supernode];

    // Compute the supernodal row pattern.
    const Int num_packed = ComputeRowPattern(
        matrix, factorization->permutation, factorization->inverse_permutation,
        factorization->supernode_sizes, factorization->supernode_starts,
        factorization->supernode_member_to_index, supernode_parents,
        main_supernode, pattern_flags.data(), row_structure.data());

    // for J = find(L(K, :))
    //   L(K:n, K) -= L(K:n, J) * (D(J, J) * L(K, J)')
    for (Int index = 0; index < num_packed; ++index) {
      const Int descendant_supernode = row_structure[index];
      CATAMARI_ASSERT(descendant_supernode < main_supernode,
                      "Looking into upper triangle.");
      const Int descendant_supernode_size =
          factorization->supernode_sizes[descendant_supernode];

      const Int descendant_main_intersect_size_beg =
          intersect_ptrs[descendant_supernode];
      const Int descendant_main_index_beg = index_ptrs[descendant_supernode];
      const Int descendant_main_value_beg = value_ptrs[descendant_supernode];
      const Int descendant_main_intersect_size =
          lower_factor.intersect_sizes[descendant_main_intersect_size_beg];

      FormScaledAdjoint(
          *factorization, descendant_supernode, descendant_main_intersect_size,
          descendant_main_value_beg, scaled_adjoint_buffer.data());

      UpdateDiagonalBlock(
          main_supernode, descendant_supernode, descendant_main_intersect_size,
          descendant_main_index_beg, descendant_main_value_beg,
          scaled_adjoint_buffer.data(), factorization, update_buffer.data());

      intersect_ptrs[descendant_supernode]++;
      index_ptrs[descendant_supernode] += descendant_main_intersect_size;
      value_ptrs[descendant_supernode] +=
          descendant_main_intersect_size * descendant_supernode_size;

      // L(KNext:n, K) -= L(KNext:n, J) * (D(J, J) * L(K, J)')
      //                = L(KNext:n, J) * Z(J, K).
      Int descendant_active_intersect_size_beg =
          intersect_ptrs[descendant_supernode];
      Int descendant_active_index_beg = index_ptrs[descendant_supernode];
      Int descendant_active_value_beg = value_ptrs[descendant_supernode];
      Int main_active_intersect_size_beg =
          lower_factor.intersect_size_offsets[main_supernode];
      Int main_active_index_beg = lower_factor.index_offsets[main_supernode];
      Int main_active_value_beg = lower_factor.value_offsets[main_supernode];
      const Int descendant_index_guard =
          lower_factor.index_offsets[descendant_supernode + 1];
      while (descendant_active_index_beg != descendant_index_guard) {
        const Int descendant_active_intersect_size =
            lower_factor.intersect_sizes[descendant_active_intersect_size_beg];
        UpdateSubdiagonalBlock(
            main_supernode, descendant_supernode,
            descendant_main_intersect_size, descendant_main_index_beg,
            descendant_active_intersect_size, descendant_active_index_beg,
            descendant_active_value_beg, scaled_adjoint_buffer.data(),
            factorization, &main_active_intersect_size_beg,
            &main_active_index_beg, &main_active_value_beg,
            update_buffer.data());

        ++descendant_active_intersect_size_beg;
        descendant_active_index_beg += descendant_active_intersect_size;
        descendant_active_value_beg +=
            descendant_active_intersect_size * descendant_supernode_size;
      }
    }

    const Int num_supernode_pivots =
        FactorDiagonalBlock(main_supernode, factorization);
    result.num_successful_pivots += num_supernode_pivots;
    if (num_supernode_pivots < main_supernode_size) {
      return result;
    }

    SolveAgainstDiagonalBlock(main_supernode, factorization);

    // Finish updating the result structure.
    const Int degree = lower_factor.index_offsets[main_supernode + 1] -
                       lower_factor.index_offsets[main_supernode];
    result.largest_supernode =
        std::max(result.largest_supernode, main_supernode_size);
    result.num_factorization_entries +=
        (main_supernode_size * (main_supernode_size + 1)) / 2 +
        main_supernode_size * degree;
    result.num_factorization_flops +=
        std::pow(1. * main_supernode_size, 3.) / 3. +
        std::pow(1. * degree, 2.) * main_supernode_size;
  }

  return result;
}

}  // namespace supernodal_ldl

template <class Field>
LDLResult LDL(const CoordinateMatrix<Field>& matrix,
              const std::vector<Int>& permutation,
              const std::vector<Int>& inverse_permutation,
              const SupernodalLDLControl& control,
              SupernodalLDLFactorization<Field>* factorization) {
  factorization->permutation = permutation;
  factorization->inverse_permutation = inverse_permutation;
  factorization->is_cholesky = control.use_cholesky;
  factorization->forward_solve_out_of_place_supernode_threshold =
      control.forward_solve_out_of_place_supernode_threshold;
  factorization->backward_solve_out_of_place_supernode_threshold =
      control.backward_solve_out_of_place_supernode_threshold;
  return supernodal_ldl::LeftLooking(matrix, control, factorization);
}

template <class Field>
LDLResult LDL(const CoordinateMatrix<Field>& matrix,
              const SupernodalLDLControl& control,
              SupernodalLDLFactorization<Field>* factorization) {
  std::vector<Int> permutation, inverse_permutation;
  return LDL(matrix, permutation, inverse_permutation, control, factorization);
}

template <class Field>
void LDLSolve(const SupernodalLDLFactorization<Field>& factorization,
              std::vector<Field>* vector) {
  const bool have_permutation = !factorization.permutation.empty();

  // Reorder the input into the permutation of the factorization.
  if (have_permutation) {
    Permute(factorization.permutation, vector);
  }

  LowerTriangularSolve(factorization, vector);
  DiagonalSolve(factorization, vector);
  LowerAdjointTriangularSolve(factorization, vector);

  // Reverse the factorization permutation.
  if (have_permutation) {
    Permute(factorization.inverse_permutation, vector);
  }
}

template <class Field>
void LowerTriangularSolve(
    const SupernodalLDLFactorization<Field>& factorization,
    std::vector<Field>* vector) {
  const Int num_supernodes = factorization.supernode_sizes.size();
  const SupernodalLowerFactor<Field>& lower_factor = factorization.lower_factor;
  const SupernodalDiagonalFactor<Field>& diagonal_factor =
      factorization.diagonal_factor;
  const bool is_cholesky = factorization.is_cholesky;

  std::vector<Field> workspace(factorization.largest_degree, Field{0});

  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int supernode_size = factorization.supernode_sizes[supernode];
    const Int supernode_start = factorization.supernode_starts[supernode];

    const Int index_beg = lower_factor.index_offsets[supernode];
    const Int degree = lower_factor.index_offsets[supernode + 1] - index_beg;
    const Int value_beg = lower_factor.value_offsets[supernode];

    ConstBlasMatrix<Field> triangular_matrix;
    triangular_matrix.height = supernode_size;
    triangular_matrix.width = supernode_size;
    triangular_matrix.leading_dim = supernode_size;
    triangular_matrix.data =
        &diagonal_factor.values[diagonal_factor.value_offsets[supernode]];

    // Solve against the diagonal block of the supernode.
    if (is_cholesky) {
      TriangularSolveLeftLower(triangular_matrix, &(*vector)[supernode_start]);
    } else {
      TriangularSolveLeftLowerUnit(triangular_matrix,
                                   &(*vector)[supernode_start]);
    }

    // Handle the external updates for this supernode.
    if (supernode_size >=
        factorization.forward_solve_out_of_place_supernode_threshold) {
      // Perform an out-of-place GEMV.
      ConstBlasMatrix<Field> subdiagonal;
      subdiagonal.height = supernode_size;
      subdiagonal.width = degree;
      subdiagonal.leading_dim = supernode_size;
      subdiagonal.data = &lower_factor.values[value_beg];

      TransposeMatrixVectorProduct(
          Field{-1}, subdiagonal, vector->data() + supernode_start,
          workspace.data());

      for (Int i = 0; i < degree; ++i) {
        const Int row = lower_factor.indices[index_beg + i];
        (*vector)[row] += workspace[i];
        workspace[i] = 0;
      }
    } else {
      for (Int j = 0; j < supernode_size; ++j) {
        const Int column = supernode_start + j;
        const Field& eta = (*vector)[column];
  
        for (Int i = 0; i < degree; ++i) {
          const Int row = lower_factor.indices[index_beg + i];
          const Field& value =
              lower_factor.values[value_beg + i * supernode_size + j];
          (*vector)[row] -= value * eta;
        }
      }
    }
  }
}

template <class Field>
void DiagonalSolve(const SupernodalLDLFactorization<Field>& factorization,
                   std::vector<Field>* vector) {
  const Int num_supernodes = factorization.supernode_sizes.size();
  const SupernodalDiagonalFactor<Field>& diagonal_factor =
      factorization.diagonal_factor;
  const bool is_cholesky = factorization.is_cholesky;
  if (is_cholesky) {
    // D is the identity.
    return;
  }

  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int supernode_size = factorization.supernode_sizes[supernode];
    const Int supernode_start = factorization.supernode_starts[supernode];

    // Handle the diagonal-block portion of the supernode.
    const Int diag_value_beg = diagonal_factor.value_offsets[supernode];
    const Field* diag_block = &diagonal_factor.values[diag_value_beg];
    for (Int j = 0; j < supernode_size; ++j) {
      const Int column = supernode_start + j;
      const Field& value = diag_block[j + j * supernode_size];
      (*vector)[column] /= value;
    }
  }
}

template <class Field>
void LowerAdjointTriangularSolve(
    const SupernodalLDLFactorization<Field>& factorization,
    std::vector<Field>* vector) {
  const Int num_supernodes = factorization.supernode_sizes.size();
  const SupernodalLowerFactor<Field>& lower_factor = factorization.lower_factor;
  const SupernodalDiagonalFactor<Field>& diagonal_factor =
      factorization.diagonal_factor;
  const bool is_cholesky = factorization.is_cholesky;

  std::vector<Field> workspace(factorization.largest_degree);

  for (Int supernode = num_supernodes - 1; supernode >= 0; --supernode) {
    const Int supernode_size = factorization.supernode_sizes[supernode];
    const Int supernode_start = factorization.supernode_starts[supernode];

    const Int index_beg = lower_factor.index_offsets[supernode];
    const Int degree = lower_factor.index_offsets[supernode + 1] - index_beg;
    const Int value_beg = lower_factor.value_offsets[supernode];

    // Handle the external updates for this supernode.
    if (supernode_size >=
        factorization.backward_solve_out_of_place_supernode_threshold) {
      for (Int i = 0; i < degree; ++i) {
        const Int row = lower_factor.indices[index_beg + i];
        workspace[i] = (*vector)[row];
      }

      ConstBlasMatrix<Field> subdiagonal;
      subdiagonal.height = supernode_size;
      subdiagonal.width = degree;
      subdiagonal.leading_dim = supernode_size;
      subdiagonal.data = &lower_factor.values[value_beg];

      ConjugateMatrixVectorProduct(
          Field{-1}, subdiagonal, workspace.data(),
          vector->data() + supernode_start);
    } else {
      for (Int j = supernode_size - 1; j >= 0; --j) {
        const Int column = supernode_start + j;
        Field& eta = (*vector)[column];
  
        // Handle the below-diagonal portion of this supernode.
        const Field* value_column = &lower_factor.values[value_beg + j];
        for (Int i = 0; i < degree; ++i) {
          const Int row = lower_factor.indices[index_beg + i];
          const Field& value = value_column[i * supernode_size];
          eta -= Conjugate(value) * (*vector)[row];
        }
      }
    }

    ConstBlasMatrix<Field> triangular_matrix;
    triangular_matrix.height = supernode_size;
    triangular_matrix.width = supernode_size;
    triangular_matrix.leading_dim = supernode_size;
    triangular_matrix.data =
        &diagonal_factor.values[diagonal_factor.value_offsets[supernode]];

    // Solve against the diagonal block of this supernode.
    if (is_cholesky) {
      TriangularSolveLeftLowerAdjoint(triangular_matrix,
                                      &(*vector)[supernode_start]);
    } else {
      TriangularSolveLeftLowerAdjointUnit(triangular_matrix,
                                          &(*vector)[supernode_start]);
    }
  }
}

template <class Field>
void PrintLowerFactor(const SupernodalLDLFactorization<Field>& factorization,
                      const std::string& label, std::ostream& os) {
  const SupernodalLowerFactor<Field>& lower_factor = factorization.lower_factor;
  const SupernodalDiagonalFactor<Field>& diag_factor =
      factorization.diagonal_factor;
  const bool is_cholesky = factorization.is_cholesky;

  auto print_entry = [&](const Int& row, const Int& column,
                         const Field& value) {
    os << row << " " << column << " " << value << "\n";
  };

  os << label << ": \n";
  const Int num_supernodes = factorization.supernode_sizes.size();
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int supernode_start = factorization.supernode_starts[supernode];
    const Int supernode_size = factorization.supernode_sizes[supernode];

    const Int index_beg = lower_factor.index_offsets[supernode];
    const Int index_end = lower_factor.index_offsets[supernode + 1];
    const Int degree = index_end - index_beg;

    const Int value_beg = lower_factor.value_offsets[supernode];

    const Int diag_offset = diag_factor.value_offsets[supernode];
    const Field* diag_block = &diag_factor.values[diag_offset];

    for (Int j = 0; j < supernode_size; ++j) {
      const Int column = supernode_start + j;

      // Print the portion in the diagonal block.
      if (is_cholesky) {
        print_entry(column, column, diag_block[j + j * supernode_size]);
      } else {
        print_entry(column, column, Field{1});
      }
      for (Int k = j + 1; k < supernode_size; ++k) {
        const Int row = supernode_start + k;
        const Field& value = diag_block[k + j * supernode_size];
        print_entry(row, column, value);
      }

      // Print the portion below the diagonal block.
      for (Int i = 0; i < degree; ++i) {
        const Int row = lower_factor.indices[index_beg + i];
        const Field& value =
            lower_factor.values[value_beg + i * supernode_size + j];
        print_entry(row, column, value);
      }
    }
  }
  os << std::endl;
}

template <class Field>
void PrintDiagonalFactor(const SupernodalLDLFactorization<Field>& factorization,
                         const std::string& label, std::ostream& os) {
  const SupernodalDiagonalFactor<Field>& diag_factor =
      factorization.diagonal_factor;
  if (factorization.is_cholesky) {
    // TODO(Jack Poulson): Print the identity.
    return;
  }

  os << label << ": \n";
  const Int num_supernodes = factorization.supernode_sizes.size();
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int diag_offset = diag_factor.value_offsets[supernode];
    const Field* diag_block = &diag_factor.values[diag_offset];

    const Int supernode_size = factorization.supernode_sizes[supernode];
    for (Int j = 0; j < supernode_size; ++j) {
      os << diag_block[j + j * supernode_size] << " ";
    }
  }
  os << std::endl;
}

}  // namespace catamari

#endif  // ifndef CATAMARI_SUPERNODAL_LDL_IMPL_H_
