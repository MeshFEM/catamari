/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SUPERNODAL_LDL_IMPL_H_
#define CATAMARI_LDL_SUPERNODAL_LDL_IMPL_H_

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"

#include "catamari/ldl/supernodal_ldl.hpp"

namespace catamari {
namespace supernodal_ldl {

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
void InitializeLeftLookingFactors(const CoordinateMatrix<Field>& matrix,
                                  const std::vector<Int>& parents,
                                  const std::vector<Int>& supernode_degrees,
                                  Factorization<Field>* factorization) {
  factorization->lower_factor.reset(new LowerFactor<Field>(
      factorization->supernode_sizes, supernode_degrees));
  factorization->diagonal_factor.reset(
      new DiagonalFactor<Field>(factorization->supernode_sizes));

  CATAMARI_ASSERT(
      supernode_degrees.size() == factorization->supernode_sizes.size(),
      "Invalid supernode degrees size.");

  // Store the largest supernode size of the factorization.
  factorization->max_supernode_size =
      *std::max_element(factorization->supernode_sizes.begin(),
                        factorization->supernode_sizes.end());

  // Store the largest degree of the factorization for use in the solve phase.
  factorization->max_degree =
      *std::max_element(supernode_degrees.begin(), supernode_degrees.end());

  FillStructureIndices(matrix, factorization->permutation,
                       factorization->inverse_permutation, parents,
                       factorization->supernode_sizes,
                       factorization->supernode_member_to_index,
                       factorization->lower_factor.get(),
                       &factorization->max_descendant_entries);

  FillNonzeros(
      matrix, factorization->permutation, factorization->inverse_permutation,
      factorization->supernode_starts, factorization->supernode_sizes,
      factorization->supernode_member_to_index,
      factorization->lower_factor.get(), factorization->diagonal_factor.get());
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
                  "Active supernode was <= the main supernode in update.");

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

      for (Int j = 0; j < descendant_main_intersect_size; ++j) {
        const Int column = descendant_main_indices[j];
        const Int j_rel = column - main_supernode_start;

        main_active_block->Entry(i_rel, j_rel) += workspace_matrix->Entry(i, j);
        workspace_matrix->Entry(i, j) = 0;
      }
    }
  }
}

template <class Field>
void LeftLookingSupernodeUpdate(Int main_supernode,
                                const CoordinateMatrix<Field>& matrix,
                                const std::vector<Int>& supernode_parents,
                                Factorization<Field>* factorization,
                                LeftLookingState<Field>* state) {
  LowerFactor<Field>& lower_factor = *factorization->lower_factor;
  DiagonalFactor<Field>& diagonal_factor = *factorization->diagonal_factor;
  const SymmetricFactorizationType factorization_type =
      factorization->factorization_type;

  BlasMatrix<Field>& main_diagonal_block =
      diagonal_factor.blocks[main_supernode];
  BlasMatrix<Field>& main_lower_block = lower_factor.blocks[main_supernode];
  const Int main_supernode_size = main_lower_block.width;

  state->pattern_flags[main_supernode] = main_supernode;

  state->rel_rows[main_supernode] = 0;
  state->intersect_ptrs[main_supernode] =
      lower_factor.IntersectionSizes(main_supernode);

  // Compute the supernodal row pattern.
  const Int num_packed = ComputeRowPattern(
      matrix, factorization->permutation, factorization->inverse_permutation,
      factorization->supernode_sizes, factorization->supernode_starts,
      factorization->supernode_member_to_index, supernode_parents,
      main_supernode, state->pattern_flags.data(), state->row_structure.data());

  // for J = find(L(K, :))
  //   L(K:n, K) -= L(K:n, J) * (D(J, J) * L(K, J)')
  for (Int index = 0; index < num_packed; ++index) {
    const Int descendant_supernode = state->row_structure[index];
    CATAMARI_ASSERT(descendant_supernode < main_supernode,
                    "Looking into upper triangle.");
    const ConstBlasMatrix<Field>& descendant_lower_block =
        lower_factor.blocks[descendant_supernode];
    const Int descendant_degree = descendant_lower_block.height;
    const Int descendant_supernode_size = descendant_lower_block.width;

    const Int descendant_main_intersect_size =
        *state->intersect_ptrs[descendant_supernode];

    const Int descendant_main_rel_row = state->rel_rows[descendant_supernode];
    ConstBlasMatrix<Field> descendant_main_matrix =
        descendant_lower_block.Submatrix(descendant_main_rel_row, 0,
                                         descendant_main_intersect_size,
                                         descendant_supernode_size);

    BlasMatrix<Field> scaled_transpose;
    scaled_transpose.height = descendant_supernode_size;
    scaled_transpose.width = descendant_main_intersect_size;
    scaled_transpose.leading_dim = descendant_supernode_size;
    scaled_transpose.data = state->scaled_transpose_buffer.data();

    FormScaledTranspose(factorization_type,
                        diagonal_factor.blocks[descendant_supernode].ToConst(),
                        descendant_main_matrix, &scaled_transpose);

    BlasMatrix<Field> workspace_matrix;
    workspace_matrix.height = descendant_main_intersect_size;
    workspace_matrix.width = descendant_main_intersect_size;
    workspace_matrix.leading_dim = descendant_main_intersect_size;
    workspace_matrix.data = state->workspace_buffer.data();

    UpdateDiagonalBlock(factorization_type, factorization->supernode_starts,
                        *factorization->lower_factor, main_supernode,
                        descendant_supernode, descendant_main_rel_row,
                        descendant_main_matrix, scaled_transpose.ToConst(),
                        &main_diagonal_block, &workspace_matrix);

    state->intersect_ptrs[descendant_supernode]++;
    state->rel_rows[descendant_supernode] += descendant_main_intersect_size;

    // L(KNext:n, K) -= L(KNext:n, J) * (D(J, J) * L(K, J)')
    //                = L(KNext:n, J) * Z(J, K).
    const Int* descendant_active_intersect_size_beg =
        state->intersect_ptrs[descendant_supernode];
    Int descendant_active_rel_row = state->rel_rows[descendant_supernode];
    const Int* main_active_intersect_sizes =
        lower_factor.IntersectionSizes(main_supernode);
    Int main_active_rel_row = 0;
    while (descendant_active_rel_row != descendant_degree) {
      const Int descendant_active_intersect_size =
          *descendant_active_intersect_size_beg;

      ConstBlasMatrix<Field> descendant_active_matrix =
          descendant_lower_block.Submatrix(descendant_active_rel_row, 0,
                                           descendant_active_intersect_size,
                                           descendant_supernode_size);

      // The width of the workspace matrix and pointer are already correct.
      workspace_matrix.height = descendant_active_intersect_size;
      workspace_matrix.leading_dim = descendant_active_intersect_size;

      SeekForMainActiveRelativeRow(
          main_supernode, descendant_supernode, descendant_active_rel_row,
          factorization->supernode_member_to_index, lower_factor,
          &main_active_rel_row, &main_active_intersect_sizes);
      const Int main_active_intersect_size = *main_active_intersect_sizes;

      BlasMatrix<Field> main_active_block = main_lower_block.Submatrix(
          main_active_rel_row, 0, main_active_intersect_size,
          main_supernode_size);

      UpdateSubdiagonalBlock(
          main_supernode, descendant_supernode, main_active_rel_row,
          descendant_main_rel_row, descendant_active_rel_row,
          factorization->supernode_starts,
          factorization->supernode_member_to_index, scaled_transpose.ToConst(),
          descendant_active_matrix, lower_factor, &main_active_block,
          &workspace_matrix);

      ++descendant_active_intersect_size_beg;
      descendant_active_rel_row += descendant_active_intersect_size;
    }
  }
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

template <class Field>
bool LeftLookingSupernodeFinalize(Int main_supernode,
                                  Factorization<Field>* factorization,
                                  LDLResult* result) {
  LowerFactor<Field>& lower_factor = *factorization->lower_factor;
  DiagonalFactor<Field>& diagonal_factor = *factorization->diagonal_factor;
  const SymmetricFactorizationType factorization_type =
      factorization->factorization_type;

  BlasMatrix<Field>& main_diagonal_block =
      diagonal_factor.blocks[main_supernode];
  BlasMatrix<Field>& main_lower_block = lower_factor.blocks[main_supernode];
  const Int main_degree = main_lower_block.height;
  const Int main_supernode_size = main_lower_block.width;

  const Int num_supernode_pivots =
      FactorDiagonalBlock(factorization_type, &main_diagonal_block);
  result->num_successful_pivots += num_supernode_pivots;
  if (num_supernode_pivots < main_supernode_size) {
    return false;
  }

  SolveAgainstDiagonalBlock(factorization_type, main_diagonal_block.ToConst(),
                            &main_lower_block);

  // Finish updating the result structure.
  result->largest_supernode =
      std::max(result->largest_supernode, main_supernode_size);
  result->num_factorization_entries +=
      (main_supernode_size * (main_supernode_size + 1)) / 2 +
      main_supernode_size * main_degree;

  // Add the approximate number of flops for the diagonal block factorization.
  result->num_factorization_flops +=
      std::pow(1. * main_supernode_size, 3.) / 3. +
      std::pow(1. * main_supernode_size, 2.) / 2.;

  // Add the approximate number of flops for the triangular solves of the
  // diagonal block against its structure.
  result->num_factorization_flops +=
      std::pow(1. * main_degree, 2.) * main_supernode_size;

  return true;
}

template <class Field>
LDLResult LeftLooking(const CoordinateMatrix<Field>& matrix,
                      const Control& control,
                      Factorization<Field>* factorization) {
  std::vector<Int> supernode_parents;
  {
    std::vector<Int> parents;
    std::vector<Int> supernode_degrees;
    FormSupernodes(matrix, control.relaxation_control, &parents,
                   &supernode_degrees, &supernode_parents, factorization);
    InitializeLeftLookingFactors(matrix, parents, supernode_degrees,
                                 factorization);
  }

  const Int num_supernodes = factorization->supernode_sizes.size();
  const Int max_supernode_size = factorization->max_supernode_size;

  LeftLookingState<Field> state;
  state.row_structure.resize(num_supernodes);
  state.pattern_flags.resize(num_supernodes);
  state.rel_rows.resize(num_supernodes);
  state.intersect_ptrs.resize(num_supernodes);
  state.scaled_transpose_buffer.resize(max_supernode_size * max_supernode_size,
                                       Field{0});
  state.workspace_buffer.resize(max_supernode_size * (max_supernode_size - 1),
                                Field{0});

  LDLResult result;

  // Note that any postordering of the supernodal elimination forest suffices.
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    LeftLookingSupernodeUpdate(supernode, matrix, supernode_parents,
                               factorization, &state);
    const bool succeeded =
        LeftLookingSupernodeFinalize(supernode, factorization, &result);
    if (!succeeded) {
      return result;
    }
  }

  return result;
}

}  // namespace supernodal_ldl

template <class Field>
LDLResult LDL(const CoordinateMatrix<Field>& matrix,
              const std::vector<Int>& permutation,
              const std::vector<Int>& inverse_permutation,
              const supernodal_ldl::Control& control,
              supernodal_ldl::Factorization<Field>* factorization) {
  factorization->permutation = permutation;
  factorization->inverse_permutation = inverse_permutation;
  factorization->factorization_type = control.factorization_type;
  factorization->forward_solve_out_of_place_supernode_threshold =
      control.forward_solve_out_of_place_supernode_threshold;
  factorization->backward_solve_out_of_place_supernode_threshold =
      control.backward_solve_out_of_place_supernode_threshold;
  return supernodal_ldl::LeftLooking(matrix, control, factorization);
}

template <class Field>
LDLResult LDL(const CoordinateMatrix<Field>& matrix,
              const supernodal_ldl::Control& control,
              supernodal_ldl::Factorization<Field>* factorization) {
  std::vector<Int> permutation, inverse_permutation;
  return LDL(matrix, permutation, inverse_permutation, control, factorization);
}

}  // namespace catamari

#endif  // ifndef CATAMARI_LDL_SUPERNODAL_LDL_IMPL_H_
