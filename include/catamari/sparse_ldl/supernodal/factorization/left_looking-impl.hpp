/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_LEFT_LOOKING_IMPL_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_LEFT_LOOKING_IMPL_H_

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"
#include "catamari/io_utils.hpp"

#include "catamari/sparse_ldl/supernodal/factorization.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
void Factorization<Field>::LeftLookingSupernodeUpdate(
    Int supernode, const CoordinateMatrix<Field>& matrix,
    LeftLookingSharedState* shared_state, PrivateState<Field>* private_state) {
  typedef ComplexBase<Field> Real;
  BlasMatrixView<Field>& diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrixView<Field>& lower_block = lower_factor_->blocks[supernode];
  const Int supernode_size = lower_block.width;
  const Int supernode_degree = lower_block.height;
  const Int supernode_offset = ordering_.supernode_offsets[supernode];

  // Scatter the pattern of this supernode into pattern_flags.
  // TODO(Jack Poulson): Switch away from pointers to Int members.
  const Int* structure = lower_factor_->StructureBeg(supernode);
  for (Int i = 0; i < supernode_degree; ++i) {
    private_state->pattern_flags[structure[i]] = i;
  }

  shared_state->rel_rows[supernode] = 0;
  shared_state->intersect_ptrs[supernode] =
      lower_factor_->IntersectionSizesBeg(supernode);

  // for J = find(L(K, :))
  //   L(K:n, K) -= L(K:n, J) * (D(J, J) * L(K, J)')
  const Int head = shared_state->descendants.heads[supernode];
  for (Int next_descendant = head; next_descendant >= 0;) {
    const Int descendant = next_descendant;
    CATAMARI_ASSERT(descendant < supernode, "Looking into upper triangle");
    const ConstBlasMatrixView<Field>& descendant_lower_block =
        lower_factor_->blocks[descendant];
    const Int descendant_degree = descendant_lower_block.height;
    const Int descendant_size = descendant_lower_block.width;

    const Int descendant_main_rel_row = shared_state->rel_rows[descendant];
    const Int intersect_size = *shared_state->intersect_ptrs[descendant];
    CATAMARI_ASSERT(intersect_size > 0, "Non-positive intersection size.");

    const Int* descendant_structure =
        lower_factor_->StructureBeg(descendant) + descendant_main_rel_row;
    CATAMARI_ASSERT(
        descendant_structure < lower_factor_->StructureEnd(descendant),
        "Relative row exceeded end of structure.");
    CATAMARI_ASSERT(
        supernode_member_to_index_[*descendant_structure] == supernode,
        "First member of structure was not intersecting supernode.");
    CATAMARI_ASSERT(
        supernode_member_to_index_[descendant_structure[intersect_size - 1]] ==
            supernode,
        "Last member of structure was not intersecting supernode.");

    const ConstBlasMatrixView<Field> descendant_main_matrix =
        descendant_lower_block.Submatrix(descendant_main_rel_row, 0,
                                         intersect_size, descendant_size);

    BlasMatrixView<Field> scaled_transpose;
    if (control_.factorization_type != kCholeskyFactorization) {
      scaled_transpose.height = descendant_size;
      scaled_transpose.width = intersect_size;
      scaled_transpose.leading_dim = descendant_size;
      scaled_transpose.data = private_state->scaled_transpose_buffer.Data();
      FormScaledTranspose(control_.factorization_type,
                          diagonal_factor_->blocks[descendant].ToConst(),
                          descendant_main_matrix, &scaled_transpose);
    }

    const Int descendant_below_main_rel_row =
        shared_state->rel_rows[descendant] + intersect_size;
    const Int descendant_main_degree =
        descendant_degree - descendant_main_rel_row;
    const Int descendant_degree_remaining =
        descendant_degree - descendant_below_main_rel_row;

    // Construct mapping of descendant structure to supernode structure.
    const bool inplace_diag_update = intersect_size == supernode_size;
    const bool inplace_subdiag_update =
        inplace_diag_update && descendant_degree_remaining == supernode_degree;
    if (!inplace_subdiag_update) {
      for (Int i_rel = 0; i_rel < descendant_main_degree; ++i_rel) {
        const Int i = descendant_structure[i_rel];
        if (i_rel < intersect_size) {
          // Store the relative index in the diagonal block.
          CATAMARI_ASSERT(
              i >= supernode_offset && i < supernode_offset + supernode_size,
              "Invalid relative diagonal block index.");
          private_state->relative_indices[i_rel] = i - supernode_offset;
        } else {
          // Store the relative index in the lower structure.
          private_state->relative_indices[i_rel] =
              private_state->pattern_flags[i];
          CATAMARI_ASSERT(
              private_state->relative_indices[i_rel] >= 0 &&
                  private_state->relative_indices[i_rel] < supernode_degree,
              "Invalid subdiagonal relative index.");
        }
      }
    }

    // Update the diagonal block.
    BlasMatrixView<Field> workspace_matrix;
    workspace_matrix.height = intersect_size;
    workspace_matrix.width = intersect_size;
    workspace_matrix.leading_dim = intersect_size;
    workspace_matrix.data = private_state->workspace_buffer.Data();
    if (inplace_diag_update) {
      CATAMARI_START_TIMER(profile.herk);
      if (control_.factorization_type == kCholeskyFactorization) {
        LowerNormalHermitianOuterProduct(Real{-1}, descendant_main_matrix,
                                         Real{1}, &diagonal_block);
      } else {
        MatrixMultiplyLowerNormalNormal(Field{-1}, descendant_main_matrix,
                                        scaled_transpose.ToConst(), Field{1},
                                        &diagonal_block);
      }
      CATAMARI_STOP_TIMER(profile.herk);
    } else {
      CATAMARI_START_TIMER(profile.herk);
      if (control_.factorization_type == kCholeskyFactorization) {
        LowerNormalHermitianOuterProduct(Real{-1}, descendant_main_matrix,
                                         Real{0}, &workspace_matrix);
      } else {
        MatrixMultiplyLowerNormalNormal(Field{-1}, descendant_main_matrix,
                                        scaled_transpose.ToConst(), Field{0},
                                        &workspace_matrix);
      }
      CATAMARI_STOP_TIMER(profile.herk);

      for (Int j_rel = 0; j_rel < intersect_size; ++j_rel) {
        const Int j = private_state->relative_indices[j_rel];
        for (Int i_rel = j_rel; i_rel < intersect_size; ++i_rel) {
          const Int i = private_state->relative_indices[i_rel];
          diagonal_block(i, j) += workspace_matrix(i_rel, j_rel);
        }
      }
    }
#ifdef CATAMARI_ENABLE_TIMERS
    profile.herk_gflops +=
        std::pow(1. * intersect_size, 2.) * descendant_size / 1.e9;
#endif  // ifdefCATAMARI_ENABLE_TIMERS

    shared_state->intersect_ptrs[descendant]++;
    shared_state->rel_rows[descendant] = descendant_below_main_rel_row;

    next_descendant = shared_state->descendants.lists[descendant];
    if (descendant_degree_remaining > 0) {
      const ConstBlasMatrixView<Field> descendant_below_main_matrix =
          descendant_lower_block.Submatrix(descendant_below_main_rel_row, 0,
                                           descendant_degree_remaining,
                                           descendant_size);

      // L(KNext:n, K) -= L(KNext:n, J) * (D(J, J) * L(K, J)')
      //                = L(KNext:n, J) * Z(J, K).
      if (inplace_subdiag_update) {
        CATAMARI_START_TIMER(profile.gemm);
        if (control_.factorization_type == kCholeskyFactorization) {
          MatrixMultiplyNormalAdjoint(Field{-1}, descendant_below_main_matrix,
                                      descendant_main_matrix, Field{1},
                                      &lower_block);
        } else {
          MatrixMultiplyNormalNormal(Field{-1}, descendant_below_main_matrix,
                                     scaled_transpose.ToConst(), Field{1},
                                     &lower_block);
        }
        CATAMARI_STOP_TIMER(profile.gemm);
      } else {
        workspace_matrix.height = descendant_degree_remaining;
        workspace_matrix.width = intersect_size;
        workspace_matrix.leading_dim = descendant_degree_remaining;

        CATAMARI_START_TIMER(profile.gemm);
        if (control_.factorization_type == kCholeskyFactorization) {
          MatrixMultiplyNormalAdjoint(Field{-1}, descendant_below_main_matrix,
                                      descendant_main_matrix, Field{0},
                                      &workspace_matrix);
        } else {
          MatrixMultiplyNormalNormal(Field{-1}, descendant_below_main_matrix,
                                     scaled_transpose.ToConst(), Field{0},
                                     &workspace_matrix);
        }
        CATAMARI_STOP_TIMER(profile.gemm);

        // Update the active diagonal block.
        for (Int j_rel = 0; j_rel < intersect_size; ++j_rel) {
          const Int j = private_state->relative_indices[j_rel];
          CATAMARI_ASSERT(j >= 0 && j < supernode_size,
                          "Invalid unpacked column index.");
          CATAMARI_ASSERT(j + ordering_.supernode_offsets[supernode] ==
                              descendant_structure[j_rel],
                          "Mismatched unpacked column structure.");
          for (Int i_rel = 0; i_rel < descendant_degree_remaining; ++i_rel) {
            const Int i =
                private_state->relative_indices[i_rel + intersect_size];
            CATAMARI_ASSERT(i >= 0 && i < supernode_degree,
                            "Invalid row index.");
            CATAMARI_ASSERT(
                structure[i] == descendant_structure[i_rel + intersect_size],
                "Mismatched row structure.");
            lower_block(i, j) += workspace_matrix(i_rel, j_rel);
          }
        }
      }
#ifdef CATAMARI_ENABLE_TIMERS
      profile.gemm_gflops += 2. * intersect_size * descendant_size *
                             descendant_degree_remaining / 1.e9;
#endif  // ifdef CATAMARI_ENABLE_TIMERS

      // Insert the descendant supernode into the list of its next ancestor.
      // NOTE: We would need a lock for this in a multithreaded setting.
      const Int next_ancestor =
          supernode_member_to_index_[descendant_structure[intersect_size]];
      shared_state->descendants.Insert(next_ancestor, descendant);
    }
  }

  if (supernode_degree > 0) {
    // Insert the supernode into the list of its parent.
    // NOTE: We would need a lock for this in a multithreaded setting.
    const Int parent = supernode_member_to_index_[structure[0]];
    shared_state->descendants.Insert(parent, supernode);
  }

  // Clear the descendant list for this node.
  shared_state->descendants.heads[supernode] = -1;
}

template <class Field>
bool Factorization<Field>::LeftLookingSupernodeFinalize(
    Int supernode, SparseLDLResult* result) {
  BlasMatrixView<Field>& diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrixView<Field>& lower_block = lower_factor_->blocks[supernode];
  const Int degree = lower_block.height;
  const Int supernode_size = lower_block.width;

  CATAMARI_START_TIMER(profile.cholesky);
  const Int num_supernode_pivots = FactorDiagonalBlock(
      control_.block_size, control_.factorization_type, &diagonal_block);
  CATAMARI_STOP_TIMER(profile.cholesky);
  result->num_successful_pivots += num_supernode_pivots;
  if (num_supernode_pivots < supernode_size) {
    return false;
  }
#ifdef CATAMARI_ENABLE_TIMERS
  profile.cholesky_gflops += std::pow(1. * supernode_size, 3.) / 3.e9;
#endif  // ifdef CATAMARI_ENABLE_TIMERS
  IncorporateSupernodeIntoLDLResult(supernode_size, degree, result);
  if (!degree) {
    return true;
  }

  CATAMARI_ASSERT(supernode_size > 0, "Supernode size was non-positive.");
  CATAMARI_START_TIMER(profile.trsm);
  SolveAgainstDiagonalBlock(control_.factorization_type,
                            diagonal_block.ToConst(), &lower_block);
  CATAMARI_STOP_TIMER(profile.trsm);
#ifdef CATAMARI_ENABLE_TIMERS
  profile.trsm_gflops += std::pow(1. * supernode_size, 2.) * degree / 1.e9;
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  return true;
}

template <class Field>
bool Factorization<Field>::LeftLookingSubtree(
    Int supernode, const CoordinateMatrix<Field>& matrix,
    LeftLookingSharedState* shared_state, PrivateState<Field>* private_state,
    SparseLDLResult* result) {
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;

  CATAMARI_START_TIMER(shared_state->inclusive_timers[supernode]);

  Buffer<int> successes(num_children);
  Buffer<SparseLDLResult> result_contributions(num_children);

  // Recurse on the children.
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    CATAMARI_ASSERT(ordering_.assembly_forest.parents[child] == supernode,
                    "Incorrect child index");

    successes[child_index] =
        LeftLookingSubtree(child, matrix, shared_state, private_state,
                           &result_contributions[child_index]);
  }

  CATAMARI_START_TIMER(shared_state->exclusive_timers[supernode]);

  // Merge the child results (stopping if a failure is detected).
  bool succeeded = true;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    if (!successes[child_index]) {
      succeeded = false;
      break;
    }
    MergeContribution(result_contributions[child_index], result);
  }

  if (succeeded) {
    InitializeBlockColumn(supernode, matrix);
    LeftLookingSupernodeUpdate(supernode, matrix, shared_state, private_state);
    succeeded = LeftLookingSupernodeFinalize(supernode, result);
  }

  CATAMARI_STOP_TIMER(shared_state->inclusive_timers[supernode]);
  CATAMARI_STOP_TIMER(shared_state->exclusive_timers[supernode]);

  return succeeded;
}

template <class Field>
SparseLDLResult Factorization<Field>::LeftLooking(
    const CoordinateMatrix<Field>& matrix) {
#ifdef CATAMARI_OPENMP
  if (omp_get_max_threads() > 1) {
    return OpenMPLeftLooking(matrix);
  }
#endif
  const Int num_supernodes = ordering_.supernode_sizes.Size();

  LeftLookingSharedState shared_state;
  shared_state.rel_rows.Resize(num_supernodes);
  shared_state.intersect_ptrs.Resize(num_supernodes);
  shared_state.descendants.Initialize(num_supernodes);
#ifdef CATAMARI_ENABLE_TIMERS
  shared_state.inclusive_timers.Resize(num_supernodes);
  shared_state.exclusive_timers.Resize(num_supernodes);
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  PrivateState<Field> private_state;
  private_state.pattern_flags.Resize(matrix.NumRows());
  private_state.relative_indices.Resize(matrix.NumRows());
  if (control_.factorization_type != kCholeskyFactorization) {
    private_state.scaled_transpose_buffer.Resize(
        left_looking_scaled_transpose_size_, Field{0});
  }
  private_state.workspace_buffer.Resize(left_looking_workspace_size_, Field{0});

  SparseLDLResult result;

  // Note that any postordering of the supernodal elimination forest suffices.
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    InitializeBlockColumn(supernode, matrix);
    LeftLookingSupernodeUpdate(supernode, matrix, &shared_state,
                               &private_state);
    const bool succeeded = LeftLookingSupernodeFinalize(supernode, &result);
    if (!succeeded) {
      return result;
    }
  }

#ifdef CATAMARI_DEBUG
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    CATAMARI_ASSERT(shared_state.rel_rows[supernode] ==
                        lower_factor_->blocks[supernode].height,
                    "Did not properly handle relative row offsets.");
  }
#endif  // ifdef CATAMARI_DEBUG

#ifdef CATAMARI_ENABLE_TIMERS
  TruncatedForestTimersToDot(
      control_.inclusive_timings_filename, shared_state.inclusive_timers,
      ordering_.assembly_forest, control_.max_timing_levels,
      control_.avoid_timing_isolated_roots);
  TruncatedForestTimersToDot(
      control_.exclusive_timings_filename, shared_state.exclusive_timers,
      ordering_.assembly_forest, control_.max_timing_levels,
      control_.avoid_timing_isolated_roots);
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  return result;
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef
        // CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_LEFT_LOOKING_IMPL_H_
