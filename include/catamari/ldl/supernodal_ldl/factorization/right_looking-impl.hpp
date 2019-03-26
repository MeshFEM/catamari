/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_RIGHT_LOOKING_IMPL_H_
#define CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_RIGHT_LOOKING_IMPL_H_

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"
#include "catamari/io_utils.hpp"

#include "catamari/ldl/supernodal_ldl/factorization.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
bool Factorization<Field>::RightLookingSupernodeFinalize(
    Int supernode, RightLookingSharedState<Field>* shared_state,
    PrivateState<Field>* private_state, LDLResult* result) {
  typedef ComplexBase<Field> Real;
  BlasMatrixView<Field>& diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrixView<Field>& lower_block = lower_factor_->blocks[supernode];
  const Int degree = lower_block.height;
  const Int supernode_size = lower_block.width;

  // Initialize this supernode's Schur complement as the zero matrix.
  Buffer<Field>& schur_complement_buffer =
      shared_state->schur_complement_buffers[supernode];
  BlasMatrixView<Field>& schur_complement =
      shared_state->schur_complements[supernode];
  schur_complement_buffer.Resize(degree * degree, Field{0});
  schur_complement.height = degree;
  schur_complement.width = degree;
  schur_complement.leading_dim = degree;
  schur_complement.data = schur_complement_buffer.Data();

  MergeChildSchurComplements(supernode, ordering_, lower_factor_.get(),
                             diagonal_factor_.get(), shared_state);

  const Int num_supernode_pivots = FactorDiagonalBlock(
      control_.block_size, control_.factorization_type, &diagonal_block);
  result->num_successful_pivots += num_supernode_pivots;
  if (num_supernode_pivots < supernode_size) {
    return false;
  }
  IncorporateSupernodeIntoLDLResult(supernode_size, degree, result);

  if (!degree) {
    // We can early exit.
    return true;
  }

  CATAMARI_ASSERT(supernode_size > 0, "Supernode size was non-positive.");
  SolveAgainstDiagonalBlock(control_.factorization_type,
                            diagonal_block.ToConst(), &lower_block);

  if (control_.factorization_type == kCholeskyFactorization) {
    LowerNormalHermitianOuterProduct(Real{-1}, lower_block.ToConst(), Real{1},
                                     &schur_complement);
  } else {
    BlasMatrixView<Field> scaled_transpose;
    scaled_transpose.height = supernode_size;
    scaled_transpose.width = degree;
    scaled_transpose.leading_dim = supernode_size;
    scaled_transpose.data = private_state->scaled_transpose_buffer.Data();
    FormScaledTranspose(control_.factorization_type, diagonal_block.ToConst(),
                        lower_block.ToConst(), &scaled_transpose);
    MatrixMultiplyLowerNormalNormal(Field{-1}, lower_block.ToConst(),
                                    scaled_transpose.ToConst(), Field{1},
                                    &schur_complement);
  }

  return true;
}

template <class Field>
bool Factorization<Field>::RightLookingSubtree(
    Int supernode, const CoordinateMatrix<Field>& matrix,
    RightLookingSharedState<Field>* shared_state,
    PrivateState<Field>* private_state, LDLResult* result) {
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;

#ifdef CATAMARI_ENABLE_TIMERS
  shared_state->inclusive_timers[supernode].Start();
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  Buffer<int> successes(num_children);
  Buffer<LDLResult> result_contributions(num_children);

  // Recurse on the children.
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    CATAMARI_ASSERT(ordering_.assembly_forest.parents[child] == supernode,
                    "Incorrect child index");
    LDLResult& result_contribution = result_contributions[child_index];
    successes[child_index] = RightLookingSubtree(
        child, matrix, shared_state, private_state, &result_contribution);
  }

#ifdef CATAMARI_ENABLE_TIMERS
  shared_state->exclusive_timers[supernode].Start();
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  // Merge the children's results (stopping if a failure is detected).
  bool succeeded = true;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    if (!successes[child_index]) {
      succeeded = false;
      break;
    }
    MergeContribution(result_contributions[child_index], result);
  }

  if (succeeded) {
    succeeded = RightLookingSupernodeFinalize(supernode, shared_state,
                                              private_state, result);
  } else {
    // Clear the child fronts.
    for (Int child_index = 0; child_index < num_children; ++child_index) {
      const Int child =
          ordering_.assembly_forest.children[child_beg + child_index];
      Buffer<Field>& child_schur_complement_buffer =
          shared_state->schur_complement_buffers[child];
      BlasMatrixView<Field>& child_schur_complement =
          shared_state->schur_complements[child];
      child_schur_complement.height = 0;
      child_schur_complement.width = 0;
      child_schur_complement.data = nullptr;
      child_schur_complement_buffer.Clear();
    }
  }

#ifdef CATAMARI_ENABLE_TIMERS
  shared_state->inclusive_timers[supernode].Stop();
  shared_state->exclusive_timers[supernode].Stop();
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  return succeeded;
}

template <class Field>
LDLResult Factorization<Field>::RightLooking(
    const CoordinateMatrix<Field>& matrix) {
#ifdef CATAMARI_OPENMP
  if (omp_get_max_threads() > 1) {
    return OpenMPRightLooking(matrix);
  }
#endif
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const Int num_roots = ordering_.assembly_forest.roots.Size();

  RightLookingSharedState<Field> shared_state;
  shared_state.schur_complement_buffers.Resize(num_supernodes);
  shared_state.schur_complements.Resize(num_supernodes);
#ifdef CATAMARI_ENABLE_TIMERS
  shared_state.inclusive_timers.Resize(num_supernodes);
  shared_state.exclusive_timers.Resize(num_supernodes);
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  PrivateState<Field> private_state;
  if (control_.factorization_type != kCholeskyFactorization) {
    private_state.scaled_transpose_buffer.Resize(max_lower_block_size_);
  }

  LDLResult result;

  Buffer<int> successes(num_roots);
  Buffer<LDLResult> result_contributions(num_roots);

  // Merge the child results (stopping if a failure is detected).
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = ordering_.assembly_forest.roots[root_index];
    LDLResult& result_contribution = result_contributions[root_index];
    successes[root_index] = RightLookingSubtree(
        root, matrix, &shared_state, &private_state, &result_contribution);
  }

  // Merge the child results (stopping if a failure is detected).
  for (Int index = 0; index < num_roots; ++index) {
    if (!successes[index]) {
      break;
    }
    MergeContribution(result_contributions[index], &result);
  }

#ifdef CATAMARI_ENABLE_TIMERS
  // TODO(Jack Poulson): Make these parameters configurable.
  const Int max_levels = 4;
  const bool avoid_isolated_roots = true;
  const std::string inclusive_filename = "inclusive.gv";
  const std::string exclusive_filename = "exclusive.gv";
  TruncatedForestTimersToDot(inclusive_filename, shared_state.inclusive_timers,
                             ordering_.assembly_forest, max_levels,
                             avoid_isolated_roots);
  TruncatedForestTimersToDot(exclusive_filename, shared_state.exclusive_timers,
                             ordering_.assembly_forest, max_levels,
                             avoid_isolated_roots);
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  return result;
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef
        // CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_RIGHT_LOOKING_IMPL_H_
