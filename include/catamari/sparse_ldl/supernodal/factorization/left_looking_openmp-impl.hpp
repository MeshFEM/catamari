/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_LEFT_LOOKING_OPENMP_IMPL_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_LEFT_LOOKING_OPENMP_IMPL_H_
#ifdef CATAMARI_OPENMP

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"
#include "catamari/io_utils.hpp"

#include "catamari/sparse_ldl/supernodal/factorization.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
void Factorization<Field>::OpenMPLeftLookingSupernodeUpdate(
    Int main_supernode, const CoordinateMatrix<Field>& matrix,
    LeftLookingSharedState* shared_state,
    Buffer<PrivateState<Field>>* private_states) {
  // The left-looking supernodal update needs to be rewritten, but it is low
  // priority since a multithreaded right-looking approach will almost always
  // be faster.

  // TODO(Jack Poulson): Rewrite this routine. A mutex will be needed for the
  // linked list updates.
}

template <class Field>
bool Factorization<Field>::OpenMPLeftLookingSupernodeFinalize(
    Int supernode, Buffer<PrivateState<Field>>* private_states,
    SparseLDLResult* result) {
  BlasMatrixView<Field>& diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrixView<Field>& lower_block = lower_factor_->blocks[supernode];
  const Int degree = lower_block.height;
  const Int supernode_size = lower_block.width;

  Int num_supernode_pivots;
  if (control_.supernodal_pivoting) {
    BlasMatrixView<Int> permutation = SupernodePermutation(supernode);
    num_supernode_pivots = PivotedFactorDiagonalBlock(
        control_.block_size, control_.factorization_type, &diagonal_block,
        &permutation);
  } else {
    #pragma omp taskgroup
    {
      const int thread = omp_get_thread_num();
      Buffer<Field>* buffer =
          &(*private_states)[thread].scaled_transpose_buffer;

      num_supernode_pivots = OpenMPFactorDiagonalBlock(
          control_.factor_tile_size, control_.block_size,
          control_.factorization_type, &diagonal_block, buffer);
      result->num_successful_pivots += num_supernode_pivots;
    }
  }
  if (num_supernode_pivots < supernode_size) {
    return false;
  }
  IncorporateSupernodeIntoLDLResult(supernode_size, degree, result);
  if (!degree) {
    return true;
  }

  CATAMARI_ASSERT(supernode_size > 0, "Supernode size was non-positive.");
  if (control_.supernodal_pivoting) {
    // Solve against P^T from the right, which is the same as applying P
    // from the right, which is the same as applying P^T to each row.
    const ConstBlasMatrixView<Int> permutation =
        SupernodePermutation(supernode);
    InversePermuteColumns(permutation, &lower_block);
  }
  #pragma omp taskgroup
  OpenMPSolveAgainstDiagonalBlock(control_.outer_product_tile_size,
                                  control_.factorization_type,
                                  diagonal_block.ToConst(), &lower_block);

  return true;
}

template <class Field>
bool Factorization<Field>::OpenMPLeftLookingSubtree(
    Int supernode, const CoordinateMatrix<Field>& matrix,
    const Buffer<double>& work_estimates, double min_parallel_work,
    LeftLookingSharedState* shared_state,
    Buffer<PrivateState<Field>>* private_states, SparseLDLResult* result) {
  const double work_estimate = work_estimates[supernode];
  if (work_estimate < min_parallel_work) {
    const int thread = omp_get_thread_num();
    return LeftLookingSubtree(supernode, matrix, shared_state,
                              &(*private_states)[thread], result);
  }

  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;

  CATAMARI_START_TIMER(shared_state->inclusive_timers[supernode]);

  Buffer<int> successes(num_children);
  Buffer<SparseLDLResult> result_contributions(num_children);

  #pragma omp taskgroup
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    #pragma omp task default(none)                                     \
        firstprivate(supernode, child_index, child, min_parallel_work, \
            shared_state, private_states)                              \
        shared(successes, matrix, work_estimates, result_contributions)
    successes[child_index] = OpenMPLeftLookingSubtree(
        child, matrix, work_estimates, min_parallel_work, shared_state,
        private_states, &result_contributions[child_index]);
  }

  CATAMARI_START_TIMER(shared_state->exclusive_timers[supernode]);

  bool succeeded = true;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    if (!successes[child_index]) {
      succeeded = false;
      break;
    }
    MergeContribution(result_contributions[child_index], result);
  }

  if (succeeded) {
    #pragma omp taskgroup
    OpenMPInitializeBlockColumn(supernode, matrix);

    // Handle the current supernode's elimination.
    #pragma omp taskgroup
    OpenMPLeftLookingSupernodeUpdate(supernode, matrix, shared_state,
                                     private_states);

    #pragma omp taskgroup
    succeeded =
        OpenMPLeftLookingSupernodeFinalize(supernode, private_states, result);
  }

  CATAMARI_STOP_TIMER(shared_state->inclusive_timers[supernode]);
  CATAMARI_STOP_TIMER(shared_state->exclusive_timers[supernode]);

  return succeeded;
}

template <class Field>
SparseLDLResult Factorization<Field>::OpenMPLeftLooking(
    const CoordinateMatrix<Field>& matrix) {
  const Int num_rows = matrix.NumRows();
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  const int max_threads = omp_get_max_threads();

  CATAMARI_START_TIMER(profile.left_looking_allocate);
  LeftLookingSharedState shared_state;
  shared_state.rel_rows.Resize(num_supernodes);
  shared_state.intersect_ptrs.Resize(num_supernodes);
  shared_state.descendants.Initialize(num_supernodes);
#ifdef CATAMARI_ENABLE_TIMERS
  shared_state.inclusive_timers.Resize(num_supernodes);
  shared_state.exclusive_timers.Resize(num_supernodes);
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  Buffer<PrivateState<Field>> private_states(max_threads);

  #pragma omp taskgroup
  for (int t = 0; t < max_threads; ++t) {
    #pragma omp task default(none) \
        firstprivate(t, num_supernodes, num_rows) shared(private_states)
    {
      PrivateState<Field>& private_state = private_states[t];
      private_state.pattern_flags.Resize(num_rows);
      private_state.relative_indices.Resize(num_rows);
      if (control_.factorization_type != kCholeskyFactorization) {
        private_state.scaled_transpose_buffer.Resize(
            left_looking_scaled_transpose_size_, Field{0});
      }
      private_state.workspace_buffer.Resize(left_looking_workspace_size_,
                                            Field{0});
    }
  }
  CATAMARI_STOP_TIMER(profile.left_looking_allocate);

  // Compute flop-count estimates so that we may prioritize the expensive
  // tasks before the cheaper ones.
  Buffer<double> work_estimates(num_supernodes);
  for (const Int& root : ordering_.assembly_forest.roots) {
    FillSubtreeWorkEstimates(root, ordering_.assembly_forest, *lower_factor_,
                             &work_estimates);
  }
  const double total_work =
      std::accumulate(work_estimates.begin(), work_estimates.end(), 0.);
  const double min_parallel_ratio_work =
      (total_work * control_.parallel_ratio_threshold) / max_threads;
  const double min_parallel_work =
      std::max(control_.min_parallel_threshold, min_parallel_ratio_work);

  SparseLDLResult result;
  Buffer<int> successes(num_roots);
  Buffer<SparseLDLResult> result_contributions(num_roots);

  if (total_work < min_parallel_work) {
    for (Int root_index = 0; root_index < num_roots; ++root_index) {
      const Int root = ordering_.assembly_forest.roots[root_index];
      successes[root_index] =
          LeftLookingSubtree(root, matrix, &shared_state, &private_states[0],
                             &result_contributions[root_index]);
    }
  } else {
    const int old_max_threads = GetMaxBlasThreads();
    SetNumBlasThreads(1);

    #pragma omp parallel
    #pragma omp single
    for (Int root_index = 0; root_index < num_roots; ++root_index) {
      const Int root = ordering_.assembly_forest.roots[root_index];
      #pragma omp task default(none)                                      \
          firstprivate(root_index, root, min_parallel_work)               \
          shared(successes, matrix, work_estimates, result_contributions, \
              shared_state, private_states)
      successes[root_index] = OpenMPLeftLookingSubtree(
          root, matrix, work_estimates, min_parallel_work, &shared_state,
          &private_states, &result_contributions[root_index]);
    }

    SetNumBlasThreads(old_max_threads);
  }

  for (Int index = 0; index < num_roots; ++index) {
    if (!successes[index]) {
      break;
    }
    MergeContribution(result_contributions[index], &result);
  }

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

#endif  // ifdef CATAMARI_OPENMP
#endif  // ifndef
// CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_LEFT_LOOKING_OPENMP_IMPL_H_
