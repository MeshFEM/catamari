/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_RIGHT_LOOKING_OPENMP_IMPL_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_RIGHT_LOOKING_OPENMP_IMPL_H_
#ifdef CATAMARI_OPENMP

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"
#include "catamari/io_utils.hpp"

#include "catamari/sparse_ldl/supernodal/factorization.hpp"

#define FINEGRAINED_PARALLELISM 0

namespace catamari {
namespace supernodal_ldl {

template <class Field>
bool Factorization<Field>::OpenMPRightLookingSupernodeFinalize(
    Int supernode, const DynamicRegularizationParams<Field>& dynamic_reg_params,
    RightLookingSharedState<Field>* shared_state,
    Buffer<PrivateState<Field>>* private_states,
    SparseLDLResult<Field>* result) {
  typedef ComplexBase<Field> Real;
  BlasMatrixView<Field> diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrixView<Field> lower_block = lower_factor_->blocks[supernode];
  const Int degree = lower_block.height;
  const Int supernode_size = lower_block.width;

  // Initialize this supernode's Schur complement as the zero matrix.
  Buffer<Field>& schur_complement_buffer =
      shared_state->schur_complement_buffers[supernode];
  {
    BlasMatrixView<Field>& schur_complement =
        shared_state->schur_complements[supernode];
    schur_complement_buffer.Resize(degree * degree, Field{0});
    schur_complement.height = degree;
    schur_complement.width = degree;
    schur_complement.leading_dim = degree;
    schur_complement.data = schur_complement_buffer.Data();
  }
  BlasMatrixView<Field> schur_complement =
      shared_state->schur_complements[supernode];

#if FINEGRAINED_PARALLELISM
  #pragma omp taskgroup
  OpenMPMergeChildSchurComplements(control_.merge_grain_size, supernode,
                                   ordering_, lower_factor_.get(),
                                   diagonal_factor_.get(), shared_state);
#else
  MergeChildSchurComplements(supernode,
                             ordering_, lower_factor_.get(),
                             diagonal_factor_.get(), shared_state);
#endif

  Int num_supernode_pivots;
  if (control_.supernodal_pivoting) {
    // TODO(Jack Poulson): Add support for OpenMP supernodal pivoting.
    BlasMatrixView<Int> permutation = SupernodePermutation(supernode);
    num_supernode_pivots = PivotedFactorDiagonalBlock(
        control_.block_size, control_.factorization_type, &diagonal_block,
        &permutation);
    result->num_successful_pivots += num_supernode_pivots;
  } else {
    // TODO(Jack Poulson): Preallocate this buffer.
    Buffer<Field> multithreaded_buffer;
    // Buffer is not for Cholesky...
    if (control_.factorization_type != kCholeskyFactorization)
        multithreaded_buffer.Resize(supernode_size * supernode_size);
    {
#if FINEGRAINED_PARALLELISM
      #pragma omp taskgroup
      num_supernode_pivots = OpenMPFactorDiagonalBlock(
          control_.factor_tile_size, control_.block_size,
          control_.factorization_type, dynamic_reg_params, &diagonal_block,
          &multithreaded_buffer, &result->dynamic_regularization);
#else
      num_supernode_pivots = FactorDiagonalBlock(
          control_.block_size,
          control_.factorization_type, dynamic_reg_params, &diagonal_block,
          &result->dynamic_regularization);
#endif
      result->num_successful_pivots += num_supernode_pivots;

    }
  }
  if (num_supernode_pivots < supernode_size) {
    return false;
  }
  IncorporateSupernodeIntoLDLResult(supernode_size, degree, result);

  if (!degree) {
    // We can early exit.
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
  SolveAgainstDiagonalBlock(control_.factorization_type,
                            diagonal_block.ToConst(), &lower_block);

  if (control_.factorization_type == kCholeskyFactorization) {
     // This appears to always be *slightly* worth parallelizing...
#if 1
    #pragma omp taskgroup
    OpenMPLowerNormalHermitianOuterProduct(control_.outer_product_tile_size,
                                           Real{-1}, lower_block.ToConst(),
                                           Real{1}, &schur_complement);
#else
    LowerNormalHermitianOuterProduct(
                                     Real{-1}, lower_block.ToConst(),
                                     Real{1}, &schur_complement);
#endif
  } else {
    const int thread = omp_get_thread_num();
    PrivateState<Field> &private_state = (*private_states)[thread];
    BlasMatrixView<Field> scaled_transpose;
    scaled_transpose.height = supernode_size;
    scaled_transpose.width = degree;
    scaled_transpose.leading_dim = supernode_size;
    scaled_transpose.data = private_state.scaled_transpose_buffer.Data();

    #pragma omp taskgroup
    OpenMPFormScaledTranspose(
        control_.outer_product_tile_size, control_.factorization_type,
        diagonal_block.ToConst(), lower_block.ToConst(), &scaled_transpose);

    // Perform the multi-threaded MatrixMultiplyLowerNormalNormal.
    #pragma omp taskgroup
    OpenMPMatrixMultiplyLowerNormalNormal(
        control_.outer_product_tile_size, Field{-1}, lower_block.ToConst(),
        scaled_transpose.ToConst(), Field{1}, &schur_complement);
  }

  return true;
}

template <class Field>
bool Factorization<Field>::OpenMPRightLookingSubtree(
    Int supernode, const CoordinateMatrix<Field>& matrix,
    const DynamicRegularizationParams<Field>& dynamic_reg_params,
    const Buffer<double>& work_estimates, double min_parallel_work,
    RightLookingSharedState<Field>* shared_state,
    Buffer<PrivateState<Field>>* private_states,
    SparseLDLResult<Field>* result) {
  const double work_estimate = work_estimates[supernode];
  if (work_estimate < min_parallel_work) {
    const int thread = omp_get_thread_num();
    return RightLookingSubtree(supernode, matrix, dynamic_reg_params,
                               shared_state, &(*private_states)[thread],
                               result);
  }

  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;

  CATAMARI_START_TIMER(shared_state->inclusive_timers[supernode]);

  Buffer<SparseLDLResult<Field>> result_contributions(num_children);

  bool fail = false;

  // Recurse on the children.
  #pragma omp taskgroup
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    const Int child_offset = ordering_.supernode_offsets[child];

    #pragma omp task default(none)                                        \
      firstprivate(supernode, child, child_index, child_offset,           \
          min_parallel_work, shared_state, private_states)                \
      shared(fail, matrix, dynamic_reg_params, result_contributions, \
             work_estimates)
    {
      DynamicRegularizationParams<Field> subparams = dynamic_reg_params;
      subparams.offset = child_offset;
      bool success = OpenMPRightLookingSubtree(
          child, matrix, subparams, work_estimates, min_parallel_work,
          shared_state, private_states, &result_contributions[child_index]);
      if (!success) fail = true;
    }
  }

  CATAMARI_START_TIMER(shared_state->exclusive_timers[supernode]);

  // Merge the child results (stopping if a failure is detected).
  bool succeeded = !fail;

  if (succeeded) {
    for (Int child_index = 0; child_index < num_children; ++child_index) {
      MergeContribution(result_contributions[child_index], result);
    }
    if (dynamic_reg_params.enabled) {
      MergeDynamicRegularizations(result_contributions, result);
    }
#if FINEGRAINED_PARALLELISM
    OpenMPInitializeBlockColumn(supernode, matrix);
#else
    InitializeBlockColumn(supernode, matrix);
#endif
    #pragma omp taskgroup
    succeeded = OpenMPRightLookingSupernodeFinalize(
        supernode, dynamic_reg_params, shared_state, private_states, result);
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

  CATAMARI_STOP_TIMER(shared_state->inclusive_timers[supernode]);
  CATAMARI_STOP_TIMER(shared_state->exclusive_timers[supernode]);

  return succeeded;
}

template <class Field>
SparseLDLResult<Field> Factorization<Field>::OpenMPRightLooking(
    const CoordinateMatrix<Field>& matrix) {
  typedef ComplexBase<Field> Real;

  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  const Int max_threads = omp_get_max_threads();

  // Set up the base state of the dynamic regularization parameters. We only
  // need to update the offset for each child.
  static const Real kEpsilon = std::numeric_limits<Real>::epsilon();
  DynamicRegularizationParams<Field> dynamic_reg_params;
  dynamic_reg_params.enabled = control_.dynamic_regularization.enabled;
  dynamic_reg_params.positive_threshold = std::pow(
      kEpsilon, control_.dynamic_regularization.positive_threshold_exponent);
  dynamic_reg_params.negative_threshold = std::pow(
      kEpsilon, control_.dynamic_regularization.negative_threshold_exponent);
  if (control_.dynamic_regularization.relative) {
    const Real matrix_max_norm = MaxNorm(matrix);
    dynamic_reg_params.positive_threshold *= matrix_max_norm;
    dynamic_reg_params.negative_threshold *= matrix_max_norm;
  }
  dynamic_reg_params.signatures = &control_.dynamic_regularization.signatures;
  dynamic_reg_params.inverse_permutation = ordering_.inverse_permutation.Empty()
                                               ? nullptr
                                               : &ordering_.inverse_permutation;

  RightLookingSharedState<Field> shared_state;
  shared_state.schur_complement_buffers.Resize(num_supernodes);
  shared_state.schur_complements.Resize(num_supernodes);
#ifdef CATAMARI_ENABLE_TIMERS
  shared_state.inclusive_timers.Resize(num_supernodes);
  shared_state.exclusive_timers.Resize(num_supernodes);
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  Buffer<PrivateState<Field>> private_states(max_threads);
  if (control_.factorization_type != kCholeskyFactorization) {
    const Int workspace_size = max_lower_block_size_;
    #pragma omp taskgroup
    for (int t = 0; t < max_threads; ++t) {
      #pragma omp task default(none) firstprivate(t, workspace_size) \
          shared(private_states)
      private_states[t].scaled_transpose_buffer.Resize(workspace_size);
    }
  }

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

  SparseLDLResult<Field> result;

  bool fail = false;

  Buffer<SparseLDLResult<Field>> result_contributions(num_roots);

  // Recurse on each tree in the elimination forest.
  if (total_work < min_parallel_work) {
    const int thread = omp_get_thread_num();
    for (Int root_index = 0; root_index < num_roots; ++root_index) {
      const Int root = ordering_.assembly_forest.roots[root_index];
      DynamicRegularizationParams<Field> subparams = dynamic_reg_params;
      subparams.offset = ordering_.supernode_offsets[root_index];
      bool success = RightLookingSubtree(
          root, matrix, subparams, &shared_state, &private_states[thread],
          &result_contributions[root_index]);
      if (!success) fail = true;
    }
  } else {
    const int old_max_threads = GetMaxBlasThreads();
    SetNumBlasThreads(1);

    #pragma omp parallel
    #pragma omp single
    for (Int root_index = 0; root_index < num_roots; ++root_index) {
      const Int root = ordering_.assembly_forest.roots[root_index];
      const Int root_offset = ordering_.supernode_offsets[root];

      // As above, one could make use of OpenMP task priorities, e.g., with an
      // integer priority of:
      //
      //   const Int task_priority = std::pow(work_estimates[child], 0.25);
      //
      #pragma omp task default(none) \
          firstprivate(root, root_index, root_offset, min_parallel_work)      \
          shared(fail, matrix, dynamic_reg_params, result_contributions, \
              shared_state, private_states, work_estimates)
      {
        DynamicRegularizationParams<Field> subparams = dynamic_reg_params;
        subparams.offset = root_offset;
        bool success = OpenMPRightLookingSubtree(
            root, matrix, subparams, work_estimates, min_parallel_work,
            &shared_state, &private_states, &result_contributions[root_index]);
        if (!success) fail = true;
      }
    }

    SetNumBlasThreads(old_max_threads);
  }

  bool succeeded = !fail;
  if (succeeded) {
    for (Int index = 0; index < num_roots; ++index) {
      MergeContribution(result_contributions[index], &result);
    }
    if (dynamic_reg_params.enabled) {
      MergeDynamicRegularizations(result_contributions, &result);
    }
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
// CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_RIGHT_LOOKING_OPENMP_IMPL_H_
