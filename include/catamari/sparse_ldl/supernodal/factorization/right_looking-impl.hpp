/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_RIGHT_LOOKING_IMPL_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_RIGHT_LOOKING_IMPL_H_

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"
#include "catamari/io_utils.hpp"
#include "catamari/norms.hpp"

#include "catamari/sparse_ldl/supernodal/factorization.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
bool Factorization<Field>::RightLookingSupernodeFinalize(
    Int supernode, const DynamicRegularizationParams<Field>& dynamic_reg_params,
    RightLookingSharedState<Field>* shared_state,
    PrivateState<Field>* private_state, SparseLDLResult<Field>* result) {
  typedef ComplexBase<Field> Real;
  BlasMatrixView<Field>& diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrixView<Field>& lower_block = lower_factor_->blocks[supernode];
  const Int degree = lower_block.height;
  const Int supernode_size = lower_block.width;

  BlasMatrixView<Field>& schur_complement = shared_state->schur_complements[supernode];

#if ALLOCATE_SCHUR_COMPLEMENT_OTF
  // Initialize this supernode's Schur complement as the zero matrix.
  Buffer<Field>& schur_complement_buffer = shared_state->schur_complement_buffers[supernode];
  {
    schur_complement_buffer.Resize(degree * degree);
    schur_complement.height = degree;
    schur_complement.width = degree;
    schur_complement.leading_dim = degree;
    schur_complement.data = schur_complement_buffer.Data();
    eigenMap(schur_complement).setZero(); // Notably faster than having `schur_complement_buffer.Resize` zero-init!
  }
#elif ZERO_SCHUR_COMPLEMENT_OTF
  eigenMap(schur_complement).setZero();
#endif

  CATAMARI_START_TIMER(profile.merge);
  MergeChildSchurComplements(supernode, ordering_, lower_factor_.get(),
                             diagonal_factor_.get(), shared_state);
  CATAMARI_STOP_TIMER(profile.merge);

  CATAMARI_START_TIMER(profile.cholesky);
  Int num_supernode_pivots;
  if (control_.supernodal_pivoting) {
    BlasMatrixView<Int> permutation = SupernodePermutation(supernode);
    num_supernode_pivots = PivotedFactorDiagonalBlock(
        control_.block_size, control_.factorization_type, &diagonal_block,
        &permutation);
  } else {
    num_supernode_pivots = FactorDiagonalBlock(
        control_.block_size, control_.factorization_type, dynamic_reg_params,
        &diagonal_block, &result->dynamic_regularization);
  }
  CATAMARI_STOP_TIMER(profile.cholesky);
#ifdef CATAMARI_ENABLE_TIMERS
  profile.cholesky_gflops += std::pow(1. * supernode_size, 3.) / 3.e9;
#endif  // ifdef CATAMARI_ENABLE_TIMERS
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
  CATAMARI_START_TIMER(profile.trsm);
  if (control_.supernodal_pivoting) {
    // Solve against P^T from the right, which is the same as applying P
    // from the right, which is the same as applying P^T to each row.
    const ConstBlasMatrixView<Int> permutation =
        SupernodePermutation(supernode);
    InversePermuteColumns(permutation, &lower_block);
  }
  SolveAgainstDiagonalBlock(control_.factorization_type,
                            diagonal_block.ToConst(), &lower_block);
  CATAMARI_STOP_TIMER(profile.trsm);
#ifdef CATAMARI_ENABLE_TIMERS
  profile.trsm_gflops += std::pow(1. * supernode_size, 2.) * degree / 1.e9;
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  CATAMARI_START_TIMER(profile.herk);
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
  CATAMARI_STOP_TIMER(profile.herk);
#ifdef CATAMARI_ENABLE_TIMERS
  profile.herk_gflops += std::pow(1. * degree, 2.) * supernode_size / 1.e9;
#endif

  return true;
}

template <class Field>
bool Factorization<Field>::RightLookingSubtree(
    Int supernode, const CoordinateMatrix<Field>& matrix,
    const DynamicRegularizationParams<Field>& dynamic_reg_params,
    RightLookingSharedState<Field>* shared_state,
    PrivateState<Field>* private_state, SparseLDLResult<Field>* result) {
  // assert(false);
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;

  CATAMARI_START_TIMER(shared_state->inclusive_timers[supernode]);

  Buffer<SparseLDLResult<Field>> result_contributions(num_children);

  // Set up a dynamic regularization structure for the children.
  DynamicRegularizationParams<Field> subparams = dynamic_reg_params;

  // Recurse on the children.
  bool succeeded = true;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    CATAMARI_ASSERT(ordering_.assembly_forest.parents[child] == supernode,
                    "Incorrect child index");
    SparseLDLResult<Field>& result_contribution =
        result_contributions[child_index];

    subparams.offset = ordering_.supernode_offsets[child];
    bool success =
        RightLookingSubtree(child, matrix, subparams, shared_state,
                            private_state, &result_contribution);
    // Early exit on failure.
    if (!success) { succeeded = false; break; }
    MergeContribution(result_contribution, result);
  }

  CATAMARI_START_TIMER(shared_state->exclusive_timers[supernode]);

  // Merge the children's results
  if (succeeded) {
      if (dynamic_reg_params.enabled)
        MergeDynamicRegularizations(result_contributions, result);
#if !LOAD_MATRIX_OUTSIDE
      InitializeBlockColumn(supernode, matrix);
#endif
      succeeded = RightLookingSupernodeFinalize(
          supernode, dynamic_reg_params, shared_state, private_state, result);
  }

#if ALLOCATE_SCHUR_COMPLEMENT_OTF
  // Clear the child fronts.
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child = ordering_.assembly_forest.children[child_beg + child_index];
    Buffer<Field>& child_schur_complement_buffer = shared_state->schur_complement_buffers[child];
    BlasMatrixView<Field>& child_schur_complement = shared_state->schur_complements[child];
    child_schur_complement.height = 0;
    child_schur_complement.width = 0;
    if (child_schur_complement.data) {
        child_schur_complement.data = nullptr;
        child_schur_complement_buffer.Clear();
    }
  }
#endif

  CATAMARI_STOP_TIMER(shared_state->inclusive_timers[supernode]);
  CATAMARI_STOP_TIMER(shared_state->exclusive_timers[supernode]);

  return succeeded;
}

template <class Field>
SparseLDLResult<Field> Factorization<Field>::RightLooking(
    const CoordinateMatrix<Field>& matrix) {
  typedef ComplexBase<Field> Real;

  const Int max_threads = get_max_num_tbb_threads();
  if (true || (max_threads > 1)) { // the tbb implementation is even faster in the single-threaded case...
    return OpenMPRightLooking(matrix);
  }
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const Int num_roots = ordering_.assembly_forest.roots.Size();

  RightLookingSharedState<Field> &shared_state = shared_state_;
  if (shared_state.schur_complements.Size() != num_supernodes) {
      shared_state.schur_complements.Resize(num_supernodes);
#if ALLOCATE_SCHUR_COMPLEMENT_OTF
      shared_state.schur_complement_buffers.Resize(num_supernodes);
#else
      {
          BENCHMARK_SCOPED_TIMER_SECTION atimer("Allocate buffers");

          Int total_size = 0;
          for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
              const Int degree = lower_factor_->blocks[supernode].height;
              total_size += degree * degree;
          }

          // std::cout << "Allocating buffer of size " << total_size << "(" << (8.0 * total_size / 1024. / 1024.) << "MB)" << std::endl;
          shared_state.schur_complement_buffers.Resize(1);
          Buffer<Field> &workspace_buffer = shared_state.schur_complement_buffers[0];
          workspace_buffer.Resize(total_size);
          Int offset = 0;
          shared_state.schur_complements.Resize(num_supernodes);
          for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
              const Int degree = lower_factor_->blocks[supernode].height;
              const Int workspace_size = degree * degree;

              auto &supernode_rhs = shared_state.schur_complements[supernode];
              supernode_rhs.height = degree;
              supernode_rhs.width = degree;
              supernode_rhs.leading_dim = degree;
              supernode_rhs.data = workspace_buffer.Data() + offset;
              offset += workspace_size;
          }
      }
#endif
  }

#if !(ALLOCATE_SCHUR_COMPLEMENT_OTF || ZERO_SCHUR_COMPLEMENT_OTF)
  {
      // Existing data in the pre-allocated schur_complements buffer must be cleared.
      BENCHMARK_SCOPED_TIMER_SECTION sztimer("setZeroParallel");
      setZeroParallel(eigenMap(shared_state.schur_complement_buffers[0]));
  }
#endif

#ifdef CATAMARI_ENABLE_TIMERS
  shared_state.inclusive_timers.Resize(num_supernodes);
  shared_state.exclusive_timers.Resize(num_supernodes);
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  PrivateState<Field> private_state;
  if (control_.factorization_type != kCholeskyFactorization) {
    private_state.scaled_transpose_buffer.Resize(max_lower_block_size_);
  }

  // Allocate the map from child structures to parent fronts.
  auto &ncdi = ordering_.assembly_forest.num_child_diag_indices;
  auto &cri =  ordering_.assembly_forest.child_rel_indices;
  if ( cri.Size() != num_supernodes) {
      cri.Resize(num_supernodes);
      ncdi.Resize(num_supernodes);
  }

  SparseLDLResult<Field> result;

  Buffer<SparseLDLResult<Field>> result_contributions(num_roots);

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

  // Factor the children.
  bool succeeded = true;
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = ordering_.assembly_forest.roots[root_index];
    SparseLDLResult<Field>& result_contribution =
        result_contributions[root_index];
    dynamic_reg_params.offset = ordering_.supernode_offsets[root];
    bool success =
        RightLookingSubtree(root, matrix, dynamic_reg_params, &shared_state,
                            &private_state, &result_contribution);
    // Early exit on failure.
    if (!success) { succeeded = false; break; }
    MergeContribution(result_contribution, &result);
  }

  if (succeeded && dynamic_reg_params.enabled) {
    MergeDynamicRegularizations(result_contributions, &result);
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

#endif  // ifndef
        // CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_RIGHT_LOOKING_IMPL_H_
