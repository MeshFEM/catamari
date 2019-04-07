/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_HERMITIAN_DPP_SUPERNODAL_RIGHT_LOOKING_OPENMP_IMPL_H_
#define CATAMARI_SPARSE_HERMITIAN_DPP_SUPERNODAL_RIGHT_LOOKING_OPENMP_IMPL_H_
#ifdef CATAMARI_OPENMP

#include <algorithm>

#include "catamari/io_utils.hpp"

#include "catamari/sparse_hermitian_dpp/supernodal.hpp"

namespace catamari {

template <class Field>
void SupernodalHermitianDPP<Field>::OpenMPRightLookingSupernodeSample(
    Int supernode, bool maximum_likelihood,
    supernodal_ldl::RightLookingSharedState<Field>* shared_state,
    Buffer<PrivateState>* private_states, std::vector<Int>* sample) const {
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

  supernodal_ldl::OpenMPMergeChildSchurComplements(
      control_.merge_grain_size, supernode, ordering_, lower_factor_.get(),
      diagonal_factor_.get(), shared_state);

  // Sample and factor the diagonal block.
  std::vector<Int> supernode_sample;
  #pragma omp taskgroup
  {
    const int thread = omp_get_thread_num();
    PrivateState& private_state = (*private_states)[thread];
    Buffer<Field>* buffer = &private_state.ldl_state.scaled_transpose_buffer;
    supernode_sample = OpenMPSampleLowerHermitianDPP(
        control_.factor_tile_size, control_.block_size, maximum_likelihood,
        &diagonal_block, &private_state.generator, buffer);
  }
  AppendSupernodeSample(supernode, supernode_sample, sample);

  if (!degree) {
    // We can early exit.
    return;
  }

  CATAMARI_ASSERT(supernode_size > 0, "Supernode size was non-positive.");
  const SymmetricFactorizationType factorization_type =
      kLDLAdjointFactorization;
  #pragma omp taskgroup
  supernodal_ldl::OpenMPSolveAgainstDiagonalBlock(
      control_.outer_product_tile_size, factorization_type,
      diagonal_block.ToConst(), &lower_block);

  // TODO(Jack Poulson): See if this can be pre-allocated.
  Buffer<Field> scaled_transpose_buffer(degree * supernode_size);
  BlasMatrixView<Field> scaled_transpose;
  scaled_transpose.height = supernode_size;
  scaled_transpose.width = degree;
  scaled_transpose.leading_dim = supernode_size;
  scaled_transpose.data = scaled_transpose_buffer.Data();

  #pragma omp taskgroup
  supernodal_ldl::OpenMPFormScaledTranspose(
      control_.outer_product_tile_size, factorization_type,
      diagonal_block.ToConst(), lower_block.ToConst(), &scaled_transpose);

  #pragma omp taskgroup
  OpenMPMatrixMultiplyLowerNormalNormal(
      control_.outer_product_tile_size, Field{-1}, lower_block.ToConst(),
      scaled_transpose.ToConst(), Field{1}, &schur_complement);
}

template <class Field>
void SupernodalHermitianDPP<Field>::OpenMPRightLookingSubtree(
    Int supernode, bool maximum_likelihood,
    supernodal_ldl::RightLookingSharedState<Field>* shared_state,
    Buffer<PrivateState>* private_states, std::vector<Int>* sample) const {
  const double work_estimate = work_estimates_[supernode];
  if (work_estimate < min_parallel_work_) {
    const int thread = omp_get_thread_num();
    PrivateState& private_state = (*private_states)[thread];
    RightLookingSubtree(supernode, maximum_likelihood, shared_state,
                        &private_state, sample);
    return;
  }

  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;

  CATAMARI_START_TIMER(shared_state->inclusive_timers[supernode]);

  // NOTE: We could alternatively avoid switch to maintaining a single, shared
  // boolean list of length 'num_rows' which flags each entry as 'in' or
  // 'out' of the sample.
  Buffer<std::vector<Int>> subsamples(num_children);

  #pragma omp taskgroup
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    CATAMARI_ASSERT(ordering_.assembly_forest.parents[child] == supernode,
                    "Incorrect child index");
    std::vector<Int>* subsample = &subsamples[child_index];

    #pragma omp task default(none)                         \
        firstprivate(child, maximum_likelihood, subsample) \
        shared(shared_state, private_states)
    OpenMPRightLookingSubtree(child, maximum_likelihood, shared_state,
                              private_states, subsample);
  }

  CATAMARI_START_TIMER(shared_state->exclusive_timers[supernode]);

  // Merge the subsamples into the current sample.
  for (const std::vector<Int>& subsample : subsamples) {
    sample->insert(sample->end(), subsample.begin(), subsample.end());
  }

  std::vector<Int> subsample;
  #pragma omp taskgroup
  OpenMPRightLookingSupernodeSample(supernode, maximum_likelihood, shared_state,
                                    private_states, &subsample);
  sample->insert(sample->end(), subsample.begin(), subsample.end());

  CATAMARI_STOP_TIMER(shared_state->inclusive_timers[supernode]);
  CATAMARI_STOP_TIMER(shared_state->exclusive_timers[supernode]);
}

template <class Field>
std::vector<Int> SupernodalHermitianDPP<Field>::OpenMPRightLookingSample(
    bool maximum_likelihood) const {
  const Int num_rows = matrix_.NumRows();
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const Int num_roots = ordering_.assembly_forest.roots.Size();

  supernodal_ldl::OpenMPFillZeros(ordering_, lower_factor_.get(),
                                  diagonal_factor_.get());

  // Initialize the factors with the input matrix.
  supernodal_ldl::OpenMPFillNonzeros(
      matrix_, ordering_, supernode_member_to_index_, lower_factor_.get(),
      diagonal_factor_.get());

  std::vector<Int> sample;
  sample.reserve(num_rows);

  supernodal_ldl::RightLookingSharedState<Field> shared_state;
  shared_state.schur_complement_buffers.Resize(num_supernodes);
  shared_state.schur_complements.Resize(num_supernodes);
#ifdef CATAMARI_ENABLE_TIMERS
  shared_state.inclusive_timers.Resize(num_supernodes);
  shared_state.exclusive_timers.Resize(num_supernodes);
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  const int max_threads = omp_get_max_threads();

  if (total_work_ < min_parallel_work_) {
    // We only need the random number generator.
    std::random_device random_device;
    PrivateState private_state;
    // TODO(Jack Poulson): Use RUNNING_ON_VALGRIND to change seeding
    // mechanism.
    private_state.generator.seed(random_device());

    for (Int root_index = 0; root_index < num_roots; ++root_index) {
      const Int root = ordering_.assembly_forest.roots[root_index];
      RightLookingSubtree(root, maximum_likelihood, &shared_state,
                          &private_state, &sample);
    }
  } else {
    std::random_device random_device;
    Buffer<PrivateState> private_states(max_threads);
    for (PrivateState& private_state : private_states) {
      private_state.ldl_state.scaled_transpose_buffer.Resize(
          max_supernode_size_ * max_supernode_size_, Field{0});
      // TODO(Jack Poulson): Use RUNNING_ON_VALGRIND to change seeding
      // mechanism.
      private_state.generator.seed(random_device());
    }

    const int old_max_threads = GetMaxBlasThreads();
    SetNumBlasThreads(1);

    // NOTE: We could alternatively avoid switch to maintaining a single, shared
    // boolean list of length 'num_rows' which flags each entry as 'in' or
    // 'out' of the sample.
    Buffer<std::vector<Int>> subsamples(num_roots);

    #pragma omp parallel
    #pragma omp single
    for (Int root_index = 0; root_index < num_roots; ++root_index) {
      const Int root = ordering_.assembly_forest.roots[root_index];
      std::vector<Int>* subsample = &subsamples[root_index];
      #pragma omp task default(none)                         \
          firstprivate(root, maximum_likelihood, subsample)  \
          shared(shared_state, private_states)
      OpenMPRightLookingSubtree(root, maximum_likelihood, &shared_state,
                                &private_states, subsample);
    }

    // Merge the subsamples into the current sample.
    for (const std::vector<Int>& subsample : subsamples) {
      sample.insert(sample.end(), subsample.begin(), subsample.end());
    }

    SetNumBlasThreads(old_max_threads);
  }

  std::sort(sample.begin(), sample.end());

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

  return sample;
}

}  // namespace catamari

#endif  // ifdef CATAMARI_OPENMP
#endif  // ifndef
        // CATAMARI_SPARSE_HERMITIAN_DPP_SUPERNODAL_RIGHT_LOOKING_OPENMP_IMPL_H_
