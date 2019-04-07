/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_HERMITIAN_DPP_SUPERNODAL_LEFT_LOOKING_OPENMP_IMPL_H_
#define CATAMARI_SPARSE_HERMITIAN_DPP_SUPERNODAL_LEFT_LOOKING_OPENMP_IMPL_H_
#ifdef CATAMARI_OPENMP

#include <algorithm>

#include "catamari/io_utils.hpp"

#include "catamari/sparse_hermitian_dpp/supernodal.hpp"

namespace catamari {

template <class Field>
void SupernodalHermitianDPP<Field>::OpenMPLeftLookingSupernodeUpdate(
    Int main_supernode, supernodal_ldl::LeftLookingSharedState* shared_state,
    Buffer<PrivateState>* private_states) const {
  const SymmetricFactorizationType factorization_type =
      kLDLAdjointFactorization;

  BlasMatrixView<Field> main_diagonal_block =
      diagonal_factor_->blocks[main_supernode];
  BlasMatrixView<Field> main_lower_block =
      lower_factor_->blocks[main_supernode];
  const Int main_supernode_size = ordering_.supernode_sizes[main_supernode];

  shared_state->rel_rows[main_supernode] = 0;
  shared_state->intersect_ptrs[main_supernode] =
      lower_factor_->IntersectionSizesBeg(main_supernode);

  const int main_thread = omp_get_thread_num();
  Buffer<Int>& pattern_flags =
      (*private_states)[main_thread].ldl_state.pattern_flags;
  Buffer<Int>& row_structure =
      (*private_states)[main_thread].ldl_state.row_structure;

  // Compute the supernodal row pattern.
  pattern_flags[main_supernode] = main_supernode;
  const Int num_packed = supernodal_ldl::ComputeRowPattern(
      matrix_, ordering_.permutation, ordering_.inverse_permutation,
      ordering_.supernode_sizes, ordering_.supernode_offsets,
      supernode_member_to_index_, ordering_.assembly_forest.parents,
      main_supernode, pattern_flags.Data(), row_structure.Data());

  // OpenMP pragmas cannot operate on object members or function results.
  const Buffer<Int>& supernode_offsets_ref = ordering_.supernode_offsets;
  const Buffer<Int>& supernode_member_to_index_ref = supernode_member_to_index_;
  supernodal_ldl::LowerFactor<Field>* const lower_factor_ptr =
      lower_factor_.get();
  Field* const main_diagonal_block_data CATAMARI_UNUSED =
      main_diagonal_block.data;
  Field* const main_lower_block_data = main_lower_block.data;

  // for J = find(L(K, :))
  //   L(K:n, K) -= L(K:n, J) * (D(J, J) * L(K, J)')
  for (Int index = 0; index < num_packed; ++index) {
    const Int descendant_supernode = row_structure[index];
    CATAMARI_ASSERT(descendant_supernode < main_supernode,
                    "Looking into upper triangle.");
    const ConstBlasMatrixView<Field> descendant_lower_block =
        lower_factor_->blocks[descendant_supernode];
    const Int descendant_degree = descendant_lower_block.height;
    const Int descendant_supernode_size = descendant_lower_block.width;

    const Int descendant_main_rel_row =
        shared_state->rel_rows[descendant_supernode];
    const Int descendant_main_intersect_size =
        *shared_state->intersect_ptrs[descendant_supernode];

    const ConstBlasMatrixView<Field> descendant_diag_block =
        diagonal_factor_->blocks[descendant_supernode].ToConst();

    #pragma omp task default(none)                                         \
        firstprivate(index, descendant_supernode, descendant_main_rel_row, \
            descendant_main_intersect_size, descendant_lower_block,        \
            descendant_diag_block, descendant_supernode_size,              \
            private_states, main_diagonal_block, main_supernode)           \
        shared(supernode_offsets_ref)                                      \
        depend(out: main_diagonal_block_data)
    {
      const int thread = omp_get_thread_num();
      PrivateState& private_state = (*private_states)[thread];

      const ConstBlasMatrixView<Field> descendant_main_matrix =
          descendant_lower_block.Submatrix(descendant_main_rel_row, 0,
                                           descendant_main_intersect_size,
                                           descendant_supernode_size);

      // BlasMatrixView<Field> scaled_transpose;
      BlasMatrixView<Field> scaled_transpose;
      scaled_transpose.height = descendant_supernode_size;
      scaled_transpose.width = descendant_main_intersect_size;
      scaled_transpose.leading_dim = descendant_supernode_size;
      scaled_transpose.data =
          private_state.ldl_state.scaled_transpose_buffer.Data();

      supernodal_ldl::FormScaledTranspose(
          factorization_type, descendant_diag_block, descendant_main_matrix,
          &scaled_transpose);

      BlasMatrixView<Field> workspace_matrix;
      workspace_matrix.height = descendant_main_intersect_size;
      workspace_matrix.width = descendant_main_intersect_size;
      workspace_matrix.leading_dim = descendant_main_intersect_size;
      workspace_matrix.data = private_state.ldl_state.workspace_buffer.Data();

      supernodal_ldl::UpdateDiagonalBlock(
          factorization_type, supernode_offsets_ref, *lower_factor_ptr,
          main_supernode, descendant_supernode, descendant_main_rel_row,
          descendant_main_matrix, scaled_transpose.ToConst(),
          &main_diagonal_block, &workspace_matrix);
    }

    shared_state->intersect_ptrs[descendant_supernode]++;
    shared_state->rel_rows[descendant_supernode] +=
        descendant_main_intersect_size;

    // L(KNext:n, K) -= L(KNext:n, J) * (D(J, J) * L(K, J)')
    //                = L(KNext:n, J) * Z(J, K).
    const Int* descendant_active_intersect_size_beg =
        shared_state->intersect_ptrs[descendant_supernode];
    Int descendant_active_rel_row =
        shared_state->rel_rows[descendant_supernode];
    const Int* main_active_intersect_sizes =
        lower_factor_->IntersectionSizesBeg(main_supernode);
    Int main_active_rel_row = 0;
    while (descendant_active_rel_row != descendant_degree) {
      const Int descendant_active_intersect_size =
          *descendant_active_intersect_size_beg;

      supernodal_ldl::SeekForMainActiveRelativeRow(
          main_supernode, descendant_supernode, descendant_active_rel_row,
          supernode_member_to_index_, *lower_factor_, &main_active_rel_row,
          &main_active_intersect_sizes);
      const Int main_active_intersect_size = *main_active_intersect_sizes;

      #pragma omp task default(none)                                         \
          firstprivate(factorization_type, index, descendant_supernode,      \
              descendant_active_rel_row, descendant_main_rel_row,            \
              main_active_rel_row, descendant_active_intersect_size,         \
              descendant_main_intersect_size,                                \
              main_active_intersect_size, descendant_supernode_size,         \
              private_states, descendant_lower_block, descendant_diag_block, \
              main_lower_block, main_supernode)                              \
          shared(supernode_offsets_ref, supernode_member_to_index_ref)       \
          depend(out: main_lower_block_data[main_active_rel_row])
      {
        const int thread = omp_get_thread_num();
        PrivateState& private_state = (*private_states)[thread];

        const ConstBlasMatrixView<Field> descendant_active_matrix =
            descendant_lower_block.Submatrix(descendant_active_rel_row, 0,
                                             descendant_active_intersect_size,
                                             descendant_supernode_size);

        const ConstBlasMatrixView<Field> descendant_main_matrix =
            descendant_lower_block.Submatrix(descendant_main_rel_row, 0,
                                             descendant_main_intersect_size,
                                             descendant_supernode_size);

        BlasMatrixView<Field> scaled_transpose;
        scaled_transpose.height = descendant_supernode_size;
        scaled_transpose.width = descendant_main_intersect_size;
        scaled_transpose.leading_dim = descendant_supernode_size;
        scaled_transpose.data =
            private_state.ldl_state.scaled_transpose_buffer.Data();

        supernodal_ldl::FormScaledTranspose(
            factorization_type, descendant_diag_block, descendant_main_matrix,
            &scaled_transpose);

        BlasMatrixView<Field> main_active_block = main_lower_block.Submatrix(
            main_active_rel_row, 0, main_active_intersect_size,
            main_supernode_size);

        BlasMatrixView<Field> workspace_matrix;
        workspace_matrix.height = descendant_active_intersect_size;
        workspace_matrix.width = descendant_main_intersect_size;
        workspace_matrix.leading_dim = descendant_active_intersect_size;
        workspace_matrix.data = private_state.ldl_state.workspace_buffer.Data();

        supernodal_ldl::UpdateSubdiagonalBlock(
            main_supernode, descendant_supernode, main_active_rel_row,
            descendant_main_rel_row, descendant_active_rel_row,
            supernode_offsets_ref, supernode_member_to_index_ref,
            scaled_transpose.ToConst(), descendant_active_matrix,
            *lower_factor_ptr, &main_active_block, &workspace_matrix);
      }

      ++descendant_active_intersect_size_beg;
      descendant_active_rel_row += descendant_active_intersect_size;
    }
  }
}

template <class Field>
void SupernodalHermitianDPP<Field>::OpenMPLeftLookingSupernodeSample(
    Int main_supernode, bool maximum_likelihood,
    Buffer<PrivateState>* private_states, std::vector<Int>* sample) const {
  const SymmetricFactorizationType factorization_type =
      kLDLAdjointFactorization;

  BlasMatrixView<Field>& main_diagonal_block =
      diagonal_factor_->blocks[main_supernode];
  BlasMatrixView<Field>& main_lower_block =
      lower_factor_->blocks[main_supernode];

  // Sample and factor the diagonal block.
  std::vector<Int> supernode_sample;
  #pragma omp taskgroup
  {
    const int thread = omp_get_thread_num();
    PrivateState& private_state = (*private_states)[thread];
    Buffer<Field>* buffer = &private_state.ldl_state.scaled_transpose_buffer;
    supernode_sample = OpenMPSampleLowerHermitianDPP(
        control_.factor_tile_size, control_.block_size, maximum_likelihood,
        &main_diagonal_block, &private_state.generator, buffer);
  }
  AppendSupernodeSample(main_supernode, supernode_sample, sample);

  #pragma omp taskgroup
  supernodal_ldl::OpenMPSolveAgainstDiagonalBlock(
      control_.outer_product_tile_size, factorization_type,
      main_diagonal_block.ToConst(), &main_lower_block);
}

template <class Field>
void SupernodalHermitianDPP<Field>::LeftLookingSubtree(
    Int supernode, bool maximum_likelihood,
    supernodal_ldl::LeftLookingSharedState* shared_state,
    PrivateState* private_state, std::vector<Int>* sample) const {
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;

  CATAMARI_START_TIMER(shared_state->inclusive_timers[supernode]);

  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    CATAMARI_ASSERT(ordering_.assembly_forest.parents[child] == supernode,
                    "Incorrect child index");
    LeftLookingSubtree(child, maximum_likelihood, shared_state, private_state,
                       sample);
  }

  CATAMARI_START_TIMER(shared_state->exclusive_timers[supernode]);

  LeftLookingSupernodeUpdate(supernode, shared_state, private_state);

  std::vector<Int> subsample;
  LeftLookingSupernodeSample(supernode, maximum_likelihood, private_state,
                             &subsample);
  sample->insert(sample->end(), subsample.begin(), subsample.end());

  CATAMARI_STOP_TIMER(shared_state->inclusive_timers[supernode]);
  CATAMARI_STOP_TIMER(shared_state->exclusive_timers[supernode]);
}

template <class Field>
void SupernodalHermitianDPP<Field>::OpenMPLeftLookingSubtree(
    Int supernode, bool maximum_likelihood,
    supernodal_ldl::LeftLookingSharedState* shared_state,
    Buffer<PrivateState>* private_states, std::vector<Int>* sample) const {
  const double work_estimate = work_estimates_[supernode];
  if (work_estimate < min_parallel_work_) {
    const int thread = omp_get_thread_num();
    LeftLookingSubtree(supernode, maximum_likelihood, shared_state,
                       &(*private_states)[thread], sample);
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
    std::vector<Int>* subsample = &subsamples[child_index];
    #pragma omp task default(none)                                      \
        firstprivate(supernode, maximum_likelihood, child_index, child, \
            shared_state, private_states, subsample)
    OpenMPLeftLookingSubtree(child, maximum_likelihood, shared_state,
                             private_states, subsample);
  }

  CATAMARI_START_TIMER(shared_state->exclusive_timers[supernode]);

  // Merge the subsamples into the current sample.
  for (const std::vector<Int>& subsample : subsamples) {
    sample->insert(sample->end(), subsample.begin(), subsample.end());
  }

  #pragma omp taskgroup
  OpenMPLeftLookingSupernodeUpdate(supernode, shared_state, private_states);

  std::vector<Int> subsample;
  #pragma omp taskgroup
  OpenMPLeftLookingSupernodeSample(supernode, maximum_likelihood,
                                   private_states, &subsample);

  sample->insert(sample->end(), subsample.begin(), subsample.end());

  CATAMARI_STOP_TIMER(shared_state->inclusive_timers[supernode]);
  CATAMARI_STOP_TIMER(shared_state->exclusive_timers[supernode]);
}

template <class Field>
std::vector<Int> SupernodalHermitianDPP<Field>::OpenMPLeftLookingSample(
    bool maximum_likelihood) const {
  const Int num_rows = ordering_.supernode_offsets.Back();
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  const int max_threads = omp_get_max_threads();

  supernodal_ldl::OpenMPFillZeros(ordering_, lower_factor_.get(),
                                  diagonal_factor_.get());

  // Initialize the factors with the input matrix.
  supernodal_ldl::OpenMPFillNonzeros(
      matrix_, ordering_, supernode_member_to_index_, lower_factor_.get(),
      diagonal_factor_.get());

  std::vector<Int> sample;
  sample.reserve(num_rows);

  supernodal_ldl::LeftLookingSharedState shared_state;
  shared_state.rel_rows.Resize(num_supernodes);
  shared_state.intersect_ptrs.Resize(num_supernodes);
#ifdef CATAMARI_ENABLE_TIMERS
  shared_state.inclusive_timers.Resize(num_supernodes);
  shared_state.exclusive_timers.Resize(num_supernodes);
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  std::random_device random_device;
  Buffer<PrivateState> private_states(max_threads);
  for (PrivateState& private_state : private_states) {
    private_state.ldl_state.row_structure.Resize(num_supernodes);
    private_state.ldl_state.pattern_flags.Resize(num_supernodes, -1);
    private_state.ldl_state.scaled_transpose_buffer.Resize(
        max_supernode_size_ * max_supernode_size_, Field{0});
    private_state.ldl_state.workspace_buffer.Resize(
        max_supernode_size_ * (max_supernode_size_ - 1), Field{0});
    private_state.generator.seed(random_device());
  }

  if (total_work_ < min_parallel_work_) {
    std::vector<Int> subsample;
    for (Int root_index = 0; root_index < num_roots; ++root_index) {
      const Int root = ordering_.assembly_forest.roots[root_index];
      LeftLookingSubtree(root, maximum_likelihood, &shared_state,
                         &private_states[0], &sample);
    }
  } else {
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
      #pragma omp task default(none)                        \
          firstprivate(maximum_likelihood, root, subsample) \
          shared(shared_state, private_states)
      OpenMPLeftLookingSubtree(root, maximum_likelihood, &shared_state,
                               &private_states, subsample);
    }

    // Merge the subtree samples into a single sample.
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
        // CATAMARI_SPARSE_HERMITIAN_DPP_SUPERNODAL_LEFT_LOOKING_OPENMP_IMPL_H_
