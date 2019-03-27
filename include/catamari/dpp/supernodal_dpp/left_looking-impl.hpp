/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SUPERNODAL_DPP_LEFT_LOOKING_IMPL_H_
#define CATAMARI_SUPERNODAL_DPP_LEFT_LOOKING_IMPL_H_

#include <algorithm>

#include "catamari/io_utils.hpp"

#include "catamari/dpp/supernodal_dpp.hpp"

namespace catamari {

template <class Field>
void SupernodalDPP<Field>::LeftLookingSupernodeUpdate(
    Int main_supernode, supernodal_ldl::LeftLookingSharedState* shared_state,
    PrivateState* private_state) const {
  const SymmetricFactorizationType factorization_type =
      kLDLAdjointFactorization;

  BlasMatrixView<Field>& main_diagonal_block =
      diagonal_factor_->blocks[main_supernode];
  BlasMatrixView<Field>& main_lower_block =
      lower_factor_->blocks[main_supernode];
  const Int main_supernode_size = ordering_.supernode_sizes[main_supernode];

  shared_state->rel_rows[main_supernode] = 0;
  shared_state->intersect_ptrs[main_supernode] =
      lower_factor_->IntersectionSizesBeg(main_supernode);

  // Compute the supernodal row pattern.
  private_state->ldl_state.pattern_flags[main_supernode] = main_supernode;
  const Int num_packed = supernodal_ldl::ComputeRowPattern(
      matrix_, ordering_.permutation, ordering_.inverse_permutation,
      ordering_.supernode_sizes, ordering_.supernode_offsets,
      supernode_member_to_index_, ordering_.assembly_forest.parents,
      main_supernode, private_state->ldl_state.pattern_flags.Data(),
      private_state->ldl_state.row_structure.Data());

  // for J = find(L(K, :))
  //   L(K:n, K) -= L(K:n, J) * (D(J, J) * L(K, J)')
  for (Int index = 0; index < num_packed; ++index) {
    const Int descendant_supernode =
        private_state->ldl_state.row_structure[index];
    CATAMARI_ASSERT(descendant_supernode < main_supernode,
                    "Looking into upper triangle.");
    const ConstBlasMatrixView<Field>& descendant_lower_block =
        lower_factor_->blocks[descendant_supernode];
    const Int descendant_degree = descendant_lower_block.height;
    const Int descendant_supernode_size = descendant_lower_block.width;

    const Int descendant_main_rel_row =
        shared_state->rel_rows[descendant_supernode];
    const Int descendant_main_intersect_size =
        *shared_state->intersect_ptrs[descendant_supernode];

    const ConstBlasMatrixView<Field> descendant_diag_block =
        diagonal_factor_->blocks[descendant_supernode].ToConst();

    const ConstBlasMatrixView<Field> descendant_main_matrix =
        descendant_lower_block.Submatrix(descendant_main_rel_row, 0,
                                         descendant_main_intersect_size,
                                         descendant_supernode_size);

    BlasMatrixView<Field> scaled_transpose;
    scaled_transpose.height = descendant_supernode_size;
    scaled_transpose.width = descendant_main_intersect_size;
    scaled_transpose.leading_dim = descendant_supernode_size;
    scaled_transpose.data =
        private_state->ldl_state.scaled_transpose_buffer.Data();

    supernodal_ldl::FormScaledTranspose(
        factorization_type, descendant_diag_block, descendant_main_matrix,
        &scaled_transpose);

    BlasMatrixView<Field> workspace_matrix;
    workspace_matrix.height = descendant_main_intersect_size;
    workspace_matrix.width = descendant_main_intersect_size;
    workspace_matrix.leading_dim = descendant_main_intersect_size;
    workspace_matrix.data = private_state->ldl_state.workspace_buffer.Data();

    supernodal_ldl::UpdateDiagonalBlock(
        factorization_type, ordering_.supernode_offsets, *lower_factor_,
        main_supernode, descendant_supernode, descendant_main_rel_row,
        descendant_main_matrix, scaled_transpose.ToConst(),
        &main_diagonal_block, &workspace_matrix);

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

      const ConstBlasMatrixView<Field> descendant_active_matrix =
          descendant_lower_block.Submatrix(descendant_active_rel_row, 0,
                                           descendant_active_intersect_size,
                                           descendant_supernode_size);

      // The width of the workspace matrix and pointer are already correct.
      workspace_matrix.height = descendant_active_intersect_size;
      workspace_matrix.leading_dim = descendant_active_intersect_size;

      supernodal_ldl::SeekForMainActiveRelativeRow(
          main_supernode, descendant_supernode, descendant_active_rel_row,
          supernode_member_to_index_, *lower_factor_, &main_active_rel_row,
          &main_active_intersect_sizes);
      const Int main_active_intersect_size = *main_active_intersect_sizes;

      BlasMatrixView<Field> main_active_block = main_lower_block.Submatrix(
          main_active_rel_row, 0, main_active_intersect_size,
          main_supernode_size);

      supernodal_ldl::UpdateSubdiagonalBlock(
          main_supernode, descendant_supernode, main_active_rel_row,
          descendant_main_rel_row, descendant_active_rel_row,
          ordering_.supernode_offsets, supernode_member_to_index_,
          scaled_transpose.ToConst(), descendant_active_matrix, *lower_factor_,
          &main_active_block, &workspace_matrix);

      ++descendant_active_intersect_size_beg;
      descendant_active_rel_row += descendant_active_intersect_size;
    }
  }
}

template <class Field>
void SupernodalDPP<Field>::LeftLookingSupernodeSample(
    Int supernode, bool maximum_likelihood, PrivateState* private_state,
    std::vector<Int>* sample) const {
  const SymmetricFactorizationType factorization_type =
      kLDLAdjointFactorization;

  BlasMatrixView<Field>& diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrixView<Field>& lower_block = lower_factor_->blocks[supernode];

  // Sample and factor the diagonal block.
  const std::vector<Int> supernode_sample =
      LowerFactorAndSampleDPP(control_.block_size, maximum_likelihood,
                              &diagonal_block, &private_state->generator);
  AppendSupernodeSample(supernode, supernode_sample, sample);

  supernodal_ldl::SolveAgainstDiagonalBlock(
      factorization_type, diagonal_block.ToConst(), &lower_block);
}

template <class Field>
std::vector<Int> SupernodalDPP<Field>::LeftLookingSample(
    bool maximum_likelihood) const {
  const Int num_rows = ordering_.supernode_offsets.Back();
  const Int num_supernodes = ordering_.supernode_sizes.Size();

  supernodal_ldl::FillZeros(ordering_, lower_factor_.get(),
                            diagonal_factor_.get());

  // Initialize the factors with the input matrix.
  supernodal_ldl::FillNonzeros(matrix_, ordering_, supernode_member_to_index_,
                               lower_factor_.get(), diagonal_factor_.get());

  std::vector<Int> sample;
  sample.reserve(num_rows);

  supernodal_ldl::LeftLookingSharedState shared_state;
  shared_state.rel_rows.Resize(num_supernodes);
  shared_state.intersect_ptrs.Resize(num_supernodes);
#ifdef CATAMARI_ENABLE_TIMERS
  shared_state.exclusive_timers.Resize(num_supernodes);
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  std::random_device random_device;
  PrivateState private_state;
  private_state.ldl_state.row_structure.Resize(num_supernodes);
  private_state.ldl_state.pattern_flags.Resize(num_supernodes);
  private_state.ldl_state.scaled_transpose_buffer.Resize(
      max_supernode_size_ * max_supernode_size_, Field{0});
  private_state.ldl_state.workspace_buffer.Resize(
      max_supernode_size_ * (max_supernode_size_ - 1), Field{0});
  private_state.generator.seed(random_device());

  // Note that any postordering of the supernodal elimination forest suffices.
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    CATAMARI_START_TIMER(shared_state.exclusive_timers[supernode]);
    LeftLookingSupernodeUpdate(supernode, &shared_state, &private_state);
    LeftLookingSupernodeSample(supernode, maximum_likelihood, &private_state,
                               &sample);
    CATAMARI_STOP_TIMER(shared_state.exclusive_timers[supernode]);
  }

  std::sort(sample.begin(), sample.end());

#ifdef CATAMARI_ENABLE_TIMERS
  TruncatedForestTimersToDot(
      control_.exclusive_timings_filename, shared_state.exclusive_timers,
      ordering_.assembly_forest, control_.max_timing_levels,
      control_.avoid_timing_isolated_roots);
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  return sample;
}

}  // namespace catamari

#endif  // ifndef CATAMARI_SUPERNODAL_DPP_LEFT_LOOKING_IMPL_H_
