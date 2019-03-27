/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SUPERNODAL_DPP_RIGHT_LOOKING_IMPL_H_
#define CATAMARI_SUPERNODAL_DPP_RIGHT_LOOKING_IMPL_H_

#include <algorithm>

#include "catamari/io_utils.hpp"

#include "catamari/dpp/supernodal_dpp.hpp"

namespace catamari {

template <class Field>
void SupernodalDPP<Field>::RightLookingSupernodeSample(
    Int supernode, bool maximum_likelihood,
    supernodal_ldl::RightLookingSharedState<Field>* shared_state,
    PrivateState* private_state, std::vector<Int>* sample) const {
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

  supernodal_ldl::MergeChildSchurComplements(
      supernode, ordering_, lower_factor_.get(), diagonal_factor_.get(),
      shared_state);

  // Sample and factor the diagonal block.
  const std::vector<Int> supernode_sample =
      LowerFactorAndSampleDPP(control_.block_size, maximum_likelihood,
                              &diagonal_block, &private_state->generator);
  AppendSupernodeSample(supernode, supernode_sample, sample);

  if (!degree) {
    // We can early exit.
    return;
  }

  CATAMARI_ASSERT(supernode_size > 0, "Supernode size was non-positive.");
  const SymmetricFactorizationType factorization_type =
      kLDLAdjointFactorization;
  supernodal_ldl::SolveAgainstDiagonalBlock(
      factorization_type, diagonal_block.ToConst(), &lower_block);

  Buffer<Field> scaled_transpose_buffer(degree * supernode_size);
  BlasMatrixView<Field> scaled_transpose;
  scaled_transpose.height = supernode_size;
  scaled_transpose.width = degree;
  scaled_transpose.leading_dim = supernode_size;
  scaled_transpose.data = scaled_transpose_buffer.Data();
  supernodal_ldl::FormScaledTranspose(factorization_type,
                                      diagonal_block.ToConst(),
                                      lower_block.ToConst(), &scaled_transpose);

  MatrixMultiplyLowerNormalNormal(Field{-1}, lower_block.ToConst(),
                                  scaled_transpose.ToConst(), Field{1},
                                  &schur_complement);
}

template <class Field>
void SupernodalDPP<Field>::RightLookingSubtree(
    Int supernode, bool maximum_likelihood,
    supernodal_ldl::RightLookingSharedState<Field>* shared_state,
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
    RightLookingSubtree(child, maximum_likelihood, shared_state, private_state,
                        sample);
  }

  CATAMARI_START_TIMER(shared_state->exclusive_timers[supernode]);

  std::vector<Int> subsample;
  RightLookingSupernodeSample(supernode, maximum_likelihood, shared_state,
                              private_state, &subsample);
  sample->insert(sample->end(), subsample.begin(), subsample.end());

  CATAMARI_STOP_TIMER(shared_state->inclusive_timers[supernode]);
  CATAMARI_STOP_TIMER(shared_state->exclusive_timers[supernode]);
}

template <class Field>
std::vector<Int> SupernodalDPP<Field>::RightLookingSample(
    bool maximum_likelihood) const {
  const Int num_rows = matrix_.NumRows();
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const Int num_roots = ordering_.assembly_forest.roots.Size();

  supernodal_ldl::FillZeros(ordering_, lower_factor_.get(),
                            diagonal_factor_.get());

  // Initialize the factors with the input matrix.
  supernodal_ldl::FillNonzeros(matrix_, ordering_, supernode_member_to_index_,
                               lower_factor_.get(), diagonal_factor_.get());

  std::vector<Int> sample;
  sample.reserve(num_rows);

  supernodal_ldl::RightLookingSharedState<Field> shared_state;
  shared_state.schur_complement_buffers.Resize(num_supernodes);
  shared_state.schur_complements.Resize(num_supernodes);
#ifdef CATAMARI_ENABLE_TIMERS
  shared_state.inclusive_timers.Resize(num_supernodes);
  shared_state.exclusive_timers.Resize(num_supernodes);
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  // We only need the random number generator.
  std::random_device random_device;
  PrivateState private_state;
  private_state.generator.seed(random_device());

  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = ordering_.assembly_forest.roots[root_index];
    RightLookingSubtree(root, maximum_likelihood, &shared_state, &private_state,
                        &sample);
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

#endif  // ifndef CATAMARI_SUPERNODAL_DPP_RIGHT_LOOKING_IMPL_H_
