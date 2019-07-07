/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_HERMITIAN_DPP_SUPERNODAL_LEFT_LOOKING_IMPL_H_
#define CATAMARI_SPARSE_HERMITIAN_DPP_SUPERNODAL_LEFT_LOOKING_IMPL_H_

#include <algorithm>

#include "catamari/dense_dpp.hpp"
#include "catamari/io_utils.hpp"

#include "catamari/sparse_hermitian_dpp/supernodal.hpp"

namespace catamari {

template <class Field>
void SupernodalHermitianDPP<Field>::LeftLookingSupernodeUpdate(
    Int supernode, supernodal_ldl::LeftLookingSharedState* shared_state,
    PrivateState* private_state) const {
  const SymmetricFactorizationType factorization_type =
      kLDLAdjointFactorization;

  BlasMatrixView<Field>& diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrixView<Field>& lower_block = lower_factor_->blocks[supernode];
  const Int supernode_size = lower_block.width;
  const Int supernode_degree = lower_block.height;
  const Int supernode_offset = ordering_.supernode_offsets[supernode];

  Int* pattern_flags = private_state->ldl_state.pattern_flags.Data();
  Int* rel_ind = private_state->ldl_state.relative_indices.Data();

  // Scatter the pattern of this supernode into pattern_flags.
  // TODO(Jack Poulson): Switch away from pointers to Int members.
  const Int* structure = lower_factor_->StructureBeg(supernode);
  for (Int i = 0; i < supernode_degree; ++i) {
    pattern_flags[structure[i]] = i;
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
    scaled_transpose.height = descendant_size;
    scaled_transpose.width = intersect_size;
    scaled_transpose.leading_dim = descendant_size;
    scaled_transpose.data =
        private_state->ldl_state.scaled_transpose_buffer.Data();
    supernodal_ldl::FormScaledTranspose(
        factorization_type, diagonal_factor_->blocks[descendant].ToConst(),
        descendant_main_matrix, &scaled_transpose);

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
      // Store the relative indices of the diagonal block.
      for (Int i_rel = 0; i_rel < intersect_size; ++i_rel) {
        const Int i = descendant_structure[i_rel];
        CATAMARI_ASSERT(
            i >= supernode_offset && i < supernode_offset + supernode_size,
            "Invalid relative diagonal block index.");
        rel_ind[i_rel] = i - supernode_offset;
      }
    }

    // Update the diagonal block.
    BlasMatrixView<Field> workspace_matrix;
    workspace_matrix.height = intersect_size;
    workspace_matrix.width = intersect_size;
    workspace_matrix.leading_dim = intersect_size;
    workspace_matrix.data = private_state->ldl_state.workspace_buffer.Data();
    if (inplace_diag_update) {
      // Apply the diagonal block update in-place.
      MatrixMultiplyLowerNormalNormal(Field{-1}, descendant_main_matrix,
                                      scaled_transpose.ToConst(), Field{1},
                                      &diagonal_block);
    } else {
      // Form the diagonal block update out-of-place.
      MatrixMultiplyLowerNormalNormal(Field{-1}, descendant_main_matrix,
                                      scaled_transpose.ToConst(), Field{0},
                                      &workspace_matrix);

      // Apply the diagonal block update.
      for (Int j_rel = 0; j_rel < intersect_size; ++j_rel) {
        const Int j = rel_ind[j_rel];
        Field* diag_col = diagonal_block.Pointer(0, j);
        const Field* workspace_col = workspace_matrix.Pointer(0, j_rel);
        for (Int i_rel = j_rel; i_rel < intersect_size; ++i_rel) {
          const Int i = rel_ind[i_rel];
          diag_col[i] += workspace_col[i_rel];
        }
      }
    }

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
        // Apply the subdiagonal block update in-place.
        MatrixMultiplyNormalNormal(Field{-1}, descendant_below_main_matrix,
                                   scaled_transpose.ToConst(), Field{1},
                                   &lower_block);
      } else {
        // Form the subdiagonal block update out-of-place.
        workspace_matrix.height = descendant_degree_remaining;
        workspace_matrix.width = intersect_size;
        workspace_matrix.leading_dim = descendant_degree_remaining;
        MatrixMultiplyNormalNormal(Field{-1}, descendant_below_main_matrix,
                                   scaled_transpose.ToConst(), Field{0},
                                   &workspace_matrix);

        // Store the relative indices on the lower structure.
        for (Int i_rel = intersect_size; i_rel < descendant_main_degree;
             ++i_rel) {
          const Int i = descendant_structure[i_rel];
          rel_ind[i_rel] = pattern_flags[i];
          CATAMARI_ASSERT(
              rel_ind[i_rel] >= 0 && rel_ind[i_rel] < supernode_degree,
              "Invalid subdiagonal relative index.");
        }

        // Apply the subdiagonal block update.
        for (Int j_rel = 0; j_rel < intersect_size; ++j_rel) {
          const Int j = rel_ind[j_rel];
          CATAMARI_ASSERT(j >= 0 && j < supernode_size,
                          "Invalid unpacked column index.");
          CATAMARI_ASSERT(j + ordering_.supernode_offsets[supernode] ==
                              descendant_structure[j_rel],
                          "Mismatched unpacked column structure.");

          Field* lower_col = lower_block.Pointer(0, j);
          const Field* workspace_col = workspace_matrix.Pointer(0, j_rel);
          for (Int i_rel = 0; i_rel < descendant_degree_remaining; ++i_rel) {
            const Int i = rel_ind[i_rel + intersect_size];
            CATAMARI_ASSERT(i >= 0 && i < supernode_degree,
                            "Invalid row index.");
            CATAMARI_ASSERT(
                structure[i] == descendant_structure[i_rel + intersect_size],
                "Mismatched row structure.");
            lower_col[i] += workspace_col[i_rel];
          }
        }
      }

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
void SupernodalHermitianDPP<Field>::LeftLookingSupernodeSample(
    Int supernode, bool maximum_likelihood, PrivateState* private_state,
    std::vector<Int>* sample) const {
  const SymmetricFactorizationType factorization_type =
      kLDLAdjointFactorization;

  BlasMatrixView<Field>& diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrixView<Field>& lower_block = lower_factor_->blocks[supernode];

  // Sample and factor the diagonal block.
  const std::vector<Int> supernode_sample =
      SampleLowerHermitianDPP(control_.block_size, maximum_likelihood,
                              &diagonal_block, &private_state->generator);
  AppendSupernodeSample(supernode, supernode_sample, sample);

  supernodal_ldl::SolveAgainstDiagonalBlock(
      factorization_type, diagonal_block.ToConst(), &lower_block);
}

template <class Field>
std::vector<Int> SupernodalHermitianDPP<Field>::LeftLookingSample(
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
  shared_state.descendants.Initialize(num_supernodes);
#ifdef CATAMARI_ENABLE_TIMERS
  shared_state.exclusive_timers.Resize(num_supernodes);
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  std::random_device random_device;
  PrivateState private_state;
  private_state.ldl_state.pattern_flags.Resize(num_rows);
  private_state.ldl_state.relative_indices.Resize(num_rows);
  private_state.ldl_state.scaled_transpose_buffer.Resize(
      left_looking_scaled_transpose_size_, Field{0});
  private_state.ldl_state.workspace_buffer.Resize(left_looking_workspace_size_,
                                                  Field{0});
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

#endif  // ifndef CATAMARI_SPARSE_HERMITIAN_DPP_SUPERNODAL_LEFT_LOOKING_IMPL_H_
