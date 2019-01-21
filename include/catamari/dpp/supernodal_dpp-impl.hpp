/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SUPERNODAL_DPP_IMPL_H_
#define CATAMARI_SUPERNODAL_DPP_IMPL_H_

#include <algorithm>

#include "catamari/dpp/supernodal_dpp.hpp"

namespace catamari {

template <class Field>
SupernodalDPP<Field>::SupernodalDPP(const CoordinateMatrix<Field>& matrix,
                                    const SymmetricOrdering& ordering,
                                    const SupernodalDPPControl& control,
                                    unsigned int random_seed)
    : matrix_(matrix),
      ordering_(ordering),
      control_(control),
      generator_(random_seed),
      unit_uniform_(ComplexBase<Field>{0}, ComplexBase<Field>{1}) {
  FormSupernodes();
  FormStructure();
}

template <class Field>
void SupernodalDPP<Field>::FormSupernodes() {
  std::vector<Int> orig_parents, orig_degrees;
  scalar_ldl::EliminationForestAndDegrees(matrix_, ordering_, &orig_parents,
                                          &orig_degrees);

  std::vector<Int> orig_supernode_sizes;
  scalar_ldl::LowerStructure scalar_structure;
  supernodal_ldl::FormFundamentalSupernodes(matrix_, ordering_, orig_parents,
                                            orig_degrees, &orig_supernode_sizes,
                                            &scalar_structure);

  std::vector<Int> orig_supernode_starts;
  OffsetScan(orig_supernode_sizes, &orig_supernode_starts);

  std::vector<Int> orig_member_to_index;
  supernodal_ldl::MemberToIndex(matrix_.NumRows(), orig_supernode_starts,
                                &orig_member_to_index);

  std::vector<Int> orig_supernode_degrees;
  supernodal_ldl::SupernodalDegrees(
      matrix_, ordering_.permutation, ordering_.inverse_permutation,
      orig_supernode_sizes, orig_supernode_starts, orig_member_to_index,
      orig_parents, &orig_supernode_degrees);

  const Int num_orig_supernodes = orig_supernode_sizes.size();
  std::vector<Int> orig_supernode_parents;
  supernodal_ldl::ConvertFromScalarToSupernodalEliminationForest(
      num_orig_supernodes, orig_parents, orig_member_to_index,
      &orig_supernode_parents);

  if (control_.relaxation_control.relax_supernodes) {
    supernodal_ldl::RelaxSupernodes(
        orig_parents, orig_supernode_sizes, orig_supernode_starts,
        orig_supernode_parents, orig_supernode_degrees, orig_member_to_index,
        scalar_structure, control_.relaxation_control, &ordering_.permutation,
        &ordering_.inverse_permutation, &parents_,
        &ordering_.assembly_forest.parents, &supernode_degrees_,
        &ordering_.supernode_sizes, &ordering_.supernode_offsets,
        &supernode_member_to_index_);
  } else {
    parents_ = orig_parents;
    supernode_degrees_ = orig_supernode_degrees;

    ordering_.supernode_sizes = orig_supernode_sizes;
    ordering_.supernode_offsets = orig_supernode_starts;
    ordering_.assembly_forest.parents = orig_supernode_parents;
    ordering_.assembly_forest.FillFromParents();

    supernode_member_to_index_ = orig_member_to_index;
  }
}

template <class Field>
void SupernodalDPP<Field>::FormStructure() {
  CATAMARI_ASSERT(supernode_degrees_.size() == ordering_.supernode_sizes.size(),
                  "Invalid supernode degrees size.");

  lower_factor_.reset(new supernodal_ldl::LowerFactor<Field>(
      ordering_.supernode_sizes, supernode_degrees_));
  diagonal_factor_.reset(
      new supernodal_ldl::DiagonalFactor<Field>(ordering_.supernode_sizes));

  supernodal_ldl::FillStructureIndices(
      matrix_, ordering_, parents_, ordering_.supernode_sizes,
      supernode_member_to_index_, lower_factor_.get());

  // TODO(Jack Poulson): Do not compute this for right-looking factorizations
  // (once support is added).
  lower_factor_->FillIntersectionSizes(ordering_.supernode_sizes,
                                       supernode_member_to_index_);

  max_supernode_size_ = *std::max_element(ordering_.supernode_sizes.begin(),
                                          ordering_.supernode_sizes.end());
}

template <class Field>
std::vector<Int> SupernodalDPP<Field>::Sample(bool maximum_likelihood) const {
  return LeftLookingSample(maximum_likelihood);
}

template <class Field>
void SupernodalDPP<Field>::LeftLookingSupernodeUpdate(
    Int main_supernode, LeftLookingSampleState* state) const {
  const SymmetricFactorizationType factorization_type =
      kLDLAdjointFactorization;

  BlasMatrix<Field>& main_diagonal_block =
      diagonal_factor_->blocks[main_supernode];
  BlasMatrix<Field>& main_lower_block = lower_factor_->blocks[main_supernode];
  const Int main_supernode_size = ordering_.supernode_sizes[main_supernode];

  state->pattern_flags[main_supernode] = main_supernode;

  state->rel_rows[main_supernode] = 0;
  state->intersect_ptrs[main_supernode] =
      lower_factor_->IntersectionSizes(main_supernode);

  // Compute the supernodal row pattern.
  const Int num_packed = supernodal_ldl::ComputeRowPattern(
      matrix_, ordering_.permutation, ordering_.inverse_permutation,
      ordering_.supernode_sizes, ordering_.supernode_offsets,
      supernode_member_to_index_, ordering_.assembly_forest.parents,
      main_supernode, state->pattern_flags.data(), state->row_structure.data());

  // for J = find(L(K, :))
  //   L(K:n, K) -= L(K:n, J) * (D(J, J) * L(K, J)')
  for (Int index = 0; index < num_packed; ++index) {
    const Int descendant_supernode = state->row_structure[index];
    CATAMARI_ASSERT(descendant_supernode < main_supernode,
                    "Looking into upper triangle.");
    const ConstBlasMatrix<Field>& descendant_lower_block =
        lower_factor_->blocks[descendant_supernode];
    const Int descendant_degree = descendant_lower_block.height;
    const Int descendant_supernode_size = descendant_lower_block.width;

    const Int descendant_main_intersect_size =
        *state->intersect_ptrs[descendant_supernode];

    const Int descendant_main_rel_row = state->rel_rows[descendant_supernode];
    ConstBlasMatrix<Field> descendant_main_matrix =
        descendant_lower_block.Submatrix(descendant_main_rel_row, 0,
                                         descendant_main_intersect_size,
                                         descendant_supernode_size);

    BlasMatrix<Field> scaled_transpose;
    scaled_transpose.height = descendant_supernode_size;
    scaled_transpose.width = descendant_main_intersect_size;
    scaled_transpose.leading_dim = descendant_supernode_size;
    scaled_transpose.data = state->scaled_transpose_buffer.data();

    supernodal_ldl::FormScaledTranspose(
        factorization_type,
        diagonal_factor_->blocks[descendant_supernode].ToConst(),
        descendant_main_matrix, &scaled_transpose);

    BlasMatrix<Field> workspace_matrix;
    workspace_matrix.height = descendant_main_intersect_size;
    workspace_matrix.width = descendant_main_intersect_size;
    workspace_matrix.leading_dim = descendant_main_intersect_size;
    workspace_matrix.data = state->workspace_buffer.data();

    supernodal_ldl::UpdateDiagonalBlock(
        factorization_type, ordering_.supernode_offsets, *lower_factor_,
        main_supernode, descendant_supernode, descendant_main_rel_row,
        descendant_main_matrix, scaled_transpose.ToConst(),
        &main_diagonal_block, &workspace_matrix);

    state->intersect_ptrs[descendant_supernode]++;
    state->rel_rows[descendant_supernode] += descendant_main_intersect_size;

    // L(KNext:n, K) -= L(KNext:n, J) * (D(J, J) * L(K, J)')
    //                = L(KNext:n, J) * Z(J, K).
    const Int* descendant_active_intersect_size_beg =
        state->intersect_ptrs[descendant_supernode];
    Int descendant_active_rel_row = state->rel_rows[descendant_supernode];
    const Int* main_active_intersect_sizes =
        lower_factor_->IntersectionSizes(main_supernode);
    Int main_active_rel_row = 0;
    while (descendant_active_rel_row != descendant_degree) {
      const Int descendant_active_intersect_size =
          *descendant_active_intersect_size_beg;

      ConstBlasMatrix<Field> descendant_active_matrix =
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

      BlasMatrix<Field> main_active_block = main_lower_block.Submatrix(
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
    Int main_supernode, bool maximum_likelihood,
    std::vector<Int>* sample) const {
  const SymmetricFactorizationType factorization_type =
      kLDLAdjointFactorization;

  BlasMatrix<Field>& main_diagonal_block =
      diagonal_factor_->blocks[main_supernode];
  BlasMatrix<Field>& main_lower_block = lower_factor_->blocks[main_supernode];

  // Sample and factor the diagonal block.
  const Int main_supernode_start = ordering_.supernode_offsets[main_supernode];
  const std::vector<Int> supernode_sample = LowerFactorAndSampleDPP(
      maximum_likelihood, &main_diagonal_block, &generator_, &unit_uniform_);
  for (const Int& index : supernode_sample) {
    const Int orig_row = main_supernode_start + index;
    if (ordering_.inverse_permutation.empty()) {
      sample->push_back(orig_row);
    } else {
      sample->push_back(ordering_.inverse_permutation[orig_row]);
    }
  }

  supernodal_ldl::SolveAgainstDiagonalBlock(
      factorization_type, main_diagonal_block.ToConst(), &main_lower_block);
}

template <class Field>
std::vector<Int> SupernodalDPP<Field>::LeftLookingSample(
    bool maximum_likelihood) const {
  const Int num_rows = ordering_.supernode_offsets.back();
  const Int num_supernodes = ordering_.supernode_sizes.size();

  // Reset the lower factor to all zeros.
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    BlasMatrix<Field>& matrix = lower_factor_->blocks[supernode];
    std::fill(matrix.data, matrix.data + matrix.leading_dim * matrix.width,
              Field{0});
  }

  // Reset the diagonal factor to all zeros.
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    BlasMatrix<Field>& matrix = diagonal_factor_->blocks[supernode];
    std::fill(matrix.data, matrix.data + matrix.leading_dim * matrix.width,
              Field{0});
  }

  // Initialize the factors with the input matrix.
  supernodal_ldl::FillNonzeros(
      matrix_, ordering_.permutation, ordering_.inverse_permutation,
      ordering_.supernode_offsets, ordering_.supernode_sizes,
      supernode_member_to_index_, lower_factor_.get(), diagonal_factor_.get());

  std::vector<Int> sample;
  sample.reserve(num_rows);

  LeftLookingSampleState state;
  state.row_structure.resize(num_supernodes);
  state.pattern_flags.resize(num_supernodes);
  state.rel_rows.resize(num_supernodes);
  state.intersect_ptrs.resize(num_supernodes);
  state.scaled_transpose_buffer.resize(
      max_supernode_size_ * max_supernode_size_, Field{0});
  state.workspace_buffer.resize(max_supernode_size_ * (max_supernode_size_ - 1),
                                Field{0});

  // Note that any postordering of the supernodal elimination forest suffices.
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    LeftLookingSupernodeUpdate(supernode, &state);
    LeftLookingSupernodeSample(supernode, maximum_likelihood, &sample);
  }

  std::sort(sample.begin(), sample.end());

  return sample;
}

}  // namespace catamari

#endif  // ifndef CATAMARI_SUPERNODAL_DPP_IMPL_H_
