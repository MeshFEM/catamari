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
                                    const SupernodalDPPControl& control)
    : matrix_(matrix), ordering_(ordering), control_(control) {
#ifdef CATAMARI_OPENMP
  if (omp_get_max_threads() > 1) {
    #pragma omp parallel
    #pragma omp single
    {
      OpenMPFormSupernodes();
      OpenMPFormStructure();
    }
    return;
  }
#endif  // ifdef CATAMARI_OPENMP
  FormSupernodes();
  FormStructure();
}

template <class Field>
void SupernodalDPP<Field>::FormSupernodes() {
  // Greedily compute a supernodal partition using the original ordering.
  AssemblyForest orig_scalar_forest;
  SymmetricOrdering fund_ordering;
  fund_ordering.permutation = ordering_.permutation;
  fund_ordering.inverse_permutation = ordering_.inverse_permutation;
  scalar_ldl::LowerStructure scalar_structure;
  supernodal_ldl::FormFundamentalSupernodes(
      matrix_, ordering_, &orig_scalar_forest, &fund_ordering.supernode_sizes,
      &scalar_structure);
  OffsetScan(fund_ordering.supernode_sizes, &fund_ordering.supernode_offsets);

  Buffer<Int> fund_member_to_index;
  supernodal_ldl::MemberToIndex(matrix_.NumRows(),
                                fund_ordering.supernode_offsets,
                                &fund_member_to_index);

  const Int num_fund_supernodes = fund_ordering.supernode_sizes.Size();
  supernodal_ldl::ConvertFromScalarToSupernodalEliminationForest(
      num_fund_supernodes, orig_scalar_forest.parents, fund_member_to_index,
      &fund_ordering.assembly_forest.parents);
  fund_ordering.assembly_forest.FillFromParents();

  Buffer<Int> fund_supernode_degrees;
  supernodal_ldl::SupernodalDegrees(matrix_, fund_ordering, orig_scalar_forest,
                                    fund_member_to_index,
                                    &fund_supernode_degrees);

  if (control_.relaxation_control.relax_supernodes) {
    supernodal_ldl::RelaxSupernodes(
        orig_scalar_forest.parents, fund_ordering.supernode_sizes,
        fund_ordering.supernode_offsets, fund_ordering.assembly_forest.parents,
        fund_supernode_degrees, fund_member_to_index, scalar_structure,
        control_.relaxation_control, &ordering_.permutation,
        &ordering_.inverse_permutation, &forest_.parents,
        &ordering_.assembly_forest.parents, &supernode_degrees_,
        &ordering_.supernode_sizes, &ordering_.supernode_offsets,
        &supernode_member_to_index_);
    forest_.FillFromParents();
    ordering_.assembly_forest.FillFromParents();
  } else {
    forest_ = orig_scalar_forest;

    ordering_.supernode_sizes = fund_ordering.supernode_sizes;
    ordering_.supernode_offsets = fund_ordering.supernode_offsets;
    ordering_.assembly_forest.parents = fund_ordering.assembly_forest.parents;
    ordering_.assembly_forest.FillFromParents();

    supernode_degrees_ = fund_supernode_degrees;
    supernode_member_to_index_ = fund_member_to_index;
  }
}

template <class Field>
void SupernodalDPP<Field>::FormStructure() {
  CATAMARI_ASSERT(supernode_degrees_.Size() == ordering_.supernode_sizes.Size(),
                  "Invalid supernode degrees size.");

  lower_factor_.reset(new supernodal_ldl::LowerFactor<Field>(
      ordering_.supernode_sizes, supernode_degrees_));
  diagonal_factor_.reset(
      new supernodal_ldl::DiagonalFactor<Field>(ordering_.supernode_sizes));

  max_supernode_size_ = *std::max_element(ordering_.supernode_sizes.begin(),
                                          ordering_.supernode_sizes.end());

  supernodal_ldl::FillStructureIndices(matrix_, ordering_, forest_,
                                       supernode_member_to_index_,
                                       lower_factor_.get());

  if (control_.algorithm == kLeftLookingLDL) {
    lower_factor_->FillIntersectionSizes(ordering_.supernode_sizes,
                                         supernode_member_to_index_);
  }
}

template <class Field>
std::vector<Int> SupernodalDPP<Field>::Sample(bool maximum_likelihood) const {
  if (control_.algorithm == kLeftLookingLDL) {
#ifdef CATAMARI_OPENMP
    if (omp_get_max_threads() > 1) {
      return OpenMPLeftLookingSample(maximum_likelihood);
    }
#endif  // ifdef CATAMARI_OPENMP
    return LeftLookingSample(maximum_likelihood);
  } else {
#ifdef CATAMARI_OPENMP
    if (omp_get_max_threads() > 1) {
      return OpenMPRightLookingSample(maximum_likelihood);
    }
#endif
    return RightLookingSample(maximum_likelihood);
  }
}

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

template <typename Field>
void SupernodalDPP<Field>::AppendSupernodeSample(
    Int supernode, const std::vector<Int>& supernode_sample,
    std::vector<Int>* sample) const {
  const Int supernode_start = ordering_.supernode_offsets[supernode];
  for (const Int& index : supernode_sample) {
    const Int orig_row = supernode_start + index;
    if (ordering_.inverse_permutation.Empty()) {
      sample->push_back(orig_row);
    } else {
      sample->push_back(ordering_.inverse_permutation[orig_row]);
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
    LeftLookingSupernodeUpdate(supernode, &shared_state, &private_state);
    LeftLookingSupernodeSample(supernode, maximum_likelihood, &private_state,
                               &sample);
  }

  std::sort(sample.begin(), sample.end());

  return sample;
}

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

  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    CATAMARI_ASSERT(ordering_.assembly_forest.parents[child] == supernode,
                    "Incorrect child index");
    RightLookingSubtree(child, maximum_likelihood, shared_state, private_state,
                        sample);
  }

  std::vector<Int> subsample;
  RightLookingSupernodeSample(supernode, maximum_likelihood, shared_state,
                              private_state, &subsample);
  sample->insert(sample->end(), subsample.begin(), subsample.end());
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

  return sample;
}

}  // namespace catamari

#include "catamari/dpp/supernodal_dpp/openmp-impl.hpp"

#endif  // ifndef CATAMARI_SUPERNODAL_DPP_IMPL_H_
