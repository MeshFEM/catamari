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
#ifdef _OPENMP
  if (omp_get_max_threads() > 1) {
    MultithreadedFormSupernodes();
    MultithreadedFormStructure();
    return;
  }
#endif  // ifdef _OPENMP
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

#ifdef _OPENMP
template <class Field>
void SupernodalDPP<Field>::MultithreadedFormSupernodes() {
  // Greedily compute a supernodal partition using the original ordering.
  AssemblyForest orig_scalar_forest;
  SymmetricOrdering fund_ordering;
  fund_ordering.permutation = ordering_.permutation;
  fund_ordering.inverse_permutation = ordering_.inverse_permutation;
  scalar_ldl::LowerStructure scalar_structure;
  supernodal_ldl::MultithreadedFormFundamentalSupernodes(
      matrix_, ordering_, &orig_scalar_forest, &fund_ordering.supernode_sizes,
      &scalar_structure);
  OffsetScan(fund_ordering.supernode_sizes, &fund_ordering.supernode_offsets);
  CATAMARI_ASSERT(fund_ordering.supernode_offsets.Back() == matrix_.NumRows(),
                  "Supernodes did not sum to the matrix size.");

  Buffer<Int> fund_member_to_index;
  supernodal_ldl::MemberToIndex(matrix_.NumRows(),
                                fund_ordering.supernode_offsets,
                                &fund_member_to_index);

  // TODO(Jack Poulson): Parallelize
  //     ConvertFromScalarToSupernodalEliminationForest.
  const Int num_fund_supernodes = fund_ordering.supernode_sizes.Size();
  Buffer<Int> fund_supernode_parents;
  supernodal_ldl::ConvertFromScalarToSupernodalEliminationForest(
      num_fund_supernodes, orig_scalar_forest.parents, fund_member_to_index,
      &fund_ordering.assembly_forest.parents);
  fund_ordering.assembly_forest.FillFromParents();

  Buffer<Int> fund_supernode_degrees;
  supernodal_ldl::MultithreadedSupernodalDegrees(
      matrix_, fund_ordering, orig_scalar_forest, fund_member_to_index,
      &fund_supernode_degrees);

  if (control_.relaxation_control.relax_supernodes) {
    // TODO(Jack Poulson): Parallelize RelaxSupernodes.
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

    supernode_member_to_index_ = fund_member_to_index;
    supernode_degrees_ = fund_supernode_degrees;
  }
}
#endif  // ifdef _OPENMP

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

#ifdef _OPENMP
template <class Field>
void SupernodalDPP<Field>::MultithreadedFormStructure() {
  CATAMARI_ASSERT(supernode_degrees_.Size() == ordering_.supernode_sizes.Size(),
                  "Invalid supernode degrees size.");

  lower_factor_.reset(new supernodal_ldl::LowerFactor<Field>(
      ordering_.supernode_sizes, supernode_degrees_));
  diagonal_factor_.reset(
      new supernodal_ldl::DiagonalFactor<Field>(ordering_.supernode_sizes));

  max_supernode_size_ = *std::max_element(ordering_.supernode_sizes.begin(),
                                          ordering_.supernode_sizes.end());

  supernodal_ldl::MultithreadedFillStructureIndices(
      control_.sort_grain_size, matrix_, ordering_, forest_,
      supernode_member_to_index_, lower_factor_.get());

  if (control_.algorithm == kLeftLookingLDL) {
    // TODO(Jack Poulson): Switch to a multithreaded equivalent.
    lower_factor_->FillIntersectionSizes(ordering_.supernode_sizes,
                                         supernode_member_to_index_);
  }
}
#endif  // ifdef _OPENMP

template <class Field>
std::vector<Int> SupernodalDPP<Field>::Sample(bool maximum_likelihood) const {
  if (control_.algorithm == kLeftLookingLDL) {
#ifdef _OPENMP
    if (omp_get_max_threads() > 1) {
      return MultithreadedLeftLookingSample(maximum_likelihood);
    }
#endif  // ifdef _OPENMP
    return LeftLookingSample(maximum_likelihood);
  } else {
#ifdef _OPENMP
    if (omp_get_max_threads() > 1) {
      return MultithreadedRightLookingSample(maximum_likelihood);
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

  BlasMatrix<Field>& main_diagonal_block =
      diagonal_factor_->blocks[main_supernode];
  BlasMatrix<Field>& main_lower_block = lower_factor_->blocks[main_supernode];
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
    const ConstBlasMatrix<Field>& descendant_lower_block =
        lower_factor_->blocks[descendant_supernode];
    const Int descendant_degree = descendant_lower_block.height;
    const Int descendant_supernode_size = descendant_lower_block.width;

    const Int descendant_main_rel_row =
        shared_state->rel_rows[descendant_supernode];
    const Int descendant_main_intersect_size =
        *shared_state->intersect_ptrs[descendant_supernode];

    const ConstBlasMatrix<Field> descendant_diag_block =
        diagonal_factor_->blocks[descendant_supernode].ToConst();

    const ConstBlasMatrix<Field> descendant_main_matrix =
        descendant_lower_block.Submatrix(descendant_main_rel_row, 0,
                                         descendant_main_intersect_size,
                                         descendant_supernode_size);

    BlasMatrix<Field> scaled_transpose;
    scaled_transpose.height = descendant_supernode_size;
    scaled_transpose.width = descendant_main_intersect_size;
    scaled_transpose.leading_dim = descendant_supernode_size;
    scaled_transpose.data =
        private_state->ldl_state.scaled_transpose_buffer.Data();

    supernodal_ldl::FormScaledTranspose(
        factorization_type, descendant_diag_block, descendant_main_matrix,
        &scaled_transpose);

    BlasMatrix<Field> workspace_matrix;
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

      const ConstBlasMatrix<Field> descendant_active_matrix =
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

#ifdef _OPENMP
template <class Field>
void SupernodalDPP<Field>::MultithreadedLeftLookingSupernodeUpdate(
    Int main_supernode, supernodal_ldl::LeftLookingSharedState* shared_state,
    Buffer<PrivateState>* private_states) const {
  const SymmetricFactorizationType factorization_type =
      kLDLAdjointFactorization;

  BlasMatrix<Field> main_diagonal_block =
      diagonal_factor_->blocks[main_supernode];
  BlasMatrix<Field> main_lower_block = lower_factor_->blocks[main_supernode];
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
    const ConstBlasMatrix<Field> descendant_lower_block =
        lower_factor_->blocks[descendant_supernode];
    const Int descendant_degree = descendant_lower_block.height;
    const Int descendant_supernode_size = descendant_lower_block.width;

    const Int descendant_main_rel_row =
        shared_state->rel_rows[descendant_supernode];
    const Int descendant_main_intersect_size =
        *shared_state->intersect_ptrs[descendant_supernode];

    const ConstBlasMatrix<Field> descendant_diag_block =
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

      const ConstBlasMatrix<Field> descendant_main_matrix =
          descendant_lower_block.Submatrix(descendant_main_rel_row, 0,
                                           descendant_main_intersect_size,
                                           descendant_supernode_size);

      // BlasMatrix<Field> scaled_transpose;
      BlasMatrix<Field> scaled_transpose;
      scaled_transpose.height = descendant_supernode_size;
      scaled_transpose.width = descendant_main_intersect_size;
      scaled_transpose.leading_dim = descendant_supernode_size;
      scaled_transpose.data =
          private_state.ldl_state.scaled_transpose_buffer.Data();

      supernodal_ldl::FormScaledTranspose(
          factorization_type, descendant_diag_block, descendant_main_matrix,
          &scaled_transpose);

      BlasMatrix<Field> workspace_matrix;
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

        const ConstBlasMatrix<Field> descendant_active_matrix =
            descendant_lower_block.Submatrix(descendant_active_rel_row, 0,
                                             descendant_active_intersect_size,
                                             descendant_supernode_size);

        const ConstBlasMatrix<Field> descendant_main_matrix =
            descendant_lower_block.Submatrix(descendant_main_rel_row, 0,
                                             descendant_main_intersect_size,
                                             descendant_supernode_size);

        BlasMatrix<Field> scaled_transpose;
        scaled_transpose.height = descendant_supernode_size;
        scaled_transpose.width = descendant_main_intersect_size;
        scaled_transpose.leading_dim = descendant_supernode_size;
        scaled_transpose.data =
            private_state.ldl_state.scaled_transpose_buffer.Data();

        supernodal_ldl::FormScaledTranspose(
            factorization_type, descendant_diag_block, descendant_main_matrix,
            &scaled_transpose);

        BlasMatrix<Field> main_active_block = main_lower_block.Submatrix(
            main_active_rel_row, 0, main_active_intersect_size,
            main_supernode_size);

        BlasMatrix<Field> workspace_matrix;
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
#endif  // ifdef _OPENMP

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

  BlasMatrix<Field>& diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrix<Field>& lower_block = lower_factor_->blocks[supernode];

  // Sample and factor the diagonal block.
  const std::vector<Int> supernode_sample = LowerFactorAndSampleDPP(
      control_.block_size, maximum_likelihood, &diagonal_block,
      &private_state->generator, &private_state->unit_uniform);
  AppendSupernodeSample(supernode, supernode_sample, sample);

  supernodal_ldl::SolveAgainstDiagonalBlock(
      factorization_type, diagonal_block.ToConst(), &lower_block);
}

#ifdef _OPENMP
template <class Field>
void SupernodalDPP<Field>::MultithreadedLeftLookingSupernodeSample(
    Int main_supernode, bool maximum_likelihood,
    Buffer<PrivateState>* private_states, std::vector<Int>* sample) const {
  const SymmetricFactorizationType factorization_type =
      kLDLAdjointFactorization;

  BlasMatrix<Field>& main_diagonal_block =
      diagonal_factor_->blocks[main_supernode];
  BlasMatrix<Field>& main_lower_block = lower_factor_->blocks[main_supernode];

  // Sample and factor the diagonal block.
  std::vector<Int> supernode_sample;
  #pragma omp taskgroup
  {
    const int thread = omp_get_thread_num();
    PrivateState& private_state = (*private_states)[thread];
    Buffer<Field>* buffer = &private_state.ldl_state.scaled_transpose_buffer;
    supernode_sample = MultithreadedLowerFactorAndSampleDPP(
        control_.factor_tile_size, control_.block_size, maximum_likelihood,
        &main_diagonal_block, &private_state.generator,
        &private_state.unit_uniform, buffer);
  }
  AppendSupernodeSample(main_supernode, supernode_sample, sample);

  #pragma omp taskgroup
  supernodal_ldl::MultithreadedSolveAgainstDiagonalBlock(
      control_.outer_product_tile_size, factorization_type,
      main_diagonal_block.ToConst(), &main_lower_block);
}
#endif  // ifdef _OPENMP

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

#ifdef _OPENMP
template <class Field>
void SupernodalDPP<Field>::LeftLookingSubtree(
    Int supernode, bool maximum_likelihood,
    supernodal_ldl::LeftLookingSharedState* shared_state,
    PrivateState* private_state, std::vector<Int>* sample) const {
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;

  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    CATAMARI_ASSERT(ordering_.assembly_forest.parents[child] == supernode,
                    "Incorrect child index");
    LeftLookingSubtree(child, maximum_likelihood, shared_state, private_state,
                       sample);
  }

  LeftLookingSupernodeUpdate(supernode, shared_state, private_state);

  std::vector<Int> subsample;
  LeftLookingSupernodeSample(supernode, maximum_likelihood, private_state,
                             &subsample);
  sample->insert(sample->end(), subsample.begin(), subsample.end());
}

template <class Field>
void SupernodalDPP<Field>::MultithreadedLeftLookingSubtree(
    Int level, Int max_parallel_levels, Int supernode, bool maximum_likelihood,
    supernodal_ldl::LeftLookingSharedState* shared_state,
    Buffer<PrivateState>* private_states, std::vector<Int>* sample) const {
  if (level >= max_parallel_levels) {
    const int thread = omp_get_thread_num();
    LeftLookingSubtree(supernode, maximum_likelihood, shared_state,
                       &(*private_states)[thread], sample);
    return;
  }

  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;

  // NOTE: We could alternatively avoid switch to maintaining a single, shared
  // boolean list of length 'num_rows' which flags each entry as 'in' or
  // 'out' of the sample.
  Buffer<std::vector<Int>> subsamples(num_children);

  #pragma omp taskgroup
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    std::vector<Int>* subsample = &subsamples[child_index];
    #pragma omp task default(none)                                \
        firstprivate(level, max_parallel_levels, supernode,       \
            maximum_likelihood, child_index, child, shared_state, \
            private_states, subsample)
    MultithreadedLeftLookingSubtree(level + 1, max_parallel_levels, child,
                                    maximum_likelihood, shared_state,
                                    private_states, subsample);
  }

  // Merge the subsamples into the current sample.
  for (const std::vector<Int>& subsample : subsamples) {
    sample->insert(sample->end(), subsample.begin(), subsample.end());
  }

  #pragma omp taskgroup
  MultithreadedLeftLookingSupernodeUpdate(supernode, shared_state,
                                          private_states);

  std::vector<Int> subsample;
  #pragma omp taskgroup
  MultithreadedLeftLookingSupernodeSample(supernode, maximum_likelihood,
                                          private_states, &subsample);

  sample->insert(sample->end(), subsample.begin(), subsample.end());
}

template <class Field>
std::vector<Int> SupernodalDPP<Field>::MultithreadedLeftLookingSample(
    bool maximum_likelihood) const {
  const Int num_rows = ordering_.supernode_offsets.Back();
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  const int max_threads = omp_get_max_threads();

  supernodal_ldl::MultithreadedFillZeros(ordering_, lower_factor_.get(),
                                         diagonal_factor_.get());

  // Initialize the factors with the input matrix.
  supernodal_ldl::MultithreadedFillNonzeros(
      matrix_, ordering_, supernode_member_to_index_, lower_factor_.get(),
      diagonal_factor_.get());

  std::vector<Int> sample;
  sample.reserve(num_rows);

  supernodal_ldl::LeftLookingSharedState shared_state;
  shared_state.rel_rows.Resize(num_supernodes);
  shared_state.intersect_ptrs.Resize(num_supernodes);

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

  // TODO(Jack Poulson): Make this value configurable.
  const Int max_parallel_levels = std::ceil(std::log2(max_threads)) + 3;

  const Int level = 0;
  if (max_parallel_levels == 0) {
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
    #pragma omp taskgroup
    for (Int root_index = 0; root_index < num_roots; ++root_index) {
      const Int root = ordering_.assembly_forest.roots[root_index];
      std::vector<Int>* subsample = &subsamples[root_index];
      #pragma omp task default(none) firstprivate(level, max_parallel_levels, \
              maximum_likelihood, root, subsample)                            \
          shared(shared_state, private_states)
      {
        MultithreadedLeftLookingSubtree(level + 1, max_parallel_levels, root,
                                        maximum_likelihood, &shared_state,
                                        &private_states, subsample);
      }
    }

    // Merge the subtree samples into a single sample.
    for (const std::vector<Int>& subsample : subsamples) {
      sample.insert(sample.end(), subsample.begin(), subsample.end());
    }

    SetNumBlasThreads(old_max_threads);
  }

  std::sort(sample.begin(), sample.end());

  return sample;
}
#endif  // ifdef _OPENMP

template <class Field>
void SupernodalDPP<Field>::RightLookingSupernodeSample(
    Int supernode, bool maximum_likelihood,
    supernodal_ldl::RightLookingSharedState<Field>* shared_state,
    PrivateState* private_state, std::vector<Int>* sample) const {
  BlasMatrix<Field>& diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrix<Field>& lower_block = lower_factor_->blocks[supernode];
  const Int degree = lower_block.height;
  const Int supernode_size = lower_block.width;

  // Initialize this supernode's Schur complement as the zero matrix.
  Buffer<Field>& schur_complement_buffer =
      shared_state->schur_complement_buffers[supernode];
  BlasMatrix<Field>& schur_complement =
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
  const std::vector<Int> supernode_sample = LowerFactorAndSampleDPP(
      control_.block_size, maximum_likelihood, &diagonal_block,
      &private_state->generator, &private_state->unit_uniform);
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
  BlasMatrix<Field> scaled_transpose;
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

#ifdef _OPENMP
template <class Field>
void SupernodalDPP<Field>::MultithreadedRightLookingSupernodeSample(
    Int supernode, bool maximum_likelihood,
    supernodal_ldl::RightLookingSharedState<Field>* shared_state,
    Buffer<PrivateState>* private_states, std::vector<Int>* sample) const {
  BlasMatrix<Field>& diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrix<Field>& lower_block = lower_factor_->blocks[supernode];
  const Int degree = lower_block.height;
  const Int supernode_size = lower_block.width;

  // Initialize this supernode's Schur complement as the zero matrix.
  Buffer<Field>& schur_complement_buffer =
      shared_state->schur_complement_buffers[supernode];
  BlasMatrix<Field>& schur_complement =
      shared_state->schur_complements[supernode];
  schur_complement_buffer.Resize(degree * degree, Field{0});
  schur_complement.height = degree;
  schur_complement.width = degree;
  schur_complement.leading_dim = degree;
  schur_complement.data = schur_complement_buffer.Data();

  supernodal_ldl::MultithreadedMergeChildSchurComplements(
      control_.merge_grain_size, supernode, ordering_, lower_factor_.get(),
      diagonal_factor_.get(), shared_state);

  // Sample and factor the diagonal block.
  std::vector<Int> supernode_sample;
  #pragma omp taskgroup
  {
    const int thread = omp_get_thread_num();
    PrivateState& private_state = (*private_states)[thread];
    Buffer<Field>* buffer = &private_state.ldl_state.scaled_transpose_buffer;
    supernode_sample = MultithreadedLowerFactorAndSampleDPP(
        control_.factor_tile_size, control_.block_size, maximum_likelihood,
        &diagonal_block, &private_state.generator, &private_state.unit_uniform,
        buffer);
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
  supernodal_ldl::MultithreadedSolveAgainstDiagonalBlock(
      control_.outer_product_tile_size, factorization_type,
      diagonal_block.ToConst(), &lower_block);

  // TODO(Jack Poulson): See if this can be pre-allocated.
  Buffer<Field> scaled_transpose_buffer(degree * supernode_size);
  BlasMatrix<Field> scaled_transpose;
  scaled_transpose.height = supernode_size;
  scaled_transpose.width = degree;
  scaled_transpose.leading_dim = supernode_size;
  scaled_transpose.data = scaled_transpose_buffer.Data();

  #pragma omp taskgroup
  supernodal_ldl::MultithreadedFormScaledTranspose(
      control_.outer_product_tile_size, factorization_type,
      diagonal_block.ToConst(), lower_block.ToConst(), &scaled_transpose);

  #pragma omp taskgroup
  MultithreadedMatrixMultiplyLowerNormalNormal(
      control_.outer_product_tile_size, Field{-1}, lower_block.ToConst(),
      scaled_transpose.ToConst(), Field{1}, &schur_complement);
}
#endif  // ifdef _OPENMP

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

#ifdef _OPENMP
template <class Field>
void SupernodalDPP<Field>::MultithreadedRightLookingSubtree(
    Int level, Int max_parallel_levels, Int supernode, bool maximum_likelihood,
    supernodal_ldl::RightLookingSharedState<Field>* shared_state,
    Buffer<PrivateState>* private_states, std::vector<Int>* sample) const {
  if (level >= max_parallel_levels) {
    const int thread = omp_get_thread_num();
    PrivateState& private_state = (*private_states)[thread];
    RightLookingSubtree(supernode, maximum_likelihood, shared_state,
                        &private_state, sample);
    return;
  }

  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;

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

    #pragma omp task default(none) firstprivate(child, level, \
        max_parallel_levels, maximum_likelihood, subsample)   \
        shared(shared_state, private_states)
    MultithreadedRightLookingSubtree(level + 1, max_parallel_levels, child,
                                     maximum_likelihood, shared_state,
                                     private_states, subsample);
  }

  // Merge the subsamples into the current sample.
  for (const std::vector<Int>& subsample : subsamples) {
    sample->insert(sample->end(), subsample.begin(), subsample.end());
  }

  std::vector<Int> subsample;
  #pragma omp taskgroup
  MultithreadedRightLookingSupernodeSample(
      supernode, maximum_likelihood, shared_state, private_states, &subsample);
  sample->insert(sample->end(), subsample.begin(), subsample.end());
}
#endif  // ifdef _OPENMP

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

#ifdef _OPENMP
template <class Field>
std::vector<Int> SupernodalDPP<Field>::MultithreadedRightLookingSample(
    bool maximum_likelihood) const {
  const Int num_rows = matrix_.NumRows();
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const Int num_roots = ordering_.assembly_forest.roots.Size();

  supernodal_ldl::MultithreadedFillZeros(ordering_, lower_factor_.get(),
                                         diagonal_factor_.get());

  // Initialize the factors with the input matrix.
  supernodal_ldl::MultithreadedFillNonzeros(
      matrix_, ordering_, supernode_member_to_index_, lower_factor_.get(),
      diagonal_factor_.get());

  std::vector<Int> sample;
  sample.reserve(num_rows);

  supernodal_ldl::RightLookingSharedState<Field> shared_state;
  shared_state.schur_complement_buffers.Resize(num_supernodes);
  shared_state.schur_complements.Resize(num_supernodes);

  const int max_threads = omp_get_max_threads();

  // TODO(Jack Poulson): Make this valuable configurable.
  const int max_parallel_levels = std::ceil(std::log2(max_threads)) + 3;

  const Int level = 0;
  if (max_parallel_levels == 0) {
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
    #pragma omp taskgroup
    for (Int root_index = 0; root_index < num_roots; ++root_index) {
      const Int root = ordering_.assembly_forest.roots[root_index];
      std::vector<Int>* subsample = &subsamples[root_index];

      #pragma omp task default(none) firstprivate(root, level, \
          max_parallel_levels, maximum_likelihood, subsample)  \
          shared(shared_state, private_states)
      MultithreadedRightLookingSubtree(level + 1, max_parallel_levels, root,
                                       maximum_likelihood, &shared_state,
                                       &private_states, subsample);
    }

    // Merge the subsamples into the current sample.
    for (const std::vector<Int>& subsample : subsamples) {
      sample.insert(sample.end(), subsample.begin(), subsample.end());
    }

    SetNumBlasThreads(old_max_threads);
  }

  std::sort(sample.begin(), sample.end());

  return sample;
}
#endif  // ifdef _OPENMP

}  // namespace catamari

#endif  // ifndef CATAMARI_SUPERNODAL_DPP_IMPL_H_
