/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_HERMITIAN_DPP_SUPERNODAL_COMMON_IMPL_H_
#define CATAMARI_SPARSE_HERMITIAN_DPP_SUPERNODAL_COMMON_IMPL_H_

#include <algorithm>

#include "catamari/sparse_hermitian_dpp/supernodal.hpp"

namespace catamari {

template <class Field>
SupernodalHermitianDPP<Field>::SupernodalHermitianDPP(
    const CoordinateMatrix<Field>& matrix, const SymmetricOrdering& ordering,
    const SupernodalHermitianDPPControl& control)
    : matrix_(matrix), ordering_(ordering), control_(control) {
#ifdef CATAMARI_OPENMP
  const int max_threads = omp_get_max_threads();
  if (max_threads > 1) {
    #pragma omp parallel
    #pragma omp single
    {
      OpenMPFormSupernodes();
      OpenMPFormStructure();
    }

    // Compute flop-count estimates so that we may prioritize the expensive
    // tasks before the cheaper ones.
    const Int num_supernodes = ordering_.supernode_sizes.Size();
    work_estimates_.Resize(num_supernodes);
    for (const Int& root : ordering_.assembly_forest.roots) {
      supernodal_ldl::FillSubtreeWorkEstimates(
          root, ordering_.assembly_forest, *lower_factor_, &work_estimates_);
    }
    total_work_ =
        std::accumulate(work_estimates_.begin(), work_estimates_.end(), 0.);
    const double min_parallel_ratio_work =
        (total_work_ * control_.parallel_ratio_threshold) / max_threads;
    min_parallel_work_ =
        std::max(control_.min_parallel_threshold, min_parallel_ratio_work);

    return;
  }
#endif  // ifdef CATAMARI_OPENMP
  FormSupernodes();
  FormStructure();
}

template <class Field>
void SupernodalHermitianDPP<Field>::FormSupernodes() {
  // Greedily compute a supernodal partition using the original ordering.
  AssemblyForest orig_scalar_forest;
  SymmetricOrdering fund_ordering;
  fund_ordering.permutation = ordering_.permutation;
  fund_ordering.inverse_permutation = ordering_.inverse_permutation;

  Buffer<Int> orig_scalar_degrees;
  scalar_ldl::EliminationForestAndDegrees(
      matrix_, ordering_, &orig_scalar_forest.parents, &orig_scalar_degrees);
  orig_scalar_forest.FillFromParents();

  supernodal_ldl::FormFundamentalSupernodes(
      orig_scalar_forest, orig_scalar_degrees, &fund_ordering.supernode_sizes);
  OffsetScan(fund_ordering.supernode_sizes, &fund_ordering.supernode_offsets);
#ifdef CATAMARI_DEBUG
  if (!supernodal_ldl::ValidFundamentalSupernodes(
          matrix_, ordering_, fund_ordering.supernode_sizes)) {
    std::cerr << "Invalid fundamental supernodes." << std::endl;
    return;
  }
#endif  // ifdef CATAMARI_DEBUG

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
        fund_supernode_degrees, fund_member_to_index,
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
void SupernodalHermitianDPP<Field>::FormStructure() {
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
std::vector<Int> SupernodalHermitianDPP<Field>::Sample(
    bool maximum_likelihood) const {
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

template <typename Field>
void SupernodalHermitianDPP<Field>::AppendSupernodeSample(
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

template <typename Field>
ComplexBase<Field> SupernodalHermitianDPP<Field>::LogLikelihood() const {
  typedef ComplexBase<Field> Real;
  const Int num_supernodes = diagonal_factor_->blocks.Size();

  Real log_likelihood = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const ConstBlasMatrixView<Field> diag_block =
        diagonal_factor_->blocks[supernode];
    const Int supernode_size = diag_block.height;
    for (Int j = 0; j < supernode_size; ++j) {
      log_likelihood += std::log(std::abs(diag_block(j, j)));
    }
  }

  return log_likelihood;
}

}  // namespace catamari

#endif  // ifndef CATAMARI_SPARSE_HERMITIAN_DPP_SUPERNODAL_COMMON_IMPL_H_
