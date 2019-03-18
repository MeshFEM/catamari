/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SUPERNODAL_DPP_COMMON_IMPL_H_
#define CATAMARI_SUPERNODAL_DPP_COMMON_IMPL_H_

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

}  // namespace catamari

#endif  // ifndef CATAMARI_SUPERNODAL_DPP_COMMON_IMPL_H_
