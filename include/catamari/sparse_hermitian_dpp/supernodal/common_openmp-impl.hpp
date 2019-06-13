/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_HERMITIAN_DPP_SUPERNODAL_COMMON_OPENMP_IMPL_H_
#define CATAMARI_SPARSE_HERMITIAN_DPP_SUPERNODAL_COMMON_OPENMP_IMPL_H_
#ifdef CATAMARI_OPENMP

#include <algorithm>

#include "catamari/sparse_hermitian_dpp/supernodal.hpp"

namespace catamari {

template <class Field>
void SupernodalHermitianDPP<Field>::OpenMPFormSupernodes() {
  // Greedily compute a supernodal partition using the original ordering.
  AssemblyForest orig_scalar_forest;
  SymmetricOrdering fund_ordering;
  fund_ordering.permutation = ordering_.permutation;
  fund_ordering.inverse_permutation = ordering_.inverse_permutation;

  Buffer<Int> orig_scalar_degrees;
  scalar_ldl::OpenMPEliminationForestAndDegrees(
      matrix_, ordering_, &orig_scalar_forest.parents, &orig_scalar_degrees);
  orig_scalar_forest.FillFromParents();

  supernodal_ldl::FormFundamentalSupernodes(
      orig_scalar_forest, orig_scalar_degrees, &fund_ordering.supernode_sizes);
  OffsetScan(fund_ordering.supernode_sizes, &fund_ordering.supernode_offsets);
  CATAMARI_ASSERT(fund_ordering.supernode_offsets.Back() == matrix_.NumRows(),
                  "Supernodes did not sum to the matrix size.");
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

  // TODO(Jack Poulson): Parallelize
  //     ConvertFromScalarToSupernodalEliminationForest.
  const Int num_fund_supernodes = fund_ordering.supernode_sizes.Size();
  Buffer<Int> fund_supernode_parents;
  supernodal_ldl::ConvertFromScalarToSupernodalEliminationForest(
      num_fund_supernodes, orig_scalar_forest.parents, fund_member_to_index,
      &fund_ordering.assembly_forest.parents);
  fund_ordering.assembly_forest.FillFromParents();

  // Convert the scalar degrees into the supernodal degrees.
  Buffer<Int> fund_supernode_degrees(num_fund_supernodes);
  for (Int supernode = 0; supernode < num_fund_supernodes; ++supernode) {
    const Int supernode_tail = fund_ordering.supernode_offsets[supernode] +
                               fund_ordering.supernode_sizes[supernode] - 1;
    fund_supernode_degrees[supernode] = orig_scalar_degrees[supernode_tail];
  }

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

    supernode_member_to_index_ = fund_member_to_index;
    supernode_degrees_ = fund_supernode_degrees;
  }
}

template <class Field>
void SupernodalHermitianDPP<Field>::OpenMPFormStructure() {
  CATAMARI_ASSERT(supernode_degrees_.Size() == ordering_.supernode_sizes.Size(),
                  "Invalid supernode degrees size.");

  lower_factor_.reset(new supernodal_ldl::LowerFactor<Field>(
      ordering_.supernode_sizes, supernode_degrees_));
  diagonal_factor_.reset(
      new supernodal_ldl::DiagonalFactor<Field>(ordering_.supernode_sizes));

  max_supernode_size_ = *std::max_element(ordering_.supernode_sizes.begin(),
                                          ordering_.supernode_sizes.end());

  supernodal_ldl::OpenMPFillStructureIndices(
      control_.sort_grain_size, matrix_, ordering_, forest_,
      supernode_member_to_index_, lower_factor_.get());

  if (control_.algorithm == kLeftLookingLDL) {
    // TODO(Jack Poulson): Switch to a multithreaded equivalent.
    lower_factor_->FillIntersectionSizes(ordering_.supernode_sizes,
                                         supernode_member_to_index_);
  }
}

}  // namespace catamari

#endif  // ifdef CATAMARI_OPENMP
#endif  // ifndef CATAMARI_SPARSE_HERMITIAN_DPP_SUPERNODAL_COMMON_OPENMP_IMPL_H_
