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
  Buffer<Int> scalar_parents;
  Buffer<Int> scalar_degrees;
  scalar_ldl::OpenMPEliminationForestAndDegrees(
      matrix_, ordering_, &scalar_parents, &scalar_degrees);

  SymmetricOrdering fund_ordering;
  fund_ordering.permutation = ordering_.permutation;
  fund_ordering.inverse_permutation = ordering_.inverse_permutation;
  supernodal_ldl::FormFundamentalSupernodes(scalar_parents, scalar_degrees,
                                            &fund_ordering.supernode_sizes);
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
      num_fund_supernodes, scalar_parents, fund_member_to_index,
      &fund_ordering.assembly_forest.parents);

  // Convert the scalar degrees into the supernodal degrees.
  Buffer<Int> fund_supernode_degrees(num_fund_supernodes);
  for (Int supernode = 0; supernode < num_fund_supernodes; ++supernode) {
    const Int supernode_tail = fund_ordering.supernode_offsets[supernode] +
                               fund_ordering.supernode_sizes[supernode] - 1;
    fund_supernode_degrees[supernode] = scalar_degrees[supernode_tail];
  }

  if (control_.relaxation_control.relax_supernodes) {
    supernodal_ldl::RelaxSupernodes(
        scalar_parents, fund_ordering, fund_supernode_degrees,
        fund_member_to_index, control_.relaxation_control, &ordering_,
        &supernode_degrees_, &supernode_member_to_index_);
  } else {
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
      control_.sort_grain_size, matrix_, ordering_, supernode_member_to_index_,
      lower_factor_.get());

  if (control_.algorithm == kLeftLookingLDL) {
    // TODO(Jack Poulson): Switch to a multithreaded equivalent.
    lower_factor_->FillIntersectionSizes(ordering_.supernode_sizes,
                                         supernode_member_to_index_);

    // Compute the maximum of the diagonal and subdiagonal update sizes.
    Int workspace_size = 0;
    Int scaled_transpose_size = 0;
    const Int num_supernodes = ordering_.supernode_sizes.Size();
    for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
      const Int supernode_size = ordering_.supernode_sizes[supernode];
      Int degree_remaining = supernode_degrees_[supernode];
      const Int* intersect_sizes_beg =
          lower_factor_->IntersectionSizesBeg(supernode);
      const Int* intersect_sizes_end =
          lower_factor_->IntersectionSizesEnd(supernode);
      for (const Int* iter = intersect_sizes_beg; iter != intersect_sizes_end;
           ++iter) {
        const Int intersect_size = *iter;
        degree_remaining -= intersect_size;

        // Handle the space for the diagonal block update.
        workspace_size =
            std::max(workspace_size, intersect_size * intersect_size);

        // Handle the space for the lower update.
        workspace_size =
            std::max(workspace_size, intersect_size * degree_remaining);

        // Increment the maximum scaled transpose size if necessary.
        scaled_transpose_size =
            std::max(scaled_transpose_size, supernode_size * intersect_size);
      }
    }
    left_looking_workspace_size_ = workspace_size;
    left_looking_scaled_transpose_size_ = scaled_transpose_size;
  }
}

}  // namespace catamari

#endif  // ifdef CATAMARI_OPENMP
#endif  // ifndef CATAMARI_SPARSE_HERMITIAN_DPP_SUPERNODAL_COMMON_OPENMP_IMPL_H_
