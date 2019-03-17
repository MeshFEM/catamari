/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_COMMON_OPENMP_IMPL_H_
#define CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_COMMON_OPENMP_IMPL_H_
#ifdef CATAMARI_OPENMP

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"

#include "catamari/ldl/supernodal_ldl/factorization.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
void Factorization<Field>::OpenMPInitialFactorizationSetup(
    const CoordinateMatrix<Field>& matrix) {
  AssemblyForest forest;
  Buffer<Int> supernode_degrees;

  #pragma omp taskgroup
  OpenMPFormSupernodes(matrix, &forest, &supernode_degrees);

  #pragma omp taskgroup
  OpenMPInitializeFactors(matrix, forest, supernode_degrees);
}

template <class Field>
void Factorization<Field>::OpenMPFormSupernodes(
    const CoordinateMatrix<Field>& matrix, AssemblyForest* forest,
    Buffer<Int>* supernode_degrees) {
  // Greedily compute a supernodal partition using the original ordering.
  AssemblyForest orig_scalar_forest;
  SymmetricOrdering fund_ordering;
  fund_ordering.permutation = ordering_.permutation;
  fund_ordering.inverse_permutation = ordering_.inverse_permutation;
  scalar_ldl::LowerStructure scalar_structure;
  OpenMPFormFundamentalSupernodes(matrix, ordering_, &orig_scalar_forest,
                                  &fund_ordering.supernode_sizes,
                                  &scalar_structure);
  OffsetScan(fund_ordering.supernode_sizes, &fund_ordering.supernode_offsets);
  CATAMARI_ASSERT(fund_ordering.supernode_offsets.Back() == matrix.NumRows(),
                  "Supernodes did not sum to the matrix size.");

  Buffer<Int> fund_member_to_index;
  MemberToIndex(matrix.NumRows(), fund_ordering.supernode_offsets,
                &fund_member_to_index);

  // TODO(Jack Poulson): Parallelize
  //     ConvertFromScalarToSupernodalEliminationForest.
  const Int num_fund_supernodes = fund_ordering.supernode_sizes.Size();
  Buffer<Int> fund_supernode_parents;
  ConvertFromScalarToSupernodalEliminationForest(
      num_fund_supernodes, orig_scalar_forest.parents, fund_member_to_index,
      &fund_ordering.assembly_forest.parents);
  fund_ordering.assembly_forest.FillFromParents();

  Buffer<Int> fund_supernode_degrees;
  OpenMPSupernodalDegrees(matrix, fund_ordering, orig_scalar_forest,
                          fund_member_to_index, &fund_supernode_degrees);

  const SupernodalRelaxationControl& relax_control =
      control_.relaxation_control;
  if (relax_control.relax_supernodes) {
    // TODO(Jack Poulson): Parallelize RelaxSupernodes.
    RelaxSupernodes(
        orig_scalar_forest.parents, fund_ordering.supernode_sizes,
        fund_ordering.supernode_offsets, fund_ordering.assembly_forest.parents,
        fund_supernode_degrees, fund_member_to_index, scalar_structure,
        relax_control, &ordering_.permutation, &ordering_.inverse_permutation,
        &forest->parents, &ordering_.assembly_forest.parents, supernode_degrees,
        &ordering_.supernode_sizes, &ordering_.supernode_offsets,
        &supernode_member_to_index_);
    forest->FillFromParents();
    ordering_.assembly_forest.FillFromParents();
  } else {
    *forest = orig_scalar_forest;

    ordering_.supernode_sizes = fund_ordering.supernode_sizes;
    ordering_.supernode_offsets = fund_ordering.supernode_offsets;
    ordering_.assembly_forest.parents = fund_ordering.assembly_forest.parents;
    ordering_.assembly_forest.FillFromParents();

    supernode_member_to_index_ = fund_member_to_index;
    *supernode_degrees = fund_supernode_degrees;
  }
}

template <class Field>
void Factorization<Field>::OpenMPInitializeFactors(
    const CoordinateMatrix<Field>& matrix, const AssemblyForest& forest,
    const Buffer<Int>& supernode_degrees) {
  lower_factor_.reset(
      new LowerFactor<Field>(ordering_.supernode_sizes, supernode_degrees));
  diagonal_factor_.reset(new DiagonalFactor<Field>(ordering_.supernode_sizes));

  CATAMARI_ASSERT(supernode_degrees.Size() == ordering_.supernode_sizes.Size(),
                  "Invalid supernode degrees size.");

  // Store the largest supernode size of the factorization.
  max_supernode_size_ = *std::max_element(ordering_.supernode_sizes.begin(),
                                          ordering_.supernode_sizes.end());

  // Store the largest degree of the factorization for use in the solve phase.
  max_degree_ =
      *std::max_element(supernode_degrees.begin(), supernode_degrees.end());

  // Compute the maximum number of entries below the diagonal block of a
  // supernode.
  max_lower_block_size_ = 0;
  const Int num_supernodes = supernode_degrees.Size();
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int lower_block_size =
        supernode_degrees[supernode] * ordering_.supernode_sizes[supernode];
    max_lower_block_size_ = std::max(max_lower_block_size_, lower_block_size);
  }

  OpenMPFillStructureIndices(control_.sort_grain_size, matrix, ordering_,
                             forest, supernode_member_to_index_,
                             lower_factor_.get());
  if (control_.algorithm == kLeftLookingLDL) {
    // TODO(Jack Poulson): Switch to a multithreaded equivalent.
    lower_factor_->FillIntersectionSizes(ordering_.supernode_sizes,
                                         supernode_member_to_index_);
  }

  OpenMPFillZeros(ordering_, lower_factor_.get(), diagonal_factor_.get());
  OpenMPFillNonzeros(matrix, ordering_, supernode_member_to_index_,
                     lower_factor_.get(), diagonal_factor_.get());
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifdef CATAMARI_OPENMP
#endif  // ifndef
        // CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_COMMON_OPENMP_IMPL_H_
