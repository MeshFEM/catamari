/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_COMMON_IMPL_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_COMMON_IMPL_H_

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"

#include "catamari/sparse_ldl/supernodal/factorization.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
void Factorization<Field>::IncorporateSupernodeIntoLDLResult(
    Int supernode_size, Int degree, SparseLDLResult* result) {
  // Finish updating the result structure.
  result->largest_supernode =
      std::max(result->largest_supernode, supernode_size);
  result->num_factorization_entries +=
      (supernode_size * (supernode_size + 1)) / 2 + supernode_size * degree;

  // Compute the number of flops for factoring the diagonal block.
  const double diagonal_flops = (IsComplex<Field>::value ? 4. : 1.) *
                                    std::pow(1. * supernode_size, 3.) / 3. +
                                std::pow(1. * supernode_size, 2.) / 2.;

  // Compute the number of flops to update the subdiagonal block.
  const double solve_flops = (IsComplex<Field>::value ? 4. : 1.) * degree *
                             std::pow(1. * supernode_size, 2.);

  // Compute the number of flops for forming the Schur complement.
  const double schur_complement_flops = (IsComplex<Field>::value ? 4. : 1.) *
                                        supernode_size *
                                        std::pow(1. * degree, 2.);

  result->num_diagonal_flops += diagonal_flops;
  result->num_subdiag_solve_flops += solve_flops;
  result->num_schur_complement_flops += schur_complement_flops;
  result->num_factorization_flops +=
      diagonal_flops + solve_flops + schur_complement_flops;
}

template <class Field>
void Factorization<Field>::MergeContribution(
    const SparseLDLResult& contribution, SparseLDLResult* result) {
  result->num_successful_pivots += contribution.num_successful_pivots;
  result->largest_supernode =
      std::max(result->largest_supernode, contribution.largest_supernode);
  result->num_factorization_entries += contribution.num_factorization_entries;

  result->num_diagonal_flops += contribution.num_diagonal_flops;
  result->num_subdiag_solve_flops += contribution.num_subdiag_solve_flops;
  result->num_schur_complement_flops += contribution.num_schur_complement_flops;
  result->num_factorization_flops += contribution.num_factorization_flops;
}

template <class Field>
void Factorization<Field>::FormSupernodes(const CoordinateMatrix<Field>& matrix,
                                          AssemblyForest* forest,
                                          Buffer<Int>* supernode_degrees) {
  // Greedily compute a supernodal partition using the original ordering.
  AssemblyForest orig_scalar_forest;
  SymmetricOrdering fund_ordering;
  fund_ordering.permutation = ordering_.permutation;
  fund_ordering.inverse_permutation = ordering_.inverse_permutation;
  scalar_ldl::LowerStructure scalar_structure;
  FormFundamentalSupernodes(matrix, ordering_, &orig_scalar_forest,
                            &fund_ordering.supernode_sizes, &scalar_structure);
  OffsetScan(fund_ordering.supernode_sizes, &fund_ordering.supernode_offsets);
  CATAMARI_ASSERT(fund_ordering.supernode_offsets.Back() == matrix.NumRows(),
                  "Supernodes did not sum to the matrix size.");

  Buffer<Int> fund_member_to_index;
  MemberToIndex(matrix.NumRows(), fund_ordering.supernode_offsets,
                &fund_member_to_index);

  const Int num_fund_supernodes = fund_ordering.supernode_sizes.Size();
  ConvertFromScalarToSupernodalEliminationForest(
      num_fund_supernodes, orig_scalar_forest.parents, fund_member_to_index,
      &fund_ordering.assembly_forest.parents);
  fund_ordering.assembly_forest.FillFromParents();

  Buffer<Int> fund_supernode_degrees;
  SupernodalDegrees(matrix, fund_ordering, orig_scalar_forest,
                    fund_member_to_index, &fund_supernode_degrees);

  const SupernodalRelaxationControl& relax_control =
      control_.relaxation_control;
  if (relax_control.relax_supernodes) {
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
void Factorization<Field>::InitializeFactors(
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

  FillStructureIndices(matrix, ordering_, forest, supernode_member_to_index_,
                       lower_factor_.get());
  if (control_.algorithm == kLeftLookingLDL) {
    lower_factor_->FillIntersectionSizes(ordering_.supernode_sizes,
                                         supernode_member_to_index_);
  }

  FillZeros(ordering_, lower_factor_.get(), diagonal_factor_.get());
  FillNonzeros(matrix, ordering_, supernode_member_to_index_,
               lower_factor_.get(), diagonal_factor_.get());
}

template <class Field>
void Factorization<Field>::InitialFactorizationSetup(
    const CoordinateMatrix<Field>& matrix) {
  AssemblyForest forest;
  Buffer<Int> supernode_degrees;
  FormSupernodes(matrix, &forest, &supernode_degrees);
  InitializeFactors(matrix, forest, supernode_degrees);
}

template <class Field>
SparseLDLResult Factorization<Field>::Factor(
    const CoordinateMatrix<Field>& matrix,
    const SymmetricOrdering& manual_ordering, const Control& control) {
  control_ = control;
  ordering_ = manual_ordering;

#ifdef CATAMARI_OPENMP
  if (omp_get_max_threads() > 1) {
    #pragma omp parallel
    #pragma omp single
    OpenMPInitialFactorizationSetup(matrix);
  } else {
    InitialFactorizationSetup(matrix);
  }
#else
  InitialFactorizationSetup(matrix);
#endif

  if (control_.algorithm == kLeftLookingLDL) {
    return LeftLooking(matrix);
  } else {
    return RightLooking(matrix);
  }
}

template <class Field>
SparseLDLResult Factorization<Field>::RefactorWithFixedSparsityPattern(
    const CoordinateMatrix<Field>& matrix) {
  // TODO(Jack Poulson): Check that the previous factorization had an identical
  // sparsity pattern.

#ifdef CATAMARI_OPENMP
  if (omp_get_max_threads() > 1) {
    #pragma omp parallel
    #pragma omp single
    {
      OpenMPFillZeros(ordering_, lower_factor_.get(), diagonal_factor_.get());
      OpenMPFillNonzeros(matrix, ordering_, supernode_member_to_index_,
                         lower_factor_.get(), diagonal_factor_.get());
    }
  } else {
    FillZeros(ordering_, lower_factor_.get(), diagonal_factor_.get());
    FillNonzeros(matrix, ordering_, supernode_member_to_index_,
                 lower_factor_.get(), diagonal_factor_.get());
  }
#else
  FillZeros(ordering_, lower_factor_.get(), diagonal_factor_.get());
  FillNonzeros(matrix, ordering_, supernode_member_to_index_,
               lower_factor_.get(), diagonal_factor_.get());
#endif

  if (control_.algorithm == kLeftLookingLDL) {
    return LeftLooking(matrix);
  } else {
    return RightLooking(matrix);
  }
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_COMMON_IMPL_H_
