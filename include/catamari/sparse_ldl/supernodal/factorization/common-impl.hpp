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
    Int supernode_size, Int degree, SparseLDLResult<Field>* result) {
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
    const SparseLDLResult<Field>& contribution,
    SparseLDLResult<Field>* result) {
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
                                          Buffer<Int>* supernode_degrees) {
  CATAMARI_START_TIMER(profile.scalar_elimination_forest);
  Buffer<Int> scalar_parents;
  Buffer<Int> scalar_degrees;
  const bool explicitly_permute = false;  // TODO(Jack Poulson): Make optional.
  if (ordering_.permutation.Empty()) {
    scalar_ldl::EliminationForestAndDegrees(matrix, &scalar_parents,
                                            &scalar_degrees);
  } else if (explicitly_permute) {
    CoordinateMatrix<Field> reordered_matrix;
    PermuteMatrix(matrix, ordering_, &reordered_matrix);
    scalar_ldl::EliminationForestAndDegrees(reordered_matrix, &scalar_parents,
                                            &scalar_degrees);
  } else {
    scalar_ldl::EliminationForestAndDegrees(matrix, ordering_, &scalar_parents,
                                            &scalar_degrees);
  }
  CATAMARI_STOP_TIMER(profile.scalar_elimination_forest);

  SymmetricOrdering fund_ordering;
  fund_ordering.permutation = ordering_.permutation;
  fund_ordering.inverse_permutation = ordering_.inverse_permutation;
  FormFundamentalSupernodes(scalar_parents, scalar_degrees,
                            &fund_ordering.supernode_sizes);
  OffsetScan(fund_ordering.supernode_sizes, &fund_ordering.supernode_offsets);
  CATAMARI_ASSERT(fund_ordering.supernode_offsets.Back() == matrix.NumRows(),
                  "Supernodes did not sum to the matrix size.");
#ifdef CATAMARI_DEBUG
  if (!supernodal_ldl::ValidFundamentalSupernodes(
          matrix, ordering_, fund_ordering.supernode_sizes)) {
    std::cerr << "Invalid fundamental supernodes." << std::endl;
    return;
  }
#endif  // ifdef CATAMARI_DEBUG

  Buffer<Int> fund_member_to_index;
  MemberToIndex(matrix.NumRows(), fund_ordering.supernode_offsets,
                &fund_member_to_index);

  CATAMARI_START_TIMER(profile.supernodal_elimination_forest);
  const Int num_fund_supernodes = fund_ordering.supernode_sizes.Size();
  ConvertFromScalarToSupernodalEliminationForest(
      num_fund_supernodes, scalar_parents, fund_member_to_index,
      &fund_ordering.assembly_forest.parents);
  CATAMARI_STOP_TIMER(profile.supernodal_elimination_forest);

  // Construct the supernodal degrees from the scalar degrees.
  Buffer<Int> fund_supernode_degrees(num_fund_supernodes);
  for (Int supernode = 0; supernode < num_fund_supernodes; ++supernode) {
    const Int supernode_tail = fund_ordering.supernode_offsets[supernode] +
                               fund_ordering.supernode_sizes[supernode] - 1;
    fund_supernode_degrees[supernode] = scalar_degrees[supernode_tail];
  }

  CATAMARI_START_TIMER(profile.relax_supernodes);
  const SupernodalRelaxationControl& relax_control =
      control_.relaxation_control;
  if (relax_control.relax_supernodes) {
    RelaxSupernodes(fund_ordering, fund_supernode_degrees, relax_control,
                    &ordering_, supernode_degrees, &supernode_member_to_index_);
  } else {
    ordering_.supernode_sizes = fund_ordering.supernode_sizes;
    ordering_.supernode_offsets = fund_ordering.supernode_offsets;
    ordering_.assembly_forest.parents = fund_ordering.assembly_forest.parents;
    ordering_.assembly_forest.FillFromParents();

    supernode_member_to_index_ = fund_member_to_index;
    *supernode_degrees = fund_supernode_degrees;
  }
  CATAMARI_STOP_TIMER(profile.relax_supernodes);
}

template <class Field>
void Factorization<Field>::InitializeFactors(
    const CoordinateMatrix<Field>& matrix,
    const Buffer<Int>& supernode_degrees) {
  lower_factor_.reset(
      new LowerFactor<Field>(ordering_.supernode_sizes, supernode_degrees));
  diagonal_factor_.reset(new DiagonalFactor<Field>(ordering_.supernode_sizes));

  CATAMARI_ASSERT(supernode_degrees.Size() == ordering_.supernode_sizes.Size(),
                  "Invalid supernode degrees size.");

  // Store the largest degree of the factorization for use in the solve phase.
  max_degree_ =
      *std::max_element(supernode_degrees.begin(), supernode_degrees.end());

  FillStructureIndices(matrix, ordering_, supernode_member_to_index_,
                       lower_factor_.get());
  if (control_.algorithm == kLeftLookingLDL) {
    lower_factor_->FillIntersectionSizes(ordering_.supernode_sizes,
                                         supernode_member_to_index_);

    // Compute the maximum of the diagonal and subdiagonal update sizes.
    Int workspace_size = 0;
    Int scaled_transpose_size = 0;
    const Int num_supernodes = supernode_degrees.Size();
    for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
      const Int supernode_size = ordering_.supernode_sizes[supernode];
      Int degree_remaining = supernode_degrees[supernode];
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

        if (control_.factorization_type != kCholeskyFactorization) {
          // Increment the maximum scaled transpose size if necessary.
          scaled_transpose_size =
              std::max(scaled_transpose_size, supernode_size * intersect_size);
        }
      }
    }
    left_looking_workspace_size_ = workspace_size;
    left_looking_scaled_transpose_size_ = scaled_transpose_size;
  } else {
    // Compute the maximum number of entries below the diagonal block of a
    // supernode.
    max_lower_block_size_ = 0;
    const Int num_supernodes = supernode_degrees.Size();
    for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
      const Int lower_block_size =
          supernode_degrees[supernode] * ordering_.supernode_sizes[supernode];
      max_lower_block_size_ = std::max(max_lower_block_size_, lower_block_size);
    }
  }

  if (control_.factorization_type != kLDLAdjointFactorization) {
    // We currently only support supernodal pivoting with LDL^H fact'ns.
    control_.supernodal_pivoting = false;
  }
  if (control_.supernodal_pivoting) {
    const Int num_rows = matrix.NumRows();
    supernode_permutations_.Resize(num_rows, 1);
  }
}

template <class Field>
BlasMatrixView<Int> Factorization<Field>::SupernodePermutation(Int supernode) {
  const Int supernode_offset = ordering_.supernode_offsets[supernode];
  const Int supernode_size = ordering_.supernode_sizes[supernode];
  return supernode_permutations_.Submatrix(supernode_offset, 0, supernode_size,
                                           1);
}

template <class Field>
ConstBlasMatrixView<Int> Factorization<Field>::SupernodePermutation(
    Int supernode) const {
  const Int supernode_offset = ordering_.supernode_offsets[supernode];
  const Int supernode_size = ordering_.supernode_sizes[supernode];
  return supernode_permutations_.Submatrix(supernode_offset, 0, supernode_size,
                                           1);
}

template <class Field>
void Factorization<Field>::InitialFactorizationSetup(
    const CoordinateMatrix<Field>& matrix) {
  Buffer<Int> supernode_degrees;
  FormSupernodes(matrix, &supernode_degrees);
  CATAMARI_START_TIMER(profile.initialize_factors);
  InitializeFactors(matrix, supernode_degrees);
  CATAMARI_STOP_TIMER(profile.initialize_factors);
}

template <class Field>
void Factorization<Field>::InitializeBlockColumn(
    Int supernode, const CoordinateMatrix<Field>& matrix) {
  BlasMatrixView<Field>& diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrixView<Field>& lower_block = lower_factor_->blocks[supernode];

  const bool self_adjoint =
      control_.factorization_type != kLDLTransposeFactorization;
  const bool have_permutation = !ordering_.permutation.Empty();
  const Int supernode_start = ordering_.supernode_offsets[supernode];
  const Int supernode_size = ordering_.supernode_sizes[supernode];
  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  const Int* index_beg = lower_factor_->StructureBeg(supernode);
  const Int* index_end = lower_factor_->StructureEnd(supernode);
  for (Int j = supernode_start; j < supernode_start + supernode_size; ++j) {
    const Int j_rel = j - supernode_start;
    const Int j_orig = have_permutation ? ordering_.inverse_permutation[j] : j;

    // Fill the diagonal block's column with zeros.
    Field* diag_column_ptr = diagonal_block.Pointer(0, j_rel);
    std::fill(diag_column_ptr, diag_column_ptr + supernode_size, Field{0});

    // Fill the lower block's column with zeros.
    Field* lower_column_ptr = lower_block.Pointer(0, j_rel);
    std::fill(lower_column_ptr, lower_column_ptr + lower_block.height,
              Field{0});

    // Insert the entries from the sparse matrix into this column.
    const Int row_beg = matrix.RowEntryOffset(j_orig);
    const Int row_end = matrix.RowEntryOffset(j_orig + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      const Int row =
          have_permutation ? ordering_.permutation[entry.column] : entry.column;
      if (row < supernode_start) {
        continue;
      }
      const Field value = self_adjoint ? Conjugate(entry.value) : entry.value;
      if (row < supernode_start + supernode_size) {
        diag_column_ptr[row - supernode_start] = value;
      } else {
        const Int* iter = std::lower_bound(index_beg, index_end, row);
        CATAMARI_ASSERT(iter != index_end, "Exceeded row indices.");
        CATAMARI_ASSERT(*iter == row, "Entry (" + std::to_string(row) + ", " +
                                          std::to_string(j) +
                                          ") wasn't in the structure.");
        const Int rel_row = std::distance(index_beg, iter);
        lower_column_ptr[rel_row] = value;
      }
    }
  }
}

template <class Field>
SparseLDLResult<Field> Factorization<Field>::Factor(
    const CoordinateMatrix<Field>& matrix,
    const SymmetricOrdering& manual_ordering, const Control<Field>& control) {
  control_ = control;
  ordering_ = manual_ordering;

#ifdef CATAMARI_ENABLE_TIMERS
  profile.Reset();
#endif  // ifdef CATAMARI_ENABLE_TIMERS

#ifdef CATAMARI_OPENMP
  if (omp_get_max_threads() > 1) {
    if (control_.algorithm == kAdaptiveLDL) {
      control_.algorithm = kRightLookingLDL;
    }
    #pragma omp parallel
    #pragma omp single
    OpenMPInitialFactorizationSetup(matrix);
  } else {
    if (control_.algorithm == kAdaptiveLDL) {
      control_.algorithm = kLeftLookingLDL;
    }
    InitialFactorizationSetup(matrix);
  }
#else
  if (control_.algorithm == kAdaptiveLDL) {
    control_.algorithm = kLeftLookingLDL;
  }
  InitialFactorizationSetup(matrix);
#endif  // ifdef CATAMARI_OPENMP

  SparseLDLResult<Field> result;
  if (control_.algorithm == kLeftLookingLDL) {
    result = LeftLooking(matrix);
  } else {
    result = RightLooking(matrix);
  }

#ifdef CATAMARI_ENABLE_TIMERS
  std::cout << profile << std::endl;
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  return result;
}

// TODO(Jack Poulson): Check that the previous factorization had an identical
// sparsity pattern.
template <class Field>
SparseLDLResult<Field> Factorization<Field>::RefactorWithFixedSparsityPattern(
    const CoordinateMatrix<Field>& matrix) {
#ifdef CATAMARI_ENABLE_TIMERS
  profile.Reset();
#endif  // ifdef CATAMARI_ENABLE_TIMERS
  SparseLDLResult<Field> result;
  if (control_.algorithm == kLeftLookingLDL) {
    result = LeftLooking(matrix);
  } else {
    result = RightLooking(matrix);
  }

#ifdef CATAMARI_ENABLE_TIMERS
  std::cout << profile << std::endl;
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  return result;
}

template <class Field>
Int Factorization<Field>::NumRows() const {
  return supernode_member_to_index_.Size();
}

template <class Field>
const Buffer<Int>& Factorization<Field>::Permutation() const {
  return ordering_.permutation;
}

template <class Field>
const Buffer<Int>& Factorization<Field>::InversePermutation() const {
  return ordering_.inverse_permutation;
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_COMMON_IMPL_H_
