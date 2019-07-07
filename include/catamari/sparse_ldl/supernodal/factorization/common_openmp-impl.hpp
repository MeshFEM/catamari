/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_COMMON_OPENMP_IMPL_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_COMMON_OPENMP_IMPL_H_
#ifdef CATAMARI_OPENMP

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"

#ifdef CATAMARI_ENABLE_TIMERS
#include "quotient/timer.hpp"
#endif  // ifdef CATAMARI_ENABLE_TIMERS

#include "catamari/sparse_ldl/supernodal/factorization.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
void Factorization<Field>::OpenMPInitialFactorizationSetup(
    const CoordinateMatrix<Field>& matrix) {
  Buffer<Int> supernode_degrees;

#ifdef CATAMARI_ENABLE_TIMERS
  quotient::Timer timer;
  timer.Start();
  #pragma omp taskgroup
  OpenMPFormSupernodes(matrix, &supernode_degrees);
  std::cout << "OpenMPFormSupernodes: " << timer.Stop() << std::endl;

  timer.Start();
  #pragma omp taskgroup
  OpenMPInitializeFactors(matrix, supernode_degrees);
  std::cout << "OpenMPInitializeFactors: " << timer.Stop() << std::endl;
#else
  #pragma omp taskgroup
  OpenMPFormSupernodes(matrix, &supernode_degrees);

  #pragma omp taskgroup
  OpenMPInitializeFactors(matrix, supernode_degrees);
#endif  // ifdef CATAMARI_ENABLE_TIMERS
}

template <class Field>
void Factorization<Field>::OpenMPFormSupernodes(
    const CoordinateMatrix<Field>& matrix, Buffer<Int>* supernode_degrees) {
  // Greedily compute a supernodal partition using the original ordering.
  AssemblyForest orig_scalar_forest;
  SymmetricOrdering fund_ordering;
  fund_ordering.permutation = ordering_.permutation;
  fund_ordering.inverse_permutation = ordering_.inverse_permutation;

  // Compute the non-supernodal elimination tree using the original ordering.
  CATAMARI_START_TIMER(profile.scalar_elimination_forest);
  Buffer<Int> scalar_degrees;
  scalar_ldl::OpenMPEliminationForestAndDegrees(
      matrix, ordering_, &orig_scalar_forest.parents, &scalar_degrees);
  orig_scalar_forest.FillFromParents();
  CATAMARI_STOP_TIMER(profile.scalar_elimination_forest);

  FormFundamentalSupernodes(orig_scalar_forest, scalar_degrees,
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

  // TODO(Jack Poulson): Parallelize
  //     ConvertFromScalarToSupernodalEliminationForest.
  CATAMARI_START_TIMER(profile.supernodal_elimination_forest);
  const Int num_fund_supernodes = fund_ordering.supernode_sizes.Size();
  Buffer<Int> fund_supernode_parents;
  ConvertFromScalarToSupernodalEliminationForest(
      num_fund_supernodes, orig_scalar_forest.parents, fund_member_to_index,
      &fund_ordering.assembly_forest.parents);
  fund_ordering.assembly_forest.FillFromParents();
  CATAMARI_STOP_TIMER(profile.supernodal_elimination_forest);

  // Convert the scalar degrees into supernodal degrees.
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
    RelaxSupernodes(orig_scalar_forest.parents, fund_ordering,
                    fund_supernode_degrees, fund_member_to_index, relax_control,
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
void Factorization<Field>::OpenMPInitializeFactors(
    const CoordinateMatrix<Field>& matrix,
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
                             supernode_member_to_index_, lower_factor_.get());
  if (control_.algorithm == kLeftLookingLDL) {
    // TODO(Jack Poulson): Switch to a multithreaded equivalent.
    lower_factor_->FillIntersectionSizes(ordering_.supernode_sizes,
                                         supernode_member_to_index_);

    // Compute the maximum of the diagonal and subdiagonal update sizes.
    // TODO(Jack Poulson): Avoid redundancy with common-impl.hpp implementation.
    Int workspace_size = 0;
    Int scaled_transpose_size = 0;
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
  }
}

template <class Field>
void Factorization<Field>::OpenMPInitializeBlockColumn(
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

  #pragma omp parallel for schedule(dynamic)
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

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifdef CATAMARI_OPENMP
#endif  // ifndef
        // CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_COMMON_OPENMP_IMPL_H_
