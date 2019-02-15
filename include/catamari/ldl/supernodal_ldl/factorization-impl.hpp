/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_IMPL_H_
#define CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_IMPL_H_

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"

#include "catamari/ldl/supernodal_ldl.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
void Factorization<Field>::IncorporateSupernodeIntoLDLResult(
    Int supernode_size, Int degree, LDLResult* result) {
  // Finish updating the result structure.
  result->largest_supernode =
      std::max(result->largest_supernode, supernode_size);
  result->num_factorization_entries +=
      (supernode_size * (supernode_size + 1)) / 2 + supernode_size * degree;

  // Add the approximate number of flops for the diagonal block factorization.
  result->num_factorization_flops += std::pow(1. * supernode_size, 3.) / 3. +
                                     std::pow(1. * supernode_size, 2.) / 2.;
}

template <class Field>
void Factorization<Field>::FormSupernodes(
    const CoordinateMatrix<Field>& matrix,
    const SupernodalRelaxationControl& control, AssemblyForest* forest,
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

  if (control.relax_supernodes) {
    RelaxSupernodes(
        orig_scalar_forest.parents, fund_ordering.supernode_sizes,
        fund_ordering.supernode_offsets, fund_ordering.assembly_forest.parents,
        fund_supernode_degrees, fund_member_to_index, scalar_structure, control,
        &ordering_.permutation, &ordering_.inverse_permutation,
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

#ifdef _OPENMP
template <class Field>
void Factorization<Field>::MultithreadedFormSupernodes(
    const CoordinateMatrix<Field>& matrix,
    const SupernodalRelaxationControl& control, AssemblyForest* forest,
    Buffer<Int>* supernode_degrees) {
  // Greedily compute a supernodal partition using the original ordering.
  AssemblyForest orig_scalar_forest;
  SymmetricOrdering fund_ordering;
  fund_ordering.permutation = ordering_.permutation;
  fund_ordering.inverse_permutation = ordering_.inverse_permutation;
  scalar_ldl::LowerStructure scalar_structure;
  MultithreadedFormFundamentalSupernodes(matrix, ordering_, &orig_scalar_forest,
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
  MultithreadedSupernodalDegrees(matrix, fund_ordering, orig_scalar_forest,
                                 fund_member_to_index, &fund_supernode_degrees);

  if (control.relax_supernodes) {
    // TODO(Jack Poulson): Parallelize RelaxSupernodes.
    RelaxSupernodes(
        orig_scalar_forest.parents, fund_ordering.supernode_sizes,
        fund_ordering.supernode_offsets, fund_ordering.assembly_forest.parents,
        fund_supernode_degrees, fund_member_to_index, scalar_structure, control,
        &ordering_.permutation, &ordering_.inverse_permutation,
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
#endif  // ifdef _OPENMP

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

  FillStructureIndices(matrix, ordering_, forest, supernode_member_to_index_,
                       lower_factor_.get());
  if (algorithm_ == kLeftLookingLDL) {
    lower_factor_->FillIntersectionSizes(ordering_.supernode_sizes,
                                         supernode_member_to_index_);
  }

  FillNonzeros(matrix, ordering_, supernode_member_to_index_,
               lower_factor_.get(), diagonal_factor_.get());
}

#ifdef _OPENMP
template <class Field>
void Factorization<Field>::MultithreadedInitializeFactors(
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

  MultithreadedFillStructureIndices(sort_grain_size_, matrix, ordering_, forest,
                                    supernode_member_to_index_,
                                    lower_factor_.get());
  if (algorithm_ == kLeftLookingLDL) {
    // TODO(Jack Poulson): Switch to a multithreaded equivalent.
    lower_factor_->FillIntersectionSizes(ordering_.supernode_sizes,
                                         supernode_member_to_index_);
  }

  MultithreadedFillNonzeros(matrix, ordering_, supernode_member_to_index_,
                            lower_factor_.get(), diagonal_factor_.get());
}
#endif  // ifdef _OPENMP

template <class Field>
void Factorization<Field>::LeftLookingSupernodeUpdate(
    Int main_supernode, const CoordinateMatrix<Field>& matrix,
    LeftLookingSharedState* shared_state, PrivateState* private_state) {
  BlasMatrix<Field>& main_diagonal_block =
      diagonal_factor_->blocks[main_supernode];
  BlasMatrix<Field>& main_lower_block = lower_factor_->blocks[main_supernode];
  const Int main_supernode_size = main_lower_block.width;

  private_state->pattern_flags[main_supernode] = main_supernode;

  shared_state->rel_rows[main_supernode] = 0;
  shared_state->intersect_ptrs[main_supernode] =
      lower_factor_->IntersectionSizesBeg(main_supernode);

  // Compute the supernodal row pattern.
  const Int num_packed = ComputeRowPattern(
      matrix, ordering_.permutation, ordering_.inverse_permutation,
      ordering_.supernode_sizes, ordering_.supernode_offsets,
      supernode_member_to_index_, ordering_.assembly_forest.parents,
      main_supernode, private_state->pattern_flags.Data(),
      private_state->row_structure.Data());

  // for J = find(L(K, :))
  //   L(K:n, K) -= L(K:n, J) * (D(J, J) * L(K, J)')
  for (Int index = 0; index < num_packed; ++index) {
    const Int descendant_supernode = private_state->row_structure[index];
    CATAMARI_ASSERT(descendant_supernode < main_supernode,
                    "Looking into upper triangle");
    const ConstBlasMatrix<Field>& descendant_lower_block =
        lower_factor_->blocks[descendant_supernode];
    const Int descendant_degree = descendant_lower_block.height;
    const Int descendant_supernode_size = descendant_lower_block.width;

    const Int descendant_main_rel_row =
        shared_state->rel_rows[descendant_supernode];
    const Int descendant_main_intersect_size =
        *shared_state->intersect_ptrs[descendant_supernode];

    const ConstBlasMatrix<Field> descendant_main_matrix =
        descendant_lower_block.Submatrix(descendant_main_rel_row, 0,
                                         descendant_main_intersect_size,
                                         descendant_supernode_size);

    BlasMatrix<Field> scaled_transpose;
    scaled_transpose.height = descendant_supernode_size;
    scaled_transpose.width = descendant_main_intersect_size;
    scaled_transpose.leading_dim = descendant_supernode_size;
    scaled_transpose.data = private_state->scaled_transpose_buffer.Data();

    FormScaledTranspose(
        factorization_type_,
        diagonal_factor_->blocks[descendant_supernode].ToConst(),
        descendant_main_matrix, &scaled_transpose);

    BlasMatrix<Field> workspace_matrix;
    workspace_matrix.height = descendant_main_intersect_size;
    workspace_matrix.width = descendant_main_intersect_size;
    workspace_matrix.leading_dim = descendant_main_intersect_size;
    workspace_matrix.data = private_state->workspace_buffer.Data();

    UpdateDiagonalBlock(factorization_type_, ordering_.supernode_offsets,
                        *lower_factor_, main_supernode, descendant_supernode,
                        descendant_main_rel_row, descendant_main_matrix,
                        scaled_transpose.ToConst(), &main_diagonal_block,
                        &workspace_matrix);

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

      SeekForMainActiveRelativeRow(
          main_supernode, descendant_supernode, descendant_active_rel_row,
          supernode_member_to_index_, *lower_factor_, &main_active_rel_row,
          &main_active_intersect_sizes);
      const Int main_active_intersect_size = *main_active_intersect_sizes;

      const ConstBlasMatrix<Field> descendant_active_matrix =
          descendant_lower_block.Submatrix(descendant_active_rel_row, 0,
                                           descendant_active_intersect_size,
                                           descendant_supernode_size);

      // The width of the workspace matrix and pointer are already correct.
      workspace_matrix.height = descendant_active_intersect_size;
      workspace_matrix.leading_dim = descendant_active_intersect_size;

      BlasMatrix<Field> main_active_block = main_lower_block.Submatrix(
          main_active_rel_row, 0, main_active_intersect_size,
          main_supernode_size);

      UpdateSubdiagonalBlock(
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

template <class Field>
bool Factorization<Field>::LeftLookingSupernodeFinalize(Int main_supernode,
                                                        LDLResult* result) {
  BlasMatrix<Field>& main_diagonal_block =
      diagonal_factor_->blocks[main_supernode];
  BlasMatrix<Field>& main_lower_block = lower_factor_->blocks[main_supernode];
  const Int main_degree = main_lower_block.height;
  const Int main_supernode_size = main_lower_block.width;

  const Int num_supernode_pivots = FactorDiagonalBlock(
      block_size_, factorization_type_, &main_diagonal_block);
  result->num_successful_pivots += num_supernode_pivots;
  if (num_supernode_pivots < main_supernode_size) {
    return false;
  }
  IncorporateSupernodeIntoLDLResult(main_supernode_size, main_degree, result);
  if (!main_degree) {
    return true;
  }

  CATAMARI_ASSERT(main_supernode_size > 0, "Supernode size was non-positive.");
  SolveAgainstDiagonalBlock(factorization_type_, main_diagonal_block.ToConst(),
                            &main_lower_block);

  return true;
}

template <class Field>
LDLResult Factorization<Field>::LeftLooking(
    const CoordinateMatrix<Field>& matrix, const Control& control) {
#ifdef _OPENMP
  if (omp_get_max_threads() > 1) {
    return MultithreadedLeftLooking(matrix, control);
  }
#endif
  algorithm_ = kLeftLookingLDL;

  {
    AssemblyForest forest;
    Buffer<Int> supernode_degrees;
    FormSupernodes(matrix, control.relaxation_control, &forest,
                   &supernode_degrees);
    InitializeFactors(matrix, forest, supernode_degrees);
  }
  const Int num_supernodes = ordering_.supernode_sizes.Size();

  LeftLookingSharedState shared_state;
  shared_state.rel_rows.Resize(num_supernodes);
  shared_state.intersect_ptrs.Resize(num_supernodes);

  PrivateState private_state;
  private_state.row_structure.Resize(num_supernodes);
  private_state.pattern_flags.Resize(num_supernodes);
  private_state.scaled_transpose_buffer.Resize(
      max_supernode_size_ * max_supernode_size_, Field{0});
  private_state.workspace_buffer.Resize(
      max_supernode_size_ * (max_supernode_size_ - 1), Field{0});

  LDLResult result;

  // Note that any postordering of the supernodal elimination forest suffices.
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    LeftLookingSupernodeUpdate(supernode, matrix, &shared_state,
                               &private_state);
    const bool succeeded = LeftLookingSupernodeFinalize(supernode, &result);
    if (!succeeded) {
      return result;
    }
  }

  return result;
}

template <class Field>
void Factorization<Field>::MergeChildSchurComplements(
    Int supernode, RightLookingSharedState* shared_state) {
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;
  BlasMatrix<Field> lower_block = lower_factor_->blocks[supernode];
  BlasMatrix<Field> diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrix<Field> schur_complement =
      shared_state->schur_complements[supernode];

  const Int supernode_size = ordering_.supernode_sizes[supernode];
  const Int supernode_start = ordering_.supernode_offsets[supernode];
  const Int* main_indices = lower_factor_->StructureBeg(supernode);
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    const Int* child_indices = lower_factor_->StructureBeg(child);
    Buffer<Field>& child_schur_complement_buffer =
        shared_state->schur_complement_buffers[child];
    BlasMatrix<Field>& child_schur_complement =
        shared_state->schur_complements[child];
    const Int child_degree = child_schur_complement.height;

    // Fill the mapping from the child structure into the parent front.
    Int num_child_diag_indices = 0;
    Buffer<Int> child_rel_indices(child_degree);
    {
      Int i_rel = supernode_size;
      for (Int i = 0; i < child_degree; ++i) {
        const Int row = child_indices[i];
        if (row < supernode_start + supernode_size) {
          child_rel_indices[i] = row - supernode_start;
          ++num_child_diag_indices;
        } else {
          while (main_indices[i_rel - supernode_size] != row) {
            ++i_rel;
            CATAMARI_ASSERT(i_rel < supernode_size + schur_complement.height,
                            "Relative index is out-of-bounds.");
          }
          child_rel_indices[i] = i_rel;
        }
      }
    }

    // Add the child Schur complement into this supernode's front.
    for (Int j = 0; j < child_degree; ++j) {
      const Int j_rel = child_rel_indices[j];
      if (j < num_child_diag_indices) {
        // Contribute into the upper-left diagonal block of the front.
        for (Int i = j; i < num_child_diag_indices; ++i) {
          const Int i_rel = child_rel_indices[i];
          diagonal_block(i_rel, j_rel) += child_schur_complement(i, j);
        }

        // Contribute into the lower-left block of the front.
        for (Int i = num_child_diag_indices; i < child_degree; ++i) {
          const Int i_rel = child_rel_indices[i];
          lower_block(i_rel - supernode_size, j_rel) +=
              child_schur_complement(i, j);
        }
      } else {
        // Contribute into the bottom-right block of the front.
        for (Int i = j; i < child_degree; ++i) {
          const Int i_rel = child_rel_indices[i];
          schur_complement(i_rel - supernode_size, j_rel - supernode_size) +=
              child_schur_complement(i, j);
        }
      }
    }

    child_schur_complement.height = 0;
    child_schur_complement.width = 0;
    child_schur_complement.data = nullptr;
    child_schur_complement_buffer.Clear();
  }
}

#ifdef _OPENMP
template <class Field>
void Factorization<Field>::MultithreadedMergeChildSchurComplements(
    Int supernode, RightLookingSharedState* shared_state) {
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;
  BlasMatrix<Field> lower_block = lower_factor_->blocks[supernode];
  BlasMatrix<Field> diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrix<Field> schur_complement =
      shared_state->schur_complements[supernode];

  const Int supernode_size = ordering_.supernode_sizes[supernode];
  const Int supernode_start = ordering_.supernode_offsets[supernode];
  const Int* main_indices = lower_factor_->StructureBeg(supernode);
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    const Int* child_indices = lower_factor_->StructureBeg(child);
    Buffer<Field>& child_schur_complement_buffer =
        shared_state->schur_complement_buffers[child];
    BlasMatrix<Field> child_schur_complement =
        shared_state->schur_complements[child];
    const Int child_degree = child_schur_complement.height;

    // Fill the mapping from the child structure into the parent front.
    Int num_child_diag_indices = 0;
    Buffer<Int> child_rel_indices(child_degree);
    {
      Int i_rel = supernode_size;
      for (Int i = 0; i < child_degree; ++i) {
        const Int row = child_indices[i];
        if (row < supernode_start + supernode_size) {
          child_rel_indices[i] = row - supernode_start;
          ++num_child_diag_indices;
        } else {
          while (main_indices[i_rel - supernode_size] != row) {
            ++i_rel;
            CATAMARI_ASSERT(i_rel < supernode_size + schur_complement.height,
                            "Relative index is out-of-bounds.");
          }
          child_rel_indices[i] = i_rel;
        }
      }
    }
    const Int* child_rel_indices_ptr = child_rel_indices.Data();

    // Add the child Schur complement into this supernode's front.
    #pragma omp taskgroup
    for (Int j_beg = 0; j_beg < child_degree; j_beg += merge_grain_size_) {
      #pragma omp task default(none)                                           \
          firstprivate(j_beg, child_degree, diagonal_block, lower_block,       \
              child_schur_complement, schur_complement, child_rel_indices_ptr, \
              num_child_diag_indices, supernode_size)
      {
        const Int j_end = std::min(child_degree, j_beg + merge_grain_size_);
        for (Int j = j_beg; j < j_end; ++j) {
          const Int j_rel = child_rel_indices_ptr[j];
          if (j < num_child_diag_indices) {
            // Contribute into the upper-left diagonal block of the front.
            for (Int i = j; i < num_child_diag_indices; ++i) {
              const Int i_rel = child_rel_indices_ptr[i];
              diagonal_block(i_rel, j_rel) += child_schur_complement(i, j);
            }

            // Contribute into the lower-left block of the front.
            for (Int i = num_child_diag_indices; i < child_degree; ++i) {
              const Int i_rel = child_rel_indices_ptr[i];
              lower_block(i_rel - supernode_size, j_rel) +=
                  child_schur_complement(i, j);
            }
          } else {
            // Contribute into the bottom-right block of the front.
            for (Int i = j; i < child_degree; ++i) {
              const Int i_rel = child_rel_indices_ptr[i];
              schur_complement(i_rel - supernode_size,
                               j_rel - supernode_size) +=
                  child_schur_complement(i, j);
            }
          }
        }
      }
    }

    shared_state->schur_complements[child].height = 0;
    shared_state->schur_complements[child].width = 0;
    shared_state->schur_complements[child].data = nullptr;
    child_schur_complement_buffer.Clear();
  }
}
#endif  // ifdef _OPENMP

template <class Field>
bool Factorization<Field>::RightLookingSupernodeFinalize(
    Int supernode, RightLookingSharedState* shared_state, LDLResult* result) {
  typedef ComplexBase<Field> Real;
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

  MergeChildSchurComplements(supernode, shared_state);

  const Int num_supernode_pivots =
      FactorDiagonalBlock(block_size_, factorization_type_, &diagonal_block);
  result->num_successful_pivots += num_supernode_pivots;
  if (num_supernode_pivots < supernode_size) {
    return false;
  }
  IncorporateSupernodeIntoLDLResult(supernode_size, degree, result);

  if (!degree) {
    // We can early exit.
    return true;
  }

  CATAMARI_ASSERT(supernode_size > 0, "Supernode size was non-positive.");
  SolveAgainstDiagonalBlock(factorization_type_, diagonal_block.ToConst(),
                            &lower_block);

  if (factorization_type_ == kCholeskyFactorization) {
    LowerNormalHermitianOuterProduct(Real{-1}, lower_block.ToConst(), Real{1},
                                     &schur_complement);
  } else {
    // TODO(Jack Poulson): See if this can be preallocated.
    Buffer<Field> scaled_transpose_buffer(degree * supernode_size);
    BlasMatrix<Field> scaled_transpose;
    scaled_transpose.height = supernode_size;
    scaled_transpose.width = degree;
    scaled_transpose.leading_dim = supernode_size;
    scaled_transpose.data = scaled_transpose_buffer.Data();
    FormScaledTranspose(factorization_type_, diagonal_block.ToConst(),
                        lower_block.ToConst(), &scaled_transpose);
    MatrixMultiplyLowerNormalNormal(Field{-1}, lower_block.ToConst(),
                                    scaled_transpose.ToConst(), Field{1},
                                    &schur_complement);
  }

  return true;
}

#ifdef _OPENMP
template <class Field>
bool Factorization<Field>::MultithreadedRightLookingSupernodeFinalize(
    Int supernode, RightLookingSharedState* shared_state, LDLResult* result) {
  typedef ComplexBase<Field> Real;
  BlasMatrix<Field> diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrix<Field> lower_block = lower_factor_->blocks[supernode];
  const Int degree = lower_block.height;
  const Int supernode_size = lower_block.width;

  // Initialize this supernode's Schur complement as the zero matrix.
  Buffer<Field>& schur_complement_buffer =
      shared_state->schur_complement_buffers[supernode];
  {
    BlasMatrix<Field>& schur_complement =
        shared_state->schur_complements[supernode];
    schur_complement_buffer.Resize(degree * degree, Field{0});
    schur_complement.height = degree;
    schur_complement.width = degree;
    schur_complement.leading_dim = degree;
    schur_complement.data = schur_complement_buffer.Data();
  }
  BlasMatrix<Field> schur_complement =
      shared_state->schur_complements[supernode];

  MultithreadedMergeChildSchurComplements(supernode, shared_state);

  Int num_supernode_pivots;
  {
    Buffer<Field> multithreaded_buffer(supernode_size * supernode_size);
    #pragma omp taskgroup
    {
      num_supernode_pivots = MultithreadedFactorDiagonalBlock(
          factor_tile_size_, block_size_, factorization_type_, &diagonal_block,
          &multithreaded_buffer);
      result->num_successful_pivots += num_supernode_pivots;
    }
  }
  if (num_supernode_pivots < supernode_size) {
    return false;
  }
  IncorporateSupernodeIntoLDLResult(supernode_size, degree, result);

  if (!degree) {
    // We can early exit.
    return true;
  }

  CATAMARI_ASSERT(supernode_size > 0, "Supernode size was non-positive.");
  #pragma omp taskgroup
  MultithreadedSolveAgainstDiagonalBlock(
      outer_product_tile_size_, factorization_type_, diagonal_block.ToConst(),
      &lower_block);

  if (factorization_type_ == kCholeskyFactorization) {
    #pragma omp taskgroup
    MultithreadedLowerNormalHermitianOuterProduct(
        outer_product_tile_size_, Real{-1}, lower_block.ToConst(), Real{1},
        &schur_complement);
  } else {
    // TODO(Jack Poulson): See if this can be preallocated.
    Buffer<Field> scaled_transpose_buffer(degree * supernode_size);
    BlasMatrix<Field> scaled_transpose;
    scaled_transpose.height = supernode_size;
    scaled_transpose.width = degree;
    scaled_transpose.leading_dim = supernode_size;
    scaled_transpose.data = scaled_transpose_buffer.Data();

    #pragma omp taskgroup
    MultithreadedFormScaledTranspose(
        outer_product_tile_size_, factorization_type_, diagonal_block.ToConst(),
        lower_block.ToConst(), &scaled_transpose);

    // Perform the multi-threaded MatrixMultiplyLowerNormalNormal.
    #pragma omp taskgroup
    MultithreadedMatrixMultiplyLowerNormalNormal(
        outer_product_tile_size_, Field{-1}, lower_block.ToConst(),
        scaled_transpose.ToConst(), Field{1}, &schur_complement);
  }

  return true;
}
#endif  // ifdef _OPENMP

template <class Field>
bool Factorization<Field>::RightLookingSubtree(
    Int supernode, const CoordinateMatrix<Field>& matrix,
    RightLookingSharedState* shared_state, LDLResult* result) {
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;

  Buffer<int> successes(num_children);
  Buffer<LDLResult> result_contributions(num_children);

  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    CATAMARI_ASSERT(ordering_.assembly_forest.parents[child] == supernode,
                    "Incorrect child index");
    LDLResult& result_contribution = result_contributions[child_index];
    successes[child_index] =
        RightLookingSubtree(child, matrix, shared_state, &result_contribution);
  }

  bool succeeded = true;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    if (!successes[child_index]) {
      succeeded = false;
      break;
    }
    const LDLResult& contribution = result_contributions[child_index];

    // TODO(Jack Poulson): Switch to calling a reduction routine.
    result->num_successful_pivots += contribution.num_successful_pivots;
    result->largest_supernode =
        std::max(result->largest_supernode, contribution.largest_supernode);
    result->num_factorization_entries += contribution.num_factorization_entries;
    result->num_factorization_flops += contribution.num_factorization_flops;
  }

  if (succeeded) {
    succeeded = RightLookingSupernodeFinalize(supernode, shared_state, result);
  } else {
    // Clear the child fronts.
    for (Int child_index = 0; child_index < num_children; ++child_index) {
      const Int child =
          ordering_.assembly_forest.children[child_beg + child_index];
      Buffer<Field>& child_schur_complement_buffer =
          shared_state->schur_complement_buffers[child];
      BlasMatrix<Field>& child_schur_complement =
          shared_state->schur_complements[child];
      child_schur_complement.height = 0;
      child_schur_complement.width = 0;
      child_schur_complement.data = nullptr;
      child_schur_complement_buffer.Clear();
    }
  }

  return succeeded;
}

#ifdef _OPENMP
template <class Field>
bool Factorization<Field>::MultithreadedRightLookingSubtree(
    Int level, Int max_parallel_levels, Int supernode,
    const CoordinateMatrix<Field>& matrix, const Buffer<double>& work_estimates,
    RightLookingSharedState* shared_state, LDLResult* result) {
  if (level >= max_parallel_levels) {
    return RightLookingSubtree(supernode, matrix, shared_state, result);
  }

  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;

  Buffer<int> successes(num_children);
  Buffer<LDLResult> result_contributions(num_children);

  #pragma omp taskgroup
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    #pragma omp task default(none)                                            \
      firstprivate(level, max_parallel_levels, supernode, child, child_index, \
          shared_state)                                                       \
      shared(successes, matrix, result_contributions, work_estimates)
    {
      LDLResult& result_contribution = result_contributions[child_index];
      successes[child_index] = MultithreadedRightLookingSubtree(
          level + 1, max_parallel_levels, child, matrix, work_estimates,
          shared_state, &result_contribution);
    }
  }

  bool succeeded = true;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    if (!successes[child_index]) {
      succeeded = false;
      break;
    }
    const LDLResult& contribution = result_contributions[child_index];

    // TODO(Jack Poulson): Switch to calling a reduction routine.
    result->num_successful_pivots += contribution.num_successful_pivots;
    result->largest_supernode =
        std::max(result->largest_supernode, contribution.largest_supernode);
    result->num_factorization_entries += contribution.num_factorization_entries;
    result->num_factorization_flops += contribution.num_factorization_flops;
  }

  if (succeeded) {
    #pragma omp taskgroup
    succeeded = MultithreadedRightLookingSupernodeFinalize(
        supernode, shared_state, result);
  } else {
    // Clear the child fronts.
    for (Int child_index = 0; child_index < num_children; ++child_index) {
      const Int child =
          ordering_.assembly_forest.children[child_beg + child_index];
      Buffer<Field>& child_schur_complement_buffer =
          shared_state->schur_complement_buffers[child];
      BlasMatrix<Field>& child_schur_complement =
          shared_state->schur_complements[child];
      child_schur_complement.height = 0;
      child_schur_complement.width = 0;
      child_schur_complement.data = nullptr;
      child_schur_complement_buffer.Clear();
    }
  }

  return succeeded;
}
#endif  // ifdef _OPENMP

template <class Field>
LDLResult Factorization<Field>::RightLooking(
    const CoordinateMatrix<Field>& matrix, const Control& control) {
#ifdef _OPENMP
  if (omp_get_max_threads() > 1) {
    return MultithreadedRightLooking(matrix, control);
  }
#endif
  algorithm_ = kRightLookingLDL;

  {
    AssemblyForest forest;
    Buffer<Int> supernode_degrees;
    FormSupernodes(matrix, control.relaxation_control, &forest,
                   &supernode_degrees);
    InitializeFactors(matrix, forest, supernode_degrees);
  }
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const Int num_roots = ordering_.assembly_forest.roots.Size();

  RightLookingSharedState shared_state;
  shared_state.schur_complement_buffers.Resize(num_supernodes);
  shared_state.schur_complements.Resize(num_supernodes);

  LDLResult result;

  Buffer<int> successes(num_roots);
  Buffer<LDLResult> result_contributions(num_roots);

  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = ordering_.assembly_forest.roots[root_index];
    LDLResult& result_contribution = result_contributions[root_index];
    successes[root_index] =
        RightLookingSubtree(root, matrix, &shared_state, &result_contribution);
  }

  for (Int index = 0; index < num_roots; ++index) {
    if (!successes[index]) {
      break;
    }

    const LDLResult& contribution = result_contributions[index];

    // TODO(Jack Poulson): Switch to calling a reduction routine.
    result.num_successful_pivots += contribution.num_successful_pivots;
    result.largest_supernode =
        std::max(result.largest_supernode, contribution.largest_supernode);
    result.num_factorization_entries += contribution.num_factorization_entries;
    result.num_factorization_flops += contribution.num_factorization_flops;
  }

  return result;
}

template <class Field>
bool Factorization<Field>::LeftLookingSubtree(
    Int supernode, const CoordinateMatrix<Field>& matrix,
    LeftLookingSharedState* shared_state, PrivateState* private_state,
    LDLResult* result) {
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;

  Buffer<int> successes(num_children);
  Buffer<LDLResult> result_contributions(num_children);

  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    CATAMARI_ASSERT(ordering_.assembly_forest.parents[child] == supernode,
                    "Incorrect child index");

    LDLResult& result_contribution = result_contributions[child_index];
    successes[child_index] = LeftLookingSubtree(
        child, matrix, shared_state, private_state, &result_contribution);
  }

  bool succeeded = true;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    if (!successes[child_index]) {
      succeeded = false;
      break;
    }
    const LDLResult& contribution = result_contributions[child_index];

    // TODO(Jack Poulson): Switch to calling a reduction routine.
    result->num_successful_pivots += contribution.num_successful_pivots;
    result->largest_supernode =
        std::max(result->largest_supernode, contribution.largest_supernode);
    result->num_factorization_entries += contribution.num_factorization_entries;
    result->num_factorization_flops += contribution.num_factorization_flops;
  }

  if (succeeded) {
    LeftLookingSupernodeUpdate(supernode, matrix, shared_state, private_state);
    succeeded = LeftLookingSupernodeFinalize(supernode, result);
  }

  return succeeded;
}

#ifdef _OPENMP
template <class Field>
void Factorization<Field>::MultithreadedLeftLookingSupernodeUpdate(
    Int main_supernode, const CoordinateMatrix<Field>& matrix,
    LeftLookingSharedState* shared_state,
    Buffer<PrivateState>* private_states) {
  BlasMatrix<Field> main_diagonal_block =
      diagonal_factor_->blocks[main_supernode];
  BlasMatrix<Field> main_lower_block = lower_factor_->blocks[main_supernode];
  const Int main_supernode_size = main_lower_block.width;

  shared_state->rel_rows[main_supernode] = 0;
  shared_state->intersect_ptrs[main_supernode] =
      lower_factor_->IntersectionSizesBeg(main_supernode);

  const int main_thread = omp_get_thread_num();
  Buffer<Int>& pattern_flags = (*private_states)[main_thread].pattern_flags;
  Buffer<Int>& row_structure = (*private_states)[main_thread].row_structure;

  // Compute the supernodal row pattern.
  pattern_flags[main_supernode] = main_supernode;
  const Int num_packed = ComputeRowPattern(
      matrix, ordering_.permutation, ordering_.inverse_permutation,
      ordering_.supernode_sizes, ordering_.supernode_offsets,
      supernode_member_to_index_, ordering_.assembly_forest.parents,
      main_supernode, pattern_flags.Data(), row_structure.Data());

  // OpenMP pragmas cannot operate on object members or function results.
  const SymmetricFactorizationType factorization_type_copy =
      factorization_type_;
  const Buffer<Int>& supernode_offsets_ref = ordering_.supernode_offsets;
  const Buffer<Int>& supernode_member_to_index_ref = supernode_member_to_index_;
  LowerFactor<Field>* const lower_factor_ptr = lower_factor_.get();
  Field* const main_diagonal_block_data CATAMARI_UNUSED =
      main_diagonal_block.data;
  Field* const main_lower_block_data = main_lower_block.data;

  // for J = find(L(K, :))
  //   L(K:n, K) -= L(K:n, J) * (D(J, J) * L(K, J)')
  for (Int index = 0; index < num_packed; ++index) {
    const Int descendant_supernode = row_structure[index];
    CATAMARI_ASSERT(descendant_supernode < main_supernode,
                    "Looking into upper triangle (multithreaded).");
    const ConstBlasMatrix<Field> descendant_lower_block =
        lower_factor_ptr->blocks[descendant_supernode];
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

      BlasMatrix<Field> scaled_transpose;
      scaled_transpose.height = descendant_supernode_size;
      scaled_transpose.width = descendant_main_intersect_size;
      scaled_transpose.leading_dim = descendant_supernode_size;
      scaled_transpose.data = private_state.scaled_transpose_buffer.Data();
      FormScaledTranspose(factorization_type_copy, descendant_diag_block,
                          descendant_main_matrix, &scaled_transpose);

      BlasMatrix<Field> workspace_matrix;
      workspace_matrix.height = descendant_main_intersect_size;
      workspace_matrix.width = descendant_main_intersect_size;
      workspace_matrix.leading_dim = descendant_main_intersect_size;
      workspace_matrix.data = private_state.workspace_buffer.Data();

      UpdateDiagonalBlock(factorization_type_copy, supernode_offsets_ref,
                          *lower_factor_ptr, main_supernode,
                          descendant_supernode, descendant_main_rel_row,
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
        lower_factor_ptr->IntersectionSizesBeg(main_supernode);
    Int main_active_rel_row = 0;
    while (descendant_active_rel_row != descendant_degree) {
      const Int descendant_active_intersect_size =
          *descendant_active_intersect_size_beg;

      SeekForMainActiveRelativeRow(
          main_supernode, descendant_supernode, descendant_active_rel_row,
          supernode_member_to_index_ref, *lower_factor_ptr,
          &main_active_rel_row, &main_active_intersect_sizes);
      const Int main_active_intersect_size = *main_active_intersect_sizes;

      #pragma omp task default(none)                                         \
          firstprivate(index, descendant_supernode,                          \
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
        scaled_transpose.data = private_state.scaled_transpose_buffer.Data();
        FormScaledTranspose(factorization_type_copy, descendant_diag_block,
                            descendant_main_matrix, &scaled_transpose);

        BlasMatrix<Field> main_active_block = main_lower_block.Submatrix(
            main_active_rel_row, 0, main_active_intersect_size,
            main_supernode_size);

        BlasMatrix<Field> workspace_matrix;
        workspace_matrix.height = descendant_active_intersect_size;
        workspace_matrix.width = descendant_main_intersect_size;
        workspace_matrix.leading_dim = descendant_active_intersect_size;
        workspace_matrix.data = private_state.workspace_buffer.Data();

        UpdateSubdiagonalBlock(
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

template <class Field>
bool Factorization<Field>::MultithreadedLeftLookingSupernodeFinalize(
    Int supernode, Buffer<PrivateState>* private_states, LDLResult* result) {
  BlasMatrix<Field>& diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrix<Field>& lower_block = lower_factor_->blocks[supernode];
  const Int degree = lower_block.height;
  const Int supernode_size = lower_block.width;

  Int num_supernode_pivots;
  #pragma omp taskgroup
  {
    const int thread = omp_get_thread_num();
    Buffer<Field>* buffer = &(*private_states)[thread].scaled_transpose_buffer;

    num_supernode_pivots = MultithreadedFactorDiagonalBlock(
        factor_tile_size_, block_size_, factorization_type_, &diagonal_block,
        buffer);
    result->num_successful_pivots += num_supernode_pivots;
  }
  if (num_supernode_pivots < supernode_size) {
    return false;
  }
  IncorporateSupernodeIntoLDLResult(supernode_size, degree, result);
  if (!degree) {
    return true;
  }

  CATAMARI_ASSERT(supernode_size > 0, "Supernode size was non-positive.");
  #pragma omp taskgroup
  MultithreadedSolveAgainstDiagonalBlock(
      outer_product_tile_size_, factorization_type_, diagonal_block.ToConst(),
      &lower_block);

  return true;
}

template <class Field>
bool Factorization<Field>::MultithreadedLeftLookingSubtree(
    Int level, Int max_parallel_levels, Int supernode,
    const CoordinateMatrix<Field>& matrix, LeftLookingSharedState* shared_state,
    Buffer<PrivateState>* private_states, LDLResult* result) {
  if (level >= max_parallel_levels) {
    const int thread = omp_get_thread_num();
    return LeftLookingSubtree(supernode, matrix, shared_state,
                              &(*private_states)[thread], result);
  }

  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;

  Buffer<int> successes(num_children);
  Buffer<LDLResult> result_contributions(num_children);

  #pragma omp taskgroup
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    #pragma omp task default(none)                                       \
        firstprivate(level, max_parallel_levels, supernode, child_index, \
            child, shared_state, private_states)                         \
        shared(successes, matrix, result_contributions)
    {
      LDLResult& result_contribution = result_contributions[child_index];
      successes[child_index] = MultithreadedLeftLookingSubtree(
          level + 1, max_parallel_levels, child, matrix, shared_state,
          private_states, &result_contribution);
    }
  }

  bool succeeded = true;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    if (!successes[child_index]) {
      succeeded = false;
      break;
    }
    const LDLResult& contribution = result_contributions[child_index];

    // TODO(Jack Poulson): Switch to calling a reduction routine.
    result->num_successful_pivots += contribution.num_successful_pivots;
    result->largest_supernode =
        std::max(result->largest_supernode, contribution.largest_supernode);
    result->num_factorization_entries += contribution.num_factorization_entries;
    result->num_factorization_flops += contribution.num_factorization_flops;
  }

  if (succeeded) {
    // Handle the current supernode's elimination.
    #pragma omp taskgroup
    MultithreadedLeftLookingSupernodeUpdate(supernode, matrix, shared_state,
                                            private_states);

    #pragma omp taskgroup
    succeeded = MultithreadedLeftLookingSupernodeFinalize(
        supernode, private_states, result);
  }

  return succeeded;
}

template <class Field>
LDLResult Factorization<Field>::MultithreadedLeftLooking(
    const CoordinateMatrix<Field>& matrix, const Control& control) {
  algorithm_ = kLeftLookingLDL;

  {
    AssemblyForest forest;
    Buffer<Int> supernode_degrees;

    #pragma omp parallel
    #pragma omp single
    {
      #pragma omp taskgroup
      MultithreadedFormSupernodes(matrix, control.relaxation_control, &forest,
                                  &supernode_degrees);

      #pragma omp taskgroup
      MultithreadedInitializeFactors(matrix, forest, supernode_degrees);
    }
  }
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  const int max_threads = omp_get_max_threads();

  LeftLookingSharedState shared_state;
  shared_state.rel_rows.Resize(num_supernodes);
  shared_state.intersect_ptrs.Resize(num_supernodes);

  Buffer<PrivateState> private_states(max_threads);
  for (int thread = 0; thread < max_threads; ++thread) {
    PrivateState& private_state = private_states[thread];
    private_state.pattern_flags.Resize(num_supernodes, -1);
    private_state.row_structure.Resize(num_supernodes);

    // TODO(Jack Poulson): Switch to a reasonably-tight upper bound for each
    // thread.
    private_state.scaled_transpose_buffer.Resize(
        max_supernode_size_ * max_supernode_size_, Field{0});

    // TODO(Jack Poulson): Switch to a reasonably-tight upper bound for each
    // thread.
    private_state.workspace_buffer.Resize(
        max_supernode_size_ * (max_supernode_size_ - 1), Field{0});
  }

  // TODO(Jack Poulson): Make this value configurable.
  const Int max_parallel_levels = std::ceil(std::log2(max_threads)) + 3;

  LDLResult result;
  Buffer<int> successes(num_roots);
  Buffer<LDLResult> result_contributions(num_roots);

  const Int level = 0;
  if (max_parallel_levels == 0) {
    for (Int root_index = 0; root_index < num_roots; ++root_index) {
      const Int root = ordering_.assembly_forest.roots[root_index];
      LDLResult& result_contribution = result_contributions[root_index];
      successes[root_index] =
          LeftLookingSubtree(root, matrix, &shared_state, &private_states[0],
                             &result_contribution);
    }
  } else {
    const int old_max_threads = GetMaxBlasThreads();
    SetNumBlasThreads(1);

    #pragma omp parallel
    #pragma omp single
    #pragma omp taskgroup
    for (Int root_index = 0; root_index < num_roots; ++root_index) {
      const Int root = ordering_.assembly_forest.roots[root_index];
      #pragma omp task default(none) firstprivate(root_index, root, level) \
          shared(successes, matrix, result_contributions, shared_state,    \
              private_states)
      {
        LDLResult& result_contribution = result_contributions[root_index];
        successes[root_index] = MultithreadedLeftLookingSubtree(
            level + 1, max_parallel_levels, root, matrix, &shared_state,
            &private_states, &result_contribution);
      }
    }

    SetNumBlasThreads(old_max_threads);
  }

  for (Int index = 0; index < num_roots; ++index) {
    if (!successes[index]) {
      break;
    }

    const LDLResult& contribution = result_contributions[index];

    // TODO(Jack Poulson): Switch to calling a reduction routine.
    result.num_successful_pivots += contribution.num_successful_pivots;
    result.largest_supernode =
        std::max(result.largest_supernode, contribution.largest_supernode);
    result.num_factorization_entries += contribution.num_factorization_entries;
    result.num_factorization_flops += contribution.num_factorization_flops;
  }

  return result;
}

template <class Field>
LDLResult Factorization<Field>::MultithreadedRightLooking(
    const CoordinateMatrix<Field>& matrix, const Control& control) {
  algorithm_ = kRightLookingLDL;

  {
    AssemblyForest forest;
    Buffer<Int> supernode_degrees;

    #pragma omp parallel
    #pragma omp single
    {
      #pragma omp taskgroup
      MultithreadedFormSupernodes(matrix, control.relaxation_control, &forest,
                                  &supernode_degrees);

      #pragma omp taskgroup
      MultithreadedInitializeFactors(matrix, forest, supernode_degrees);
    }
  }

  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  const Int max_threads = omp_get_max_threads();

  RightLookingSharedState shared_state;
  shared_state.schur_complement_buffers.Resize(num_supernodes);
  shared_state.schur_complements.Resize(num_supernodes);

  // Compute flop-count estimates so that we may prioritize the expensive
  // tasks before the cheaper ones.
  Buffer<double> work_estimates(num_supernodes);
  for (const Int& root : ordering_.assembly_forest.roots) {
    FillSubtreeWorkEstimates(root, ordering_.assembly_forest, *lower_factor_,
                             &work_estimates);
  }

  LDLResult result;

  Buffer<int> successes(num_roots);
  Buffer<LDLResult> result_contributions(num_roots);

  // TODO(Jack Poulson): Make this value configurable.
  const Int max_parallel_levels = std::ceil(std::log2(max_threads)) + 3;

  const Int level = 0;
  if (max_parallel_levels == 0) {
    for (Int root_index = 0; root_index < num_roots; ++root_index) {
      const Int root = ordering_.assembly_forest.roots[root_index];
      LDLResult& result_contribution = result_contributions[root_index];
      successes[root_index] = RightLookingSubtree(root, matrix, &shared_state,
                                                  &result_contribution);
    }
  } else {
    const int old_max_threads = GetMaxBlasThreads();
    SetNumBlasThreads(1);

    #pragma omp parallel
    #pragma omp single
    #pragma omp taskgroup
    for (Int root_index = 0; root_index < num_roots; ++root_index) {
      const Int root = ordering_.assembly_forest.roots[root_index];

      // As above, one could make use of OpenMP task priorities, e.g., with an
      // integer priority of:
      //
      //   const Int task_priority = std::pow(work_estimates[child], 0.25);
      //
      #pragma omp task default(none) firstprivate(root, root_index, level) \
          shared(successes, matrix, result_contributions, shared_state,    \
              work_estimates)
      {
        LDLResult& result_contribution = result_contributions[root_index];
        successes[root_index] = MultithreadedRightLookingSubtree(
            level + 1, max_parallel_levels, root, matrix, work_estimates,
            &shared_state, &result_contribution);
      }
    }

    SetNumBlasThreads(old_max_threads);
  }

  for (Int index = 0; index < num_roots; ++index) {
    if (!successes[index]) {
      break;
    }

    const LDLResult& contribution = result_contributions[index];

    // TODO(Jack Poulson): Switch to calling a reduction routine.
    result.num_successful_pivots += contribution.num_successful_pivots;
    result.largest_supernode =
        std::max(result.largest_supernode, contribution.largest_supernode);
    result.num_factorization_entries += contribution.num_factorization_entries;
    result.num_factorization_flops += contribution.num_factorization_flops;
  }

  return result;
}
#endif  // ifdef _OPENMP

template <class Field>
LDLResult Factorization<Field>::Factor(const CoordinateMatrix<Field>& matrix,
                                       const SymmetricOrdering& manual_ordering,
                                       const Control& control) {
  ordering_ = manual_ordering;
  factorization_type_ = control.factorization_type;
  block_size_ = control.block_size;

#ifdef _OPENMP
  factor_tile_size_ = control.factor_tile_size;
  outer_product_tile_size_ = control.outer_product_tile_size;
  merge_grain_size_ = control.merge_grain_size;
  sort_grain_size_ = control.sort_grain_size;
#endif  // ifdef _OPENMP

  forward_solve_out_of_place_supernode_threshold_ =
      control.forward_solve_out_of_place_supernode_threshold;
  backward_solve_out_of_place_supernode_threshold_ =
      control.backward_solve_out_of_place_supernode_threshold;

  const bool use_leftlooking = control.algorithm == kLeftLookingLDL;
  if (use_leftlooking) {
    return LeftLooking(matrix, control);
  } else {
    return RightLooking(matrix, control);
  }
}

template <class Field>
void Factorization<Field>::Solve(BlasMatrix<Field>* matrix) const {
  const bool have_permutation = !ordering_.permutation.Empty();

  // TODO(Jack Poulson): Add multithreaded tree parallelism.

  // Reorder the input into the permutation of the factorization.
  if (have_permutation) {
    Permute(ordering_.permutation, matrix);
  }

  LowerTriangularSolve(matrix);
  DiagonalSolve(matrix);
  LowerTransposeTriangularSolve(matrix);

  // Reverse the factorization permutation.
  if (have_permutation) {
    Permute(ordering_.inverse_permutation, matrix);
  }
}

template <class Field>
void Factorization<Field>::LowerTriangularSolve(
    BlasMatrix<Field>* matrix) const {
  const Int num_rhs = matrix->width;
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const bool is_cholesky = factorization_type_ == kCholeskyFactorization;

  Buffer<Field> workspace(max_degree_ * num_rhs, Field{0});

  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const ConstBlasMatrix<Field> triangular_matrix =
        diagonal_factor_->blocks[supernode];

    const Int supernode_size = ordering_.supernode_sizes[supernode];
    const Int supernode_start = ordering_.supernode_offsets[supernode];
    BlasMatrix<Field> matrix_supernode =
        matrix->Submatrix(supernode_start, 0, supernode_size, num_rhs);

    // Solve against the diagonal block of the supernode.
    if (is_cholesky) {
      LeftLowerTriangularSolves(triangular_matrix, &matrix_supernode);
    } else {
      LeftLowerUnitTriangularSolves(triangular_matrix, &matrix_supernode);
    }

    const ConstBlasMatrix<Field> subdiagonal = lower_factor_->blocks[supernode];
    if (!subdiagonal.height) {
      continue;
    }

    // Handle the external updates for this supernode.
    const Int* indices = lower_factor_->StructureBeg(supernode);
    if (supernode_size >= forward_solve_out_of_place_supernode_threshold_) {
      // Perform an out-of-place GEMM.
      BlasMatrix<Field> work_matrix;
      work_matrix.height = subdiagonal.height;
      work_matrix.width = num_rhs;
      work_matrix.leading_dim = subdiagonal.height;
      work_matrix.data = workspace.Data();

      // Store the updates in the workspace.
      MatrixMultiplyNormalNormal(Field{-1}, subdiagonal,
                                 matrix_supernode.ToConst(), Field{0},
                                 &work_matrix);

      // Accumulate the workspace into the solution matrix.
      for (Int j = 0; j < num_rhs; ++j) {
        for (Int i = 0; i < subdiagonal.height; ++i) {
          const Int row = indices[i];
          matrix->Entry(row, j) += work_matrix(i, j);
        }
      }
    } else {
      for (Int j = 0; j < num_rhs; ++j) {
        for (Int k = 0; k < supernode_size; ++k) {
          const Field& eta = matrix_supernode(k, j);
          for (Int i = 0; i < subdiagonal.height; ++i) {
            const Int row = indices[i];
            matrix->Entry(row, j) -= subdiagonal(i, k) * eta;
          }
        }
      }
    }
  }
}

template <class Field>
void Factorization<Field>::DiagonalSolve(BlasMatrix<Field>* matrix) const {
  const Int num_rhs = matrix->width;
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const bool is_cholesky = factorization_type_ == kCholeskyFactorization;
  if (is_cholesky) {
    // D is the identity.
    return;
  }

  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const ConstBlasMatrix<Field> diagonal_matrix =
        diagonal_factor_->blocks[supernode];

    const Int supernode_size = ordering_.supernode_sizes[supernode];
    const Int supernode_start = ordering_.supernode_offsets[supernode];
    BlasMatrix<Field> matrix_supernode =
        matrix->Submatrix(supernode_start, 0, supernode_size, num_rhs);

    // Handle the diagonal-block portion of the supernode.
    for (Int j = 0; j < num_rhs; ++j) {
      for (Int i = 0; i < supernode_size; ++i) {
        matrix_supernode(i, j) /= diagonal_matrix(i, i);
      }
    }
  }
}

template <class Field>
void Factorization<Field>::LowerTransposeTriangularSolve(
    BlasMatrix<Field>* matrix) const {
  const Int num_rhs = matrix->width;
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const bool is_selfadjoint = factorization_type_ != kLDLTransposeFactorization;

  Buffer<Field> packed_input_buf(max_degree_ * num_rhs);

  for (Int supernode = num_supernodes - 1; supernode >= 0; --supernode) {
    const Int supernode_size = ordering_.supernode_sizes[supernode];
    const Int supernode_start = ordering_.supernode_offsets[supernode];
    const Int* indices = lower_factor_->StructureBeg(supernode);

    BlasMatrix<Field> matrix_supernode =
        matrix->Submatrix(supernode_start, 0, supernode_size, num_rhs);

    const ConstBlasMatrix<Field> subdiagonal = lower_factor_->blocks[supernode];
    if (subdiagonal.height) {
      // Handle the external updates for this supernode.
      if (supernode_size >= backward_solve_out_of_place_supernode_threshold_) {
        // Fill the work matrix.
        BlasMatrix<Field> work_matrix;
        work_matrix.height = subdiagonal.height;
        work_matrix.width = num_rhs;
        work_matrix.leading_dim = subdiagonal.height;
        work_matrix.data = packed_input_buf.Data();
        for (Int j = 0; j < num_rhs; ++j) {
          for (Int i = 0; i < subdiagonal.height; ++i) {
            const Int row = indices[i];
            work_matrix(i, j) = matrix->Entry(row, j);
          }
        }

        if (is_selfadjoint) {
          MatrixMultiplyAdjointNormal(Field{-1}, subdiagonal,
                                      work_matrix.ToConst(), Field{1},
                                      &matrix_supernode);
        } else {
          MatrixMultiplyTransposeNormal(Field{-1}, subdiagonal,
                                        work_matrix.ToConst(), Field{1},
                                        &matrix_supernode);
        }
      } else {
        for (Int k = 0; k < supernode_size; ++k) {
          for (Int i = 0; i < subdiagonal.height; ++i) {
            const Int row = indices[i];
            for (Int j = 0; j < num_rhs; ++j) {
              if (is_selfadjoint) {
                matrix_supernode(k, j) -=
                    Conjugate(subdiagonal(i, k)) * matrix->Entry(row, j);
              } else {
                matrix_supernode(k, j) -=
                    subdiagonal(i, k) * matrix->Entry(row, j);
              }
            }
          }
        }
      }
    }

    // Solve against the diagonal block of this supernode.
    const ConstBlasMatrix<Field> triangular_matrix =
        diagonal_factor_->blocks[supernode];
    if (factorization_type_ == kCholeskyFactorization) {
      LeftLowerAdjointTriangularSolves(triangular_matrix, &matrix_supernode);
    } else if (factorization_type_ == kLDLAdjointFactorization) {
      LeftLowerAdjointUnitTriangularSolves(triangular_matrix,
                                           &matrix_supernode);
    } else {
      LeftLowerTransposeUnitTriangularSolves(triangular_matrix,
                                             &matrix_supernode);
    }
  }
}

template <class Field>
void Factorization<Field>::PrintDiagonalFactor(const std::string& label,
                                               std::ostream& os) const {
  if (factorization_type_ == kCholeskyFactorization) {
    // TODO(Jack Poulson): Print the identity.
    return;
  }

  os << label << ": \n";
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const ConstBlasMatrix<Field>& diag_matrix =
        diagonal_factor_->blocks[supernode];
    for (Int j = 0; j < diag_matrix.height; ++j) {
      os << diag_matrix(j, j) << " ";
    }
  }
  os << std::endl;
}

template <class Field>
void Factorization<Field>::PrintLowerFactor(const std::string& label,
                                            std::ostream& os) const {
  const bool is_cholesky = factorization_type_ == kCholeskyFactorization;

  auto print_entry = [&](const Int& row, const Int& column,
                         const Field& value) {
    os << row << " " << column << " " << value << "\n";
  };

  os << label << ": \n";
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int supernode_start = ordering_.supernode_offsets[supernode];
    const Int* indices = lower_factor_->StructureBeg(supernode);

    const ConstBlasMatrix<Field>& diag_matrix =
        diagonal_factor_->blocks[supernode];
    const ConstBlasMatrix<Field>& lower_matrix =
        lower_factor_->blocks[supernode];

    for (Int j = 0; j < diag_matrix.height; ++j) {
      const Int column = supernode_start + j;

      // Print the portion in the diagonal block.
      if (is_cholesky) {
        print_entry(column, column, diag_matrix(j, j));
      } else {
        print_entry(column, column, Field{1});
      }
      for (Int k = j + 1; k < diag_matrix.height; ++k) {
        const Int row = supernode_start + k;
        print_entry(row, column, diag_matrix(k, j));
      }

      // Print the portion below the diagonal block.
      for (Int i = 0; i < lower_matrix.height; ++i) {
        const Int row = indices[i];
        print_entry(row, column, lower_matrix(i, j));
      }
    }
  }
  os << std::endl;
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_IMPL_H_
