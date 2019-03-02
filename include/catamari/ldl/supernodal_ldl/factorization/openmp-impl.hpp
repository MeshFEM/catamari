/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_OPENMP_IMPL_H_
#define CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_OPENMP_IMPL_H_
#ifdef CATAMARI_OPENMP

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"

#include "catamari/ldl/supernodal_ldl/factorization.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
void Factorization<Field>::OpenMPFormSupernodes(
    const CoordinateMatrix<Field>& matrix,
    const SupernodalRelaxationControl& control, AssemblyForest* forest,
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

  OpenMPFillStructureIndices(sort_grain_size_, matrix, ordering_, forest,
                             supernode_member_to_index_, lower_factor_.get());
  if (algorithm_ == kLeftLookingLDL) {
    // TODO(Jack Poulson): Switch to a multithreaded equivalent.
    lower_factor_->FillIntersectionSizes(ordering_.supernode_sizes,
                                         supernode_member_to_index_);
  }

  OpenMPFillNonzeros(matrix, ordering_, supernode_member_to_index_,
                     lower_factor_.get(), diagonal_factor_.get());
}

template <class Field>
bool Factorization<Field>::OpenMPRightLookingSupernodeFinalize(
    Int supernode, RightLookingSharedState<Field>* shared_state,
    Buffer<PrivateState<Field>>* private_states, LDLResult* result) {
  typedef ComplexBase<Field> Real;
  BlasMatrixView<Field> diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrixView<Field> lower_block = lower_factor_->blocks[supernode];
  const Int degree = lower_block.height;
  const Int supernode_size = lower_block.width;

  const int thread = omp_get_thread_num();
  PrivateState<Field>* private_state = &(*private_states)[thread];

  // Initialize this supernode's Schur complement as the zero matrix.
  Buffer<Field>& schur_complement_buffer =
      shared_state->schur_complement_buffers[supernode];
  {
    BlasMatrixView<Field>& schur_complement =
        shared_state->schur_complements[supernode];
    schur_complement_buffer.Resize(degree * degree, Field{0});
    schur_complement.height = degree;
    schur_complement.width = degree;
    schur_complement.leading_dim = degree;
    schur_complement.data = schur_complement_buffer.Data();
  }
  BlasMatrixView<Field> schur_complement =
      shared_state->schur_complements[supernode];

  OpenMPMergeChildSchurComplements(merge_grain_size_, supernode, ordering_,
                                   lower_factor_.get(), diagonal_factor_.get(),
                                   shared_state);

  Int num_supernode_pivots;
  {
    // TODO(Jack Poulson): Preallocate this buffer.
    Buffer<Field> multithreaded_buffer(supernode_size * supernode_size);
    #pragma omp taskgroup
    {
      num_supernode_pivots = OpenMPFactorDiagonalBlock(
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
  OpenMPSolveAgainstDiagonalBlock(outer_product_tile_size_, factorization_type_,
                                  diagonal_block.ToConst(), &lower_block);

  if (factorization_type_ == kCholeskyFactorization) {
    #pragma omp taskgroup
    OpenMPLowerNormalHermitianOuterProduct(outer_product_tile_size_, Real{-1},
                                           lower_block.ToConst(), Real{1},
                                           &schur_complement);
  } else {
    BlasMatrixView<Field> scaled_transpose;
    scaled_transpose.height = supernode_size;
    scaled_transpose.width = degree;
    scaled_transpose.leading_dim = supernode_size;
    scaled_transpose.data = private_state->scaled_transpose_buffer.Data();

    #pragma omp taskgroup
    OpenMPFormScaledTranspose(outer_product_tile_size_, factorization_type_,
                              diagonal_block.ToConst(), lower_block.ToConst(),
                              &scaled_transpose);

    // Perform the multi-threaded MatrixMultiplyLowerNormalNormal.
    #pragma omp taskgroup
    OpenMPMatrixMultiplyLowerNormalNormal(
        outer_product_tile_size_, Field{-1}, lower_block.ToConst(),
        scaled_transpose.ToConst(), Field{1}, &schur_complement);
  }

  return true;
}

template <class Field>
bool Factorization<Field>::OpenMPRightLookingSubtree(
    Int level, Int max_parallel_levels, Int supernode,
    const CoordinateMatrix<Field>& matrix, const Buffer<double>& work_estimates,
    RightLookingSharedState<Field>* shared_state,
    Buffer<PrivateState<Field>>* private_states, LDLResult* result) {
  if (level >= max_parallel_levels) {
    const int thread = omp_get_thread_num();
    return RightLookingSubtree(supernode, matrix, shared_state,
                               &(*private_states)[thread], result);
  }

  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;

  Buffer<int> successes(num_children);
  Buffer<LDLResult> result_contributions(num_children);

  // Recurse on the children.
  #pragma omp taskgroup
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    #pragma omp task default(none)                                            \
      firstprivate(level, max_parallel_levels, supernode, child, child_index, \
          shared_state, private_states)                                       \
      shared(successes, matrix, result_contributions, work_estimates)
    {
      LDLResult& result_contribution = result_contributions[child_index];
      successes[child_index] = OpenMPRightLookingSubtree(
          level + 1, max_parallel_levels, child, matrix, work_estimates,
          shared_state, private_states, &result_contribution);
    }
  }

  // Merge the child results (stopping if a failure is detected).
  bool succeeded = true;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    if (!successes[child_index]) {
      succeeded = false;
      break;
    }
    MergeContribution(result_contributions[child_index], result);
  }

  if (succeeded) {
    #pragma omp taskgroup
    succeeded = OpenMPRightLookingSupernodeFinalize(supernode, shared_state,
                                                    private_states, result);
  } else {
    // Clear the child fronts.
    for (Int child_index = 0; child_index < num_children; ++child_index) {
      const Int child =
          ordering_.assembly_forest.children[child_beg + child_index];
      Buffer<Field>& child_schur_complement_buffer =
          shared_state->schur_complement_buffers[child];
      BlasMatrixView<Field>& child_schur_complement =
          shared_state->schur_complements[child];
      child_schur_complement.height = 0;
      child_schur_complement.width = 0;
      child_schur_complement.data = nullptr;
      child_schur_complement_buffer.Clear();
    }
  }

  return succeeded;
}

template <class Field>
void Factorization<Field>::OpenMPLeftLookingSupernodeUpdate(
    Int main_supernode, const CoordinateMatrix<Field>& matrix,
    LeftLookingSharedState* shared_state,
    Buffer<PrivateState<Field>>* private_states) {
  BlasMatrixView<Field> main_diagonal_block =
      diagonal_factor_->blocks[main_supernode];
  BlasMatrixView<Field> main_lower_block =
      lower_factor_->blocks[main_supernode];
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
    const ConstBlasMatrixView<Field> descendant_lower_block =
        lower_factor_ptr->blocks[descendant_supernode];
    const Int descendant_degree = descendant_lower_block.height;
    const Int descendant_supernode_size = descendant_lower_block.width;

    const Int descendant_main_rel_row =
        shared_state->rel_rows[descendant_supernode];
    const Int descendant_main_intersect_size =
        *shared_state->intersect_ptrs[descendant_supernode];

    const ConstBlasMatrixView<Field> descendant_diag_block =
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
      PrivateState<Field>& private_state = (*private_states)[thread];

      const ConstBlasMatrixView<Field> descendant_main_matrix =
          descendant_lower_block.Submatrix(descendant_main_rel_row, 0,
                                           descendant_main_intersect_size,
                                           descendant_supernode_size);

      BlasMatrixView<Field> scaled_transpose;
      scaled_transpose.height = descendant_supernode_size;
      scaled_transpose.width = descendant_main_intersect_size;
      scaled_transpose.leading_dim = descendant_supernode_size;
      scaled_transpose.data = private_state.scaled_transpose_buffer.Data();
      FormScaledTranspose(factorization_type_copy, descendant_diag_block,
                          descendant_main_matrix, &scaled_transpose);

      BlasMatrixView<Field> workspace_matrix;
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
        PrivateState<Field>& private_state = (*private_states)[thread];

        const ConstBlasMatrixView<Field> descendant_active_matrix =
            descendant_lower_block.Submatrix(descendant_active_rel_row, 0,
                                             descendant_active_intersect_size,
                                             descendant_supernode_size);

        const ConstBlasMatrixView<Field> descendant_main_matrix =
            descendant_lower_block.Submatrix(descendant_main_rel_row, 0,
                                             descendant_main_intersect_size,
                                             descendant_supernode_size);

        BlasMatrixView<Field> scaled_transpose;
        scaled_transpose.height = descendant_supernode_size;
        scaled_transpose.width = descendant_main_intersect_size;
        scaled_transpose.leading_dim = descendant_supernode_size;
        scaled_transpose.data = private_state.scaled_transpose_buffer.Data();
        FormScaledTranspose(factorization_type_copy, descendant_diag_block,
                            descendant_main_matrix, &scaled_transpose);

        BlasMatrixView<Field> main_active_block = main_lower_block.Submatrix(
            main_active_rel_row, 0, main_active_intersect_size,
            main_supernode_size);

        BlasMatrixView<Field> workspace_matrix;
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
bool Factorization<Field>::OpenMPLeftLookingSupernodeFinalize(
    Int supernode, Buffer<PrivateState<Field>>* private_states,
    LDLResult* result) {
  BlasMatrixView<Field>& diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrixView<Field>& lower_block = lower_factor_->blocks[supernode];
  const Int degree = lower_block.height;
  const Int supernode_size = lower_block.width;

  Int num_supernode_pivots;
  #pragma omp taskgroup
  {
    const int thread = omp_get_thread_num();
    Buffer<Field>* buffer = &(*private_states)[thread].scaled_transpose_buffer;

    num_supernode_pivots =
        OpenMPFactorDiagonalBlock(factor_tile_size_, block_size_,
                                  factorization_type_, &diagonal_block, buffer);
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
  OpenMPSolveAgainstDiagonalBlock(outer_product_tile_size_, factorization_type_,
                                  diagonal_block.ToConst(), &lower_block);

  return true;
}

template <class Field>
bool Factorization<Field>::OpenMPLeftLookingSubtree(
    Int level, Int max_parallel_levels, Int supernode,
    const CoordinateMatrix<Field>& matrix, LeftLookingSharedState* shared_state,
    Buffer<PrivateState<Field>>* private_states, LDLResult* result) {
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
      successes[child_index] = OpenMPLeftLookingSubtree(
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
    MergeContribution(result_contributions[child_index], result);
  }

  if (succeeded) {
    // Handle the current supernode's elimination.
    #pragma omp taskgroup
    OpenMPLeftLookingSupernodeUpdate(supernode, matrix, shared_state,
                                     private_states);

    #pragma omp taskgroup
    succeeded =
        OpenMPLeftLookingSupernodeFinalize(supernode, private_states, result);
  }

  return succeeded;
}

template <class Field>
LDLResult Factorization<Field>::OpenMPLeftLooking(
    const CoordinateMatrix<Field>& matrix, const Control& control) {
  algorithm_ = kLeftLookingLDL;

  {
    AssemblyForest forest;
    Buffer<Int> supernode_degrees;

    #pragma omp parallel
    #pragma omp single
    {
      #pragma omp taskgroup
      OpenMPFormSupernodes(matrix, control.relaxation_control, &forest,
                           &supernode_degrees);

      #pragma omp taskgroup
      OpenMPInitializeFactors(matrix, forest, supernode_degrees);
    }
  }
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  const int max_threads = omp_get_max_threads();

  LeftLookingSharedState shared_state;
  shared_state.rel_rows.Resize(num_supernodes);
  shared_state.intersect_ptrs.Resize(num_supernodes);

  Buffer<PrivateState<Field>> private_states(max_threads);

  const int max_supernode_size = max_supernode_size_;
  #pragma omp taskgroup
  for (int t = 0; t < max_threads; ++t) {
    #pragma omp task default(none)                          \
        firstprivate(t, num_supernodes, max_supernode_size) \
        shared(private_states)
    {
      PrivateState<Field>& private_state = private_states[t];
      private_state.pattern_flags.Resize(num_supernodes, -1);
      private_state.row_structure.Resize(num_supernodes);

      // TODO(Jack Poulson): Switch to a reasonably-tight upper bound for each
      // thread.
      private_state.scaled_transpose_buffer.Resize(
          max_supernode_size * max_supernode_size, Field{0});

      // TODO(Jack Poulson): Switch to a reasonably-tight upper bound for each
      // thread.
      private_state.workspace_buffer.Resize(
          max_supernode_size * (max_supernode_size - 1), Field{0});
    }
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
        successes[root_index] = OpenMPLeftLookingSubtree(
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
    MergeContribution(result_contributions[index], &result);
  }

  return result;
}

template <class Field>
LDLResult Factorization<Field>::OpenMPRightLooking(
    const CoordinateMatrix<Field>& matrix, const Control& control) {
  algorithm_ = kRightLookingLDL;

  {
    AssemblyForest forest;
    Buffer<Int> supernode_degrees;

    #pragma omp parallel
    #pragma omp single
    {
      #pragma omp taskgroup
      OpenMPFormSupernodes(matrix, control.relaxation_control, &forest,
                           &supernode_degrees);

      #pragma omp taskgroup
      OpenMPInitializeFactors(matrix, forest, supernode_degrees);
    }
  }

  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  const Int max_threads = omp_get_max_threads();

  RightLookingSharedState<Field> shared_state;
  shared_state.schur_complement_buffers.Resize(num_supernodes);
  shared_state.schur_complements.Resize(num_supernodes);

  Buffer<PrivateState<Field>> private_states(max_threads);
  if (factorization_type_ != kCholeskyFactorization) {
    const Int workspace_size = max_lower_block_size_;
    #pragma omp taskgroup
    for (int t = 0; t < max_threads; ++t) {
      #pragma omp task default(none) firstprivate(t, workspace_size) \
          shared(private_states)
      private_states[t].scaled_transpose_buffer.Resize(workspace_size);
    }
  }

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

  // Recurse on each tree in the elimination forest.
  const Int level = 0;
  if (max_parallel_levels == 0) {
    const int thread = omp_get_thread_num();
    for (Int root_index = 0; root_index < num_roots; ++root_index) {
      const Int root = ordering_.assembly_forest.roots[root_index];
      LDLResult& result_contribution = result_contributions[root_index];
      successes[root_index] =
          RightLookingSubtree(root, matrix, &shared_state,
                              &private_states[thread], &result_contribution);
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
              private_states, work_estimates)
      {
        LDLResult& result_contribution = result_contributions[root_index];
        successes[root_index] = OpenMPRightLookingSubtree(
            level + 1, max_parallel_levels, root, matrix, work_estimates,
            &shared_state, &private_states, &result_contribution);
      }
    }

    SetNumBlasThreads(old_max_threads);
  }

  for (Int index = 0; index < num_roots; ++index) {
    if (!successes[index]) {
      break;
    }
    MergeContribution(result_contributions[index], &result);
  }

  return result;
}

template <class Field>
void Factorization<Field>::OpenMPLowerSupernodalTrapezoidalSolve(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    RightLookingSharedState<Field>* shared_state) const {
  // Eliminate this supernode.
  const Int num_rhs = right_hand_sides->width;
  const bool is_cholesky = factorization_type_ == kCholeskyFactorization;
  const ConstBlasMatrixView<Field> triangular_right_hand_sides =
      diagonal_factor_->blocks[supernode];

  const Int supernode_size = ordering_.supernode_sizes[supernode];
  const Int supernode_start = ordering_.supernode_offsets[supernode];
  BlasMatrixView<Field> right_hand_sides_supernode =
      right_hand_sides->Submatrix(supernode_start, 0, supernode_size, num_rhs);

  // Solve against the diagonal block of the supernode.
  if (is_cholesky) {
    LeftLowerTriangularSolves(triangular_right_hand_sides,
                              &right_hand_sides_supernode);
  } else {
    LeftLowerUnitTriangularSolves(triangular_right_hand_sides,
                                  &right_hand_sides_supernode);
  }

  const ConstBlasMatrixView<Field> subdiagonal =
      lower_factor_->blocks[supernode];
  if (!subdiagonal.height) {
    return;
  }

  // Store the updates in the workspace.
  MatrixMultiplyNormalNormal(Field{-1}, subdiagonal,
                             right_hand_sides_supernode.ToConst(), Field{1},
                             &shared_state->schur_complements[supernode]);
}

template <class Field>
void Factorization<Field>::OpenMPLowerTriangularSolveRecursion(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    RightLookingSharedState<Field>* shared_state) const {
  // Recurse on this supernode's children.
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;
  #pragma omp taskgroup
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    #pragma omp task default(none) firstprivate(child) \
        shared(right_hand_sides, shared_state)
    OpenMPLowerTriangularSolveRecursion(child, right_hand_sides, shared_state);
  }

  const Int supernode_start = ordering_.supernode_sizes[supernode];
  const Int supernode_size = ordering_.supernode_offsets[supernode];
  const Int* main_indices = lower_factor_->StructureBeg(supernode);

  // Merge the child Schur complements into the parent.
  const Int degree = lower_factor_->blocks[supernode].height;
  const Int num_rhs = right_hand_sides->width;
  const Int workspace_size = degree * num_rhs;
  shared_state->schur_complement_buffers[supernode].Resize(workspace_size,
                                                           Field{0});
  BlasMatrixView<Field>& main_right_hand_sides =
      shared_state->schur_complements[supernode];
  main_right_hand_sides.height = degree;
  main_right_hand_sides.width = num_rhs;
  main_right_hand_sides.leading_dim = degree;
  main_right_hand_sides.data =
      shared_state->schur_complement_buffers[supernode].Data();
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    const Int* child_indices = lower_factor_->StructureBeg(child);
    BlasMatrixView<Field>& child_right_hand_sides =
        shared_state->schur_complements[child];
    const Int child_degree = child_right_hand_sides.height;

    Int i_rel = supernode_size;
    for (Int i = 0; i < child_degree; ++i) {
      const Int row = child_indices[i];
      if (row < supernode_start + supernode_size) {
        for (Int j = 0; j < num_rhs; ++j) {
          right_hand_sides->Entry(row, j) += child_right_hand_sides(i, j);
        }
      } else {
        while (main_indices[i_rel - supernode_size] != row) {
          ++i_rel;
        }
        const Int main_rel = i_rel - supernode_size;
        for (Int j = 0; j < num_rhs; ++j) {
          main_right_hand_sides(main_rel, j) += child_right_hand_sides(i, j);
        }
      }
    }

    shared_state->schur_complement_buffers[child].Clear();
    child_right_hand_sides.height = 0;
    child_right_hand_sides.width = 0;
    child_right_hand_sides.leading_dim = 0;
    child_right_hand_sides.data = nullptr;
  }

  // Perform this supernode's trapezoidal solve.
  OpenMPLowerSupernodalTrapezoidalSolve(supernode, right_hand_sides,
                                        shared_state);
}

template <class Field>
void Factorization<Field>::OpenMPLowerTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  // Set up the shared state.
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  RightLookingSharedState<Field> shared_state;
  shared_state.schur_complement_buffers.Resize(num_supernodes);
  shared_state.schur_complements.Resize(num_supernodes);

  // Recurse on each tree in the elimination forest.
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  #pragma omp taskgroup
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = ordering_.assembly_forest.roots[root_index];
    #pragma omp task default(none) firstprivate(root) \
        shared(right_hand_sides, shared_state)
    OpenMPLowerTriangularSolveRecursion(root, right_hand_sides, &shared_state);
  }
}

template <class Field>
void Factorization<Field>::OpenMPDiagonalSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  if (factorization_type_ == kCholeskyFactorization) {
    // D is the identity.
    return;
  }

  const SymmetricOrdering* ordering_ptr = &ordering_;
  const DiagonalFactor<Field>* diagonal_factor_ptr = diagonal_factor_.get();

  const Int num_supernodes = ordering_.supernode_sizes.Size();
  #pragma omp taskgroup
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    #pragma omp task default(none) firstprivate(supernode) \
        shared(right_hand_sides, ordering_ptr, diagonal_factor_ptr)
    {
      const ConstBlasMatrixView<Field> diagonal_right_hand_sides =
          diagonal_factor_ptr->blocks[supernode];

      const Int num_rhs = right_hand_sides->width;
      const Int supernode_size = ordering_ptr->supernode_sizes[supernode];
      const Int supernode_start = ordering_ptr->supernode_offsets[supernode];
      BlasMatrixView<Field> right_hand_sides_supernode =
          right_hand_sides->Submatrix(supernode_start, 0, supernode_size,
                                      num_rhs);

      // Handle the diagonal-block portion of the supernode.
      for (Int j = 0; j < num_rhs; ++j) {
        for (Int i = 0; i < supernode_size; ++i) {
          right_hand_sides_supernode(i, j) /= diagonal_right_hand_sides(i, i);
        }
      }
    }
  }
}

template <class Field>
void Factorization<Field>::OpenMPLowerTransposeTriangularSolveRecursion(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    Buffer<Buffer<Field>>* private_packed_input_bufs) const {
  // Perform this supernode's trapezoidal solve.
  const int thread = omp_get_thread_num();
  Buffer<Field>* packed_input_buf = &(*private_packed_input_bufs)[thread];
  LowerTransposeSupernodalTrapezoidalSolve(supernode, right_hand_sides,
                                           packed_input_buf);

  // Recurse on this supernode's children.
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;
  #pragma omp taskgroup
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    #pragma omp task default(none) firstprivate(child) \
        shared(right_hand_sides, private_packed_input_bufs)
    OpenMPLowerTransposeTriangularSolveRecursion(child, right_hand_sides,
                                                 private_packed_input_bufs);
  }
}

template <class Field>
void Factorization<Field>::OpenMPLowerTransposeTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  // Allocate each thread's workspace.
  const int max_threads = omp_get_max_threads();
  const Int workspace_size = max_degree_ * right_hand_sides->width;
  Buffer<Buffer<Field>> private_packed_input_bufs(max_threads);
  #pragma omp taskgroup
  for (int t = 0; t < max_threads; ++t) {
    #pragma omp task default(none) firstprivate(t, workspace_size) \
        shared(private_packed_input_bufs)
    private_packed_input_bufs[t].Resize(workspace_size);
  }

  // Recurse from each root of the elimination forest.
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  #pragma omp taskgroup
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = ordering_.assembly_forest.roots[root_index];
    #pragma omp task default(none) firstprivate(root) \
        shared(right_hand_sides, private_packed_input_bufs)
    OpenMPLowerTransposeTriangularSolveRecursion(root, right_hand_sides,
                                                 &private_packed_input_bufs);
  }
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifdef CATAMARI_OPENMP
#endif  // ifndef CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_OPENMP_IMPL_H_
