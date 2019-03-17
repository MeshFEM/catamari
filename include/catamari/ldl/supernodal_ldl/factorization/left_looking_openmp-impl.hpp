/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_LEFT_LOOKING_OPENMP_IMPL_H_
#define CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_LEFT_LOOKING_OPENMP_IMPL_H_
#ifdef CATAMARI_OPENMP

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"

#include "catamari/ldl/supernodal_ldl/factorization.hpp"

namespace catamari {
namespace supernodal_ldl {

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
      control_.factorization_type;
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

    num_supernode_pivots = OpenMPFactorDiagonalBlock(
        control_.factor_tile_size, control_.block_size,
        control_.factorization_type, &diagonal_block, buffer);
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
  OpenMPSolveAgainstDiagonalBlock(control_.outer_product_tile_size,
                                  control_.factorization_type,
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
    const CoordinateMatrix<Field>& matrix) {
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

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifdef CATAMARI_OPENMP
#endif  // ifndef
        // CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_LEFT_LOOKING_OPENMP_IMPL_H_
