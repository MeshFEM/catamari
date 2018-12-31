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

// Create the packed downlinks from the uplinks of an elimination forest.
inline void EliminationChildrenFromParents(
    const std::vector<Int>& supernode_parents,
    std::vector<Int>* supernode_children,
    std::vector<Int>* supernode_child_offsets, std::vector<Int>* roots) {
  const Int num_supernodes = supernode_parents.size();

  // Compute the number of children (initially stored in
  // 'supernode_child_offsets') of each supernode. Along the way, count the
  // number of trees in the forest.
  Int num_roots = 0;
  supernode_child_offsets->clear();
  supernode_child_offsets->resize(num_supernodes + 1, 0);
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int parent = supernode_parents[supernode];
    if (parent >= 0) {
      ++(*supernode_child_offsets)[parent];
    } else {
      ++num_roots;
    }
  }

  // Compute the child offsets using an in-place scan.
  Int num_total_children = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int num_children = (*supernode_child_offsets)[supernode];
    (*supernode_child_offsets)[supernode] = num_total_children;
    num_total_children += num_children;
  }
  (*supernode_child_offsets)[num_supernodes] = num_total_children;

  // Pack the children into the 'supernode_children' buffer.
  supernode_children->resize(num_total_children);
  roots->reserve(num_roots);
  std::vector<Int> offsets_copy = *supernode_child_offsets;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int parent = supernode_parents[supernode];
    if (parent >= 0) {
      (*supernode_children)[offsets_copy[parent]++] = supernode;
    } else {
      roots->push_back(supernode);
    }
  }
}

template <class Field>
void Factorization<Field>::FormSupernodes(
    const CoordinateMatrix<Field>& matrix,
    const SupernodalRelaxationControl& control, std::vector<Int>* parents,
    std::vector<Int>* supernode_degrees, std::vector<Int>* supernode_parents) {
  // Compute the non-supernodal elimination tree using the original ordering.
  std::vector<Int> orig_parents, orig_degrees;
  scalar_ldl::EliminationForestAndDegrees(
      matrix, permutation, inverse_permutation, &orig_parents, &orig_degrees);

  // Greedily compute a supernodal partition using the original ordering.
  std::vector<Int> orig_supernode_sizes;
  scalar_ldl::LowerStructure scalar_structure;
  FormFundamentalSupernodes(matrix, permutation, inverse_permutation,
                            orig_parents, orig_degrees, &orig_supernode_sizes,
                            &scalar_structure);

#ifdef CATAMARI_DEBUG
  {
    Int supernode_size_sum = 0;
    for (const Int& supernode_size : orig_supernode_sizes) {
      supernode_size_sum += supernode_size;
    }
    CATAMARI_ASSERT(supernode_size_sum == matrix.NumRows(),
                    "Supernodes did not sum to the matrix size.");
  }
#endif

  std::vector<Int> orig_supernode_starts;
  OffsetScan(orig_supernode_sizes, &orig_supernode_starts);

  std::vector<Int> orig_member_to_index;
  MemberToIndex(matrix.NumRows(), orig_supernode_starts, &orig_member_to_index);

  std::vector<Int> orig_supernode_degrees;
  SupernodalDegrees(matrix, permutation, inverse_permutation,
                    orig_supernode_sizes, orig_supernode_starts,
                    orig_member_to_index, orig_parents,
                    &orig_supernode_degrees);

  const Int num_orig_supernodes = orig_supernode_sizes.size();
  std::vector<Int> orig_supernode_parents;
  ConvertFromScalarToSupernodalEliminationForest(
      num_orig_supernodes, orig_parents, orig_member_to_index,
      &orig_supernode_parents);

  if (control.relax_supernodes) {
    RelaxSupernodes(orig_parents, orig_supernode_sizes, orig_supernode_starts,
                    orig_supernode_parents, orig_supernode_degrees,
                    orig_member_to_index, scalar_structure, control,
                    &permutation, &inverse_permutation, parents,
                    supernode_parents, supernode_degrees, &supernode_sizes,
                    &supernode_starts, &supernode_member_to_index);
  } else {
    *parents = orig_parents;
    *supernode_degrees = orig_supernode_degrees;
    *supernode_parents = orig_supernode_parents;
    supernode_sizes = orig_supernode_sizes;
    supernode_starts = orig_supernode_starts;
    supernode_member_to_index = orig_member_to_index;
  }
}

template <class Field>
void Factorization<Field>::InitializeLeftLookingFactors(
    const CoordinateMatrix<Field>& matrix, const std::vector<Int>& parents,
    const std::vector<Int>& supernode_degrees) {
  lower_factor.reset(
      new LowerFactor<Field>(supernode_sizes, supernode_degrees));
  diagonal_factor.reset(new DiagonalFactor<Field>(supernode_sizes));

  CATAMARI_ASSERT(supernode_degrees.size() == supernode_sizes.size(),
                  "Invalid supernode degrees size.");

  // Store the largest supernode size of the factorization.
  max_supernode_size =
      *std::max_element(supernode_sizes.begin(), supernode_sizes.end());

  // Store the largest degree of the factorization for use in the solve phase.
  max_degree =
      *std::max_element(supernode_degrees.begin(), supernode_degrees.end());

  FillStructureIndices(matrix, permutation, inverse_permutation, parents,
                       supernode_sizes, supernode_member_to_index,
                       lower_factor.get(), &max_descendant_entries);

  FillNonzeros(matrix, permutation, inverse_permutation, supernode_starts,
               supernode_sizes, supernode_member_to_index, lower_factor.get(),
               diagonal_factor.get());
}

template <class Field>
void Factorization<Field>::LeftLookingSupernodeUpdate(
    Int main_supernode, const CoordinateMatrix<Field>& matrix,
    const std::vector<Int>& supernode_parents,
    LeftLookingSharedState* shared_state,
    LeftLookingPrivateState* private_state) {
  BlasMatrix<Field>& main_diagonal_block =
      diagonal_factor->blocks[main_supernode];
  BlasMatrix<Field>& main_lower_block = lower_factor->blocks[main_supernode];
  const Int main_supernode_size = main_lower_block.width;

  private_state->pattern_flags[main_supernode] = main_supernode;

  shared_state->rel_rows[main_supernode] = 0;
  shared_state->intersect_ptrs[main_supernode] =
      lower_factor->IntersectionSizes(main_supernode);

  // Compute the supernodal row pattern.
  const Int num_packed = ComputeRowPattern(
      matrix, permutation, inverse_permutation, supernode_sizes,
      supernode_starts, supernode_member_to_index, supernode_parents,
      main_supernode, private_state->pattern_flags.data(),
      private_state->row_structure.data());

  // for J = find(L(K, :))
  //   L(K:n, K) -= L(K:n, J) * (D(J, J) * L(K, J)')
  for (Int index = 0; index < num_packed; ++index) {
    const Int descendant_supernode = private_state->row_structure[index];
    CATAMARI_ASSERT(descendant_supernode < main_supernode,
                    "Looking into upper triangle.");
    const ConstBlasMatrix<Field>& descendant_lower_block =
        lower_factor->blocks[descendant_supernode];
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
    scaled_transpose.data = private_state->scaled_transpose_buffer.data();

    FormScaledTranspose(factorization_type,
                        diagonal_factor->blocks[descendant_supernode].ToConst(),
                        descendant_main_matrix, &scaled_transpose);

    BlasMatrix<Field> workspace_matrix;
    workspace_matrix.height = descendant_main_intersect_size;
    workspace_matrix.width = descendant_main_intersect_size;
    workspace_matrix.leading_dim = descendant_main_intersect_size;
    workspace_matrix.data = private_state->workspace_buffer.data();

    UpdateDiagonalBlock(
        factorization_type, supernode_starts, *lower_factor, main_supernode,
        descendant_supernode, descendant_main_rel_row, descendant_main_matrix,
        scaled_transpose.ToConst(), &main_diagonal_block, &workspace_matrix);

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
        lower_factor->IntersectionSizes(main_supernode);
    Int main_active_rel_row = 0;
    while (descendant_active_rel_row != descendant_degree) {
      const Int descendant_active_intersect_size =
          *descendant_active_intersect_size_beg;

      SeekForMainActiveRelativeRow(
          main_supernode, descendant_supernode, descendant_active_rel_row,
          supernode_member_to_index, *lower_factor, &main_active_rel_row,
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
          descendant_main_rel_row, descendant_active_rel_row, supernode_starts,
          supernode_member_to_index, scaled_transpose.ToConst(),
          descendant_active_matrix, *lower_factor, &main_active_block,
          &workspace_matrix);

      ++descendant_active_intersect_size_beg;
      descendant_active_rel_row += descendant_active_intersect_size;
    }
  }
}

template <class Field>
bool Factorization<Field>::LeftLookingSupernodeFinalize(Int main_supernode,
                                                        LDLResult* result) {
  BlasMatrix<Field>& main_diagonal_block =
      diagonal_factor->blocks[main_supernode];
  BlasMatrix<Field>& main_lower_block = lower_factor->blocks[main_supernode];
  const Int main_degree = main_lower_block.height;
  const Int main_supernode_size = main_lower_block.width;

  const Int num_supernode_pivots =
      FactorDiagonalBlock(factorization_type, &main_diagonal_block);
  result->num_successful_pivots += num_supernode_pivots;
  if (num_supernode_pivots < main_supernode_size) {
    return false;
  }

  SolveAgainstDiagonalBlock(factorization_type, main_diagonal_block.ToConst(),
                            &main_lower_block);

  // Finish updating the result structure.
  result->largest_supernode =
      std::max(result->largest_supernode, main_supernode_size);
  result->num_factorization_entries +=
      (main_supernode_size * (main_supernode_size + 1)) / 2 +
      main_supernode_size * main_degree;

  // Add the approximate number of flops for the diagonal block factorization.
  result->num_factorization_flops +=
      std::pow(1. * main_supernode_size, 3.) / 3. +
      std::pow(1. * main_supernode_size, 2.) / 2.;

  // Add the approximate number of flops for the triangular solves of the
  // diagonal block against its structure.
  result->num_factorization_flops +=
      std::pow(1. * main_degree, 2.) * main_supernode_size;

  return true;
}

template <class Field>
LDLResult Factorization<Field>::LeftLooking(
    const CoordinateMatrix<Field>& matrix, const Control& control) {
#ifdef _OPENMP
  return MultithreadedLeftLooking(matrix, control);
#endif

  std::vector<Int> supernode_parents;
  {
    std::vector<Int> parents;
    std::vector<Int> supernode_degrees;
    FormSupernodes(matrix, control.relaxation_control, &parents,
                   &supernode_degrees, &supernode_parents);
    InitializeLeftLookingFactors(matrix, parents, supernode_degrees);
  }
  const Int num_supernodes = supernode_sizes.size();

  LeftLookingSharedState shared_state;
  shared_state.rel_rows.resize(num_supernodes);
  shared_state.intersect_ptrs.resize(num_supernodes);

  LeftLookingPrivateState private_state;
  private_state.row_structure.resize(num_supernodes);
  private_state.pattern_flags.resize(num_supernodes);
  private_state.scaled_transpose_buffer.resize(
      max_supernode_size * max_supernode_size, Field{0});
  private_state.workspace_buffer.resize(
      max_supernode_size * (max_supernode_size - 1), Field{0});

  LDLResult result;

  // Note that any postordering of the supernodal elimination forest suffices.
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    LeftLookingSupernodeUpdate(supernode, matrix, supernode_parents,
                               &shared_state, &private_state);
    const bool succeeded = LeftLookingSupernodeFinalize(supernode, &result);
    if (!succeeded) {
      return result;
    }
  }

  return result;
}

#ifdef _OPENMP
template <class Field>
Int MultithreadedFactorDiagonalBlock(
    Int num_threads, SymmetricFactorizationType factorization_type,
    BlasMatrix<Field>* diagonal_block) {
  Int num_pivots;

  // TODO(Jack Poulson): Decide if this parallel construct is of use.
  #pragma omp parallel num_threads(num_threads)
  {
    const int old_num_threads = SetNumLocalBlasThreads(num_threads);
    #pragma omp single
    {
      if (factorization_type == kCholeskyFactorization) {
        num_pivots = LowerCholeskyFactorization(diagonal_block);
      } else if (factorization_type == kLDLAdjointFactorization) {
        num_pivots = LowerLDLAdjointFactorization(diagonal_block);
      } else {
        num_pivots = LowerLDLTransposeFactorization(diagonal_block);
      }
    }  // pragma omp sequential
    SetNumLocalBlasThreads(old_num_threads);
  }  // pragma omp parallel

  return num_pivots;
}

template <class Field>
void Factorization<Field>::MultithreadedLeftLookingSupernodeUpdate(
    Int thread_offset, Int num_threads, Int main_supernode,
    const CoordinateMatrix<Field>& matrix,
    const std::vector<Int>& supernode_parents,
    LeftLookingSharedState* shared_state,
    std::vector<LeftLookingPrivateState>* private_states) {
  BlasMatrix<Field>& main_diagonal_block =
      diagonal_factor->blocks[main_supernode];
  BlasMatrix<Field>& main_lower_block = lower_factor->blocks[main_supernode];
  const Int main_supernode_size = main_lower_block.width;

  shared_state->rel_rows[main_supernode] = 0;
  shared_state->intersect_ptrs[main_supernode] =
      lower_factor->IntersectionSizes(main_supernode);

  LeftLookingPrivateState& main_private_state =
      (*private_states)[thread_offset];
  std::vector<Int>& main_pattern_flags = main_private_state.pattern_flags;
  std::vector<Int>& main_row_structure = main_private_state.row_structure;

  // Compute the supernodal row pattern.
  main_pattern_flags[main_supernode] = main_supernode;
  const Int num_packed = ComputeRowPattern(
      matrix, permutation, inverse_permutation, supernode_sizes,
      supernode_starts, supernode_member_to_index, supernode_parents,
      main_supernode, main_pattern_flags.data(), main_row_structure.data());

  const SymmetricFactorizationType factorization_type_copy = factorization_type;
  const std::vector<Int>& supernode_starts_ref = supernode_starts;
  const std::vector<Int>& supernode_member_to_index_ref =
      supernode_member_to_index;

  LowerFactor<Field>* const lower_factor_ptr = lower_factor.get();
  Field* const main_diagonal_block_data CATAMARI_UNUSED =
      main_diagonal_block.data;
  Field* const main_lower_block_data = main_lower_block.data;

  // Pack the scaled transposes.
  std::vector<BlasMatrix<Field>> scaled_transposes(num_packed);
  std::vector<Field>& scaled_transpose_buffer =
      (*private_states)[thread_offset].scaled_transpose_buffer;

  // OpenMP array index dependencies cannot operate directly on an
  // std::vector, so we extract the underlying array.
  BlasMatrix<Field>* scaled_transposes_data = scaled_transposes.data();

  #pragma omp parallel num_threads(num_threads) default(none)                \
      shared(main_supernode, main_diagonal_block, shared_state,              \
             private_states, main_row_structure,  supernode_starts_ref,      \
             supernode_member_to_index_ref, main_lower_block, thread_offset, \
             scaled_transposes, scaled_transposes_data,                      \
             scaled_transpose_buffer)
  {
    const int old_num_threads = SetNumLocalBlasThreads(1);
    #pragma omp single
    {
      {
        Int offset = 0;
        for (Int index = 0; index < num_packed; ++index) {
          const Int descendant_supernode = main_row_structure[index];
          const ConstBlasMatrix<Field>& descendant_lower_block =
              lower_factor_ptr->blocks[descendant_supernode];
          const Int descendant_supernode_size = descendant_lower_block.width;
          const Int descendant_main_rel_row =
              shared_state->rel_rows[descendant_supernode];
          const Int descendant_main_intersect_size =
              *shared_state->intersect_ptrs[descendant_supernode];

          const ConstBlasMatrix<Field> descendant_diag_block =
              diagonal_factor->blocks[descendant_supernode].ToConst();

          const ConstBlasMatrix<Field> descendant_main_matrix =
              descendant_lower_block.Submatrix(descendant_main_rel_row, 0,
                                               descendant_main_intersect_size,
                                               descendant_supernode_size);

          scaled_transposes[index].height = descendant_supernode_size;
          scaled_transposes[index].width = descendant_main_intersect_size;
          scaled_transposes[index].leading_dim = descendant_supernode_size;
          scaled_transposes[index].data = &scaled_transpose_buffer[offset];

          #pragma omp task default(none)                 \
              firstprivate(index, descendant_diag_block, \
                  descendant_main_matrix)                \
              shared(scaled_transposes)                  \
              depend(out: scaled_transposes_data[index])
          FormScaledTranspose(factorization_type_copy, descendant_diag_block,
                              descendant_main_matrix,
                              &scaled_transposes[index]);

          offset += descendant_supernode_size * descendant_main_intersect_size;
        }
        CATAMARI_ASSERT(offset <= Int(scaled_transpose_buffer.size()),
                        "Offset extended beyond buffer length.");
      }

      // for J = find(L(K, :))
      //   L(K:n, K) -= L(K:n, J) * (D(J, J) * L(K, J)')
      for (Int index = 0; index < num_packed; ++index) {
        const Int descendant_supernode = main_row_structure[index];
        CATAMARI_ASSERT(descendant_supernode < main_supernode,
                        "Looking into upper triangle.");
        const ConstBlasMatrix<Field> descendant_lower_block =
            lower_factor_ptr->blocks[descendant_supernode];
        const Int descendant_degree = descendant_lower_block.height;
        const Int descendant_supernode_size = descendant_lower_block.width;

        const Int descendant_main_rel_row =
            shared_state->rel_rows[descendant_supernode];
        const Int descendant_main_intersect_size =
            *shared_state->intersect_ptrs[descendant_supernode];

        #pragma omp task default(none)                                         \
            firstprivate(index, descendant_supernode, descendant_main_rel_row, \
                descendant_main_intersect_size, descendant_lower_block)        \
            shared(private_states, main_supernode, scaled_transposes,          \
                main_diagonal_block, supernode_starts_ref, thread_offset)      \
            depend(in: scaled_transposes_data[index])                          \
            depend(out: main_diagonal_block_data)
        {
          const int global_thread = thread_offset + omp_get_thread_num();
          LeftLookingPrivateState& private_state =
              (*private_states)[global_thread];

          const ConstBlasMatrix<Field> scaled_transpose =
              scaled_transposes[index].ToConst();

          const ConstBlasMatrix<Field> descendant_main_matrix =
              descendant_lower_block.Submatrix(descendant_main_rel_row, 0,
                                               descendant_main_intersect_size,
                                               descendant_supernode_size);

          BlasMatrix<Field> workspace_matrix;
          workspace_matrix.height = descendant_main_intersect_size;
          workspace_matrix.width = descendant_main_intersect_size;
          workspace_matrix.leading_dim = descendant_main_intersect_size;
          workspace_matrix.data = private_state.workspace_buffer.data();

          UpdateDiagonalBlock(factorization_type_copy, supernode_starts_ref,
                              *lower_factor_ptr, main_supernode,
                              descendant_supernode, descendant_main_rel_row,
                              descendant_main_matrix, scaled_transpose,
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
            lower_factor_ptr->IntersectionSizes(main_supernode);
        Int main_active_rel_row = 0;
        while (descendant_active_rel_row != descendant_degree) {
          const Int descendant_active_intersect_size =
              *descendant_active_intersect_size_beg;

          SeekForMainActiveRelativeRow(
              main_supernode, descendant_supernode, descendant_active_rel_row,
              supernode_member_to_index_ref, *lower_factor_ptr,
              &main_active_rel_row, &main_active_intersect_sizes);
          const Int main_active_intersect_size = *main_active_intersect_sizes;

          #pragma omp task default(none)                                  \
              firstprivate(index, descendant_supernode,                   \
                  descendant_active_rel_row, descendant_main_rel_row,     \
                  main_active_rel_row, descendant_active_intersect_size,  \
                  descendant_main_intersect_size,                         \
                  main_active_intersect_size, descendant_supernode_size,  \
                  private_states, descendant_lower_block)                 \
              shared(main_supernode, scaled_transposes, main_lower_block, \
                  supernode_starts_ref, supernode_member_to_index_ref,    \
                  thread_offset)                                          \
              depend(in: scaled_transposes_data[index])                   \
              depend(out: main_lower_block_data[main_active_rel_row])
          {
            const int global_thread = thread_offset + omp_get_thread_num();
            LeftLookingPrivateState& private_state =
                (*private_states)[global_thread];

            const ConstBlasMatrix<Field> scaled_transpose =
                scaled_transposes[index].ToConst();

            const ConstBlasMatrix<Field> descendant_active_matrix =
                descendant_lower_block.Submatrix(
                    descendant_active_rel_row, 0,
                    descendant_active_intersect_size,
                    descendant_supernode_size);

            BlasMatrix<Field> main_active_block = main_lower_block.Submatrix(
                main_active_rel_row, 0, main_active_intersect_size,
                main_supernode_size);

            BlasMatrix<Field> workspace_matrix;
            workspace_matrix.height = descendant_active_intersect_size;
            workspace_matrix.width = descendant_main_intersect_size;
            workspace_matrix.leading_dim = descendant_active_intersect_size;
            workspace_matrix.data = private_state.workspace_buffer.data();

            UpdateSubdiagonalBlock(
                main_supernode, descendant_supernode, main_active_rel_row,
                descendant_main_rel_row, descendant_active_rel_row,
                supernode_starts_ref, supernode_member_to_index_ref,
                scaled_transpose, descendant_active_matrix, *lower_factor_ptr,
                &main_active_block, &workspace_matrix);
          }

          ++descendant_active_intersect_size_beg;
          descendant_active_rel_row += descendant_active_intersect_size;
        }
      }
    }  // pragma omp sequential
    SetNumLocalBlasThreads(old_num_threads);
  }  // pragma omp parallel
}

template <class Field>
bool Factorization<Field>::MultithreadedLeftLookingSupernodeFinalize(
    Int num_threads, Int main_supernode, LDLResult* result) {
  BlasMatrix<Field>& main_diagonal_block =
      diagonal_factor->blocks[main_supernode];
  BlasMatrix<Field>& main_lower_block = lower_factor->blocks[main_supernode];
  const Int main_degree = main_lower_block.height;
  const Int main_supernode_size = main_lower_block.width;

  const Int num_supernode_pivots = MultithreadedFactorDiagonalBlock(
      num_threads, factorization_type, &main_diagonal_block);
  result->num_successful_pivots += num_supernode_pivots;
  if (num_supernode_pivots < main_supernode_size) {
    return false;
  }

  // TODO(Jack Poulson): Decide if this parallel construct is of use.
  #pragma omp parallel num_threads(num_threads)
  {
    const int old_num_threads = SetNumLocalBlasThreads(num_threads);
    #pragma omp single
    {
      SolveAgainstDiagonalBlock(
          factorization_type, main_diagonal_block.ToConst(), &main_lower_block);
    }
    SetNumLocalBlasThreads(old_num_threads);
  }

  // Finish updating the result structure.
  result->largest_supernode =
      std::max(result->largest_supernode, main_supernode_size);
  result->num_factorization_entries +=
      (main_supernode_size * (main_supernode_size + 1)) / 2 +
      main_supernode_size * main_degree;

  // Add the approximate number of flops for the diagonal block factorization.
  result->num_factorization_flops +=
      std::pow(1. * main_supernode_size, 3.) / 3. +
      std::pow(1. * main_supernode_size, 2.) / 2.;

  // Add the approximate number of flops for the triangular solves of the
  // diagonal block against its structure.
  result->num_factorization_flops +=
      std::pow(1. * main_degree, 2.) * main_supernode_size;

  return true;
}

template <class Field>
bool Factorization<Field>::LeftLookingSubtree(
    Int supernode, const CoordinateMatrix<Field>& matrix,
    const std::vector<Int>& supernode_parents,
    const std::vector<Int>& supernode_children,
    const std::vector<Int>& supernode_child_offsets,
    LeftLookingSharedState* shared_state,
    LeftLookingPrivateState* private_state, LDLResult* result) {
  const Int child_beg = supernode_child_offsets[supernode];
  const Int child_end = supernode_child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;

  std::vector<int> successes(num_children);
  std::vector<LDLResult> result_contributions(num_children);

  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child = supernode_children[child_beg + child_index];
    CATAMARI_ASSERT(supernode_parents[child] == supernode,
                    "Incorrect child index");
    LDLResult& result_contribution = result_contributions[child_index];
    successes[child_index] =
        LeftLookingSubtree(child, matrix, supernode_parents, supernode_children,
                           supernode_child_offsets, shared_state, private_state,
                           &result_contribution);
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
    LeftLookingSupernodeUpdate(supernode, matrix, supernode_parents,
                               shared_state, private_state);
    succeeded = LeftLookingSupernodeFinalize(supernode, result);
  }

  return succeeded;
}

template <class Field>
bool Factorization<Field>::MultithreadedLeftLookingSubtree(
    Int thread_offset, Int num_threads, Int supernode,
    const CoordinateMatrix<Field>& matrix,
    const std::vector<Int>& supernode_parents,
    const std::vector<Int>& supernode_children,
    const std::vector<Int>& supernode_child_offsets,
    LeftLookingSharedState* shared_state,
    std::vector<LeftLookingPrivateState>* private_states, LDLResult* result) {
  const Int child_beg = supernode_child_offsets[supernode];
  const Int child_end = supernode_child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;

  std::vector<int> successes(num_children);
  std::vector<LDLResult> result_contributions(num_children);

  if (num_threads == 1 || num_threads < num_children) {
    #pragma omp parallel for num_threads(num_threads) default(none) \
        shared(thread_offset, successes, matrix, supernode_parents, \
           supernode_children, supernode_child_offsets,             \
           result_contributions, shared_state, private_states)
    for (Int child_index = 0; child_index < num_children; ++child_index) {
      const Int child = supernode_children[child_beg + child_index];
      CATAMARI_ASSERT(supernode_parents[child] == supernode,
                      "Incorrect child index");
      const int thread = omp_get_thread_num();
      const int global_thread = thread_offset + thread;

      const int old_num_threads = SetNumLocalBlasThreads(1);
      LDLResult& result_contribution = result_contributions[child_index];
      successes[child_index] = LeftLookingSubtree(
          child, matrix, supernode_parents, supernode_children,
          supernode_child_offsets, shared_state,
          &(*private_states)[global_thread], &result_contribution);
      SetNumLocalBlasThreads(old_num_threads);
    }
  } else {
    // TODO(Jack Poulson): Come up with a more intelligent subtree mapping.
    std::vector<Int> num_threads_per_tree(num_children);
    const Int num_threads_base = num_threads / num_children;
    for (Int j = 0; j < num_children; ++j) {
      if (j < num_children - 1) {
        num_threads_per_tree[j] = num_threads_base;
      } else {
        num_threads_per_tree[j] =
            num_threads - (num_children - 1) * num_threads_base;
      }
    }

    std::vector<Int> thread_offsets;
    OffsetScan(num_threads_per_tree, &thread_offsets);

    #pragma omp parallel for num_threads(num_children) default(none)     \
        shared(successes, matrix, supernode_parents, supernode_children, \
            supernode_child_offsets, result_contributions, shared_state, \
            private_states, num_threads_per_tree, thread_offset, thread_offsets)
    for (Int child_index = 0; child_index < num_children; ++child_index) {
      const Int child = supernode_children[child_beg + child_index];
      CATAMARI_ASSERT(supernode_parents[child] == supernode,
                      "Incorrect child index");

      const int thread = omp_get_thread_num();
      const int child_thread_offset = thread_offset + thread_offsets[thread];
      const int num_threads_child = num_threads_per_tree[thread];

      LDLResult& result_contribution = result_contributions[child_index];
      successes[child_index] = MultithreadedLeftLookingSubtree(
          child_thread_offset, num_threads_child, child, matrix,
          supernode_parents, supernode_children, supernode_child_offsets,
          shared_state, private_states, &result_contribution);
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
    MultithreadedLeftLookingSupernodeUpdate(
        thread_offset, num_threads, supernode, matrix, supernode_parents,
        shared_state, private_states);
    succeeded = MultithreadedLeftLookingSupernodeFinalize(num_threads,
                                                          supernode, result);
  }

  return succeeded;
}

template <class Field>
LDLResult Factorization<Field>::MultithreadedLeftLooking(
    const CoordinateMatrix<Field>& matrix, const Control& control) {
  std::vector<Int> supernode_parents;
  {
    std::vector<Int> parents;
    std::vector<Int> supernode_degrees;
    FormSupernodes(matrix, control.relaxation_control, &parents,
                   &supernode_degrees, &supernode_parents);
    InitializeLeftLookingFactors(matrix, parents, supernode_degrees);
  }
  const Int num_supernodes = supernode_sizes.size();
  const Int max_threads = omp_get_max_threads();

  LeftLookingSharedState shared_state;
  shared_state.rel_rows.resize(num_supernodes);
  shared_state.intersect_ptrs.resize(num_supernodes);

  std::vector<LeftLookingPrivateState> private_states(max_threads);
  for (int thread = 0; thread < max_threads; ++thread) {
    LeftLookingPrivateState& private_state = private_states[thread];
    private_state.pattern_flags.resize(num_supernodes, -1);
    private_state.row_structure.resize(num_supernodes);

    // TODO(Jack Poulson): Switch to a reasonably-tight upper bound for each
    // thread.
    private_state.scaled_transpose_buffer.resize(max_descendant_entries,
                                                 Field{0});

    // TODO(Jack Poulson): Switch to a reasonably-tight upper bound for each
    // thread.
    private_state.workspace_buffer.resize(
        max_supernode_size * (max_supernode_size - 1), Field{0});
  }

  // Form the downlinks for the supernodal elimination tree.
  std::vector<Int> supernode_children;
  std::vector<Int> supernode_child_offsets;
  std::vector<Int> roots;
  EliminationChildrenFromParents(supernode_parents, &supernode_children,
                                 &supernode_child_offsets, &roots);
  const Int num_roots = roots.size();

  omp_set_nested(1);
  omp_set_dynamic(0);

  LDLResult result;

  std::vector<int> successes(num_roots);
  std::vector<LDLResult> result_contributions(num_roots);

  // Decide how many of the current processes to split amongst each of the
  // children.
  // TODO(Jack Poulson): Come up with a more intelligent assignment mechanism.
  if (max_threads == 1 || max_threads < num_roots) {
    #pragma omp parallel for num_threads(max_threads) default(none)            \
        shared(roots, successes, matrix, supernode_parents,                    \
            supernode_children, supernode_child_offsets, result_contributions, \
            shared_state, private_states)
    for (Int root_index = 0; root_index < num_roots; ++root_index) {
      const Int root = roots[root_index];
      const int thread_num = omp_get_thread_num();

      const int old_num_threads = SetNumLocalBlasThreads(1);
      LDLResult& result_contribution = result_contributions[root_index];
      successes[root_index] = LeftLookingSubtree(
          root, matrix, supernode_parents, supernode_children,
          supernode_child_offsets, &shared_state, &private_states[thread_num],
          &result_contribution);
      SetNumLocalBlasThreads(old_num_threads);
    }
  } else {
    std::vector<Int> num_threads_per_tree(num_roots);
    const Int num_threads_base = max_threads / num_roots;
    for (Int j = 0; j < num_roots; ++j) {
      if (j < num_roots - 1) {
        num_threads_per_tree[j] = num_threads_base;
      } else {
        num_threads_per_tree[j] =
            max_threads - (num_roots - 1) * num_threads_base;
      }
    }

    std::vector<Int> thread_offsets;
    OffsetScan(num_threads_per_tree, &thread_offsets);

    #pragma omp parallel for num_threads(num_roots) default(none)              \
        shared(roots, successes, matrix, supernode_parents,                    \
            supernode_children, supernode_child_offsets, result_contributions, \
            shared_state, num_threads_per_tree, thread_offsets, private_states)
    for (Int root_index = 0; root_index < num_roots; ++root_index) {
      const Int root = roots[root_index];
      const int child_thread = omp_get_thread_num();
      const int child_thread_offset = thread_offsets[child_thread];
      const int num_threads_child = num_threads_per_tree[child_thread];

      LDLResult& result_contribution = result_contributions[root_index];
      successes[root_index] = MultithreadedLeftLookingSubtree(
          child_thread_offset, num_threads_child, root, matrix,
          supernode_parents, supernode_children, supernode_child_offsets,
          &shared_state, &private_states, &result_contribution);
    }
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
LDLResult Factorization<Field>::Factor(
    const CoordinateMatrix<Field>& matrix,
    const std::vector<Int>& manual_permutation,
    const std::vector<Int>& inverse_manual_permutation,
    const Control& control) {
  permutation = manual_permutation;
  inverse_permutation = inverse_manual_permutation;
  factorization_type = control.factorization_type;
  forward_solve_out_of_place_supernode_threshold =
      control.forward_solve_out_of_place_supernode_threshold;
  backward_solve_out_of_place_supernode_threshold =
      control.backward_solve_out_of_place_supernode_threshold;
  return LeftLooking(matrix, control);
}

template <class Field>
LDLResult Factorization<Field>::Factor(const CoordinateMatrix<Field>& matrix,
                                       const Control& control) {
  std::vector<Int> manual_permutation, inverse_manual_permutation;
  return Factor(matrix, manual_permutation, inverse_manual_permutation,
                control);
}

template <class Field>
void Factorization<Field>::Solve(BlasMatrix<Field>* matrix) const {
  const bool have_permutation = !permutation.empty();

  // Reorder the input into the permutation of the factorization.
  if (have_permutation) {
    Permute(permutation, matrix);
  }

  LowerTriangularSolve(matrix);
  DiagonalSolve(matrix);
  LowerTransposeTriangularSolve(matrix);

  // Reverse the factorization permutation.
  if (have_permutation) {
    Permute(inverse_permutation, matrix);
  }
}

template <class Field>
void Factorization<Field>::LowerTriangularSolve(
    BlasMatrix<Field>* matrix) const {
  const Int num_rhs = matrix->width;
  const Int num_supernodes = supernode_sizes.size();
  const bool is_cholesky = factorization_type == kCholeskyFactorization;

  std::vector<Field> workspace(max_degree * num_rhs, Field{0});

  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const ConstBlasMatrix<Field> triangular_matrix =
        diagonal_factor->blocks[supernode];

    const Int supernode_size = supernode_sizes[supernode];
    const Int supernode_start = supernode_starts[supernode];
    BlasMatrix<Field> matrix_supernode =
        matrix->Submatrix(supernode_start, 0, supernode_size, num_rhs);

    // Solve against the diagonal block of the supernode.
    if (is_cholesky) {
      LeftLowerTriangularSolves(triangular_matrix, &matrix_supernode);
    } else {
      LeftLowerUnitTriangularSolves(triangular_matrix, &matrix_supernode);
    }

    const ConstBlasMatrix<Field> subdiagonal = lower_factor->blocks[supernode];
    if (!subdiagonal.height) {
      continue;
    }

    // Handle the external updates for this supernode.
    const Int* indices = lower_factor->Structure(supernode);
    if (supernode_size >= forward_solve_out_of_place_supernode_threshold) {
      // Perform an out-of-place GEMM.
      BlasMatrix<Field> work_matrix;
      work_matrix.height = subdiagonal.height;
      work_matrix.width = num_rhs;
      work_matrix.leading_dim = subdiagonal.height;
      work_matrix.data = workspace.data();

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
  const Int num_supernodes = supernode_sizes.size();
  const bool is_cholesky = factorization_type == kCholeskyFactorization;
  if (is_cholesky) {
    // D is the identity.
    return;
  }

  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const ConstBlasMatrix<Field> diagonal_matrix =
        diagonal_factor->blocks[supernode];

    const Int supernode_size = supernode_sizes[supernode];
    const Int supernode_start = supernode_starts[supernode];
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
  const Int num_supernodes = supernode_sizes.size();
  const bool is_selfadjoint = factorization_type != kLDLTransposeFactorization;

  std::vector<Field> packed_input_buf(max_degree * num_rhs);

  for (Int supernode = num_supernodes - 1; supernode >= 0; --supernode) {
    const Int supernode_size = supernode_sizes[supernode];
    const Int supernode_start = supernode_starts[supernode];
    const Int* indices = lower_factor->Structure(supernode);

    BlasMatrix<Field> matrix_supernode =
        matrix->Submatrix(supernode_start, 0, supernode_size, num_rhs);

    const ConstBlasMatrix<Field> subdiagonal = lower_factor->blocks[supernode];
    if (subdiagonal.height) {
      // Handle the external updates for this supernode.
      if (supernode_size >= backward_solve_out_of_place_supernode_threshold) {
        // Fill the work matrix.
        BlasMatrix<Field> work_matrix;
        work_matrix.height = subdiagonal.height;
        work_matrix.width = num_rhs;
        work_matrix.leading_dim = subdiagonal.height;
        work_matrix.data = packed_input_buf.data();
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
        diagonal_factor->blocks[supernode];
    if (factorization_type == kCholeskyFactorization) {
      LeftLowerAdjointTriangularSolves(triangular_matrix, &matrix_supernode);
    } else if (factorization_type == kLDLAdjointFactorization) {
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
  if (factorization_type == kCholeskyFactorization) {
    // TODO(Jack Poulson): Print the identity.
    return;
  }

  os << label << ": \n";
  const Int num_supernodes = supernode_sizes.size();
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const ConstBlasMatrix<Field>& diag_matrix =
        diagonal_factor->blocks[supernode];
    for (Int j = 0; j < diag_matrix.height; ++j) {
      os << diag_matrix(j, j) << " ";
    }
  }
  os << std::endl;
}

template <class Field>
void Factorization<Field>::PrintLowerFactor(const std::string& label,
                                            std::ostream& os) const {
  const bool is_cholesky = factorization_type == kCholeskyFactorization;

  auto print_entry = [&](const Int& row, const Int& column,
                         const Field& value) {
    os << row << " " << column << " " << value << "\n";
  };

  os << label << ": \n";
  const Int num_supernodes = supernode_sizes.size();
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int supernode_start = supernode_starts[supernode];
    const Int* indices = lower_factor->Structure(supernode);

    const ConstBlasMatrix<Field>& diag_matrix =
        diagonal_factor->blocks[supernode];
    const ConstBlasMatrix<Field>& lower_matrix =
        lower_factor->blocks[supernode];

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

template <class Field>
void FillStructureIndices(const CoordinateMatrix<Field>& matrix,
                          const std::vector<Int>& permutation,
                          const std::vector<Int>& inverse_permutation,
                          const std::vector<Int>& parents,
                          const std::vector<Int>& supernode_sizes,
                          const std::vector<Int>& supernode_member_to_index,
                          LowerFactor<Field>* lower_factor,
                          Int* max_descendant_entries) {
  const Int num_rows = matrix.NumRows();
  const Int num_supernodes = supernode_sizes.size();
  const bool have_permutation = !permutation.empty();

  // A data structure for marking whether or not a node is in the pattern of
  // the active row of the lower-triangular factor.
  std::vector<Int> pattern_flags(num_rows);

  // A data structure for marking whether or not a supernode is in the pattern
  // of the active row of the lower-triangular factor.
  std::vector<Int> supernode_pattern_flags(num_supernodes);

  // A set of pointers for keeping track of where to insert supernode pattern
  // indices.
  std::vector<Int*> supernode_ptrs(num_supernodes);
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    supernode_ptrs[supernode] = lower_factor->Structure(supernode);
  }

  // Fill in the structure indices.
  const std::vector<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int main_supernode = supernode_member_to_index[row];
    pattern_flags[row] = row;
    supernode_pattern_flags[main_supernode] = row;

    const Int orig_row = have_permutation ? inverse_permutation[row] : row;
    const Int row_beg = matrix.RowEntryOffset(orig_row);
    const Int row_end = matrix.RowEntryOffset(orig_row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      Int descendant =
          have_permutation ? permutation[entry.column] : entry.column;
      Int descendant_supernode = supernode_member_to_index[descendant];

      if (descendant >= row) {
        if (have_permutation) {
          continue;
        } else {
          // We are traversing the strictly lower triangle and know that the
          // indices are sorted.
          break;
        }
      }

      // Look for new entries in the pattern by walking up to the root of this
      // subtree of the elimination forest.
      while (pattern_flags[descendant] != row) {
        // Mark 'descendant' as in the pattern of this row.
        pattern_flags[descendant] = row;

        descendant_supernode = supernode_member_to_index[descendant];
        CATAMARI_ASSERT(descendant_supernode <= main_supernode,
                        "Descendant supernode was larger than main supernode.");
        if (descendant_supernode == main_supernode) {
          break;
        }

        if (supernode_pattern_flags[descendant_supernode] != row) {
          supernode_pattern_flags[descendant_supernode] = row;
          CATAMARI_ASSERT(
              descendant_supernode < main_supernode,
              "Descendant supernode was as large as main supernode.");
          CATAMARI_ASSERT(
              supernode_ptrs[descendant_supernode] >=
                      lower_factor->Structure(descendant_supernode) &&
                  supernode_ptrs[descendant_supernode] <
                      lower_factor->Structure(descendant_supernode + 1),
              "Left supernode's indices.");
          *supernode_ptrs[descendant_supernode] = row;
          ++supernode_ptrs[descendant_supernode];
        }

        // Move up to the parent in this subtree of the elimination forest.
        // Moving to the parent will increase the index (but remain bounded
        // from above by 'row').
        descendant = parents[descendant];
      }
    }
  }

#ifdef CATAMARI_DEBUG
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int* index_beg = lower_factor->Structure(supernode);
    const Int* index_end = lower_factor->Structure(supernode + 1);
    CATAMARI_ASSERT(supernode_ptrs[supernode] == index_end,
                    "Supernode pointers did not match index offsets.");
    bool sorted = true;
    Int last_row = -1;
    for (const Int* row_ptr = index_beg; row_ptr != index_end; ++row_ptr) {
      const Int row = *row_ptr;
      if (row <= last_row) {
        sorted = false;
        break;
      }
      last_row = row;
    }

    if (!sorted) {
      std::cerr << "Supernode " << supernode << " did not have sorted indices."
                << std::endl;
      for (const Int* row_ptr = index_beg; row_ptr != index_end; ++row_ptr) {
        std::cout << *row_ptr << " ";
      }
      std::cout << std::endl;
    }
  }
#endif

  lower_factor->FillIntersectionSizes(
      supernode_sizes, supernode_member_to_index, max_descendant_entries);
}

template <class Field>
void FillNonzeros(const CoordinateMatrix<Field>& matrix,
                  const std::vector<Int>& permutation,
                  const std::vector<Int>& inverse_permutation,
                  const std::vector<Int>& supernode_starts,
                  const std::vector<Int>& supernode_sizes,
                  const std::vector<Int>& supernode_member_to_index,
                  LowerFactor<Field>* lower_factor,
                  DiagonalFactor<Field>* diagonal_factor) {
  const Int num_rows = matrix.NumRows();
  const bool have_permutation = !permutation.empty();

  const std::vector<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int supernode = supernode_member_to_index[row];
    const Int supernode_start = supernode_starts[supernode];

    const Int orig_row = have_permutation ? inverse_permutation[row] : row;
    const Int row_beg = matrix.RowEntryOffset(orig_row);
    const Int row_end = matrix.RowEntryOffset(orig_row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      const Int column =
          have_permutation ? permutation[entry.column] : entry.column;
      const Int column_supernode = supernode_member_to_index[column];

      if (column_supernode == supernode) {
        // Insert the value into the diagonal block.
        const Int rel_row = row - supernode_start;
        const Int rel_column = column - supernode_start;
        diagonal_factor->blocks[supernode](rel_row, rel_column) = entry.value;
        continue;
      }

      if (column_supernode > supernode) {
        if (have_permutation) {
          continue;
        } else {
          break;
        }
      }

      // Insert the value into the subdiagonal block.
      const Int* column_index_beg = lower_factor->Structure(column_supernode);
      const Int* column_index_end =
          lower_factor->Structure(column_supernode + 1);
      const Int* iter =
          std::lower_bound(column_index_beg, column_index_end, row);
      CATAMARI_ASSERT(iter != column_index_end, "Exceeded column indices.");
      CATAMARI_ASSERT(*iter == row, "Entry (" + std::to_string(row) + ", " +
                                        std::to_string(column) +
                                        ") wasn't in the structure.");
      const Int rel_row = std::distance(column_index_beg, iter);
      const Int rel_column = column - supernode_starts[column_supernode];
      lower_factor->blocks[column_supernode](rel_row, rel_column) = entry.value;
    }
  }
}

template <class Field>
Int ComputeRowPattern(const CoordinateMatrix<Field>& matrix,
                      const std::vector<Int>& permutation,
                      const std::vector<Int>& inverse_permutation,
                      const std::vector<Int>& supernode_sizes,
                      const std::vector<Int>& supernode_starts,
                      const std::vector<Int>& member_to_index,
                      const std::vector<Int>& supernode_parents,
                      Int main_supernode, Int* pattern_flags,
                      Int* row_structure) {
  Int num_packed = 0;
  const bool have_permutation = !permutation.empty();
  const std::vector<MatrixEntry<Field>>& entries = matrix.Entries();

  // Take the union of the row patterns of each row in the supernode.
  const Int main_supernode_size = supernode_sizes[main_supernode];
  const Int main_supernode_start = supernode_starts[main_supernode];
  for (Int row = main_supernode_start;
       row < main_supernode_start + main_supernode_size; ++row) {
    const Int orig_row = have_permutation ? inverse_permutation[row] : row;
    const Int row_beg = matrix.RowEntryOffset(orig_row);
    const Int row_end = matrix.RowEntryOffset(orig_row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      const Int descendant =
          have_permutation ? permutation[entry.column] : entry.column;

      Int descendant_supernode = member_to_index[descendant];
      if (descendant_supernode >= main_supernode) {
        if (have_permutation) {
          continue;
        } else {
          break;
        }
      }

      // Walk up to the root of the current subtree of the elimination
      // forest, stopping if we encounter a member already marked as in the
      // row pattern.
      while (pattern_flags[descendant_supernode] != main_supernode) {
        // Place 'descendant_supernode' into the pattern of this supernode.
        row_structure[num_packed++] = descendant_supernode;
        pattern_flags[descendant_supernode] = main_supernode;

        descendant_supernode = supernode_parents[descendant_supernode];
      }
    }
  }

  return num_packed;
}

template <class Field>
void FormScaledTranspose(SymmetricFactorizationType factorization_type,
                         const ConstBlasMatrix<Field>& diagonal_block,
                         const ConstBlasMatrix<Field>& matrix,
                         BlasMatrix<Field>* scaled_transpose) {
  if (factorization_type == kCholeskyFactorization) {
    for (Int j = 0; j < matrix.width; ++j) {
      for (Int i = 0; i < matrix.height; ++i) {
        scaled_transpose->Entry(j, i) = Conjugate(matrix(i, j));
      }
    }
  } else if (factorization_type == kLDLAdjointFactorization) {
    for (Int j = 0; j < matrix.width; ++j) {
      const Field& delta = diagonal_block(j, j);
      for (Int i = 0; i < matrix.height; ++i) {
        scaled_transpose->Entry(j, i) = delta * Conjugate(matrix(i, j));
      }
    }
  } else {
    for (Int j = 0; j < matrix.width; ++j) {
      const Field& delta = diagonal_block(j, j);
      for (Int i = 0; i < matrix.height; ++i) {
        scaled_transpose->Entry(j, i) = delta * matrix(i, j);
      }
    }
  }
}

template <class Field>
void UpdateDiagonalBlock(SymmetricFactorizationType factorization_type,
                         const std::vector<Int>& supernode_starts,
                         const LowerFactor<Field>& lower_factor,
                         Int main_supernode, Int descendant_supernode,
                         Int descendant_main_rel_row,
                         const ConstBlasMatrix<Field>& descendant_main_matrix,
                         const ConstBlasMatrix<Field>& scaled_transpose,
                         BlasMatrix<Field>* main_diag_block,
                         BlasMatrix<Field>* workspace_matrix) {
  typedef ComplexBase<Field> Real;
  const Int main_supernode_size = main_diag_block->height;
  const Int descendant_main_intersect_size = scaled_transpose.width;

  const bool inplace_update =
      descendant_main_intersect_size == main_supernode_size;
  BlasMatrix<Field>* accumulation_block =
      inplace_update ? main_diag_block : workspace_matrix;

  if (factorization_type == kCholeskyFactorization) {
    LowerNormalHermitianOuterProduct(Real{-1}, descendant_main_matrix, Real{1},
                                     accumulation_block);
  } else {
    MatrixMultiplyLowerNormalNormal(Field{-1}, descendant_main_matrix,
                                    scaled_transpose, Field{1},
                                    accumulation_block);
  }

  if (!inplace_update) {
    // Apply the out-of-place update and zero the buffer.
    const Int main_supernode_start = supernode_starts[main_supernode];
    const Int* descendant_main_indices =
        lower_factor.Structure(descendant_supernode) + descendant_main_rel_row;

    for (Int j = 0; j < descendant_main_intersect_size; ++j) {
      const Int column = descendant_main_indices[j];
      const Int j_rel = column - main_supernode_start;
      for (Int i = j; i < descendant_main_intersect_size; ++i) {
        const Int row = descendant_main_indices[i];
        const Int i_rel = row - main_supernode_start;
        main_diag_block->Entry(i_rel, j_rel) += workspace_matrix->Entry(i, j);
        workspace_matrix->Entry(i, j) = 0;
      }
    }
  }
}

template <class Field>
void SeekForMainActiveRelativeRow(
    Int main_supernode, Int descendant_supernode, Int descendant_active_rel_row,
    const std::vector<Int>& supernode_member_to_index,
    const LowerFactor<Field>& lower_factor, Int* main_active_rel_row,
    const Int** main_active_intersect_sizes) {
  const Int* main_indices = lower_factor.Structure(main_supernode);
  const Int* descendant_indices = lower_factor.Structure(descendant_supernode);
  const Int descendant_active_supernode_start =
      descendant_indices[descendant_active_rel_row];
  const Int active_supernode =
      supernode_member_to_index[descendant_active_supernode_start];
  CATAMARI_ASSERT(active_supernode > main_supernode,
                  "Active supernode was <= the main supernode in update.");

  Int main_active_intersect_size = **main_active_intersect_sizes;
  Int main_active_first_row = main_indices[*main_active_rel_row];
  while (supernode_member_to_index[main_active_first_row] < active_supernode) {
    *main_active_rel_row += main_active_intersect_size;
    ++*main_active_intersect_sizes;

    main_active_first_row = main_indices[*main_active_rel_row];
    main_active_intersect_size = **main_active_intersect_sizes;
  }
#ifdef CATAMARI_DEBUG
  const Int main_active_supernode =
      supernode_member_to_index[main_active_first_row];
  CATAMARI_ASSERT(main_active_supernode == active_supernode,
                  "Did not find active supernode.");
#endif
}

template <class Field>
void UpdateSubdiagonalBlock(
    Int main_supernode, Int descendant_supernode, Int main_active_rel_row,
    Int descendant_main_rel_row, Int descendant_active_rel_row,
    const std::vector<Int>& supernode_starts,
    const std::vector<Int>& supernode_member_to_index,
    const ConstBlasMatrix<Field>& scaled_transpose,
    const ConstBlasMatrix<Field>& descendant_active_matrix,
    const LowerFactor<Field>& lower_factor,
    BlasMatrix<Field>* main_active_block, BlasMatrix<Field>* workspace_matrix) {
  const Int main_supernode_size = lower_factor.blocks[main_supernode].width;
  const Int main_active_intersect_size = main_active_block->height;
  const Int descendant_main_intersect_size = scaled_transpose.width;
  const Int descendant_active_intersect_size = descendant_active_matrix.height;
  const bool inplace_update =
      main_active_intersect_size == descendant_active_intersect_size &&
      main_supernode_size == descendant_main_intersect_size;

  BlasMatrix<Field>* accumulation_matrix =
      inplace_update ? main_active_block : workspace_matrix;
  MatrixMultiplyNormalNormal(Field{-1}, descendant_active_matrix,
                             scaled_transpose, Field{1}, accumulation_matrix);

  if (!inplace_update) {
    const Int main_supernode_start = supernode_starts[main_supernode];

    const Int* main_indices = lower_factor.Structure(main_supernode);
    const Int* main_active_indices = main_indices + main_active_rel_row;

    const Int* descendant_indices =
        lower_factor.Structure(descendant_supernode);
    const Int* descendant_main_indices =
        descendant_indices + descendant_main_rel_row;
    const Int* descendant_active_indices =
        descendant_indices + descendant_active_rel_row;

    Int i_rel = 0;
    for (Int i = 0; i < descendant_active_intersect_size; ++i) {
      const Int row = descendant_active_indices[i];

      // Both the main and descendant supernodal intersections can be sparse,
      // and different. Thus, we must scan through the intersection indices to
      // find the appropriate relative index here.
      //
      // Scan downwards in the main active indices until we find the equivalent
      // row from the descendant active indices.
      while (main_active_indices[i_rel] != row) {
        ++i_rel;
      }

      for (Int j = 0; j < descendant_main_intersect_size; ++j) {
        const Int column = descendant_main_indices[j];
        const Int j_rel = column - main_supernode_start;

        main_active_block->Entry(i_rel, j_rel) += workspace_matrix->Entry(i, j);
        workspace_matrix->Entry(i, j) = 0;
      }
    }
  }
}

template <class Field>
Int FactorDiagonalBlock(SymmetricFactorizationType factorization_type,
                        BlasMatrix<Field>* diagonal_block) {
  Int num_pivots;
  if (factorization_type == kCholeskyFactorization) {
    num_pivots = LowerCholeskyFactorization(diagonal_block);
  } else if (factorization_type == kLDLAdjointFactorization) {
    num_pivots = LowerLDLAdjointFactorization(diagonal_block);
  } else {
    num_pivots = LowerLDLTransposeFactorization(diagonal_block);
  }
  return num_pivots;
}

template <class Field>
void SolveAgainstDiagonalBlock(SymmetricFactorizationType factorization_type,
                               const ConstBlasMatrix<Field>& triangular_matrix,
                               BlasMatrix<Field>* lower_matrix) {
  if (!lower_matrix->height) {
    return;
  }
  if (factorization_type == kCholeskyFactorization) {
    // Solve against the lower-triangular matrix L(K, K)' from the right.
    RightLowerAdjointTriangularSolves(triangular_matrix, lower_matrix);
  } else if (factorization_type == kLDLAdjointFactorization) {
    // Solve against D(K, K) L(K, K)' from the right.
    RightDiagonalTimesLowerAdjointUnitTriangularSolves(triangular_matrix,
                                                       lower_matrix);
  } else {
    // Solve against D(K, K) L(K, K)^T from the right.
    RightDiagonalTimesLowerTransposeUnitTriangularSolves(triangular_matrix,
                                                         lower_matrix);
  }
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_IMPL_H_
