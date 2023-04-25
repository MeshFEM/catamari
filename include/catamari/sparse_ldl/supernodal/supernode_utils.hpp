/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_SUPERNODE_UTILS_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_SUPERNODE_UTILS_H_

#include "catamari/buffer.hpp"
#include "catamari/sparse_ldl/scalar.hpp"
#include "catamari/symmetric_ordering.hpp"

#include "catamari_config.hh"
#include "factorization/SchurComplementStorage.hpp"

#if defined(CATAMARI_ENABLE_TIMERS) || CUSTOM_TIMERS
#include "quotient/timer.hpp"
#endif

namespace catamari {

struct SupernodalRelaxationControl {
  // If true, relaxed supernodes are created in a manner similar to the
  // suggestion from:
  //   Ashcraft and Grime, "The impact of relaxed supernode partitions on the
  //   multifrontal method", 1989.
  bool relax_supernodes = false;

  // A list of pairs of bounding combined supernode sizes and explicit zero
  // ratios for deciding child/parent supernode mergability.
  //
  // These constants match that of CHOLMOD (Davis et al.).
  std::vector<std::pair<Int, float>> cutoff_pairs{
      std::make_pair(4, 1.f),
      std::make_pair(16, 0.8f),
      std::make_pair(48, 0.1f),
      std::make_pair(std::numeric_limits<Int>::max(), 0.05f),
  };
};

namespace supernodal_ldl {

// A data structure for representing whether or not a child supernode can be
// merged with its parent and, if so, how many explicit zeros would be in the
// combined supernode's block column.
struct MergableStatus {
  // Whether or not the supernode can be merged.
  bool mergable;

  // How many explicit zeros would be stored in the merged supernode.
  Int num_merged_zeros;
};

// A data structure for maintaining singly linked lists eminating from each
// supernode of a left-looking factorization.
struct LinkedLists {
  // A workspace of size 'num_supernodes' which contains the links in singly
  // linked lists started by the non-negative indices of 'heads'.
  Buffer<Int> lists;

  // A map from each supernode to either the index of the start of a linked list
  // in 'lists' or an empty list (signified by a negative index).
  Buffer<Int> heads;

  // Initializes empty linked lists eminating from each supernode.
  void Initialize(Int num_supernodes) {
    lists.Resize(num_supernodes);
    heads.Resize(num_supernodes, -1);
  }

  // Inserts 'target_supernode' into the linked list associated with the
  // 'source_supernode'.
  void Insert(Int source_supernode, Int target_supernode) {
    lists[target_supernode] = heads[source_supernode];
    heads[source_supernode] = target_supernode;
  }
};

struct LeftLookingSharedState {
  // The relative index of the active supernode within each supernode's
  // structure.
  Buffer<Int> rel_rows;

  // Pointers to the active supernode intersection size within each
  // supernode's structure.
  Buffer<const Int*> intersect_ptrs;

  // Left-looking factorizations make use of linked lists of the descendants
  // of each supernode.
  LinkedLists descendants;

#ifdef CATAMARI_ENABLE_TIMERS
  // A separate timer for each supernode's inclusive processing time.
  Buffer<quotient::Timer> inclusive_timers;

  // A separate timer for each supernode's exclusive processing time.
  Buffer<quotient::Timer> exclusive_timers;
#endif  // ifdef CATAMARI_ENABLE_TIMERS
};

#include <atomic>

template <typename Field>
struct RightLookingSharedState {
  RightLookingSharedState() { unsetFailed(); }
  // The Schur complement matrices for each of the supernodes in the
  // multifrontal method. Each front should only be allocated while it is
  // actively in use.
  Buffer<BlasMatrixView<Field>> schur_complements;

  // The underlying buffers for the Schur complement portions of the fronts.
  // They are allocated and deallocated as the factorization progresses.
  // (Julian Panetta: We no longer use this during factorization;
  //  we use the `schur_complement_storage` stack below instead!)
  Buffer<Buffer<Field>> schur_complement_buffers;

#if CUSTOM_TIMERS
  Buffer<quotient::Timer> custom_timers;
#endif

  Buffer<SchurComplementStorage<Field>> schur_complement_storage;

  void unsetFailed() { m_fail.store(false, std::memory_order_relaxed); }
  void   setFailed() { m_fail.store(true, std::memory_order_relaxed); }
  bool   hasFailed() const { return m_fail.load(std::memory_order_relaxed); }

#ifdef CATAMARI_ENABLE_TIMERS
  // A separate timer for each supernode's inclusive processing time.
  Buffer<quotient::Timer> inclusive_timers;

  // A separate timer for each supernode's exclusive processing time.
  Buffer<quotient::Timer> exclusive_timers;
#endif  // ifdef CATAMARI_ENABLE_TIMERS

private:
  std::atomic<bool> m_fail; // Global flag to indicate factorization failure and accelerate early-exit in parallel case.
};

template <typename Field>
struct PrivateState {
  // A data structure for marking whether or not a (super)node is in the pattern
  // of the active row of the lower-triangular factor.
  Buffer<Int> pattern_flags;

  // A buffer for storing the relative indices mapping a descendant supernode's
  // structure into one of its ancestor's structure.
  Buffer<Int> relative_indices;

  // A buffer for storing (scaled) transposed descendant blocks.
  Buffer<Field> scaled_transpose_buffer;

  // A buffer for storing updates to the current supernode column.
  Buffer<Field> workspace_buffer;
};

// Fills 'member_to_index' with a length 'num_rows' array whose i'th index
// is the index of the supernode containing column 'i'.
void MemberToIndex(Int num_rows, const Buffer<Int>& supernode_starts,
                   Buffer<Int>* member_to_index);

// Builds an elimination forest over the supernodes from an elimination forest
// over the nodes.
void ConvertFromScalarToSupernodalEliminationForest(
    Int num_supernodes, const Buffer<Int>& parents,
    const Buffer<Int>& member_to_index, Buffer<Int>* supernode_parents);

// Checks that a valid set of supernodes has been provided by explicitly
// computing each row pattern and ensuring that each intersects entire
// supernodes.
template <class Field>
bool ValidFundamentalSupernodes(const CoordinateMatrix<Field>& matrix,
                                const SymmetricOrdering& ordering,
                                const Buffer<Int>& supernode_sizes);

// Compute an unrelaxed supernodal partition using the existing ordering.
// We require that supernodes have dense diagonal blocks and equal structures
// below the diagonal block.
void FormFundamentalSupernodes(const Buffer<Int>& scalar_parents,
                               const Buffer<Int>& scalar_degrees,
                               Buffer<Int>* supernode_sizes);

// Returns whether or not the child supernode can be merged into its parent by
// counting the number of explicit zeros that would be introduced by the
// merge.
//
// Consider the possibility of merging a child supernode '0' with parent
// supernode '1', which would result in an expanded supernode of the form:
//
//    -------------
//   | L00  |      |
//   |------|------|
//   | L10  | L11  |
//   ---------------
//   |      |      |
//   | L20  | L21  |
//   |      |      |
//    -------------
//
// Because we have a fundamental supernode partition, we know that L10 is
// nonzero in only its first row.
//
// The reason that downstream explicit zeros would not be introduced is the
// same as the reason explicit zeros are not introduced into L21; supernode
// 1 is the parent of supernode 0.
//
MergableStatus MergableSupernode(Int child_size, Int child_degree,
                                 Int parent_size, Int parent_degree,
                                 Int num_child_zeros, Int num_parent_zeros,
                                 const SupernodalRelaxationControl& control);

void MergeChildren(Int parent, const Buffer<Int>& orig_supernode_starts,
                   const Buffer<Int>& orig_supernode_sizes,
                   const Buffer<Int>& orig_supernode_degrees,
                   const Buffer<Int>& children,
                   const Buffer<Int>& child_offsets,
                   const SupernodalRelaxationControl& control,
                   Buffer<Int>* supernode_sizes, Buffer<Int>* num_zeros,
                   Buffer<Int>* last_merged_child, Buffer<Int>* merge_parents);

// Walk up the tree in the original postordering, merging supernodes as we
// progress. The 'relaxed_permutation' and 'relaxed_inverse_permutation'
// variables are also inputs.
void RelaxSupernodes(const SymmetricOrdering& orig_ordering,
                     const Buffer<Int>& orig_supernode_degrees,
                     const SupernodalRelaxationControl& control,
                     SymmetricOrdering* relaxed_ordering,
                     Buffer<Int>* relaxed_supernode_degrees,
                     Buffer<Int>* relaxed_supernode_member_to_index);

// Fills an estimate of the work required to eliminate the subtree in a
// right-looking factorization.
template <class Field>
void FillSubtreeWorkEstimates(Int root, const AssemblyForest& supernode_forest,
                              const LowerFactor<Field>& lower_factor,
                              Buffer<double>* work_estimates);

// Fills in the structure indices for the lower factor.
template <class Field>
void FillStructureIndices(const CoordinateMatrix<Field>& matrix,
                          const SymmetricOrdering& ordering,
                          const Buffer<Int>& supernode_member_to_index,
                          LowerFactor<Field>* lower_factor);
#ifdef CATAMARI_OPENMP
template <class Field>
void OpenMPFillStructureIndices(Int sort_grain_size,
                                const CoordinateMatrix<Field>& matrix,
                                const SymmetricOrdering& ordering,
                                const Buffer<Int>& supernode_member_to_index,
                                LowerFactor<Field>* lower_factor);
#endif  // ifdef CATAMARI_OPENMP

// Fill in the nonzeros from the original sparse matrix.
template <class Field>
void FillNonzeros(const CoordinateMatrix<Field>& matrix,
                  const SymmetricOrdering& ordering,
                  const Buffer<Int>& supernode_member_to_index,
                  LowerFactor<Field>* lower_factor,
                  DiagonalFactor<Field>* diagonal_factor);
#ifdef CATAMARI_OPENMP
template <class Field>
void OpenMPFillNonzeros(const CoordinateMatrix<Field>& matrix,
                        const SymmetricOrdering& ordering,
                        const Buffer<Int>& supernode_member_to_index,
                        LowerFactor<Field>* lower_factor,
                        DiagonalFactor<Field>* diagonal_factor);
#endif  // ifdef CATAMARI_OPENMP

// Explicitly fill the factorization blocks with zeros.
template <class Field>
void FillZeros(const SymmetricOrdering& ordering,
               LowerFactor<Field>* lower_factor,
               DiagonalFactor<Field>* diagonal_factor);

#ifdef CATAMARI_OPENMP
template <class Field>
void OpenMPFillZeros(const SymmetricOrdering& ordering,
                     LowerFactor<Field>* lower_factor,
                     DiagonalFactor<Field>* diagonal_factor);
#endif  // ifdef CATAMARI_OPENMP

// Store the scaled adjoint update matrix, Z(d, m) = D(d, d) L(m, d)', or
// the scaled transpose, Z(d, m) = D(d, d) L(m, d)^T.
template <class Field>
void FormScaledTranspose(SymmetricFactorizationType factorization_type,
                         const ConstBlasMatrixView<Field>& diagonal_block,
                         const ConstBlasMatrixView<Field>& matrix,
                         BlasMatrixView<Field>* scaled_transpose);

#ifdef CATAMARI_OPENMP
template <class Field>
void OpenMPFormScaledTranspose(Int tile_size,
                               SymmetricFactorizationType factorization_type,
                               const ConstBlasMatrixView<Field>& diagonal_block,
                               const ConstBlasMatrixView<Field>& matrix,
                               BlasMatrixView<Field>* scaled_transpose);
#endif  // ifdef CATAMARI_OPENMP

// Adds the schur complements of a given supernode's children onto its front.
template <class Field>
void MergeChildSchurComplements(Int supernode,
                                const SymmetricOrdering& ordering,
                                LowerFactor<Field>* lower_factor,
                                DiagonalFactor<Field>* diagonal_factor,
                                RightLookingSharedState<Field>* shared_state);

// Adds the schur complements of a single child onto its parent supernode's front.
template <class Field>
void MergeChildSchurComplement(Int supernode, Int child,
                               const SymmetricOrdering& ordering,
                               const LowerFactor<Field> *lower_factor,
                               const BlasMatrixView<Field> &child_schur_complement,
                               BlasMatrixView<Field> lower_block,
                               BlasMatrixView<Field> diagonal_block,
                               BlasMatrixView<Field> schur_complement,
                               bool freshShurComplement);

#ifdef CATAMARI_OPENMP
template <class Field>
void OpenMPMergeChildSchurComplements(
    Int merge_grain_size, Int supernode, const SymmetricOrdering& ordering,
    LowerFactor<Field>* lower_factor, DiagonalFactor<Field>* diagonal_factor,
    RightLookingSharedState<Field>* shared_state);
#endif  // ifdef CATAMARI_OPENMP

// L(m, m) -= L(m, d) * (D(d, d) * L(m, d)')
//          = L(m, d) * Z(:, m).
//
// The update is to the main_supernode_size x main_supernode_size dense
// diagonal block, L(m, m), with the densified L(m, d) matrix being
// descendant_main_intersect_size x descendant_supernode_size, and Z(:, m)
// being descendant_supernode_size x descendant_main_intersect_size.
template <class Field>
void UpdateDiagonalBlock(
    SymmetricFactorizationType factorization_type,
    const Buffer<Int>& supernode_starts, const LowerFactor<Field>& lower_factor,
    Int main_supernode, Int descendant_supernode, Int descendant_main_rel_row,
    const ConstBlasMatrixView<Field>& descendant_main_matrix,
    const ConstBlasMatrixView<Field>& scaled_transpose,
    BlasMatrixView<Field>* main_diag_block,
    BlasMatrixView<Field>* workspace_matrix);

// L(a, m) -= L(a, d) * (D(d, d) * L(m, d)')
//          = L(a, d) * Z(:, m).
//
// The update is to the main_active_intersect_size x main_supernode_size
// densified matrix L(a, m) using the
// descendant_active_intersect_size x descendant_supernode_size densified
// L(a, d) and thd escendant_supernode_size x descendant_main_intersect_size
// Z(:, m).
template <class Field>
void UpdateSubdiagonalBlock(
    SymmetricFactorizationType factorization_type, Int main_supernode,
    Int descendant_supernode, Int main_active_rel_row,
    Int descendant_main_rel_row,
    const ConstBlasMatrixView<Field>& descendant_main_matrix,
    Int descendant_active_rel_row, const Buffer<Int>& supernode_starts,
    const Buffer<Int>& supernode_member_to_index,
    const ConstBlasMatrixView<Field>& scaled_transpose,
    const ConstBlasMatrixView<Field>& descendant_active_matrix,
    const LowerFactor<Field>& lower_factor,
    BlasMatrixView<Field>* main_active_block,
    BlasMatrixView<Field>* workspace_matrix);

// Perform an in-place LDL' factorization of the supernodal diagonal block.
template <class Field>
Int FactorDiagonalBlock(
    Int block_size, SymmetricFactorizationType factorization_type,
    const DynamicRegularizationParams<Field>& dynamic_reg_params,
    BlasMatrixView<Field>* diagonal_block,
    std::vector<std::pair<Int, ComplexBase<Field>>>* dynamic_regularization);

#ifdef CATAMARI_OPENMP
// Perform an in-place LDL' factorization of the supernodal diagonal block.
template <class Field>
Int OpenMPFactorDiagonalBlock(
    Int tile_size, Int block_size,
    SymmetricFactorizationType factorization_type,
    const DynamicRegularizationParams<Field>& dynamic_reg_params,
    BlasMatrixView<Field>* diagonal_block, Buffer<Field>* buffer,
    std::vector<std::pair<Int, ComplexBase<Field>>>* dynamic_regularization);
#endif  // ifdef CATAMARI_OPENMP

// L(KNext:n, K) /= D(K, K) L(K, K)', or /= D(K, K) L(K, K)^T.
template <class Field>
void SolveAgainstDiagonalBlock(
    SymmetricFactorizationType factorization_type,
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* lower_matrix);

#ifdef CATAMARI_OPENMP
// L(KNext:n, K) /= D(K, K) L(K, K)', or /= D(K, K) L(K, K)^T.
template <class Field>
void OpenMPSolveAgainstDiagonalBlock(
    Int tile_size, SymmetricFactorizationType factorization_type,
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* lower_matrix);
#endif  // ifdef CATAMARI_OPENMP

}  // namespace supernodal_ldl
}  // namespace catamari

#include "catamari/sparse_ldl/supernodal/supernode_utils-impl.hpp"

#endif  // ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_SUPERNODE_UTILS_H_
