/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SUPERNODAL_LDL_SUPERNODE_UTILS_H_
#define CATAMARI_LDL_SUPERNODAL_LDL_SUPERNODE_UTILS_H_

#include "catamari/ldl/scalar_ldl.hpp"

namespace catamari {

struct SupernodalRelaxationControl {
  // If true, relaxed supernodes are created in a manner similar to the
  // suggestion from:
  //   Ashcraft and Grime, "The impact of relaxed supernode partitions on the
  //   multifrontal method", 1989.
  bool relax_supernodes = false;

  // The allowable number of explicit zeros in any relaxed supernode.
  Int allowable_supernode_zeros = 128;

  // The allowable ratio of explicit zeros in any relaxed supernode. This
  // should be interpreted as an *alternative* allowance for a supernode merge.
  // If the number of explicit zeros that would be introduced is less than or
  // equal to 'allowable_supernode_zeros', *or* the ratio of explicit zeros to
  // nonzeros is bounded by 'allowable_supernode_zero_ratio', then the merge
  // can procede.
  float allowable_supernode_zero_ratio = 0.01;
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

// Fills 'member_to_index' with a length 'num_rows' array whose i'th index
// is the index of the supernode containing column 'i'.
void MemberToIndex(Int num_rows, const std::vector<Int>& supernode_starts,
                   std::vector<Int>* member_to_index);

// Create the packed downlinks from the uplinks of an elimination forest.
void EliminationForestAndRootsFromParents(const std::vector<Int>& parents,
                                          std::vector<Int>* children,
                                          std::vector<Int>* child_offsets,
                                          std::vector<Int>* roots);

// Builds an elimination forest over the supernodes from an elimination forest
// over the nodes.
void ConvertFromScalarToSupernodalEliminationForest(
    Int num_supernodes, const std::vector<Int>& parents,
    const std::vector<Int>& member_to_index,
    std::vector<Int>* supernode_parents);

// Checks that a valid set of supernodes has been provided by explicitly
// computing each row pattern and ensuring that each intersects entire
// supernodes.
template <class Field>
bool ValidFundamentalSupernodes(const CoordinateMatrix<Field>& matrix,
                                const SymmetricOrdering& ordering,
                                const std::vector<Int>& supernode_sizes);

// Compute an unrelaxed supernodal partition using the existing ordering.
// We require that supernodes have dense diagonal blocks and equal structures
// below the diagonal block.
template <class Field>
void FormFundamentalSupernodes(const CoordinateMatrix<Field>& matrix,
                               const SymmetricOrdering& ordering,
                               const std::vector<Int>& parents,
                               const std::vector<Int>& degrees,
                               std::vector<Int>* supernode_sizes,
                               scalar_ldl::LowerStructure* scalar_structure);
#ifdef _OPENMP
template <class Field>
void MultithreadedFormFundamentalSupernodes(
    const CoordinateMatrix<Field>& matrix, const SymmetricOrdering& ordering,
    const std::vector<Int>& parents, const std::vector<Int>& degrees,
    std::vector<Int>* supernode_sizes,
    scalar_ldl::LowerStructure* scalar_structure);
#endif  // ifdef _OPENMP

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
// Because it is assumed that supernode 1 is the parent of supernode 0,
// the only places explicit nonzeros can be introduced are in:
//   * L10: if supernode 0 was not fully-connected to supernode 1.
//   * L20: if supernode 0's structure (with supernode 1 removed) does not
//     contain every member of the structure of supernode 1.
//
// Counting the number of explicit zeros that would be introduced is thus
// simply a matter of counting these mismatches.
//
// The reason that downstream explicit zeros would not be introduced is the
// same as the reason explicit zeros are not introduced into L21; supernode
// 1 is the parent of supernode 0.
//
MergableStatus MergableSupernode(
    Int child_tail, Int parent_tail, Int child_size, Int parent_size,
    Int num_child_explicit_zeros, Int num_parent_explicit_zeros,
    const std::vector<Int>& orig_member_to_index,
    const scalar_ldl::LowerStructure& scalar_structure,
    const SupernodalRelaxationControl& control);

void MergeChildren(Int parent, const std::vector<Int>& orig_supernode_starts,
                   const std::vector<Int>& orig_supernode_sizes,
                   const std::vector<Int>& orig_member_to_index,
                   const std::vector<Int>& children,
                   const std::vector<Int>& child_offsets,
                   const scalar_ldl::LowerStructure& scalar_structure,
                   const SupernodalRelaxationControl& control,
                   std::vector<Int>* supernode_sizes,
                   std::vector<Int>* num_explicit_zeros,
                   std::vector<Int>* last_merged_child,
                   std::vector<Int>* merge_parents);

// Walk up the tree in the original postordering, merging supernodes as we
// progress. The 'relaxed_permutation' and 'relaxed_inverse_permutation'
// variables are also inputs.
void RelaxSupernodes(const std::vector<Int>& orig_parents,
                     const std::vector<Int>& orig_supernode_sizes,
                     const std::vector<Int>& orig_supernode_starts,
                     const std::vector<Int>& orig_supernode_parents,
                     const std::vector<Int>& orig_supernode_degrees,
                     const std::vector<Int>& orig_member_to_index,
                     const scalar_ldl::LowerStructure& scalar_structure,
                     const SupernodalRelaxationControl& control,
                     std::vector<Int>* relaxed_permutation,
                     std::vector<Int>* relaxed_inverse_permutation,
                     std::vector<Int>* relaxed_parents,
                     std::vector<Int>* relaxed_supernode_parents,
                     std::vector<Int>* relaxed_supernode_degrees,
                     std::vector<Int>* relaxed_supernode_sizes,
                     std::vector<Int>* relaxed_supernode_starts,
                     std::vector<Int>* relaxed_supernode_member_to_index);

// Computes the sizes of the structures of a supernodal LDL' factorization.
template <class Field>
void SupernodalDegrees(const CoordinateMatrix<Field>& matrix,
                       const std::vector<Int>& permutation,
                       const std::vector<Int>& inverse_permutation,
                       const std::vector<Int>& supernode_sizes,
                       const std::vector<Int>& supernode_starts,
                       const std::vector<Int>& member_to_index,
                       const std::vector<Int>& parents,
                       std::vector<Int>* supernode_degrees);

// Fills an estimate of the work required to eliminate the subtree in a
// right-looking factorization.
template <class Field>
void FillSubtreeWorkEstimates(Int root, const AssemblyForest& supernode_ofrest,
                              const LowerFactor<Field>& lower_factor,
                              std::vector<double>* work_estimates);

// Fills in the structure indices for the lower factor.
template <class Field>
void FillStructureIndices(const CoordinateMatrix<Field>& matrix,
                          const SymmetricOrdering& ordering,
                          const std::vector<Int>& parents,
                          const std::vector<Int>& supernode_sizes,
                          const std::vector<Int>& supernode_member_to_index,
                          LowerFactor<Field>* lower_factor);
#ifdef _OPENMP
template <class Field>
void MultithreadedFillStructureIndices(
    const CoordinateMatrix<Field>& matrix, const SymmetricOrdering& ordering,
    const std::vector<Int>& parents, const std::vector<Int>& supernode_sizes,
    const std::vector<Int>& supernode_member_to_index,
    LowerFactor<Field>* lower_factor);
#endif  // ifdef _OPENMP

// Fill in the nonzeros from the original sparse matrix.
template <class Field>
void FillNonzeros(const CoordinateMatrix<Field>& matrix,
                  const std::vector<Int>& permutation,
                  const std::vector<Int>& inverse_permutation,
                  const std::vector<Int>& supernode_starts,
                  const std::vector<Int>& supernode_sizes,
                  const std::vector<Int>& supernode_member_to_index,
                  LowerFactor<Field>* lower_factor,
                  DiagonalFactor<Field>* diagonal_factor);

// Computes the supernodal nonzero pattern of L(row, :) in
// row_structure[0 : num_packed - 1].
template <class Field>
Int ComputeRowPattern(const CoordinateMatrix<Field>& matrix,
                      const std::vector<Int>& permutation,
                      const std::vector<Int>& inverse_permutation,
                      const std::vector<Int>& supernode_sizes,
                      const std::vector<Int>& supernode_starts,
                      const std::vector<Int>& member_to_index,
                      const std::vector<Int>& supernode_parents,
                      Int main_supernode, Int* pattern_flags,
                      Int* row_structure);

// Store the scaled adjoint update matrix, Z(d, m) = D(d, d) L(m, d)', or
// the scaled transpose, Z(d, m) = D(d, d) L(m, d)^T.
template <class Field>
void FormScaledTranspose(SymmetricFactorizationType factorization_type,
                         const ConstBlasMatrix<Field>& diagonal_block,
                         const ConstBlasMatrix<Field>& matrix,
                         BlasMatrix<Field>* scaled_transpose);

// Moves the pointers for the main supernode down to the active supernode of
// the descendant column block.
template <class Field>
void SeekForMainActiveRelativeRow(
    Int main_supernode, Int descendant_supernode, Int descendant_active_rel_row,
    const std::vector<Int>& supernode_member_to_index,
    const LowerFactor<Field>& lower_factor, Int* main_active_rel_row,
    const Int** main_active_intersect_sizes);

// L(m, m) -= L(m, d) * (D(d, d) * L(m, d)')
//          = L(m, d) * Z(:, m).
//
// The update is to the main_supernode_size x main_supernode_size dense
// diagonal block, L(m, m), with the densified L(m, d) matrix being
// descendant_main_intersect_size x descendant_supernode_size, and Z(:, m)
// being descendant_supernode_size x descendant_main_intersect_size.
template <class Field>
void UpdateDiagonalBlock(SymmetricFactorizationType factorization_type,
                         const std::vector<Int>& supernode_starts,
                         const LowerFactor<Field>& lower_factor,
                         Int main_supernode, Int descendant_supernode,
                         Int descendant_main_rel_row,
                         const ConstBlasMatrix<Field>& descendant_main_matrix,
                         const ConstBlasMatrix<Field>& scaled_transpose,
                         BlasMatrix<Field>* main_diag_block,
                         BlasMatrix<Field>* workspace_matrix);

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
    Int main_supernode, Int descendant_supernode, Int main_active_rel_row,
    Int descendant_main_rel_row, Int descendant_active_rel_row,
    const std::vector<Int>& supernode_starts,
    const std::vector<Int>& supernode_member_to_index,
    const ConstBlasMatrix<Field>& scaled_transpose,
    const ConstBlasMatrix<Field>& descendant_active_matrix,
    const LowerFactor<Field>& lower_factor,
    BlasMatrix<Field>* main_active_block, BlasMatrix<Field>* workspace_matrix);

// Perform an in-place LDL' factorization of the supernodal diagonal block.
template <class Field>
Int FactorDiagonalBlock(SymmetricFactorizationType factorization_type,
                        BlasMatrix<Field>* diagonal_block);

#ifdef _OPENMP
// Perform an in-place LDL' factorization of the supernodal diagonal block.
template <class Field>
Int MultithreadedFactorDiagonalBlock(
    Int tile_size, SymmetricFactorizationType factorization_type,
    BlasMatrix<Field>* diagonal_block, std::vector<Field>* buffer);
#endif  // ifdef _OPENMP

// L(KNext:n, K) /= D(K, K) L(K, K)', or /= D(K, K) L(K, K)^T.
template <class Field>
void SolveAgainstDiagonalBlock(SymmetricFactorizationType factorization_type,
                               const ConstBlasMatrix<Field>& triangular_matrix,
                               BlasMatrix<Field>* lower_matrix);

#ifdef _OPENMP
// L(KNext:n, K) /= D(K, K) L(K, K)', or /= D(K, K) L(K, K)^T.
template <class Field>
void MultithreadedSolveAgainstDiagonalBlock(
    Int tile_size, SymmetricFactorizationType factorization_type,
    const ConstBlasMatrix<Field>& triangular_matrix,
    BlasMatrix<Field>* lower_matrix);
#endif  // ifdef _OPENMP

}  // namespace supernodal_ldl
}  // namespace catamari

#include "catamari/ldl/supernodal_ldl/supernode_utils-impl.hpp"

#endif  // ifndef CATAMARI_LDL_SUPERNODAL_LDL_SUPERNODE_UTILS_H_
