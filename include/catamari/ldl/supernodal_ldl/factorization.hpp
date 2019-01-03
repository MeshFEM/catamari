/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_H_
#define CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_H_

#include "catamari/ldl/supernodal_ldl/diagonal_factor.hpp"
#include "catamari/ldl/supernodal_ldl/lower_factor.hpp"
#include "catamari/ldl/supernodal_ldl/supernode_utils.hpp"

namespace catamari {
namespace supernodal_ldl {

// Configuration options for supernodal LDL' factorization.
struct Control {
  // Determines the style of the factorization.
  SymmetricFactorizationType factorization_type;

  // Configuration for the supernodal relaxation.
  SupernodalRelaxationControl relaxation_control;

  // The minimal supernode size for an out-of-place trapezoidal solve to be
  // used.
  Int forward_solve_out_of_place_supernode_threshold = 10;

  // The minimal supernode size for an out-of-place trapezoidal solve to be
  // used.
  Int backward_solve_out_of_place_supernode_threshold = 10;
};

// The user-facing data structure for storing a supernodal LDL' factorization.
template <class Field>
class Factorization {
 public:
  // Marks the type of factorization employed.
  SymmetricFactorizationType factorization_type;

  // An array of length 'num_supernodes'; the i'th member is the size of the
  // i'th supernode.
  std::vector<Int> supernode_sizes;

  // An array of length 'num_supernodes + 1'; the i'th member, for
  // 0 <= i < num_supernodes, is the principal member of the i'th supernode.
  // The last member is equal to 'num_rows'.
  std::vector<Int> supernode_starts;

  // An array of length 'num_rows'; the i'th member is the index of the
  // supernode containing column 'i'.
  std::vector<Int> supernode_member_to_index;

  // The largest supernode size in the factorization.
  Int max_supernode_size;

  // The largest degree of a supernode in the factorization.
  Int max_degree;

  // The largest number of entries in the block row to the left of a diagonal
  // block.
  // NOTE: This is only needed for multithreaded factorizations.
  Int max_descendant_entries;

  // If the following is nonempty, then, if the permutation is a matrix P, the
  // matrix P A P' has been factored. Typically, this permutation is the
  // composition of a fill-reducing ordering and a supernodal relaxation
  // permutation.
  std::vector<Int> permutation;

  // The inverse of the above permutation (if it is nontrivial).
  std::vector<Int> inverse_permutation;

  // The subdiagonal-block portion of the lower-triangular factor.
  std::unique_ptr<LowerFactor<Field>> lower_factor;

  // The block-diagonal factor.
  std::unique_ptr<DiagonalFactor<Field>> diagonal_factor;

  // The minimal supernode size for an out-of-place trapezoidal solve to be
  // used.
  Int forward_solve_out_of_place_supernode_threshold;

  // The minimal supernode size for an out-of-place trapezoidal solve to be
  // used.
  Int backward_solve_out_of_place_supernode_threshold;

  // Factors the given matrix using an automatically determined permutation.
  LDLResult Factor(const CoordinateMatrix<Field>& matrix,
                   const Control& control);

  // Factors the given matrix using the prescribed permutation.
  LDLResult Factor(const CoordinateMatrix<Field>& matrix,
                   const std::vector<Int>& manual_permutation,
                   const std::vector<Int>& inverse_manual_permutation,
                   const Control& control);

  // Solve a set of linear systems using the factorization.
  void Solve(BlasMatrix<Field>* matrix) const;

  // Solves a set of linear systems using the lower-triangular factor.
  void LowerTriangularSolve(BlasMatrix<Field>* matrix) const;

  // Solves a set of linear systems using the diagonal factor.
  void DiagonalSolve(BlasMatrix<Field>* matrix) const;

  // Solves a set of linear systems using the trasnpose (or adjoint) of the
  // lower-triangular factor.
  void LowerTransposeTriangularSolve(BlasMatrix<Field>* matrix) const;

  // Prints the diagonal of the factorization.
  void PrintDiagonalFactor(const std::string& label, std::ostream& os) const;

  // Prints the unit lower-triangular matrix.
  void PrintLowerFactor(const std::string& label, std::ostream& os) const;

 private:
  struct LeftLookingSharedState {
    // The relative index of the active supernode within each supernode's
    // structure.
    std::vector<Int> rel_rows;

    // Pointers to the active supernode intersection size within each
    // supernode's structure.
    std::vector<const Int*> intersect_ptrs;
  };

  struct LeftLookingPrivateState {
    // An integer workspace for storing the supernodes in the current row
    // pattern.
    std::vector<Int> row_structure;

    // A data structure for marking whether or not a supernode is in the pattern
    // of the active row of the lower-triangular factor.
    std::vector<Int> pattern_flags;

    // A buffer for storing (scaled) transposed descendant blocks.
    std::vector<Field> scaled_transpose_buffer;

    // A buffer for storing updates to the current supernode column.
    std::vector<Field> workspace_buffer;
  };

  // Form the (possibly relaxed) supernodes for the factorization.
  void FormSupernodes(const CoordinateMatrix<Field>& matrix,
                      const SupernodalRelaxationControl& control,
                      std::vector<Int>* parents,
                      std::vector<Int>* supernode_degrees,
                      std::vector<Int>* supernode_parents);

  LDLResult LeftLooking(const CoordinateMatrix<Field>& matrix,
                        const Control& control);

  void InitializeLeftLookingFactors(const CoordinateMatrix<Field>& matrix,
                                    const std::vector<Int>& parents,
                                    const std::vector<Int>& supernode_degrees);

  void LeftLookingSupernodeUpdate(Int main_supernode,
                                  const CoordinateMatrix<Field>& matrix,
                                  const std::vector<Int>& supernode_parents,
                                  LeftLookingSharedState* shared_state,
                                  LeftLookingPrivateState* private_state);

  bool LeftLookingSupernodeFinalize(Int main_supernode, LDLResult* result);

#ifdef _OPENMP
  void MultithreadedLeftLookingSupernodeUpdate(
      Int main_supernode, const CoordinateMatrix<Field>& matrix,
      const std::vector<Int>& supernode_parents,
      LeftLookingSharedState* shared_state,
      std::vector<LeftLookingPrivateState>* private_states);

  bool MultithreadedLeftLookingSupernodeFinalize(
      Int main_supernode, std::vector<LeftLookingPrivateState>* private_states,
      LDLResult* result);

  bool LeftLookingSubtree(Int supernode, const CoordinateMatrix<Field>& matrix,
                          const std::vector<Int>& supernode_parents,
                          const std::vector<Int>& supernode_children,
                          const std::vector<Int>& supernode_child_offsets,
                          LeftLookingSharedState* shared_state,
                          LeftLookingPrivateState* private_state,
                          LDLResult* result);

  bool MultithreadedLeftLookingSubtree(
      Int level, Int max_parallel_levels, Int supernode,
      const CoordinateMatrix<Field>& matrix,
      const std::vector<Int>& supernode_parents,
      const std::vector<Int>& supernode_children,
      const std::vector<Int>& supernode_child_offsets,
      LeftLookingSharedState* shared_state,
      std::vector<LeftLookingPrivateState>* private_states, LDLResult* result);

  LDLResult MultithreadedLeftLooking(const CoordinateMatrix<Field>& matrix,
                                     const Control& control);
#endif  // ifdef _OPENMP
};

// Fills in the structure indices for the lower factor.
template <class Field>
void FillStructureIndices(const CoordinateMatrix<Field>& matrix,
                          const std::vector<Int>& permutation,
                          const std::vector<Int>& inverse_permutation,
                          const std::vector<Int>& parents,
                          const std::vector<Int>& supernode_sizes,
                          const std::vector<Int>& supernode_member_to_index,
                          LowerFactor<Field>* lower_factor,
                          Int* max_descendant_entries);

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

// Moves the pointers for the main supernode down to the active supernode of
// the descendant column block.
template <class Field>
void SeekForMainActiveRelativeRow(
    Int main_supernode, Int descendant_supernode, Int descendant_active_rel_row,
    const std::vector<Int>& supernode_member_to_index,
    const LowerFactor<Field>& lower_factor, Int* main_active_rel_row,
    const Int** main_active_intersect_sizes);

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
    BlasMatrix<Field>* diagonal_block);
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

#include "catamari/ldl/supernodal_ldl/factorization-impl.hpp"

#endif  // ifndef CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_H_
