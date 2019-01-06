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

  // The choice of either left-looking or right-looking LDL' factorization.
  // There is currently no supernodal up-looking support.
  LDLAlgorithm algorithm = kRightLookingLDL;

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
  SymmetricOrdering ordering;

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

  // Factors the given matrix using the prescribed permutation.
  LDLResult Factor(const CoordinateMatrix<Field>& matrix,
                   const SymmetricOrdering& manual_ordering,
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

  // Incorporates the details and work required to process the supernode with
  // the given size and degree into the factorization result.
  static void IncorporateSupernodeIntoLDLResult(Int supernode_size, Int degree,
                                                LDLResult* result);

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

  struct RightLookingSharedState {
    // The Schur complement matrices for each of the supernodes in the
    // multifrontal method. Each front should only be allocated while it is
    // actively in use.
    std::vector<BlasMatrix<Field>> schur_complements;

    // The underlying buffers for the Schur complement portions of the fronts.
    // They are allocated and deallocated as the factorization progresses.
    std::vector<std::vector<Field>> schur_complement_buffers;
  };

  // Form the (possibly relaxed) supernodes for the factorization.
  void FormSupernodes(const CoordinateMatrix<Field>& matrix,
                      const SupernodalRelaxationControl& control,
                      std::vector<Int>* parents,
                      std::vector<Int>* supernode_degrees,
                      std::vector<Int>* supernode_parents);

  void InitializeFactors(const CoordinateMatrix<Field>& matrix,
                         const std::vector<Int>& parents,
                         const std::vector<Int>& supernode_degrees);

  LDLResult LeftLooking(const CoordinateMatrix<Field>& matrix,
                        const Control& control);

  LDLResult RightLooking(const CoordinateMatrix<Field>& matrix,
                         const Control& control);

  bool LeftLookingSubtree(Int supernode, const CoordinateMatrix<Field>& matrix,
                          const std::vector<Int>& supernode_parents,
                          const std::vector<Int>& supernode_children,
                          const std::vector<Int>& supernode_child_offsets,
                          LeftLookingSharedState* shared_state,
                          LeftLookingPrivateState* private_state,
                          LDLResult* result);

  bool RightLookingSubtree(Int supernode, const CoordinateMatrix<Field>& matrix,
                           const std::vector<Int>& supernode_parents,
                           const std::vector<Int>& supernode_children,
                           const std::vector<Int>& supernode_child_offsets,
                           RightLookingSharedState* shared_state,
                           LDLResult* result);

  void LeftLookingSupernodeUpdate(Int main_supernode,
                                  const CoordinateMatrix<Field>& matrix,
                                  const std::vector<Int>& supernode_parents,
                                  LeftLookingSharedState* shared_state,
                                  LeftLookingPrivateState* private_state);

  bool LeftLookingSupernodeFinalize(Int main_supernode, LDLResult* result);

  void MergeChildSchurComplements(
      Int supernode, const std::vector<Int>& supernode_children,
      const std::vector<Int>& supernode_child_offsets,
      RightLookingSharedState* shared_state);

  bool RightLookingSupernodeFinalize(
      Int supernode, const std::vector<Int>& supernode_children,
      const std::vector<Int>& supernode_child_offsets,
      RightLookingSharedState* shared_state, LDLResult* result);

#ifdef _OPENMP
  void MultithreadedLeftLookingSupernodeUpdate(
      Int main_supernode, const CoordinateMatrix<Field>& matrix,
      const std::vector<Int>& supernode_parents,
      LeftLookingSharedState* shared_state,
      std::vector<LeftLookingPrivateState>* private_states);

  bool MultithreadedLeftLookingSupernodeFinalize(
      Int main_supernode, std::vector<LeftLookingPrivateState>* private_states,
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

  LDLResult MultithreadedRightLooking(const CoordinateMatrix<Field>& matrix,
                                      const Control& control);

  bool MultithreadedRightLookingSubtree(
      Int level, Int max_parallel_levels, Int supernode,
      const CoordinateMatrix<Field>& matrix,
      const std::vector<Int>& supernode_parents,
      const std::vector<Int>& supernode_children,
      const std::vector<Int>& supernode_child_offsets,
      const std::vector<double>& work_estimates,
      RightLookingSharedState* shared_state, LDLResult* result);

  void MultithreadedMergeChildSchurComplements(
      Int supernode, const std::vector<Int>& supernode_children,
      const std::vector<Int>& supernode_child_offsets,
      RightLookingSharedState* shared_state);

  bool MultithreadedRightLookingSupernodeFinalize(
      Int supernode, const std::vector<Int>& supernode_children,
      const std::vector<Int>& supernode_child_offsets,
      RightLookingSharedState* shared_state, LDLResult* result);
#endif  // ifdef _OPENMP
};

}  // namespace supernodal_ldl
}  // namespace catamari

#include "catamari/ldl/supernodal_ldl/factorization-impl.hpp"

#endif  // ifndef CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_H_
