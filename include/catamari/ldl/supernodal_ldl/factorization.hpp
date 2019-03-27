/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_H_
#define CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_H_

#include "catamari/buffer.hpp"
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

  // The algorithmic block size for the factorization.
  Int block_size = 64;

#ifdef CATAMARI_OPENMP
  // The size of the matrix tiles for factorization OpenMP tasks.
  Int factor_tile_size = 128;

  // The size of the matrix tiles for dense outer product OpenMP tasks.
  Int outer_product_tile_size = 240;

  // The number of columns to group into a single task when multithreading
  // the addition of child Schur complement updates onto the parent.
  Int merge_grain_size = 500;

  // The number of columns to group into a single task when multithreading
  // the scalar structure formation.
  Int sort_grain_size = 200;

  // The minimum ratio of the amount of work in a subtree relative to the
  // nominal amount of flops assigned to each thread (total_work / max_threads)
  // before OpenMP subtasks are launched in the subtree.
  double parallel_ratio_threshold = 0.02;

  // The minimum number of flops in a subtree before OpenMP subtasks are
  // generated.
  double min_parallel_threshold = 1e5;
#endif  // ifdef CATAMARI_OPENMP

#ifdef CATAMARI_ENABLE_TIMERS
  // The max number of levels of the supernodal tree to visualize timings of.
  Int max_timing_levels = 4;

  // Whether isolated diagonal entries should have their timings visualized.
  bool avoid_timing_isolated_roots = true;

  // The name of the Graphviz file for the inclusive timing annotations.
  std::string inclusive_timings_filename = "inclusive.gv";

  // The name of the Graphviz file for the exclusive timing annotations.
  std::string exclusive_timings_filename = "exclusive.gv";
#endif  // ifdef CATAMARI_ENABLE_TIMERS
};

// The user-facing data structure for storing a supernodal LDL' factorization.
template <class Field>
class Factorization {
 public:
  // Factors the given matrix using the prescribed permutation.
  LDLResult Factor(const CoordinateMatrix<Field>& matrix,
                   const SymmetricOrdering& manual_ordering,
                   const Control& control);

  // Factors the given matrix after having previously factored another matrix
  // with the same sparsity pattern.
  LDLResult RefactorWithFixedSparsityPattern(
      const CoordinateMatrix<Field>& matrix);

  // Solve a set of linear systems using the factorization.
  void Solve(BlasMatrixView<Field>* right_hand_sides) const;

  // Solves a set of linear systems using the lower-triangular factor.
  void LowerTriangularSolve(BlasMatrixView<Field>* right_hand_sides) const;

#ifdef CATAMARI_OPENMP
  void OpenMPLowerTriangularSolve(
      BlasMatrixView<Field>* right_hand_sides) const;
#endif  // ifdef CATAMARI_OPENMP

  // Solves a set of linear systems using the diagonal factor.
  void DiagonalSolve(BlasMatrixView<Field>* right_hand_sides) const;

#ifdef CATAMARI_OPENMP
  void OpenMPDiagonalSolve(BlasMatrixView<Field>* right_hand_sides) const;
#endif  // ifdef CATAMARI_OPENMP

  // Solves a set of linear systems using the trasnpose (or adjoint) of the
  // lower-triangular factor.
  void LowerTransposeTriangularSolve(
      BlasMatrixView<Field>* right_hand_sides) const;

#ifdef CATAMARI_OPENMP
  void OpenMPLowerTransposeTriangularSolve(
      BlasMatrixView<Field>* right_hand_sides) const;
#endif  // ifdef CATAMARI_OPENMP

  // Prints the diagonal of the factorization.
  void PrintDiagonalFactor(const std::string& label, std::ostream& os) const;

  // Prints the unit lower-triangular matrix.
  void PrintLowerFactor(const std::string& label, std::ostream& os) const;

  // Incorporates the details and work required to process the supernode with
  // the given size and degree into the factorization result.
  static void IncorporateSupernodeIntoLDLResult(Int supernode_size, Int degree,
                                                LDLResult* result);

  // Adds in the contribution of a subtree into an overall result.
  static void MergeContribution(const LDLResult& contribution,
                                LDLResult* result);

 private:
  // The control structure for the factorization.
  Control control_;

  // The representation of the permutation matrix P so that P A P' should be
  // factored. Typically, this permutation is the composition of a
  // fill-reducing ordering and a supernodal relaxation permutation.
  SymmetricOrdering ordering_;

  // An array of length 'num_rows'; the i'th member is the index of the
  // supernode containing column 'i'.
  Buffer<Int> supernode_member_to_index_;

  // The largest supernode size in the factorization.
  Int max_supernode_size_;

  // The largest degree of a supernode in the factorization.
  Int max_degree_;

  // The largest number of entries in any supernode's lower block.
  Int max_lower_block_size_;

  // The subdiagonal-block portion of the lower-triangular factor.
  std::unique_ptr<LowerFactor<Field>> lower_factor_;

  // The block-diagonal factor.
  std::unique_ptr<DiagonalFactor<Field>> diagonal_factor_;

  // Performs the initial analysis (and factorization initialization) for a
  // particular sparisty pattern. Subsequent factorizations with the same
  // sparsity pattern can reuse the symbolic analysis.
  void InitialFactorizationSetup(const CoordinateMatrix<Field>& matrix);
#ifdef CATAMARI_OPENMP
  void OpenMPInitialFactorizationSetup(const CoordinateMatrix<Field>& matrix);
#endif  // ifdef CATAMARI_OPENMP

  // TODO(Jack Poulson): Add ReinitializeFactorization.

  // Form the (possibly relaxed) supernodes for the factorization.
  void FormSupernodes(const CoordinateMatrix<Field>& matrix,
                      AssemblyForest* forest, Buffer<Int>* supernode_degrees);
#ifdef CATAMARI_OPENMP
  void OpenMPFormSupernodes(const CoordinateMatrix<Field>& matrix,
                            AssemblyForest* forest,
                            Buffer<Int>* supernode_degrees);
#endif  // ifdef CATAMARI_OPENMP

  void InitializeFactors(const CoordinateMatrix<Field>& matrix,
                         const AssemblyForest& forest,
                         const Buffer<Int>& supernode_degrees);
#ifdef CATAMARI_OPENMP
  void OpenMPInitializeFactors(const CoordinateMatrix<Field>& matrix,
                               const AssemblyForest& forest,
                               const Buffer<Int>& supernode_degrees);
#endif  // ifdef CATAMARI_OPENMP

  LDLResult LeftLooking(const CoordinateMatrix<Field>& matrix);
#ifdef CATAMARI_OPENMP
  LDLResult OpenMPLeftLooking(const CoordinateMatrix<Field>& matrix);
#endif  // ifdef CATAMARI_OPENMP

  LDLResult RightLooking(const CoordinateMatrix<Field>& matrix);
#ifdef CATAMARI_OPENMP
  LDLResult OpenMPRightLooking(const CoordinateMatrix<Field>& matrix);
#endif  // ifdef CATAMARI_OPENMP

  bool LeftLookingSubtree(Int supernode, const CoordinateMatrix<Field>& matrix,
                          LeftLookingSharedState* shared_state,
                          PrivateState<Field>* private_state,
                          LDLResult* result);
#ifdef CATAMARI_OPENMP
  bool OpenMPLeftLookingSubtree(Int supernode,
                                const CoordinateMatrix<Field>& matrix,
                                const Buffer<double>& work_estimates,
                                double min_parallel_work,
                                LeftLookingSharedState* shared_state,
                                Buffer<PrivateState<Field>>* private_states,
                                LDLResult* result);
#endif  // ifdef CATAMARI_OPENMP

  bool RightLookingSubtree(Int supernode, const CoordinateMatrix<Field>& matrix,
                           RightLookingSharedState<Field>* shared_state,
                           PrivateState<Field>* private_state,
                           LDLResult* result);
#ifdef CATAMARI_OPENMP
  bool OpenMPRightLookingSubtree(Int supernode,
                                 const CoordinateMatrix<Field>& matrix,
                                 const Buffer<double>& work_estimates,
                                 double min_parallel_work,
                                 RightLookingSharedState<Field>* shared_state,
                                 Buffer<PrivateState<Field>>* private_states,
                                 LDLResult* result);
#endif  // ifdef CATAMARI_OPENMP

  void LeftLookingSupernodeUpdate(Int main_supernode,
                                  const CoordinateMatrix<Field>& matrix,
                                  LeftLookingSharedState* shared_state,
                                  PrivateState<Field>* private_state);
#ifdef CATAMARI_OPENMP
  void OpenMPLeftLookingSupernodeUpdate(
      Int main_supernode, const CoordinateMatrix<Field>& matrix,
      LeftLookingSharedState* shared_state,
      Buffer<PrivateState<Field>>* private_states);
#endif  // ifdef CATAMARI_OPENMP

  bool LeftLookingSupernodeFinalize(Int main_supernode, LDLResult* result);
#ifdef CATAMARI_OPENMP
  bool OpenMPLeftLookingSupernodeFinalize(
      Int supernode, Buffer<PrivateState<Field>>* private_states,
      LDLResult* result);
#endif  // ifdef CATAMARI_OPENMP

  bool RightLookingSupernodeFinalize(
      Int supernode, RightLookingSharedState<Field>* shared_state,
      PrivateState<Field>* private_state, LDLResult* result);
#ifdef CATAMARI_OPENMP
  bool OpenMPRightLookingSupernodeFinalize(
      Int supernode, RightLookingSharedState<Field>* shared_state,
      Buffer<PrivateState<Field>>* private_state, LDLResult* result);
#endif  // ifdef CATAMARI_OPENMP

  // Performs the portion of the lower-triangular solve corresponding to the
  // subtree with the given root supernode.
  void LowerTriangularSolveRecursion(Int supernode,
                                     BlasMatrixView<Field>* right_hand_sides,
                                     Buffer<Field>* workspace) const;
#ifdef CATAMARI_OPENMP
  void OpenMPLowerTriangularSolveRecursion(
      Int supernode, BlasMatrixView<Field>* right_hand_sides,
      RightLookingSharedState<Field>* shared_state) const;
#endif  // ifdef CATAMARI_OPENMP

  // Performs the trapezoidal solve associated with a particular supernode.
  void LowerSupernodalTrapezoidalSolve(Int supernode,
                                       BlasMatrixView<Field>* right_hand_sides,
                                       Buffer<Field>* workspace) const;
#ifdef CATAMARI_OPENMP
  void OpenMPLowerSupernodalTrapezoidalSolve(
      Int supernode, BlasMatrixView<Field>* right_hand_sides,
      RightLookingSharedState<Field>* shared_state) const;
#endif  // ifdef CATAMARI_OPENMP

  // Performs the portion of the transposed lower-triangular solve
  // corresponding to the subtree with the given root supernode.
  void LowerTransposeTriangularSolveRecursion(
      Int supernode, BlasMatrixView<Field>* right_hand_sides,
      Buffer<Field>* packed_input_buf) const;
#ifdef CATAMARI_OPENMP
  void OpenMPLowerTransposeTriangularSolveRecursion(
      Int supernode, BlasMatrixView<Field>* right_hand_sides,
      Buffer<Buffer<Field>>* private_packed_input_bufs) const;
#endif  // ifdef CATAMARI_OPENMP

  // Performs the trapezoidal solve associated with a particular supernode.
  void LowerTransposeSupernodalTrapezoidalSolve(
      Int supernode, BlasMatrixView<Field>* right_hand_sides,
      Buffer<Field>* workspace) const;
};

}  // namespace supernodal_ldl
}  // namespace catamari

#include "catamari/ldl/supernodal_ldl/factorization/common-impl.hpp"
#include "catamari/ldl/supernodal_ldl/factorization/common_openmp-impl.hpp"
#include "catamari/ldl/supernodal_ldl/factorization/io-impl.hpp"
#include "catamari/ldl/supernodal_ldl/factorization/left_looking-impl.hpp"
#include "catamari/ldl/supernodal_ldl/factorization/left_looking_openmp-impl.hpp"
#include "catamari/ldl/supernodal_ldl/factorization/right_looking-impl.hpp"
#include "catamari/ldl/supernodal_ldl/factorization/right_looking_openmp-impl.hpp"
#include "catamari/ldl/supernodal_ldl/factorization/solve-impl.hpp"
#include "catamari/ldl/supernodal_ldl/factorization/solve_openmp-impl.hpp"

#endif  // ifndef CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_H_
