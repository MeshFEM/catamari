/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_H_

#include "catamari/buffer.hpp"
#include "catamari/sparse_ldl/supernodal/diagonal_factor.hpp"
#include "catamari/sparse_ldl/supernodal/lower_factor.hpp"
#include "catamari/sparse_ldl/supernodal/supernode_utils.hpp"

#ifdef CATAMARI_ENABLE_TIMERS
#include "quotient/timer.hpp"
#endif  // ifdef CATAMARI_ENABLE_TIMERS

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

#ifdef CATAMARI_ENABLE_TIMERS
struct FactorizationProfile {
  quotient::Timer scalar_elimination_forest;

  quotient::Timer supernodal_elimination_forest;

  quotient::Timer relax_supernodes;

  quotient::Timer initialize_factors;

  quotient::Timer gemm;
  double gemm_gflops = 0;

  quotient::Timer herk;
  double herk_gflops = 0;

  quotient::Timer trsm;
  double trsm_gflops = 0;

  quotient::Timer cholesky;
  double cholesky_gflops = 0;

  quotient::Timer merge;

  FactorizationProfile()
      : scalar_elimination_forest("scalar_elimination_forest"),
        supernodal_elimination_forest("supernodal_elimination_forest"),
        relax_supernodes("relax_supernodes"),
        initialize_factors("initialize_factors"),
        gemm("gemm"),
        herk("herk"),
        trsm("trsm"),
        cholesky("cholesky"),
        merge("merge") {}

  void Reset() {
    scalar_elimination_forest.Reset(scalar_elimination_forest.Name());
    supernodal_elimination_forest.Reset(supernodal_elimination_forest.Name());
    relax_supernodes.Reset(relax_supernodes.Name());
    initialize_factors.Reset(initialize_factors.Name());
    gemm.Reset(gemm.Name());
    gemm_gflops = 0;
    herk.Reset(herk.Name());
    herk_gflops = 0;
    trsm.Reset(trsm.Name());
    trsm_gflops = 0;
    cholesky.Reset(cholesky.Name());
    cholesky_gflops = 0;
    merge.Reset(merge.Name());
  }
};

std::ostream& operator<<(std::ostream& os,
                         const FactorizationProfile& profile) {
  os << profile.scalar_elimination_forest << "\n"
     << profile.supernodal_elimination_forest << "\n"
     << profile.relax_supernodes << "\n"
     << profile.initialize_factors << "\n"
     << profile.merge << "\n"
     << profile.gemm << " (GFlops: " << profile.gemm_gflops
     << ", GFlop/sec: " << profile.gemm_gflops / profile.gemm.TotalSeconds()
     << ")\n"
     << profile.herk << " (GFlops: " << profile.herk_gflops
     << ", GFlop/sec: " << profile.herk_gflops / profile.herk.TotalSeconds()
     << ")\n"
     << profile.trsm << " (GFlops: " << profile.trsm_gflops
     << ", GFlop/sec: " << profile.trsm_gflops / profile.trsm.TotalSeconds()
     << ")\n"
     << profile.cholesky << " (GFlops: " << profile.cholesky_gflops
     << ", GFlop/sec: "
     << profile.cholesky_gflops / profile.cholesky.TotalSeconds() << ")\n";
  return os;
}
#endif  // ifdef CATAMARI_ENABLE_TIMERS

// The user-facing data structure for storing a supernodal LDL' factorization.
template <class Field>
class Factorization {
 public:
#ifdef CATAMARI_ENABLE_TIMERS
  FactorizationProfile profile;
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  // Factors the given matrix using the prescribed permutation.
  SparseLDLResult Factor(const CoordinateMatrix<Field>& matrix,
                         const SymmetricOrdering& manual_ordering,
                         const Control& control);

  // Factors the given matrix after having previously factored another matrix
  // with the same sparsity pattern.
  SparseLDLResult RefactorWithFixedSparsityPattern(
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
                                                SparseLDLResult* result);

  // Adds in the contribution of a subtree into an overall result.
  static void MergeContribution(const SparseLDLResult& contribution,
                                SparseLDLResult* result);

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

  // Initializes a supernodal block column of the factorization using the
  // input matrix.
  void InitializeBlockColumn(Int supernode,
                             const CoordinateMatrix<Field>& matrix);
#ifdef CATAMARI_OPENMP
  void OpenMPInitializeBlockColumn(Int supernode,
                                   const CoordinateMatrix<Field>& matrix);
#endif  // ifdef CATAMARI_OPENMP

  void InitializeFactors(const CoordinateMatrix<Field>& matrix,
                         const AssemblyForest& forest,
                         const Buffer<Int>& supernode_degrees);
#ifdef CATAMARI_OPENMP
  void OpenMPInitializeFactors(const CoordinateMatrix<Field>& matrix,
                               const AssemblyForest& forest,
                               const Buffer<Int>& supernode_degrees);
#endif  // ifdef CATAMARI_OPENMP

  SparseLDLResult LeftLooking(const CoordinateMatrix<Field>& matrix);
#ifdef CATAMARI_OPENMP
  SparseLDLResult OpenMPLeftLooking(const CoordinateMatrix<Field>& matrix);
#endif  // ifdef CATAMARI_OPENMP

  SparseLDLResult RightLooking(const CoordinateMatrix<Field>& matrix);
#ifdef CATAMARI_OPENMP
  SparseLDLResult OpenMPRightLooking(const CoordinateMatrix<Field>& matrix);
#endif  // ifdef CATAMARI_OPENMP

  bool LeftLookingSubtree(Int supernode, const CoordinateMatrix<Field>& matrix,
                          LeftLookingSharedState* shared_state,
                          PrivateState<Field>* private_state,
                          SparseLDLResult* result);
#ifdef CATAMARI_OPENMP
  bool OpenMPLeftLookingSubtree(Int supernode,
                                const CoordinateMatrix<Field>& matrix,
                                const Buffer<double>& work_estimates,
                                double min_parallel_work,
                                LeftLookingSharedState* shared_state,
                                Buffer<PrivateState<Field>>* private_states,
                                SparseLDLResult* result);
#endif  // ifdef CATAMARI_OPENMP

  bool RightLookingSubtree(Int supernode, const CoordinateMatrix<Field>& matrix,
                           RightLookingSharedState<Field>* shared_state,
                           PrivateState<Field>* private_state,
                           SparseLDLResult* result);
#ifdef CATAMARI_OPENMP
  bool OpenMPRightLookingSubtree(Int supernode,
                                 const CoordinateMatrix<Field>& matrix,
                                 const Buffer<double>& work_estimates,
                                 double min_parallel_work,
                                 RightLookingSharedState<Field>* shared_state,
                                 Buffer<PrivateState<Field>>* private_states,
                                 SparseLDLResult* result);
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

  bool LeftLookingSupernodeFinalize(Int main_supernode,
                                    SparseLDLResult* result);
#ifdef CATAMARI_OPENMP
  bool OpenMPLeftLookingSupernodeFinalize(
      Int supernode, Buffer<PrivateState<Field>>* private_states,
      SparseLDLResult* result);
#endif  // ifdef CATAMARI_OPENMP

  bool RightLookingSupernodeFinalize(
      Int supernode, RightLookingSharedState<Field>* shared_state,
      PrivateState<Field>* private_state, SparseLDLResult* result);
#ifdef CATAMARI_OPENMP
  bool OpenMPRightLookingSupernodeFinalize(
      Int supernode, RightLookingSharedState<Field>* shared_state,
      Buffer<PrivateState<Field>>* private_state, SparseLDLResult* result);
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

#include "catamari/sparse_ldl/supernodal/factorization/common-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/common_openmp-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/io-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/left_looking-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/left_looking_openmp-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/right_looking-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/right_looking_openmp-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/solve-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/solve_openmp-impl.hpp"

#endif  // ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_H_
