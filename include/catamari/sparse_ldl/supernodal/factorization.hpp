/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_H_

#include "catamari/blas_matrix.hpp"
#include "catamari/buffer.hpp"
#include "catamari/sparse_ldl/supernodal/diagonal_factor.hpp"
#include "catamari/sparse_ldl/supernodal/lower_factor.hpp"
#include "catamari/sparse_ldl/supernodal/supernode_utils.hpp"
#include <tbb/task_group.h>
#include <stdexcept>

#ifdef CATAMARI_ENABLE_TIMERS
#include "quotient/timer.hpp"
#endif  // ifdef CATAMARI_ENABLE_TIMERS

#include <Eigen/Dense>

#define LOAD_MATRIX_OUTSIDE 1

namespace catamari {

template<class Field>
auto eigenMap(BlasMatrixView<Field> &bm) {
    if (bm.leading_dim != bm.height) throw std::runtime_error("fail!");
    return Eigen::Map<Eigen::Matrix<Field, Eigen::Dynamic, Eigen::Dynamic>>(bm.Data(), bm.height, bm.width);
}

template<class Field>
auto eigenMap(Buffer<Field> &b) {
    return Eigen::Map<Eigen::Matrix<Field, Eigen::Dynamic, 1>>(b.Data(), b.Size());
}

template<class Field>
auto eigenMap(const ConstBlasMatrixView<Field> &bm) {
    if (bm.leading_dim != bm.height) throw std::runtime_error("fail!");
    return Eigen::Map<const Eigen::Matrix<Field, Eigen::Dynamic, Eigen::Dynamic>>(bm.Data(), bm.height, bm.width);
}
template<class Field>

auto eigenMap(Field *ptr, int size) {
    return Eigen::Map<Eigen::Array<Field, Eigen::Dynamic, 1>>(ptr, size);
}

namespace supernodal_ldl {

// Configuration options for supernodal LDL' factorization.
template <typename Field>
struct Control {
  // Determines the style of the factorization.
  SymmetricFactorizationType factorization_type = kLDLAdjointFactorization;

  // Configuration for the supernodal relaxation.
  SupernodalRelaxationControl relaxation_control;

  // The choice of either left-looking or right-looking LDL' factorization.
  // There is currently no supernodal up-looking support.
  LDLAlgorithm algorithm = kAdaptiveLDL;

  // Whether pivoting within each supernodal diagonal block should be enabled.
  bool supernodal_pivoting = false;

  // The amount of dynamic regularization -- if any -- to use.
  DynamicRegularizationControl<Field> dynamic_regularization;

  // The minimal supernode size for an out-of-place trapezoidal solve to be
  // used.
  Int forward_solve_out_of_place_supernode_threshold = 20;

  // The minimal supernode size for an out-of-place trapezoidal solve to be
  // used.
  Int backward_solve_out_of_place_supernode_threshold = 8;

  // The algorithmic block size for the factorization.
  Int block_size = 64;

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

  quotient::Timer gemm_unpack;

  quotient::Timer herk;
  double herk_gflops = 0;

  quotient::Timer herk_unpack;

  quotient::Timer trsm;
  double trsm_gflops = 0;

  quotient::Timer cholesky;
  double cholesky_gflops = 0;

  quotient::Timer merge;

  quotient::Timer left_looking;
  quotient::Timer left_looking_allocate;
  quotient::Timer left_looking_update;
  quotient::Timer left_looking_finalize;

  FactorizationProfile()
      : scalar_elimination_forest("scalar_elimination_forest"),
        supernodal_elimination_forest("supernodal_elimination_forest"),
        relax_supernodes("relax_supernodes"),
        initialize_factors("initialize_factors"),
        gemm("gemm"),
        gemm_unpack("gemm_unpack"),
        herk("herk"),
        herk_unpack("herk_unpack"),
        trsm("trsm"),
        cholesky("cholesky"),
        merge("merge"),
        left_looking("left_looking"),
        left_looking_allocate("left_looking_allocate"),
        left_looking_update("left_looking_update"),
        left_looking_finalize("left_looking_finalize") {}

  void Reset() {
    scalar_elimination_forest.Reset(scalar_elimination_forest.Name());
    supernodal_elimination_forest.Reset(supernodal_elimination_forest.Name());
    relax_supernodes.Reset(relax_supernodes.Name());
    initialize_factors.Reset(initialize_factors.Name());
    gemm.Reset(gemm.Name());
    gemm_gflops = 0;
    gemm_unpack.Reset(gemm_unpack.Name());
    herk.Reset(herk.Name());
    herk_gflops = 0;
    herk_unpack.Reset(herk_unpack.Name());
    trsm.Reset(trsm.Name());
    trsm_gflops = 0;
    cholesky.Reset(cholesky.Name());
    cholesky_gflops = 0;
    merge.Reset(merge.Name());
    left_looking.Reset(left_looking.Name());
    left_looking_allocate.Reset(left_looking_allocate.Name());
    left_looking_update.Reset(left_looking_update.Name());
    left_looking_finalize.Reset(left_looking_finalize.Name());
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
     << profile.gemm_unpack << "\n"
     << profile.herk << " (GFlops: " << profile.herk_gflops
     << ", GFlop/sec: " << profile.herk_gflops / profile.herk.TotalSeconds()
     << ")\n"
     << profile.herk_unpack << "\n"
     << profile.trsm << " (GFlops: " << profile.trsm_gflops
     << ", GFlop/sec: " << profile.trsm_gflops / profile.trsm.TotalSeconds()
     << ")\n"
     << profile.cholesky << " (GFlops: " << profile.cholesky_gflops
     << ", GFlop/sec: "
     << profile.cholesky_gflops / profile.cholesky.TotalSeconds() << ")\n";
  if (profile.left_looking.TotalSeconds() > 0.) {
    os << profile.left_looking << "\n"
       << profile.left_looking_allocate << "\n"
       << profile.left_looking_update << "\n"
       << profile.left_looking_finalize << std::endl;
  }
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
  SparseLDLResult<Field> Factor(const CoordinateMatrix<Field>& matrix,
                                const SymmetricOrdering& manual_ordering,
                                const Control<Field>& control);

  // Factors the given matrix after having previously factored another matrix
  // with the same sparsity pattern.
  SparseLDLResult<Field> RefactorWithFixedSparsityPattern(
      const CoordinateMatrix<Field>& matrix);

  // Factors the given matrix after having previously factored another matrix
  // with the same sparsity pattern -- the control structure is allowed to
  // change in minor ways.
  SparseLDLResult<Field> RefactorWithFixedSparsityPattern(
      const CoordinateMatrix<Field>& matrix, const Control<Field>& control);

  // Returns the number of rows in the last factored matrix.
  Int NumRows() const;

  // Solve a set of linear systems using the factorization.
  void Solve(BlasMatrixView<Field>* right_hand_sides, bool already_permuted = false) const;

  // Solves a set of linear systems using the lower-triangular factor.
  void LowerTriangularSolve(BlasMatrixView<Field>* right_hand_sides) const;

  void OpenMPLowerTriangularSolve(
      BlasMatrixView<Field>* right_hand_sides,
      RightLookingSharedState<Field>* shared_state) const;

  // Solves a set of linear systems using the diagonal factor.
  void DiagonalSolve(BlasMatrixView<Field>* right_hand_sides) const;

  void OpenMPDiagonalSolve(BlasMatrixView<Field>* right_hand_sides) const;

  // Solves a set of linear systems using the trasnpose (or adjoint) of the
  // lower-triangular factor.
  void LowerTransposeTriangularSolve(
      BlasMatrixView<Field>* right_hand_sides) const;

  void OpenMPLowerTransposeTriangularSolve(
      BlasMatrixView<Field>* right_hand_sides,
      RightLookingSharedState<Field>* shared_state) const;

  // Prints the diagonal of the factorization.
  void PrintDiagonalFactor(const std::string& label, std::ostream& os) const;

  // Prints the unit lower-triangular matrix.
  void PrintLowerFactor(const std::string& label, std::ostream& os) const;

  // Returns a view of the given supernode's permutation vector.
  // NOTE: This is only valid when control.supernodal_pivoting is true.
  BlasMatrixView<Int> SupernodePermutation(Int supernode);

  // Returns a const view of the given supernode's permutation vector.
  // NOTE: This is only valid when control.supernodal_pivoting is true.
  ConstBlasMatrixView<Int> SupernodePermutation(Int supernode) const;

  // Returns an immutable reference to the permutation mapping the original
  // matrix indices into those used for the factorization.
  const Buffer<Int>& Permutation() const;

  // Returns an immutable reference to the permutation mapping the factorization
  // indices into those of the original matrix.
  const Buffer<Int>& InversePermutation() const;

  // Incorporates the details and work required to process the supernode with
  // the given size and degree into the factorization result.
  static void IncorporateSupernodeIntoLDLResult(Int supernode_size, Int degree,
                                                SparseLDLResult<Field>* result);

  // Adds in the contribution of a subtree into an overall result.
  static void MergeContribution(const SparseLDLResult<Field>& contribution,
                                SparseLDLResult<Field>* result);

 private:
  // The control structure for the factorization.
  Control<Field> control_;

  // The representation of the permutation matrix P so that P A P' should be
  // factored. Typically, this permutation is the composition of a
  // fill-reducing ordering and a supernodal relaxation permutation.
public:
  SymmetricOrdering ordering_;
private:

  // An array of length 'num_rows'; the i'th member is the index of the
  // supernode containing column 'i'.
  Buffer<Int> supernode_member_to_index_;

  // The largest degree of a supernode in the factorization.
  Int max_degree_;

  // The largest number of entries in any supernode's lower block.
  Int max_lower_block_size_;

  // The maximum workspace needed for a diagonal or subdiagonal update of a
  // block column of a supernode.
  Int left_looking_workspace_size_;

  // The maximum workspace needed for storing the scaled transpose of an
  // intersection of a block column with an ancestor supernode.
  // This is only nonzero for LDL^T and LDL^H factorizations.
  Int left_looking_scaled_transpose_size_;

public:
  // The subdiagonal-block portion of the lower-triangular factor.
  std::unique_ptr<LowerFactor<Field>> lower_factor_;

  // The block-diagonal factor.
  std::unique_ptr<DiagonalFactor<Field>> diagonal_factor_;
private:

  // If supernodal_pivoting is enabled, all of the supernode permutation
  // vectors are stored within this single buffer.
  BlasMatrix<Int> supernode_permutations_;

  // Julian Panetta: cache work estimates
  Buffer<double> work_estimates_;
  double total_work_;
  mutable Buffer<Field> permute_scratch_;
  mutable RightLookingSharedState<Field> solve_shared_state_;

  // Julian Panetta: cache right-looking shared state
  RightLookingSharedState<Field> shared_state_;

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
                      Buffer<Int>* supernode_degrees);
#ifdef CATAMARI_OPENMP
  void OpenMPFormSupernodes(const CoordinateMatrix<Field>& matrix,
                            Buffer<Int>* supernode_degrees);
#endif  // ifdef CATAMARI_OPENMP

  // Initializes a supernodal block column of the factorization using the
  // input matrix.
public:
  void InitializeBlockColumn(Int supernode,
                             const CoordinateMatrix<Field>& matrix);
private:
#ifdef CATAMARI_OPENMP
  void OpenMPInitializeBlockColumn(Int supernode,
                                   const CoordinateMatrix<Field>& matrix);
#endif  // ifdef CATAMARI_OPENMP

  void InitializeFactors(const CoordinateMatrix<Field>& matrix,
                         const Buffer<Int>& supernode_degrees);
#ifdef CATAMARI_OPENMP
  void OpenMPInitializeFactors(const CoordinateMatrix<Field>& matrix,
                               const Buffer<Int>& supernode_degrees);
#endif  // ifdef CATAMARI_OPENMP

  SparseLDLResult<Field> LeftLooking(const CoordinateMatrix<Field>& matrix);

  SparseLDLResult<Field> RightLooking(const CoordinateMatrix<Field>& matrix);
  SparseLDLResult<Field> OpenMPRightLooking(
      const CoordinateMatrix<Field>& matrix);

  bool LeftLookingSubtree(
      Int supernode, const CoordinateMatrix<Field>& matrix,
      const DynamicRegularizationParams<Field>& dynamic_reg_params,
      LeftLookingSharedState* shared_state, PrivateState<Field>* private_state,
      SparseLDLResult<Field>* result);

  bool RightLookingSubtree(
      Int supernode, const CoordinateMatrix<Field>& matrix,
      const DynamicRegularizationParams<Field>& dynamic_reg_params,
      RightLookingSharedState<Field>* shared_state,
      PrivateState<Field>* private_state, SparseLDLResult<Field>* result);
  bool OpenMPRightLookingSubtree(
      Int supernode, const CoordinateMatrix<Field>& matrix,
      const DynamicRegularizationParams<Field>& dynamic_reg_params,
      const Buffer<double>& work_estimates, double min_parallel_work,
      RightLookingSharedState<Field>* shared_state,
      Buffer<PrivateState<Field>>* private_states,
      SparseLDLResult<Field>* result);

  void LeftLookingSupernodeUpdate(Int main_supernode,
                                  const CoordinateMatrix<Field>& matrix,
                                  LeftLookingSharedState* shared_state,
                                  PrivateState<Field>* private_state);

  bool LeftLookingSupernodeFinalize(
      Int main_supernode,
      const DynamicRegularizationParams<Field>& dynamic_reg_params,
      SparseLDLResult<Field>* result);

  bool RightLookingSupernodeFinalize(
      Int supernode,
      const DynamicRegularizationParams<Field>& dynamic_reg_params,
      RightLookingSharedState<Field>* shared_state,
      PrivateState<Field>* private_state, SparseLDLResult<Field>* result);
  bool OpenMPRightLookingSupernodeFinalize(
      Int supernode,
      const DynamicRegularizationParams<Field>& dynamic_reg_params,
      RightLookingSharedState<Field>* shared_state,
      Buffer<PrivateState<Field>>* private_state,
      SparseLDLResult<Field>* result);

  // Performs the portion of the lower-triangular solve corresponding to the
  // subtree with the given root supernode.
  void LowerTriangularSolveRecursion(Int supernode,
                                     BlasMatrixView<Field>* right_hand_sides,
                                     Buffer<Field>* workspace) const;
  void OpenMPLowerTriangularSolveRecursion(
      Int supernode, BlasMatrixView<Field>* right_hand_sides,
      RightLookingSharedState<Field>* shared_state, int level) const;

  // Performs the trapezoidal solve associated with a particular supernode.
  void LowerSupernodalTrapezoidalSolve(Int supernode,
                                       BlasMatrixView<Field>* right_hand_sides,
                                       Buffer<Field>* workspace) const;
  void OpenMPLowerSupernodalTrapezoidalSolve(
      Int supernode, BlasMatrixView<Field>* right_hand_sides,
      BlasMatrixView<Field> *supernode_schur_complement) const;

  // Performs the portion of the transposed lower-triangular solve
  // corresponding to the subtree with the given root supernode.
  void LowerTransposeTriangularSolveRecursion(
      Int supernode, BlasMatrixView<Field>* right_hand_sides,
      Buffer<Field>* packed_input_buf) const;
  void OpenMPLowerTransposeTriangularSolveRecursion(
      Int supernode, BlasMatrixView<Field>* right_hand_sides,
      RightLookingSharedState<Field>* shared_state, int level, tbb::task_group &tg) const;

  // Performs the trapezoidal solve associated with a particular supernode.
  void LowerTransposeSupernodalTrapezoidalSolve(
      Int supernode, BlasMatrixView<Field>* right_hand_sides,
      Buffer<Field>* workspace) const;

  void LowerTransposeSupernodalTrapezoidalSolve(
      Int supernode, BlasMatrixView<Field>* right_hand_sides,
      BlasMatrixView<Field> &work_right_hand_sides) const;
};

}  // namespace supernodal_ldl
}  // namespace catamari

#include "catamari/sparse_ldl/supernodal/factorization/common-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/common_openmp-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/io-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/left_looking-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/right_looking-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/right_looking_openmp-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/solve-impl.hpp"
#include "catamari/sparse_ldl/supernodal/factorization/solve_openmp-impl.hpp"

#endif  // ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_H_
