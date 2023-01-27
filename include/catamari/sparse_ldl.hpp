/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_H_
#define CATAMARI_SPARSE_LDL_H_

#include <limits>
#include <memory>
#include <stdexcept>

#include "catamari/sparse_ldl/scalar.hpp"
#include "catamari/sparse_ldl/supernodal.hpp"
#include "quotient/minimum_degree.hpp"

namespace catamari {

enum SupernodalStrategy {
  // Use a scalar factorization.
  kScalarFactorization,

  // Use a supernodal factorization.
  kSupernodalFactorization,

  // Adaptively choose between scalar and supernodal factorization.
  kAdaptiveSupernodalStrategy,
};

// Configuration options for LDL' factorization.
template <typename Field>
struct SparseLDLControl {
  // The configuration options for the Minimum Degree reordering.
  quotient::MinimumDegreeControl md_control;

  // Whether or not a supernodal factorization should be used.
  SupernodalStrategy supernodal_strategy = kAdaptiveSupernodalStrategy;

  // The minimum number of factorization flops before a supernodal approach will
  // be selected by the kAdaptiveSupernodalStrategy approach.
  double supernodal_flop_threshold = 1e5;

  // The number of flops per factorization nonzero before a supernodal approach
  // will be selected by the kAdaptiveSupernodalStrategy approach.
  double supernodal_intensity_threshold = 40;

  // The configuration options for the scalar LDL factorization.
  scalar_ldl::Control<Field> scalar_control;

  // The configuration options for the supernodal LDL factorization.
  supernodal_ldl::Control<Field> supernodal_control;

  // If Ruiz equilibration should be used to (hopefully) increase the accuracy
  // of the factorization and solve.
  bool equilibrate = false;

  // If the high-level logic should print progress information.
  bool verbose = false;

  // Sets the factorization type for both the scalar and supernodal control
  // structures.
  void SetFactorizationType(SymmetricFactorizationType type) {
    scalar_control.factorization_type = type;
    supernodal_control.factorization_type = type;
  }

  // Sets the dynamic regularization control parameters in both the scalar and
  // supernodal control structures to the given settings.
  void SetDynamicRegularization(
      const DynamicRegularizationControl<Field>& dynamic_regularization) {
    scalar_control.dynamic_regularization = dynamic_regularization;
    supernodal_control.dynamic_regularization = dynamic_regularization;
  }
};

// Configuration options for solving linear systems with iterative refinement.
template <typename Real>
struct RefinedSolveControl {
  // The desired relative error (in the max norm) of the solution. Iteration
  // stops early if this tolerance is achieved.
  Real relative_tol = 10 * std::numeric_limits<Real>::epsilon();

  // The maximum number of iterations of iterative refinement to perform.
  Int max_iters = 3;

  // Whether (typically) higher-precision arithmetic should be used for the
  // iterative refinement.
  bool promote = false;

  // Whether convergence progress information should be printed.
  bool verbose = false;
};

// The return value of iterative refinement.
template <typename Real>
struct RefinedSolveStatus {
  // The number of performed refinement iterations.
  Int num_iterations;

  // The maximum of the relative max norms of the residual matrix, i.e.,
  //
  //   max_j || b_j - A x_j ||_{max} / || b_j ||_{max},
  //
  // where we replace any division by zero with division by one. That is, we
  // use absolute residual norms for any zero right-hand sides.
  Real residual_relative_max_norm;
};

// A wrapper for the scalar and supernodal factorization data structures.
template <class Field>
class SparseLDL {
 public:
  // The underlying real datatype of the scalar type.
  typedef ComplexBase<Field> Real;

  // Whether or not a supernodal factorization was used. If it is true, only
  // 'supernodal_factorization' should be non-null, and vice versa.
  bool is_supernodal;

  // The scalar LDL factorization data structure.
  std::unique_ptr<scalar_ldl::Factorization<Field>> scalar_factorization;

  // The supernodal LDL factorization data structure.
  std::unique_ptr<supernodal_ldl::Factorization<Field>>
      supernodal_factorization;

  // The default constructor.
  SparseLDL();

  // Performs the factorization using an automatically determined ordering.
  SparseLDLResult<Field> Factor(const CoordinateMatrix<Field>& matrix,
                                const SparseLDLControl<Field>& control,
                                bool symbolic_only = false);

  // Performs the factorization using a prescribed ordering.
  SparseLDLResult<Field> Factor(const CoordinateMatrix<Field>& matrix,
                                const SymmetricOrdering& ordering,
                                const SparseLDLControl<Field>& control,
                                bool symbolic_only = false);

  // Returns the diagonal perturbation -- in the original ordering -- given
  // the list of diagonal dynamic regularization permutations in the
  // factorization ordering.
  void DynamicRegularizationDiagonal(const SparseLDLResult<Field>& result,
                                     BlasMatrix<Real>* diagonal) const;

  // Factors a new matrix with the same sparsity pattern as a previous
  // factorization.
  SparseLDLResult<Field> RefactorWithFixedSparsityPattern(
      const CoordinateMatrix<Field>& matrix);

  // Factors a new matrix with the same sparsity pattern as a previous
  // factorization -- the control structure is allowed to change in minor ways,
  // for example, in its dynamic regularization choices.
  SparseLDLResult<Field> RefactorWithFixedSparsityPattern(
      const CoordinateMatrix<Field>& matrix,
      const SparseLDLControl<Field>& control);

  SparseLDLResult<Field> RefactorWithFixedSparsityPattern(const ConversionPlan &cplan, const Field *Ax, Field sigma = 0, const Field *Bx = nullptr) {
      if (is_supernodal) {
        return supernodal_factorization->RefactorWithFixedSparsityPattern(cplan, Ax, sigma, Bx);
      }
      throw std::runtime_error("Implemented for supernodal only");
  }

  // Returns the number of rows of the last factored matrix.
  Int NumRows() const;

  // Solves a set of linear systems using the factorization.
  void Solve(BlasMatrixView<Field>* right_hand_sides, bool already_permuted = false) const;

  // Solves a set of linear systems using iterative refinement.
  RefinedSolveStatus<Real> RefinedSolve(
      const CoordinateMatrix<Field>& matrix,
      const RefinedSolveControl<Real>& control,
      BlasMatrixView<Field>* right_hand_sides) const;

  // Solve with iterative refinement and a diagonal scaling:
  //
  //     (D A D) (inv(D) x) = (D b).
  //
  // This can be useful in situations where the right-hand side consists of
  // several subgroups, and the relative accuracy of each subgroup is desired
  // to be controlled. One can thus construct the diagonal matrix D to be
  // piecewise constant, with each piece being set to the inverse of the norm
  // of the subgroup of the right-hand side vector.
  RefinedSolveStatus<ComplexBase<Field>> DiagonallyScaledRefinedSolve(
      const CoordinateMatrix<Field>& matrix,
      const ConstBlasMatrixView<Real>& scaling,
      const RefinedSolveControl<Real>& control,
      BlasMatrixView<Field>* right_hand_sides) const;

  // Solves a set of linear systems using iterative refinement in the presence
  // of dynamic regularization.
  RefinedSolveStatus<Real> DynamicallyRegularizedRefinedSolve(
      const CoordinateMatrix<Field>& matrix,
      const SparseLDLResult<Field>& result,
      const RefinedSolveControl<Real>& control,
      BlasMatrixView<Field>* right_hand_sides) const;

  // Solve with iterative refinement and a diagonal scaling in the presence of
  // dynamic regularization:
  //
  //     (D (A + regularization) D) (inv(D) x) = (D b).
  //
  // This can be useful in situations where the right-hand side consists of
  // several subgroups, and the relative accuracy of each subgroup is desired
  // to be controlled. One can thus construct the diagonal matrix D to be
  // piecewise constant, with each piece being set to the inverse of the norm
  // of the subgroup of the right-hand side vector.
  RefinedSolveStatus<ComplexBase<Field>>
  DiagonallyScaledDynamicallyRegularizedRefinedSolve(
      const CoordinateMatrix<Field>& matrix,
      const SparseLDLResult<Field>& result,
      const ConstBlasMatrixView<Real>& scaling,
      const RefinedSolveControl<Real>& control,
      BlasMatrixView<Field>* right_hand_sides) const;

  // Solves a set of linear systems using the lower-triangular factor.
  void LowerTriangularSolve(BlasMatrixView<Field>* right_hand_sides) const;

  // Solves a set of linear systems using the diagonal factor.
  void DiagonalSolve(BlasMatrixView<Field>* right_hand_sides) const;

  // Solves a set of linear systems using the transpose (or adjoint) of the
  // lower-triangular matrix.
  void LowerTransposeTriangularSolve(
      BlasMatrixView<Field>* right_hand_sides) const;

  // Prints the lower-triangular factor of the factorization.
  void PrintLowerFactor(const std::string& label, std::ostream& os) const;

  // Prints the diagonal factor of the factorization.
  void PrintDiagonalFactor(const std::string& label, std::ostream& os) const;

  // Returns an immutable reference to the permutation mapping from the original
  // indices into the factorization's.
  const Buffer<Int>& Permutation() const;

  // Returns an immutable reference to the permutation mapping from the
  // factorization indices back to the original matrix ones.
  const Buffer<Int>& InversePermutation() const;

 private:
  // An (optional) diagonal scaling meant to improve accuracy.
  bool have_equilibration_;
  BlasMatrix<Real> equilibration_;

  // Solves a set of linear systems using iterative refinement.
  RefinedSolveStatus<Real> RefinedSolveHelper(
      const CoordinateMatrix<Field>& matrix,
      const RefinedSolveControl<Real>& control,
      BlasMatrixView<Field>* right_hand_sides) const;

  // Solves a set of linear systems using iterative refinement with
  // higher-precision forward multiplies.
  RefinedSolveStatus<Real> PromotedRefinedSolveHelper(
      const CoordinateMatrix<Field>& matrix,
      const RefinedSolveControl<Real>& control,
      BlasMatrixView<Field>* right_hand_sides_lower) const;

  // Solve with iterative refinement and a diagonal scaling:
  //
  //     (D A D) (inv(D) x) = (D b).
  //
  // This can be useful in situations where the right-hand side consists of
  // several subgroups, and the relative accuracy of each subgroup is desired
  // to be controlled. One can thus construct the diagonal matrix D to be
  // piecewise constant, with each piece being set to the inverse of the norm
  // of the subgroup of the right-hand side vector.
  RefinedSolveStatus<ComplexBase<Field>> DiagonallyScaledRefinedSolveHelper(
      const CoordinateMatrix<Field>& matrix,
      const ConstBlasMatrixView<Real>& scaling,
      const RefinedSolveControl<Real>& control,
      BlasMatrixView<Field>* right_hand_sides) const;

  // Uses a higher-precision to solve with iterative refinement and a diagonal
  // scaling:
  //
  //     (D A D) (inv(D) x) = (D b).
  //
  // This can be useful in situations where the right-hand side consists of
  // several subgroups, and the relative accuracy of each subgroup is desired
  // to be controlled. One can thus construct the diagonal matrix D to be
  // piecewise constant, with each piece being set to the inverse of the norm
  // of the subgroup of the right-hand side vector.
  RefinedSolveStatus<ComplexBase<Field>>
  PromotedDiagonallyScaledRefinedSolveHelper(
      const CoordinateMatrix<Field>& matrix,
      const ConstBlasMatrixView<Real>& scaling,
      const RefinedSolveControl<Real>& control,
      BlasMatrixView<Field>* right_hand_sides_lower) const;

  // Solves a set of linear systems using iterative refinement in the presence
  // of dynamic regularization.
  RefinedSolveStatus<Real> DynamicallyRegularizedRefinedSolveHelper(
      const CoordinateMatrix<Field>& matrix,
      const SparseLDLResult<Field>& result,
      const RefinedSolveControl<Real>& control,
      BlasMatrixView<Field>* right_hand_sides) const;

  // Solves a set of linear systems using iterative refinement in the presence
  // of dynamic regularization.
  RefinedSolveStatus<Real> PromotedDynamicallyRegularizedRefinedSolveHelper(
      const CoordinateMatrix<Field>& matrix,
      const SparseLDLResult<Field>& result,
      const RefinedSolveControl<Real>& control,
      BlasMatrixView<Field>* right_hand_sides_lower) const;

  // Solve with iterative refinement and a diagonal scaling in the presence of
  // dynamic regularization:
  //
  //     (D (A + regularization) D) (inv(D) x) = (D b).
  //
  // This can be useful in situations where the right-hand side consists of
  // several subgroups, and the relative accuracy of each subgroup is desired
  // to be controlled. One can thus construct the diagonal matrix D to be
  // piecewise constant, with each piece being set to the inverse of the norm
  // of the subgroup of the right-hand side vector.
  RefinedSolveStatus<ComplexBase<Field>>
  DiagonallyScaledDynamicallyRegularizedRefinedSolveHelper(
      const CoordinateMatrix<Field>& matrix,
      const SparseLDLResult<Field>& result,
      const ConstBlasMatrixView<Real>& scaling,
      const RefinedSolveControl<Real>& control,
      BlasMatrixView<Field>* right_hand_sides) const;

  // Solve with higher-precision iterative refinement and a diagonal scaling in
  // the presence of dynamic regularization:
  //
  //     (D (A + regularization) D) (inv(D) x) = (D b).
  //
  // This can be useful in situations where the right-hand side consists of
  // several subgroups, and the relative accuracy of each subgroup is desired
  // to be controlled. One can thus construct the diagonal matrix D to be
  // piecewise constant, with each piece being set to the inverse of the norm
  // of the subgroup of the right-hand side vector.
  RefinedSolveStatus<ComplexBase<Field>>
  PromotedDiagonallyScaledDynamicallyRegularizedRefinedSolveHelper(
      const CoordinateMatrix<Field>& matrix,
      const SparseLDLResult<Field>& result,
      const ConstBlasMatrixView<Real>& scaling,
      const RefinedSolveControl<Real>& control,
      BlasMatrixView<Field>* right_hand_sides_lower) const;
};

// Append the children's dynamic regularizations.
template <class Field>
void MergeDynamicRegularizations(
    const Buffer<SparseLDLResult<Field>>& children_results,
    SparseLDLResult<Field>* result);

}  // namespace catamari

#include "catamari/sparse_ldl-impl.hpp"

#endif  // ifndef CATAMARI_SPARSE_LDL_H_
