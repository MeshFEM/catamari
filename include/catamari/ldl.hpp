/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_H_
#define CATAMARI_LDL_H_

#include <limits>
#include <memory>

#include "catamari/ldl/scalar_ldl.hpp"
#include "catamari/ldl/supernodal_ldl.hpp"
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
struct LDLControl {
  quotient::MinimumDegreeControl md_control;

  // Whether or not a supernodal factorization should be used.
  SupernodalStrategy supernodal_strategy = kAdaptiveSupernodalStrategy;

  // The configuration options for the scalar LDL factorization.
  scalar_ldl::Control scalar_control;

  // The configuration options for the supernodal LDL factorization.
  supernodal_ldl::Control supernodal_control;

  // Sets the factorization type for both the scalar and supernodal control
  // structures.
  void SetFactorizationType(SymmetricFactorizationType type) {
    scalar_control.factorization_type = type;
    supernodal_control.factorization_type = type;
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

  // Whether
  bool verbose = false;
};

// A wrapper for the scalar and supernodal factorization data structures.
template <class Field>
class LDLFactorization {
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

  // Performs the factorization using an automatically determined ordering.
  LDLResult Factor(const CoordinateMatrix<Field>& matrix,
                   const LDLControl& control);

  // Performs the factorization using a prescribed ordering.
  LDLResult Factor(const CoordinateMatrix<Field>& matrix,
                   const SymmetricOrdering& ordering,
                   const LDLControl& control);

  // Solves a set of linear systems using the factorization.
  void Solve(BlasMatrixView<Field>* right_hand_sides) const;

  // Solves a set of linear systems using iterative refinement.
  Int RefinedSolve(const CoordinateMatrix<Field>& matrix,
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
};

}  // namespace catamari

#include "catamari/ldl-impl.hpp"

#endif  // ifndef CATAMARI_LDL_H_
