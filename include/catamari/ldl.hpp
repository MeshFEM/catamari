/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_H_
#define CATAMARI_LDL_H_

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

// A wrapper for the scalar and supernodal factorization data structures.
template <class Field>
class LDLFactorization {
 public:
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
  void Solve(BlasMatrixView<Field>* matrix) const;

  // Solves a set of linear systems using the lower-triangular factor.
  void LowerTriangularSolve(BlasMatrixView<Field>* matrix) const;

  // Solves a set of linear systems using the diagonal factor.
  void DiagonalSolve(BlasMatrixView<Field>* matrix) const;

  // Solves a set of linear systems using the transpose (or adjoint) of the
  // lower-triangular matrix.
  void LowerTransposeTriangularSolve(BlasMatrixView<Field>* matrix) const;

  // Prints the lower-triangular factor of the factorization.
  void PrintLowerFactor(const std::string& label, std::ostream& os) const;

  // Prints the diagonal factor of the factorization.
  void PrintDiagonalFactor(const std::string& label, std::ostream& os) const;
};

}  // namespace catamari

#include "catamari/ldl-impl.hpp"

#endif  // ifndef CATAMARI_LDL_H_
