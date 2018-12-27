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

#include "catamari/supernodal_ldl.hpp"
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

// A wrapper for the scalar and supernodal factorization data structures.
template <class Field>
struct LDLFactorization {
  // Whether or not a supernodal factorization was used. If it is true, only
  // 'supernodal_factorization' should be non-null, and vice versa.
  bool is_supernodal;

  // The scalar LDL factorization data structure.
  std::unique_ptr<ScalarLDLFactorization<Field>> scalar_factorization;

  // The supernodal LDL factorization data structure.
  std::unique_ptr<supernodal_ldl::Factorization<Field>>
      supernodal_factorization;
};

// Configuration options for LDL' factorization.
struct LDLControl {
  quotient::MinimumDegreeControl md_control;

  // Whether or not a supernodal factorization should be used.
  SupernodalStrategy supernodal_strategy = kAdaptiveSupernodalStrategy;

  // The configuration options for the scalar LDL factorization.
  ScalarLDLControl scalar_control;

  // The configuration options for the supernodal LDL factorization.
  supernodal_ldl::Control supernodal_control;
};

// Performs an LDL' factorization in the minimum-degree ordering.
template <class Field>
LDLResult LDL(const CoordinateMatrix<Field>& matrix, const LDLControl& control,
              LDLFactorization<Field>* factorization);

// Performs an LDL' factorization in a permuted ordering.
template <class Field>
LDLResult LDL(const CoordinateMatrix<Field>& matrix,
              const std::vector<Int>& permutation,
              const std::vector<Int>& inverse_permutation,
              const LDLControl& control,
              LDLFactorization<Field>* factorization);

// Solve A x = b via the substitution (L D L') x = b and the sequence:
//   x := L' \ (D \ (L \ b)).
template <class Field>
void LDLSolve(const LDLFactorization<Field>& factorization,
              BlasMatrix<Field>* matrix);

// Solves L x = b using a lower triangular matrix L.
template <class Field>
void LowerTriangularSolve(const LDLFactorization<Field>& factorization,
                          BlasMatrix<Field>* matrix);

// Solves D x = b using a diagonal matrix D.
template <class Field>
void DiagonalSolve(const LDLFactorization<Field>& factorization,
                   BlasMatrix<Field>* matrix);

// Solves L' x = b using a lower triangular matrix L.
template <class Field>
void LowerTransposeTriangularSolve(const LDLFactorization<Field>& factorization,
                                   BlasMatrix<Field>* matrix);

// Prints the unit-diagonal lower-triangular factor of the LDL' factorization.
template <class Field>
void PrintLowerFactor(const LDLFactorization<Field>& factorization,
                      const std::string& label, std::ostream& os);

// Prints the diagonal factor of the LDL' factorization.
template <class Field>
void PrintDiagonalFactor(const LDLFactorization<Field>& factorization,
                         const std::string& label, std::ostream& os);

}  // namespace catamari

#include "catamari/ldl-impl.hpp"

#endif  // ifndef CATAMARI_LDL_H_
