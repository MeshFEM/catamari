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
//
// TODO(Jack Poulson): Decide how to support the traditional option of using
// a supernodal LDL for sufficiently dense factorizations and a scalar
// up-looking approach otherwise.
template <class Field>
struct LDLFactorization {
  // Whether or not a supernodal factorization was used. If it is true, only
  // 'supernodal_factorization' should be non-null, and vice versa.
  bool is_supernodal;

  // The scalar LDL factorization data structure.
  std::unique_ptr<ScalarLDLFactorization<Field>> scalar_factorization;

  // The supernodal LDL factorization data structure.
  std::unique_ptr<SupernodalLDLFactorization<Field>> supernodal_factorization;
};

// Configuration options for LDL' factorization.
struct LDLControl {
  // Whether or not a supernodal factorization should be used.
  SupernodalStrategy supernodal_strategy = kAdaptiveSupernodalStrategy;

  // The configuration options for the scalar LDL factorization.
  ScalarLDLControl scalar_control;

  // The configuration options for the supernodal LDL factorization.
  SupernodalLDLControl supernodal_control;
};

// Performs an LDL' factorization in the natural ordering.
template <class Field>
Int LDL(
    const CoordinateMatrix<Field>& matrix, const LDLControl& control,
    LDLFactorization<Field>* factorization);

// Solve A x = b via the substitution (L D L') x = b and the sequence:
//   x := L' \ (D \ (L \ b)).
template <class Field>
void LDLSolve(
    const LDLFactorization<Field>& factorization, std::vector<Field>* vector);

// Solves L x = b using a lower triangular matrix L.
template <class Field>
void LowerTriangularSolve(
    const LDLFactorization<Field>& factorization, std::vector<Field>* vector);

// Solves D x = b using a diagonal matrix D.
template <class Field>
void DiagonalSolve(
    const LDLFactorization<Field>& factorization, std::vector<Field>* vector);

// Solves L' x = b using a lower triangular matrix L.
template <class Field>
void LowerAdjointTriangularSolve(
    const LDLFactorization<Field>& factorization, std::vector<Field>* vector);

// Prints the unit-diagonal lower-triangular factor of the LDL' factorization.
template <class Field>
void PrintLowerFactor(
    const LDLFactorization<Field>& factorization, const std::string& label,
    std::ostream& os);

// Prints the diagonal factor of the LDL' factorization.
template <class Field>
void PrintDiagonalFactor(
    const LDLFactorization<Field>& factorization, const std::string& label,
    std::ostream& os);

}  // namespace catamari

#include "catamari/ldl-impl.hpp"

#endif  // ifndef CATAMARI_LDL_H_
