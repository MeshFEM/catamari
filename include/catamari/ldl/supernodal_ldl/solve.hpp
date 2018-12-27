/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SUPERNODAL_LDL_SOLVE_H_
#define CATAMARI_LDL_SUPERNODAL_LDL_SOLVE_H_

#include "catamari/ldl/supernodal_ldl/factorization.hpp"

namespace catamari {
namespace supernodal_ldl {

// Solve (P A P') (P X) = (P B) via the substitution (L D L') (P X) = (P B) or
// (L D L^T) (P X) = (P B).
template <class Field>
void Solve(const Factorization<Field>& factorization,
           BlasMatrix<Field>* matrix);

// Solves L X = B using a lower triangular matrix L.
template <class Field>
void LowerTriangularSolve(const Factorization<Field>& factorization,
                          BlasMatrix<Field>* matrix);

// Solves D X = B using a diagonal matrix D.
template <class Field>
void DiagonalSolve(const Factorization<Field>& factorization,
                   BlasMatrix<Field>* matrix);

// Solves L' X = B or L^T X = B using a lower triangular matrix L.
template <class Field>
void LowerTransposeTriangularSolve(const Factorization<Field>& factorization,
                                   BlasMatrix<Field>* matrix);

}  // namespace supernodal_ldl
}  // namespace catamari

#include "catamari/ldl/supernodal_ldl/solve-impl.hpp"

#endif  // ifndef CATAMARI_LDL_SUPERNODAL_LDL_SOLVE_H_
