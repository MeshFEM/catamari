/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_EQUILIBRATE_SYMMETRIC_MATRIX_H_
#define CATAMARI_EQUILIBRATE_SYMMETRIC_MATRIX_H_

#include "catamari/blas_matrix.hpp"
#include "catamari/coordinate_matrix.hpp"

namespace catamari {

// Rescales a symmetric or Hermitian matrix in-place and returns the
// corresponding scaling. Our approach is essentially the simple algorithm
// described in [Ruiz-2001].
//
// A diagonal matrix D is computed, which transforms systems
//
//     A x = b,
//
// into the form:
//
//     A_{rescaled} (D x) = (inv(D) b),
//
// where
//
//     A_{rescaled} = inv(D) A inv(D).
//
template <typename Field>
void EquilibrateSymmetricMatrix(CoordinateMatrix<Field>* matrix,
                                BlasMatrix<ComplexBase<Field>>* scaling,
                                bool verbose = false);

// References:
//
// [Ruiz-2001]
//   Daniel Ruiz, "A Scaling Algorithm to Equilibrate Both Rows and Columns
//   Norms in Matrices", Technical Report RAL-TR-2001-034, Rutherford Appleton
//   Laboratory, 2001.
//

}  // namespace catamari

#include "catamari/equilibrate_symmetric_matrix-impl.hpp"

#endif  // ifndef CATAMARI_EQUILIBRATE_SYMMETRIC_MATRIX_H_
