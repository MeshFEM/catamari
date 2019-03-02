/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_NORMS_H_
#define CATAMARI_NORMS_H_

#include "catamari/blas_matrix_view.hpp"
#include "catamari/coordinate_matrix.hpp"
#include "catamari/scalar_functions.hpp"

namespace catamari {

// Returns the Frobenius norm of a dense matrix.
template <typename Field>
ComplexBase<Field> EuclideanNorm(const ConstBlasMatrixView<Field>& matrix) {
  typedef ComplexBase<Field> Real;
  Real scale = 0;
  Real scaled_square = 1;
  const Int height = matrix.height;
  const Int width = matrix.width;
  for (Int j = 0; j < width; ++j) {
    for (Int i = 0; i < height; ++i) {
      catamari::UpdateScaledSquare(matrix(i, j), &scale, &scaled_square);
    }
  }
  return scale * catamari::Sqrt(scaled_square);
}

// Returns the Frobenius norm of a sparse matrix.
template <typename Field>
ComplexBase<Field> EuclideanNorm(const CoordinateMatrix<Field>& matrix) {
  typedef ComplexBase<Field> Real;
  Real scale = 0;
  Real scaled_square = 1;
  for (const MatrixEntry<Field>& entry : matrix.Entries()) {
    catamari::UpdateScaledSquare(entry.value, &scale, &scaled_square);
  }
  return scale * catamari::Sqrt(scaled_square);
}

}  // namespace catamari

#endif  // ifndef CATAMARI_NORMS_H_
