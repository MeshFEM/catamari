/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_BLAS_MATRIX_H_
#define CATAMARI_BLAS_MATRIX_H_

#include "catamari/blas_matrix_view.hpp"

namespace catamari {

// A BLAS-like, column-major matrix which handles its own resource allocation.
template <typename Field>
struct BlasMatrix {
  // A description of a BLAS-style, column-major matrix.
  BlasMatrixView<Field> view;

  // The underlying data buffer for the BLAS-style matrix.
  Buffer<Field> data;

  // Initializes the matrix to 0 x 0.
  BlasMatrix();

  // Copy constructs a BLAS matrix by reducing the leading dimension to the
  // height.
  BlasMatrix(const BlasMatrix<Field>& matrix);

  // Copies a BLAS matrix by reducing the leading dimension to the height.
  BlasMatrix<Field>& operator=(const BlasMatrix<Field>& matrix);

  // Resizes the matrix without initialization.
  void Resize(const Int& height, const Int& width);

  // Resizes the matrix with initialization to a given value.
  void Resize(const Int& height, const Int& width, const Field& value);

  // Returns a modifiable reference to an entry of the matrix.
  Field& operator()(Int row, Int column);

  // Returns an immutable reference to an entry of the matrix.
  const Field& operator()(Int row, Int column) const;

  // Returns a modifiable reference to an entry of the matrix.
  Field& Entry(Int row, Int column);

  // Returns an immutable reference to an entry of the matrix.
  const Field& Entry(Int row, Int column) const;
};

}  // namespace catamari

#include "catamari/blas_matrix-impl.hpp"

#endif  // ifndef CATAMARI_BLAS_MATRIX_H_
