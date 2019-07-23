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
#include "catamari/buffer.hpp"
#include "catamari/macros.hpp"

namespace catamari {

// A BLAS-like, column-major matrix which handles its own resource allocation.
template <typename T>
struct BlasMatrix {
  // A description of a BLAS-style, column-major matrix.
  BlasMatrixView<T> view;

  // The underlying data buffer for the BLAS-style matrix.
  Buffer<T> data;

  // Initializes the matrix to 0 x 0.
  BlasMatrix();

  // Builds a height x width matrix without initialization.
  BlasMatrix(const Int& height, const Int& width);

  // Builds a height x width matrix with initialization.
  BlasMatrix(const Int& height, const Int& width, const T& value);

  // Builds a height x width matrix without initialization.
  BlasMatrix(const Int& height, const Int& width, const Int& leading_dim);

  // Builds a height x width matrix with initialization.
  BlasMatrix(const Int& height, const Int& width, const Int& leading_dim,
             const T& value);

  // Copy constructs a BLAS matrix by reducing the leading dimension to the
  // height.
  BlasMatrix(const BlasMatrix<T>& matrix);

  // Copy constructs a BLAS matrix view by reducing the leading dimension to the
  // height.
  BlasMatrix(const BlasMatrixView<T>& matrix);

  // Copy constructs a BLAS matrix view by reducing the leading dimension to the
  // height.
  BlasMatrix(const ConstBlasMatrixView<T>& matrix);

  // Copies a BLAS matrix by reducing the leading dimension to the height.
  BlasMatrix<T>& operator=(const BlasMatrix<T>& matrix);

  // Copies a BLAS matrix view by reducing the leading dimension to the height.
  BlasMatrix<T>& operator=(const BlasMatrixView<T>& matrix);

  // Copies a BLAS matrix view by reducing the leading dimension to the height.
  BlasMatrix<T>& operator=(const ConstBlasMatrixView<T>& matrix);

  // Resizes the matrix without initialization.
  void Resize(const Int& height, const Int& width);

  // Resizes the matrix with initialization to a given value.
  void Resize(const Int& height, const Int& width, const T& value);

  // Resizes the matrix with a particular leading dim without initialization.
  void Resize(const Int& height, const Int& width, const Int& leading_dim);

  // Resizes the matrix with a particular leading dim with initialization.
  void Resize(const Int& height, const Int& width, const Int& leading_dim,
              const T& value);

  // Returns the number of rows of the matrix.
  Int Height() const CATAMARI_NOEXCEPT;

  // Returns the number of columns of the matrix.
  Int Width() const CATAMARI_NOEXCEPT;

  // Returns the leading dimension of the matrix.
  Int LeadingDimension() const CATAMARI_NOEXCEPT;

  // Returns an immutable data pointer for the matrix.
  const T* Data() const CATAMARI_NOEXCEPT;

  // Returns a mutable data pointer for the matrix.
  T* Data() CATAMARI_NOEXCEPT;

  // Returns a modifiable reference to an entry of the matrix.
  T& operator()(Int row, Int column = 0);

  // Returns an immutable reference to an entry of the matrix.
  const T& operator()(Int row, Int column = 0) const;

  // Returns a modifiable reference to an entry of the matrix.
  T& Entry(Int row, Int column = 0);

  // Returns an immutable reference to an entry of the matrix.
  const T& Entry(Int row, Int column = 0) const;

  // Returns a modifiable pointer to a particular entry of the matrix.
  T* Pointer(Int row, Int column);

  // Returns an immutable pointer to a particular entry of the matrix.
  const T* Pointer(Int row, Int column) const;

  // Returns a modifiable view of the matrix.
  BlasMatrixView<T> View();

  // Returns a representation of the submatrix starting at position
  // (row_beg, column_beg) that has the given number of rows and columns.
  BlasMatrixView<T> Submatrix(Int row_beg, Int column_beg, Int num_rows,
                              Int num_columns);

  // Returns a representation of the submatrix starting at position
  // (row_beg, column_beg) that has the given number of rows and columns.
  ConstBlasMatrixView<T> Submatrix(Int row_beg, Int column_beg, Int num_rows,
                                   Int num_columns) const;

  // Returns a constant view of the matrix.
  const ConstBlasMatrixView<T> ConstView() const;
};

// Pretty-prints the BlasMatrix.
template <class T>
std::ostream& operator<<(std::ostream& os, const BlasMatrix<T>& matrix);
template <class T>
void Print(const BlasMatrix<T>& matrix, const std::string& label,
           std::ostream& os);

}  // namespace catamari

#include "catamari/blas_matrix-impl.hpp"

#endif  // ifndef CATAMARI_BLAS_MATRIX_H_
