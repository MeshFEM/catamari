/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_BLAS_MATRIX_H_
#define CATAMARI_BLAS_MATRIX_H_

#include "catamari/integers.hpp"

namespace catamari {

// Forward declaration.
template <class T>
struct BlasMatrix;

// A data structure for manipulating a constant BLAS-style column-major matrix.
template <class T>
struct ConstBlasMatrix {
  // The number of rows of the matrix.
  Int height;

  // The number of columns of the matrix.
  Int width;

  // The stride between consecutive entries in the same row of the matrix.
  Int leading_dim;

  // The pointer to the top-left entry of the matrix.
  const T* data;

  // Returns a const pointer to the entry in position (row, column).
  const T* Pointer(Int row, Int column) const;

  // Returns a const reference to the entry in position (row, column).
  const T& operator()(Int row, Int column) const;

  // Returns a const reference to the entry in position (row, column).
  const T& Entry(Int row, Int column) const;

  // Returns a representation of the submatrix starting at position
  // (row_beg, column_beg) that has the given number of rows and columns.
  ConstBlasMatrix<T> Submatrix(Int row_beg, Int column_beg, Int num_rows,
                               Int num_columns) const;

  // A default constructor.
  ConstBlasMatrix();

  // A copy constructor from a mutable matrix.
  ConstBlasMatrix(const BlasMatrix<T>& matrix);

  // An assignment operator from a mutable matrix.
  ConstBlasMatrix<T>& operator=(const BlasMatrix<T>& matrix);
};

// A data structure for manipulating a BLAS-style column-major matrix.
template <class T>
struct BlasMatrix {
  // The number of rows of the matrix.
  Int height;

  // The number of columns of the matrix.
  Int width;

  // The stride between consecutive entries in the same row of the matrix.
  Int leading_dim;

  // The pointer to the top-left entry of the matrix.
  T* data;

  // Returns a constant equivalent of the current state.
  ConstBlasMatrix<T> ToConst() const;

  // Returns a pointer to the entry in position (row, column).
  T* Pointer(Int row, Int column);

  // Returns a const pointer to the entry in position (row, column).
  const T* Pointer(Int row, Int column) const;

  // Returns a reference to the entry in position (row, column).
  T& operator()(Int row, Int column);

  // Returns a const reference to the entry in position (row, column).
  const T& operator()(Int row, Int column) const;

  // Returns a reference to the entry in position (row, column).
  T& Entry(Int row, Int column);

  // Returns a const reference to the entry in position (row, column).
  const T& Entry(Int row, Int column) const;

  // Returns a representation of the submatrix starting at position
  // (row_beg, column_beg) that has the given number of rows and columns.
  BlasMatrix<T> Submatrix(Int row_beg, Int column_beg, Int num_rows,
                          Int num_columns);

  // Returns a constant representation of the submatrix starting at position
  // (row_beg, column_beg) that has the given number of rows and columns.
  ConstBlasMatrix<T> Submatrix(Int row_beg, Int column_beg, Int num_rows,
                               Int num_columns) const;
};

}  // namespace catamari

#include "catamari/blas_matrix-impl.hpp"

#endif  // ifndef CATAMARI_BLAS_MATRIX_H_
