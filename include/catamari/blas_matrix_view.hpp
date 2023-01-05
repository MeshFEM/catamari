/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_BLAS_MATRIX_VIEW_H_
#define CATAMARI_BLAS_MATRIX_VIEW_H_

#include <iostream>
#include <string>

#include "catamari/integers.hpp"
#include "catamari/macros.hpp"

namespace catamari {

// Forward declaration.
template <class T>
struct BlasMatrixView;

// A data structure for manipulating a view of a constant BLAS-style
// column-major matrix.
template <class T>
struct ConstBlasMatrixView {
  // The number of rows of the matrix.
  Int height;

  // The number of columns of the matrix.
  Int width;

  // The stride between consecutive entries in the same row of the matrix.
  Int leading_dim;

  // The pointer to the top-left entry of the matrix.
  const T* data;

  // Returns the number of rows of the matrix.
  Int Height() const CATAMARI_NOEXCEPT;

  // Returns the number of columns of the matrix.
  Int Width() const CATAMARI_NOEXCEPT;

  // Returns the leading dimension of the matrix.
  Int LeadingDimension() const CATAMARI_NOEXCEPT;

  // Returns an immutable pointer to the top-left entry of the matrix.
  const T* Data() const CATAMARI_NOEXCEPT;

  // Returns a const pointer to the entry in position (row, column).
  const T* Pointer(Int row, Int column) const;

  // Returns a const reference to the entry in position (row, column).
  const T& operator()(Int row, Int column = 0) const;

  // Enable ConstBlasMatrixView to masquerade as a const Buffer or some other 1D array
  // to avoid code duplication in function templates (potentially dangerous).
  const T& operator[](Int row) const{ return data[row]; }

  // Returns a const reference to the entry in position (row, column).
  const T& Entry(Int row, Int column = 0) const;

  // Returns a representation of the submatrix starting at position
  // (row_beg, column_beg) that has the given number of rows and columns.
  ConstBlasMatrixView<T> Submatrix(Int row_beg, Int column_beg, Int num_rows,
                                   Int num_columns) const;

  // A default constructor.
  ConstBlasMatrixView() CATAMARI_NOEXCEPT;

  // A copy constructor from a mutable matrix.
  ConstBlasMatrixView(const BlasMatrixView<T>& matrix) CATAMARI_NOEXCEPT;

  // An assignment operator from a mutable matrix.
  ConstBlasMatrixView<T>& operator=(const BlasMatrixView<T>& matrix)
      CATAMARI_NOEXCEPT;
};

// A data structure for manipulating a view of a BLAS-style column-major matrix.
template <class T>
struct BlasMatrixView {
  // The number of rows of the matrix.
  Int height;

  // The number of columns of the matrix.
  Int width;

  // The stride between consecutive entries in the same row of the matrix.
  Int leading_dim;

  // The pointer to the top-left entry of the matrix.
  T* data;

  // Returns the number of rows of the matrix.
  Int Height() const CATAMARI_NOEXCEPT;

  // Returns the number of columns of the matrix.
  Int Width() const CATAMARI_NOEXCEPT;

  // Returns the leading dimension of the matrix.
  Int LeadingDimension() const CATAMARI_NOEXCEPT;

  // Returns a pointer to the top-left entry of the matrix.
  T* Data() CATAMARI_NOEXCEPT;

  // Returns an immutable pointer to the top-left entry of the matrix.
  const T* Data() const CATAMARI_NOEXCEPT;

  // Returns a constant equivalent of the current state.
  ConstBlasMatrixView<T> ToConst() const CATAMARI_NOEXCEPT;

  // Returns a pointer to the entry in position (row, column).
  T* Pointer(Int row, Int column);

  // Returns a const pointer to the entry in position (row, column).
  const T* Pointer(Int row, Int column) const;

  // Returns a reference to the entry in position (row, column).
  T& operator()(Int row, Int column = 0);

  // Returns a const reference to the entry in position (row, column).
  const T& operator()(Int row, Int column = 0) const;

  // Enable BlasMatrixView to masquerade as a Buffer or some other 1D array
  // to avoid code duplication in function templates (potentially dangerous).
        T& operator[](Int row)      { return data[row]; }
  const T& operator[](Int row) const{ return data[row]; }

  // Returns a reference to the entry in position (row, column).
  T& Entry(Int row, Int column = 0);

  // Returns a const reference to the entry in position (row, column).
  const T& Entry(Int row, Int column = 0) const;

  // Returns a representation of the submatrix starting at position
  // (row_beg, column_beg) that has the given number of rows and columns.
  BlasMatrixView<T> Submatrix(Int row_beg, Int column_beg, Int num_rows,
                              Int num_columns);

  // Returns a constant representation of the submatrix starting at position
  // (row_beg, column_beg) that has the given number of rows and columns.
  ConstBlasMatrixView<T> Submatrix(Int row_beg, Int column_beg, Int num_rows,
                                   Int num_columns) const;
};

// Pretty-prints the BlasMatrixView.
template <class T>
std::ostream& operator<<(std::ostream& os,
                         const ConstBlasMatrixView<T>& matrix);
template <class T>
std::ostream& operator<<(std::ostream& os, const BlasMatrixView<T>& matrix);
template <class T>
void Print(const ConstBlasMatrixView<T>& matrix, const std::string& label,
           std::ostream& os);
template <class T>
void Print(const BlasMatrixView<T>& matrix, const std::string& label,
           std::ostream& os);

}  // namespace catamari

#include "catamari/blas_matrix_view-impl.hpp"

#endif  // ifndef CATAMARI_BLAS_MATRIX_VIEW_H_
