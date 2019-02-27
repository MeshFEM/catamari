/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_BLAS_MATRIX_VIEW_IMPL_H_
#define CATAMARI_BLAS_MATRIX_VIEW_IMPL_H_

#include "catamari/blas_matrix_view.hpp"

namespace catamari {

template <class T>
const T* ConstBlasMatrixView<T>::Pointer(Int row, Int column) const {
  return &data[row + column * leading_dim];
}

template <class T>
const T& ConstBlasMatrixView<T>::operator()(Int row, Int column) const {
  return data[row + column * leading_dim];
}

template <class T>
const T& ConstBlasMatrixView<T>::Entry(Int row, Int column) const {
  return data[row + column * leading_dim];
}

template <class T>
ConstBlasMatrixView<T> ConstBlasMatrixView<T>::Submatrix(
    Int row_beg, Int column_beg, Int num_rows, Int num_columns) const {
  ConstBlasMatrixView<T> submatrix;
  submatrix.height = num_rows;
  submatrix.width = num_columns;
  submatrix.leading_dim = leading_dim;
  submatrix.data = Pointer(row_beg, column_beg);
  return submatrix;
}

template <class T>
ConstBlasMatrixView<T>::ConstBlasMatrixView() {}

template <class T>
ConstBlasMatrixView<T>::ConstBlasMatrixView(const BlasMatrixView<T>& matrix)
    : height(matrix.height),
      width(matrix.width),
      leading_dim(matrix.leading_dim),
      data(matrix.data) {}

template <class T>
ConstBlasMatrixView<T>& ConstBlasMatrixView<T>::operator=(
    const BlasMatrixView<T>& matrix) {
  height = matrix.height;
  width = matrix.width;
  leading_dim = matrix.leading_dim;
  data = matrix.leading_dim;
}

template <class T>
ConstBlasMatrixView<T> BlasMatrixView<T>::ToConst() const {
  ConstBlasMatrixView<T> const_matrix = *this;
  return const_matrix;
}

template <class T>
T* BlasMatrixView<T>::Pointer(Int row, Int column) {
  return &data[row + column * leading_dim];
}

template <class T>
const T* BlasMatrixView<T>::Pointer(Int row, Int column) const {
  return &data[row + column * leading_dim];
}

template <class T>
T& BlasMatrixView<T>::operator()(Int row, Int column) {
  return data[row + column * leading_dim];
}

template <class T>
const T& BlasMatrixView<T>::operator()(Int row, Int column) const {
  return data[row + column * leading_dim];
}

template <class T>
T& BlasMatrixView<T>::Entry(Int row, Int column) {
  return data[row + column * leading_dim];
}

template <class T>
const T& BlasMatrixView<T>::Entry(Int row, Int column) const {
  return data[row + column * leading_dim];
}

template <class T>
BlasMatrixView<T> BlasMatrixView<T>::Submatrix(Int row_beg, Int column_beg,
                                               Int num_rows, Int num_columns) {
  BlasMatrixView<T> submatrix;
  submatrix.height = num_rows;
  submatrix.width = num_columns;
  submatrix.leading_dim = leading_dim;
  submatrix.data = Pointer(row_beg, column_beg);
  return submatrix;
}

template <class T>
ConstBlasMatrixView<T> BlasMatrixView<T>::Submatrix(Int row_beg, Int column_beg,
                                                    Int num_rows,
                                                    Int num_columns) const {
  ConstBlasMatrixView<T> submatrix;
  submatrix.height = num_rows;
  submatrix.width = num_columns;
  submatrix.leading_dim = leading_dim;
  submatrix.data = Pointer(row_beg, column_beg);
  return submatrix;
}

}  // namespace catamari

#endif  // ifndef CATAMARI_BLAS_MATRIX_VIEW_IMPL_H_
