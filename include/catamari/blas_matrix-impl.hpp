/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_BLAS_MATRIX_IMPL_H_
#define CATAMARI_BLAS_MATRIX_IMPL_H_

#include "catamari/blas_matrix.hpp"

namespace catamari {

template <class T>
const T* ConstBlasMatrix<T>::Pointer(Int row, Int column) const {
  return &data[row + column * leading_dim];
}

template <class T>
const T& ConstBlasMatrix<T>::operator()(Int row, Int column) const {
  return data[row + column * leading_dim];
}

template <class T>
const T& ConstBlasMatrix<T>::Entry(Int row, Int column) const {
  return data[row + column * leading_dim];
}

template <class T>
ConstBlasMatrix<T> ConstBlasMatrix<T>::Submatrix(Int row_beg, Int column_beg,
                                                 Int num_rows,
                                                 Int num_columns) const {
  ConstBlasMatrix<T> submatrix;
  submatrix.height = num_rows;
  submatrix.width = num_columns;
  submatrix.leading_dim = leading_dim;
  submatrix.data = Pointer(row_beg, column_beg);
  return submatrix;
}

template <class T>
ConstBlasMatrix<T>::ConstBlasMatrix() {}

template <class T>
ConstBlasMatrix<T>::ConstBlasMatrix(const BlasMatrix<T>& matrix)
    : height(matrix.height),
      width(matrix.width),
      leading_dim(matrix.leading_dim),
      data(matrix.data) {}

template <class T>
ConstBlasMatrix<T>& ConstBlasMatrix<T>::operator=(const BlasMatrix<T>& matrix) {
  height = matrix.height;
  width = matrix.width;
  leading_dim = matrix.leading_dim;
  data = matrix.leading_dim;
}

template <class T>
ConstBlasMatrix<T> BlasMatrix<T>::ToConst() const {
  ConstBlasMatrix<T> const_matrix = *this;
  return const_matrix;
}

template <class T>
T* BlasMatrix<T>::Pointer(Int row, Int column) {
  return &data[row + column * leading_dim];
}

template <class T>
const T* BlasMatrix<T>::Pointer(Int row, Int column) const {
  return &data[row + column * leading_dim];
}

template <class T>
T& BlasMatrix<T>::operator()(Int row, Int column) {
  return data[row + column * leading_dim];
}

template <class T>
const T& BlasMatrix<T>::operator()(Int row, Int column) const {
  return data[row + column * leading_dim];
}

template <class T>
T& BlasMatrix<T>::Entry(Int row, Int column) {
  return data[row + column * leading_dim];
}

template <class T>
const T& BlasMatrix<T>::Entry(Int row, Int column) const {
  return data[row + column * leading_dim];
}

template <class T>
BlasMatrix<T> BlasMatrix<T>::Submatrix(Int row_beg, Int column_beg,
                                       Int num_rows, Int num_columns) {
  BlasMatrix<T> submatrix;
  submatrix.height = num_rows;
  submatrix.width = num_columns;
  submatrix.leading_dim = leading_dim;
  submatrix.data = Pointer(row_beg, column_beg);
  return submatrix;
}

template <class T>
ConstBlasMatrix<T> BlasMatrix<T>::Submatrix(Int row_beg, Int column_beg,
                                            Int num_rows,
                                            Int num_columns) const {
  ConstBlasMatrix<T> submatrix;
  submatrix.height = num_rows;
  submatrix.width = num_columns;
  submatrix.leading_dim = leading_dim;
  submatrix.data = Pointer(row_beg, column_beg);
  return submatrix;
}

}  // namespace catamari

#endif  // ifndef CATAMARI_BLAS_MATRIX_IMPL_H_
