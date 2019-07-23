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
Int ConstBlasMatrixView<T>::Height() const CATAMARI_NOEXCEPT {
  return height;
}

template <class T>
Int ConstBlasMatrixView<T>::Width() const CATAMARI_NOEXCEPT {
  return width;
}

template <class T>
Int ConstBlasMatrixView<T>::LeadingDimension() const CATAMARI_NOEXCEPT {
  return leading_dim;
}

template <class T>
const T* ConstBlasMatrixView<T>::Data() const CATAMARI_NOEXCEPT {
  return data;
}

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
ConstBlasMatrixView<T>::ConstBlasMatrixView() CATAMARI_NOEXCEPT {}

template <class T>
ConstBlasMatrixView<T>::ConstBlasMatrixView(const BlasMatrixView<T>& matrix)
    CATAMARI_NOEXCEPT : height(matrix.height),
                        width(matrix.width),
                        leading_dim(matrix.leading_dim),
                        data(matrix.data) {}

template <class T>
ConstBlasMatrixView<T>& ConstBlasMatrixView<T>::operator=(
    const BlasMatrixView<T>& matrix) CATAMARI_NOEXCEPT {
  height = matrix.height;
  width = matrix.width;
  leading_dim = matrix.leading_dim;
  data = matrix.leading_dim;
  return *this;
}

template <class T>
Int BlasMatrixView<T>::Height() const CATAMARI_NOEXCEPT {
  return height;
}

template <class T>
Int BlasMatrixView<T>::Width() const CATAMARI_NOEXCEPT {
  return width;
}

template <class T>
Int BlasMatrixView<T>::LeadingDimension() const CATAMARI_NOEXCEPT {
  return leading_dim;
}

template <class T>
T* BlasMatrixView<T>::Data() CATAMARI_NOEXCEPT {
  return data;
}

template <class T>
const T* BlasMatrixView<T>::Data() const CATAMARI_NOEXCEPT {
  return data;
}

template <class T>
ConstBlasMatrixView<T> BlasMatrixView<T>::ToConst() const CATAMARI_NOEXCEPT {
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

template <class T>
std::ostream& operator<<(std::ostream& os,
                         const ConstBlasMatrixView<T>& matrix) {
  for (Int i = 0; i < matrix.height; ++i) {
    for (Int j = 0; j < matrix.width; ++j) {
      os << matrix(i, j) << " ";
    }
    os << "\n";
  }
  return os;
}

template <class T>
std::ostream& operator<<(std::ostream& os, const BlasMatrixView<T>& matrix) {
  return os << matrix.ToConst();
}

template <class T>
void Print(const ConstBlasMatrixView<T>& matrix, const std::string& label,
           std::ostream& os) {
  os << label << ":\n";
  os << matrix << std::endl;
}

template <class T>
void Print(const BlasMatrixView<T>& matrix, const std::string& label,
           std::ostream& os) {
  os << label << ":\n";
  os << matrix << std::endl;
}

}  // namespace catamari

#endif  // ifndef CATAMARI_BLAS_MATRIX_VIEW_IMPL_H_
