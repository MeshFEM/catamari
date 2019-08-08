/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_BLAS_MATRIX_IMPL_H_
#define CATAMARI_BLAS_MATRIX_IMPL_H_

#include "catamari/blas_matrix.hpp"

namespace catamari {

template <typename T>
BlasMatrix<T>::BlasMatrix() {
  view.height = 0;
  view.width = 0;
  view.leading_dim = 0;
  view.data = nullptr;
}

template <typename T>
BlasMatrix<T>::BlasMatrix(const Int& height, const Int& width) {
  view.height = 0;
  view.width = 0;
  view.leading_dim = 0;
  view.data = nullptr;
  Resize(height, width);
}

template <typename T>
BlasMatrix<T>::BlasMatrix(const Int& height, const Int& width, const T& value) {
  view.height = 0;
  view.width = 0;
  view.leading_dim = 0;
  view.data = nullptr;
  Resize(height, width, value);
}

template <typename T>
BlasMatrix<T>::BlasMatrix(const Int& height, const Int& width,
                          const Int& leading_dim) {
  view.height = 0;
  view.width = 0;
  view.leading_dim = 0;
  view.data = nullptr;
  Resize(height, width, leading_dim);
}

template <typename T>
BlasMatrix<T>::BlasMatrix(const Int& height, const Int& width,
                          const Int& leading_dim, const T& value) {
  view.height = 0;
  view.width = 0;
  view.leading_dim = 0;
  view.data = nullptr;
  Resize(height, width, leading_dim, value);
}

template <typename T>
BlasMatrix<T>::BlasMatrix(const BlasMatrix<T>& matrix) {
  const Int height = matrix.view.height;
  const Int width = matrix.view.width;

  data.Resize(height * width);
  view.height = height;
  view.width = width;
  view.leading_dim = height;
  view.data = data.Data();

  // Copy each individual column so that the leading dimension does not
  // impact the copy time.
  for (Int j = 0; j < width; ++j) {
    std::copy(&matrix(0, j), &matrix(height, j), &view(0, j));
  }
}

template <typename T>
BlasMatrix<T>::BlasMatrix(const BlasMatrixView<T>& matrix)
    : BlasMatrix<T>(matrix.ToConst()) {}

template <typename T>
BlasMatrix<T>::BlasMatrix(const ConstBlasMatrixView<T>& matrix) {
  const Int height = matrix.height;
  const Int width = matrix.width;

  data.Resize(height * width);
  view.height = height;
  view.width = width;
  view.leading_dim = height;
  view.data = data.Data();

  // Copy each individual column so that the leading dimension does not
  // impact the copy time.
  for (Int j = 0; j < width; ++j) {
    std::copy(&matrix(0, j), &matrix(height, j), &view(0, j));
  }
}

template <typename T>
BlasMatrix<T>& BlasMatrix<T>::operator=(const BlasMatrix<T>& matrix) {
  if (this != &matrix) {
    const Int height = matrix.view.height;
    const Int width = matrix.view.width;

    data.Resize(height * width);
    view.height = height;
    view.width = width;
    view.leading_dim = height;
    view.data = data.Data();

    // Copy each individual column so that the leading dimension does not
    // impact the copy time.
    for (Int j = 0; j < width; ++j) {
      std::copy(&matrix(0, j), &matrix(height, j), &view(0, j));
    }
  }
  return *this;
}

template <typename T>
BlasMatrix<T>& BlasMatrix<T>::operator=(const BlasMatrixView<T>& matrix) {
  *this = matrix.ToConst();
  return *this;
}

template <typename T>
BlasMatrix<T>& BlasMatrix<T>::operator=(const ConstBlasMatrixView<T>& matrix) {
  const Int height = matrix.height;
  const Int width = matrix.width;

  data.Resize(height * width);
  view.height = height;
  view.width = width;
  view.leading_dim = height;
  view.data = data.Data();

  // Copy each individual column so that the leading dimension does not
  // impact the copy time.
  for (Int j = 0; j < width; ++j) {
    std::copy(&matrix(0, j), &matrix(height, j), &view(0, j));
  }
  return *this;
}

template <typename T>
void BlasMatrix<T>::Resize(const Int& height, const Int& width) {
  if (height == view.height && width == view.width) {
    return;
  }
  // TODO(Jack Poulson): Decide on how to handle 0 leading dimensions.
  data.Resize(height * width);
  view.height = height;
  view.width = width;
  view.leading_dim = height;
  view.data = data.Data();
}

template <typename T>
void BlasMatrix<T>::Resize(const Int& height, const Int& width,
                           const Int& leading_dim) {
  if (height == view.height && width == view.width) {
    return;
  }
  if (leading_dim < height) {
    throw std::invalid_argument(
        "The leading dimension must be at least as large as the height.");
  }
  // TODO(Jack Poulson): Decide on how to handle 0 leading dimensions.
  data.Resize(leading_dim * width);
  view.height = height;
  view.width = width;
  view.leading_dim = leading_dim;
  view.data = data.Data();
}

template <typename T>
void BlasMatrix<T>::Resize(const Int& height, const Int& width,
                           const T& value) {
  // TODO(Jack Poulson): Decide on how to handle 0 leading dimensions.
  data.Resize(height * width, value);
  view.height = height;
  view.width = width;
  view.leading_dim = height;
  view.data = data.Data();
}

template <typename T>
void BlasMatrix<T>::Resize(const Int& height, const Int& width,
                           const Int& leading_dim, const T& value) {
  if (leading_dim < height) {
    throw std::invalid_argument(
        "The leading dimension must be at least as large as the height.");
  }
  // TODO(Jack Poulson): Decide on how to handle 0 leading dimensions.
  data.Resize(leading_dim * width, value);
  view.height = height;
  view.width = width;
  view.leading_dim = leading_dim;
  view.data = data.Data();
}

template <typename T>
Int BlasMatrix<T>::Height() const CATAMARI_NOEXCEPT {
  return view.height;
}

template <typename T>
Int BlasMatrix<T>::Width() const CATAMARI_NOEXCEPT {
  return view.width;
}

template <typename T>
Int BlasMatrix<T>::LeadingDimension() const CATAMARI_NOEXCEPT {
  return view.leading_dim;
}

template <typename T>
const T* BlasMatrix<T>::Data() const CATAMARI_NOEXCEPT {
  return view.data;
}

template <typename T>
T* BlasMatrix<T>::Data() CATAMARI_NOEXCEPT {
  return view.data;
}

template <typename T>
T& BlasMatrix<T>::operator()(Int row, Int column) {
  return view(row, column);
}

template <typename T>
const T& BlasMatrix<T>::operator()(Int row, Int column) const {
  return view(row, column);
}

template <typename T>
T& BlasMatrix<T>::Entry(Int row, Int column) {
  return view(row, column);
}

template <typename T>
const T& BlasMatrix<T>::Entry(Int row, Int column) const {
  return view(row, column);
}

template <typename T>
T* BlasMatrix<T>::Pointer(Int row, Int column) {
  return view.Pointer(row, column);
}

template <typename T>
const T* BlasMatrix<T>::Pointer(Int row, Int column) const {
  return view.Pointer(row, column);
}

template <typename T>
BlasMatrixView<T> BlasMatrix<T>::View() {
  return view;
}

template <typename T>
const ConstBlasMatrixView<T> BlasMatrix<T>::ConstView() const {
  return view.ToConst();
}

template <typename T>
BlasMatrixView<T> BlasMatrix<T>::Submatrix(Int row_beg, Int column_beg,
                                           Int num_rows, Int num_columns) {
  return View().Submatrix(row_beg, column_beg, num_rows, num_columns);
}

template <typename T>
ConstBlasMatrixView<T> BlasMatrix<T>::Submatrix(Int row_beg, Int column_beg,
                                                Int num_rows,
                                                Int num_columns) const {
  return ConstView().Submatrix(row_beg, column_beg, num_rows, num_columns);
}

template <class T>
std::ostream& operator<<(std::ostream& os, const BlasMatrix<T>& matrix) {
  return os << matrix.ConstView();
}

template <class T>
void Print(const BlasMatrix<T>& matrix, const std::string& label,
           std::ostream& os) {
  os << label << ":\n";
  os << matrix.ConstView() << std::endl;
}

}  // namespace catamari

#endif  // ifndef CATAMARI_BLAS_MATRIX_IMPL_H_
