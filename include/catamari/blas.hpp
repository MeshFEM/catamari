/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_BLAS_H_
#define CATAMARI_BLAS_H_

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

template <class Field>
void MatrixVectorProduct(
    const Field& alpha, const ConstBlasMatrix<Field>& matrix,
    const Field* input_vector, Field* result);

template <class Field>
void ConjugateMatrixVectorProduct(
    const Field& alpha, const ConstBlasMatrix<Field>& matrix,
    const Field* input_vector, Field* result);

template <class Field>
void TransposeMatrixVectorProduct(
    const Field& alpha, const ConstBlasMatrix<Field>& matrix,
    const Field* input_vector, Field* result);

template <class Field>
void TriangularSolveLeftLower(const ConstBlasMatrix<Field>& triangular_matrix,
                              Field* vector);

template <class Field>
void TriangularSolveLeftLowerUnit(
    const ConstBlasMatrix<Field>& triangular_matrix, Field* vector);

template <class Field>
void TriangularSolveLeftLowerAdjoint(
    const ConstBlasMatrix<Field>& triangular_matrix, Field* vector);

template <class Field>
void TriangularSolveLeftLowerAdjointUnit(
    const ConstBlasMatrix<Field>& triangular_matrix, Field* vector);

template <class Field>
void MatrixMultiplyConjugateNormal(const Field& alpha,
                                   const ConstBlasMatrix<Field>& left_matrix,
                                   const ConstBlasMatrix<Field>& right_matrix,
                                   const Field& beta,
                                   BlasMatrix<Field>* output_matrix);

template <class Field>
void MatrixMultiplyTransposeNormal(const Field& alpha,
                                   const ConstBlasMatrix<Field>& left_matrix,
                                   const ConstBlasMatrix<Field>& right_matrix,
                                   const Field& beta,
                                   BlasMatrix<Field>* output_matrix);

template <class Field>
void LowerTransposeHermitianOuterProduct(
    const Field& alpha, const ConstBlasMatrix<Field>& left_matrix,
    const Field& beta, BlasMatrix<Field>* output_matrix);

template <class Field>
void MatrixMultiplyLowerNormalTranspose(
    const Field& alpha, const ConstBlasMatrix<Field>& left_matrix,
    const ConstBlasMatrix<Field>& right_matrix, const Field& beta,
    BlasMatrix<Field>* output_matrix);

template <class Field>
void MatrixMultiplyLowerTransposeNormal(
    const Field& alpha, const ConstBlasMatrix<Field>& left_matrix,
    const ConstBlasMatrix<Field>& right_matrix, const Field& beta,
    BlasMatrix<Field>* output_matrix);

template <class Field>
void LeftLowerTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix, BlasMatrix<Field>* matrix);

template <class Field>
void LeftLowerUnitTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix, BlasMatrix<Field>* matrix);

template <class Field>
void LeftLowerConjugateTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix, BlasMatrix<Field>* matrix);

template <class Field>
void LeftLowerAdjointTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix, BlasMatrix<Field>* matrix);

template <class Field>
void LeftLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix, BlasMatrix<Field>* matrix);

template <class Field>
void DiagonalTimesLeftLowerConjugateTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix, BlasMatrix<Field>* matrix);

template <class Field>
void RightLowerAdjointTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix, BlasMatrix<Field>* matrix);

template <class Field>
void RightLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix, BlasMatrix<Field>* matrix);

}  // namespace catamari

#include "catamari/blas-impl.hpp"

#endif  // ifndef CATAMARI_BLAS_H_
