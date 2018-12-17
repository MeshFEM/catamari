/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_BLAS_IMPL_H_
#define CATAMARI_BLAS_IMPL_H_

#include "catamari/blas.hpp"

#ifdef CATAMARI_HAVE_MKL

// TODO(Jack Poulson): Attempt to support 64-bit BLAS when Int = long long int.
typedef int BlasInt;

#define CATAMARI_HAVE_BLAS

#define BLAS_SYMBOL(name) name##_

// TODO(Jack Poulson): Decide when to avoid enabling this function. It seems to
// be slightly slower than doing twice as much work by running Gemm then
// setting the strictly upper triangle of the result to zero.
#define CATAMARI_USE_GEMMT

#elif defined(CATAMARI_HAVE_OPENBLAS)

// TODO(Jack Poulson): Attempt to support 64-bit BLAS when Int = long long int.
typedef int BlasInt;

#define CATAMARI_HAVE_BLAS
#define BLAS_SYMBOL(name) name##_

#endif  // ifdef CATAMARI_HAVE_OPENBLAS

#ifdef CATAMARI_HAVE_BLAS
extern "C" {

void BLAS_SYMBOL(ssyr)(const char* uplo, const BlasInt* height,
                       const float* alpha, const float* vector,
                       const BlasInt* stride, float* matrix,
                       const BlasInt* leading_dim);

void BLAS_SYMBOL(dsyr)(const char* uplo, const BlasInt* height,
                       const double* alpha, const double* vector,
                       const BlasInt* stride, double* matrix,
                       const BlasInt* leading_dim);

void BLAS_SYMBOL(dgemv)(const char* trans, const BlasInt* height,
                        const BlasInt* width, const double* alpha,
                        const double* matrix, const BlasInt* leading_dim,
                        const double* input_vector, const BlasInt* input_stride,
                        const double* beta, double* result,
                        const BlasInt* result_stride);

void BLAS_SYMBOL(sgemv)(const char* trans, const BlasInt* height,
                        const BlasInt* width, const float* alpha,
                        const float* matrix, const BlasInt* leading_dim,
                        const float* input_vector, const BlasInt* input_stride,
                        const float* beta, float* result,
                        const BlasInt* result_stride);

void BLAS_SYMBOL(dtrsv)(const char* uplo, const char* trans, const char* diag,
                        const BlasInt* height, const double* triangular_matrix,
                        const BlasInt* triang_leading_dim, double* vector,
                        const BlasInt* stride);

void BLAS_SYMBOL(strsv)(const char* uplo, const char* trans, const char* diag,
                        const BlasInt* height, const float* triangular_matrix,
                        const BlasInt* triang_leading_dim, float* vector,
                        const BlasInt* stride);

void BLAS_SYMBOL(dgemm)(const char* trans_left, const char* trans_right,
                        const BlasInt* output_height,
                        const BlasInt* output_width,
                        const BlasInt* contraction_size, const double* alpha,
                        const double* left_matrix,
                        const BlasInt* left_leading_dim,
                        const double* right_matrix,
                        const BlasInt* right_leading_dim, const double* beta,
                        double* output_matrix,
                        const BlasInt* output_leading_dim);

void BLAS_SYMBOL(sgemm)(
    const char* trans_left, const char* trans_right,
    const BlasInt* output_height, const BlasInt* output_width,
    const BlasInt* contraction_size, const float* alpha,
    const float* left_matrix, const BlasInt* left_leading_dim,
    const float* right_matrix, const BlasInt* right_leading_dim,
    const float* beta, float* output_matrix, const BlasInt* output_leading_dim);

#ifdef CATAMARI_HAVE_MKL
void BLAS_SYMBOL(dgemmt)(const char* uplo, const char* trans_left,
                         const char* trans_right, const BlasInt* height,
                         const BlasInt* rank, const double* alpha,
                         const double* left_matrix,
                         const BlasInt* left_leading_dim,
                         const double* right_matrix,
                         const BlasInt* right_leading_dim, const double* beta,
                         double* matrix, const BlasInt* leading_dim);

void BLAS_SYMBOL(sgemmt)(const char* uplo, const char* trans_left,
                         const char* trans_right, const BlasInt* height,
                         const BlasInt* rank, const float* alpha,
                         const float* left_matrix,
                         const BlasInt* left_leading_dim,
                         const float* right_matrix,
                         const BlasInt* right_leading_dim, const float* beta,
                         float* matrix, const BlasInt* leading_dim);
#endif  // ifdef CATAMARI_HAVE_MKL

void BLAS_SYMBOL(dsyrk)(const char* uplo, const char* trans,
                        const BlasInt* height, const BlasInt* rank,
                        const double* alpha, const double* factor,
                        const BlasInt* factor_leading_dim, const double* beta,
                        double* matrix, const BlasInt* leading_dim);

void BLAS_SYMBOL(ssyrk)(const char* uplo, const char* trans,
                        const BlasInt* height, const BlasInt* rank,
                        const float* alpha, const float* factor,
                        const BlasInt* factor_leading_dim, const float* beta,
                        float* matrix, const BlasInt* leading_dim);

void BLAS_SYMBOL(dtrsm)(const char* side, const char* uplo,
                        const char* trans_triang, const char* diag,
                        const BlasInt* height, const BlasInt* width,
                        const double* alpha, const double* triang_matrix,
                        const BlasInt* triang_leading_dim, double* matrix,
                        const BlasInt* leading_dim);

void BLAS_SYMBOL(strsm)(const char* side, const char* uplo,
                        const char* trans_triang, const char* diag,
                        const BlasInt* height, const BlasInt* width,
                        const float* alpha, const float* triang_matrix,
                        const BlasInt* triang_leading_dim, float* matrix,
                        const BlasInt* leading_dim);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

namespace catamari {

template <class T>
inline const T* ConstBlasMatrix<T>::Pointer(Int row, Int column) const {
  return &data[row + column * leading_dim];
}

template <class T>
inline const T& ConstBlasMatrix<T>::operator()(Int row, Int column) const {
  return data[row + column * leading_dim];
}

template <class T>
inline const T& ConstBlasMatrix<T>::Entry(Int row, Int column) const {
  return data[row + column * leading_dim];
}

template <class T>
inline ConstBlasMatrix<T> ConstBlasMatrix<T>::Submatrix(Int row_beg,
                                                        Int column_beg,
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
inline ConstBlasMatrix<T>::ConstBlasMatrix() {}

template <class T>
inline ConstBlasMatrix<T>::ConstBlasMatrix(const BlasMatrix<T>& matrix)
    : height(matrix.height),
      width(matrix.width),
      leading_dim(matrix.leading_dim),
      data(matrix.data) {}

template <class T>
inline ConstBlasMatrix<T>& ConstBlasMatrix<T>::operator=(
    const BlasMatrix<T>& matrix) {
  height = matrix.height;
  width = matrix.width;
  leading_dim = matrix.leading_dim;
  data = matrix.leading_dim;
}

template <class T>
inline T* BlasMatrix<T>::Pointer(Int row, Int column) {
  return &data[row + column * leading_dim];
}

template <class T>
inline const T* BlasMatrix<T>::Pointer(Int row, Int column) const {
  return &data[row + column * leading_dim];
}

template <class T>
inline T& BlasMatrix<T>::operator()(Int row, Int column) {
  return data[row + column * leading_dim];
}

template <class T>
inline const T& BlasMatrix<T>::operator()(Int row, Int column) const {
  return data[row + column * leading_dim];
}

template <class T>
inline T& BlasMatrix<T>::Entry(Int row, Int column) {
  return data[row + column * leading_dim];
}

template <class T>
inline const T& BlasMatrix<T>::Entry(Int row, Int column) const {
  return data[row + column * leading_dim];
}

template <class T>
inline BlasMatrix<T> BlasMatrix<T>::Submatrix(Int row_beg, Int column_beg,
                                              Int num_rows, Int num_columns) {
  BlasMatrix<T> submatrix;
  submatrix.height = num_rows;
  submatrix.width = num_columns;
  submatrix.leading_dim = leading_dim;
  submatrix.data = Pointer(row_beg, column_beg);
  return submatrix;
}

template <class T>
inline ConstBlasMatrix<T> BlasMatrix<T>::Submatrix(Int row_beg, Int column_beg,
                                                   Int num_rows,
                                                   Int num_columns) const {
  ConstBlasMatrix<T> submatrix;
  submatrix.height = num_rows;
  submatrix.width = num_columns;
  submatrix.leading_dim = leading_dim;
  submatrix.data = Pointer(row_beg, column_beg);
  return submatrix;
}

template <class Field>
void MatrixVectorProduct(
    const Field& alpha, const ConstBlasMatrix<Field>& matrix,
    const Field* input_vector, Field* result) {
  const Int height = matrix.height;
  const Int width = matrix.width;
  const Int leading_dim = matrix.leading_dim;
  for (Int j = 0; j < width; ++j) {
    for (Int i = 0; i < height; ++i) {
      result[i] += alpha * matrix(i, j) * input_vector[j];
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void MatrixVectorProduct(
    const float& alpha, const ConstBlasMatrix<float>& matrix,
    const float* input_vector, float* result) {
  const char trans = 'N';
  const BlasInt height_blas = matrix.height;
  const BlasInt width_blas = matrix.width;
  const BlasInt leading_dim_blas = matrix.leading_dim;
  const BlasInt unit_stride_blas = 1; 
  const float beta = 1;
  BLAS_SYMBOL(sgemv)(&trans, &height_blas, &width_blas, &alpha, matrix.data,
                     &leading_dim_blas, input_vector, &unit_stride_blas,
                     &beta, result, &unit_stride_blas);
}

template <>
inline void MatrixVectorProduct(
    const double& alpha, const ConstBlasMatrix<double>& matrix,
    const double* input_vector, double* result) {
  const char trans = 'N';
  const BlasInt height_blas = matrix.height;
  const BlasInt width_blas = matrix.width;
  const BlasInt leading_dim_blas = matrix.leading_dim;
  const BlasInt unit_stride_blas = 1; 
  const double beta = 1;
  BLAS_SYMBOL(dgemv)(&trans, &height_blas, &width_blas, &alpha, matrix.data,
                     &leading_dim_blas, input_vector, &unit_stride_blas,
                     &beta, result, &unit_stride_blas);
}
#endif

template <class Field>
void ConjugateMatrixVectorProduct(
    const Field& alpha, const ConstBlasMatrix<Field>& matrix,
    const Field* input_vector, Field* result) {
  const Int height = matrix.height;
  const Int width = matrix.width;
  const Int leading_dim = matrix.leading_dim;
  for (Int j = 0; j < width; ++j) {
    for (Int i = 0; i < height; ++i) {
      result[i] += alpha * Conjugate(matrix(i, j)) * input_vector[j];
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void ConjugateMatrixVectorProduct(
    const float& alpha, const ConstBlasMatrix<float>& matrix,
    const float* input_vector, float* result) {
  MatrixVectorProduct(alpha, matrix, input_vector, result);
}

template <>
inline void ConjugateMatrixVectorProduct(
    const double& alpha, const ConstBlasMatrix<double>& matrix,
    const double* input_vector, double* result) {
  MatrixVectorProduct(alpha, matrix, input_vector, result);
}
#endif

template <class Field>
void TransposeMatrixVectorProduct(
    const Field& alpha, const ConstBlasMatrix<Field>& matrix,
    const Field* input_vector, Field* result) {
  const Int height = matrix.height;
  const Int width = matrix.width;
  for (Int j = 0; j < width; ++j) {
    for (Int i = 0; i < height; ++i) {
      result[j] += alpha * matrix(i, j) * input_vector[i];
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void TransposeMatrixVectorProduct(
    const float& alpha, const ConstBlasMatrix<float>& matrix,
    const float* input_vector, float* result) {
  const char trans = 'T';
  const BlasInt height_blas = matrix.height;
  const BlasInt width_blas = matrix.width;
  const BlasInt leading_dim_blas = matrix.leading_dim;
  const BlasInt unit_stride_blas = 1; 
  const float beta = 1;
  BLAS_SYMBOL(sgemv)(&trans, &height_blas, &width_blas, &alpha, matrix.data,
                     &leading_dim_blas, input_vector, &unit_stride_blas,
                     &beta, result, &unit_stride_blas);
}

template <>
inline void TransposeMatrixVectorProduct(
    const double& alpha, const ConstBlasMatrix<double>& matrix,
    const double* input_vector, double* result) {
  const char trans = 'T';
  const BlasInt height_blas = matrix.height;
  const BlasInt width_blas = matrix.width;
  const BlasInt leading_dim_blas = matrix.leading_dim;
  const BlasInt unit_stride_blas = 1; 
  const double beta = 1;
  BLAS_SYMBOL(dgemv)(&trans, &height_blas, &width_blas, &alpha, matrix.data,
                     &leading_dim_blas, input_vector, &unit_stride_blas,
                     &beta, result, &unit_stride_blas);
}
#endif

template <class Field>
void TriangularSolveLeftLower(const ConstBlasMatrix<Field>& triangular_matrix,
                              Field* vector) {
  for (Int j = 0; j < triangular_matrix.height; ++j) {
    const Field* triang_column = triangular_matrix.Pointer(0, j);
    vector[j] /= triang_column[j];
    const Field eta = vector[j];
    for (Int i = j + 1; i < triangular_matrix.height; ++i) {
      vector[i] -= triang_column[i] * eta;
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void TriangularSolveLeftLower(
    const ConstBlasMatrix<float>& triangular_matrix, float* vector) {
  const char uplo = 'L';
  const char trans = 'N';
  const char diag = 'N';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(strsv)
  (&uplo, &trans, &diag, &height_blas, triangular_matrix.data,
   &triang_leading_dim_blas, vector, &unit_stride_blas);
}

template <>
inline void TriangularSolveLeftLower(
    const ConstBlasMatrix<double>& triangular_matrix, double* vector) {
  const char uplo = 'L';
  const char trans = 'N';
  const char diag = 'N';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(dtrsv)
  (&uplo, &trans, &diag, &height_blas, triangular_matrix.data,
   &triang_leading_dim_blas, vector, &unit_stride_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void TriangularSolveLeftLowerUnit(
    const ConstBlasMatrix<Field>& triangular_matrix, Field* vector) {
  for (Int j = 0; j < triangular_matrix.height; ++j) {
    const Field* triang_column = triangular_matrix.Pointer(0, j);
    const Field eta = vector[j];
    for (Int i = j + 1; i < triangular_matrix.height; ++i) {
      vector[i] -= triang_column[i] * eta;
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void TriangularSolveLeftLowerUnit(
    const ConstBlasMatrix<float>& triangular_matrix, float* vector) {
  const char uplo = 'L';
  const char trans = 'N';
  const char diag = 'U';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(strsv)
  (&uplo, &trans, &diag, &height_blas, triangular_matrix.data,
   &triang_leading_dim_blas, vector, &unit_stride_blas);
}

template <>
inline void TriangularSolveLeftLowerUnit(
    const ConstBlasMatrix<double>& triangular_matrix, double* vector) {
  const char uplo = 'L';
  const char trans = 'N';
  const char diag = 'U';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(dtrsv)
  (&uplo, &trans, &diag, &height_blas, triangular_matrix.data,
   &triang_leading_dim_blas, vector, &unit_stride_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void TriangularSolveLeftLowerAdjoint(
    const ConstBlasMatrix<Field>& triangular_matrix, Field* vector) {
  for (Int j = triangular_matrix.height - 1; j >= 0; --j) {
    const Field* triang_column = triangular_matrix.Pointer(0, j);
    Field& eta = vector[j];
    for (Int i = j + 1; i < triangular_matrix.height; ++i) {
      eta -= Conjugate(triang_column[i]) * vector[i];
    }
    eta /= Conjugate(triang_column[j]);
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void TriangularSolveLeftLowerAdjoint(
    const ConstBlasMatrix<float>& triangular_matrix, float* vector) {
  const char uplo = 'L';
  const char trans = 'T';
  const char diag = 'N';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(strsv)
  (&uplo, &trans, &diag, &height_blas, triangular_matrix.data,
   &triang_leading_dim_blas, vector, &unit_stride_blas);
}

template <>
inline void TriangularSolveLeftLowerAdjoint(
    const ConstBlasMatrix<double>& triangular_matrix, double* vector) {
  const char uplo = 'L';
  const char trans = 'T';
  const char diag = 'N';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(dtrsv)
  (&uplo, &trans, &diag, &height_blas, triangular_matrix.data,
   &triang_leading_dim_blas, vector, &unit_stride_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void TriangularSolveLeftLowerAdjointUnit(
    const ConstBlasMatrix<Field>& triangular_matrix, Field* vector) {
  for (Int j = triangular_matrix.height - 1; j >= 0; --j) {
    const Field* triang_column = triangular_matrix.Pointer(0, j);
    Field& eta = vector[j];
    for (Int i = j + 1; i < triangular_matrix.height; ++i) {
      eta -= Conjugate(triang_column[i]) * vector[i];
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void TriangularSolveLeftLowerAdjointUnit(
    const ConstBlasMatrix<float>& triangular_matrix, float* vector) {
  const char uplo = 'L';
  const char trans = 'T';
  const char diag = 'U';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(strsv)
  (&uplo, &trans, &diag, &height_blas, triangular_matrix.data,
   &triang_leading_dim_blas, vector, &unit_stride_blas);
}

template <>
inline void TriangularSolveLeftLowerAdjointUnit(
    const ConstBlasMatrix<double>& triangular_matrix, double* vector) {
  const char uplo = 'L';
  const char trans = 'T';
  const char diag = 'U';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(dtrsv)
  (&uplo, &trans, &diag, &height_blas, triangular_matrix.data,
   &triang_leading_dim_blas, vector, &unit_stride_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void MatrixMultiplyTransposeNormal(const Field& alpha,
                                   const ConstBlasMatrix<Field>& left_matrix,
                                   const ConstBlasMatrix<Field>& right_matrix,
                                   const Field& beta,
                                   BlasMatrix<Field>* output_matrix) {
  const Int output_height = output_matrix->height;
  const Int output_width = output_matrix->width;
  const Int contraction_size = left_matrix.height;
  for (Int j = 0; j < output_width; ++j) {
    for (Int i = 0; i < output_height; ++i) {
      Field& output_entry = output_matrix->Entry(i, j);
      output_entry *= beta;
      for (Int k = 0; k < contraction_size; ++k) {
        output_entry += alpha * left_matrix(k, i) * right_matrix(k, j);
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void MatrixMultiplyTransposeNormal(
    const float& alpha, const ConstBlasMatrix<float>& left_matrix,
    const ConstBlasMatrix<float>& right_matrix, const float& beta,
    BlasMatrix<float>* output_matrix) {
  const char trans_left = 'T';
  const char trans_right = 'N';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt output_width_blas = output_matrix->width;
  const BlasInt contraction_size_blas = left_matrix.height;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(sgemm)
  (&trans_left, &trans_right, &output_height_blas, &output_width_blas,
   &contraction_size_blas, &alpha, left_matrix.data, &left_leading_dim_blas,
   right_matrix.data, &right_leading_dim_blas, &beta, output_matrix->data,
   &output_leading_dim_blas);
}

template <>
inline void MatrixMultiplyTransposeNormal(
    const double& alpha, const ConstBlasMatrix<double>& left_matrix,
    const ConstBlasMatrix<double>& right_matrix, const double& beta,
    BlasMatrix<double>* output_matrix) {
  const char trans_left = 'T';
  const char trans_right = 'N';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt output_width_blas = output_matrix->width;
  const BlasInt contraction_size_blas = left_matrix.height;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(dgemm)
  (&trans_left, &trans_right, &output_height_blas, &output_width_blas,
   &contraction_size_blas, &alpha, left_matrix.data, &left_leading_dim_blas,
   right_matrix.data, &right_leading_dim_blas, &beta, output_matrix->data,
   &output_leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void LowerTransposeHermitianOuterProduct(
    const Field& alpha, const ConstBlasMatrix<Field>& left_matrix,
    const Field& beta, BlasMatrix<Field>* output_matrix) {
  const Int output_height = output_matrix->height;
  const Int output_width = output_matrix->width;
  const Int contraction_size = left_matrix.height;
  for (Int j = 0; j < output_height; ++j) {
    for (Int i = j; i < output_height; ++i) {
      Field& output_entry = output_matrix->Entry(i, j);
      output_entry *= beta;
      for (Int k = 0; k < contraction_size; ++k) {
        output_entry +=
            alpha * left_matrix(k, i) * Conjugate(left_matrix(k, j));
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void LowerTransposeHermitianOuterProduct(
    const float& alpha, const ConstBlasMatrix<float>& left_matrix,
    const float& beta, BlasMatrix<float>* output_matrix) {
  const char uplo = 'L';
  const char trans = 'T';
  const BlasInt height_blas = output_matrix->height;
  const BlasInt rank_blas = left_matrix.height;
  const BlasInt factor_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(ssyrk)
  (&uplo, &trans, &height_blas, &rank_blas, &alpha, left_matrix.data,
   &factor_leading_dim_blas, &beta, output_matrix->data, &leading_dim_blas);
}

template <>
inline void LowerTransposeHermitianOuterProduct(
    const double& alpha, const ConstBlasMatrix<double>& left_matrix,
    const double& beta, BlasMatrix<double>* output_matrix) {
  const char uplo = 'L';
  const char trans = 'T';
  const BlasInt height_blas = output_matrix->height;
  const BlasInt rank_blas = left_matrix.height;
  const BlasInt factor_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(dsyrk)
  (&uplo, &trans, &height_blas, &rank_blas, &alpha, left_matrix.data,
   &factor_leading_dim_blas, &beta, output_matrix->data, &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void LowerNormalHermitianOuterProduct(
    const Field& alpha, const ConstBlasMatrix<Field>& left_matrix,
    const ConstBlasMatrix<Field>& right_matrix, const Field& beta,
    BlasMatrix<Field>* output_matrix) {
  const Int output_height = output_matrix->height;
  const Int contraction_size = left_matrix.width;
  for (Int j = 0; j < output_height; ++j) {
    for (Int i = j; i < output_height; ++i) {
      Field& output_entry = output_matrix->Entry(i, j);
      output_entry *= beta;
      for (Int k = 0; k < contraction_size; ++k) {
        output_entry +=
            alpha * left_matrix(i, k) * Conjugate(right_matrix(j, k));
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void LowerNormalHermitianOuterProduct(
    const float& alpha, const ConstBlasMatrix<float>& left_matrix,
    const ConstBlasMatrix<float>& right_matrix, const float& beta,
    BlasMatrix<float>* output_matrix) {
  const char uplo = 'L';
  const char trans = 'N';
  const BlasInt height_blas = output_matrix->height;
  const BlasInt rank_blas = left_matrix.width;
  const BlasInt factor_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(ssyrk)
  (&uplo, &trans, &height_blas, &rank_blas, &alpha, left_matrix.data,
   &factor_leading_dim_blas, &beta, output_matrix->data, &leading_dim_blas);
}

template <>
inline void LowerNormalHermitianOuterProduct(
    const double& alpha, const ConstBlasMatrix<double>& left_matrix,
    const ConstBlasMatrix<double>& right_matrix, const double& beta,
    BlasMatrix<double>* output_matrix) {
  const char uplo = 'L';
  const char trans = 'N';
  const BlasInt height_blas = output_matrix->height;
  const BlasInt rank_blas = left_matrix.width;
  const BlasInt factor_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(dsyrk)
  (&uplo, &trans, &height_blas, &rank_blas, &alpha, left_matrix.data,
   &factor_leading_dim_blas, &beta, output_matrix->data, &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void MatrixMultiplyLowerNormalTranspose(
    const Field& alpha, const ConstBlasMatrix<Field>& left_matrix,
    const ConstBlasMatrix<Field>& right_matrix, const Field& beta,
    BlasMatrix<Field>* output_matrix) {
  const Int output_height = output_matrix->height;
  const Int contraction_size = left_matrix.height;
  for (Int j = 0; j < output_height; ++j) {
    for (Int i = j; i < output_height; ++i) {
      Field& output_entry = output_matrix->Entry(i, j);
      output_entry *= beta;
      for (Int k = 0; k < contraction_size; ++k) {
        output_entry += alpha * left_matrix(i, k) * right_matrix(j, k);
      }
    }
  }
}

#ifdef CATAMARI_USE_GEMMT
template <>
inline void MatrixMultiplyLowerNormalTranspose(
    const float& alpha, const ConstBlasMatrix<float>& left_matrix,
    const ConstBlasMatrix<float>& right_matrix, const float& beta,
    BlasMatrix<float>* output_matrix) {
  const char uplo = 'L';
  const char trans_left = 'N';
  const char trans_right = 'T';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt contraction_size_blas = left_matrix.width;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(sgemmt)
  (&uplo, &trans_left, &trans_right, &output_height_blas,
   &contraction_size_blas, &alpha, left_matrix.data, &left_leading_dim_blas,
   right_matrix.data, &right_leading_dim_blas, &beta, output_matrix->data,
   &output_leading_dim_blas);
}

template <>
inline void MatrixMultiplyLowerNormalTranspose(
    const double& alpha, const ConstBlasMatrix<double>& left_matrix,
    const ConstBlasMatrix<double>& right_matrix, const double& beta,
    BlasMatrix<double>* output_matrix) {
  const char uplo = 'L';
  const char trans_left = 'N';
  const char trans_right = 'T';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt contraction_size_blas = left_matrix.width;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(dgemmt)
  (&uplo, &trans_left, &trans_right, &output_height_blas,
   &contraction_size_blas, &alpha, left_matrix.data, &left_leading_dim_blas,
   right_matrix.data, &right_leading_dim_blas, &beta, output_matrix->data,
   &output_leading_dim_blas);
}
#elif defined(CATAMARI_HAVE_BLAS)
template <>
inline void MatrixMultiplyLowerNormalTranspose(
    const float& alpha, const ConstBlasMatrix<float>& left_matrix,
    const ConstBlasMatrix<float>& right_matrix, const float& beta,
    BlasMatrix<float>* output_matrix) {
  MatrixMultiplyNormalTranspose(alpha, left_matrix, right_matrix, beta,
                                output_matrix);

  // Explicitly zero out the strictly-upper triangle of C.
  const Int output_height = output_matrix->height;
  for (Int j = 1; j < output_height; ++j) {
    for (Int i = 0; i < j; ++i) {
      output_matrix->Entry(i, j) = 0;
    }
  }
}

template <>
inline void MatrixMultiplyLowerNormalTranspose(
    const double& alpha, const ConstBlasMatrix<double>& left_matrix,
    const ConstBlasMatrix<double>& right_matrix, const double& beta,
    BlasMatrix<double>* output_matrix) {
  MatrixMultiplyNormalTranspose(alpha, left_matrix, right_matrix, beta,
                                output_matrix);

  // Explicitly zero out the strictly-upper triangle of C.
  const Int output_height = output_matrix->height;
  for (Int j = 1; j < output_height; ++j) {
    for (Int i = 0; i < j; ++i) {
      output_matrix->Entry(i, j) = 0;
    }
  }
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void MatrixMultiplyLowerTransposeNormal(
    const Field& alpha, const ConstBlasMatrix<Field>& left_matrix,
    const ConstBlasMatrix<Field>& right_matrix, const Field& beta,
    BlasMatrix<Field>* output_matrix) {
  const Int output_height = output_matrix->height;
  const Int contraction_size = left_matrix.height;
  for (Int j = 0; j < output_height; ++j) {
    for (Int i = j; i < output_height; ++i) {
      Field& output_entry = output_matrix->Entry(i, j);
      output_entry *= beta;
      for (Int k = 0; k < contraction_size; ++k) {
        output_entry += alpha * left_matrix(k, i) * right_matrix(k, j);
      }
    }
  }
}

#ifdef CATAMARI_USE_GEMMT
template <>
inline void MatrixMultiplyLowerTransposeNormal(
    const float& alpha, const ConstBlasMatrix<float>& left_matrix,
    const ConstBlasMatrix<float>& right_matrix, const float& beta,
    BlasMatrix<float>* output_matrix) {
  const char uplo = 'L';
  const char trans_left = 'T';
  const char trans_right = 'N';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt contraction_size_blas = left_matrix.height;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(sgemmt)
  (&uplo, &trans_left, &trans_right, &output_height_blas,
   &contraction_size_blas, &alpha, left_matrix.data, &left_leading_dim_blas,
   right_matrix.data, &right_leading_dim_blas, &beta, output_matrix->data,
   &output_leading_dim_blas);
}

template <>
inline void MatrixMultiplyLowerTransposeNormal(
    const double& alpha, const ConstBlasMatrix<double>& left_matrix,
    const ConstBlasMatrix<double>& right_matrix, const double& beta,
    BlasMatrix<double>* output_matrix) {
  const char uplo = 'L';
  const char trans_left = 'T';
  const char trans_right = 'N';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt contraction_size_blas = left_matrix.height;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(dgemmt)
  (&uplo, &trans_left, &trans_right, &output_height_blas,
   &contraction_size_blas, &alpha, left_matrix.data, &left_leading_dim_blas,
   right_matrix.data, &right_leading_dim_blas, &beta, output_matrix->data,
   &output_leading_dim_blas);
}
#elif defined(CATAMARI_HAVE_BLAS)
template <>
inline void MatrixMultiplyLowerTransposeNormal(
    const float& alpha, const ConstBlasMatrix<float>& left_matrix,
    const ConstBlasMatrix<float>& right_matrix, const float& beta,
    BlasMatrix<float>* output_matrix) {
  MatrixMultiplyTransposeNormal(alpha, left_matrix, right_matrix, beta,
                                output_matrix);

  // Explicitly zero out the strictly-upper triangle of C.
  const Int output_height = output_matrix->height;
  for (Int j = 1; j < output_height; ++j) {
    for (Int i = 0; i < j; ++i) {
      output_matrix->Entry(i, j) = 0;
    }
  }
}

template <>
inline void MatrixMultiplyLowerTransposeNormal(
    const float& alpha, const ConstBlasMatrix<float>& left_matrix,
    const ConstBlasMatrix<float>& right_matrix, const float& beta,
    BlasMatrix<float>* output_matrix) {
  MatrixMultiplyTransposeNormal(alpha, left_matrix, right_matrix, beta,
                                output_matrix);

  // Explicitly zero out the strictly-upper triangle of C.
  const Int output_height = output_matrix->height;
  for (Int j = 1; j < output_height; ++j) {
    for (Int i = 0; i < j; ++i) {
      output_matrix->Entry(i, j) = 0;
    }
  }
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void LeftLowerConjugateTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix,
    BlasMatrix<Field>* matrix) {
  const Int width = matrix->width;
  for (Int j = 0; j < width; ++j) {
    Field* input_column = matrix->Pointer(0, j);

    // Interleave with a subsequent solve against D.
    for (Int i = 0; i < triangular_matrix.height; ++i) {
      const Field* l_column = triangular_matrix.Pointer(0, i);
      input_column[i] /= Conjugate(l_column[i]);
      for (Int k = i + 1; k < triangular_matrix.height; ++k) {
        input_column[k] -= Conjugate(l_column[k]) * input_column[i];
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void LeftLowerConjugateTriangularSolves(
    const ConstBlasMatrix<float>& triangular_matrix,
    BlasMatrix<float>* matrix) {
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'N';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const float alpha = 1.f;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(strsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void LeftLowerConjugateTriangularSolves(
    const ConstBlasMatrix<double>& triangular_matrix,
    BlasMatrix<double>* matrix) {
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'N';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const double alpha = 1.f;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(dtrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void DiagonalTimesLeftLowerConjugateUnitTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix,
    BlasMatrix<Field>* matrix) {
  const Int width = matrix->width;
  for (Int j = 0; j < width; ++j) {
    Field* input_column = matrix->Pointer(0, j);

    // Interleave with a subsequent solve against D.
    for (Int i = 0; i < triangular_matrix.height; ++i) {
      const Field* l_column = triangular_matrix.Pointer(0, i);
      for (Int k = i + 1; k < triangular_matrix.height; ++k) {
        input_column[k] -= Conjugate(l_column[k]) * input_column[i];
      }
      input_column[i] /= l_column[i];
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void DiagonalTimesLeftLowerConjugateUnitTriangularSolves(
    const ConstBlasMatrix<float>& triangular_matrix,
    BlasMatrix<float>* matrix) {
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'N';
  const char diag = 'U';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const float alpha = 1.f;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(strsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);

  // Solve against the diagonal.
  for (BlasInt j = 0; j < width_blas; ++j) {
    for (BlasInt i = 0; i < height_blas; ++i) {
      matrix->Entry(i, j) /= triangular_matrix(i, i);
    }
  }
}

template <>
inline void DiagonalTimesLeftLowerConjugateUnitTriangularSolves(
    const ConstBlasMatrix<double>& triangular_matrix,
    BlasMatrix<double>* matrix) {
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'N';
  const char diag = 'U';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const double alpha = 1.;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(dtrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);

  // Solve against the diagonal.
  for (BlasInt j = 0; j < width_blas; ++j) {
    for (BlasInt i = 0; i < height_blas; ++i) {
      matrix->Entry(i, j) /= triangular_matrix(i, i);
    }
  }
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void RightLowerAdjointTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix,
    BlasMatrix<Field>* matrix) {
  const Int height = matrix->height;
  const Int width = matrix->width;
  for (Int i = 0; i < height; ++i) {
    for (Int j = 0; j < width; ++j) {
      matrix->Entry(i, j) /= RealPart(triangular_matrix(j, j));
      const Field eta = matrix->Entry(i, j);
      for (Int k = j + 1; k < width; ++k) {
        matrix->Entry(i, k) -= eta * Conjugate(triangular_matrix(k, j));
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void RightLowerAdjointTriangularSolves(
    const ConstBlasMatrix<float>& triangular_matrix,
    BlasMatrix<float>* matrix) {
  const char side = 'R';
  const char uplo = 'L';
  const char trans_triang = 'T';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const float alpha = 1.f;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BLAS_SYMBOL(strsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void RightLowerAdjointTriangularSolves(
    const ConstBlasMatrix<double>& triangular_matrix,
    BlasMatrix<double>* matrix) {
  const char side = 'R';
  const char uplo = 'L';
  const char trans_triang = 'T';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const double alpha = 1.f;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BLAS_SYMBOL(dtrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void RightLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix,
    BlasMatrix<Field>* matrix) {
  const Int height = matrix->height;
  const Int width = matrix->width;
  for (Int i = 0; i < height; ++i) {
    for (Int j = 0; j < width; ++j) {
      const Field eta = matrix->Entry(i, j);
      for (Int k = j + 1; k < width; ++k) {
        matrix->Entry(i, k) -= eta * Conjugate(triangular_matrix(k, j));
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void RightLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrix<float>& triangular_matrix,
    BlasMatrix<float>* matrix) {
  const char side = 'R';
  const char uplo = 'L';
  const char trans_triang = 'T';
  const char diag = 'U';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const float alpha = 1.f;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BLAS_SYMBOL(strsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void RightLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrix<double>& triangular_matrix,
    BlasMatrix<double>* matrix) {
  const char side = 'R';
  const char uplo = 'L';
  const char trans_triang = 'T';
  const char diag = 'U';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const double alpha = 1.f;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BLAS_SYMBOL(dtrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

}  // namespace catamari

#endif  // ifndef CATAMARI_BLAS_IMPL_H_
