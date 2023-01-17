/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_BASIC_LINEAR_ALGEBRA_IMPL_H_
#define CATAMARI_DENSE_BASIC_LINEAR_ALGEBRA_IMPL_H_

#include <stdexcept>
#include "catamari/blas.hpp"
#include "catamari/macros.hpp"

#include "catamari/dense_basic_linear_algebra.hpp"
#include "../../../../../src/lib/MeshFEM/Parallelism.hh"

namespace catamari {

template <class Field>
void ConjugateMatrix(BlasMatrixView<Field>* matrix) {
  const Int height = matrix->height;
  const Int width = matrix->width;
  for (Int j = 0; j < width; ++j) {
    for (Int i = 0; i < height; ++i) {
      matrix->Entry(i, j) = Conjugate(matrix->Entry(i, j));
    }
  }
}

template <class Field>
void MatrixVectorProduct(const Field& alpha,
                         const ConstBlasMatrixView<Field>& matrix,
                         const Field* input_vector, Field* result) {
  const Int height = matrix.height;
  const Int width = matrix.width;
  for (Int j = 0; j < width; ++j) {
    for (Int i = 0; i < height; ++i) {
      result[i] += alpha * matrix(i, j) * input_vector[j];
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void MatrixVectorProduct(const float& alpha,
                                const ConstBlasMatrixView<float>& matrix,
                                const float* input_vector, float* result) {
  const char trans = 'N';
  const BlasInt height_blas = matrix.height;
  const BlasInt width_blas = matrix.width;
  const BlasInt leading_dim_blas = matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  const float beta = 1;
  BLAS_SYMBOL(sgemv)
  (&trans, &height_blas, &width_blas, &alpha, matrix.data, &leading_dim_blas,
   input_vector, &unit_stride_blas, &beta, result, &unit_stride_blas);
}

template <>
inline void MatrixVectorProduct(const double& alpha,
                                const ConstBlasMatrixView<double>& matrix,
                                const double* input_vector, double* result) {
  const char trans = 'N';
  const BlasInt height_blas = matrix.height;
  const BlasInt width_blas = matrix.width;
  const BlasInt leading_dim_blas = matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  const double beta = 1;
  BLAS_SYMBOL(dgemv)
  (&trans, &height_blas, &width_blas, &alpha, matrix.data, &leading_dim_blas,
   input_vector, &unit_stride_blas, &beta, result, &unit_stride_blas);
}

template <>
inline void MatrixVectorProduct(
    const Complex<float>& alpha,
    const ConstBlasMatrixView<Complex<float>>& matrix,
    const Complex<float>* input_vector, Complex<float>* result) {
  const char trans = 'N';
  const BlasInt height_blas = matrix.height;
  const BlasInt width_blas = matrix.width;
  const BlasInt leading_dim_blas = matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  const BlasComplexFloat beta = 1;
  BLAS_SYMBOL(cgemv)
  (&trans, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexFloat*>(&alpha),
   reinterpret_cast<const BlasComplexFloat*>(matrix.data), &leading_dim_blas,
   reinterpret_cast<const BlasComplexFloat*>(input_vector), &unit_stride_blas,
   reinterpret_cast<const BlasComplexFloat*>(&beta),
   reinterpret_cast<BlasComplexFloat*>(result), &unit_stride_blas);
}

template <>
inline void MatrixVectorProduct(
    const Complex<double>& alpha,
    const ConstBlasMatrixView<Complex<double>>& matrix,
    const Complex<double>* input_vector, Complex<double>* result) {
  const char trans = 'N';
  const BlasInt height_blas = matrix.height;
  const BlasInt width_blas = matrix.width;
  const BlasInt leading_dim_blas = matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  const BlasComplexDouble beta = 1;
  BLAS_SYMBOL(zgemv)
  (&trans, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexDouble*>(&alpha),
   reinterpret_cast<const BlasComplexDouble*>(matrix.data), &leading_dim_blas,
   reinterpret_cast<const BlasComplexDouble*>(input_vector), &unit_stride_blas,
   reinterpret_cast<const BlasComplexDouble*>(&beta),
   reinterpret_cast<BlasComplexDouble*>(result), &unit_stride_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void ConjugateMatrixVectorProduct(const Field& alpha,
                                  const ConstBlasMatrixView<Field>& matrix,
                                  const Field* input_vector, Field* result) {
  const Int height = matrix.height;
  const Int width = matrix.width;
  for (Int j = 0; j < width; ++j) {
    for (Int i = 0; i < height; ++i) {
      result[i] += alpha * Conjugate(matrix(i, j)) * input_vector[j];
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void ConjugateMatrixVectorProduct(
    const float& alpha, const ConstBlasMatrixView<float>& matrix,
    const float* input_vector, float* result) {
  MatrixVectorProduct(alpha, matrix, input_vector, result);
}

template <>
inline void ConjugateMatrixVectorProduct(
    const double& alpha, const ConstBlasMatrixView<double>& matrix,
    const double* input_vector, double* result) {
  MatrixVectorProduct(alpha, matrix, input_vector, result);
}

// Make use of the equivalence:
//
//   alpha conj(A) x + y = conj(conj(alpha) A conj(x) + conj(y))
//
// in order to make use of GEMV for complex inputs.

template <>
inline void ConjugateMatrixVectorProduct(
    const Complex<float>& alpha,
    const ConstBlasMatrixView<Complex<float>>& matrix,
    const Complex<float>* input_vector, Complex<float>* result) {
  // Make a conjugated copy of the input vector.
  Buffer<Complex<float>> conjugated_input(matrix.width);
  for (Int i = 0; i < matrix.width; ++i) {
    conjugated_input[i] = Conjugate(input_vector[i]);
  }

  BlasMatrixView<Complex<float>> result_matrix;
  result_matrix.height = matrix.height;
  result_matrix.width = 1;
  result_matrix.leading_dim = matrix.height;
  result_matrix.data = result;

  ConjugateMatrix(&result_matrix);
  MatrixVectorProduct(Conjugate(alpha), matrix, conjugated_input.Data(),
                      result);
  ConjugateMatrix(&result_matrix);
}

template <>
inline void ConjugateMatrixVectorProduct(
    const Complex<double>& alpha,
    const ConstBlasMatrixView<Complex<double>>& matrix,
    const Complex<double>* input_vector, Complex<double>* result) {
  // Make a conjugated copy of the input vector.
  Buffer<Complex<double>> conjugated_input(matrix.width);
  for (Int i = 0; i < matrix.width; ++i) {
    conjugated_input[i] = Conjugate(input_vector[i]);
  }

  BlasMatrixView<Complex<double>> result_matrix;
  result_matrix.height = matrix.height;
  result_matrix.width = 1;
  result_matrix.leading_dim = matrix.height;
  result_matrix.data = result;

  ConjugateMatrix(&result_matrix);
  MatrixVectorProduct(Conjugate(alpha), matrix, conjugated_input.Data(),
                      result);
  ConjugateMatrix(&result_matrix);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void TransposeMatrixVectorProduct(const Field& alpha,
                                  const ConstBlasMatrixView<Field>& matrix,
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
    const float& alpha, const ConstBlasMatrixView<float>& matrix,
    const float* input_vector, float* result) {
  const char trans = 'T';
  const BlasInt height_blas = matrix.height;
  const BlasInt width_blas = matrix.width;
  const BlasInt leading_dim_blas = matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  const float beta = 1;
  BLAS_SYMBOL(sgemv)
  (&trans, &height_blas, &width_blas, &alpha, matrix.data, &leading_dim_blas,
   input_vector, &unit_stride_blas, &beta, result, &unit_stride_blas);
}

template <>
inline void TransposeMatrixVectorProduct(
    const double& alpha, const ConstBlasMatrixView<double>& matrix,
    const double* input_vector, double* result) {
  const char trans = 'T';
  const BlasInt height_blas = matrix.height;
  const BlasInt width_blas = matrix.width;
  const BlasInt leading_dim_blas = matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  const double beta = 1;
  BLAS_SYMBOL(dgemv)
  (&trans, &height_blas, &width_blas, &alpha, matrix.data, &leading_dim_blas,
   input_vector, &unit_stride_blas, &beta, result, &unit_stride_blas);
}

template <>
inline void TransposeMatrixVectorProduct(
    const Complex<float>& alpha,
    const ConstBlasMatrixView<Complex<float>>& matrix,
    const Complex<float>* input_vector, Complex<float>* result) {
  const char trans = 'T';
  const BlasInt height_blas = matrix.height;
  const BlasInt width_blas = matrix.width;
  const BlasInt leading_dim_blas = matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  const Complex<float> beta = 1;
  BLAS_SYMBOL(cgemv)
  (&trans, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexFloat*>(&alpha),
   reinterpret_cast<const BlasComplexFloat*>(matrix.data), &leading_dim_blas,
   reinterpret_cast<const BlasComplexFloat*>(input_vector), &unit_stride_blas,
   reinterpret_cast<const BlasComplexFloat*>(&beta),
   reinterpret_cast<BlasComplexFloat*>(result), &unit_stride_blas);
}

template <>
inline void TransposeMatrixVectorProduct(
    const Complex<double>& alpha,
    const ConstBlasMatrixView<Complex<double>>& matrix,
    const Complex<double>* input_vector, Complex<double>* result) {
  const char trans = 'T';
  const BlasInt height_blas = matrix.height;
  const BlasInt width_blas = matrix.width;
  const BlasInt leading_dim_blas = matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  const Complex<double> beta = 1;
  BLAS_SYMBOL(zgemv)
  (&trans, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexDouble*>(&alpha),
   reinterpret_cast<const BlasComplexDouble*>(matrix.data), &leading_dim_blas,
   reinterpret_cast<const BlasComplexDouble*>(input_vector), &unit_stride_blas,
   reinterpret_cast<const BlasComplexDouble*>(&beta),
   reinterpret_cast<BlasComplexDouble*>(result), &unit_stride_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void TriangularSolveLeftLower(
    const ConstBlasMatrixView<Field>& triangular_matrix, Field* vector) {
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
    const ConstBlasMatrixView<float>& triangular_matrix, float* vector) {
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
    const ConstBlasMatrixView<double>& triangular_matrix, double* vector) {
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

template <>
inline void TriangularSolveLeftLower(
    const ConstBlasMatrixView<Complex<float>>& triangular_matrix,
    Complex<float>* vector) {
  const char uplo = 'L';
  const char trans = 'N';
  const char diag = 'N';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(ctrsv)
  (&uplo, &trans, &diag, &height_blas,
   reinterpret_cast<const BlasComplexFloat*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexFloat*>(vector),
   &unit_stride_blas);
}

template <>
inline void TriangularSolveLeftLower(
    const ConstBlasMatrixView<Complex<double>>& triangular_matrix,
    Complex<double>* vector) {
  const char uplo = 'L';
  const char trans = 'N';
  const char diag = 'N';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(ztrsv)
  (&uplo, &trans, &diag, &height_blas,
   reinterpret_cast<const BlasComplexDouble*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexDouble*>(vector),
   &unit_stride_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void TriangularSolveLeftLowerUnit(
    const ConstBlasMatrixView<Field>& triangular_matrix, Field* vector) {
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
    const ConstBlasMatrixView<float>& triangular_matrix, float* vector) {
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
    const ConstBlasMatrixView<double>& triangular_matrix, double* vector) {
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

template <>
inline void TriangularSolveLeftLowerUnit(
    const ConstBlasMatrixView<Complex<float>>& triangular_matrix,
    Complex<float>* vector) {
  const char uplo = 'L';
  const char trans = 'N';
  const char diag = 'U';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(ctrsv)
  (&uplo, &trans, &diag, &height_blas,
   reinterpret_cast<const BlasComplexFloat*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexFloat*>(vector),
   &unit_stride_blas);
}

template <>
inline void TriangularSolveLeftLowerUnit(
    const ConstBlasMatrixView<Complex<double>>& triangular_matrix,
    Complex<double>* vector) {
  const char uplo = 'L';
  const char trans = 'N';
  const char diag = 'U';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(ztrsv)
  (&uplo, &trans, &diag, &height_blas,
   reinterpret_cast<const BlasComplexDouble*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexDouble*>(vector),
   &unit_stride_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void TriangularSolveLeftLowerAdjoint(
    const ConstBlasMatrixView<Field>& triangular_matrix, Field* vector) {
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
    const ConstBlasMatrixView<float>& triangular_matrix, float* vector) {
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
    const ConstBlasMatrixView<double>& triangular_matrix, double* vector) {
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

template <>
inline void TriangularSolveLeftLowerAdjoint(
    const ConstBlasMatrixView<Complex<float>>& triangular_matrix,
    Complex<float>* vector) {
  const char uplo = 'L';
  const char trans = 'C';
  const char diag = 'N';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(ctrsv)
  (&uplo, &trans, &diag, &height_blas,
   reinterpret_cast<const BlasComplexFloat*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexFloat*>(vector),
   &unit_stride_blas);
}

template <>
inline void TriangularSolveLeftLowerAdjoint(
    const ConstBlasMatrixView<Complex<double>>& triangular_matrix,
    Complex<double>* vector) {
  const char uplo = 'L';
  const char trans = 'C';
  const char diag = 'N';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(ztrsv)
  (&uplo, &trans, &diag, &height_blas,
   reinterpret_cast<const BlasComplexDouble*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexDouble*>(vector),
   &unit_stride_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void TriangularSolveLeftLowerAdjointUnit(
    const ConstBlasMatrixView<Field>& triangular_matrix, Field* vector) {
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
    const ConstBlasMatrixView<float>& triangular_matrix, float* vector) {
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
    const ConstBlasMatrixView<double>& triangular_matrix, double* vector) {
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

template <>
inline void TriangularSolveLeftLowerAdjointUnit(
    const ConstBlasMatrixView<Complex<float>>& triangular_matrix,
    Complex<float>* vector) {
  const char uplo = 'L';
  const char trans = 'C';
  const char diag = 'U';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(ctrsv)
  (&uplo, &trans, &diag, &height_blas,
   reinterpret_cast<const BlasComplexFloat*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexFloat*>(vector),
   &unit_stride_blas);
}

template <>
inline void TriangularSolveLeftLowerAdjointUnit(
    const ConstBlasMatrixView<Complex<double>>& triangular_matrix,
    Complex<double>* vector) {
  const char uplo = 'L';
  const char trans = 'C';
  const char diag = 'U';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(ztrsv)
  (&uplo, &trans, &diag, &height_blas,
   reinterpret_cast<const BlasComplexDouble*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexDouble*>(vector),
   &unit_stride_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void TriangularSolveLeftUpper(
    const ConstBlasMatrixView<Field>& triangular_matrix, Field* vector) {
  for (Int j = triangular_matrix.height - 1; j >= 0; --j) {
    const Field* triangular_column = triangular_matrix.Pointer(0, j);
    Field& eta = vector[j];
    eta /= triangular_column[j];
    for (Int i = 0; i < j; ++i) {
      vector[i] -= eta * triangular_column[i];
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void TriangularSolveLeftUpper(
    const ConstBlasMatrixView<float>& triangular_matrix, float* vector) {
  const char uplo = 'U';
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
inline void TriangularSolveLeftUpper(
    const ConstBlasMatrixView<double>& triangular_matrix, double* vector) {
  const char uplo = 'U';
  const char trans = 'N';
  const char diag = 'N';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(dtrsv)
  (&uplo, &trans, &diag, &height_blas, triangular_matrix.data,
   &triang_leading_dim_blas, vector, &unit_stride_blas);
}

template <>
inline void TriangularSolveLeftUpper(
    const ConstBlasMatrixView<Complex<float>>& triangular_matrix,
    Complex<float>* vector) {
  const char uplo = 'U';
  const char trans = 'N';
  const char diag = 'N';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(ctrsv)
  (&uplo, &trans, &diag, &height_blas,
   reinterpret_cast<const BlasComplexFloat*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexFloat*>(vector),
   &unit_stride_blas);
}

template <>
inline void TriangularSolveLeftUpper(
    const ConstBlasMatrixView<Complex<double>>& triangular_matrix,
    Complex<double>* vector) {
  const char uplo = 'U';
  const char trans = 'N';
  const char diag = 'N';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(ztrsv)
  (&uplo, &trans, &diag, &height_blas,
   reinterpret_cast<const BlasComplexDouble*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexDouble*>(vector),
   &unit_stride_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void MatrixMultiplyNormalNormalDynamicBLASDispatch(const Field& alpha,
                                const ConstBlasMatrixView<Field>& left_matrix,
                                const ConstBlasMatrixView<Field>& right_matrix,
                                const Field& beta,
                                BlasMatrixView<Field>* output_matrix) {
#ifdef CATAMARI_HAVE_BLAS
    if (left_matrix.height > 15) // only use BLAS call for large enough matrices
        return MatrixMultiplyNormalNormal(alpha, left_matrix, right_matrix, beta, output_matrix);
#endif
    CATAMARI_ASSERT( left_matrix.height == output_matrix->height, "Output height was incompatible");
    CATAMARI_ASSERT(right_matrix. width == output_matrix->width,  "Output width was incompatible");
    CATAMARI_ASSERT( left_matrix. width ==  right_matrix.height,  "Contraction dimensions were incompatible.");
    const Int output_height    = output_matrix->height;
    const Int output_width     = output_matrix->width;
    const Int contraction_size = left_matrix.width;
    if (alpha == Field(-1)) {
        for (Int j = 0; j < output_width; ++j) {
            Field *out_col = output_matrix->Pointer(0, j);
            const Field *right_col = right_matrix.Pointer(0, j);
            for (Int i = 0; i < output_height; ++i) {
                Field output_entry = out_col[i] * beta;
                for (Int k = 0; k < contraction_size; ++k) {
                    output_entry -= left_matrix(i, k) * right_col[k];
                }
                out_col[i] = output_entry;
            }
        }
    }
    else {
        for (Int j = 0; j < output_width; ++j) {
            Field *out_col = output_matrix->Pointer(0, j);
            const Field *right_col = right_matrix.Pointer(0, j);
            for (Int i = 0; i < output_height; ++i) {
                Field output_entry = out_col[i] * beta;
                for (Int k = 0; k < contraction_size; ++k) {
                    output_entry += alpha * left_matrix(i, k) * right_col[k];
                }
                out_col[i] = output_entry;
            }
        }
    }
}

template <class Field>
void MatrixMultiplyNormalNormal(const Field& alpha,
                                const ConstBlasMatrixView<Field>& left_matrix,
                                const ConstBlasMatrixView<Field>& right_matrix,
                                const Field& beta,
                                BlasMatrixView<Field>* output_matrix) {
  CATAMARI_ASSERT(left_matrix.height == output_matrix->height,
                  "Output height was incompatible");
  CATAMARI_ASSERT(right_matrix.width == output_matrix->width,
                  "Output width was incompatible");
  CATAMARI_ASSERT(left_matrix.width == right_matrix.height,
                  "Contraction dimensions were incompatible.");
  const Int output_height = output_matrix->height;
  const Int output_width = output_matrix->width;
  const Int contraction_size = left_matrix.width;
  for (Int j = 0; j < output_width; ++j) {
    for (Int i = 0; i < output_height; ++i) {
      Field& output_entry = output_matrix->Entry(i, j);
      output_entry *= beta;
      for (Int k = 0; k < contraction_size; ++k) {
        output_entry += alpha * left_matrix(i, k) * right_matrix(k, j);
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void MatrixMultiplyNormalNormal(
    const float& alpha, const ConstBlasMatrixView<float>& left_matrix,
    const ConstBlasMatrixView<float>& right_matrix, const float& beta,
    BlasMatrixView<float>* output_matrix) {
  CATAMARI_ASSERT(left_matrix.height == output_matrix->height,
                  "Output height was incompatible");
  CATAMARI_ASSERT(right_matrix.width == output_matrix->width,
                  "Output width was incompatible");
  CATAMARI_ASSERT(left_matrix.width == right_matrix.height,
                  "Contraction dimensions were incompatible.");
  const char trans_left = 'N';
  const char trans_right = 'N';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt output_width_blas = output_matrix->width;
  const BlasInt contraction_size_blas = left_matrix.width;
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
inline void MatrixMultiplyNormalNormal(
    const double& alpha, const ConstBlasMatrixView<double>& left_matrix,
    const ConstBlasMatrixView<double>& right_matrix, const double& beta,
    BlasMatrixView<double>* output_matrix) {
  CATAMARI_ASSERT(left_matrix.height == output_matrix->height,
                  "Output height was incompatible");
  CATAMARI_ASSERT(right_matrix.width == output_matrix->width,
                  "Output width was incompatible");
  CATAMARI_ASSERT(left_matrix.width == right_matrix.height,
                  "Contraction dimensions were incompatible.");
  const char trans_left = 'N';
  const char trans_right = 'N';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt output_width_blas = output_matrix->width;
  const BlasInt contraction_size_blas = left_matrix.width;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(dgemm)
  (&trans_left, &trans_right, &output_height_blas, &output_width_blas,
   &contraction_size_blas, &alpha, left_matrix.data, &left_leading_dim_blas,
   right_matrix.data, &right_leading_dim_blas, &beta, output_matrix->data,
   &output_leading_dim_blas);
}

template <>
inline void MatrixMultiplyNormalNormal(
    const Complex<float>& alpha,
    const ConstBlasMatrixView<Complex<float>>& left_matrix,
    const ConstBlasMatrixView<Complex<float>>& right_matrix,
    const Complex<float>& beta, BlasMatrixView<Complex<float>>* output_matrix) {
  CATAMARI_ASSERT(left_matrix.height == output_matrix->height,
                  "Output height was incompatible");
  CATAMARI_ASSERT(right_matrix.width == output_matrix->width,
                  "Output width was incompatible");
  CATAMARI_ASSERT(left_matrix.width == right_matrix.height,
                  "Contraction dimensions were incompatible.");
  const char trans_left = 'N';
  const char trans_right = 'N';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt output_width_blas = output_matrix->width;
  const BlasInt contraction_size_blas = left_matrix.width;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(cgemm)
  (&trans_left, &trans_right, &output_height_blas, &output_width_blas,
   &contraction_size_blas, reinterpret_cast<const BlasComplexFloat*>(&alpha),
   reinterpret_cast<const BlasComplexFloat*>(left_matrix.data),
   &left_leading_dim_blas,
   reinterpret_cast<const BlasComplexFloat*>(right_matrix.data),
   &right_leading_dim_blas, reinterpret_cast<const BlasComplexFloat*>(&beta),
   reinterpret_cast<BlasComplexFloat*>(output_matrix->data),
   &output_leading_dim_blas);
}

template <>
inline void MatrixMultiplyNormalNormal(
    const Complex<double>& alpha,
    const ConstBlasMatrixView<Complex<double>>& left_matrix,
    const ConstBlasMatrixView<Complex<double>>& right_matrix,
    const Complex<double>& beta,
    BlasMatrixView<Complex<double>>* output_matrix) {
  CATAMARI_ASSERT(left_matrix.height == output_matrix->height,
                  "Output height was incompatible");
  CATAMARI_ASSERT(right_matrix.width == output_matrix->width,
                  "Output width was incompatible");
  CATAMARI_ASSERT(left_matrix.width == right_matrix.height,
                  "Contraction dimensions were incompatible.");
  CATAMARI_ASSERT(left_matrix.leading_dim >= left_matrix.height,
                  "Left matrix had too small a leading dimension.");
  CATAMARI_ASSERT(right_matrix.leading_dim >= right_matrix.height,
                  "Right matrix had too small a leading dimension.");
  CATAMARI_ASSERT(output_matrix->leading_dim >= output_matrix->height,
                  "Output matrix had too small a leading dimension.");
  const char trans_left = 'N';
  const char trans_right = 'N';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt output_width_blas = output_matrix->width;
  const BlasInt contraction_size_blas = left_matrix.width;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(zgemm)
  (&trans_left, &trans_right, &output_height_blas, &output_width_blas,
   &contraction_size_blas, reinterpret_cast<const BlasComplexDouble*>(&alpha),
   reinterpret_cast<const BlasComplexDouble*>(left_matrix.data),
   &left_leading_dim_blas,
   reinterpret_cast<const BlasComplexDouble*>(right_matrix.data),
   &right_leading_dim_blas, reinterpret_cast<const BlasComplexDouble*>(&beta),
   reinterpret_cast<BlasComplexDouble*>(output_matrix->data),
   &output_leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void MatrixMultiplyNormalTranspose(
    const Field& alpha, const ConstBlasMatrixView<Field>& left_matrix,
    const ConstBlasMatrixView<Field>& right_matrix, const Field& beta,
    BlasMatrixView<Field>* output_matrix) {
  CATAMARI_ASSERT(left_matrix.height == output_matrix->height,
                  "Output height was incompatible");
  CATAMARI_ASSERT(right_matrix.height == output_matrix->width,
                  "Output width was incompatible");
  CATAMARI_ASSERT(left_matrix.width == right_matrix.width,
                  "Contraction dimensions were incompatible.");
  const Int output_height = output_matrix->height;
  const Int output_width = output_matrix->width;
  const Int contraction_size = left_matrix.width;
  for (Int j = 0; j < output_width; ++j) {
    for (Int i = 0; i < output_height; ++i) {
      Field& output_entry = output_matrix->Entry(i, j);
      output_entry *= beta;
      for (Int k = 0; k < contraction_size; ++k) {
        output_entry += alpha * left_matrix(i, k) * right_matrix(j, k);
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void MatrixMultiplyNormalTranspose(
    const float& alpha, const ConstBlasMatrixView<float>& left_matrix,
    const ConstBlasMatrixView<float>& right_matrix, const float& beta,
    BlasMatrixView<float>* output_matrix) {
  CATAMARI_ASSERT(left_matrix.height == output_matrix->height,
                  "Output height was incompatible");
  CATAMARI_ASSERT(right_matrix.height == output_matrix->width,
                  "Output width was incompatible");
  CATAMARI_ASSERT(left_matrix.width == right_matrix.width,
                  "Contraction dimensions were incompatible.");
  const char trans_left = 'N';
  const char trans_right = 'T';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt output_width_blas = output_matrix->width;
  const BlasInt contraction_size_blas = left_matrix.width;
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
inline void MatrixMultiplyNormalTranspose(
    const double& alpha, const ConstBlasMatrixView<double>& left_matrix,
    const ConstBlasMatrixView<double>& right_matrix, const double& beta,
    BlasMatrixView<double>* output_matrix) {
  CATAMARI_ASSERT(left_matrix.height == output_matrix->height,
                  "Output height was incompatible");
  CATAMARI_ASSERT(right_matrix.height == output_matrix->width,
                  "Output width was incompatible");
  CATAMARI_ASSERT(left_matrix.width == right_matrix.width,
                  "Contraction dimensions were incompatible.");
  const char trans_left = 'N';
  const char trans_right = 'T';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt output_width_blas = output_matrix->width;
  const BlasInt contraction_size_blas = left_matrix.width;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(dgemm)
  (&trans_left, &trans_right, &output_height_blas, &output_width_blas,
   &contraction_size_blas, &alpha, left_matrix.data, &left_leading_dim_blas,
   right_matrix.data, &right_leading_dim_blas, &beta, output_matrix->data,
   &output_leading_dim_blas);
}

template <>
inline void MatrixMultiplyNormalTranspose(
    const Complex<float>& alpha,
    const ConstBlasMatrixView<Complex<float>>& left_matrix,
    const ConstBlasMatrixView<Complex<float>>& right_matrix,
    const Complex<float>& beta, BlasMatrixView<Complex<float>>* output_matrix) {
  CATAMARI_ASSERT(left_matrix.height == output_matrix->height,
                  "Output height was incompatible");
  CATAMARI_ASSERT(right_matrix.height == output_matrix->width,
                  "Output width was incompatible");
  CATAMARI_ASSERT(left_matrix.width == right_matrix.width,
                  "Contraction dimensions were incompatible.");
  const char trans_left = 'N';
  const char trans_right = 'T';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt output_width_blas = output_matrix->width;
  const BlasInt contraction_size_blas = left_matrix.width;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(cgemm)
  (&trans_left, &trans_right, &output_height_blas, &output_width_blas,
   &contraction_size_blas, reinterpret_cast<const BlasComplexFloat*>(&alpha),
   reinterpret_cast<const BlasComplexFloat*>(left_matrix.data),
   &left_leading_dim_blas,
   reinterpret_cast<const BlasComplexFloat*>(right_matrix.data),
   &right_leading_dim_blas, reinterpret_cast<const BlasComplexFloat*>(&beta),
   reinterpret_cast<BlasComplexFloat*>(output_matrix->data),
   &output_leading_dim_blas);
}

template <>
inline void MatrixMultiplyNormalTranspose(
    const Complex<double>& alpha,
    const ConstBlasMatrixView<Complex<double>>& left_matrix,
    const ConstBlasMatrixView<Complex<double>>& right_matrix,
    const Complex<double>& beta,
    BlasMatrixView<Complex<double>>* output_matrix) {
  CATAMARI_ASSERT(left_matrix.height == output_matrix->height,
                  "Output height was incompatible");
  CATAMARI_ASSERT(right_matrix.height == output_matrix->width,
                  "Output width was incompatible");
  CATAMARI_ASSERT(left_matrix.width == right_matrix.width,
                  "Contraction dimensions were incompatible.");
  const char trans_left = 'N';
  const char trans_right = 'T';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt output_width_blas = output_matrix->width;
  const BlasInt contraction_size_blas = left_matrix.width;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(zgemm)
  (&trans_left, &trans_right, &output_height_blas, &output_width_blas,
   &contraction_size_blas, reinterpret_cast<const BlasComplexDouble*>(&alpha),
   reinterpret_cast<const BlasComplexDouble*>(left_matrix.data),
   &left_leading_dim_blas,
   reinterpret_cast<const BlasComplexDouble*>(right_matrix.data),
   &right_leading_dim_blas, reinterpret_cast<const BlasComplexDouble*>(&beta),
   reinterpret_cast<BlasComplexDouble*>(output_matrix->data),
   &output_leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void MatrixMultiplyNormalAdjoint(const Field& alpha,
                                 const ConstBlasMatrixView<Field>& left_matrix,
                                 const ConstBlasMatrixView<Field>& right_matrix,
                                 const Field& beta,
                                 BlasMatrixView<Field>* output_matrix) {
  CATAMARI_ASSERT(left_matrix.height == output_matrix->height,
                  "Output height was incompatible");
  CATAMARI_ASSERT(right_matrix.height == output_matrix->width,
                  "Output width was incompatible");
  CATAMARI_ASSERT(left_matrix.width == right_matrix.width,
                  "Contraction dimensions were incompatible.");
  const Int output_height = output_matrix->height;
  const Int output_width = output_matrix->width;
  const Int contraction_size = left_matrix.width;
  for (Int j = 0; j < output_width; ++j) {
    for (Int i = 0; i < output_height; ++i) {
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
inline void MatrixMultiplyNormalAdjoint(
    const float& alpha, const ConstBlasMatrixView<float>& left_matrix,
    const ConstBlasMatrixView<float>& right_matrix, const float& beta,
    BlasMatrixView<float>* output_matrix) {
  MatrixMultiplyNormalTranspose(alpha, left_matrix, right_matrix, beta,
                                output_matrix);
}

template <>
inline void MatrixMultiplyNormalAdjoint(
    const double& alpha, const ConstBlasMatrixView<double>& left_matrix,
    const ConstBlasMatrixView<double>& right_matrix, const double& beta,
    BlasMatrixView<double>* output_matrix) {
  MatrixMultiplyNormalTranspose(alpha, left_matrix, right_matrix, beta,
                                output_matrix);
}

template <>
inline void MatrixMultiplyNormalAdjoint(
    const Complex<float>& alpha,
    const ConstBlasMatrixView<Complex<float>>& left_matrix,
    const ConstBlasMatrixView<Complex<float>>& right_matrix,
    const Complex<float>& beta, BlasMatrixView<Complex<float>>* output_matrix) {
  CATAMARI_ASSERT(left_matrix.height == output_matrix->height,
                  "Output height was incompatible");
  CATAMARI_ASSERT(right_matrix.height == output_matrix->width,
                  "Output width was incompatible");
  CATAMARI_ASSERT(left_matrix.width == right_matrix.width,
                  "Contraction dimensions were incompatible.");
  const char trans_left = 'N';
  const char trans_right = 'C';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt output_width_blas = output_matrix->width;
  const BlasInt contraction_size_blas = left_matrix.width;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(cgemm)
  (&trans_left, &trans_right, &output_height_blas, &output_width_blas,
   &contraction_size_blas, reinterpret_cast<const BlasComplexFloat*>(&alpha),
   reinterpret_cast<const BlasComplexFloat*>(left_matrix.data),
   &left_leading_dim_blas,
   reinterpret_cast<const BlasComplexFloat*>(right_matrix.data),
   &right_leading_dim_blas, reinterpret_cast<const BlasComplexFloat*>(&beta),
   reinterpret_cast<BlasComplexFloat*>(output_matrix->data),
   &output_leading_dim_blas);
}

template <>
inline void MatrixMultiplyNormalAdjoint(
    const Complex<double>& alpha,
    const ConstBlasMatrixView<Complex<double>>& left_matrix,
    const ConstBlasMatrixView<Complex<double>>& right_matrix,
    const Complex<double>& beta,
    BlasMatrixView<Complex<double>>* output_matrix) {
  CATAMARI_ASSERT(left_matrix.height == output_matrix->height,
                  "Output height was incompatible");
  CATAMARI_ASSERT(right_matrix.height == output_matrix->width,
                  "Output width was incompatible");
  CATAMARI_ASSERT(left_matrix.width == right_matrix.width,
                  "Contraction dimensions were incompatible.");
  const char trans_left = 'N';
  const char trans_right = 'C';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt output_width_blas = output_matrix->width;
  const BlasInt contraction_size_blas = left_matrix.width;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(zgemm)
  (&trans_left, &trans_right, &output_height_blas, &output_width_blas,
   &contraction_size_blas, reinterpret_cast<const BlasComplexDouble*>(&alpha),
   reinterpret_cast<const BlasComplexDouble*>(left_matrix.data),
   &left_leading_dim_blas,
   reinterpret_cast<const BlasComplexDouble*>(right_matrix.data),
   &right_leading_dim_blas, reinterpret_cast<const BlasComplexDouble*>(&beta),
   reinterpret_cast<BlasComplexDouble*>(output_matrix->data),
   &output_leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void MatrixMultiplyTransposeNormal(
    const Field& alpha, const ConstBlasMatrixView<Field>& left_matrix,
    const ConstBlasMatrixView<Field>& right_matrix, const Field& beta,
    BlasMatrixView<Field>* output_matrix) {
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
    const float& alpha, const ConstBlasMatrixView<float>& left_matrix,
    const ConstBlasMatrixView<float>& right_matrix, const float& beta,
    BlasMatrixView<float>* output_matrix) {
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
    const double& alpha, const ConstBlasMatrixView<double>& left_matrix,
    const ConstBlasMatrixView<double>& right_matrix, const double& beta,
    BlasMatrixView<double>* output_matrix) {
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

template <>
inline void MatrixMultiplyTransposeNormal(
    const Complex<float>& alpha,
    const ConstBlasMatrixView<Complex<float>>& left_matrix,
    const ConstBlasMatrixView<Complex<float>>& right_matrix,
    const Complex<float>& beta, BlasMatrixView<Complex<float>>* output_matrix) {
  const char trans_left = 'T';
  const char trans_right = 'N';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt output_width_blas = output_matrix->width;
  const BlasInt contraction_size_blas = left_matrix.height;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(cgemm)
  (&trans_left, &trans_right, &output_height_blas, &output_width_blas,
   &contraction_size_blas, reinterpret_cast<const BlasComplexFloat*>(&alpha),
   reinterpret_cast<const BlasComplexFloat*>(left_matrix.data),
   &left_leading_dim_blas,
   reinterpret_cast<const BlasComplexFloat*>(right_matrix.data),
   &right_leading_dim_blas, reinterpret_cast<const BlasComplexFloat*>(&beta),
   reinterpret_cast<BlasComplexFloat*>(output_matrix->data),
   &output_leading_dim_blas);
}

template <>
inline void MatrixMultiplyTransposeNormal(
    const Complex<double>& alpha,
    const ConstBlasMatrixView<Complex<double>>& left_matrix,
    const ConstBlasMatrixView<Complex<double>>& right_matrix,
    const Complex<double>& beta,
    BlasMatrixView<Complex<double>>* output_matrix) {
  const char trans_left = 'T';
  const char trans_right = 'N';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt output_width_blas = output_matrix->width;
  const BlasInt contraction_size_blas = left_matrix.height;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(zgemm)
  (&trans_left, &trans_right, &output_height_blas, &output_width_blas,
   &contraction_size_blas, reinterpret_cast<const BlasComplexDouble*>(&alpha),
   reinterpret_cast<const BlasComplexDouble*>(left_matrix.data),
   &left_leading_dim_blas,
   reinterpret_cast<const BlasComplexDouble*>(right_matrix.data),
   &right_leading_dim_blas, reinterpret_cast<const BlasComplexDouble*>(&beta),
   reinterpret_cast<BlasComplexDouble*>(output_matrix->data),
   &output_leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void MatrixMultiplyAdjointNormal(const Field& alpha,
                                 const ConstBlasMatrixView<Field>& left_matrix,
                                 const ConstBlasMatrixView<Field>& right_matrix,
                                 const Field& beta,
                                 BlasMatrixView<Field>* output_matrix) {
  const Int output_height = output_matrix->height;
  const Int output_width = output_matrix->width;
  const Int contraction_size = left_matrix.height;
  for (Int j = 0; j < output_width; ++j) {
    for (Int i = 0; i < output_height; ++i) {
      Field& output_entry = output_matrix->Entry(i, j);
      output_entry *= beta;
      for (Int k = 0; k < contraction_size; ++k) {
        output_entry +=
            alpha * Conjugate(left_matrix(k, i)) * right_matrix(k, j);
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void MatrixMultiplyAdjointNormal(
    const float& alpha, const ConstBlasMatrixView<float>& left_matrix,
    const ConstBlasMatrixView<float>& right_matrix, const float& beta,
    BlasMatrixView<float>* output_matrix) {
  MatrixMultiplyTransposeNormal(alpha, left_matrix, right_matrix, beta,
                                output_matrix);
}

template <>
inline void MatrixMultiplyAdjointNormal(
    const double& alpha, const ConstBlasMatrixView<double>& left_matrix,
    const ConstBlasMatrixView<double>& right_matrix, const double& beta,
    BlasMatrixView<double>* output_matrix) {
  MatrixMultiplyTransposeNormal(alpha, left_matrix, right_matrix, beta,
                                output_matrix);
}

template <>
inline void MatrixMultiplyAdjointNormal(
    const Complex<float>& alpha,
    const ConstBlasMatrixView<Complex<float>>& left_matrix,
    const ConstBlasMatrixView<Complex<float>>& right_matrix,
    const Complex<float>& beta, BlasMatrixView<Complex<float>>* output_matrix) {
  const char trans_left = 'C';
  const char trans_right = 'N';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt output_width_blas = output_matrix->width;
  const BlasInt contraction_size_blas = left_matrix.height;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(cgemm)
  (&trans_left, &trans_right, &output_height_blas, &output_width_blas,
   &contraction_size_blas, reinterpret_cast<const BlasComplexFloat*>(&alpha),
   reinterpret_cast<const BlasComplexFloat*>(left_matrix.data),
   &left_leading_dim_blas,
   reinterpret_cast<const BlasComplexFloat*>(right_matrix.data),
   &right_leading_dim_blas, reinterpret_cast<const BlasComplexFloat*>(&beta),
   reinterpret_cast<BlasComplexFloat*>(output_matrix->data),
   &output_leading_dim_blas);
}

template <>
inline void MatrixMultiplyAdjointNormal(
    const Complex<double>& alpha,
    const ConstBlasMatrixView<Complex<double>>& left_matrix,
    const ConstBlasMatrixView<Complex<double>>& right_matrix,
    const Complex<double>& beta,
    BlasMatrixView<Complex<double>>* output_matrix) {
  const char trans_left = 'C';
  const char trans_right = 'N';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt output_width_blas = output_matrix->width;
  const BlasInt contraction_size_blas = left_matrix.height;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(zgemm)
  (&trans_left, &trans_right, &output_height_blas, &output_width_blas,
   &contraction_size_blas, reinterpret_cast<const BlasComplexDouble*>(&alpha),
   reinterpret_cast<const BlasComplexDouble*>(left_matrix.data),
   &left_leading_dim_blas,
   reinterpret_cast<const BlasComplexDouble*>(right_matrix.data),
   &right_leading_dim_blas, reinterpret_cast<const BlasComplexDouble*>(&beta),
   reinterpret_cast<BlasComplexDouble*>(output_matrix->data),
   &output_leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void LowerNormalHermitianOuterProductDynamicBLASDispatch(
    const ComplexBase<Field>& alpha,
    const Eigen::Matrix<Field, Eigen::Dynamic, Eigen::Dynamic> left_mat_transpose,
    const ComplexBase<Field>& beta, BlasMatrixView<Field>* output_matrix) {
  const Int output_height = output_matrix->height;
  const Int contraction_size = left_mat_transpose.rows();

    if (output_height * output_height * contraction_size > 1000)
    {
        // return LowerNormalHermitianOuterProduct(alpha, left_matrix, beta, output_matrix);
        const char uplo = 'L';
        const char trans = 'T';
        const BlasInt height_blas = output_matrix->height;
        const BlasInt rank_blas = contraction_size;
        const BlasInt factor_leading_dim_blas = rank_blas;
        const BlasInt leading_dim_blas = output_matrix->leading_dim;
        BLAS_SYMBOL(dsyrk)
        (&uplo, &trans, &height_blas, &rank_blas, &alpha, left_mat_transpose.data(),
         &factor_leading_dim_blas, &beta, output_matrix->data, &leading_dim_blas);
        return;
    }

    if (beta == Field(0)) {
        for (Int j = 0; j < output_height; ++j) {
            Field *out_col = output_matrix->Pointer(0, j);
            for (Int i = j; i < output_height; ++i) {
                Field val = left_mat_transpose.col(i).dot(left_mat_transpose.col(j));
                // Field val = 0;
                // for (Int k = 0; k < contraction_size; ++k)
                //     val += left_mat_transpose(k, i) * Conjugate(left_mat_transpose(k, j));
                out_col[i] = alpha * val;
            }
        }
    }
    else {
        for (Int j = 0; j < output_height; ++j) {
            Field *out_col = output_matrix->Pointer(0, j);
            for (Int i = j; i < output_height; ++i) {
                Field val = left_mat_transpose.col(i).dot(left_mat_transpose.col(j));
                // Field val = 0;
                // for (Int k = 0; k < contraction_size; ++k)
                //     val += left_mat_transpose(k, i) * Conjugate(left_mat_transpose(k, j));
                out_col[i] = alpha * val + beta * out_col[i];
            }
        }
    }
}

template <class Field>
void LowerNormalHermitianOuterProductDynamicBLASDispatch(
    const ComplexBase<Field>& alpha,
    const ConstBlasMatrixView<Field>& left_matrix,
    const ComplexBase<Field>& beta, BlasMatrixView<Field>* output_matrix) {
  const Int output_height = output_matrix->height;
  const Int contraction_size = left_matrix.width;

#ifdef CATAMARI_HAVE_BLAS
    // Only use BLAS for large enough jobs.
    // Rather than basing the treshold on a flop estimate (h^2 c) it seems better to base it
    // separately on `output_height` and `contraction_size`; the naive implementation below
    // performs well if the output matrix fits in cache or if only a rank 1 matrix is added.
    if (output_height > 128 && contraction_size > 1) {
        return LowerNormalHermitianOuterProduct(alpha, left_matrix, beta, output_matrix);
    }
#endif

  using EVec = Eigen::Matrix<Field, Eigen::Dynamic, 1>;
  int k_start = 0;
  if (beta == Field(0)) {
      const Field *col_k = left_matrix.Pointer(0, 0);
      for (Int j = 0; j < output_height; ++j) {
        Field alpha_l_jk = alpha * Conjugate(col_k[j]);
        Field *out_col = output_matrix->Pointer(0, j);
        for (Int i = j; i < output_height; ++i)
              out_col[i] = alpha_l_jk * col_k[i];
      }
      k_start = 1;
  }
  else if (beta != Field(1)) {
      for (Int j = 0; j < output_height; ++j)
        for (Int i = j; i < output_height; ++i)
            output_matrix->Entry(i, j) *= beta;
  }
  for (Int k = k_start; k < contraction_size; ++k) {
      const Field *col_k = left_matrix.Pointer(0, k);
      for (Int j = 0; j < output_height; ++j) {
          Field alpha_l_jk = alpha * Conjugate(col_k[j]);
#if 0
          // Eigen version seems slower...
          const Int len = output_height - j;
          Eigen::Map<EVec>(output_matrix->Pointer(j, j), len) += alpha_l_jk * Eigen::Map<const EVec>(col_k + j, len);
#else
          Field *out_col = output_matrix->Pointer(0, j);
          for (Int i = j; i < output_height; ++i) {
              out_col[i] += alpha_l_jk * col_k[i];
          }
      }
  }
#endif
}

template <class Field>
void LowerNormalHermitianOuterProduct(
    const ComplexBase<Field>& alpha,
    const ConstBlasMatrixView<Field>& left_matrix,
    const ComplexBase<Field>& beta, BlasMatrixView<Field>* output_matrix) {
  const Int output_height = output_matrix->height;
  const Int contraction_size = left_matrix.width;

  int k_start = 0;
  if (beta != Field(1)) {
      for (Int j = 0; j < output_height; ++j)
        for (Int i = j; i < output_height; ++i)
            output_matrix->Entry(i, j) *= beta;
  }
  else if (beta == Field(0)) {
      Int k = 0;
      const Field *col_k = left_matrix.Pointer(0, k);
      for (Int j = 0; j < output_height; ++j) {
          Field alpha_l_jk = alpha * Conjugate(col_k[j]);
          Field *out_col = output_matrix->Pointer(0, j);
          for (Int i = j; i < output_height; ++i) {
              out_col[i] = alpha_l_jk * col_k[i];
          }
      }
      k_start = 1;
  }
  for (Int k = k_start; k < contraction_size; ++k) {
      const Field *col_k = left_matrix.Pointer(0, k);
      for (Int j = 0; j < output_height; ++j) {
          Field alpha_l_jk = alpha * Conjugate(col_k[j]);
          Field *out_col = output_matrix->Pointer(0, j);
          for (Int i = j; i < output_height; ++i) {
              out_col[i] += alpha_l_jk * col_k[i];
          }
      }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void LowerNormalHermitianOuterProduct(
    const float& alpha, const ConstBlasMatrixView<float>& left_matrix,
    const float& beta, BlasMatrixView<float>* output_matrix) {
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
    const double& alpha, const ConstBlasMatrixView<double>& left_matrix,
    const double& beta, BlasMatrixView<double>* output_matrix) {
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

template <>
inline void LowerNormalHermitianOuterProduct(
    const float& alpha, const ConstBlasMatrixView<Complex<float>>& left_matrix,
    const float& beta, BlasMatrixView<Complex<float>>* output_matrix) {
  const char uplo = 'L';
  const char trans = 'N';
  const BlasInt height_blas = output_matrix->height;
  const BlasInt rank_blas = left_matrix.width;
  const BlasInt factor_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(cherk)
  (&uplo, &trans, &height_blas, &rank_blas, &alpha,
   reinterpret_cast<const BlasComplexFloat*>(left_matrix.data),
   &factor_leading_dim_blas, &beta,
   reinterpret_cast<BlasComplexFloat*>(output_matrix->data), &leading_dim_blas);
}

template <>
inline void LowerNormalHermitianOuterProduct(
    const double& alpha,
    const ConstBlasMatrixView<Complex<double>>& left_matrix, const double& beta,
    BlasMatrixView<Complex<double>>* output_matrix) {
  const char uplo = 'L';
  const char trans = 'N';
  const BlasInt height_blas = output_matrix->height;
  const BlasInt rank_blas = left_matrix.width;
  const BlasInt factor_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(zherk)
  (&uplo, &trans, &height_blas, &rank_blas, &alpha,
   reinterpret_cast<const BlasComplexDouble*>(left_matrix.data),
   &factor_leading_dim_blas, &beta,
   reinterpret_cast<BlasComplexDouble*>(output_matrix->data),
   &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void MatrixMultiplyLowerNormalNormal(
    const Field& alpha, const ConstBlasMatrixView<Field>& left_matrix,
    const ConstBlasMatrixView<Field>& right_matrix, const Field& beta,
    BlasMatrixView<Field>* output_matrix) {
  const Int output_height = output_matrix->height;
  const Int contraction_size = left_matrix.width;
  for (Int j = 0; j < output_height; ++j) {
    for (Int i = j; i < output_height; ++i) {
      Field& output_entry = output_matrix->Entry(i, j);
      output_entry *= beta;
      for (Int k = 0; k < contraction_size; ++k) {
        output_entry += alpha * left_matrix(i, k) * right_matrix(k, j);
      }
    }
  }
}

#ifdef CATAMARI_USE_GEMMT
template <>
inline void MatrixMultiplyLowerNormalNormal(
    const float& alpha, const ConstBlasMatrixView<float>& left_matrix,
    const ConstBlasMatrixView<float>& right_matrix, const float& beta,
    BlasMatrixView<float>* output_matrix) {
  const char uplo = 'L';
  const char trans_left = 'N';
  const char trans_right = 'N';
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
inline void MatrixMultiplyLowerNormalNormal(
    const double& alpha, const ConstBlasMatrixView<double>& left_matrix,
    const ConstBlasMatrixView<double>& right_matrix, const double& beta,
    BlasMatrixView<double>* output_matrix) {
  const char uplo = 'L';
  const char trans_left = 'N';
  const char trans_right = 'N';
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

template <>
inline void MatrixMultiplyLowerNormalNormal(
    const Complex<float>& alpha,
    const ConstBlasMatrixView<Complex<float>>& left_matrix,
    const ConstBlasMatrixView<Complex<float>>& right_matrix,
    const Complex<float>& beta, BlasMatrixView<Complex<float>>* output_matrix) {
  const char uplo = 'L';
  const char trans_left = 'N';
  const char trans_right = 'N';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt contraction_size_blas = left_matrix.width;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(cgemmt)
  (&uplo, &trans_left, &trans_right, &output_height_blas,
   &contraction_size_blas, &alpha, left_matrix.data, &left_leading_dim_blas,
   right_matrix.data, &right_leading_dim_blas, &beta, output_matrix->data,
   &output_leading_dim_blas);
}

template <>
inline void MatrixMultiplyLowerNormalNormal(
    const Complex<double>& alpha,
    const ConstBlasMatrixView<Complex<double>>& left_matrix,
    const ConstBlasMatrixView<Complex<double>>& right_matrix,
    const Complex<double>& beta,
    BlasMatrixView<Complex<double>>* output_matrix) {
  const char uplo = 'L';
  const char trans_left = 'N';
  const char trans_right = 'N';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt contraction_size_blas = left_matrix.width;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(zgemmt)
  (&uplo, &trans_left, &trans_right, &output_height_blas,
   &contraction_size_blas, &alpha, left_matrix.data, &left_leading_dim_blas,
   right_matrix.data, &right_leading_dim_blas, &beta, output_matrix->data,
   &output_leading_dim_blas);
}
#elif defined(CATAMARI_HAVE_BLAS)
template <>
inline void MatrixMultiplyLowerNormalNormal(
    const float& alpha, const ConstBlasMatrixView<float>& left_matrix,
    const ConstBlasMatrixView<float>& right_matrix, const float& beta,
    BlasMatrixView<float>* output_matrix) {
  MatrixMultiplyNormalNormal(alpha, left_matrix, right_matrix, beta,
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
inline void MatrixMultiplyLowerNormalNormal(
    const double& alpha, const ConstBlasMatrixView<double>& left_matrix,
    const ConstBlasMatrixView<double>& right_matrix, const double& beta,
    BlasMatrixView<double>* output_matrix) {
  MatrixMultiplyNormalNormal(alpha, left_matrix, right_matrix, beta,
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
inline void MatrixMultiplyLowerNormalNormal(
    const Complex<float>& alpha,
    const ConstBlasMatrixView<Complex<float>>& left_matrix,
    const ConstBlasMatrixView<Complex<float>>& right_matrix,
    const Complex<float>& beta, BlasMatrixView<Complex<float>>* output_matrix) {
  MatrixMultiplyNormalNormal(alpha, left_matrix, right_matrix, beta,
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
inline void MatrixMultiplyLowerNormalNormal(
    const Complex<double>& alpha,
    const ConstBlasMatrixView<Complex<double>>& left_matrix,
    const ConstBlasMatrixView<Complex<double>>& right_matrix,
    const Complex<double>& beta,
    BlasMatrixView<Complex<double>>* output_matrix) {
  MatrixMultiplyNormalNormal(alpha, left_matrix, right_matrix, beta,
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
void MatrixMultiplyLowerNormalTranspose(
    const Field& alpha, const ConstBlasMatrixView<Field>& left_matrix,
    const ConstBlasMatrixView<Field>& right_matrix, const Field& beta,
    BlasMatrixView<Field>* output_matrix) {
  const Int output_height = output_matrix->height;
  const Int contraction_size = left_matrix.width;
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
    const float& alpha, const ConstBlasMatrixView<float>& left_matrix,
    const ConstBlasMatrixView<float>& right_matrix, const float& beta,
    BlasMatrixView<float>* output_matrix) {
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
    const double& alpha, const ConstBlasMatrixView<double>& left_matrix,
    const ConstBlasMatrixView<double>& right_matrix, const double& beta,
    BlasMatrixView<double>* output_matrix) {
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

template <>
inline void MatrixMultiplyLowerNormalTranspose(
    const Complex<float>& alpha,
    const ConstBlasMatrixView<Complex<float>>& left_matrix,
    const ConstBlasMatrixView<Complex<float>>& right_matrix,
    const Complex<float>& beta, BlasMatrixView<Complex<float>>* output_matrix) {
  const char uplo = 'L';
  const char trans_left = 'N';
  const char trans_right = 'T';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt contraction_size_blas = left_matrix.width;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(cgemmt)
  (&uplo, &trans_left, &trans_right, &output_height_blas,
   &contraction_size_blas, &alpha, left_matrix.data, &left_leading_dim_blas,
   right_matrix.data, &right_leading_dim_blas, &beta, output_matrix->data,
   &output_leading_dim_blas);
}

template <>
inline void MatrixMultiplyLowerNormalTranspose(
    const Complex<double>& alpha,
    const ConstBlasMatrixView<Complex<double>>& left_matrix,
    const ConstBlasMatrixView<Complex<double>>& right_matrix,
    const Complex<double>& beta,
    BlasMatrixView<Complex<double>>* output_matrix) {
  const char uplo = 'L';
  const char trans_left = 'N';
  const char trans_right = 'T';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt contraction_size_blas = left_matrix.width;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(zgemmt)
  (&uplo, &trans_left, &trans_right, &output_height_blas,
   &contraction_size_blas, &alpha, left_matrix.data, &left_leading_dim_blas,
   right_matrix.data, &right_leading_dim_blas, &beta, output_matrix->data,
   &output_leading_dim_blas);
}
#elif defined(CATAMARI_HAVE_BLAS)
template <>
inline void MatrixMultiplyLowerNormalTranspose(
    const float& alpha, const ConstBlasMatrixView<float>& left_matrix,
    const ConstBlasMatrixView<float>& right_matrix, const float& beta,
    BlasMatrixView<float>* output_matrix) {
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
    const double& alpha, const ConstBlasMatrixView<double>& left_matrix,
    const ConstBlasMatrixView<double>& right_matrix, const double& beta,
    BlasMatrixView<double>* output_matrix) {
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
    const Complex<float>& alpha,
    const ConstBlasMatrixView<Complex<float>>& left_matrix,
    const ConstBlasMatrixView<Complex<float>>& right_matrix,
    const Complex<float>& beta, BlasMatrixView<Complex<float>>* output_matrix) {
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
    const Complex<double>& alpha,
    const ConstBlasMatrixView<Complex<double>>& left_matrix,
    const ConstBlasMatrixView<Complex<double>>& right_matrix,
    const Complex<double>& beta,
    BlasMatrixView<Complex<double>>* output_matrix) {
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
    const Field& alpha, const ConstBlasMatrixView<Field>& left_matrix,
    const ConstBlasMatrixView<Field>& right_matrix, const Field& beta,
    BlasMatrixView<Field>* output_matrix) {
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
    const float& alpha, const ConstBlasMatrixView<float>& left_matrix,
    const ConstBlasMatrixView<float>& right_matrix, const float& beta,
    BlasMatrixView<float>* output_matrix) {
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
    const double& alpha, const ConstBlasMatrixView<double>& left_matrix,
    const ConstBlasMatrixView<double>& right_matrix, const double& beta,
    BlasMatrixView<double>* output_matrix) {
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

template <>
inline void MatrixMultiplyLowerTransposeNormal(
    const Complex<float>& alpha,
    const ConstBlasMatrixView<Complex<float>>& left_matrix,
    const ConstBlasMatrixView<Complex<float>>& right_matrix,
    const Complex<float>& beta, BlasMatrixView<Complex<float>>* output_matrix) {
  const char uplo = 'L';
  const char trans_left = 'T';
  const char trans_right = 'N';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt contraction_size_blas = left_matrix.height;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(cgemmt)
  (&uplo, &trans_left, &trans_right, &output_height_blas,
   &contraction_size_blas, &alpha, left_matrix.data, &left_leading_dim_blas,
   right_matrix.data, &right_leading_dim_blas, &beta, output_matrix->data,
   &output_leading_dim_blas);
}

template <>
inline void MatrixMultiplyLowerTransposeNormal(
    const Complex<double>& alpha,
    const ConstBlasMatrixView<Complex<double>>& left_matrix,
    const ConstBlasMatrixView<Complex<double>>& right_matrix,
    const Complex<double>& beta,
    BlasMatrixView<Complex<double>>* output_matrix) {
  const char uplo = 'L';
  const char trans_left = 'T';
  const char trans_right = 'N';
  const BlasInt output_height_blas = output_matrix->height;
  const BlasInt contraction_size_blas = left_matrix.height;
  const BlasInt left_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt right_leading_dim_blas = right_matrix.leading_dim;
  const BlasInt output_leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(zgemmt)
  (&uplo, &trans_left, &trans_right, &output_height_blas,
   &contraction_size_blas, &alpha, left_matrix.data, &left_leading_dim_blas,
   right_matrix.data, &right_leading_dim_blas, &beta, output_matrix->data,
   &output_leading_dim_blas);
}
#elif defined(CATAMARI_HAVE_BLAS)
template <>
inline void MatrixMultiplyLowerTransposeNormal(
    const float& alpha, const ConstBlasMatrixView<float>& left_matrix,
    const ConstBlasMatrixView<float>& right_matrix, const float& beta,
    BlasMatrixView<float>* output_matrix) {
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
    const double& alpha, const ConstBlasMatrixView<double>& left_matrix,
    const ConstBlasMatrixView<double>& right_matrix, const double& beta,
    BlasMatrixView<double>* output_matrix) {
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
    const Complex<float>& alpha,
    const ConstBlasMatrixView<Complex<float>>& left_matrix,
    const ConstBlasMatrixView<Complex<float>>& right_matrix,
    const Complex<float>& beta, BlasMatrixView<Complex<float>>* output_matrix) {
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
    const Complex<double>& alpha,
    const ConstBlasMatrixView<Complex<double>>& left_matrix,
    const ConstBlasMatrixView<Complex<double>>& right_matrix,
    const Complex<double>& beta,
    BlasMatrixView<Complex<double>>* output_matrix) {
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
void LeftLowerTriangularSolvesDynamicBLASDispatch(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");

#ifdef CATAMARI_HAVE_BLAS
    if (triangular_matrix.height > 5) // only use BLAS call for large enough matrices
        return LeftLowerTriangularSolves(triangular_matrix, matrix);
#endif

  const Int width = matrix->width;
  for (Int j = 0; j < width; ++j) {
    Field* input_column = matrix->Pointer(0, j);

    // Interleave with a subsequent solve against D.
    for (Int i = 0; i < triangular_matrix.height; ++i) {
      const Field* l_column = triangular_matrix.Pointer(0, i);
      Field val = (input_column[i] /= l_column[i]);
      for (Int k = i + 1; k < triangular_matrix.height; ++k)
        input_column[k] -= l_column[k] * val;
    }
  }
}

template <class Field>
void LeftLowerTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const Int width = matrix->width;
  for (Int j = 0; j < width; ++j) {
    Field* input_column = matrix->Pointer(0, j);

    // Interleave with a subsequent solve against D.
    for (Int i = 0; i < triangular_matrix.height; ++i) {
      const Field* l_column = triangular_matrix.Pointer(0, i);
      Field val = (input_column[i] /= l_column[i]);
      for (Int k = i + 1; k < triangular_matrix.height; ++k)
        input_column[k] -= l_column[k] * val;
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void LeftLowerTriangularSolves(
    const ConstBlasMatrixView<float>& triangular_matrix,
    BlasMatrixView<float>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'N';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const float alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(strsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void LeftLowerTriangularSolves(
    const ConstBlasMatrixView<double>& triangular_matrix,
    BlasMatrixView<double>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'N';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const double alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(dtrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void LeftLowerTriangularSolves(
    const ConstBlasMatrixView<Complex<float>>& triangular_matrix,
    BlasMatrixView<Complex<float>>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'N';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const Complex<float> alpha = 1.f;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(ctrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexFloat*>(&alpha),
   reinterpret_cast<const BlasComplexFloat*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexFloat*>(matrix->data),
   &leading_dim_blas);
}

template <>
inline void LeftLowerTriangularSolves(
    const ConstBlasMatrixView<Complex<double>>& triangular_matrix,
    BlasMatrixView<Complex<double>>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'N';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const Complex<double> alpha = 1.;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(ztrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexDouble*>(&alpha),
   reinterpret_cast<const BlasComplexDouble*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexDouble*>(matrix->data),
   &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void LeftLowerUnitTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const Int width = matrix->width;
  for (Int j = 0; j < width; ++j) {
    Field* input_column = matrix->Pointer(0, j);

    // Interleave with a subsequent solve against D.
    for (Int i = 0; i < triangular_matrix.height; ++i) {
      const Field* l_column = triangular_matrix.Pointer(0, i);
      for (Int k = i + 1; k < triangular_matrix.height; ++k) {
        input_column[k] -= l_column[k] * input_column[i];
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void LeftLowerUnitTriangularSolves(
    const ConstBlasMatrixView<float>& triangular_matrix,
    BlasMatrixView<float>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'N';
  const char diag = 'U';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const float alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(strsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void LeftLowerUnitTriangularSolves(
    const ConstBlasMatrixView<double>& triangular_matrix,
    BlasMatrixView<double>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'N';
  const char diag = 'U';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const double alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(dtrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void LeftLowerUnitTriangularSolves(
    const ConstBlasMatrixView<Complex<float>>& triangular_matrix,
    BlasMatrixView<Complex<float>>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'N';
  const char diag = 'U';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const Complex<float> alpha = 1.f;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(ctrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexFloat*>(&alpha),
   reinterpret_cast<const BlasComplexFloat*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexFloat*>(matrix->data),
   &leading_dim_blas);
}

template <>
inline void LeftLowerUnitTriangularSolves(
    const ConstBlasMatrixView<Complex<double>>& triangular_matrix,
    BlasMatrixView<Complex<double>>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'N';
  const char diag = 'U';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const Complex<double> alpha = 1.;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(ztrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexDouble*>(&alpha),
   reinterpret_cast<const BlasComplexDouble*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexDouble*>(matrix->data),
   &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void LeftLowerAdjointTriangularSolvesDynamicBLASDispatch(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix) {
    CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                    triangular_matrix.height == matrix->height,
                    "Incompatible matrix dimensions");
#ifdef CATAMARI_HAVE_BLAS
    if (triangular_matrix.height > 5) // only use BLAS call for large enough matrices
        return LeftLowerAdjointTriangularSolves(triangular_matrix, matrix);
#endif

    const Int width = matrix->width;
    for (Int j = 0; j < width; ++j) {
        Field *matrix_col_j = matrix->Pointer(0, j);
        for (Int k = triangular_matrix.height - 1; k >= 0; --k) {
            const Field* tri_col = triangular_matrix.Pointer(0, k);
            Field eta = matrix_col_j[k];
            for (Int i = k + 1; i < triangular_matrix.height; ++i)
                eta -= Conjugate(tri_col[i]) * matrix_col_j[i];
            eta /= Conjugate(tri_col[k]);
            matrix_col_j[k] = eta;
        }
    }
}

template <class Field>
void LeftLowerAdjointTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const Int width = matrix->width;
  for (Int j = 0; j < width; ++j) {
    const Field *matrix_col_j = matrix->Pointer(0, j);
    for (Int k = triangular_matrix.height - 1; k >= 0; --k) {
      const Field* tri_col = triangular_matrix.Pointer(0, k);
      Field eta = matrix->Entry(k, j);
      for (Int i = k + 1; i < triangular_matrix.height; ++i) {
        eta -= Conjugate(tri_col[i]) * matrix_col_j[i];
      }
      eta /= Conjugate(tri_col[k]);
      matrix->Entry(k, j) = eta;
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void LeftLowerAdjointTriangularSolves(
    const ConstBlasMatrixView<float>& triangular_matrix,
    BlasMatrixView<float>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'T';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const float alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(strsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void LeftLowerAdjointTriangularSolves(
    const ConstBlasMatrixView<double>& triangular_matrix,
    BlasMatrixView<double>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'T';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const double alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(dtrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void LeftLowerAdjointTriangularSolves(
    const ConstBlasMatrixView<Complex<float>>& triangular_matrix,
    BlasMatrixView<Complex<float>>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'C';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const Complex<float> alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(ctrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexFloat*>(&alpha),
   reinterpret_cast<const BlasComplexFloat*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexFloat*>(matrix->data),
   &leading_dim_blas);
}

template <>
inline void LeftLowerAdjointTriangularSolves(
    const ConstBlasMatrixView<Complex<double>>& triangular_matrix,
    BlasMatrixView<Complex<double>>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'C';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const Complex<double> alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(ztrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexDouble*>(&alpha),
   reinterpret_cast<const BlasComplexDouble*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexDouble*>(matrix->data),
   &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void LeftLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const Int width = matrix->width;
  for (Int j = 0; j < width; ++j) {
    for (Int k = triangular_matrix.height - 1; k >= 0; --k) {
      const Field* triang_column = triangular_matrix.Pointer(0, k);
      Field& eta = matrix->Entry(k, j);
      for (Int i = k + 1; i < triangular_matrix.height; ++i) {
        eta -= Conjugate(triang_column[i]) * matrix->Entry(i, j);
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void LeftLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrixView<float>& triangular_matrix,
    BlasMatrixView<float>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'T';
  const char diag = 'U';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const float alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(strsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void LeftLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrixView<double>& triangular_matrix,
    BlasMatrixView<double>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'T';
  const char diag = 'U';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const double alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(dtrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void LeftLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrixView<Complex<float>>& triangular_matrix,
    BlasMatrixView<Complex<float>>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'C';
  const char diag = 'U';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const Complex<float> alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(ctrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexFloat*>(&alpha),
   reinterpret_cast<const BlasComplexFloat*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexFloat*>(matrix->data),
   &leading_dim_blas);
}

template <>
inline void LeftLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrixView<Complex<double>>& triangular_matrix,
    BlasMatrixView<Complex<double>>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'C';
  const char diag = 'U';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const Complex<double> alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(ztrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexDouble*>(&alpha),
   reinterpret_cast<const BlasComplexDouble*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexDouble*>(matrix->data),
   &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void LeftLowerTransposeUnitTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const Int width = matrix->width;
  for (Int j = 0; j < width; ++j) {
    for (Int k = triangular_matrix.height - 1; k >= 0; --k) {
      const Field* triang_column = triangular_matrix.Pointer(0, k);
      Field& eta = matrix->Entry(k, j);
      for (Int i = k + 1; i < triangular_matrix.height; ++i) {
        eta -= triang_column[i] * matrix->Entry(i, j);
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void LeftLowerTransposeUnitTriangularSolves(
    const ConstBlasMatrixView<float>& triangular_matrix,
    BlasMatrixView<float>* matrix) {
  LeftLowerAdjointUnitTriangularSolves(triangular_matrix, matrix);
}

template <>
inline void LeftLowerTransposeUnitTriangularSolves(
    const ConstBlasMatrixView<double>& triangular_matrix,
    BlasMatrixView<double>* matrix) {
  LeftLowerAdjointUnitTriangularSolves(triangular_matrix, matrix);
}

template <>
inline void LeftLowerTransposeUnitTriangularSolves(
    const ConstBlasMatrixView<Complex<float>>& triangular_matrix,
    BlasMatrixView<Complex<float>>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'T';
  const char diag = 'U';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const Complex<float> alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(ctrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexFloat*>(&alpha),
   reinterpret_cast<const BlasComplexFloat*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexFloat*>(matrix->data),
   &leading_dim_blas);
}

template <>
inline void LeftLowerTransposeUnitTriangularSolves(
    const ConstBlasMatrixView<Complex<double>>& triangular_matrix,
    BlasMatrixView<Complex<double>>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'T';
  const char diag = 'U';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const Complex<double> alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;

  BLAS_SYMBOL(ztrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexDouble*>(&alpha),
   reinterpret_cast<const BlasComplexDouble*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexDouble*>(matrix->data),
   &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void RightLowerAdjointTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix) {
  const Int height = matrix->height;
  const Int width = matrix->width;
  for (Int i = 0; i < height; ++i) {
    for (Int j = 0; j < width; ++j) {
      matrix->Entry(i, j) /= triangular_matrix(j, j);
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
    const ConstBlasMatrixView<float>& triangular_matrix,
    BlasMatrixView<float>* matrix) {
  const char side = 'R';
  const char uplo = 'L';
  const char trans_triang = 'T';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const float alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BLAS_SYMBOL(strsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void RightLowerAdjointTriangularSolves(
    const ConstBlasMatrixView<double>& triangular_matrix,
    BlasMatrixView<double>* matrix) {
  const char side = 'R';
  const char uplo = 'L';
  const char trans_triang = 'T';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const double alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BLAS_SYMBOL(dtrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void RightLowerAdjointTriangularSolves(
    const ConstBlasMatrixView<Complex<float>>& triangular_matrix,
    BlasMatrixView<Complex<float>>* matrix) {
  const char side = 'R';
  const char uplo = 'L';
  const char trans_triang = 'C';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const Complex<float> alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BLAS_SYMBOL(ctrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexFloat*>(&alpha),
   reinterpret_cast<const BlasComplexFloat*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexFloat*>(matrix->data),
   &leading_dim_blas);
}

template <>
inline void RightLowerAdjointTriangularSolves(
    const ConstBlasMatrixView<Complex<double>>& triangular_matrix,
    BlasMatrixView<Complex<double>>* matrix) {
  const char side = 'R';
  const char uplo = 'L';
  const char trans_triang = 'C';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const Complex<double> alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BLAS_SYMBOL(ztrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexDouble*>(&alpha),
   reinterpret_cast<const BlasComplexDouble*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexDouble*>(matrix->data),
   &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void RightLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix) {
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
    const ConstBlasMatrixView<float>& triangular_matrix,
    BlasMatrixView<float>* matrix) {
  const char side = 'R';
  const char uplo = 'L';
  const char trans_triang = 'T';
  const char diag = 'U';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const float alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BLAS_SYMBOL(strsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void RightLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrixView<double>& triangular_matrix,
    BlasMatrixView<double>* matrix) {
  const char side = 'R';
  const char uplo = 'L';
  const char trans_triang = 'T';
  const char diag = 'U';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const double alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BLAS_SYMBOL(dtrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void RightLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrixView<Complex<float>>& triangular_matrix,
    BlasMatrixView<Complex<float>>* matrix) {
  const char side = 'R';
  const char uplo = 'L';
  const char trans_triang = 'C';
  const char diag = 'U';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const Complex<float> alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BLAS_SYMBOL(ctrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexFloat*>(&alpha),
   reinterpret_cast<const BlasComplexFloat*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexFloat*>(matrix->data),
   &leading_dim_blas);
}

template <>
inline void RightLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrixView<Complex<double>>& triangular_matrix,
    BlasMatrixView<Complex<double>>* matrix) {
  const char side = 'R';
  const char uplo = 'L';
  const char trans_triang = 'C';
  const char diag = 'U';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const Complex<double> alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BLAS_SYMBOL(ztrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexDouble*>(&alpha),
   reinterpret_cast<const BlasComplexDouble*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexDouble*>(matrix->data),
   &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void RightDiagonalTimesLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix) {
  const Int height = matrix->height;
  const Int width = matrix->width;
  for (Int i = 0; i < height; ++i) {
    for (Int j = 0; j < width; ++j) {
      const Field eta = matrix->Entry(i, j);
      for (Int k = j + 1; k < width; ++k) {
        matrix->Entry(i, k) -= eta * Conjugate(triangular_matrix(k, j));
      }
      matrix->Entry(i, j) /= triangular_matrix(j, j);
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void RightDiagonalTimesLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrixView<float>& triangular_matrix,
    BlasMatrixView<float>* matrix) {
  RightLowerAdjointUnitTriangularSolves(triangular_matrix, matrix);

  // Solve against the diagonal.
  const Int height = matrix->height;
  const Int width = matrix->width;
  for (BlasInt j = 0; j < width; ++j) {
    for (BlasInt i = 0; i < height; ++i) {
      matrix->Entry(i, j) /= triangular_matrix(j, j);
    }
  }
}

template <>
inline void RightDiagonalTimesLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrixView<double>& triangular_matrix,
    BlasMatrixView<double>* matrix) {
  RightLowerAdjointUnitTriangularSolves(triangular_matrix, matrix);

  // Solve against the diagonal.
  const Int height = matrix->height;
  const Int width = matrix->width;
  for (BlasInt j = 0; j < width; ++j) {
    for (BlasInt i = 0; i < height; ++i) {
      matrix->Entry(i, j) /= triangular_matrix(j, j);
    }
  }
}

template <>
inline void RightDiagonalTimesLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrixView<Complex<float>>& triangular_matrix,
    BlasMatrixView<Complex<float>>* matrix) {
  RightLowerAdjointUnitTriangularSolves(triangular_matrix, matrix);

  // Solve against the diagonal.
  const Int height = matrix->height;
  const Int width = matrix->width;
  for (BlasInt j = 0; j < width; ++j) {
    for (BlasInt i = 0; i < height; ++i) {
      matrix->Entry(i, j) /= triangular_matrix(j, j);
    }
  }
}

template <>
inline void RightDiagonalTimesLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrixView<Complex<double>>& triangular_matrix,
    BlasMatrixView<Complex<double>>* matrix) {
  RightLowerAdjointUnitTriangularSolves(triangular_matrix, matrix);

  // Solve against the diagonal.
  const Int height = matrix->height;
  const Int width = matrix->width;
  for (BlasInt j = 0; j < width; ++j) {
    for (BlasInt i = 0; i < height; ++i) {
      matrix->Entry(i, j) /= triangular_matrix(j, j);
    }
  }
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void RightLowerTransposeTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix) {
  const Int height = matrix->height;
  const Int width = matrix->width;
  for (Int i = 0; i < height; ++i) {
    for (Int j = 0; j < width; ++j) {
      matrix->Entry(i, j) /= triangular_matrix(j, j);
      const Field eta = matrix->Entry(i, j);
      for (Int k = j + 1; k < width; ++k) {
        matrix->Entry(i, k) -= eta * triangular_matrix(k, j);
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void RightLowerTransposeTriangularSolves(
    const ConstBlasMatrixView<float>& triangular_matrix,
    BlasMatrixView<float>* matrix) {
  RightLowerAdjointTriangularSolves(triangular_matrix, matrix);
}

template <>
inline void RightLowerTransposeTriangularSolves(
    const ConstBlasMatrixView<double>& triangular_matrix,
    BlasMatrixView<double>* matrix) {
  RightLowerAdjointTriangularSolves(triangular_matrix, matrix);
}

template <>
inline void RightLowerTransposeTriangularSolves(
    const ConstBlasMatrixView<Complex<float>>& triangular_matrix,
    BlasMatrixView<Complex<float>>* matrix) {
  const char side = 'R';
  const char uplo = 'L';
  const char trans_triang = 'T';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const Complex<float> alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BLAS_SYMBOL(ctrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexFloat*>(&alpha),
   reinterpret_cast<const BlasComplexFloat*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexFloat*>(matrix->data),
   &leading_dim_blas);
}

template <>
inline void RightLowerTransposeTriangularSolves(
    const ConstBlasMatrixView<Complex<double>>& triangular_matrix,
    BlasMatrixView<Complex<double>>* matrix) {
  const char side = 'R';
  const char uplo = 'L';
  const char trans_triang = 'T';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const Complex<double> alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BLAS_SYMBOL(ztrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexDouble*>(&alpha),
   reinterpret_cast<const BlasComplexDouble*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexDouble*>(matrix->data),
   &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void RightLowerTransposeUnitTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix) {
  const Int height = matrix->height;
  const Int width = matrix->width;
  for (Int i = 0; i < height; ++i) {
    for (Int j = 0; j < width; ++j) {
      const Field eta = matrix->Entry(i, j);
      for (Int k = j + 1; k < width; ++k) {
        matrix->Entry(i, k) -= eta * triangular_matrix(k, j);
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void RightLowerTransposeUnitTriangularSolves(
    const ConstBlasMatrixView<float>& triangular_matrix,
    BlasMatrixView<float>* matrix) {
  RightLowerAdjointUnitTriangularSolves(triangular_matrix, matrix);
}

template <>
inline void RightLowerTransposeUnitTriangularSolves(
    const ConstBlasMatrixView<double>& triangular_matrix,
    BlasMatrixView<double>* matrix) {
  RightLowerAdjointUnitTriangularSolves(triangular_matrix, matrix);
}

template <>
inline void RightLowerTransposeUnitTriangularSolves(
    const ConstBlasMatrixView<Complex<float>>& triangular_matrix,
    BlasMatrixView<Complex<float>>* matrix) {
  const char side = 'R';
  const char uplo = 'L';
  const char trans_triang = 'T';
  const char diag = 'U';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const Complex<float> alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BLAS_SYMBOL(ctrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexFloat*>(&alpha),
   reinterpret_cast<const BlasComplexFloat*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexFloat*>(matrix->data),
   &leading_dim_blas);
}

template <>
inline void RightLowerTransposeUnitTriangularSolves(
    const ConstBlasMatrixView<Complex<double>>& triangular_matrix,
    BlasMatrixView<Complex<double>>* matrix) {
  const char side = 'R';
  const char uplo = 'L';
  const char trans_triang = 'T';
  const char diag = 'U';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const Complex<double> alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BLAS_SYMBOL(ztrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexDouble*>(&alpha),
   reinterpret_cast<const BlasComplexDouble*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexDouble*>(matrix->data),
   &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void RightDiagonalTimesLowerTransposeUnitTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix) {
  const Int height = matrix->height;
  const Int width = matrix->width;
  for (Int i = 0; i < height; ++i) {
    for (Int j = 0; j < width; ++j) {
      const Field eta = matrix->Entry(i, j);
      for (Int k = j + 1; k < width; ++k) {
        matrix->Entry(i, k) -= eta * triangular_matrix(k, j);
      }
      matrix->Entry(i, j) /= triangular_matrix(j, j);
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void RightDiagonalTimesLowerTransposeUnitTriangularSolves(
    const ConstBlasMatrixView<float>& triangular_matrix,
    BlasMatrixView<float>* matrix) {
  RightDiagonalTimesLowerAdjointUnitTriangularSolves(triangular_matrix, matrix);
}

template <>
inline void RightDiagonalTimesLowerTransposeUnitTriangularSolves(
    const ConstBlasMatrixView<double>& triangular_matrix,
    BlasMatrixView<double>* matrix) {
  RightDiagonalTimesLowerAdjointUnitTriangularSolves(triangular_matrix, matrix);
}

template <>
inline void RightDiagonalTimesLowerTransposeUnitTriangularSolves(
    const ConstBlasMatrixView<Complex<float>>& triangular_matrix,
    BlasMatrixView<Complex<float>>* matrix) {
  RightLowerTransposeUnitTriangularSolves(triangular_matrix, matrix);

  // Solve against the diagonal.
  const Int height = matrix->height;
  const Int width = matrix->width;
  for (BlasInt j = 0; j < width; ++j) {
    for (BlasInt i = 0; i < height; ++i) {
      matrix->Entry(i, j) /= triangular_matrix(j, j);
    }
  }
}

template <>
inline void RightDiagonalTimesLowerTransposeUnitTriangularSolves(
    const ConstBlasMatrixView<Complex<double>>& triangular_matrix,
    BlasMatrixView<Complex<double>>* matrix) {
  RightLowerTransposeUnitTriangularSolves(triangular_matrix, matrix);

  // Solve against the diagonal.
  const Int height = matrix->height;
  const Int width = matrix->width;
  for (BlasInt j = 0; j < width; ++j) {
    for (BlasInt i = 0; i < height; ++i) {
      matrix->Entry(i, j) /= triangular_matrix(j, j);
    }
  }
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void RightUpperTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix) {
  const Int height = matrix->height;
  const Int width = matrix->width;
  for (Int i = 0; i < height; ++i) {
    for (Int j = 0; j < width; ++j) {
      matrix->Entry(i, j) /= triangular_matrix(j, j);
      const Field eta = matrix->Entry(i, j);
      for (Int k = j + 1; k < width; ++k) {
        matrix->Entry(i, k) -= eta * triangular_matrix(j, k);
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void RightUpperTriangularSolves(
    const ConstBlasMatrixView<float>& triangular_matrix,
    BlasMatrixView<float>* matrix) {
  const char side = 'R';
  const char uplo = 'U';
  const char trans_triang = 'N';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const float alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BLAS_SYMBOL(strsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void RightUpperTriangularSolves(
    const ConstBlasMatrixView<double>& triangular_matrix,
    BlasMatrixView<double>* matrix) {
  const char side = 'R';
  const char uplo = 'U';
  const char trans_triang = 'N';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const double alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BLAS_SYMBOL(dtrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void RightUpperTriangularSolves(
    const ConstBlasMatrixView<Complex<float>>& triangular_matrix,
    BlasMatrixView<Complex<float>>* matrix) {
  const char side = 'R';
  const char uplo = 'U';
  const char trans_triang = 'N';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const Complex<float> alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BLAS_SYMBOL(ctrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexFloat*>(&alpha),
   reinterpret_cast<const BlasComplexFloat*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexFloat*>(matrix->data),
   &leading_dim_blas);
}

template <>
inline void RightUpperTriangularSolves(
    const ConstBlasMatrixView<Complex<double>>& triangular_matrix,
    BlasMatrixView<Complex<double>>* matrix) {
  const char side = 'R';
  const char uplo = 'U';
  const char trans_triang = 'N';
  const char diag = 'N';
  const BlasInt height_blas = matrix->height;
  const BlasInt width_blas = matrix->width;
  const Complex<double> alpha = 1;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt leading_dim_blas = matrix->leading_dim;
  BLAS_SYMBOL(ztrsm)
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas,
   reinterpret_cast<const BlasComplexDouble*>(&alpha),
   reinterpret_cast<const BlasComplexDouble*>(triangular_matrix.data),
   &triang_leading_dim_blas, reinterpret_cast<BlasComplexDouble*>(matrix->data),
   &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

// In-place permutation
// Perm can be, e.g., Buffer<Int>, ConstBlasMatrixView<Int>
template <class Perm, class Field>
void Permute(const Perm &permutation, BlasMatrixView<Field>* matrix) {
  Buffer<Field> column_copy(matrix->height);
  for (Int j = 0; j < matrix->width; ++j) {
    // Make a copy of the current column.
    for (Int i = 0; i < matrix->height; ++i) {
      column_copy[i] = matrix->Entry(i, j);
    }

    // Apply the permutation.
    for (Int i = 0; i < matrix->height; ++i) {
      matrix->Entry(permutation[i], j) = column_copy[i];
    }
  }
}

// Out-of-place permutation
// Perm can be, e.g., Buffer<Int>, ConstBlasMatrixView<Int>
template <class Perm, class Field>
void Permute(const Perm &permutation, const BlasMatrixView<Field> &in, BlasMatrixView<Field> *out) {
    if (in.width != out->width || in.height != out->height) throw std::runtime_error("Size mismatch");
    for (Int j = 0; j < out->width; ++j) {
        // Apply the permutation.
        const Field *in_ptr = in.Pointer(0, j);
        Field *out_ptr = out->Pointer(0, j);
        //parallel_for_range(out->height, [&](Int i) {
        for (Int i = 0; i < out->height; ++i)
            out_ptr[permutation[i]] = in_ptr[i];
        //});
    }
}

template <class Field>
void PermuteColumns(const Buffer<Int>& permutation,
                    BlasMatrixView<Field>* matrix) {
  Buffer<Field> row_copy(matrix->width);
  for (Int i = 0; i < matrix->height; ++i) {
    // Make a copy of the current row.
    for (Int j = 0; j < matrix->width; ++j) {
      row_copy[j] = matrix->Entry(i, j);
    }

    // Apply the permutation.
    for (Int j = 0; j < matrix->width; ++j) {
      matrix->Entry(i, permutation[j]) = row_copy[j];
    }
  }
}

template <class Field>
void PermuteColumns(const ConstBlasMatrixView<Int>& permutation,
                    BlasMatrixView<Field>* matrix) {
  Buffer<Field> row_copy(matrix->width);
  for (Int i = 0; i < matrix->height; ++i) {
    // Make a copy of the current row.
    for (Int j = 0; j < matrix->width; ++j) {
      row_copy[j] = matrix->Entry(i, j);
    }

    // Apply the permutation.
    for (Int j = 0; j < matrix->width; ++j) {
      matrix->Entry(i, permutation(j)) = row_copy[j];
    }
  }
}

template <class Field>
void InversePermute(const Buffer<Int>& permutation,
                    BlasMatrixView<Field>* matrix) {
  Buffer<Field> column_copy(matrix->height);
  for (Int j = 0; j < matrix->width; ++j) {
    // Make a copy of the current column.
    for (Int i = 0; i < matrix->height; ++i) {
      column_copy[i] = matrix->Entry(i, j);
    }

    // Apply the inverse permutation.
    for (Int i = 0; i < matrix->height; ++i) {
      matrix->Entry(i, j) = column_copy[permutation[i]];
    }
  }
}

template <class Field>
void InversePermute(const ConstBlasMatrixView<Int>& permutation,
                    BlasMatrixView<Field>* matrix) {
  Buffer<Field> column_copy(matrix->height);
  for (Int j = 0; j < matrix->width; ++j) {
    // Make a copy of the current column.
    for (Int i = 0; i < matrix->height; ++i) {
      column_copy[i] = matrix->Entry(i, j);
    }

    // Apply the inverse permutation.
    for (Int i = 0; i < matrix->height; ++i) {
      matrix->Entry(i, j) = column_copy[permutation(i)];
    }
  }
}

template <class Field>
void InversePermuteColumns(const Buffer<Int>& permutation,
                           BlasMatrixView<Field>* matrix) {
  Buffer<Field> row_copy(matrix->width);
  for (Int i = 0; i < matrix->height; ++i) {
    // Make a copy of the current row.
    for (Int j = 0; j < matrix->width; ++j) {
      row_copy[j] = matrix->Entry(i, j);
    }

    // Apply the inverse permutation.
    for (Int j = 0; j < matrix->width; ++j) {
      matrix->Entry(i, j) = row_copy[permutation[j]];
    }
  }
}

template <class Field>
void InversePermuteColumns(const ConstBlasMatrixView<Int>& permutation,
                           BlasMatrixView<Field>* matrix) {
  Buffer<Field> row_copy(matrix->width);
  for (Int i = 0; i < matrix->height; ++i) {
    // Make a copy of the current row.
    for (Int j = 0; j < matrix->width; ++j) {
      row_copy[j] = matrix->Entry(i, j);
    }

    // Apply the inverse permutation.
    for (Int j = 0; j < matrix->width; ++j) {
      matrix->Entry(i, j) = row_copy[permutation(j)];
    }
  }
}

}  // namespace catamari

#include "catamari/dense_basic_linear_algebra/openmp-impl.hpp"

#endif  // ifndef CATAMARI_DENSE_BASIC_LINEAR_ALGEBRA_IMPL_H_
