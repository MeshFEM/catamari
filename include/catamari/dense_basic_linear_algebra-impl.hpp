/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_BASIC_LINEAR_ALGEBRA_IMPL_H_
#define CATAMARI_DENSE_BASIC_LINEAR_ALGEBRA_IMPL_H_

#include "catamari/blas.hpp"
#include "catamari/macros.hpp"

#include "catamari/dense_basic_linear_algebra.hpp"

namespace catamari {

template <class Field>
void ConjugateMatrix(BlasMatrix<Field>* matrix) {
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
                         const ConstBlasMatrix<Field>& matrix,
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
inline void MatrixVectorProduct(const float& alpha,
                                const ConstBlasMatrix<float>& matrix,
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
                                const ConstBlasMatrix<double>& matrix,
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
inline void MatrixVectorProduct(const Complex<float>& alpha,
                                const ConstBlasMatrix<Complex<float>>& matrix,
                                const Complex<float>* input_vector,
                                Complex<float>* result) {
  const char trans = 'N';
  const BlasInt height_blas = matrix.height;
  const BlasInt width_blas = matrix.width;
  const BlasInt leading_dim_blas = matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  const BlasComplexFloat beta = 1;
  BLAS_SYMBOL(cgemv)
  (&trans, &height_blas, &width_blas, &alpha, matrix.data, &leading_dim_blas,
   input_vector, &unit_stride_blas, &beta, result, &unit_stride_blas);
}

template <>
inline void MatrixVectorProduct(const Complex<double>& alpha,
                                const ConstBlasMatrix<Complex<double>>& matrix,
                                const Complex<double>* input_vector,
                                Complex<double>* result) {
  const char trans = 'N';
  const BlasInt height_blas = matrix.height;
  const BlasInt width_blas = matrix.width;
  const BlasInt leading_dim_blas = matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  const BlasComplexDouble beta = 1;
  BLAS_SYMBOL(zgemv)
  (&trans, &height_blas, &width_blas, &alpha, matrix.data, &leading_dim_blas,
   input_vector, &unit_stride_blas, &beta, result, &unit_stride_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void ConjugateMatrixVectorProduct(const Field& alpha,
                                  const ConstBlasMatrix<Field>& matrix,
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
inline void ConjugateMatrixVectorProduct(const float& alpha,
                                         const ConstBlasMatrix<float>& matrix,
                                         const float* input_vector,
                                         float* result) {
  MatrixVectorProduct(alpha, matrix, input_vector, result);
}

template <>
inline void ConjugateMatrixVectorProduct(const double& alpha,
                                         const ConstBlasMatrix<double>& matrix,
                                         const double* input_vector,
                                         double* result) {
  MatrixVectorProduct(alpha, matrix, input_vector, result);
}

// Make use of the equivalence:
//
//   alpha conj(A) x + y = conj(conj(alpha) A conj(x) + conj(y))
//
// in order to make use of GEMV for complex inputs.

template <>
inline void ConjugateMatrixVectorProduct(
    const Complex<float>& alpha, const ConstBlasMatrix<Complex<float>>& matrix,
    const Complex<float>* input_vector, Complex<float>* result) {
  // Make a conjugated copy of the input vector.
  std::vector<Complex<float>> conjugated_input(matrix.width);
  for (Int i = 0; i < matrix.width; ++i) {
    conjugated_input[i] = Conjugate(input_vector[i]);
  }

  BlasMatrix<Complex<float>> result_matrix;
  result_matrix.height = matrix.height;
  result_matrix.width = 1;
  result_matrix.leading_dim = matrix.height;
  result_matrix.data = result;

  ConjugateMatrix(&result_matrix);
  MatrixVectorProduct(Conjugate(alpha), matrix, conjugated_input.data(),
                      result);
  ConjugateMatrix(&result_matrix);
}

template <>
inline void ConjugateMatrixVectorProduct(
    const Complex<double>& alpha,
    const ConstBlasMatrix<Complex<double>>& matrix,
    const Complex<double>* input_vector, Complex<double>* result) {
  // Make a conjugated copy of the input vector.
  std::vector<Complex<double>> conjugated_input(matrix.width);
  for (Int i = 0; i < matrix.width; ++i) {
    conjugated_input[i] = Conjugate(input_vector[i]);
  }

  BlasMatrix<Complex<double>> result_matrix;
  result_matrix.height = matrix.height;
  result_matrix.width = 1;
  result_matrix.leading_dim = matrix.height;
  result_matrix.data = result;

  ConjugateMatrix(&result_matrix);
  MatrixVectorProduct(Conjugate(alpha), matrix, conjugated_input.data(),
                      result);
  ConjugateMatrix(&result_matrix);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void TransposeMatrixVectorProduct(const Field& alpha,
                                  const ConstBlasMatrix<Field>& matrix,
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
inline void TransposeMatrixVectorProduct(const float& alpha,
                                         const ConstBlasMatrix<float>& matrix,
                                         const float* input_vector,
                                         float* result) {
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
inline void TransposeMatrixVectorProduct(const double& alpha,
                                         const ConstBlasMatrix<double>& matrix,
                                         const double* input_vector,
                                         double* result) {
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
    const Complex<float>& alpha, const ConstBlasMatrix<Complex<float>>& matrix,
    const Complex<float>* input_vector, Complex<float>* result) {
  const char trans = 'T';
  const BlasInt height_blas = matrix.height;
  const BlasInt width_blas = matrix.width;
  const BlasInt leading_dim_blas = matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  const Complex<float> beta = 1;
  BLAS_SYMBOL(cgemv)
  (&trans, &height_blas, &width_blas, &alpha, matrix.data, &leading_dim_blas,
   input_vector, &unit_stride_blas, &beta, result, &unit_stride_blas);
}

template <>
inline void TransposeMatrixVectorProduct(
    const Complex<double>& alpha,
    const ConstBlasMatrix<Complex<double>>& matrix,
    const Complex<double>* input_vector, Complex<double>* result) {
  const char trans = 'T';
  const BlasInt height_blas = matrix.height;
  const BlasInt width_blas = matrix.width;
  const BlasInt leading_dim_blas = matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  const Complex<double> beta = 1;
  BLAS_SYMBOL(zgemv)
  (&trans, &height_blas, &width_blas, &alpha, matrix.data, &leading_dim_blas,
   input_vector, &unit_stride_blas, &beta, result, &unit_stride_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

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

template <>
inline void TriangularSolveLeftLower(
    const ConstBlasMatrix<Complex<float>>& triangular_matrix,
    Complex<float>* vector) {
  const char uplo = 'L';
  const char trans = 'N';
  const char diag = 'N';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(ctrsv)
  (&uplo, &trans, &diag, &height_blas, triangular_matrix.data,
   &triang_leading_dim_blas, vector, &unit_stride_blas);
}

template <>
inline void TriangularSolveLeftLower(
    const ConstBlasMatrix<Complex<double>>& triangular_matrix,
    Complex<double>* vector) {
  const char uplo = 'L';
  const char trans = 'N';
  const char diag = 'N';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(ztrsv)
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

template <>
inline void TriangularSolveLeftLowerUnit(
    const ConstBlasMatrix<Complex<float>>& triangular_matrix,
    Complex<float>* vector) {
  const char uplo = 'L';
  const char trans = 'N';
  const char diag = 'U';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(ctrsv)
  (&uplo, &trans, &diag, &height_blas, triangular_matrix.data,
   &triang_leading_dim_blas, vector, &unit_stride_blas);
}

template <>
inline void TriangularSolveLeftLowerUnit(
    const ConstBlasMatrix<Complex<double>>& triangular_matrix,
    Complex<double>* vector) {
  const char uplo = 'L';
  const char trans = 'N';
  const char diag = 'U';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(ztrsv)
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

template <>
inline void TriangularSolveLeftLowerAdjoint(
    const ConstBlasMatrix<Complex<float>>& triangular_matrix,
    Complex<float>* vector) {
  const char uplo = 'L';
  const char trans = 'C';
  const char diag = 'N';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(ctrsv)
  (&uplo, &trans, &diag, &height_blas, triangular_matrix.data,
   &triang_leading_dim_blas, vector, &unit_stride_blas);
}

template <>
inline void TriangularSolveLeftLowerAdjoint(
    const ConstBlasMatrix<Complex<double>>& triangular_matrix,
    Complex<double>* vector) {
  const char uplo = 'L';
  const char trans = 'C';
  const char diag = 'N';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(ztrsv)
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

template <>
inline void TriangularSolveLeftLowerAdjointUnit(
    const ConstBlasMatrix<Complex<float>>& triangular_matrix,
    Complex<float>* vector) {
  const char uplo = 'L';
  const char trans = 'C';
  const char diag = 'U';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(ctrsv)
  (&uplo, &trans, &diag, &height_blas, triangular_matrix.data,
   &triang_leading_dim_blas, vector, &unit_stride_blas);
}

template <>
inline void TriangularSolveLeftLowerAdjointUnit(
    const ConstBlasMatrix<Complex<double>>& triangular_matrix,
    Complex<double>* vector) {
  const char uplo = 'L';
  const char trans = 'C';
  const char diag = 'U';
  const BlasInt height_blas = triangular_matrix.height;
  const BlasInt triang_leading_dim_blas = triangular_matrix.leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(ztrsv)
  (&uplo, &trans, &diag, &height_blas, triangular_matrix.data,
   &triang_leading_dim_blas, vector, &unit_stride_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void MatrixMultiplyNormalNormal(const Field& alpha,
                                const ConstBlasMatrix<Field>& left_matrix,
                                const ConstBlasMatrix<Field>& right_matrix,
                                const Field& beta,
                                BlasMatrix<Field>* output_matrix) {
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
    const float& alpha, const ConstBlasMatrix<float>& left_matrix,
    const ConstBlasMatrix<float>& right_matrix, const float& beta,
    BlasMatrix<float>* output_matrix) {
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
    const double& alpha, const ConstBlasMatrix<double>& left_matrix,
    const ConstBlasMatrix<double>& right_matrix, const double& beta,
    BlasMatrix<double>* output_matrix) {
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
    const ConstBlasMatrix<Complex<float>>& left_matrix,
    const ConstBlasMatrix<Complex<float>>& right_matrix,
    const Complex<float>& beta, BlasMatrix<Complex<float>>* output_matrix) {
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
   &contraction_size_blas, &alpha, left_matrix.data, &left_leading_dim_blas,
   right_matrix.data, &right_leading_dim_blas, &beta, output_matrix->data,
   &output_leading_dim_blas);
}

template <>
inline void MatrixMultiplyNormalNormal(
    const Complex<double>& alpha,
    const ConstBlasMatrix<Complex<double>>& left_matrix,
    const ConstBlasMatrix<Complex<double>>& right_matrix,
    const Complex<double>& beta, BlasMatrix<Complex<double>>* output_matrix) {
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
  BLAS_SYMBOL(zgemm)
  (&trans_left, &trans_right, &output_height_blas, &output_width_blas,
   &contraction_size_blas, &alpha, left_matrix.data, &left_leading_dim_blas,
   right_matrix.data, &right_leading_dim_blas, &beta, output_matrix->data,
   &output_leading_dim_blas);
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

template <>
inline void MatrixMultiplyTransposeNormal(
    const Complex<float>& alpha,
    const ConstBlasMatrix<Complex<float>>& left_matrix,
    const ConstBlasMatrix<Complex<float>>& right_matrix,
    const Complex<float>& beta, BlasMatrix<Complex<float>>* output_matrix) {
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
   &contraction_size_blas, &alpha, left_matrix.data, &left_leading_dim_blas,
   right_matrix.data, &right_leading_dim_blas, &beta, output_matrix->data,
   &output_leading_dim_blas);
}

template <>
inline void MatrixMultiplyTransposeNormal(
    const Complex<double>& alpha,
    const ConstBlasMatrix<Complex<double>>& left_matrix,
    const ConstBlasMatrix<Complex<double>>& right_matrix,
    const Complex<double>& beta, BlasMatrix<Complex<double>>* output_matrix) {
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
   &contraction_size_blas, &alpha, left_matrix.data, &left_leading_dim_blas,
   right_matrix.data, &right_leading_dim_blas, &beta, output_matrix->data,
   &output_leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void MatrixMultiplyAdjointNormal(const Field& alpha,
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
        output_entry +=
            alpha * Conjugate(left_matrix(k, i)) * right_matrix(k, j);
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void MatrixMultiplyAdjointNormal(
    const float& alpha, const ConstBlasMatrix<float>& left_matrix,
    const ConstBlasMatrix<float>& right_matrix, const float& beta,
    BlasMatrix<float>* output_matrix) {
  MatrixMultiplyTransposeNormal(alpha, left_matrix, right_matrix, beta,
                                output_matrix);
}

template <>
inline void MatrixMultiplyAdjointNormal(
    const double& alpha, const ConstBlasMatrix<double>& left_matrix,
    const ConstBlasMatrix<double>& right_matrix, const double& beta,
    BlasMatrix<double>* output_matrix) {
  MatrixMultiplyTransposeNormal(alpha, left_matrix, right_matrix, beta,
                                output_matrix);
}

template <>
inline void MatrixMultiplyAdjointNormal(
    const Complex<float>& alpha,
    const ConstBlasMatrix<Complex<float>>& left_matrix,
    const ConstBlasMatrix<Complex<float>>& right_matrix,
    const Complex<float>& beta, BlasMatrix<Complex<float>>* output_matrix) {
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
   &contraction_size_blas, &alpha, left_matrix.data, &left_leading_dim_blas,
   right_matrix.data, &right_leading_dim_blas, &beta, output_matrix->data,
   &output_leading_dim_blas);
}

template <>
inline void MatrixMultiplyAdjointNormal(
    const Complex<double>& alpha,
    const ConstBlasMatrix<Complex<double>>& left_matrix,
    const ConstBlasMatrix<Complex<double>>& right_matrix,
    const Complex<double>& beta, BlasMatrix<Complex<double>>* output_matrix) {
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
   &contraction_size_blas, &alpha, left_matrix.data, &left_leading_dim_blas,
   right_matrix.data, &right_leading_dim_blas, &beta, output_matrix->data,
   &output_leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void LowerNormalHermitianOuterProduct(const ComplexBase<Field>& alpha,
                                      const ConstBlasMatrix<Field>& left_matrix,
                                      const ComplexBase<Field>& beta,
                                      BlasMatrix<Field>* output_matrix) {
  const Int output_height = output_matrix->height;
  const Int output_width = output_matrix->width;
  const Int contraction_size = left_matrix.width;
  for (Int j = 0; j < output_height; ++j) {
    for (Int i = j; i < output_height; ++i) {
      Field& output_entry = output_matrix->Entry(i, j);
      output_entry *= beta;
      for (Int k = 0; k < contraction_size; ++k) {
        output_entry +=
            alpha * left_matrix(i, k) * Conjugate(left_matrix(j, k));
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void LowerNormalHermitianOuterProduct(
    const float& alpha, const ConstBlasMatrix<float>& left_matrix,
    const float& beta, BlasMatrix<float>* output_matrix) {
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
    const double& beta, BlasMatrix<double>* output_matrix) {
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
    const float& alpha, const ConstBlasMatrix<Complex<float>>& left_matrix,
    const float& beta, BlasMatrix<Complex<float>>* output_matrix) {
  const char uplo = 'L';
  const char trans = 'N';
  const BlasInt height_blas = output_matrix->height;
  const BlasInt rank_blas = left_matrix.width;
  const BlasInt factor_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(cherk)
  (&uplo, &trans, &height_blas, &rank_blas, &alpha, left_matrix.data,
   &factor_leading_dim_blas, &beta, output_matrix->data, &leading_dim_blas);
}

template <>
inline void LowerNormalHermitianOuterProduct(
    const double& alpha, const ConstBlasMatrix<Complex<double>>& left_matrix,
    const double& beta, BlasMatrix<Complex<double>>* output_matrix) {
  const char uplo = 'L';
  const char trans = 'N';
  const BlasInt height_blas = output_matrix->height;
  const BlasInt rank_blas = left_matrix.width;
  const BlasInt factor_leading_dim_blas = left_matrix.leading_dim;
  const BlasInt leading_dim_blas = output_matrix->leading_dim;
  BLAS_SYMBOL(zherk)
  (&uplo, &trans, &height_blas, &rank_blas, &alpha, left_matrix.data,
   &factor_leading_dim_blas, &beta, output_matrix->data, &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void MatrixMultiplyLowerNormalNormal(const Field& alpha,
                                     const ConstBlasMatrix<Field>& left_matrix,
                                     const ConstBlasMatrix<Field>& right_matrix,
                                     const Field& beta,
                                     BlasMatrix<Field>* output_matrix) {
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
    const float& alpha, const ConstBlasMatrix<float>& left_matrix,
    const ConstBlasMatrix<float>& right_matrix, const float& beta,
    BlasMatrix<float>* output_matrix) {
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
    const double& alpha, const ConstBlasMatrix<double>& left_matrix,
    const ConstBlasMatrix<double>& right_matrix, const double& beta,
    BlasMatrix<double>* output_matrix) {
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
    const ConstBlasMatrix<Complex<float>>& left_matrix,
    const ConstBlasMatrix<Complex<float>>& right_matrix,
    const Complex<float>& beta, BlasMatrix<Complex<float>>* output_matrix) {
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
    const ConstBlasMatrix<Complex<double>>& left_matrix,
    const ConstBlasMatrix<Complex<double>>& right_matrix,
    const Complex<double>& beta, BlasMatrix<Complex<double>>* output_matrix) {
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
    const float& alpha, const ConstBlasMatrix<float>& left_matrix,
    const ConstBlasMatrix<float>& right_matrix, const float& beta,
    BlasMatrix<float>* output_matrix) {
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
    const double& alpha, const ConstBlasMatrix<double>& left_matrix,
    const ConstBlasMatrix<double>& right_matrix, const double& beta,
    BlasMatrix<double>* output_matrix) {
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
    const ConstBlasMatrix<Complex<float>>& left_matrix,
    const ConstBlasMatrix<Complex<float>>& right_matrix,
    const Complex<float>& beta, BlasMatrix<Complex<float>>* output_matrix) {
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
    const ConstBlasMatrix<Complex<double>>& left_matrix,
    const ConstBlasMatrix<Complex<double>>& right_matrix,
    const Complex<double>& beta, BlasMatrix<Complex<double>>* output_matrix) {
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

template <>
inline void MatrixMultiplyLowerNormalTranspose(
    const Complex<float>& alpha,
    const ConstBlasMatrix<Complex<float>>& left_matrix,
    const ConstBlasMatrix<Complex<float>>& right_matrix,
    const Complex<float>& beta, BlasMatrix<Complex<float>>* output_matrix) {
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
    const ConstBlasMatrix<Complex<double>>& left_matrix,
    const ConstBlasMatrix<Complex<double>>& right_matrix,
    const Complex<double>& beta, BlasMatrix<Complex<double>>* output_matrix) {
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

template <>
inline void MatrixMultiplyLowerNormalTranspose(
    const Complex<float>& alpha,
    const ConstBlasMatrix<Complex<float>>& left_matrix,
    const ConstBlasMatrix<Complex<float>>& right_matrix,
    const Complex<float>& beta, BlasMatrix<Complex<float>>* output_matrix) {
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
    const ConstBlasMatrix<Complex<double>>& left_matrix,
    const ConstBlasMatrix<Complex<double>>& right_matrix,
    const Complex<double>& beta, BlasMatrix<Complex<double>>* output_matrix) {
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

template <>
inline void MatrixMultiplyLowerTransposeNormal(
    const Complex<float>& alpha,
    const ConstBlasMatrix<Complex<float>>& left_matrix,
    const ConstBlasMatrix<Complex<float>>& right_matrix,
    const Complex<float>& beta, BlasMatrix<Complex<float>>* output_matrix) {
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
    const ConstBlasMatrix<Complex<double>>& left_matrix,
    const ConstBlasMatrix<Complex<double>>& right_matrix,
    const Complex<double>& beta, BlasMatrix<Complex<double>>* output_matrix) {
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
    const double& alpha, const ConstBlasMatrix<double>& left_matrix,
    const ConstBlasMatrix<double>& right_matrix, const double& beta,
    BlasMatrix<double>* output_matrix) {
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
    const ConstBlasMatrix<Complex<float>>& left_matrix,
    const ConstBlasMatrix<Complex<float>>& right_matrix,
    const Complex<float>& beta, BlasMatrix<Complex<float>>* output_matrix) {
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
    const ConstBlasMatrix<Complex<double>>& left_matrix,
    const ConstBlasMatrix<Complex<double>>& right_matrix,
    const Complex<double>& beta, BlasMatrix<Complex<double>>* output_matrix) {
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
void LeftLowerTriangularSolves(const ConstBlasMatrix<Field>& triangular_matrix,
                               BlasMatrix<Field>* matrix) {
  CATAMARI_ASSERT(triangular_matrix.height == triangular_matrix.width &&
                      triangular_matrix.height == matrix->height,
                  "Incompatible matrix dimensions");
  const Int width = matrix->width;
  for (Int j = 0; j < width; ++j) {
    Field* input_column = matrix->Pointer(0, j);

    // Interleave with a subsequent solve against D.
    for (Int i = 0; i < triangular_matrix.height; ++i) {
      const Field* l_column = triangular_matrix.Pointer(0, i);
      input_column[i] /= l_column[i];
      for (Int k = i + 1; k < triangular_matrix.height; ++k) {
        input_column[k] -= l_column[k] * input_column[i];
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void LeftLowerTriangularSolves(
    const ConstBlasMatrix<float>& triangular_matrix,
    BlasMatrix<float>* matrix) {
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
    const ConstBlasMatrix<double>& triangular_matrix,
    BlasMatrix<double>* matrix) {
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
    const ConstBlasMatrix<Complex<float>>& triangular_matrix,
    BlasMatrix<Complex<float>>* matrix) {
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
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void LeftLowerTriangularSolves(
    const ConstBlasMatrix<Complex<double>>& triangular_matrix,
    BlasMatrix<Complex<double>>* matrix) {
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
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void LeftLowerUnitTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix,
    BlasMatrix<Field>* matrix) {
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
    const ConstBlasMatrix<float>& triangular_matrix,
    BlasMatrix<float>* matrix) {
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
    const ConstBlasMatrix<double>& triangular_matrix,
    BlasMatrix<double>* matrix) {
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
    const ConstBlasMatrix<Complex<float>>& triangular_matrix,
    BlasMatrix<Complex<float>>* matrix) {
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
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void LeftLowerUnitTriangularSolves(
    const ConstBlasMatrix<Complex<double>>& triangular_matrix,
    BlasMatrix<Complex<double>>* matrix) {
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
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void LeftLowerAdjointTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix,
    BlasMatrix<Field>* matrix) {
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
      eta /= Conjugate(triang_column[k]);
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void LeftLowerAdjointTriangularSolves(
    const ConstBlasMatrix<float>& triangular_matrix,
    BlasMatrix<float>* matrix) {
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
    const ConstBlasMatrix<double>& triangular_matrix,
    BlasMatrix<double>* matrix) {
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
    const ConstBlasMatrix<Complex<float>>& triangular_matrix,
    BlasMatrix<Complex<float>>* matrix) {
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
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void LeftLowerAdjointTriangularSolves(
    const ConstBlasMatrix<Complex<double>>& triangular_matrix,
    BlasMatrix<Complex<double>>* matrix) {
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
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void LeftLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix,
    BlasMatrix<Field>* matrix) {
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
    const ConstBlasMatrix<float>& triangular_matrix,
    BlasMatrix<float>* matrix) {
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
    const ConstBlasMatrix<double>& triangular_matrix,
    BlasMatrix<double>* matrix) {
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
    const ConstBlasMatrix<Complex<float>>& triangular_matrix,
    BlasMatrix<Complex<float>>* matrix) {
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
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void LeftLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrix<Complex<double>>& triangular_matrix,
    BlasMatrix<Complex<double>>* matrix) {
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
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
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
    const ConstBlasMatrix<double>& triangular_matrix,
    BlasMatrix<double>* matrix) {
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
    const ConstBlasMatrix<Complex<float>>& triangular_matrix,
    BlasMatrix<Complex<float>>* matrix) {
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
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void RightLowerAdjointTriangularSolves(
    const ConstBlasMatrix<Complex<double>>& triangular_matrix,
    BlasMatrix<Complex<double>>* matrix) {
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
    const ConstBlasMatrix<double>& triangular_matrix,
    BlasMatrix<double>* matrix) {
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
    const ConstBlasMatrix<Complex<float>>& triangular_matrix,
    BlasMatrix<Complex<float>>* matrix) {
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
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}

template <>
inline void RightLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrix<Complex<double>>& triangular_matrix,
    BlasMatrix<Complex<double>>* matrix) {
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
  (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
   triangular_matrix.data, &triang_leading_dim_blas, matrix->data,
   &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void RightDiagonalTimesLowerAdjointUnitTriangularSolves(
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
      matrix->Entry(i, j) /= triangular_matrix(j, j);
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void RightDiagonalTimesLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrix<float>& triangular_matrix,
    BlasMatrix<float>* matrix) {
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
    const ConstBlasMatrix<double>& triangular_matrix,
    BlasMatrix<double>* matrix) {
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
    const ConstBlasMatrix<Complex<float>>& triangular_matrix,
    BlasMatrix<Complex<float>>* matrix) {
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
    const ConstBlasMatrix<Complex<double>>& triangular_matrix,
    BlasMatrix<Complex<double>>* matrix) {
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

}  // namespace catamari

#endif  // ifndef CATAMARI_DENSE_BASIC_LINEAR_ALGEBRA_IMPL_H_
