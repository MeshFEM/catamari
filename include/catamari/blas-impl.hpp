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

#ifdef CATAMARI_HAVE_OPENBLAS
extern "C" {

// TODO(Jack Poulson): Attempt to support 64-bit BLAS when Int = long long int.

void dgemm_(
    const char* trans_left, const char* trans_right,
    const int* output_height, const int* output_width,
    const int* contraction_size, const double* alpha, const double* left_matrix,
    const int* left_leading_dim, const double* right_matrix,
    const int* right_leading_dim, const double* beta, double* output_matrix,
    const int* output_leading_dim);

void sgemm_(
    const char* trans_left, const char* trans_right,
    const int* output_height, const int* output_width,
    const int* contraction_size, const float* alpha, const float* left_matrix,
    const int* left_leading_dim, const float* right_matrix,
    const int* right_leading_dim, const float* beta, float* output_matrix,
    const int* output_leading_dim);

void dtrsm_(
    const char* side, const char* uplo, const char* trans_triang,
    const char* diag, const int* height, const int* width, const double* alpha,
    const double* triang_matrix, const int* triang_leading_dim, double* matrix,
    const int* leading_dim);

void strsm_(
    const char* side, const char* uplo, const char* trans_triang,
    const char* diag, const int* height, const int* width, const float* alpha,
    const float* triang_matrix, const int* triang_leading_dim, float* matrix,
    const int* leading_dim);

}
#endif

namespace catamari {

template <class Field>
void MatrixMultiplyTransposeNormal(
    Int output_height, Int output_width, Int contraction_size,
    const Field& alpha, const Field* left_matrix, Int left_leading_dim,
    const Field* right_matrix, Int right_leading_dim, const Field& beta,
    Field* output_matrix, Int output_leading_dim) {
  for (Int j = 0; j < output_width; ++j) {
    for (Int i = 0; i < output_height; ++i) {
      Field& output_entry = output_matrix[i + j * output_leading_dim];
      output_entry *= beta;
      for (Int k = 0; k < contraction_size; ++k) {
        output_entry +=
            alpha * left_matrix[k + i * left_leading_dim] *
            right_matrix[k + j * right_leading_dim];
      }
    }
  }
}

#ifdef CATAMARI_HAVE_OPENBLAS
template <>
void MatrixMultiplyTransposeNormal(
    Int output_height, Int output_width, Int contraction_size,
    const float& alpha, const float* left_matrix, Int left_leading_dim,
    const float* right_matrix, Int right_leading_dim, const float& beta,
    float* output_matrix, Int output_leading_dim) {
  const char trans_left = 'T';
  const char trans_right = 'N';
  const int output_height_32 = output_height;
  const int output_width_32 = output_width;
  const int contraction_size_32 = contraction_size;
  const int left_leading_dim_32 = left_leading_dim;
  const int right_leading_dim_32 = right_leading_dim;
  const int output_leading_dim_32 = output_leading_dim;
  sgemm_(
      &trans_left, &trans_right, &output_height_32, &output_width_32,
      &contraction_size_32, &alpha, left_matrix, &left_leading_dim_32,
      right_matrix, &right_leading_dim_32, &beta, output_matrix,
      &output_leading_dim_32);
}

template <>
void MatrixMultiplyTransposeNormal(
    Int output_height, Int output_width, Int contraction_size,
    const double& alpha, const double* left_matrix, Int left_leading_dim,
    const double* right_matrix, Int right_leading_dim, const double& beta,
    double* output_matrix, Int output_leading_dim) {
  const char trans_left = 'T';
  const char trans_right = 'N';
  const int output_height_32 = output_height;
  const int output_width_32 = output_width;
  const int contraction_size_32 = contraction_size;
  const int left_leading_dim_32 = left_leading_dim;
  const int right_leading_dim_32 = right_leading_dim;
  const int output_leading_dim_32 = output_leading_dim;
  dgemm_(
      &trans_left, &trans_right, &output_height_32, &output_width_32,
      &contraction_size_32, &alpha, left_matrix, &left_leading_dim_32,
      right_matrix, &right_leading_dim_32, &beta, output_matrix,
      &output_leading_dim_32);
}
#endif

template <class Field>
void MatrixMultiplyTransposeNormalLower(
    Int output_height, Int contraction_size,
    const Field& alpha, const Field* left_matrix, Int left_leading_dim,
    const Field* right_matrix, Int right_leading_dim, const Field& beta,
    Field* output_matrix, Int output_leading_dim) {
  for (Int j = 0; j < output_height; ++j) {
    for (Int i = j; i < output_height; ++i) {
      Field& output_entry = output_matrix[i + j * output_leading_dim];
      output_entry *= beta;
      for (Int k = 0; k < contraction_size; ++k) {
        output_entry +=
            alpha * left_matrix[k + i * left_leading_dim] *
            right_matrix[k + j * right_leading_dim];
      }
    }
  }
}

#ifdef CATAMARI_HAVE_OPENBLAS
template <>
void MatrixMultiplyTransposeNormalLower(
    Int output_height, Int contraction_size,
    const float& alpha, const float* left_matrix, Int left_leading_dim,
    const float* right_matrix, Int right_leading_dim, const float& beta,
    float* output_matrix, Int output_leading_dim) {
  // TODO(Jack Poulson): Save a factor of two if we have MKL via GEMMT.
  MatrixMultiplyTransposeNormal(
      output_height, output_height, contraction_size, alpha, left_matrix,
      left_leading_dim, right_matrix, right_leading_dim, beta, output_matrix,
      output_leading_dim);

  // Explicitly zero out the strictly-upper triangle of C.
  for (Int j = 1; j < output_height; ++j) {
    for (Int i = 0; i < j; ++i) {
      output_matrix[i + j * output_leading_dim] = 0;
    }
  }
}

template <>
void MatrixMultiplyTransposeNormalLower(
    Int output_height, Int contraction_size,
    const double& alpha, const double* left_matrix, Int left_leading_dim,
    const double* right_matrix, Int right_leading_dim, const double& beta,
    double* output_matrix, Int output_leading_dim) {
  // TODO(Jack Poulson): Save a factor of two if we have MKL via GEMMT.
  MatrixMultiplyTransposeNormal(
      output_height, output_height, contraction_size, alpha, left_matrix,
      left_leading_dim, right_matrix, right_leading_dim, beta, output_matrix,
      output_leading_dim);

  // Explicitly zero out the strictly-upper triangle of C.
  for (Int j = 1; j < output_height; ++j) {
    for (Int i = 0; i < j; ++i) {
      output_matrix[i + j * output_leading_dim] = 0;
    }
  }
}
#endif

template <class Field>
void DiagonalTimesConjugateUnitLowerTriangularSolves(
  Int height, Int width, const Field* triangular_matrix, Int triang_leading_dim,
  Field* matrix, Int leading_dim) {
  for (Int j = 0; j < width; ++j) {
    Field* input_column = &matrix[j * leading_dim];

    // Interleave with a subsequent solve against D.
    for (Int i = 0; i < height; ++i) {
      const Field* l_column = &triangular_matrix[i * triang_leading_dim];
      for (Int k = i + 1; k < height; ++k) {
        input_column[k] -= Conjugate(l_column[k]) * input_column[i];
      }
      input_column[i] /= l_column[i];
    }
  }
}

#ifdef CATAMARI_HAVE_OPENBLAS
template <>
void DiagonalTimesConjugateUnitLowerTriangularSolves(
  Int height, Int width, const float* triangular_matrix, Int triang_leading_dim,
  float* matrix, Int leading_dim) {

  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'N';
  const char diag = 'U';
  const int height_32 = height;
  const int width_32 = width;
  const float alpha = 1.f;
  const int triang_leading_dim_32 = triang_leading_dim;
  const int leading_dim_32 = leading_dim;

  strsm_(
      &side, &uplo, &trans_triang, &diag, &height_32, &width_32, &alpha,
      triangular_matrix, &triang_leading_dim_32, matrix, &leading_dim_32);

  // Solve against the diagonal.
  for (Int j = 0; j < width; ++j) {
    for (Int i = 0; i < height; ++i) {
      matrix[i + j * leading_dim] /=
          triangular_matrix[i + i * triang_leading_dim];
    }
  }
}

template <>
void DiagonalTimesConjugateUnitLowerTriangularSolves(
  Int height, Int width, const double* triangular_matrix, Int triang_leading_dim,
  double* matrix, Int leading_dim) {

  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'N';
  const char diag = 'U';
  const int height_32 = height;
  const int width_32 = width;
  const double alpha = 1.;
  const int triang_leading_dim_32 = triang_leading_dim;
  const int leading_dim_32 = leading_dim;

  dtrsm_(
      &side, &uplo, &trans_triang, &diag, &height_32, &width_32, &alpha,
      triangular_matrix, &triang_leading_dim_32, matrix, &leading_dim_32);

  // Solve against the diagonal.
  for (Int j = 0; j < width; ++j) {
    for (Int i = 0; i < height; ++i) {
      matrix[i + j * leading_dim] /=
          triangular_matrix[i + i * triang_leading_dim];
    }
  }
}
#endif

}  // namespace catamari

#endif  // ifndef CATAMARI_BLAS_IMPL_H_
