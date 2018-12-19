/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_BLAS_H_
#define CATAMARI_BLAS_H_

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

#endif  // ifndef CATAMARI_BLAS_H_
