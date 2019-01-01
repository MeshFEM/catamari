/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_BLAS_H_
#define CATAMARI_BLAS_H_

#include <complex>

#ifdef CATAMARI_HAVE_MKL

// TODO(Jack Poulson): Attempt to support 64-bit BLAS when Int = long long int.
typedef int BlasInt;
typedef std::complex<float> BlasComplexFloat;
typedef std::complex<double> BlasComplexDouble;

#define MKL_INT BlasInt
#define MKL_Complex8 BlasComplexFloat
#define MKL_Complex16 BlasComplexDouble
#include "mkl.h"

#define CATAMARI_HAVE_BLAS_PROTOS
#define CATAMARI_HAVE_LAPACK_PROTOS

#define BLAS_SYMBOL(name) name

// TODO(Jack Poulson): Decide when to avoid enabling this function. It seems to
// be slightly slower than doing twice as much work by running Gemm then
// setting the strictly upper triangle of the result to zero.
#define CATAMARI_USE_GEMMT

#elif defined(CATAMARI_HAVE_OPENBLAS)

// TODO(Jack Poulson): Attempt to support 64-bit BLAS when Int = long long int.
typedef int BlasInt;
typedef std::complex<float> BlasComplexFloat;
typedef std::complex<float> BlasComplexDouble;

extern "C" {

// Sets the maximum number of OpenBLAS threads.
void openblas_set_num_threads(int num_threads);

// Gets the current maximum number of OpenBLAS threads.
int openblas_get_num_threads();

}  // extern "C"

#define BLAS_SYMBOL(name) name##_

#endif  // ifdef CATAMARI_HAVE_MKL

#if defined(CATAMARI_HAVE_BLAS) && !defined(CATAMARI_HAVE_BLAS_PROTOS)
#define CATAMARI_HAVE_BLAS_PROTOS

extern "C" {

void BLAS_SYMBOL(ssyr)(const char* uplo, const BlasInt* height,
                       const float* alpha, const float* vector,
                       const BlasInt* stride, float* matrix,
                       const BlasInt* leading_dim);

void BLAS_SYMBOL(dsyr)(const char* uplo, const BlasInt* height,
                       const double* alpha, const double* vector,
                       const BlasInt* stride, double* matrix,
                       const BlasInt* leading_dim);

void BLAS_SYMBOL(cher)(const char* uplo, const BlasInt* height,
                       const float* alpha, const BlasComplexFloat* vector,
                       const BlasInt* stride, BlasComplexFloat* matrix,
                       const BlasInt* leading_dim);

void BLAS_SYMBOL(zher)(const char* uplo, const BlasInt* height,
                       const double* alpha, const BlasComplexDouble* vector,
                       const BlasInt* stride, BlasComplexDouble* matrix,
                       const BlasInt* leading_dim);

void BLAS_SYMBOL(sgemv)(const char* trans, const BlasInt* height,
                        const BlasInt* width, const float* alpha,
                        const float* matrix, const BlasInt* leading_dim,
                        const float* input_vector, const BlasInt* input_stride,
                        const float* beta, float* result,
                        const BlasInt* result_stride);

void BLAS_SYMBOL(dgemv)(const char* trans, const BlasInt* height,
                        const BlasInt* width, const double* alpha,
                        const double* matrix, const BlasInt* leading_dim,
                        const double* input_vector, const BlasInt* input_stride,
                        const double* beta, double* result,
                        const BlasInt* result_stride);

void BLAS_SYMBOL(cgemv)(const char* trans, const BlasInt* height,
                        const BlasInt* width, const BlasComplexFloat* alpha,
                        const BlasComplexFloat* matrix,
                        const BlasInt* leading_dim,
                        const BlasComplexFloat* input_vector,
                        const BlasInt* input_stride,
                        const BlasComplexFloat* beta, BlasComplexFloat* result,
                        const BlasInt* result_stride);

void BLAS_SYMBOL(zgemv)(
    const char* trans, const BlasInt* height, const BlasInt* width,
    const BlasComplexDouble* alpha, const BlasComplexDouble* matrix,
    const BlasInt* leading_dim, const BlasComplexDouble* input_vector,
    const BlasInt* input_stride, const BlasComplexDouble* beta,
    BlasComplexDouble* result, const BlasInt* result_stride);

void BLAS_SYMBOL(strsv)(const char* uplo, const char* trans, const char* diag,
                        const BlasInt* height, const float* triangular_matrix,
                        const BlasInt* triang_leading_dim, float* vector,
                        const BlasInt* stride);

void BLAS_SYMBOL(dtrsv)(const char* uplo, const char* trans, const char* diag,
                        const BlasInt* height, const double* triangular_matrix,
                        const BlasInt* triang_leading_dim, double* vector,
                        const BlasInt* stride);

void BLAS_SYMBOL(ctrsv)(const char* uplo, const char* trans, const char* diag,
                        const BlasInt* height,
                        const BlasComplexFloat* triangular_matrix,
                        const BlasInt* triang_leading_dim,
                        BlasComplexFloat* vector, const BlasInt* stride);

void BLAS_SYMBOL(ztrsv)(const char* uplo, const char* trans, const char* diag,
                        const BlasInt* height,
                        const BlasComplexDouble* triangular_matrix,
                        const BlasInt* triang_leading_dim,
                        BlasComplexDouble* vector, const BlasInt* stride);

void BLAS_SYMBOL(sgemm)(
    const char* trans_left, const char* trans_right,
    const BlasInt* output_height, const BlasInt* output_width,
    const BlasInt* contraction_size, const float* alpha,
    const float* left_matrix, const BlasInt* left_leading_dim,
    const float* right_matrix, const BlasInt* right_leading_dim,
    const float* beta, float* output_matrix, const BlasInt* output_leading_dim);

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

void BLAS_SYMBOL(cgemm)(
    const char* trans_left, const char* trans_right,
    const BlasInt* output_height, const BlasInt* output_width,
    const BlasInt* contraction_size, const BlasComplexFloat* alpha,
    const BlasComplexFloat* left_matrix, const BlasInt* left_leading_dim,
    const BlasComplexFloat* right_matrix, const BlasInt* right_leading_dim,
    const BlasComplexFloat* beta, BlasComplexFloat* output_matrix,
    const BlasInt* output_leading_dim);

void BLAS_SYMBOL(zgemm)(
    const char* trans_left, const char* trans_right,
    const BlasInt* output_height, const BlasInt* output_width,
    const BlasInt* contraction_size, const BlasComplexDouble* alpha,
    const BlasComplexDouble* left_matrix, const BlasInt* left_leading_dim,
    const BlasComplexDouble* right_matrix, const BlasInt* right_leading_dim,
    const BlasComplexDouble* beta, BlasComplexDouble* output_matrix,
    const BlasInt* output_leading_dim);

void BLAS_SYMBOL(ssyrk)(const char* uplo, const char* trans,
                        const BlasInt* height, const BlasInt* rank,
                        const float* alpha, const float* factor,
                        const BlasInt* factor_leading_dim, const float* beta,
                        float* matrix, const BlasInt* leading_dim);

void BLAS_SYMBOL(dsyrk)(const char* uplo, const char* trans,
                        const BlasInt* height, const BlasInt* rank,
                        const double* alpha, const double* factor,
                        const BlasInt* factor_leading_dim, const double* beta,
                        double* matrix, const BlasInt* leading_dim);

void BLAS_SYMBOL(cherk)(const char* uplo, const char* trans,
                        const BlasInt* height, const BlasInt* rank,
                        const float* alpha, const BlasComplexFloat* factor,
                        const BlasInt* factor_leading_dim, const float* beta,
                        BlasComplexFloat* matrix, const BlasInt* leading_dim);

void BLAS_SYMBOL(zherk)(const char* uplo, const char* trans,
                        const BlasInt* height, const BlasInt* rank,
                        const double* alpha, const BlasComplexDouble* factor,
                        const BlasInt* factor_leading_dim, const double* beta,
                        BlasComplexDouble* matrix, const BlasInt* leading_dim);

void BLAS_SYMBOL(strsm)(const char* side, const char* uplo,
                        const char* trans_triang, const char* diag,
                        const BlasInt* height, const BlasInt* width,
                        const float* alpha, const float* triang_matrix,
                        const BlasInt* triang_leading_dim, float* matrix,
                        const BlasInt* leading_dim);

void BLAS_SYMBOL(dtrsm)(const char* side, const char* uplo,
                        const char* trans_triang, const char* diag,
                        const BlasInt* height, const BlasInt* width,
                        const double* alpha, const double* triang_matrix,
                        const BlasInt* triang_leading_dim, double* matrix,
                        const BlasInt* leading_dim);

void BLAS_SYMBOL(ctrsm)(const char* side, const char* uplo,
                        const char* trans_triang, const char* diag,
                        const BlasInt* height, const BlasInt* width,
                        const BlasComplexFloat* alpha,
                        const BlasComplexFloat* triang_matrix,
                        const BlasInt* triang_leading_dim,
                        BlasComplexFloat* matrix, const BlasInt* leading_dim);

void BLAS_SYMBOL(ztrsm)(const char* side, const char* uplo,
                        const char* trans_triang, const char* diag,
                        const BlasInt* height, const BlasInt* width,
                        const BlasComplexDouble* alpha,
                        const BlasComplexDouble* triang_matrix,
                        const BlasInt* triang_leading_dim,
                        BlasComplexDouble* matrix, const BlasInt* leading_dim);
}
#endif  // if defined(CATAMARI_HAVE_BLAS) && !defined(CATAMARI_HAVE_BLAS_PROTOS)

namespace catamari {

// Returns the maximum number of BLAS threads.
inline int GetMaxBlasThreads() {
#ifdef CATAMARI_HAVE_MKL
  return mkl_get_max_threads();
#elif defined(CATAMARI_HAVE_OPENBLAS)
  return openblas_get_max_threads();
#else
  return 1;
#endif  // ifdef CATAMARI_HAVE_MKL
}

// When possible, sets the global number of threads to be used by BLAS.
inline void SetNumBlasThreads(int num_threads) {
#ifdef CATAMARI_HAVE_MKL
  mkl_set_num_threads(num_threads);
#elif defined(CATAMARI_HAVE_OPENBLAS)
  openblas_set_num_threads(num_threads);
#endif  // ifdef CATAMARI_HAVE_MKL
}

// When possible, sets the thread-local number of threads to be used by BLAS.
//
// OpenBLAS does not support any equivalent of mkl_set_num_threads_local, so
// the best-practice is to instead disable threading in such cases.
inline int SetNumLocalBlasThreads(int num_threads) {
#ifdef CATAMARI_HAVE_MKL
  return mkl_set_num_threads_local(num_threads);
#elif defined(CATAMARI_HAVE_OPENBLAS)
  const int old_num_threads = openblas_get_num_threads();
  openblas_set_num_threads(1);
  return old_num_threads;
#else
  return 1;
#endif  // ifdef CATAMARI_HAVE_MKL
}

}  // namespace catamari

#endif  // ifndef CATAMARI_BLAS_H_
