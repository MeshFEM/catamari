/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_BLAS_MKL_H_
#define CATAMARI_BLAS_MKL_H_

#include <complex>

#define CATAMARI_HAVE_BLAS_PROTOS
#define CATAMARI_HAVE_LAPACK_PROTOS

#define BLAS_SYMBOL(name) name

// TODO(Jack Poulson): Decide when to avoid enabling this function. It seems to
// be slightly slower than doing twice as much work by running Gemm then
// setting the strictly upper triangle of the result to zero.
#define CATAMARI_USE_GEMMT

// TODO(Jack Poulson): Attempt to support 64-bit BLAS when Int = long long int.
typedef int BlasInt;
typedef std::complex<float> BlasComplexFloat;
typedef std::complex<double> BlasComplexDouble;

#define MKL_INT BlasInt
#define MKL_Complex8 BlasComplexFloat
#define MKL_Complex16 BlasComplexDouble
#include <mkl.h>

#endif  // ifndef CATAMARI_BLAS_MKL_H_
