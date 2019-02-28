/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_BLAS_OPENBLAS_H_
#define CATAMARI_BLAS_OPENBLAS_H_

#include <complex>

#define BLAS_SYMBOL(name) name##_

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

#endif  // ifndef CATAMARI_BLAS_OPENBLAS_H_
