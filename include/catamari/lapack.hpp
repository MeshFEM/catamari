/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LAPACK_H_
#define CATAMARI_LAPACK_H_

#include "catamari/blas.hpp"

#ifdef CATAMARI_HAVE_LAPACK

#define LAPACK_SYMBOL(name) name##_

#ifndef CATAMARI_HAVE_LAPACK_PROTOS
#define CATAMARI_HAVE_LAPACK_PROTOS

extern "C" {

void LAPACK_SYMBOL(spotrf)(const char* uplo, const BlasInt* n, float* matrix,
                           const BlasInt* leading_dim, BlasInt* info);

void LAPACK_SYMBOL(dpotrf)(const char* uplo, const BlasInt* n, double* matrix,
                           const BlasInt* leading_dim, BlasInt* info);

void LAPACK_SYMBOL(cpotrf)(const char* uplo, const BlasInt* n,
                           BlasComplexFloat* matrix, const BlasInt* leading_dim,
                           BlasInt* info);

void LAPACK_SYMBOL(zpotrf)(const char* uplo, const BlasInt* n,
                           BlasComplexDouble* matrix,
                           const BlasInt* leading_dim, BlasInt* info);
}
#endif  // ifndef CATAMARI_HAVE_LAPACK_PROTOS
#endif  // ifdef CATAMARI_HAVE_LAPACK

#endif  // ifndef CATAMARI_LAPACK_H_
