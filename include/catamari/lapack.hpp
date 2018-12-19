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
extern "C" {

#define LAPACK_SYMBOL(name) name##_

void LAPACK_SYMBOL(spotrf)(const char* uplo, const BlasInt* n, float* matrix,
                           const BlasInt* leading_dim, BlasInt* info);

void LAPACK_SYMBOL(dpotrf)(const char* uplo, const BlasInt* n, double* matrix,
                           const BlasInt* leading_dim, BlasInt* info);
}
#endif  // ifdef CATAMARI_HAVE_LAPACK

#endif  // ifndef CATAMARI_LAPACK_H_
