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

#include "catamari/blas/mkl.hpp"

#elif defined(CATAMARI_HAVE_OPENBLAS)

#include "catmari/blas/openblas.hpp"

#endif  // ifdef CATAMARI_HAVE_MKL

#if defined(CATAMARI_HAVE_BLAS) && !defined(CATAMARI_HAVE_BLAS_PROTOS)

#include "catamari/blas/protos.hpp"

#endif  // if defined(CATAMARI_HAVE_BLAS) && !defined(CATAMARI_HAVE_BLAS_PROTOS)

namespace catamari {

// Returns the maximum number of BLAS threads.
int GetMaxBlasThreads();

// When possible, sets the global number of threads to be used by BLAS.
void SetNumBlasThreads(int num_threads);

// When possible, sets the thread-local number of threads to be used by BLAS.
//
// OpenBLAS does not support any equivalent of mkl_set_num_threads_local, so
// the best-practice is to instead disable threading in such cases.
int SetNumLocalBlasThreads(int num_threads);

}  // namespace catamari

#include "catamari/blas-impl.hpp"

#endif  // ifndef CATAMARI_BLAS_H_
