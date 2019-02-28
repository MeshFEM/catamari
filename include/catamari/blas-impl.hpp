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

namespace catamari {

inline int GetMaxBlasThreads() {
#ifdef CATAMARI_HAVE_MKL
  return mkl_get_max_threads();
#elif defined(CATAMARI_HAVE_OPENBLAS)
  return openblas_get_max_threads();
#else
  return 1;
#endif  // ifdef CATAMARI_HAVE_MKL
}

inline void SetNumBlasThreads(int num_threads) {
#ifdef CATAMARI_HAVE_MKL
  mkl_set_num_threads(num_threads);
#elif defined(CATAMARI_HAVE_OPENBLAS)
  openblas_set_num_threads(num_threads);
#endif  // ifdef CATAMARI_HAVE_MKL
}

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

#endif  // ifndef CATAMARI_BLAS_IMPL_H_
