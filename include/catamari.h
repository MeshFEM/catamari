/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_C_H_
#define CATAMARI_C_H_
#include <random>

#include "catamari/integers.h"
#include "catamari/macros.h"

#include "catamari/blas_matrix_view.h"
#include "catamari/buffer.h"
#include "catamari/dense_dpp.h"

std::mt19937* CatamariGenerator();

#ifdef __cplusplus
extern "C" {
#endif  // ifdef __cplusplus

CATAMARI_EXPORT void CatamariHas64BitInts(bool* has_64_bit_ints);

CATAMARI_EXPORT void CatamariHasOpenMP(bool* has_openmp);

CATAMARI_EXPORT void CatamariInitGenerator(unsigned int random_seed);

#ifdef __cplusplus
}  // extern "C"
#endif  // ifdef __cplusplus

#endif  // ifndef CATAMARI_C_H_
