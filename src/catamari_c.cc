/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <random>

#include "catamari/blas_matrix_view.hpp"
#include "catamari/complex.hpp"

#include "catamari.h"

namespace {

std::mt19937 g_generator;

}  // anonymous namespace

void CatamariHas64BitInts(bool* has_64_bit_ints) {
  *has_64_bit_ints = sizeof(catamari::Int) == 8;
}

void CatamariHasOpenMP(bool* has_openmp) {
#ifdef CATAMARI_OPENMP
  *has_openmp = true;
#else
  *has_openmp = false;
#endif
}

void CatamariInitGenerator(unsigned int random_seed) {
  g_generator.seed(random_seed);
}

std::mt19937* CatamariGenerator() { return &g_generator; }
