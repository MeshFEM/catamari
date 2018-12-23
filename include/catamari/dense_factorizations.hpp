/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_FACTORIZATIONS_H_
#define CATAMARI_DENSE_FACTORIZATIONS_H_

#include <random>

#include "catamari/integers.hpp"

namespace catamari {

template <class Field>
Int LowerCholeskyFactorization(BlasMatrix<Field>* matrix);

template <class Field>
Int LowerLDLAdjointFactorization(BlasMatrix<Field>* matrix);

template <class Field>
Int LowerLDLTransposeFactorization(BlasMatrix<Field>* matrix);

template <class Field>
std::vector<Int> LowerFactorAndSampleDPP(
    bool maximum_likelihood, BlasMatrix<Field>* matrix, std::mt19937* generator,
    std::uniform_real_distribution<ComplexBase<Field>>* uniform_dist);

}  // namespace catamari

#include "catamari/dense_factorizations-impl.hpp"

#endif  // ifndef CATAMARI_DENSE_FACTORIZATIONS_H_
