/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LAPACK_H_
#define CATAMARI_LAPACK_H_

#include "catamari/integers.hpp"

namespace catamari {

template <class Field>
Int LowerCholeskyFactorization(BlasMatrix<Field>* matrix);

template <class Field>
Int LowerLDLAdjointFactorization(BlasMatrix<Field>* matrix);

template <class Field>
std::vector<Int> LowerFactorAndSampleDPP(
    BlasMatrix<Field>* matrix, std::mt19937* generator,
    std::uniform_real_distribution<ComplexBase<Field>>* uniform_dist);

}  // namespace catamari

#include "catamari/lapack-impl.hpp"

#endif  // ifndef CATAMARI_LAPACK_H_
