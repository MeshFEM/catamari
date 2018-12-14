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
Int LowerCholeskyFactorization(Int height, Field* A, Int leading_dim);

template <class Field>
Int LowerLDLAdjointFactorization(Int height, Field* A, Int leading_dim);

}  // namespace catamari

#include "catamari/lapack-impl.hpp"

#endif  // ifndef CATAMARI_LAPACK_H_
