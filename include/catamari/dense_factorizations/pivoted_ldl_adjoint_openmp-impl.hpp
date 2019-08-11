/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_FACTORIZATIONS_PIVOTED_LDL_ADJOINT_OPENMP_IMPL_H_
#define CATAMARI_DENSE_FACTORIZATIONS_PIVOTED_LDL_ADJOINT_OPENMP_IMPL_H_

#include <cmath>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/lapack.hpp"

#include "catamari/dense_factorizations.hpp"

namespace catamari {

template <class Field>
Int OpenMPPivotedLDLAdjointFactorization(Int tile_size, Int block_size,
                                         BlasMatrixView<Field>* matrix,
                                         BlasMatrixView<Int>* permutation) {
  return PivotedLDLAdjointFactorization(block_size, matrix, permutation);
}

}  // namespace catamari

#endif  // ifndef
        // CATAMARI_DENSE_FACTORIZATIONS_PIVOTED_LDL_ADJOINT_OPENMP_IMPL_H_
