/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_DIAGONAL_FACTOR_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_DIAGONAL_FACTOR_H_

#include <vector>

#include "catamari/blas_matrix_view.hpp"
#include "catamari/buffer.hpp"
#include "catamari_config.hh"

namespace catamari {
namespace supernodal_ldl {

// Stores the (dense) diagonal blocks for the supernodes.
template <class Field>
class DiagonalFactor {
 public:
  // Representations of the diagonal blocks of the factorization.
  Buffer<BlasMatrixView<Field>> blocks;

  DiagonalFactor(const Buffer<Int>& supernode_sizes, BlasMatrixView<Field> storage);
};

}  // namespace supernodal_ldl
}  // namespace catamari

#include "catamari/sparse_ldl/supernodal/diagonal_factor-impl.hpp"

#endif  // ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_DIAGONAL_FACTOR_H_
