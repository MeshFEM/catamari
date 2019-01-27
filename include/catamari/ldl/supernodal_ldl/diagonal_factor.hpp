/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SUPERNODAL_LDL_DIAGONAL_FACTOR_H_
#define CATAMARI_LDL_SUPERNODAL_LDL_DIAGONAL_FACTOR_H_

#include <vector>

#include "catamari/blas_matrix.hpp"
#include "catamari/buffer.hpp"

namespace catamari {
namespace supernodal_ldl {

// Stores the (dense) diagonal blocks for the supernodes.
template <class Field>
class DiagonalFactor {
 public:
  // Representations of the diagonal blocks of the factorization.
  Buffer<BlasMatrix<Field>> blocks;

  DiagonalFactor(const Buffer<Int>& supernode_sizes);

 private:
  // The concatenation of the numerical values of the supernodal diagonal
  // blocks (stored in a column-major manner in each block).
  Buffer<Field> values_;
};

}  // namespace supernodal_ldl
}  // namespace catamari

#include "catamari/ldl/supernodal_ldl/diagonal_factor-impl.hpp"

#endif  // ifndef CATAMARI_LDL_SUPERNODAL_LDL_DIAGONAL_FACTOR_H_
