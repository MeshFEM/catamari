/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_DIAGONAL_FACTOR_IMPL_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_DIAGONAL_FACTOR_IMPL_H_

#include "catamari/sparse_ldl/supernodal/diagonal_factor.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
DiagonalFactor<Field>::DiagonalFactor(const Buffer<Int>& supernode_sizes, BlasMatrixView<Field> storage) {
  const Int num_supernodes = supernode_sizes.Size();

  blocks.Resize(num_supernodes);
  Int num_diagonal_entries = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int supernode_size = supernode_sizes[supernode];

    blocks[supernode].height = supernode_size;
    blocks[supernode].width = supernode_size;
    blocks[supernode].leading_dim = supernode_size;
    blocks[supernode].data = storage.Data() + num_diagonal_entries;

    num_diagonal_entries += supernode_size * supernode_size;
  }
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_DIAGONAL_FACTOR_IMPL_H_
