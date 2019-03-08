/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SUPERNODAL_LDL_DIAGONAL_FACTOR_IMPL_H_
#define CATAMARI_LDL_SUPERNODAL_LDL_DIAGONAL_FACTOR_IMPL_H_

#include "catamari/ldl/supernodal_ldl/diagonal_factor.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
DiagonalFactor<Field>::DiagonalFactor(const Buffer<Int>& supernode_sizes) {
  const Int num_supernodes = supernode_sizes.Size();

  Buffer<Int> diag_value_offsets(num_supernodes + 1);
  Int num_diagonal_entries = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int supernode_size = supernode_sizes[supernode];
    diag_value_offsets[supernode] = num_diagonal_entries;
    num_diagonal_entries += supernode_size * supernode_size;
  }
  diag_value_offsets[num_supernodes] = num_diagonal_entries;
  values_.Resize(num_diagonal_entries);

  blocks.Resize(num_supernodes);
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int supernode_size = supernode_sizes[supernode];
    blocks[supernode].height = supernode_size;
    blocks[supernode].width = supernode_size;
    blocks[supernode].leading_dim = supernode_size;
    blocks[supernode].data = &values_[diag_value_offsets[supernode]];
  }
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_LDL_SUPERNODAL_LDL_DIAGONAL_FACTOR_IMPL_H_
