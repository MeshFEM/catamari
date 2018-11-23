/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SUPERNODAL_LDL_H_
#define CATAMARI_SUPERNODAL_LDL_H_

#include "catamari/ldl.hpp"

namespace catamari {

// The representation of the portion of the unit-lower triangular factor
// that is below the supernodal diagonal blocks.
template <class Field>
struct SupernodalLowerFactor {
  std::vector<Int> supernode_index_offsets;

  std::vector<Int> supernode_value_offsets;

  std::vector<Int> indices;

  std::vector<Int> values;
};

// Stores the (dense) diagonal blocks for the supernodes.
template <class Field>
struct SupernodalDiagonalFactor {
  std::vector<Int> supernode_value_offsets;

  std::vector<Int> values;
};

template <class Field>
struct SupernodalLDLFactorization {
  // An array of length 'num_supernodes'; the i'th member is the size of the
  // i'th supernode.
  std::vector<Int> supernode_sizes;

  // An array of length 'num_supernodes + 1'; the i'th member, for
  // 0 <= i < num_supernodes, is the principal member of the i'th supernode.
  // The last member is equal to 'num_rows'.
  std::vector<Int> supernode_offsets;

  // An array of length 'num_rows'; the i'th member is the index of the
  // supernode containing column 'i'.
  std::vector<Int> supernode_container;

  // The subdiagonal-block portion of the lower-triangular factor.
  SupernodalLowerFactor<Field> lower_factor;

  // The block-diagonal factor.
  SupernodalDiagonalFactor<Field> diagonal_factor;
};

}  // namespace catamari

#include "catamari/supernodal_ldl-impl.hpp"

#endif  // ifndef CATAMARI_SUPERNODAL_LDL_H_
