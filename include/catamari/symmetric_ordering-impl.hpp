/*
 * Copyright (c) 2018-2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SYMMETRIC_ORDERING_IMPL_H_
#define CATAMARI_SYMMETRIC_ORDERING_IMPL_H_

#include "catamari/symmetric_ordering.hpp"

namespace catamari {

inline void AssemblyForest::FillFromParents() {
  const Int num_indices = parents.Size();

  // Compute the number of children (initially stored in 'child_offsets') of
  // each vertex. Along the way, count the number of trees in the forest.
  Int num_roots = 0;
  child_offsets.Resize(num_indices + 1, 0);
  for (Int index = 0; index < num_indices; ++index) {
    const Int parent = parents[index];
    if (parent >= 0) {
      ++child_offsets[parent];
    } else {
      ++num_roots;
    }
  }

  // Compute the child offsets using an in-place scan.
  Int num_total_children = 0;
  for (Int index = 0; index < num_indices; ++index) {
    const Int num_children = child_offsets[index];
    child_offsets[index] = num_total_children;
    num_total_children += num_children;
  }
  child_offsets[num_indices] = num_total_children;

  // Pack the children into the 'children' buffer.
  children.Resize(num_total_children);
  roots.Resize(num_roots);
  Int counter = 0;
  Buffer<Int> offsets_copy = child_offsets;
  for (Int index = 0; index < num_indices; ++index) {
    const Int parent = parents[index];
    if (parent >= 0) {
      children[offsets_copy[parent]++] = index;
    } else {
      roots[counter++] = index;
    }
  }
}

}  // namespace catamari

#endif  // ifndef CATAMARI_SYMMETRIC_ORDERING_IMPL_H_
