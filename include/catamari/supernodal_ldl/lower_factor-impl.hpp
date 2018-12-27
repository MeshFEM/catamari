/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SUPERNODAL_LDL_LOWER_FACTOR_IMPL_H_
#define CATAMARI_SUPERNODAL_LDL_LOWER_FACTOR_IMPL_H_

#include "catamari/supernodal_ldl/lower_factor.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
LowerFactor<Field>::LowerFactor(const std::vector<Int>& supernode_sizes,
                                const std::vector<Int>& supernode_degrees) {
  const Int num_supernodes = supernode_sizes.size();

  Int degree_sum = 0;
  Int num_entries = 0;
  structure_index_offsets_.resize(num_supernodes + 1);
  std::vector<Int> lower_value_offsets(num_supernodes + 1);
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int degree = supernode_degrees[supernode];
    const Int supernode_size = supernode_sizes[supernode];

    structure_index_offsets_[supernode] = degree_sum;
    lower_value_offsets[supernode] = num_entries;

    degree_sum += degree;
    num_entries += degree * supernode_size;
  }
  structure_index_offsets_[num_supernodes] = degree_sum;
  lower_value_offsets[num_supernodes] = num_entries;

  structure_indices_.resize(degree_sum);
  values_.resize(num_entries, Field{0});

  blocks.resize(num_supernodes);
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int degree = supernode_degrees[supernode];
    const Int supernode_size = supernode_sizes[supernode];

    blocks[supernode].height = degree;
    blocks[supernode].width = supernode_size;
    blocks[supernode].leading_dim = degree;
    blocks[supernode].data = &values_[lower_value_offsets[supernode]];
  }
}

template <class Field>
Int* LowerFactor<Field>::Structure(Int supernode) {
  return &structure_indices_[structure_index_offsets_[supernode]];
}

template <class Field>
const Int* LowerFactor<Field>::Structure(Int supernode) const {
  return &structure_indices_[structure_index_offsets_[supernode]];
}

template <class Field>
Int* LowerFactor<Field>::IntersectionSizes(Int supernode) {
  return &intersect_sizes_[intersect_size_offsets_[supernode]];
}

template <class Field>
const Int* LowerFactor<Field>::IntersectionSizes(Int supernode) const {
  return &intersect_sizes_[intersect_size_offsets_[supernode]];
}

template <class Field>
void LowerFactor<Field>::FillIntersectionSizes(
    const std::vector<Int>& supernode_sizes,
    const std::vector<Int>& supernode_member_to_index,
    Int* max_descendant_entries) {
  const Int num_supernodes = blocks.size();

  // Compute the supernode offsets.
  Int num_supernode_intersects = 0;
  intersect_size_offsets_.resize(num_supernodes + 1);
  for (Int column_supernode = 0; column_supernode < num_supernodes;
       ++column_supernode) {
    intersect_size_offsets_[column_supernode] = num_supernode_intersects;
    Int last_supernode = -1;

    const Int* index_beg = Structure(column_supernode);
    const Int* index_end = Structure(column_supernode + 1);
    for (const Int* row_ptr = index_beg; row_ptr != index_end; ++row_ptr) {
      const Int row = *row_ptr;
      const Int supernode = supernode_member_to_index[row];
      if (supernode != last_supernode) {
        last_supernode = supernode;
        ++num_supernode_intersects;
      }
    }
  }
  intersect_size_offsets_[num_supernodes] = num_supernode_intersects;

  // Fill the supernode intersection sizes (and simultaneously compute the
  // number of intersecting descendant entries for each supernode).
  intersect_sizes_.resize(num_supernode_intersects);
  num_supernode_intersects = 0;
  std::vector<Int> num_descendant_entries(num_supernodes, 0);
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    Int last_supernode = -1;
    Int intersect_size = 0;
    const Int supernode_size = supernode_sizes[supernode];

    const Int* index_beg = Structure(supernode);
    const Int* index_end = Structure(supernode + 1);
    for (const Int* row_ptr = index_beg; row_ptr != index_end; ++row_ptr) {
      const Int row = *row_ptr;
      const Int row_supernode = supernode_member_to_index[row];
      if (row_supernode != last_supernode) {
        if (last_supernode != -1) {
          // Close out the supernodal intersection.
          intersect_sizes_[num_supernode_intersects++] = intersect_size;
          num_descendant_entries[row_supernode] +=
              intersect_size * supernode_size;
        }
        last_supernode = row_supernode;
        intersect_size = 0;
      }
      ++intersect_size;
    }
    if (last_supernode != -1) {
      // Close out the last intersection count for this column supernode.
      intersect_sizes_[num_supernode_intersects++] = intersect_size;
    }
  }
  CATAMARI_ASSERT(
      num_supernode_intersects == static_cast<Int>(intersect_sizes_.size()),
      "Incorrect number of supernode intersections");

  // NOTE: This only needs to be computed for multithreaded factorizations,
  // and the same applies to the num_descendant_entries array.
  *max_descendant_entries = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    *max_descendant_entries =
        std::max(*max_descendant_entries, num_descendant_entries[supernode]);
  }
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_SUPERNODAL_LDL_LOWER_FACTOR_IMPL_H_
