/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SYMMETRIC_ORDERING_H_
#define CATAMARI_SYMMETRIC_ORDERING_H_

#include "quotient/integers.hpp"

namespace catamari {

// A representation of a (scalar or supernodal) assembly forest via its up and
// down links.
struct AssemblyForest {
  // The parents in the assembly forest.The length of the array is the number
  // of (super)nodes.
  std::vector<Int> parents;

  // The packed downlinks of the forest; the uplinks are given by 'parents'.
  std::vector<Int> children;

  // The offsets associated with the packed downlinks. The children of
  // (super)node 'j' would be stored between indices 'child_offsets[j]' and
  // 'child_offsets[j + 1]'. The length of this array is one more than the
  // number of (super)nodes.
  std::vector<Int> child_offsets;

  // The indices of the root supernodes in the forest.
  std::vector<Int> roots;

  void FillFromParents();
};

// A mechanism for passing reordering information into the factorization.
struct SymmetricOrdering {
  // If non-empty, the permutation mapping the original matrix ordering into the
  // factorization ordering.
  std::vector<Int> permutation;

  // If non-empty, the inverse of the permutation mapping the original matrix
  // ordering into the factorization ordering.
  std::vector<Int> inverse_permutation;

  // The length of each (reordering) supernode in the new ordering. The
  // length of the array is the number of supernodes.
  //
  // This member is optional, but, if coupled with 'permuted_assembly_parents',
  // allows for the factorization to parallelize its symbolic analysis.
  std::vector<Int> supernode_sizes;

  // The array of length 'num_supernodes + 1' such that permuted supernode 'j'
  // corresponds to indices 'permuted_supernode_offsets[j]' through
  // 'permuted_supernode_offsets[j + 1]'.
  std::vector<Int> supernode_offsets;

  // The (optional) supernodal assembly forest in the permuted ordering.
  AssemblyForest assembly_forest;
};

inline void AssemblyForest::FillFromParents() {
  const Int num_indices = parents.size();

  // Compute the number of children (initially stored in 'child_offsets') of
  // each vertex. Along the way, count the number of trees in the forest.
  Int num_roots = 0;
  child_offsets.clear();
  child_offsets.resize(num_indices + 1, 0);
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
  children.resize(num_total_children);
  roots.clear();
  roots.reserve(num_roots);
  std::vector<Int> offsets_copy = child_offsets;
  for (Int index = 0; index < num_indices; ++index) {
    const Int parent = parents[index];
    if (parent >= 0) {
      children[offsets_copy[parent]++] = index;
    } else {
      roots.push_back(index);
    }
  }
}

}  // namespace catamari

#endif  // ifndef CATAMARI_SYMMETRIC_ORDERING_H_
