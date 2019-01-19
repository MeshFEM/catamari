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
  std::vector<Int> permuted_supernode_sizes;

  // The (optional) supernodal assembly forest in the permuted ordering.
  AssemblyForest permuted_assembly_forest;
};

}  // namespace catamari

#endif  // ifndef CATAMARI_SYMMETRIC_ORDERING_H_
