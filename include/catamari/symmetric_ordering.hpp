/*
 * Copyright (c) 2018-2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SYMMETRIC_ORDERING_H_
#define CATAMARI_SYMMETRIC_ORDERING_H_

#include "catamari/buffer.hpp"
#include "catamari/coordinate_matrix.hpp"
#include "quotient/integers.hpp"

namespace catamari {

// A representation of a (scalar or supernodal) assembly forest via its up and
// down links.
struct AssemblyForest {
  // The parents in the assembly forest.The length of the array is the number
  // of (super)nodes.
  Buffer<Int> parents;

  // The packed downlinks of the forest; the uplinks are given by 'parents'.
  Buffer<Int> children;

  // The offsets associated with the packed downlinks. The children of
  // (super)node 'j' would be stored between indices 'child_offsets[j]' and
  // 'child_offsets[j + 1]'. The length of this array is one more than the
  // number of (super)nodes.
  Buffer<Int> child_offsets;

  // The indices of the root supernodes in the forest.
  Buffer<Int> roots;

  // Julian Panetta: mapping from each child structure to its parent front.
  Buffer<Buffer<Int>> child_rel_indices;
  Buffer<Int> num_child_diag_indices;

  // Fills the children and root list from the parent list.
  void FillFromParents();

  // Returns the number of children for the node with the given index.
  Int NumChildren(Int index) const;
};

// A mechanism for passing reordering information into the factorization.
struct SymmetricOrdering {
  // If non-empty, the permutation mapping the original matrix ordering into the
  // factorization ordering.
  Buffer<Int> permutation;

  // If non-empty, the inverse of the permutation mapping the original matrix
  // ordering into the factorization ordering.
  Buffer<Int> inverse_permutation;

  // The length of each (reordering) supernode in the new ordering. The
  // length of the array is the number of supernodes.
  //
  // This member is optional, but, if coupled with 'permuted_assembly_parents',
  // allows for the factorization to parallelize its symbolic analysis.
  Buffer<Int> supernode_sizes;

  // The array of length 'num_supernodes + 1' such that permuted supernode 'j'
  // corresponds to indices 'permuted_supernode_offsets[j]' through
  // 'permuted_supernode_offsets[j + 1]'.
  Buffer<Int> supernode_offsets;

  // The (optional) supernodal assembly forest in the permuted ordering.
  AssemblyForest assembly_forest;
};

// Permutes a symmetric matrix using the given reordering.
template <class Field>
void PermuteMatrix(const CoordinateMatrix<Field>& matrix,
                   const SymmetricOrdering& ordering,
                   CoordinateMatrix<Field>* reordered_matrix);

}  // namespace catamari

#include "catamari/symmetric_ordering-impl.hpp"

#endif  // ifndef CATAMARI_SYMMETRIC_ORDERING_H_
