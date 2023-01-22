/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_LOWER_FACTOR_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_LOWER_FACTOR_H_

#include <vector>

#include "catamari/blas_matrix_view.hpp"
#include "catamari/buffer.hpp"
#include "catamari_config.hh"

namespace catamari {
namespace supernodal_ldl {

// The representation of the portion of the unit-lower triangular factor
// that is below the supernodal diagonal blocks.
template <class Field>
class LowerFactor {
 public:
  // Representations of the densified subdiagonal blocks of the factorization.
  Buffer<BlasMatrixView<Field>> blocks;

  LowerFactor(const Buffer<Int>& supernode_sizes,
              const Buffer<Int>& supernode_degrees,
              BlasMatrixView<Field> storage);

  // Returns a pointer to the beginning of the structure of a supernode.
  Int* StructureBeg(Int supernode);

  // Returns an immutable pointer to the beginning of the structure of a
  // supernode.
  const Int* StructureBeg(Int supernode) const;

  // Returns a pointer to the end of the structure of a supernode.
  Int* StructureEnd(Int supernode);

  // Returns an immutable pointer to the end of the structure of a supernode.
  const Int* StructureEnd(Int supernode) const;

  // Returns a pointer to the beginning of the supernodal intersection sizes
  // of a supernode.
  Int* IntersectionSizesBeg(Int supernode);

  // Returns an immutable pointer to the beginning of the supernodal
  // intersection sizes of a supernode.
  const Int* IntersectionSizesBeg(Int supernode) const;

  // Returns a pointer to the end of the supernodal intersection sizes of a
  // supernode.
  Int* IntersectionSizesEnd(Int supernode);

  // Returns an immutable pointer to the end of the supernodal intersection
  // sizes of a supernode.
  const Int* IntersectionSizesEnd(Int supernode) const;

  void FillIntersectionSizes(const Buffer<Int>& supernode_sizes,
                             const Buffer<Int>& supernode_member_to_index);

 private:
  // The concatenation of the structures of the supernodes. The structure of
  // supernode j is stored between indices index_offsets[j] and
  // index_offsets[j + 1].
  Buffer<Int> structure_indices_;

  // An array of length 'num_supernodes + 1'; the j'th index is the sum of the
  // degrees (excluding the diagonal blocks) of supernodes 0 through j - 1.
  Buffer<Int> structure_index_offsets_;

  // The concatenation of the number of rows in each supernodal intersection.
  // The supernodal intersection sizes for supernode j are stored in indices
  // intersect_size_offsets[j] through intersect_size_offsets[j + 1].
  Buffer<Int> intersect_sizes_;

  // An array of length 'num_supernodes + 1'; the j'th index is the sum of the
  // number of supernodes that supernodes 0 through j - 1 individually intersect
  // with.
  Buffer<Int> intersect_size_offsets_;
};

}  // namespace supernodal_ldl
}  // namespace catamari

#include "catamari/sparse_ldl/supernodal/lower_factor-impl.hpp"

#endif  // ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_LOWER_FACTOR_H_
