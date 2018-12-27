/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SUPERNODAL_LDL_LOWER_FACTOR_H_
#define CATAMARI_SUPERNODAL_LDL_LOWER_FACTOR_H_

#include <vector>

#include "catamari/blas_matrix.hpp"

namespace catamari {
namespace supernodal_ldl {

// The representation of the portion of the unit-lower triangular factor
// that is below the supernodal diagonal blocks.
template <class Field>
class LowerFactor {
 public:
  // Representations of the densified subdiagonal blocks of the factorization.
  std::vector<BlasMatrix<Field>> blocks;

  LowerFactor(const std::vector<Int>& supernode_sizes,
              const std::vector<Int>& supernode_degrees);

  Int* Structure(Int supernode);

  const Int* Structure(Int supernode) const;

  Int* IntersectionSizes(Int supernode);

  const Int* IntersectionSizes(Int supernode) const;

  void FillIntersectionSizes(const std::vector<Int>& supernode_sizes,
                             const std::vector<Int>& supernode_member_to_index,
                             Int* max_descendant_entries);

 private:
  // The concatenation of the structures of the supernodes. The structure of
  // supernode j is stored between indices index_offsets[j] and
  // index_offsets[j + 1].
  std::vector<Int> structure_indices_;

  // An array of length 'num_supernodes + 1'; the j'th index is the sum of the
  // degrees (excluding the diagonal blocks) of supernodes 0 through j - 1.
  std::vector<Int> structure_index_offsets_;

  // The concatenation of the number of rows in each supernodal intersection.
  // The supernodal intersection sizes for supernode j are stored in indices
  // intersect_size_offsets[j] through intersect_size_offsets[j + 1].
  std::vector<Int> intersect_sizes_;

  // An array of length 'num_supernodes + 1'; the j'th index is the sum of the
  // number of supernodes that supernodes 0 through j - 1 individually intersect
  // with.
  std::vector<Int> intersect_size_offsets_;

  // The concatenation of the numerical values of the supernodal structures.
  // The entries of supernode j are stored between indices value_offsets[j] and
  // value_offsets[j + 1] in a column-major manner.
  std::vector<Field> values_;
};

}  // namespace supernodal_ldl
}  // namespace catamari

#include "catamari/supernodal_ldl/lower_factor-impl.hpp"

#endif  // ifndef CATAMARI_SUPERNODAL_LDL_LOWER_FACTOR_H_
