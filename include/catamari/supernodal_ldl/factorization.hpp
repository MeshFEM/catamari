/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SUPERNODAL_LDL_FACTORIZATION_H_
#define CATAMARI_SUPERNODAL_LDL_FACTORIZATION_H_

#include "catamari/supernodal_ldl/diagonal_factor.hpp"
#include "catamari/supernodal_ldl/lower_factor.hpp"

namespace catamari {
namespace supernodal_ldl {

// The user-facing data structure for storing a supernodal LDL' factorization.
template <class Field>
struct Factorization {
  // Marks the type of factorization employed.
  SymmetricFactorizationType factorization_type;

  // An array of length 'num_supernodes'; the i'th member is the size of the
  // i'th supernode.
  std::vector<Int> supernode_sizes;

  // An array of length 'num_supernodes + 1'; the i'th member, for
  // 0 <= i < num_supernodes, is the principal member of the i'th supernode.
  // The last member is equal to 'num_rows'.
  std::vector<Int> supernode_starts;

  // An array of length 'num_rows'; the i'th member is the index of the
  // supernode containing column 'i'.
  std::vector<Int> supernode_member_to_index;

  // The largest supernode size in the factorization.
  Int max_supernode_size;

  // The largest degree of a supernode in the factorization.
  Int max_degree;

  // The largest number of entries in the block row to the left of a diagonal
  // block.
  // NOTE: This is only needed for multithreaded factorizations.
  Int max_descendant_entries;

  // If the following is nonempty, then, if the permutation is a matrix P, the
  // matrix P A P' has been factored. Typically, this permutation is the
  // composition of a fill-reducing ordering and a supernodal relaxation
  // permutation.
  std::vector<Int> permutation;

  // The inverse of the above permutation (if it is nontrivial).
  std::vector<Int> inverse_permutation;

  // The subdiagonal-block portion of the lower-triangular factor.
  std::unique_ptr<LowerFactor<Field>> lower_factor;

  // The block-diagonal factor.
  std::unique_ptr<DiagonalFactor<Field>> diagonal_factor;

  // The minimal supernode size for an out-of-place trapezoidal solve to be
  // used.
  Int forward_solve_out_of_place_supernode_threshold;

  // The minimal supernode size for an out-of-place trapezoidal solve to be
  // used.
  Int backward_solve_out_of_place_supernode_threshold;
};

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_SUPERNODAL_LDL_FACTORIZATION_H_
