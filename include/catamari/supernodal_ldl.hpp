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
  // An array of length 'num_supernodes + 1'; the j'th index is the sum of the
  // degrees (excluding the diagonal blocks) of supernodes 0 through j - 1.
  std::vector<Int> index_offsets;

  // The concatenation of the structures of the supernodes. The structure of
  // supernode j is stored between indices index_offsets[j] and
  // index_offsets[j + 1].
  std::vector<Int> indices;

  // An array of length 'num_supernodes + 1'; the j'th index is the sum of the
  // number of supernodes that supernodes 0 through j - 1 individually intersect
  // with.
  std::vector<Int> intersection_size_offsets;

  // The concatenation of the number of rows in each supernodal intersection.
  // The supernodal intersection sizes for supernode j are stored in indices
  // intersection_size_offsets[j] through intersection_size_offsets[j + 1].
  std::vector<Int> intersection_sizes;

  // An array of length 'num_supernodes + 1'; the j'th index is the sum of the
  // number of nonzero entries (the degree times the supernode size) of
  // supernodes 0 through j - 1.
  std::vector<Int> value_offsets;

  // The concatenation of the numerical values of the supernodal structures.
  // The entries of supernode j are stored between indices value_offsets[j] and
  // value_offsets[j + 1].
  std::vector<Field> values;
};

// Stores the (dense) diagonal blocks for the supernodes.
template <class Field>
struct SupernodalDiagonalFactor {
  // An array of length 'num_supernodes + 1'; the j'th index is the sum of the
  // number of nonzero entries (the square of the supernode size) of supernodes
  // 0 through j - 1.
  std::vector<Int> value_offsets;

  // The concatenation of the numerical values of the supernodal diagonal
  // blocks.
  std::vector<Field> values;
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

// Performs a supernodal LDL' factorization in the natural ordering.
template <class Field>
Int LDL(const CoordinateMatrix<Field>& matrix,
        const std::vector<Int>& supernode_sizes, LDLAlgorithm algorithm,
        SupernodalLDLFactorization<Field>* factorization);

// Solve A x = b via the substitution (L D L') x = b and the sequence:
//   x := L' \ (D \ (L \ b)).
template <class Field>
void LDLSolve(const SupernodalLDLFactorization<Field>& factorization,
              std::vector<Field>* vector);

// Solves L x = b using a unit-lower triangular matrix L.
template <class Field>
void UnitLowerTriangularSolve(
    const SupernodalLDLFactorization<Field>& factorization,
    std::vector<Field>* vector);

// Solves D x = b using a diagonal matrix D.
template <class Field>
void DiagonalSolve(const SupernodalLDLFactorization<Field>& factorization,
                   std::vector<Field>* vector);

// Solves L' x = b using a unit-lower triangular matrix L.
template <class Field>
void UnitLowerAdjointTriangularSolve(
    const SupernodalLDLFactorization<Field>& factorization,
    std::vector<Field>* vector);

}  // namespace catamari

#include "catamari/supernodal_ldl-impl.hpp"

#endif  // ifndef CATAMARI_SUPERNODAL_LDL_H_
