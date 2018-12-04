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
  std::vector<Int> intersect_size_offsets;

  // The concatenation of the number of rows in each supernodal intersection.
  // The supernodal intersection sizes for supernode j are stored in indices
  // intersect_size_offsets[j] through intersect_size_offsets[j + 1].
  std::vector<Int> intersect_sizes;

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

// The user-facing data structure for storing a supernodal LDL' factorization.
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

  // Creating relaxed supernodes generally involves a permutation; if the
  // following is nonempty, then, if the permutation is a matrix P, the matrix
  // P A P' has been factored.
  std::vector<Int> permutation;

  // The inverse of the above permutation (if it is nontrivial).
  std::vector<Int> inverse_permutation;
};

// Configuration options for supernodal LDL' factorization.
struct SupernodalLDLControl {
  // If true, relaxed supernodes are created in a manner similar to the
  // suggestion from:
  //   Ashcraft and Grime, "The impact of relaxed supernode partitions on the
  //   multifrontal method", 1989.
  bool relax_supernodes = true;

  // The allowable number of explicit zeros in any relaxed supernode.
  Int allowable_supernode_zeros = 128;

  // The allowable ratio of explicit zeros in any relaxed supernode. This
  // should be interpreted as an *alternative* allowance for a supernode merge.
  // If the number of explicit zeros that would be introduced is less than or
  // equal to 'allowable_supernode_zeros', *or* the ratio of explicit zeros to
  // nonzeros is bounded by 'allowable_supernode_zero_ratio', then the merge
  // can procede.
  float allowable_supernode_zero_ratio = 0.01;;
};

// Performs a supernodal LDL' factorization in the natural ordering.
template <class Field>
Int LDL(const CoordinateMatrix<Field>& matrix,
        const SupernodalLDLControl& control,
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

// Prints the unit-diagonal lower-triangular factor of the LDL' factorization.
template <class Field>
void PrintLowerFactor(
    const SupernodalLDLFactorization<Field>& factorization,
    const std::string& label, std::ostream& os);

// Prints the diagonal factor of the LDL' factorization.
template <class Field>
void PrintDiagonalFactor(
    const SupernodalLDLFactorization<Field>& factorization,
    const std::string& label, std::ostream& os);

}  // namespace catamari

#include "catamari/supernodal_ldl-impl.hpp"

#endif  // ifndef CATAMARI_SUPERNODAL_LDL_H_
