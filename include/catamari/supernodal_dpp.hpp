/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SUPERNODAL_DPP_H_
#define CATAMARI_SUPERNODAL_DPP_H_

#include <random>

#include "catamari/scalar_ldl.hpp"
#include "catamari/supernodal_ldl.hpp"

namespace catamari {

struct SupernodalDPPControl {
  SupernodalRelaxationControl relaxation_control;
};

// The user-facing data structure for storing a supernodal LDL'-based DPP
// sampler.
template <class Field>
class SupernodalDPP {
 public:
  SupernodalDPP(const CoordinateMatrix<Field>& matrix,
                const std::vector<Int>& permutation,
                const std::vector<Int>& inverse_permutation,
                const SupernodalDPPControl& control, unsigned int random_seed);

  // Return a sample from the DPP. If 'maximum_likelihood' is true, then each
  // pivot is kept based upon which choice is most likely.
  std::vector<Int> Sample(bool maximum_likelihood) const;

 private:
  typedef ComplexBase<Field> Real;

  // A copy of the input matrix.
  CoordinateMatrix<Field> matrix_;

  // If the following is nonempty, then, if the permutation is a matrix P, the
  // matrix P A P' has been factored. Typically, this permutation is the
  // composition of a fill-reducing ordering and a supernodal relaxation
  // permutation.
  std::vector<Int> permutation_;

  // The inverse of the above permutation (if it is nontrivial).
  std::vector<Int> inverse_permutation_;

  // The up-links of the scalar elimination tree.
  std::vector<Int> parents_;

  // The up-links of the supernodal elimination tree.
  std::vector<Int> supernode_parents_;

  // An array of length 'num_supernodes'; the i'th member is the size of the
  // i'th supernode.
  std::vector<Int> supernode_sizes_;

  // An array of length 'num_supernodes + 1'; the i'th member, for
  // 0 <= i < num_supernodes, is the principal member of the i'th supernode.
  // The last member is equal to 'num_rows'.
  std::vector<Int> supernode_starts_;

  // An array of length 'num_rows'; the i'th member is the index of the
  // supernode containing column 'i'.
  std::vector<Int> supernode_member_to_index_;

  // The degrees of the supernodes.
  std::vector<Int> supernode_degrees_;

  // The largest number of entries in the block row to the left of a diagonal
  // block.
  // NOTE: This is only needed for multithreaded sampling.
  Int max_descendant_entries_;

  // The subdiagonal-block portion of the lower-triangular factor.
  mutable std::unique_ptr<SupernodalLowerFactor<Field>> lower_factor_;

  // The block-diagonal factor.
  mutable std::unique_ptr<SupernodalDiagonalFactor<Field>> diagonal_factor_;

  // The controls tructure for the DPP sampler.
  const SupernodalDPPControl control_;

  // A random number generator.
  mutable std::mt19937 generator_;

  // A uniform distribution over [0, 1].
  mutable std::uniform_real_distribution<Real> unit_uniform_;

  void FormSupernodes();

  void FormStructure();

  // Return a sample from the DPP.
  std::vector<Int> LeftLookingSample(bool maximum_likelihood) const;
};

}  // namespace catamari

#include "catamari/supernodal_dpp-impl.hpp"

#endif  // ifndef CATAMARI_SUPERNODAL_DPP_H_
