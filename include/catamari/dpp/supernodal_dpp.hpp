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

#include "catamari/ldl.hpp"

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

  struct LeftLookingSampleState {
    // An integer workspace for storing the supernodes in the current row
    // pattern.
    std::vector<Int> row_structure;

    // A data structure for marking whether or not a supernode is in the pattern
    // of the active row of the lower-triangular factor.
    std::vector<Int> pattern_flags;

    // The relative index of the active supernode within each supernode's
    // structure.
    std::vector<Int> rel_rows;

    // Pointers to the active supernode intersection size within each
    // supernode's structure.
    std::vector<const Int*> intersect_ptrs;

    // A buffer for storing (scaled) transposed descendant blocks.
    std::vector<Field> scaled_transpose_buffer;

    // A buffer for storing updates to the current supernode column.
    std::vector<Field> workspace_buffer;
  };

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

  // The size of the largest supernode of the factorization.
  Int max_supernode_size_;

  // The largest number of entries in the block row to the left of a diagonal
  // block.
  // NOTE: This is only needed for multithreaded sampling.
  Int max_descendant_entries_;

  // The subdiagonal-block portion of the lower-triangular factor.
  mutable std::unique_ptr<supernodal_ldl::LowerFactor<Field>> lower_factor_;

  // The block-diagonal factor.
  mutable std::unique_ptr<supernodal_ldl::DiagonalFactor<Field>>
      diagonal_factor_;

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

  // Updates a supernode using its descendants.
  void LeftLookingSupernodeUpdate(Int main_supernode,
                                  LeftLookingSampleState* state) const;

  // Appends a supernode's contribution to the current sample.
  void LeftLookingSupernodeSample(Int main_supernode, bool maximum_likelihood,
                                  std::vector<Int>* sample) const;
};

}  // namespace catamari

#include "catamari/dpp/supernodal_dpp-impl.hpp"

#endif  // ifndef CATAMARI_SUPERNODAL_DPP_H_
