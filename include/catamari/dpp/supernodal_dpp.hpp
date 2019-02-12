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

#include "catamari/buffer.hpp"
#include "catamari/ldl.hpp"

namespace catamari {

struct SupernodalDPPControl {
  // Configuration for the supernodal relaxation.
  SupernodalRelaxationControl relaxation_control;

  // The algorithmic block size for the factorization.
  Int block_size = 64;

#ifdef _OPENMP
  // The size of the matrix tiles for factorization OpenMP tasks.
  Int factor_tile_size = 128;

  // The size of the matrix tiles for dense outer product OpenMP tasks.
  Int outer_product_tile_size = 240;

  // The number of columns to group into a single task when multithreading
  // the addition of child Schur complement updates onto the parent.
  Int merge_grain_size = 500;

  // The number of columns to group into a single task when multithreading
  // the scalar structure formation.
  Int sort_grain_size = 200;
#endif  // ifdef _OPENMP
};

// The user-facing data structure for storing a supernodal LDL'-based DPP
// sampler.
//
// TODO(Jack Poulson): Add support for a right-looking supernodal factorization.
template <class Field>
class SupernodalDPP {
 public:
  SupernodalDPP(const CoordinateMatrix<Field>& matrix,
                const SymmetricOrdering& ordering,
                const SupernodalDPPControl& control, unsigned int random_seed);

  // Return a sample from the DPP. If 'maximum_likelihood' is true, then each
  // pivot is kept based upon which choice is most likely.
  std::vector<Int> Sample(bool maximum_likelihood) const;

 private:
  typedef ComplexBase<Field> Real;

  struct LeftLookingSharedState {
    // The relative index of the active supernode within each supernode's
    // structure.
    Buffer<Int> rel_rows;

    // Pointers to the active supernode intersection size within each
    // supernode's structure.
    Buffer<const Int*> intersect_ptrs;
  };

  struct RightLookingSharedState {
    // The Schur complement matrices for each of the supernodes in the
    // multifrontal method. Each front should only be allocated while it is
    // actively in use.
    Buffer<BlasMatrix<Field>> schur_complements;

    // The underlying buffers for the Schur complement portions of the fronts.
    // They are allocated and deallocated as the factorization progresses.
    Buffer<Buffer<Field>> schur_complement_buffers;
  };

  struct PrivateState {
    // An integer workspace for storing the supernodes in the current row
    // pattern.
    Buffer<Int> row_structure;

    // A data structure for marking whether or not a supernode is in the pattern
    // of the active row of the lower-triangular factor.
    Buffer<Int> pattern_flags;

    // A buffer for storing (scaled) transposed descendant blocks.
    Buffer<Field> scaled_transpose_buffer;

    // A buffer for storing updates to the current supernode column.
    Buffer<Field> workspace_buffer;
  };

  // A copy of the input matrix.
  CoordinateMatrix<Field> matrix_;

  // A representation of the permutation and supernodes associated with a
  // reordering choice for the DPP sampling.
  SymmetricOrdering ordering_;

  // The scalar elimination forest.
  AssemblyForest forest_;

  // An array of length 'num_rows'; the i'th member is the index of the
  // supernode containing column 'i'.
  Buffer<Int> supernode_member_to_index_;

  // The degrees of the supernodes.
  Buffer<Int> supernode_degrees_;

  // The size of the largest supernode of the factorization.
  Int max_supernode_size_;

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

#ifdef _OPENMP
  void MultithreadedFormSupernodes();
#endif  // ifdef _OPENMP

  void FormStructure();

#ifdef _OPENMP
  void MultithreadedFormStructure();
#endif  // ifdef _OPENMP

  // Return a sample from the DPP.
  std::vector<Int> LeftLookingSample(bool maximum_likelihood) const;

#ifdef _OPENMP
  std::vector<Int> MultithreadedLeftLookingSample(
      bool maximum_likelihood) const;
#endif  // ifdef _OPENMP

#ifdef _OPENMP
  void LeftLookingSubtree(Int supernode, bool maximum_likelihood,
                          LeftLookingSharedState* shared_state,
                          PrivateState* private_state,
                          std::vector<Int>* sample) const;

  void MultithreadedLeftLookingSubtree(Int level, Int max_parallel_levels,
                                       Int supernode, bool maximum_likelihood,
                                       LeftLookingSharedState* shared_state,
                                       Buffer<PrivateState>* private_states,
                                       std::vector<Int>* subsample) const;
#endif  // ifdef _OPENMP

  // Updates a supernode using its descendants.
  void LeftLookingSupernodeUpdate(Int main_supernode,
                                  LeftLookingSharedState* shared_state,
                                  PrivateState* private_state) const;

#ifdef _OPENMP
  void MultithreadedLeftLookingSupernodeUpdate(
      Int main_supernode, LeftLookingSharedState* shared_state,
      Buffer<PrivateState>* private_states) const;
#endif  // ifdef _OPENMP

  // Appends a supernode's contribution to the current sample.
  void LeftLookingSupernodeSample(Int main_supernode, bool maximum_likelihood,
                                  std::vector<Int>* sample) const;

#ifdef _OPENMP
  void MultithreadedLeftLookingSupernodeSample(
      Int main_supernode, bool maximum_likelihood,
      Buffer<PrivateState>* private_states, std::vector<Int>* sample) const;
#endif  // ifdef _OPENMP
};

}  // namespace catamari

#include "catamari/dpp/supernodal_dpp-impl.hpp"

#endif  // ifndef CATAMARI_SUPERNODAL_DPP_H_
