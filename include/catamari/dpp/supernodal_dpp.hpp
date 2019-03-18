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

  // The choice of either left-looking or right-looking DPP sampling.
  // There is currently only support for left-looking and right-looking.
  LDLAlgorithm algorithm = kRightLookingLDL;

  // The algorithmic block size for the factorization.
  Int block_size = 64;

#ifdef CATAMARI_OPENMP
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
#endif  // ifdef CATAMARI_OPENMP
};

// The user-facing data structure for storing a supernodal LDL'-based DPP
// sampler.
template <class Field>
class SupernodalDPP {
 public:
  SupernodalDPP(const CoordinateMatrix<Field>& matrix,
                const SymmetricOrdering& ordering,
                const SupernodalDPPControl& control);

  // Return a sample from the DPP. If 'maximum_likelihood' is true, then each
  // pivot is kept based upon which choice is most likely.
  std::vector<Int> Sample(bool maximum_likelihood) const;

 private:
  typedef ComplexBase<Field> Real;

  struct PrivateState {
    supernodal_ldl::PrivateState<Field> ldl_state;

    // A random number generator.
    std::mt19937 generator;
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

  void FormSupernodes();

#ifdef CATAMARI_OPENMP
  void OpenMPFormSupernodes();
#endif  // ifdef CATAMARI_OPENMP

  void FormStructure();

#ifdef CATAMARI_OPENMP
  void OpenMPFormStructure();
#endif  // ifdef CATAMARI_OPENMP

  // Return a sample from the DPP using a left-looking algorithm.
  std::vector<Int> LeftLookingSample(bool maximum_likelihood) const;

#ifdef CATAMARI_OPENMP
  std::vector<Int> OpenMPLeftLookingSample(bool maximum_likelihood) const;
#endif  // ifdef CATAMARI_OPENMP

#ifdef CATAMARI_OPENMP
  void LeftLookingSubtree(Int supernode, bool maximum_likelihood,
                          supernodal_ldl::LeftLookingSharedState* shared_state,
                          PrivateState* private_state,
                          std::vector<Int>* sample) const;

  void OpenMPLeftLookingSubtree(
      Int level, Int max_parallel_levels, Int supernode,
      bool maximum_likelihood,
      supernodal_ldl::LeftLookingSharedState* shared_state,
      Buffer<PrivateState>* private_states, std::vector<Int>* subsample) const;
#endif  // ifdef CATAMARI_OPENMP

  // Updates a supernode using its descendants.
  void LeftLookingSupernodeUpdate(
      Int main_supernode, supernodal_ldl::LeftLookingSharedState* shared_state,
      PrivateState* private_state) const;

#ifdef CATAMARI_OPENMP
  void OpenMPLeftLookingSupernodeUpdate(
      Int main_supernode, supernodal_ldl::LeftLookingSharedState* shared_state,
      Buffer<PrivateState>* private_states) const;
#endif  // ifdef CATAMARI_OPENMP

  // Appends a supernode's contribution to the current sample.
  void LeftLookingSupernodeSample(Int supernode, bool maximum_likelihood,
                                  PrivateState* private_state,
                                  std::vector<Int>* sample) const;

#ifdef CATAMARI_OPENMP
  void OpenMPLeftLookingSupernodeSample(Int main_supernode,
                                        bool maximum_likelihood,
                                        Buffer<PrivateState>* private_states,
                                        std::vector<Int>* sample) const;
#endif  // ifdef CATAMARI_OPENMP

  // Return a sample from the DPP using a right-looking algorithm.
  std::vector<Int> RightLookingSample(bool maximum_likelihood) const;

#ifdef CATAMARI_OPENMP
  std::vector<Int> OpenMPRightLookingSample(bool maximum_likelihood) const;
#endif  // ifdef CATAMARI_OPENMP

  void RightLookingSubtree(
      Int supernode, bool maximum_likelihood,
      supernodal_ldl::RightLookingSharedState<Field>* shared_state,
      PrivateState* private_state, std::vector<Int>* sample) const;

#ifdef CATAMARI_OPENMP
  void OpenMPRightLookingSubtree(
      Int level, Int max_parallel_levels, Int supernode,
      bool maximum_likelihood,
      supernodal_ldl::RightLookingSharedState<Field>* shared_state,
      Buffer<PrivateState>* private_states, std::vector<Int>* sample) const;
#endif  // ifdef CATAMARI_OPENMP

  void RightLookingSupernodeSample(
      Int supernode, bool maximum_likelihood,
      supernodal_ldl::RightLookingSharedState<Field>* shared_state,
      PrivateState* private_state, std::vector<Int>* sample) const;

#ifdef CATAMARI_OPENMP
  void OpenMPRightLookingSupernodeSample(
      Int supernode, bool maximum_likelihood,
      supernodal_ldl::RightLookingSharedState<Field>* shared_state,
      Buffer<PrivateState>* private_states, std::vector<Int>* sample) const;
#endif  // ifdef CATAMARI_OPENMP

  // Appends a supernode sample into an unsorted sample vector in the
  // original ordering.
  void AppendSupernodeSample(Int supernode,
                             const std::vector<Int>& supernode_sample,
                             std::vector<Int>* sample) const;
};

}  // namespace catamari

#include "catamari/dpp/supernodal_dpp/common-impl.hpp"
#include "catamari/dpp/supernodal_dpp/common_openmp-impl.hpp"
#include "catamari/dpp/supernodal_dpp/left_looking-impl.hpp"
#include "catamari/dpp/supernodal_dpp/left_looking_openmp-impl.hpp"
#include "catamari/dpp/supernodal_dpp/right_looking-impl.hpp"
#include "catamari/dpp/supernodal_dpp/right_looking_openmp-impl.hpp"

#endif  // ifndef CATAMARI_SUPERNODAL_DPP_H_
