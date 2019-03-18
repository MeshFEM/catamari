/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SCALAR_DPP_H_
#define CATAMARI_SCALAR_DPP_H_

#include <random>

#include "catamari/buffer.hpp"
#include "catamari/ldl.hpp"

namespace catamari {

struct ScalarDPPControl {
  // Currently, the only choice is 'up-looking' scalar DPP sampling.
  LDLAlgorithm algorithm = kUpLookingLDL;
};

// The user-facing data structure for storing a scalar LDL'-based DPP sampler.
template <class Field>
class ScalarDPP {
 public:
  ScalarDPP(const CoordinateMatrix<Field>& matrix,
            const SymmetricOrdering& ordering, const ScalarDPPControl& control);

  // Return a sample from the DPP. If 'maximum_likelihood' is true, then each
  // pivot is kept based upon which choice is most likely.
  std::vector<Int> Sample(bool maximum_likelihood) const;

 private:
  typedef ComplexBase<Field> Real;

  struct UpLookingState {
    scalar_ldl::UpLookingState<Field> ldl_state;

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

  // The subdiagonal-block portion of the lower-triangular factor.
  mutable std::unique_ptr<scalar_ldl::LowerFactor<Field>> lower_factor_;

  // The block-diagonal factor.
  mutable std::unique_ptr<scalar_ldl::DiagonalFactor<Real>> diagonal_factor_;

  // The controls tructure for the DPP sampler.
  const ScalarDPPControl control_;

  // Return a sample from the DPP using an up-looking algorithm.
  std::vector<Int> UpLookingSample(bool maximum_likelihood) const;

  // Initializes for running an up-looking factorization.
  void UpLookingSetup() CATAMARI_NOEXCEPT;

  // For each index 'i' in the structure of each column 'column' of L formed
  // so far:
  //   L(row, i) -= (L(row, column) * d(column)) * conj(L(i, column)).
  // L(row, row) is similarly updated, within d, then L(row, column) is
  // finalized.
  void UpLookingRowUpdate(Int row, const Int* column_beg, const Int* column_end,
                          Int* column_update_ptrs,
                          Field* row_workspace) const CATAMARI_NOEXCEPT;
};

}  // namespace catamari

#include "catamari/dpp/scalar_dpp-impl.hpp"

#endif  // ifndef CATAMARI_SCALARL_DPP_H_
