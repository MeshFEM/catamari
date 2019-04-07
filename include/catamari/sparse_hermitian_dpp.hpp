/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_HERMITIAN_DPP_H_
#define CATAMARI_SPARSE_HERMITIAN_DPP_H_

#include "catamari/sparse_hermitian_dpp/scalar.hpp"
#include "catamari/sparse_hermitian_dpp/supernodal.hpp"
#include "quotient/minimum_degree.hpp"

namespace catamari {

struct SparseHermitianDPPControl {
  // The configuration options for the Minimum Degree reordering.
  quotient::MinimumDegreeControl md_control;

  // Whether or not a supernodal factorization should be used.
  SupernodalStrategy supernodal_strategy = kAdaptiveSupernodalStrategy;

  // The configuration options for the scalar DPP sampler.
  ScalarHermitianDPPControl scalar_control;

  // The configuration options for the supernodal DPP sampler.
  SupernodalHermitianDPPControl supernodal_control;
};

// The user-facing data structure for storing an LDL'-based DPP sampler.
template <class Field>
class SparseHermitianDPP {
 public:
  // Constructs a DPP using an automatically-determined (Minimum Degree)
  // reordering.
  SparseHermitianDPP(const CoordinateMatrix<Field>& matrix,
                     const SparseHermitianDPPControl& control);

  // Constructs a DPP using a user-specified ordering.
  SparseHermitianDPP(const CoordinateMatrix<Field>& matrix,
                     const SymmetricOrdering& ordering,
                     const SparseHermitianDPPControl& control);

  // Return a sample from the DPP. If 'maximum_likelihood' is true, then each
  // pivot is kept based upon which choice is most likely.
  std::vector<Int> Sample(bool maximum_likelihood) const;

  // Returns the log-likelihood of the last DPP sample.
  ComplexBase<Field> LogLikelihood() const;

 private:
  // Whether or not a supernodal sampler was used. If it is true, only
  // 'supernodal_dpp_' should be non-null, and vice versa.
  bool is_supernodal_;

  // The scalar DPP sampling structure.
  std::unique_ptr<ScalarHermitianDPP<Field>> scalar_dpp_;

  // The supernodal DPP sampling structure.
  std::unique_ptr<SupernodalHermitianDPP<Field>> supernodal_dpp_;
};

}  // namespace catamari

#include "catamari/sparse_hermitian_dpp-impl.hpp"

#endif  // ifndef CATAMARI_SPARSE_HERMITIAN_DPP_H_
