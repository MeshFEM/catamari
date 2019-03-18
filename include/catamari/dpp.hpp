/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DPP_H_
#define CATAMARI_DPP_H_

#include "catamari/dpp/scalar_dpp.hpp"
#include "catamari/dpp/supernodal_dpp.hpp"
#include "quotient/minimum_degree.hpp"

namespace catamari {

struct DPPControl {
  // The configuration options for the Minimum Degree reordering.
  quotient::MinimumDegreeControl md_control;

  // Whether or not a supernodal factorization should be used.
  SupernodalStrategy supernodal_strategy = kAdaptiveSupernodalStrategy;

  // The configuration options for the scalar DPP sampler.
  ScalarDPPControl scalar_control;

  // The configuration options for the supernodal DPP sampler.
  SupernodalDPPControl supernodal_control;
};

// The user-facing data structure for storing an LDL'-based DPP sampler.
template <class Field>
class DPP {
 public:
  // Constructs a DPP using an automatically-determined (Minimum Degree)
  // reordering.
  DPP(const CoordinateMatrix<Field>& matrix, const DPPControl& control);

  // Constructs a DPP using a user-specified ordering.
  DPP(const CoordinateMatrix<Field>& matrix, const SymmetricOrdering& ordering,
      const DPPControl& control);

  // Return a sample from the DPP. If 'maximum_likelihood' is true, then each
  // pivot is kept based upon which choice is most likely.
  std::vector<Int> Sample(bool maximum_likelihood) const;

 private:
  // Whether or not a supernodal sampler was used. If it is true, only
  // 'supernodal_dpp_' should be non-null, and vice versa.
  bool is_supernodal_;

  // The scalar DPP sampling structure.
  std::unique_ptr<ScalarDPP<Field>> scalar_dpp_;

  // The supernodal DPP sampling structure.
  std::unique_ptr<SupernodalDPP<Field>> supernodal_dpp_;
};

}  // namespace catamari

#include "catamari/dpp-impl.hpp"

#endif  // ifndef CATAMARI_DPP_H_
