/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DPP_H_
#define CATAMARI_DPP_H_

#include "catamari/dpp/supernodal_dpp.hpp"
#include "quotient/minimum_degree.hpp"

namespace catamari {

struct DPPControl {
  quotient::MinimumDegreeControl md_control;
  SupernodalDPPControl supernodal_control;
};

// The user-facing data structure for storing an LDL'-based DPP sampler.
template <class Field>
class DPP {
 public:
  DPP(const CoordinateMatrix<Field>& matrix, const DPPControl& control);

  // Return a sample from the DPP. If 'maximum_likelihood' is true, then each
  // pivot is kept based upon which choice is most likely.
  std::vector<Int> Sample(bool maximum_likelihood) const;

 private:
  std::unique_ptr<SupernodalDPP<Field>> supernodal_dpp_;
};

}  // namespace catamari

#include "catamari/dpp-impl.hpp"

#endif  // ifndef CATAMARI_DPP_H_
