/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DPP_IMPL_H_
#define CATAMARI_DPP_IMPL_H_

#include "catamari/dpp.hpp"

namespace catamari {

template <class Field>
DPP<Field>::DPP(const CoordinateMatrix<Field>& matrix,
                const DPPControl& control) {
  std::unique_ptr<quotient::CoordinateGraph> graph = matrix.CoordinateGraph();
  const quotient::MinimumDegreeResult analysis =
      quotient::MinimumDegree(*graph, control.md_control);
  const std::vector<Int> permutation = analysis.Permutation();

  const Int num_rows = permutation.size();
  std::vector<Int> inverse_permutation(num_rows);
  for (Int row = 0; row < num_rows; ++row) {
    inverse_permutation[permutation[row]] = row;
  }

  supernodal_dpp_.reset(new SupernodalDPP<Field>(
      matrix, permutation, inverse_permutation, control.supernodal_control,
      control.random_seed));
}

template <class Field>
std::vector<Int> DPP<Field>::Sample(bool maximum_likelihood) const {
  return supernodal_dpp_->Sample(maximum_likelihood);
}

}  // namespace catamari

#endif  // ifndef CATAMARI_DPP_IMPL_H_
