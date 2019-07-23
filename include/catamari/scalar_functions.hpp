/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SCALAR_FUNCTIONS_H_
#define CATAMARI_SCALAR_FUNCTIONS_H_

#include "catamari/complex.hpp"

namespace catamari {

// A helper function for computing the Euclidean norm of a matrix. Given an
// implicit representation 'value = scale * sqrt(scaled_square)', it updates
// 'scale' and 'scaled_square' so that
// 'value + update = scale * sqrt(scaled_square)'.
template <typename Field>
void UpdateScaledSquare(const Field& update, ComplexBase<Field>* scale,
                        ComplexBase<Field>* scaled_square) {
  typedef ComplexBase<Field> Real;
  const Real abs_update = std::abs(update);
  if (abs_update == Real(0)) {
    return;
  }

  if (abs_update <= *scale) {
    const Real rel_scale = abs_update / *scale;
    *scaled_square += rel_scale * rel_scale;
  } else {
    const Real rel_scale = *scale / abs_update;
    *scaled_square = *scaled_square * rel_scale * rel_scale + Real{1};
    *scale = abs_update;
  }
}

}  // namespace catamari

#endif  // ifndef CATAMARI_SCALAR_FUNCTIONS_H_
