/*
 * Copyright (c) 2018-2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DYNAMIC_REGULARIZATION_H_
#define CATAMARI_DYNAMIC_REGULARIZATION_H_

#include <vector>

#include "catamari/buffer.hpp"
#include "catamari/integers.hpp"

namespace catamari {

// A structure for supporting dynamic regularization in an LDL^H factorization.
// It is especially useful for Hermitian quasi-(semi)definite matrices.
template <typename Field>
struct DynamicRegularizationControl {
  typedef ComplexBase<Field> Real;

  // Whether dynamic regularization is enabled.
  bool enabled = false;

  // A map from the factorization indices to whether the pivot should be
  // enforced as positive or negative.
  Buffer<bool> signatures;

  // If a pivot with a positive signature has a value less than the maximum
  // entry magnitude in the matrix times machine epsilon raised to this power,
  // then it will be increased up to it.
  Real positive_threshold_exponent = Real(1);

  // If a pivot with a negative signature has a negated value less than the
  // maximum entry magnitude in the matrix times machine epsilon raised to this
  // power, then it will be increased up to it.
  Real negative_threshold_exponent = Real(1);
};

// A structure for use in processing diagonal blocks of a dynamically
// regularized factorization.
template <typename Field>
struct DynamicRegularizationParams {
  typedef ComplexBase<Field> Real;

  // Whether dynamic regularization is enabled.
  bool enabled = false;

  // The minimum value of a positive pivot.
  Real positive_threshold;

  // The minimum (negated) value of a negative pivot.
  Real negative_threshold;

  // The integer offset of the current subproblem to use when determining the
  // signature of a pivot or inserting a new regularization entry.
  Int offset = 0;

  // A pointer to the global specification of whether each index should be
  // interpreted as a positive or negative pivot. The sibling 'offset' parameter
  // is needed to access this diagonal block's indices.
  const Buffer<bool>* signatures;
};

}  // namespace catamari

#endif  // ifndef CATAMARI_DYNAMIC_REGULARIZATION_H_
