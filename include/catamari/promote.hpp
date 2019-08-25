/*
 * Copyright (c) 2018-2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_PROMOTE_H_
#define CATAMARI_PROMOTE_H_

#include "catamari/complex.hpp"

namespace catamari {

namespace promote {

template <typename Real>
struct PromoteHelper {
  typedef Real type;
};

template <>
struct PromoteHelper<float> {
  typedef double type;
};

template <>
struct PromoteHelper<double> {
  typedef mantis::DoubleMantissa<double> type;
};

template <typename Real>
struct PromoteHelper<Complex<Real>> {
  typedef Complex<typename PromoteHelper<Real>::type> type;
};

}  // namespace promote

template <typename Field>
using Promote = typename promote::PromoteHelper<Field>::type;

}  // namespace catamari

#endif  // ifndef CATAMARI_PROMOTE_H_
