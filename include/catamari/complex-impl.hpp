/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_COMPLEX_IMPL_H_
#define CATAMARI_COMPLEX_IMPL_H_

#include "catamari/complex.hpp"

namespace catamari {

template<class Real>
Real RealPart(const Real& value) CATAMARI_NOEXCEPT { return value; }

template<class Real>
Real RealPart(const Complex<Real>& value) CATAMARI_NOEXCEPT {
  return value.real();
}

template<class Real>
Real ImagPart(const Real& value) CATAMARI_NOEXCEPT { return 0; }

template<class Real>
Real ImagPart(const Complex<Real>& value) CATAMARI_NOEXCEPT {
  return value.imag();
}

} // namespace catamari

#endif // ifndef CATAMARI_COMPLEX_IMPL_H_
