/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_COMPLEX_H_
#define CATAMARI_COMPLEX_H_

#include "quotient/complex.hpp"

namespace catamari {

using quotient::Complex;
using quotient::ComplexBase;
using quotient::IsComplex;
using quotient::IsReal;

using quotient::DisableIf;
using quotient::EnableIf;

using quotient::operator-;
using quotient::operator+;
using quotient::operator*;
using quotient::operator/;

using quotient::Conjugate;
using quotient::ImagPart;
using quotient::RealPart;

}  // namespace catamari

#endif  // ifndef CATAMARI_COMPLEX_H_
