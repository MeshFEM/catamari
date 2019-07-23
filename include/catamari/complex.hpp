/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_COMPLEX_H_
#define CATAMARI_COMPLEX_H_

#include "mantis/complex.hpp"

namespace catamari {

using mantis::Complex;
using mantis::ComplexBase;
using mantis::IsComplex;
using mantis::IsReal;

using mantis::DisableIf;
using mantis::EnableIf;

using mantis::operator-;
using mantis::operator+;
using mantis::operator*;
using mantis::operator/;

using mantis::Conjugate;
using mantis::ImagPart;
using mantis::RealPart;

}  // namespace catamari

#endif  // ifndef CATAMARI_COMPLEX_H_
