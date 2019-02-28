/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LAPACK_H_
#define CATAMARI_LAPACK_H_

#include "catamari/blas.hpp"

#ifdef CATAMARI_HAVE_LAPACK

#define LAPACK_SYMBOL(name) name##_

#ifndef CATAMARI_HAVE_LAPACK_PROTOS

#include "catamari/lapack/protos.hpp"

#endif  // ifndef CATAMARI_HAVE_LAPACK_PROTOS

#endif  // ifdef CATAMARI_HAVE_LAPACK

#endif  // ifndef CATAMARI_LAPACK_H_
