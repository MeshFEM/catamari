/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_MACROS_H_
#define CATAMARI_MACROS_H_

#include "quotient/macros.hpp"

#define CATAMARI_ASSERT(condition, msg) QUOTIENT_ASSERT(condition, msg)

#define CATAMARI_NOEXCEPT QUOTIENT_NOEXCEPT

#ifdef __GNUG__
#define CATAMARI_UNUSED __attribute__((unused))
#elif defined(__clang__)
#define CATAMARI_UNUSED __attribute__((unused))
#else
#define CATAMARI_UNUSED
#endif  // ifdef __GNUG__

#endif  // ifndef CATAMARI_MACROS_H_
