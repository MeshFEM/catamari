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

#define CATAMARI_UNUSED QUOTIENT_UNUSED

#ifdef CATAMARI_ENABLE_TIMERS
#define CATAMARI_START_TIMER(timer) timer.Start()
#define CATAMARI_STOP_TIMER(timer) timer.Stop()
#else
#define CATAMARI_START_TIMER(timer)
#define CATAMARI_STOP_TIMER(timer)
#endif  // ifdef CATAMARI_ENABLE_TIMERS

#endif  // ifndef CATAMARI_MACROS_H_
