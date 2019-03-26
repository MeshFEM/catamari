/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_IO_UTILS_H_
#define CATAMARI_IO_UTILS_H_

#include "catamari/buffer.hpp"
#include "catamari/integers.hpp"
#include "catamari/symmetric_ordering.hpp"
#include "quotient/timer.hpp"

namespace catamari {

// Writes out a GraphViz DOT file for the first 'max_levels' levels of the
// forest, annotated by the total time from the given timers. Said times are
// typically the inclusive or exclusive processing time of a supernode.
//
// The 'avoid_isolated_roots' option provides a means of skipping visualization
// of isolated diagonal entries.
void TruncatedForestTimersToDot(const std::string& filename,
                                const Buffer<quotient::Timer>& timers,
                                const AssemblyForest& forest, Int max_levels,
                                bool avoid_isolated_roots);

}  // namespace catamari

#include "catamari/io_utils-impl.hpp"

#endif  // ifndef CATAMARI_IO_UTILS_H_
