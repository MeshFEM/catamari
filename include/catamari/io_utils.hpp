/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_IO_UTILS_H_
#define CATAMARI_IO_UTILS_H_

#ifdef CATAMARI_HAVE_LIBTIFF
#include <tiffio.h>
#endif  // ifdef CATAMARI_HAVE_LIBTIFF

#include "catamari/buffer.hpp"
#include "catamari/integers.hpp"
#include "catamari/symmetric_ordering.hpp"
#include "quotient/timer.hpp"

namespace catamari {

#ifdef CATAMARI_HAVE_LIBTIFF
// A representation of a TIFF pixel (without alpha).
struct Pixel {
  char red;
  char green;
  char blue;
};
#endif  // ifdef CATAMARI_HAVE_LIBTIFF

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

#ifdef CATAMARI_HAVE_LIBTIFF
// Packs a pixel into the given position in a TIFF image buffer.
void PackPixel(Int i, Int j, Int width, Pixel pixel, std::vector<char>* image);

// Saves a row-major RGB pixel map into a TIFF image.
void WriteTIFF(const std::string& filename, std::size_t height,
               std::size_t width, const std::size_t samples_per_pixel,
               const std::vector<char>& image);
#endif  // ifdef CATAMARI_HAVE_LIBTIFF

}  // namespace catamari

#include "catamari/io_utils-impl.hpp"

#endif  // ifndef CATAMARI_IO_UTILS_H_
