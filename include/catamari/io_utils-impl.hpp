/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_IO_UTILS_IMPL_H_
#define CATAMARI_IO_UTILS_IMPL_H_

#include <cstring>
#include <fstream>
#include <iostream>
#include <ostream>
#include <string>

#include "catamari/io_utils.hpp"

namespace catamari {

inline void TruncatedNodeLabelRecursion(Int supernode,
                                        const Buffer<quotient::Timer>& timers,
                                        const AssemblyForest& forest, Int level,
                                        Int max_levels, std::ofstream& file) {
  if (level > max_levels) {
    return;
  }

  // Write out this node's label.
  file << "  " << supernode << " [label=\"" << timers[supernode].TotalSeconds()
       << "\"]\n";

  const Int child_beg = forest.child_offsets[supernode];
  const Int child_end = forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child = forest.children[child_beg + child_index];
    TruncatedNodeLabelRecursion(child, timers, forest, level + 1, max_levels,
                                file);
  }
}

inline void TruncatedNodeEdgeRecursion(Int supernode,
                                       const Buffer<quotient::Timer>& timers,
                                       const AssemblyForest& forest, Int level,
                                       Int max_levels, std::ofstream& file) {
  if (level >= max_levels) {
    return;
  }

  // Write out the child edge connections.
  const Int child_beg = forest.child_offsets[supernode];
  const Int child_end = forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child = forest.children[child_beg + child_index];
    file << "  " << supernode << " -> " << child << "\n";
    TruncatedNodeEdgeRecursion(child, timers, forest, level + 1, max_levels,
                               file);
  }
}

inline void TruncatedForestTimersToDot(const std::string& filename,
                                       const Buffer<quotient::Timer>& timers,
                                       const AssemblyForest& forest,
                                       Int max_levels,
                                       bool avoid_isolated_roots) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Could not open " << filename << std::endl;
    return;
  }

  const Int num_roots = forest.roots.Size();

  // Write out the header.
  file << "digraph g {\n";

  // Write out the node labels.
  Int level = 0;
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = forest.roots[root_index];
    if (!avoid_isolated_roots ||
        forest.child_offsets[root] != forest.child_offsets[root + 1]) {
      TruncatedNodeLabelRecursion(root, timers, forest, level, max_levels,
                                  file);
    }
  }

  file << "\n";

  // Write out the (truncated) connectivity.
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = forest.roots[root_index];
    if (!avoid_isolated_roots ||
        forest.child_offsets[root] != forest.child_offsets[root + 1]) {
      TruncatedNodeEdgeRecursion(root, timers, forest, level, max_levels, file);
    }
  }

  // Write out the footer.
  file << "}\n";
}

#ifdef CATAMARI_HAVE_LIBTIFF
inline void PackPixel(Int i, Int j, Int width, Pixel pixel,
                      std::vector<char>* image) {
  const Int samples_per_pixel = 3;
  Int offset = samples_per_pixel * (j + i * width);
  (*image)[offset++] = pixel.red;
  (*image)[offset++] = pixel.green;
  (*image)[offset++] = pixel.blue;
}

inline void WriteTIFF(const std::string& filename, std::size_t height,
                      std::size_t width, const std::size_t samples_per_pixel,
                      const std::vector<char>& image) {
  TIFF* tiff_img = TIFFOpen(filename.c_str(), "w");

  TIFFSetField(tiff_img, TIFFTAG_IMAGEWIDTH, width);
  TIFFSetField(tiff_img, TIFFTAG_IMAGELENGTH, height);
  TIFFSetField(tiff_img, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel);
  TIFFSetField(tiff_img, TIFFTAG_BITSPERSAMPLE, 8);
  TIFFSetField(tiff_img, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
  TIFFSetField(tiff_img, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(tiff_img, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);

  const std::size_t line_size = samples_per_pixel * width;
  const std::size_t scanline_size = TIFFScanlineSize(tiff_img);
  const std::size_t row_size = std::max(line_size, scanline_size);
  unsigned char* row_buf = (unsigned char*)_TIFFmalloc(row_size);

  for (uint32 row = 0; row < height; ++row) {
    std::memcpy(row_buf, &image[(height - row - 1) * line_size], line_size);
    if (TIFFWriteScanline(tiff_img, row_buf, row, 0) < 0) {
      std::cout << "Could not write row." << std::endl;
      break;
    }
  }

  TIFFClose(tiff_img);
  if (row_buf) {
    _TIFFfree(row_buf);
  }
}
#endif  // ifdef CATAMARI_HAVE_LIBTIFF

}  // namespace catamari

#endif  // ifndef CATAMARI_IO_UTILS_IMPL_H_
