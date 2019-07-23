/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// Generates a (maximum-likelihood) sample from a Determinantal Point Process
// which generates a uniform sample from the set of spanning trees of either
// Z^2, Z^3, or a hexagonal tiling of the plane.
//
// Please see
//
//   Yuval Wigderson,
//   http://web.stanford.edu/~yuvalwig/math/teaching/UniformSpanningTrees.pdf
//
// which cites
//
//   Lyons and Peres, Probability on Trees and Networks.
//
// These references are useful for introducing the star space and the definition
// of the transfer current matrix in terms of an orthogonal projector onto the
// star space. We build our DPP by computing a column-pivoted QR factorization
// of a basis for the star space, explicitly computing the Gramian of the
// thin Q factor, and sampling from the resulting SPSD kernel matrix.
//
// Please also see the foundational paper:
//
//   Burton and Pemantle, Local Characteristics, Entropy and Limit Theorems for
//   Spanning Trees and Domino Tilings Via Transfer-Impedances,
//   The Annals of Probability, 21 (3) 1993.
//   URL: https://projecteuclid.org/euclid.aop/1176989121.
//
#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

#include "catamari/blas_matrix.hpp"
#include "catamari/dense_dpp.hpp"
#include "catamari/io_utils.hpp"
#include "quotient/timer.hpp"
#include "specify.hpp"

// Column-pivoted QR factorization is an integral portion of this example
// driver, and so the entire driver is disabled if LAPACK support was not
// detected.
#ifdef CATAMARI_HAVE_LAPACK

using catamari::BlasMatrix;
using catamari::BlasMatrixView;
using catamari::Complex;
using catamari::ComplexBase;
using catamari::ConstBlasMatrixView;
using catamari::Int;
using quotient::Buffer;

extern "C" {

void LAPACK_SYMBOL(dgeqpf)(const BlasInt* height, const BlasInt* width,
                           double* matrix, const BlasInt* leading_dim,
                           BlasInt* column_pivots, double* reflector_scalars,
                           double* work, BlasInt* info);

void LAPACK_SYMBOL(dorgqr)(const BlasInt* height, const BlasInt* width,
                           const BlasInt* num_reflectors, double* matrix,
                           const BlasInt* leading_dim,
                           const double* reflector_scalars, double* work,
                           const BlasInt* work_size, BlasInt* info);

void LAPACK_SYMBOL(sgeqpf)(const BlasInt* height, const BlasInt* width,
                           float* matrix, const BlasInt* leading_dim,
                           BlasInt* column_pivots, float* reflector_scalars,
                           float* work, BlasInt* info);

void LAPACK_SYMBOL(sorgqr)(const BlasInt* height, const BlasInt* width,
                           const BlasInt* num_reflectors, float* matrix,
                           const BlasInt* leading_dim,
                           const float* reflector_scalars, float* work,
                           const BlasInt* work_size, BlasInt* info);

}  // extern "C"

namespace {

#ifdef CATAMARI_HAVE_LIBTIFF
// Returns whether the given value exists within a sorted list.
template <typename Iterator, typename T>
bool IndexExists(Iterator beg, Iterator end, T value) {
  auto iter = std::lower_bound(beg, end, value);
  return iter != end && *iter == value;
}

// Prints out an (x, y) slice of a Z^3 spanning tree.
inline void WriteGridXYSliceToTIFF(const std::string& filename, Int x_size,
                                   Int y_size, Int z_size, Int z,
                                   const std::vector<Int>& sample,
                                   const Int box_size,
                                   const catamari::Pixel& background_pixel,
                                   const catamari::Pixel& active_pixel,
                                   bool negate) {
  const Int num_x_edges = (x_size - 1) * y_size * z_size;
  const Int num_y_edges = x_size * (y_size - 1) * z_size;

  const auto x_beg = sample.begin();
  const auto x_end =
      std::lower_bound(sample.begin(), sample.end(), num_x_edges);
  const auto y_beg = x_end;
  const auto y_end =
      std::lower_bound(y_beg, sample.end(), num_x_edges + num_y_edges);

  const Int height = box_size * (y_size - 1) + 1;
  const Int width = box_size * (x_size - 1) + 1;
  const Int samples_per_pixel = 3;
  std::vector<char> image(width * height * samples_per_pixel);

  // Fill the background pixels.
  for (Int i = 0; i < height; ++i) {
    for (Int j = 0; j < width; ++j) {
      catamari::PackPixel(i, j, width, background_pixel, &image);
    }
  }

  for (Int y = 0; y < y_size; ++y) {
    // Write any horizontal connections in row 'y' and depth 'z'.
    for (Int x = 0; x < x_size - 1; ++x) {
      const Int x_ind = x + y * (x_size - 1) + z * (x_size - 1) * y_size;
      const bool found = IndexExists(x_beg, x_end, x_ind);
      if (negate != found) {
        const Int i = y * box_size;
        const Int j_offset = x * box_size;
        for (Int j = j_offset; j <= j_offset + box_size; ++j) {
          catamari::PackPixel(i, j, width, active_pixel, &image);
        }
      }
    }

    if (y < y_size - 1) {
      // Write any vertical edges from this row.
      for (Int x = 0; x < x_size; ++x) {
        const Int y_ind =
            num_x_edges + x + y * x_size + z * x_size * (y_size - 1);
        const bool found = IndexExists(y_beg, y_end, y_ind);
        if (negate != found) {
          const Int j = x * box_size;
          const Int i_offset = y * box_size;
          for (Int i = i_offset; i <= i_offset + box_size; ++i) {
            catamari::PackPixel(i, j, width, active_pixel, &image);
          }
        }
      }
    }
  }

  catamari::WriteTIFF(filename, height, width, samples_per_pixel, image);
}

// Prints out an (x, z) slice of a Z^3 spanning tree.
inline void WriteGridXZSliceToTIFF(const std::string& filename, Int x_size,
                                   Int y_size, Int z_size, Int y,
                                   const std::vector<Int>& sample,
                                   const Int box_size,
                                   const catamari::Pixel& background_pixel,
                                   const catamari::Pixel& active_pixel,
                                   bool negate) {
  const Int num_x_edges = (x_size - 1) * y_size * z_size;
  const Int num_y_edges = x_size * (y_size - 1) * z_size;

  const auto x_beg = sample.begin();
  const auto x_end =
      std::lower_bound(sample.begin(), sample.end(), num_x_edges);
  const auto y_beg = x_end;
  const auto y_end =
      std::lower_bound(y_beg, sample.end(), num_x_edges + num_y_edges);
  const auto z_beg = y_end;
  const auto z_end = sample.end();

  const Int height = box_size * (z_size - 1) + 1;
  const Int width = box_size * (x_size - 1) + 1;
  const Int samples_per_pixel = 3;
  std::vector<char> image(width * height * samples_per_pixel);

  // Fill the background pixels.
  for (Int i = 0; i < height; ++i) {
    for (Int j = 0; j < width; ++j) {
      catamari::PackPixel(i, j, width, background_pixel, &image);
    }
  }

  for (Int z = 0; z < z_size; ++z) {
    // Write any horizontal connections in row 'y' and depth 'z'.
    for (Int x = 0; x < x_size - 1; ++x) {
      const Int x_ind = x + y * (x_size - 1) + z * (x_size - 1) * y_size;
      const bool found = IndexExists(x_beg, x_end, x_ind);
      if (negate != found) {
        const Int i = z * box_size;
        const Int j_offset = x * box_size;
        for (Int j = j_offset; j <= j_offset + box_size; ++j) {
          catamari::PackPixel(i, j, width, active_pixel, &image);
        }
      }
    }

    if (z < z_size - 1) {
      // Write any depth edges from this location.
      for (Int x = 0; x < x_size; ++x) {
        const Int z_ind =
            num_x_edges + num_y_edges + x + y * x_size + z * x_size * y_size;
        const bool found = IndexExists(z_beg, z_end, z_ind);
        if (negate != found) {
          const Int j = x * box_size;
          const Int i_offset = z * box_size;
          for (Int i = i_offset; i <= i_offset + box_size; ++i) {
            catamari::PackPixel(i, j, width, active_pixel, &image);
          }
        }
      }
    }
  }

  catamari::WriteTIFF(filename, height, width, samples_per_pixel, image);
}

// Prints out an (y, z) slice of a Z^3 spanning tree.
inline void WriteGridYZSliceToTIFF(const std::string& filename, Int x_size,
                                   Int y_size, Int z_size, Int x,
                                   const std::vector<Int>& sample,
                                   const Int box_size,
                                   const catamari::Pixel& background_pixel,
                                   const catamari::Pixel& active_pixel,
                                   bool negate) {
  const Int num_x_edges = (x_size - 1) * y_size * z_size;
  const Int num_y_edges = x_size * (y_size - 1) * z_size;

  const auto x_end =
      std::lower_bound(sample.begin(), sample.end(), num_x_edges);
  const auto y_beg = x_end;
  const auto y_end =
      std::lower_bound(y_beg, sample.end(), num_x_edges + num_y_edges);
  const auto z_beg = y_end;
  const auto z_end = sample.end();

  const Int height = box_size * (z_size - 1) + 1;
  const Int width = box_size * (y_size - 1) + 1;
  const Int samples_per_pixel = 3;
  std::vector<char> image(width * height * samples_per_pixel);

  // Fill the background pixels.
  for (Int i = 0; i < height; ++i) {
    for (Int j = 0; j < width; ++j) {
      catamari::PackPixel(i, j, width, background_pixel, &image);
    }
  }

  for (Int z = 0; z < z_size; ++z) {
    // Write out the vertical edges.
    for (Int y = 0; y < y_size - 1; ++y) {
      const Int y_ind =
          num_x_edges + x + y * x_size + z * x_size * (y_size - 1);
      const bool found = IndexExists(y_beg, y_end, y_ind);
      if (negate != found) {
        const Int i = z * box_size;
        const Int j_offset = y * box_size;
        for (Int j = j_offset; j <= j_offset + box_size; ++j) {
          catamari::PackPixel(i, j, width, active_pixel, &image);
        }
      }
    }

    if (z < z_size - 1) {
      // Write any depth edges from this location.
      for (Int y = 0; y < y_size; ++y) {
        const Int z_ind =
            num_x_edges + num_y_edges + x + y * x_size + z * x_size * y_size;
        const bool found = IndexExists(z_beg, z_end, z_ind);
        if (negate != found) {
          const Int j = y * box_size;
          const Int i_offset = z * box_size;
          for (Int i = i_offset; i <= i_offset + box_size; ++i) {
            catamari::PackPixel(i, j, width, active_pixel, &image);
          }
        }
      }
    }
  }

  catamari::WriteTIFF(filename, height, width, samples_per_pixel, image);
}

// Writes out the Z^3 sample to TIFF.
inline void WriteGridSampleToTIFF(Int x_size, Int y_size, Int z_size, Int round,
                                  bool maximum_likelihood,
                                  const std::vector<Int>& sample, Int box_size,
                                  bool negate,
                                  const catamari::Pixel& background_pixel,
                                  const catamari::Pixel& active_pixel,
                                  const std::string& tag) {
  for (Int z = 0; z < z_size; ++z) {
    const std::string filename =
        "grid-xy-" + std::string(maximum_likelihood ? "ml-" : "") +
        std::to_string(z) + "-" + std::to_string(round) + "-" + tag + ".tif";
    WriteGridXYSliceToTIFF(filename, x_size, y_size, z_size, z, sample,
                           box_size, background_pixel, active_pixel, negate);
  }
  if (z_size > 1) {
    for (Int y = 0; y < y_size; ++y) {
      const std::string filename =
          "grid-xz-" + std::string(maximum_likelihood ? "ml-" : "") +
          std::to_string(y) + "-" + std::to_string(round) + "-" + tag + ".tif";
      WriteGridXZSliceToTIFF(filename, x_size, y_size, z_size, y, sample,
                             box_size, background_pixel, active_pixel, negate);
    }
    for (Int x = 0; x < x_size; ++x) {
      const std::string filename =
          "grid-yz-" + std::string(maximum_likelihood ? "ml-" : "") +
          std::to_string(x) + "-" + std::to_string(round) + "-" + tag + ".tif";
      WriteGridYZSliceToTIFF(filename, x_size, y_size, z_size, x, sample,
                             box_size, background_pixel, active_pixel, negate);
    }
  }
}

// Prints out an (x, y) slice of a Z^3 spanning tree.
inline void WriteHexagonalToTIFF(const std::string& filename, Int x_size,
                                 Int y_size, const std::vector<Int>& sample,
                                 const Int cell_size,
                                 const catamari::Pixel& background_pixel,
                                 const catamari::Pixel& active_pixel,
                                 bool negate) {
  const Int num_horz_edges =
      // Count the main set of horizontal connections for the grid of rings.
      (2 * x_size - 1) * y_size +
      // Count the caps for the x_size x y_size grid of rings.
      x_size;

  const auto horz_beg = sample.begin();
  const auto horz_end =
      std::lower_bound(sample.begin(), sample.end(), num_horz_edges);
  const auto diag_beg = horz_end;
  const auto diag_end = sample.end();

  const Int vert_box_size = 2 * cell_size;
  const Int horz_box_size = 4 * cell_size;

  const Int height = vert_box_size * y_size + 1;
  const Int width = horz_box_size * x_size + 1;
  const Int samples_per_pixel = 3;
  std::vector<char> image(width * height * samples_per_pixel);

  // Fill the background pixels.
  for (Int i = 0; i < height; ++i) {
    for (Int j = 0; j < width; ++j) {
      catamari::PackPixel(i, j, width, background_pixel, &image);
    }
  }

  for (Int x = 0; x < x_size; ++x) {
    for (Int y = 0; y < y_size; ++y) {
      /*
      Given a horizontal edge offset of h and a diagonal edge offset of d,
      the edge numbers are of the form:

                 x     x
                /       \
              d+2       d+3
              /           \
             x             x -h+1- x
              \           /
               d        d+1
                \       /
                 x -h- x

      */
      const Int horz_offset = 2 * x + (2 * x_size - 1) * y;
      const Int diag_offset = num_horz_edges + 4 * (x + y * x_size);

      // Handle the 'h' edge if it exists.
      {
        const Int index = horz_offset;
        const bool found = IndexExists(horz_beg, horz_end, index);
        if (negate != found) {
          const Int j_offset = x * horz_box_size + cell_size;
          const Int i = y * vert_box_size;
          for (Int j = j_offset; j <= j_offset + cell_size; ++j) {
            catamari::PackPixel(i, j, width, active_pixel, &image);
          }
        }
      }

      // Handle the 'h+1' edge if it exists.
      if (x != x_size - 1) {
        const Int index = horz_offset + 1;
        const bool found = IndexExists(horz_beg, horz_end, index);
        if (negate != found) {
          const Int j_offset = x * horz_box_size + 3 * cell_size;
          const Int i = y * vert_box_size + cell_size;
          for (Int j = j_offset; j <= j_offset + cell_size; ++j) {
            catamari::PackPixel(i, j, width, active_pixel, &image);
          }
        }
      }

      // Handle the 'd' edge if it exists.
      {
        const Int index = diag_offset;
        const bool found = IndexExists(diag_beg, diag_end, index);
        if (negate != found) {
          const Int j_offset = x * horz_box_size;
          const Int i_offset = y * vert_box_size + cell_size;
          for (Int k = 0; k <= cell_size; ++k) {
            const Int i = i_offset - k;
            const Int j = j_offset + k;
            catamari::PackPixel(i, j, width, active_pixel, &image);
          }
        }
      }

      // Handle the 'd+1' edge if it exists.
      {
        const Int index = diag_offset + 1;
        const bool found = IndexExists(diag_beg, diag_end, index);
        if (negate != found) {
          const Int j_offset = x * horz_box_size + 2 * cell_size;
          const Int i_offset = y * vert_box_size;
          for (Int k = 0; k <= cell_size; ++k) {
            const Int i = i_offset + k;
            const Int j = j_offset + k;
            catamari::PackPixel(i, j, width, active_pixel, &image);
          }
        }
      }

      // Handle the 'd+2' edge if it exists.
      {
        const Int index = diag_offset + 2;
        const bool found = IndexExists(diag_beg, diag_end, index);
        if (negate != found) {
          const Int j_offset = x * horz_box_size;
          const Int i_offset = y * vert_box_size + cell_size;
          for (Int k = 0; k <= cell_size; ++k) {
            const Int i = i_offset + k;
            const Int j = j_offset + k;
            catamari::PackPixel(i, j, width, active_pixel, &image);
          }
        }
      }

      // Handle the 'd+3' edge if it exists.
      {
        const Int index = diag_offset + 3;
        const bool found = IndexExists(diag_beg, diag_end, index);
        if (negate != found) {
          const Int j_offset = x * horz_box_size + 2 * cell_size;
          const Int i_offset = (y + 1) * vert_box_size;
          for (Int k = 0; k <= cell_size; ++k) {
            const Int i = i_offset - k;
            const Int j = j_offset + k;
            catamari::PackPixel(i, j, width, active_pixel, &image);
          }
        }
      }
    }

    // Fill the cap if it exists.
    {
      const Int index = (2 * x_size - 1) * y_size + x;
      const bool found = IndexExists(horz_beg, horz_end, index);
      if (negate != found) {
        const Int j_offset = x * horz_box_size + cell_size;
        const Int i = y_size * vert_box_size;
        for (Int k = 0; k <= cell_size; ++k) {
          const Int j = j_offset + k;
          catamari::PackPixel(i, j, width, active_pixel, &image);
        }
      }
    }
  }

  catamari::WriteTIFF(filename, height, width, samples_per_pixel, image);
}
#endif  // ifdef CATAMARI_HAVE_LIBTIFF

// Prints the Z^2 spanning tree to std::cout.
inline void AsciiDisplaySample(Int x_size, Int y_size,
                               const std::vector<Int>& sample,
                               char missing_char, char horizontal_sampled_char,
                               char vertical_sampled_char, bool negate) {
  const Int num_horizontal_edges = (x_size - 1) * y_size;

  const auto horizontal_beg = sample.begin();
  const auto horizontal_end =
      std::lower_bound(sample.begin(), sample.end(), num_horizontal_edges);
  const auto vertical_beg = horizontal_end;
  const auto vertical_end = sample.end();

  for (Int y = 0; y < y_size; ++y) {
    // Print the horizontal edges which exist.
    for (Int x = 0; x < x_size - 1; ++x) {
      const Int horizontal_ind = x + y * (x_size - 1);
      const bool found =
          IndexExists(horizontal_beg, horizontal_end, horizontal_ind);

      std::cout << missing_char;
      if (negate != found) {
        std::cout << horizontal_sampled_char;
      } else {
        std::cout << missing_char;
      }
    }
    std::cout << missing_char << "\n";

    if (y < y_size - 1) {
      // Print the vertical edges which exist.
      for (Int x = 0; x < x_size; ++x) {
        const Int vertical_ind = num_horizontal_edges + x + y * x_size;
        const bool found =
            IndexExists(vertical_beg, vertical_end, vertical_ind);

        if (negate != found) {
          std::cout << vertical_sampled_char;
        } else {
          std::cout << missing_char;
        }
        if (x != x_size - 1) {
          std::cout << missing_char;
        }
      }
      std::cout << "\n";
    }
  }
  std::cout << std::endl;
}

// Overwrites an overcomplete basis for the star space with a minimal orthogonal
// basis. (This is the single-precision variant.)
void OverwriteWithOrthogonalBasis(BlasMatrix<float>* matrix) {
  const BlasInt height = matrix->Height();
  const BlasInt width = matrix->Width();
  const BlasInt min_dim = std::min(height, width);
  const BlasInt leading_dim = matrix->LeadingDimension();

  const BlasInt block_size = 64;
  const BlasInt work_size = std::max(block_size, BlasInt(3)) * width;
  quotient::Buffer<BlasInt> column_pivots(width, 0.);
  quotient::Buffer<float> reflector_scalars(min_dim);
  quotient::Buffer<float> work(work_size);
  BlasInt info;
  std::cout << "Calling sgeqpf..." << std::endl;
  LAPACK_SYMBOL(sgeqpf)
  (&height, &width, matrix->Data(), &leading_dim, column_pivots.Data(),
   reflector_scalars.Data(), work.Data(), &info);
  if (info != 0) {
    std::cerr << "sgeqpf info: " << info << std::endl;
  }

  // Compute the rank.
  const float min_diag_abs = 10 * min_dim * 1e-7;
  BlasInt rank = 0;
  for (Int j = 0; j < min_dim; ++j) {
    const float diag_abs = std::abs(matrix->Entry(j, j));
    if (diag_abs < min_diag_abs) {
      std::cout << "Dropping " << diag_abs << std::endl;
      break;
    }
    ++rank;
  }
  std::cout << "min_kept: " << std::abs(matrix->Entry(rank - 1, rank - 1))
            << std::endl;
  std::cout << "height: " << height << ", width: " << width
            << ", rank: " << rank << std::endl;
  matrix->Resize(height, rank);

  LAPACK_SYMBOL(sorgqr)
  (&height, &rank, &rank, matrix->Data(), &leading_dim,
   reflector_scalars.Data(), work.Data(), &work_size, &info);
  if (info != 0) {
    std::cerr << "sorgqr info: " << info << std::endl;
  }
}

// Overwrites an overcomplete basis for the star space with a minimal orthogonal
// basis. (This is the double-precision variant.)
void OverwriteWithOrthogonalBasis(BlasMatrix<double>* matrix) {
  const BlasInt height = matrix->Height();
  const BlasInt width = matrix->Width();
  const BlasInt min_dim = std::min(height, width);
  const BlasInt leading_dim = matrix->LeadingDimension();

  const BlasInt block_size = 64;
  const BlasInt work_size = std::max(block_size, BlasInt(3)) * width;
  quotient::Buffer<BlasInt> column_pivots(width, 0.);
  quotient::Buffer<double> reflector_scalars(min_dim);
  quotient::Buffer<double> work(work_size);
  BlasInt info;
  LAPACK_SYMBOL(dgeqpf)
  (&height, &width, matrix->Data(), &leading_dim, column_pivots.Data(),
   reflector_scalars.Data(), work.Data(), &info);
  if (info != 0) {
    std::cerr << "dgeqpf info: " << info << std::endl;
  }

  // Compute the rank.
  const double min_diag_abs = 10 * min_dim * 1e-15;
  BlasInt rank = 0;
  for (Int j = 0; j < min_dim; ++j) {
    const double diag_abs = std::abs(matrix->Entry(j, j));
    if (diag_abs < min_diag_abs) {
      std::cout << "Dropping " << diag_abs << std::endl;
      break;
    }
    ++rank;
  }
  std::cout << "min_kept: " << std::abs(matrix->Entry(rank - 1, rank - 1))
            << std::endl;
  std::cout << "height: " << height << ", width: " << width
            << ", rank: " << rank << std::endl;
  matrix->Resize(height, rank);

  LAPACK_SYMBOL(dorgqr)
  (&height, &rank, &rank, matrix->Data(), &leading_dim,
   reflector_scalars.Data(), work.Data(), &work_size, &info);
  if (info != 0) {
    std::cerr << "dorgqr info: " << info << std::endl;
  }
}

// We order the subset of Z^3 x edges first, then the y edges, then the z edges,
// each lexicographically by its coordinates. The orientations are
// lexicographic as well.
template <typename Field>
void GridStarSpaceBasisGramian(Int x_size, Int y_size, Int z_size,
                               BlasMatrix<Field>* matrix) {
  const Int num_vertices = x_size * y_size * z_size;
  const Int num_x_edges = (x_size - 1) * y_size * z_size;
  const Int num_y_edges = x_size * (y_size - 1) * z_size;
  const Int num_z_edges = x_size * y_size * (z_size - 1);
  const Int num_edges = num_x_edges + num_y_edges + num_z_edges;

  // Build an (overcomplete) basis for the star space.
  BlasMatrix<Field> star_space_basis;
  star_space_basis.Resize(num_edges, num_vertices, Field{0});
  for (Int x = 0; x < x_size; ++x) {
    for (Int y = 0; y < y_size; ++y) {
      for (Int z = 0; z < z_size; ++z) {
        const Int vertex = x + y * x_size + z * x_size * y_size;

        const Int x_ind_y_stride = x_size - 1;
        const Int x_ind_z_stride = x_ind_y_stride * y_size;
        const Int x_ind = x + y * x_ind_y_stride + z * x_ind_z_stride;
        if (x > 0) {
          // Receive from the edge to our left.
          const Int left_ind = x_ind - 1;
          star_space_basis(left_ind, vertex) = 1;
        }
        if (x < x_size - 1) {
          // This edge pushes to the right.
          star_space_basis(x_ind, vertex) = -1;
        }

        const Int y_ind_y_stride = x_size;
        const Int y_ind_z_stride = y_ind_y_stride * (y_size - 1);
        const Int y_ind =
            num_x_edges + x + y * y_ind_y_stride + z * y_ind_z_stride;
        if (y > 0) {
          // Receive from the edge below us.
          const Int down_ind = y_ind - y_ind_y_stride;
          star_space_basis(down_ind, vertex) = 1;
        }
        if (y < y_size - 1) {
          // This edge pushes upward.
          star_space_basis(y_ind, vertex) = -1;
        }

        const Int z_ind_y_stride = x_size;
        const Int z_ind_z_stride = z_ind_y_stride * y_size;
        const Int z_ind = num_x_edges + num_y_edges + x + y * z_ind_y_stride +
                          z * z_ind_z_stride;
        if (z > 0) {
          // Receive from the edge out of the plane.
          const Int out_ind = z_ind - z_ind_z_stride;
          star_space_basis(out_ind, vertex) = 1;
        }
        if (z < z_size - 1) {
          // This edge pushes into the plane.
          star_space_basis(z_ind, vertex) = -1;
        }
      }
    }
  }

  // Build an orthogonal basis for the star space via a QR factorization.
  BlasMatrix<Field> star_space_orthog = star_space_basis;
  OverwriteWithOrthogonalBasis(&star_space_orthog);

  matrix->Resize(num_edges, num_edges, Field{0});
  catamari::LowerNormalHermitianOuterProduct(
      Field{1}, star_space_orthog.ConstView(), Field{1}, &matrix->view);
}

/*
  We order the hexagonal edges in the following form, which corresponds to an
  x_size of 3 and a y_size of 2:

    24 - 25       26 - 27       28 - 29
    /     \       /      \     /       \
   14      15 - 18       19 - 22       23
    \      /      \      /     \       /
    12 - 13       16 - 17       20 - 21
    /      \     /       \     /       \
   2        3 - 6         7 - 10       11
    \      /     \       /     \       /
     0 - 1         4 - 5         8 - 9

The horizontal edges are ordered first, then the diagonal edges.

*/
template <typename Field>
void HexagonalStarSpaceBasisGramian(Int x_size, Int y_size,
                                    BlasMatrix<Field>* matrix) {
  const Int num_vertices =
      // Count the bottom four vertices in the x_size x y_size grid of rings.
      4 * x_size * y_size +
      // Count the caps for the x_size x y_size grid of rings.
      2 * x_size;

  const Int num_horz_edges =
      // Count the main set of horizontal connections for the grid of rings.
      (2 * x_size - 1) * y_size +
      // Count the caps for the x_size x y_size grid of rings.
      x_size;

  const Int num_diag_edges = 4 * x_size * y_size;

  const Int num_edges = num_horz_edges + num_diag_edges;

  // Build an (overcomplete) basis for the star space.
  BlasMatrix<Field> star_space_basis;
  star_space_basis.Resize(num_edges, num_vertices, Field{0});
  for (Int x = 0; x < x_size; ++x) {
    for (Int y = 0; y < y_size; ++y) {
      const Int vertex_offset = 4 * (x + y * x_size);
      const Int up_vertex_offset = y == y_size - 1 ? 4 * x_size * y_size + 2 * x
                                                   : vertex_offset + 4 * x_size;

      /*
      For a vertex offset of 'v', and for w = v + 4 * x_size, we handle the
      following snippet
      (where the v+3 to v+6 edge may not exist):

            w    w+1
           /       \
          v+2      v+3 - v+6
           \       /
            v - v+1

      Given a horizontal edge offset of h and a diagonal edge offset of d,
      the edge numbers are of the form:

                 w      w+1
                /         \
              d+2         d+3
              /             \
            v+2             v+3 -h+1- v+6
              \             /
               d          d+1
                \         /
                 v  -h- v+1

      and the orientations of the edges are:

        h   = (v,   v+1),
        h+1 = (v+3, v+6),
        d   = (v,   v+2),
        d+1 = (v+1, v+3),
        d+2 = (v+2, w  ),
        d+3 = (v+3, w+1).

      */

      const Int horz_offset = 2 * x + (2 * x_size - 1) * y;
      const Int diag_offset = num_horz_edges + 4 * (x + y * x_size);

      // Handle horizontal edge h = (v, v+1).
      star_space_basis(horz_offset, vertex_offset) = -1;
      star_space_basis(horz_offset, vertex_offset + 1) = 1;

      if (x < x_size - 1) {
        // Handle horizontal edge h+1 = (v+3, v+6).
        star_space_basis(horz_offset + 1, vertex_offset + 3) = -1;
        star_space_basis(horz_offset + 1, vertex_offset + 6) = 1;
      }

      // Handle edge d = (v, v+2).
      star_space_basis(diag_offset, vertex_offset) = -1;
      star_space_basis(diag_offset, vertex_offset + 2) = 1;

      // Handle edge d+1 = (v+1, v+3).
      star_space_basis(diag_offset + 1, vertex_offset + 1) = -1;
      star_space_basis(diag_offset + 1, vertex_offset + 3) = 1;

      // Handle edge d+2 = (v+2, w).
      star_space_basis(diag_offset + 2, vertex_offset + 2) = -1;
      star_space_basis(diag_offset + 2, up_vertex_offset) = 1;

      // Handle edge d+3 = (v+3, w+1).
      star_space_basis(diag_offset + 3, vertex_offset + 3) = -1;
      star_space_basis(diag_offset + 3, up_vertex_offset + 1) = 1;
    }

    // We place a horizontal cap on the grid at (w, w+1).
    const Int vertex_offset = 4 * x_size * y_size + 2 * x;
    const Int cap_offset = (2 * x_size - 1) * y_size + x;
    star_space_basis(cap_offset, vertex_offset) = -1;
    star_space_basis(cap_offset, vertex_offset + 1) = 1;
  }

  // Build an orthogonal basis for the star space via a QR factorization.
  BlasMatrix<Field> star_space_orthog = star_space_basis;
  OverwriteWithOrthogonalBasis(&star_space_orthog);

  matrix->Resize(num_edges, num_edges, Field{0});
  catamari::LowerNormalHermitianOuterProduct(
      Field{1}, star_space_orthog.ConstView(), Field{1}, &matrix->view);
}

template <typename Field>
std::vector<Int> SampleDPP(Int block_size, bool maximum_likelihood,
                           BlasMatrixView<Field>* matrix,
                           std::mt19937* generator) {
  const bool is_complex = catamari::IsComplex<Field>::value;
  const Int matrix_size = matrix->height;
  quotient::Timer timer;
  timer.Start();

  const std::vector<Int> sample = catamari::SampleLowerHermitianDPP(
      block_size, maximum_likelihood, matrix, generator);

  const double runtime = timer.Stop();
  const double flops =
      (is_complex ? 4 : 1) * std::pow(1. * matrix_size, 3.) / 3.;
  const double gflops_per_sec = flops / (1.e9 * runtime);
  std::cout << "Sequential DPP time: " << runtime << " seconds." << std::endl;
  std::cout << "Sequential DPP GFlop/s: " << gflops_per_sec << std::endl;

  const ComplexBase<Field> log_likelihood =
      catamari::DPPLogLikelihood(matrix->ToConst());
  std::cout << "Sequential DPP log-likelihood: " << log_likelihood << std::endl;

  return sample;
}

#ifdef CATAMARI_OPENMP
template <typename Field>
std::vector<Int> OpenMPSampleDPP(Int tile_size, Int block_size,
                                 bool maximum_likelihood,
                                 BlasMatrixView<Field>* matrix,
                                 std::mt19937* generator,
                                 Buffer<Field>* extra_buffer) {
  const bool is_complex = catamari::IsComplex<Field>::value;
  const Int matrix_size = matrix->height;
  quotient::Timer timer;
  timer.Start();

  const int old_max_threads = catamari::GetMaxBlasThreads();
  catamari::SetNumBlasThreads(1);

  std::vector<Int> sample;
  #pragma omp parallel
  #pragma omp single
  sample = catamari::OpenMPSampleLowerHermitianDPP(tile_size, block_size,
                                                   maximum_likelihood, matrix,
                                                   generator, extra_buffer);

  catamari::SetNumBlasThreads(old_max_threads);

  const double runtime = timer.Stop();
  const double flops =
      (is_complex ? 4 : 1) * std::pow(1. * matrix_size, 3.) / 3.;
  const double gflops_per_sec = flops / (1.e9 * runtime);
  std::cout << "OpenMP DPP time: " << runtime << " seconds." << std::endl;
  std::cout << "OpenMP DPP GFlop/s: " << gflops_per_sec << std::endl;

  const ComplexBase<Field> log_likelihood =
      catamari::DPPLogLikelihood(matrix->ToConst());
  std::cout << "OpenMP DPP log-likelihood: " << log_likelihood << std::endl;

  return sample;
}
#endif  // ifdef CATAMARI_OPENMP

template <typename Field>
void RunGridDPPTests(bool maximum_likelihood, Int x_size, Int y_size,
                     Int z_size, Int block_size, Int CATAMARI_UNUSED tile_size,
                     Int num_rounds, unsigned int random_seed,
                     bool ascii_display, bool CATAMARI_UNUSED write_tiff,
                     bool negate) {
  // ASCII display configuration.
  const char missing_char = ' ';
  const char horizontal_sampled_char = '-';
  const char vertical_sampled_char = '|';

#ifdef CATAMARI_HAVE_LIBTIFF
  // TIFF configuration.
  const catamari::Pixel background_pixel{char(255), char(255), char(255)};
  const catamari::Pixel active_pixel{char(255), char(0), char(0)};
  const Int box_size = 10;
#endif  // ifdef CATAMARI_HAVE_LIBTIFF

  BlasMatrix<Field> matrix;
  GridStarSpaceBasisGramian(x_size, y_size, z_size, &matrix);

  // A uniform spanning tree should have the initial edge cover two vertices and
  // each additional edge should touch one new vertex.
  const Int expected_sample_size = x_size * y_size * z_size - 1;
  std::cout << "Expected rank: " << expected_sample_size << std::endl;
  std::cout << "Ground set size: " << matrix.Height() << std::endl;

  std::mt19937 generator(random_seed);
  BlasMatrix<Field> matrix_copy;
  Buffer<Field> extra_buffer;
  for (Int round = 0; round < num_rounds; ++round) {
#ifdef CATAMARI_OPENMP
    // Sample using the OpenMP DPP sampler.
    matrix_copy = matrix;
    extra_buffer.Resize(matrix.Height() * matrix.Height());
    const std::vector<Int> omp_sample =
        ::OpenMPSampleDPP(tile_size, block_size, maximum_likelihood,
                          &matrix_copy.view, &generator, &extra_buffer);
    if (Int(omp_sample.size()) != expected_sample_size) {
      std::cerr << "ERROR: Sampled " << omp_sample.size() << " instead of "
                << expected_sample_size << " edges." << std::endl;
    }
#ifdef CATAMARI_HAVE_LIBTIFF
    if (write_tiff) {
      const std::string tag = std::to_string(x_size) + std::string("-") +
                              std::to_string(y_size) + std::string("-") +
                              std::to_string(z_size) + std::string("-omp-") +
                              typeid(Field).name();
      WriteGridSampleToTIFF(x_size, y_size, z_size, round, maximum_likelihood,
                            omp_sample, box_size, negate, background_pixel,
                            active_pixel, tag);
    }
#endif  // ifdef CATAMARI_HAVE_LIBTIFF
    if (z_size == 1 && ascii_display) {
      AsciiDisplaySample(x_size, y_size, omp_sample, missing_char,
                         horizontal_sampled_char, vertical_sampled_char,
                         negate);
    }
#endif  // ifdef CATAMARI_OPENMP

    // Sample using the sequential DPP sampler.
    matrix_copy = matrix;
    std::vector<Int> sample = ::SampleDPP(block_size, maximum_likelihood,
                                          &matrix_copy.view, &generator);
    if (Int(sample.size()) != expected_sample_size) {
      std::cerr << "ERROR: Sampled " << sample.size() << " instead of "
                << expected_sample_size << " edges." << std::endl;
    }
#ifdef CATAMARI_HAVE_LIBTIFF
    if (write_tiff) {
      const std::string tag = std::to_string(x_size) + std::string("-") +
                              std::to_string(y_size) + std::string("-") +
                              std::to_string(z_size) + std::string("-") +
                              typeid(Field).name();
      WriteGridSampleToTIFF(x_size, y_size, z_size, round, maximum_likelihood,
                            sample, box_size, negate, background_pixel,
                            active_pixel, tag);
    }
#endif  // ifdef CATAMARI_HAVE_LIBTIFF
    if (z_size == 1 && ascii_display) {
      AsciiDisplaySample(x_size, y_size, sample, missing_char,
                         horizontal_sampled_char, vertical_sampled_char,
                         negate);
    }
  }
}

template <typename Field>
void RunHexagonalDPPTests(bool maximum_likelihood, Int x_size, Int y_size,
                          Int block_size, Int CATAMARI_UNUSED tile_size,
                          Int num_rounds, unsigned int random_seed,
                          bool CATAMARI_UNUSED write_tiff, bool negate) {
#ifdef CATAMARI_HAVE_LIBTIFF
  // TIFF configuration.
  const catamari::Pixel background_pixel{char(255), char(255), char(255)};
  const catamari::Pixel active_pixel{char(255), char(0), char(0)};
  const Int cell_size = 5;
#endif  // ifdef CATAMARI_HAVE_LIBTIFF

  BlasMatrix<Field> matrix;
  HexagonalStarSpaceBasisGramian(x_size, y_size, &matrix);

  std::mt19937 generator(random_seed);
  BlasMatrix<Field> matrix_copy;
  Buffer<Field> extra_buffer;
  for (Int round = 0; round < num_rounds; ++round) {
#ifdef CATAMARI_OPENMP
    // Sample using the OpenMP sampler.
    matrix_copy = matrix;
    extra_buffer.Resize(matrix.Height() * matrix.Height());
    const std::vector<Int> omp_sample =
        ::OpenMPSampleDPP(tile_size, block_size, maximum_likelihood,
                          &matrix_copy.view, &generator, &extra_buffer);
#ifdef CATAMARI_HAVE_LIBTIFF
    if (write_tiff) {
      const std::string tag = std::string("omp-") + typeid(Field).name();
      const std::string filename =
          "hexagonal-" + std::string(maximum_likelihood ? "ml-" : "") +
          std::to_string(round) + "-" + tag + ".tif";
      WriteHexagonalToTIFF(filename, x_size, y_size, omp_sample, cell_size,
                           background_pixel, active_pixel, negate);
    }
#endif  // ifdef CATAMARI_HAVE_LIBTIFF
#endif  // ifdef CATAMARI_OPENMP

    // Sample using the sequential sampler.
    matrix_copy = matrix;
    const std::vector<Int> sample = ::SampleDPP(block_size, maximum_likelihood,
                                                &matrix_copy.view, &generator);
#ifdef CATAMARI_HAVE_LIBTIFF
    if (write_tiff) {
      const std::string tag = typeid(Field).name();
      const std::string filename =
          "hexagonal-" + std::string(maximum_likelihood ? "ml-" : "") +
          std::to_string(round) + "-" + tag + ".tif";
      WriteHexagonalToTIFF(filename, x_size, y_size, sample, cell_size,
                           background_pixel, active_pixel, negate);
    }
#endif  // ifdef CATAMARI_HAVE_LIBTIFF
  }
}

}  // anonymous namespace

int main(int argc, char** argv) {
  specify::ArgumentParser parser(argc, argv);
  const bool negate =
      parser.OptionalInput<bool>("negate", "Negate the DPP sample?", false);
  const bool hexagonal = parser.OptionalInput<bool>(
      "hexagonal", "Use a hexagonal 2D tiling?", false);
  const Int x_size =
      parser.OptionalInput<Int>("x_size", "The x dimension of the graph.", 50);
  const Int y_size =
      parser.OptionalInput<Int>("y_size", "The y dimension of the graph.", 50);
  const Int z_size = parser.OptionalInput<Int>(
      "z_size",
      "The z dimension of the graph. If it is equal to one, the graph is "
      "treated as 2D.",
      1);
  const Int tile_size = parser.OptionalInput<Int>(
      "tile_size", "The tile size for multithreaded factorization.", 128);
  const Int block_size = parser.OptionalInput<Int>(
      "block_size", "The block_size for dense factorization.", 64);
  const Int num_rounds = parser.OptionalInput<Int>(
      "num_rounds", "The number of rounds of factorizations.", 2);
  const unsigned int random_seed = parser.OptionalInput<unsigned int>(
      "random_seed", "The random seed for the DPP.", 17u);
  const bool maximum_likelihood = parser.OptionalInput<bool>(
      "maximum_likelihood", "Take a maximum likelihood DPP sample?", false);
  const bool ascii_display = parser.OptionalInput<bool>(
      "ascii_display", "Display the results in ASCII?", false);
  const bool write_tiff = parser.OptionalInput<bool>(
      "write_tiff", "Write out the results into a TIFF file?", true);
  if (!parser.OK()) {
    return 0;
  }
  if (hexagonal && z_size != 1) {
    std::cerr << "Hexagonal tilings are only supported for 2D grids."
              << std::endl;
    return 0;
  }

  if (hexagonal) {
    std::cout << "Single-precision hexagonal:" << std::endl;
    RunHexagonalDPPTests<float>(maximum_likelihood, x_size, y_size, block_size,
                                tile_size, num_rounds, random_seed, write_tiff,
                                negate);
    std::cout << std::endl;

    std::cout << "Double-precision hexagonal:" << std::endl;
    RunHexagonalDPPTests<double>(maximum_likelihood, x_size, y_size, block_size,
                                 tile_size, num_rounds, random_seed, write_tiff,
                                 negate);
  } else {
    std::cout << "Single-precision grid:" << std::endl;
    RunGridDPPTests<float>(maximum_likelihood, x_size, y_size, z_size,
                           block_size, tile_size, num_rounds, random_seed,
                           ascii_display, write_tiff, negate);
    std::cout << std::endl;

    std::cout << "Double-precision grid:" << std::endl;
    RunGridDPPTests<double>(maximum_likelihood, x_size, y_size, z_size,
                            block_size, tile_size, num_rounds, random_seed,
                            ascii_display, write_tiff, negate);
  }

  return 0;
}
#else
int main(int CATAMARI_UNUSED argc, char* CATAMARI_UNUSED argv[]) { return 0; }
#endif  // ifdef CATAMARI_HAVE_LAPACK
