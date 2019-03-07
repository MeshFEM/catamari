/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// Generates a (maximum-likelihood) sample from a Determinantal Point Process
// which generates a uniform sample from the set of spanning trees of a 2D
// grid graph.
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
#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

#include "catamari/blas_matrix.hpp"
#include "catamari/dense_factorizations.hpp"
#include "catamari/ldl.hpp"
#include "quotient/timer.hpp"
#include "specify.hpp"

#ifdef CATAMARI_HAVE_LIBTIFF
#include <tiffio.h>
#endif  // ifdef CATAMARI_HAVE_LIBTIFF

using catamari::BlasMatrix;
using catamari::BlasMatrixView;
using catamari::Complex;
using catamari::ComplexBase;
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
void WriteTIFF(const std::string& filename, std::size_t height,
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

// A representation of a TIFF pixel (without alpha).
struct Pixel {
  char red;
  char green;
  char blue;
};

// Prints the 2D spanning tree.
inline void WriteSampleToTIFF(const std::string& filename, Int x_size,
                              Int y_size, const std::vector<Int>& sample,
                              const Int box_size, const Pixel& background_pixel,
                              const Pixel& active_pixel) {
  const Int num_horizontal_edges = (x_size - 1) * y_size;

  const auto horizontal_beg = sample.begin();
  const auto horizontal_end =
      std::lower_bound(sample.begin(), sample.end(), num_horizontal_edges);
  const auto vertical_beg = horizontal_end;
  const auto vertical_end = sample.end();

  const Int height = box_size * (y_size - 1) + 1;
  const Int width = box_size * (x_size - 1) + 1;
  const Int samples_per_pixel = 3;
  std::vector<char> image(width * width * samples_per_pixel);

  // Fill the background pixels.
  for (Int i = 0; i < height; ++i) {
    for (Int j = 0; j < width; ++j) {
      Int offset = samples_per_pixel * (j + i * width);
      image[offset++] = background_pixel.red;
      image[offset++] = background_pixel.green;
      image[offset++] = background_pixel.blue;
    }
  }

  for (Int y = 0; y < y_size; ++y) {
    // Write any horizontal connections in row 'y'.
    for (Int x = 0; x < x_size - 1; ++x) {
      const Int horizontal_ind = x + y * (x_size - 1);
      auto iter =
          std::lower_bound(horizontal_beg, horizontal_end, horizontal_ind);
      const bool found = iter != horizontal_end && *iter == horizontal_ind;

      if (found) {
        const Int i = y * box_size;
        const Int j_offset = x * box_size;
        for (Int j = j_offset; j < j_offset + box_size; ++j) {
          Int offset = samples_per_pixel * (j + i * width);
          image[offset++] = active_pixel.red;
          image[offset++] = active_pixel.green;
          image[offset++] = active_pixel.blue;
        }
      }
    }

    if (y < y_size - 1) {
      // Write any vertical edges from this row.
      for (Int x = 0; x < x_size; ++x) {
        const Int vertical_ind = num_horizontal_edges + x + y * x_size;
        auto iter = std::lower_bound(vertical_beg, vertical_end, vertical_ind);
        const bool found = iter != vertical_end && *iter == vertical_ind;

        if (found) {
          const Int j = x * box_size;
          const Int i_offset = y * box_size;
          for (Int i = i_offset; i < i_offset + box_size; ++i) {
            Int offset = samples_per_pixel * (j + i * width);
            image[offset++] = active_pixel.red;
            image[offset++] = active_pixel.green;
            image[offset++] = active_pixel.blue;
          }
        }
      }
    }
  }

  WriteTIFF(filename, height, width, samples_per_pixel, image);
}
#endif  // ifdef CATAMARI_HAVE_LIBTIFF

// Prints the 2D spanning tree.
inline void AsciiDisplaySample(Int x_size, Int y_size,
                               const std::vector<Int>& sample,
                               char missing_char, char horizontal_sampled_char,
                               char vertical_sampled_char) {
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
      auto iter =
          std::lower_bound(horizontal_beg, horizontal_end, horizontal_ind);
      const bool found = iter != horizontal_end && *iter == horizontal_ind;

      std::cout << missing_char;
      if (found) {
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
        auto iter = std::lower_bound(vertical_beg, vertical_end, vertical_ind);
        const bool found = iter != vertical_end && *iter == vertical_ind;

        if (found) {
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

void OverwriteWithOrthogonalBasis(BlasMatrix<float>* matrix) {
  const BlasInt height = matrix->view.height;
  const BlasInt width = matrix->view.width;
  const BlasInt min_dim = std::min(height, width);
  const BlasInt leading_dim = matrix->view.leading_dim;

  const BlasInt block_size = 64;
  const BlasInt work_size = std::max(block_size, BlasInt(3)) * width;
  quotient::Buffer<BlasInt> column_pivots(width, 0.);
  quotient::Buffer<float> reflector_scalars(min_dim);
  quotient::Buffer<float> work(work_size);
  BlasInt info;
  std::cout << "Calling sgeqpf..." << std::endl;
  sgeqpf(&height, &width, matrix->view.data, &leading_dim, column_pivots.Data(),
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

  sorgqr(&height, &rank, &rank, matrix->view.data, &leading_dim,
         reflector_scalars.Data(), work.Data(), &work_size, &info);
  if (info != 0) {
    std::cerr << "sorgqr info: " << info << std::endl;
  }
}

void OverwriteWithOrthogonalBasis(BlasMatrix<double>* matrix) {
  const BlasInt height = matrix->view.height;
  const BlasInt width = matrix->view.width;
  const BlasInt min_dim = std::min(height, width);
  const BlasInt leading_dim = matrix->view.leading_dim;

  const BlasInt block_size = 64;
  const BlasInt work_size = std::max(block_size, BlasInt(3)) * width;
  quotient::Buffer<BlasInt> column_pivots(width, 0.);
  quotient::Buffer<double> reflector_scalars(min_dim);
  quotient::Buffer<double> work(work_size);
  BlasInt info;
  dgeqpf(&height, &width, matrix->view.data, &leading_dim, column_pivots.Data(),
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

  dorgqr(&height, &rank, &rank, matrix->view.data, &leading_dim,
         reflector_scalars.Data(), work.Data(), &work_size, &info);
  if (info != 0) {
    std::cerr << "dorgqr info: " << info << std::endl;
  }
}

template <typename Field>
void InitializeMatrix(Int x_size, Int y_size, BlasMatrix<Field>* matrix) {
  const Int num_vertices = x_size * y_size;

  // We order the edges horizontally first, then vertically, each
  // lexicographically by their (x, y) coordinates. The orientations are
  // lexicographic as well.
  const Int num_horizontal_edges = (x_size - 1) * y_size;
  const Int num_vertical_edges = x_size * (y_size - 1);
  const Int num_edges = num_horizontal_edges + num_vertical_edges;

  // Build an (overcomplete) basis for the star space.
  BlasMatrix<Field> star_space_basis;
  star_space_basis.Resize(num_edges, num_vertices, Field{0});
  for (Int x = 0; x < x_size; ++x) {
    for (Int y = 0; y < y_size; ++y) {
      const Int vertex = x + y * x_size;

      const Int horizontal_ind = x + y * (x_size - 1);
      if (x > 0) {
        // Receive from the edge to our left.
        const Int left_ind = horizontal_ind - 1;
        star_space_basis(left_ind, vertex) = 1;
      }
      if (x < x_size - 1) {
        // This edge pushes to the right.
        star_space_basis(horizontal_ind, vertex) = -1;
      }

      const Int vertical_ind = num_horizontal_edges + x + y * x_size;
      if (y > 0) {
        // Receive from the edge below us.
        const Int down_ind = vertical_ind - x_size;
        star_space_basis(down_ind, vertex) = 1;
      }
      if (y < y_size - 1) {
        // This edge pushes upward.
        star_space_basis(vertical_ind, vertex) = -1;
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

template <typename Field>
std::vector<Int> SampleDPP(Int block_size, bool maximum_likelihood,
                           BlasMatrixView<Field>* matrix,
                           std::mt19937* generator) {
  const bool is_complex = catamari::IsComplex<Field>::value;
  const Int matrix_size = matrix->height;
  quotient::Timer timer;
  timer.Start();

  const std::vector<Int> sample = LowerFactorAndSampleDPP(
      block_size, maximum_likelihood, matrix, generator);

  const double runtime = timer.Stop();
  const double flops =
      (is_complex ? 4 : 1) * std::pow(1. * matrix_size, 3.) / 3.;
  const double gflops_per_sec = flops / (1.e9 * runtime);
  std::cout << "Sequential DPP GFlop/s: " << gflops_per_sec << std::endl;

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
  sample =
      OpenMPLowerFactorAndSampleDPP(tile_size, block_size, maximum_likelihood,
                                    matrix, generator, extra_buffer);

  catamari::SetNumBlasThreads(old_max_threads);

  const double runtime = timer.Stop();
  const double flops =
      (is_complex ? 4 : 1) * std::pow(1. * matrix_size, 3.) / 3.;
  const double gflops_per_sec = flops / (1.e9 * runtime);
  std::cout << "OpenMP DPP GFlop/s: " << gflops_per_sec << std::endl;

  return sample;
}
#endif  // ifdef CATAMARI_OPENMP

template <typename Field>
void RunDPPTests(bool maximum_likelihood, Int x_size, Int y_size,
                 Int block_size, Int CATAMARI_UNUSED tile_size, Int num_rounds,
                 unsigned int random_seed, bool ascii_display,
                 bool CATAMARI_UNUSED write_tiff) {
  const Int matrix_size = x_size * y_size;
  BlasMatrix<Field> matrix;
  Buffer<Field> extra_buffer(matrix_size * matrix_size);

  // ASCII display configuration.
  const char missing_char = ' ';
  const char horizontal_sampled_char = '-';
  const char vertical_sampled_char = '|';

#ifdef CATAMARI_HAVE_LIBTIFF
  // TIFF configuration.
  const Pixel background_pixel{char(255), char(255), char(255)};
  const Pixel active_pixel{char(255), char(0), char(0)};
  const Int box_size = 10;
#endif  // ifdef CATAMARI_HAVE_LIBTIFF

  std::mt19937 generator(random_seed);
  for (Int round = 0; round < num_rounds; ++round) {
#ifdef CATAMARI_OPENMP
    InitializeMatrix(x_size, y_size, &matrix);
    const std::vector<Int> omp_sample =
        OpenMPSampleDPP(tile_size, block_size, maximum_likelihood, &matrix.view,
                        &generator, &extra_buffer);
#ifdef CATAMARI_HAVE_LIBTIFF
    if (write_tiff) {
      const std::string omp_filename = "sample-omp-" + std::to_string(round) +
                                       "-" + typeid(Field).name() + ".tif";
      WriteSampleToTIFF(omp_filename, x_size, y_size, omp_sample, box_size,
                        background_pixel, active_pixel);
    }
#endif  // ifdef CATAMARI_HAVE_LIBTIFF
    if (ascii_display) {
      AsciiDisplaySample(x_size, y_size, omp_sample, missing_char,
                         horizontal_sampled_char, vertical_sampled_char);
    }
#endif  // ifdef CATAMARI_OPENMP

    InitializeMatrix(x_size, y_size, &matrix);
    const std::vector<Int> sample =
        SampleDPP(block_size, maximum_likelihood, &matrix.view, &generator);
#ifdef CATAMARI_HAVE_LIBTIFF
    if (write_tiff) {
      const std::string filename = "sample-" + std::to_string(round) + "-" +
                                   typeid(Field).name() + ".tif";
      WriteSampleToTIFF(filename, x_size, y_size, sample, box_size,
                        background_pixel, active_pixel);
    }
#endif  // ifdef CATAMARI_HAVE_LIBTIFF
    if (ascii_display) {
      AsciiDisplaySample(x_size, y_size, sample, missing_char,
                         horizontal_sampled_char, vertical_sampled_char);
    }
  }
}

}  // anonymous namespace

int main(int argc, char** argv) {
  specify::ArgumentParser parser(argc, argv);
  const Int x_size = parser.OptionalInput<Int>(
      "x_size", "The x dimension of the 2D graph.", 50);
  const Int y_size = parser.OptionalInput<Int>(
      "y_size", "The y dimension of the 2D graph.", 50);
  const Int tile_size = parser.OptionalInput<Int>(
      "tile_size", "The tile size for multithreaded factorization.", 128);
  const Int block_size = parser.OptionalInput<Int>(
      "block_size", "The block_size for dense factorization.", 64);
  const Int num_rounds = parser.OptionalInput<Int>(
      "num_rounds", "The number of rounds of factorizations.", 2);
  const unsigned int random_seed = parser.OptionalInput<unsigned int>(
      "random_seed", "The random seed for the DPP.", 17u);
  const bool maximum_likelihood = parser.OptionalInput<bool>(
      "maximum_likelihood", "Take a maximum likelihood DPP sample?", true);
  const bool ascii_display = parser.OptionalInput<bool>(
      "ascii_display", "Display the results in ASCII?", true);
  const bool write_tiff = parser.OptionalInput<bool>(
      "write_tiff", "Write out the results into a TIFF file?", true);
  if (!parser.OK()) {
    return 0;
  }

  std::cout << "Single-precision:" << std::endl;
  RunDPPTests<float>(maximum_likelihood, x_size, y_size, block_size, tile_size,
                     num_rounds, random_seed, ascii_display, write_tiff);
  std::cout << std::endl;

  std::cout << "Double-precision:" << std::endl;
  RunDPPTests<double>(maximum_likelihood, x_size, y_size, block_size, tile_size,
                      num_rounds, random_seed, ascii_display, write_tiff);

  return 0;
}
