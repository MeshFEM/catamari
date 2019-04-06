/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// Samples a dense, nonsymmetric Determinantal Point Process defining a uniform
// distribution over domino tilings of the Aztec diamond. Please see the review
// paper:
//
//   Sunil Chhita, Kurt Johansson, and Benjamin Young,
//   "Asymptotic Domino Statistics in the Aztec Diamond",
//   Annals of Applied Probability, 25 (3), pp. 1232--1278, 2015.
//
// Note that there appears to be a mising negative sign on the down-left edge
// from 'b_0' in Fig. 4 of said publication.
//
// TODO(Jack Poulson): Make use of Figure 1 of:
//
//   Mark Adler, Sunil Chhita, Kurt Johansson, and Pierre van Moerbeke,
//   "Tacnode GUE-minor Processes and Double Aztec Diamonds", Mar. 21, 2013.
//   URL: https://arxiv.org/abs/1303.5279
//
// to extend support to the double Aztec diamond.
//
#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

#include "catamari/blas_matrix.hpp"
#include "catamari/dense_factorizations.hpp"
#include "catamari/io_utils.hpp"
#include "catamari/ldl.hpp"
#include "quotient/timer.hpp"
#include "specify.hpp"

// Inverting a complex symmetric matrix in place is an integral part of this
// driver, and so the entire driver is disabled if LAPACK support was not
// detected.
#ifdef CATAMARI_HAVE_LAPACK

using catamari::BlasMatrix;
using catamari::BlasMatrixView;
using catamari::Complex;
using catamari::ComplexBase;
using catamari::CoordinateMatrix;
using catamari::Int;
using quotient::Buffer;

extern "C" {

void LAPACK_SYMBOL(cgetrf)(const BlasInt* height, const BlasInt* width,
                           BlasComplexFloat* matrix, const BlasInt* leading_dim,
                           BlasInt* pivots, BlasInt* info);

void LAPACK_SYMBOL(zgetrf)(const BlasInt* height, const BlasInt* width,
                           BlasComplexDouble* matrix,
                           const BlasInt* leading_dim, BlasInt* pivots,
                           BlasInt* info);

void LAPACK_SYMBOL(cgetri)(const BlasInt* height, BlasComplexFloat* matrix,
                           const BlasInt* leading_dim, const BlasInt* pivots,
                           BlasComplexFloat* work, const BlasInt* work_size,
                           BlasInt* info);

void LAPACK_SYMBOL(zgetri)(const BlasInt* height, BlasComplexDouble* matrix,
                           const BlasInt* leading_dim, const BlasInt* pivots,
                           BlasComplexDouble* work, const BlasInt* work_size,
                           BlasInt* info);

}  // extern "C"

namespace {

#ifdef CATAMARI_HAVE_LIBTIFF
// Returns whether the given value exists within a sorted list.
template <typename Iterator, typename T>
bool IndexExists(Iterator beg, Iterator end, T value) {
  auto iter = std::lower_bound(beg, end, value);
  return iter != end && *iter == value;
}

#endif  // ifdef CATAMARI_HAVE_LIBTIFF

template <typename Field>
std::vector<Int> SampleNonHermitianDPP(Int block_size, bool maximum_likelihood,
                                       BlasMatrixView<Field>* matrix,
                                       std::mt19937* generator) {
  const bool is_complex = catamari::IsComplex<Field>::value;
  const Int matrix_size = matrix->height;
  quotient::Timer timer;
  timer.Start();

  const std::vector<Int> sample = LowerFactorAndSampleNonHermitianDPP(
      block_size, maximum_likelihood, matrix, generator);

  const double runtime = timer.Stop();
  const double flops =
      (is_complex ? 4 : 1) * 2 * std::pow(1. * matrix_size, 3.) / 3.;
  const double gflops_per_sec = flops / (1.e9 * runtime);
  std::cout << "Sequential time: " << runtime << " seconds." << std::endl;
  std::cout << "Sequential DPP GFlop/s: " << gflops_per_sec << std::endl;

  const ComplexBase<Field> log_likelihood = catamari::DPPLogLikelihood(*matrix);
  std::cout << "Sequential DPP log-likelihood: " << log_likelihood << std::endl;

  return sample;
}

#ifdef CATAMARI_OPENMP
template <typename Field>
std::vector<Int> OpenMPSampleNonHermitianDPP(Int tile_size, Int block_size,
                                             bool maximum_likelihood,
                                             BlasMatrixView<Field>* matrix,
                                             std::mt19937* generator) {
  const bool is_complex = catamari::IsComplex<Field>::value;
  const Int matrix_size = matrix->height;
  quotient::Timer timer;
  timer.Start();

  const int old_max_threads = catamari::GetMaxBlasThreads();
  catamari::SetNumBlasThreads(1);

  std::vector<Int> sample;
  #pragma omp parallel
  #pragma omp single
  sample = OpenMPLowerFactorAndSampleNonHermitianDPP(
      tile_size, block_size, maximum_likelihood, matrix, generator);

  catamari::SetNumBlasThreads(old_max_threads);

  const double runtime = timer.Stop();
  const double flops =
      (is_complex ? 4 : 1) * 2 * std::pow(1. * matrix_size, 3.) / 3.;
  const double gflops_per_sec = flops / (1.e9 * runtime);
  std::cout << "OpenMP time: " << runtime << " seconds." << std::endl;
  std::cout << "OpenMP DPP GFlop/s: " << gflops_per_sec << std::endl;

  const ComplexBase<Field> log_likelihood = catamari::DPPLogLikelihood(*matrix);
  std::cout << "OpenMP DPP log-likelihood: " << log_likelihood << std::endl;

  return sample;
}
#endif  // ifdef CATAMARI_OPENMP

// Constructs the Kasteleyn matrix for an Aztec diamond.
template <typename Real>
void KasteleynMatrix(Int diamond_size,
                     CoordinateMatrix<Complex<Real>>* matrix) {
  const Int i1_length = diamond_size + 1;
  const Int i2_length = diamond_size;
  const Int num_vertices = i1_length * i2_length;
  matrix->Resize(num_vertices, num_vertices);
  matrix->ReserveEntryAdditions(4 * num_vertices);

  // Iterate over the coordinates of the black vertices.
  for (Int i1 = 0; i1 < i1_length; ++i1) {
    for (Int i2 = 0; i2 < i2_length; ++i2) {
      const bool negate = (i1 + i2) % 2;
      const Real scale = negate ? Real{-1} : Real{1};
      const Int black_index = i1 + i2 * i1_length;

      if (i1 > 0) {
        // Connect down and left (via -e1) to (i2, i1-1) in the white
        // coordinates.
        const Int white_dl_index = i2 + (i1 - 1) * i1_length;
        matrix->QueueEntryAddition(black_index, white_dl_index, -scale);

        // Connect up and left (via -e2) to (i2+1, i1-1) in the white
        // coordinates.
        const Int white_ul_index = (i2 + 1) + (i1 - 1) * i1_length;
        matrix->QueueEntryAddition(black_index, white_ul_index,
                                   Complex<Real>(0, scale));
      }

      if (i1 < diamond_size) {
        // Connect up and right (via e1) to (i2+1, i1) in the white
        // coordinates.
        const Int white_ur_index = (i2 + 1) + i1 * i1_length;
        matrix->QueueEntryAddition(black_index, white_ur_index, scale);

        // Connect down and right (via e2) to (i2, i1) in the white coordinates.
        const Int white_dr_index = i2 + i1 * i1_length;
        matrix->QueueEntryAddition(black_index, white_dr_index,
                                   Complex<Real>(0, -scale));
      }
    }
  }

  matrix->FlushEntryQueues();
}

// Fills a dense matrix from a sparse matrix.
template <typename Field>
void ConvertToDense(const CoordinateMatrix<Field>& sparse_matrix,
                    BlasMatrix<Field>* matrix) {
  const Int num_rows = sparse_matrix.NumRows();
  const Int num_cols = sparse_matrix.NumColumns();

  matrix->Resize(num_rows, num_cols, Field{0});
  for (const catamari::MatrixEntry<Field>& entry : sparse_matrix.Entries()) {
    matrix->Entry(entry.row, entry.column) = entry.value;
  }
}

// Inverts a single-precision complex dense matrix in-place.
void InvertMatrix(BlasMatrix<Complex<float>>* matrix) {
  const BlasInt num_rows = matrix->view.height;
  const BlasInt leading_dim = matrix->view.leading_dim;

  Buffer<BlasInt> pivots(num_rows);

  const BlasInt work_size = 128 * num_rows;
  Buffer<Complex<float>> work(work_size);

  BlasInt info;

  LAPACK_SYMBOL(cgetrf)
  (&num_rows, &num_rows, matrix->view.data, &leading_dim, pivots.Data(), &info);
  if (info) {
    std::cerr << "Exited cgetrf with info=" << info << std::endl;
    return;
  }

  LAPACK_SYMBOL(cgetri)
  (&num_rows, matrix->view.data, &leading_dim, pivots.Data(), work.Data(),
   &work_size, &info);
  if (info) {
    std::cerr << "Exited cgetri with info=" << info << std::endl;
  }
}

// Inverts a double-precision complex dense matrix in-place.
void InvertMatrix(BlasMatrix<Complex<double>>* matrix) {
  const BlasInt num_rows = matrix->view.height;
  const BlasInt leading_dim = matrix->view.leading_dim;

  Buffer<BlasInt> pivots(num_rows);

  const BlasInt work_size = 128 * num_rows;
  Buffer<Complex<double>> work(work_size);

  BlasInt info;

  LAPACK_SYMBOL(zgetrf)
  (&num_rows, &num_rows, matrix->view.data, &leading_dim, pivots.Data(), &info);
  if (info) {
    std::cerr << "Exited zgetrf with info=" << info << std::endl;
    return;
  }

  LAPACK_SYMBOL(zgetri)
  (&num_rows, matrix->view.data, &leading_dim, pivots.Data(), work.Data(),
   &work_size, &info);
  if (info) {
    std::cerr << "Exited zgetri with info=" << info << std::endl;
  }
}

// Forms the dense inverse of the Kasteleyn matrix.
// Note that we are not free to use a naive LDL^T sparse-direct factorization,
// as there are several zero diagonal entries in the Kasteleyn matrix.
template <typename Real>
void InvertKasteleynMatrix(
    const CoordinateMatrix<Complex<Real>> kasteleyn_matrix,
    BlasMatrix<Complex<Real>>* inverse_kasteleyn_matrix) {
  ConvertToDense(kasteleyn_matrix, inverse_kasteleyn_matrix);
  InvertMatrix(inverse_kasteleyn_matrix);
}

template <typename Real>
void KenyonMatrix(Int diamond_size,
                  const CoordinateMatrix<Complex<Real>>& kasteleyn_matrix,
                  const BlasMatrix<Complex<Real>>& inverse_kasteleyn_matrix,
                  BlasMatrix<Complex<Real>>* kenyon_matrix) {
  const Int num_edges = 4 * diamond_size * diamond_size;
  const Int i1_length = diamond_size + 1;

  BlasMatrix<Complex<Real>> dense_kasteleyn;
  ConvertToDense(kasteleyn_matrix, &dense_kasteleyn);

  /*
    We can group the edges into diamonds in the form:

      w
     / \
    b   b,
     \ /
      w


   starting in the bottom-left and working rightward and then upward. Within
   each diamond, we order the edges as


        w
       / \
      2   3
     /     \
    b       b.
     \     /
      0   1
       \ /
        w
  */

  // The Kenyon matrix has entries of the form:
  //
  //   L(e_i, e_j) = sqrt(K(b_i, w_i)) inv(K)(w_j, b_i) sqrt(K(b_j, w_j)).
  //
  kenyon_matrix->Resize(num_edges, num_edges);
  for (Int i1 = 0; i1 < diamond_size; ++i1) {
    for (Int i2 = 0; i2 < diamond_size; ++i2) {
      const Int i_tile_offset = 4 * (i1 + i2 * diamond_size);
      const Int i_black_left = i1 + i2 * i1_length;
      const Int i_black_right = (i1 + 1) + i2 * i1_length;
      const Int i_white_bottom = i2 + i1 * i1_length;
      const Int i_white_top = (i2 + 1) + i1 * i1_length;

      const Int i_blacks[4] = {i_black_left, i_black_right, i_black_left,
                               i_black_right};
      const Int i_whites[4] = {i_white_bottom, i_white_bottom, i_white_top,
                               i_white_top};

      for (Int j2 = 0; j2 < diamond_size; ++j2) {
        for (Int j1 = 0; j1 < diamond_size; ++j1) {
          const Int j_tile_offset = 4 * (j1 + j2 * diamond_size);
          const Int j_white_bottom = j2 + j1 * i1_length;
          const Int j_white_top = (j2 + 1) + j1 * i1_length;

          const Int j_whites[4] = {j_white_bottom, j_white_bottom, j_white_top,
                                   j_white_top};

          for (Int i_edge = 0; i_edge < 4; ++i_edge) {
            const Complex<Real> i_kasteleyn_value =
                dense_kasteleyn(i_blacks[i_edge], i_whites[i_edge]);
            for (Int j_edge = 0; j_edge < 4; ++j_edge) {
              const Complex<Real> inverse_kasteleyn_value =
                  inverse_kasteleyn_matrix(j_whites[j_edge], i_blacks[i_edge]);
              kenyon_matrix->Entry(i_tile_offset + i_edge,
                                   j_tile_offset + j_edge) =
                  i_kasteleyn_value * inverse_kasteleyn_value;
            }
          }
        }
      }
    }
  }
}

// Prints the sample to the console.
void PrintSample(const std::vector<Int>& sample) {
  std::ostringstream os;
  os << "Sample: ";
  for (const Int& index : sample) {
    os << index << " ";
  }
  os << std::endl;
  std::cout << os.str();
}

#ifdef CATAMARI_HAVE_LIBTIFF
// A configuration
struct DominoTIFFConfig {
  // The pixel width for each vertex.
  Int box_size = 10;

  // The RGB values of the source portion of north dimers.
  catamari::Pixel north_source_pixel{char(170), char(170), char(0)};

  // The RGB values of the target portion of north dimers.
  catamari::Pixel north_target_pixel{char(255), char(255), char(0)};

  // The RGB values of the source portion of south dimers.
  catamari::Pixel south_source_pixel{char(0), char(170), char(0)};

  // The RGB values of the target portion of south dimers.
  catamari::Pixel south_target_pixel{char(153), char(255), char(153)};

  // The RGB values of the source portion of east dimers.
  catamari::Pixel east_source_pixel{char(170), char(0), char(0)};

  // The RGB values of the target portion of east dimers.
  catamari::Pixel east_target_pixel{char(255), char(153), char(153)};

  // The RGB values of the source portion of west dimers.
  catamari::Pixel west_source_pixel{char(0), char(0), char(170)};

  // The RGB values of the target portion of west dimers.
  catamari::Pixel west_target_pixel{char(102), char(102), char(255)};

  // The RGB value of the background pixel.
  catamari::Pixel background_pixel{char(255), char(255), char(255)};

  // The RGB value of the tile boundary pixel.
  catamari::Pixel boundary_pixel{char(0), char(0), char(0)};
};

// Writes out a TIFF image for the domino tiling of the Aztec diamond.
void WriteTilingToTIFF(const std::string& filename, Int diamond_size,
                       const std::vector<Int>& sample,
                       const DominoTIFFConfig& config) {
  const Int height = config.box_size * 2 * diamond_size;
  const Int width = height;
  const Int samples_per_pixel = 3;
  std::vector<char> image(width * height * samples_per_pixel);

  // Fill in the background pixels.
  for (Int i = 0; i < height; ++i) {
    for (Int j = 0; j < width; ++j) {
      catamari::PackPixel(i, j, width, config.background_pixel, &image);
    }
  }

  const Int i_origin = 0;
  const Int j_origin = config.box_size * diamond_size;

  for (const Int& index : sample) {
    // Determine which face the index is in.
    const Int face_index = index / 4;

    // Determine the i1 and i2 components of the black vertex at the left of the
    // face.
    const Int i1_left = face_index % diamond_size;
    const Int i2_left = face_index / diamond_size;

    // Determine the (x1, x2) components of the black vertex at the left of the
    // face.
    const Int x1_left = 2 * i1_left;
    const Int x2_left = 1 + 2 * i2_left;

    // Determine the (x1, x2) components of the right black vertex.
    const Int x1_right = x1_left + 2;
    const Int x2_right = x2_left;

    // Determine the (x1, x2) components of the bottom white vertex.
    const Int x1_bottom = x1_left + 1;
    const Int x2_bottom = x2_left - 1;

    // Determine the (x1, x2) components of the top white vertex.
    const Int x1_top = x1_left + 1;
    const Int x2_top = x2_left + 1;

    // Determine which edge in the face we are handling.
    const Int face_edge = index - 4 * face_index;

    // Determine the (x1, x2) components of the source (black) vertex of the
    // edge.
    const Int x1_source = face_edge % 2 ? x1_right : x1_left;
    const Int x2_source = face_edge % 2 ? x2_right : x2_left;

    // Determine the (x1, x2) components of the target (white) vertex of the
    // edge.
    const Int x1_target = face_edge / 2 ? x1_top : x1_bottom;
    const Int x2_target = face_edge / 2 ? x2_top : x2_bottom;

    // Convert the source and target (x1, x2) coordinates into an offset of the
    // (e1, -e2) coord's via inverting the linear system
    //
    //     | 1,  1 | | alpha + 1/2 | = | x1 |,
    //     | 1, -1 | | beta  + 1/2 |   | x2 |
    //
    // so   alpha = (x1 + x2 - 1) / 2, beta = (x1 - x2 - 1) / 2.
    //
    const Int alpha_source = (x1_source + x2_source - 1) / 2;
    const Int beta_source = (x1_source - x2_source - 1) / 2;
    const Int alpha_target = (x1_target + x2_target - 1) / 2;
    const Int beta_target = (x1_target - x2_target - 1) / 2;

    catamari::Pixel source_pixel, target_pixel;
    if (face_edge == 0) {
      // East
      source_pixel = config.east_source_pixel;
      target_pixel = config.east_target_pixel;
    } else if (face_edge == 1) {
      // South
      source_pixel = config.south_source_pixel;
      target_pixel = config.south_target_pixel;
    } else if (face_edge == 2) {
      // North
      source_pixel = config.north_source_pixel;
      target_pixel = config.north_target_pixel;
    } else if (face_edge == 3) {
      // West
      source_pixel = config.west_source_pixel;
      target_pixel = config.west_target_pixel;
    } else {
      std::cerr << "Impossible fallback." << std::endl;
      source_pixel = config.background_pixel;
      target_pixel = config.background_pixel;
    }

    // Draw the source box.
    const Int i_source_offset = i_origin + alpha_source * config.box_size;
    const Int j_source_offset = j_origin + beta_source * config.box_size;
    for (Int k = 0; k < config.box_size; ++k) {
      const Int i = i_source_offset + k;
      for (Int l = 0; l < config.box_size; ++l) {
        const Int j = j_source_offset + l;

        if (k == config.box_size - 1 && !(face_edge == 2)) {
          // Only draw the north wall if not a northern domino.
          catamari::PackPixel(i, j, width, config.boundary_pixel, &image);
        } else if (k == 0 && !(face_edge == 1)) {
          // Only draw the south wall if not a southern domino.
          catamari::PackPixel(i, j, width, config.boundary_pixel, &image);
        } else if (l == config.box_size - 1 && !(face_edge == 0)) {
          // Only draw the east wall if not an eastern domino.
          catamari::PackPixel(i, j, width, config.boundary_pixel, &image);
        } else if (l == 0 && !(face_edge == 3)) {
          // Only draw the west wall if not a western domino.
          catamari::PackPixel(i, j, width, config.boundary_pixel, &image);
        } else {
          catamari::PackPixel(i, j, width, source_pixel, &image);
        }
      }
    }

    // Draw the target box.
    const Int i_target_offset = i_origin + alpha_target * config.box_size;
    const Int j_target_offset = j_origin + beta_target * config.box_size;
    for (Int k = 0; k < config.box_size; ++k) {
      const Int i = i_target_offset + k;
      for (Int l = 0; l < config.box_size; ++l) {
        const Int j = j_target_offset + l;

        if (k == config.box_size - 1 && !(face_edge == 1)) {
          // Only draw the north wall if not a southern domino.
          catamari::PackPixel(i, j, width, config.boundary_pixel, &image);
        } else if (k == 0 && !(face_edge == 2)) {
          // Only draw the south wall if not a northern domino.
          catamari::PackPixel(i, j, width, config.boundary_pixel, &image);
        } else if (l == config.box_size - 1 && !(face_edge == 3)) {
          // Only draw the east wall if not a western domino.
          catamari::PackPixel(i, j, width, config.boundary_pixel, &image);
        } else if (l == 0 && !(face_edge == 0)) {
          // Only draw the west wall if not an eastern domino.
          catamari::PackPixel(i, j, width, config.boundary_pixel, &image);
        } else {
          catamari::PackPixel(i, j, width, target_pixel, &image);
        }
      }
    }
  }

  catamari::WriteTIFF(filename, height, width, samples_per_pixel, image);
}
#endif  // ifdef CATAMARI_HAVE_LIBTIFF

// Samples the Aztec diamond domino tiling a requested number of times using
// both
// sequential and OpenMP-parallelized algorithms.
template <typename Real>
void DominoTilings(bool maximum_likelihood, Int diamond_size, Int block_size,
                   Int tile_size, Int num_rounds, unsigned int random_seed,
                   bool write_tiff, Int box_size) {
  CoordinateMatrix<Complex<Real>> kasteleyn_matrix;
  KasteleynMatrix(diamond_size, &kasteleyn_matrix);

  BlasMatrix<Complex<Real>> inverse_kasteleyn_matrix;
  InvertKasteleynMatrix(kasteleyn_matrix, &inverse_kasteleyn_matrix);

  BlasMatrix<Complex<Real>> kenyon_matrix;
  KenyonMatrix(diamond_size, kasteleyn_matrix, inverse_kasteleyn_matrix,
               &kenyon_matrix);

  const Int expected_sample_size = diamond_size * (diamond_size + 1);

  // Use the default of blue for west, red for east, yellow for north, and
  // green for south.
  DominoTIFFConfig config;
  config.box_size = box_size;

  // Use blue for north/south and red for east/west.
  DominoTIFFConfig dual_config = config;
  dual_config.north_source_pixel = config.west_source_pixel;
  dual_config.north_target_pixel = config.west_target_pixel;
  dual_config.south_source_pixel = config.west_source_pixel;
  dual_config.south_target_pixel = config.west_target_pixel;
  dual_config.east_source_pixel = config.east_source_pixel;
  dual_config.east_target_pixel = config.east_target_pixel;
  dual_config.west_source_pixel = config.east_source_pixel;
  dual_config.west_target_pixel = config.east_target_pixel;

  std::mt19937 generator(random_seed);
  BlasMatrix<Complex<Real>> kenyon_copy;
  for (Int round = 0; round < num_rounds; ++round) {
#ifdef CATAMARI_OPENMP
    // Sample using the OpenMP DPP sampler.
    kenyon_copy = kenyon_matrix;
    const std::vector<Int> omp_sample =
        OpenMPSampleNonHermitianDPP(tile_size, block_size, maximum_likelihood,
                                    &kenyon_copy.view, &generator);
    if (Int(omp_sample.size()) != expected_sample_size) {
      std::cerr << "ERROR: Sampled " << omp_sample.size() << " instead of "
                << expected_sample_size << " dimers." << std::endl;
    }
#ifdef CATAMARI_HAVE_LIBTIFF
    if (write_tiff) {
      const std::string tag = std::to_string(diamond_size) +
                              std::string("-omp-") + typeid(Real).name();
      const std::string dual_tag = std::to_string(diamond_size) +
                                   std::string("-omp-dual-") +
                                   typeid(Real).name();
      const std::string filename =
          "aztec-" + std::string(maximum_likelihood ? "ml-" : "") +
          std::to_string(round) + "-" + tag + ".tif";
      const std::string dual_filename =
          "aztec-" + std::string(maximum_likelihood ? "ml-" : "") +
          std::to_string(round) + "-" + dual_tag + ".tif";
      WriteTilingToTIFF(filename, diamond_size, omp_sample, config);
      WriteTilingToTIFF(dual_filename, diamond_size, omp_sample, dual_config);
    } else {
      PrintSample(omp_sample);
    }
#else
    PrintSample(omp_sample);
#endif  // ifdef CATAMARI_HAVE_LIBTIFF
#endif  // ifdef CATAMARI_OPENMP

    // Sample using the sequential DPP sampler.
    kenyon_copy = kenyon_matrix;
    const std::vector<Int> sample = SampleNonHermitianDPP(
        block_size, maximum_likelihood, &kenyon_copy.view, &generator);
    if (Int(sample.size()) != expected_sample_size) {
      std::cerr << "ERROR: Sampled " << sample.size() << " instead of "
                << expected_sample_size << " dimers." << std::endl;
    }
#ifdef CATAMARI_HAVE_LIBTIFF
    if (write_tiff) {
      const std::string tag =
          std::to_string(diamond_size) + "-" + typeid(Real).name();
      const std::string dual_tag = std::to_string(diamond_size) +
                                   std::string("-dual-") + typeid(Real).name();
      const std::string filename =
          "aztec-" + std::string(maximum_likelihood ? "ml-" : "") +
          std::to_string(round) + "-" + tag + ".tif";
      const std::string dual_filename =
          "aztec-" + std::string(maximum_likelihood ? "ml-" : "") +
          std::to_string(round) + "-" + dual_tag + ".tif";
      WriteTilingToTIFF(filename, diamond_size, sample, config);
      WriteTilingToTIFF(dual_filename, diamond_size, sample, dual_config);
    } else {
      PrintSample(sample);
    }
#else
    PrintSample(sample);
#endif  // ifdef CATAMARI_HAVE_LIBTIFF
  }
}

}  // anonymous namespace

int main(int argc, char** argv) {
  specify::ArgumentParser parser(argc, argv);
  const Int diamond_size = parser.OptionalInput<Int>(
      "diamond_size", "The dimension of the Aztec diamond.", 30);
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
  const bool write_tiff = parser.OptionalInput<bool>(
      "write_tiff", "Write out the results into a TIFF file?", true);
  const Int box_size = parser.OptionalInput<Int>(
      "box_size", "The pixel width of each TIFF vertex.", 10);
  if (!parser.OK()) {
    return 0;
  }

  std::cout << "Single-precision:" << std::endl;
  DominoTilings<float>(maximum_likelihood, diamond_size, block_size, tile_size,
                       num_rounds, random_seed, write_tiff, box_size);
  std::cout << std::endl;

  std::cout << "Double-precision:" << std::endl;
  DominoTilings<double>(maximum_likelihood, diamond_size, block_size, tile_size,
                        num_rounds, random_seed, write_tiff, box_size);

  return 0;
}
#else
int main(int CATAMARI_UNUSED argc, char* CATAMARI_UNUSED argv[]) { return 0; }
#endif  // ifdef CATAMARI_HAVE_LAPACK
