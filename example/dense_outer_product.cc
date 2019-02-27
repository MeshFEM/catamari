/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <iostream>

#include "catamari/blas_matrix.hpp"
#include "catamari/dense_basic_linear_algebra.hpp"
#include "quotient/timer.hpp"
#include "specify.hpp"

using catamari::BlasMatrix;
using catamari::BlasMatrixView;
using catamari::Complex;
using catamari::Int;
using quotient::Buffer;

namespace {

template <typename Field>
void InitializeMatrix(Int height, Int width, BlasMatrix<Field>* matrix) {
  matrix->Resize(height, width);
  for (Int j = 0; j < width; ++j) {
    for (Int i = 0; i < height; ++i) {
      matrix->Entry(i, j) = i + j;
    }
  }
}

template <typename Field>
void RunMatrixMultiplyLowerNormalNormal(Int height, Int rank) {
  const bool is_complex = catamari::IsComplex<Field>::value;
  quotient::Timer timer;

  BlasMatrix<Field> left_matrix, right_matrix, output_matrix;

  InitializeMatrix(height, rank, &left_matrix);
  InitializeMatrix(rank, height, &right_matrix);
  InitializeMatrix(height, height, &output_matrix);

  timer.Start();

  MatrixMultiplyLowerNormalNormal(Field{-1}, left_matrix.ConstView(),
                                  right_matrix.ConstView(), Field{1},
                                  &output_matrix.view);

  const double runtime = timer.Stop();
  const double flops =
      (is_complex ? 4. : 1.) * std::pow(1. * height, 2.) * rank;
  const double gflops_per_sec = flops / (1.e9 * runtime);

  std::cout << "Sequential GFlops/sec: " << gflops_per_sec << std::endl;
}

#ifdef _OPENMP
template <typename Field>
void RunOpenMPMatrixMultiplyLowerNormalNormal(Int tile_size, Int height,
                                              Int rank) {
  const bool is_complex = catamari::IsComplex<Field>::value;
  quotient::Timer timer;

  BlasMatrix<Field> left_matrix, right_matrix, output_matrix;

  InitializeMatrix(height, rank, &left_matrix);
  InitializeMatrix(rank, height, &right_matrix);
  InitializeMatrix(height, height, &output_matrix);

  timer.Start();

  #pragma omp parallel
  #pragma omp single
  OpenMPMatrixMultiplyLowerNormalNormal(
      tile_size, Field{-1}, left_matrix.ConstView(), right_matrix.ConstView(),
      Field{1}, &output_matrix.view);

  const double runtime = timer.Stop();
  const double flops =
      (is_complex ? 4. : 1.) * std::pow(1. * height, 2.) * rank;
  const double gflops_per_sec = flops / (1.e9 * runtime);

  std::cout << "OpenMP GFlops/sec: " << gflops_per_sec << std::endl;
}
#endif  // ifdef _OPENMP

}  // anonymous namespace

int main(int argc, char** argv) {
  specify::ArgumentParser parser(argc, argv);
  const Int height =
      parser.OptionalInput<Int>("height", "The height of the matrix.", 2000);
  const Int rank =
      parser.OptionalInput<Int>("rank", "The rank of the update.", 128);
#ifdef _OPENMP
  const Int tile_size = parser.OptionalInput<Int>(
      "tile_size", "The tile size for multithreaded factorization.", 128);
#endif  // ifdef _OPENMP
  const Int num_rounds = parser.OptionalInput<Int>(
      "num_rounds", "The number of rounds of factorizations.", 3);
  if (!parser.OK()) {
    return 0;
  }

  for (Int round = 0; round < num_rounds; ++round) {
#ifdef _OPENMP
    RunOpenMPMatrixMultiplyLowerNormalNormal<Complex<double>>(tile_size, height,
                                                              rank);
#endif  // ifdef _OPENMP
    RunMatrixMultiplyLowerNormalNormal<Complex<double>>(height, rank);
  }

  return 0;
}
