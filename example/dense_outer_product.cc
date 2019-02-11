/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cmath>
#include <iostream>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "quotient/timer.hpp"
#include "specify.hpp"

using catamari::BlasMatrix;
using catamari::Complex;
using catamari::ComplexBase;
using catamari::Conjugate;
using catamari::ConstBlasMatrix;
using catamari::Int;
using quotient::Buffer;

namespace {

template <typename Field>
void InitializeMatrix(Int height, Int width, BlasMatrix<Field>* matrix,
                      Buffer<Field>* buffer) {
  matrix->height = height;
  matrix->width = width;
  matrix->leading_dim = height;

  const Int buffer_size = matrix->leading_dim * matrix->width;
  buffer->Resize(buffer_size);
  matrix->data = buffer->Data();

  for (Int j = 0; j < width; ++j) {
    for (Int i = 0; i < height; ++i) {
      matrix->Entry(i, j) = i + j;
    }
  }
}

template <typename Field>
void RunMatrixMultiplyLowerNormalNormal(Int tile_size, Int height, Int rank) {
  const bool is_complex = catamari::IsComplex<Field>::value;
  quotient::Timer timer;

  Buffer<Field> left_buffer, right_buffer, output_buffer;
  BlasMatrix<Field> left_matrix, right_matrix, output_matrix;

  InitializeMatrix(height, rank, &left_matrix, &left_buffer);
  InitializeMatrix(rank, height, &right_matrix, &right_buffer);
  InitializeMatrix(height, height, &output_matrix, &output_buffer);

  timer.Start();

  #pragma omp parallel
  #pragma omp single
  MultithreadedMatrixMultiplyLowerNormalNormal(
      tile_size, Field{-1}, left_matrix.ToConst(), right_matrix.ToConst(),
      Field{1}, &output_matrix);

  const double runtime = timer.Stop();
  const double flops =
      (is_complex ? 4. : 1.) * std::pow(1. * height, 2.) * rank;
  const double gflops_per_sec = flops / (1.e9 * runtime);

  std::cout << "GFlops/sec: " << gflops_per_sec << std::endl;
}

}  // anonymous namespace

int main(int argc, char** argv) {
  specify::ArgumentParser parser(argc, argv);
  const Int height =
      parser.OptionalInput<Int>("height", "The height of the matrix.", 2000);
  const Int rank =
      parser.OptionalInput<Int>("rank", "The rank of the update.", 128);
  const Int tile_size = parser.OptionalInput<Int>(
      "tile_size", "The tile size for multithreaded factorization.", 128);
  const Int num_rounds = parser.OptionalInput<Int>(
      "num_rounds", "The number of rounds of factorizations.", 2);
  if (!parser.OK()) {
    return 0;
  }

  for (Int round = 0; round < num_rounds; ++round) {
    RunMatrixMultiplyLowerNormalNormal<Complex<double>>(tile_size, height,
                                                        rank);
  }

  return 0;
}
