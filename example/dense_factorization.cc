/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cmath>
#include <iostream>

#include "catamari/blas_matrix.hpp"
#include "catamari/dense_factorizations.hpp"
#include "catamari/ldl.hpp"
#include "quotient/timer.hpp"
#include "specify.hpp"

using catamari::BlasMatrix;
using catamari::BlasMatrixView;
using catamari::Complex;
using catamari::Int;
using quotient::Buffer;

namespace {

template <typename Field>
void InitializeMatrix(Int matrix_size, BlasMatrix<Field>* matrix) {
  matrix->Resize(matrix_size, matrix_size);
  for (Int j = 0; j < matrix_size; ++j) {
    matrix->Entry(j, j) = 2 * matrix_size;
    for (Int i = j + 1; i < matrix_size; ++i) {
      matrix->Entry(i, j) = Field{-1};
    }
  }
}

template <typename Field>
void RunCholeskyFactorization(Int tile_size, Int block_size,
                              BlasMatrixView<Field>* matrix) {
  const bool is_complex = catamari::IsComplex<Field>::value;
  const Int matrix_size = matrix->height;
  quotient::Timer timer;
  timer.Start();

#ifdef _OPENMP
  #pragma omp parallel
  #pragma omp single
  MultithreadedLowerCholeskyFactorization(tile_size, block_size, matrix);
#else
  LowerCholeskyFactorization(block_size, matrix);
#endif

  const double runtime = timer.Stop();
  const double flops =
      (is_complex ? 4 : 1) * std::pow(1. * matrix_size, 3.) / 3.;
  const double gflops_per_sec = flops / (1.e9 * runtime);
  std::cout << "Cholesky GFlops/sec: " << gflops_per_sec << std::endl;
}

template <typename Field>
void RunLDLAdjointFactorization(Int tile_size, Int block_size,
                                BlasMatrixView<Field>* matrix,
                                Buffer<Field>* extra_buffer) {
  const bool is_complex = catamari::IsComplex<Field>::value;
  const Int matrix_size = matrix->height;
  quotient::Timer timer;
  timer.Start();

#ifdef _OPENMP
  #pragma omp parallel
  #pragma omp single
  MultithreadedLowerLDLAdjointFactorization(tile_size, block_size, matrix,
                                            extra_buffer);
#else
  LowerLDLAdjointFactorization(block_size, matrix);
#endif

  const double runtime = timer.Stop();
  const double flops =
      (is_complex ? 4 : 1) * std::pow(1. * matrix_size, 3.) / 3.;
  const double gflops_per_sec = flops / (1.e9 * runtime);
  std::cout << "LDLAdjoint GFlops/sec: " << gflops_per_sec << std::endl;
}

template <typename Field>
void RunLDLTransposeFactorization(Int tile_size, Int block_size,
                                  BlasMatrixView<Field>* matrix,
                                  Buffer<Field>* extra_buffer) {
  const bool is_complex = catamari::IsComplex<Field>::value;
  const Int matrix_size = matrix->height;
  quotient::Timer timer;
  timer.Start();

#ifdef _OPENMP
  #pragma omp parallel
  #pragma omp single
  MultithreadedLowerLDLTransposeFactorization(tile_size, block_size, matrix,
                                              extra_buffer);
#else
  LowerLDLTransposeFactorization(block_size, matrix);
#endif

  const double runtime = timer.Stop();
  const double flops =
      (is_complex ? 4 : 1) * std::pow(1. * matrix_size, 3.) / 3.;
  const double gflops_per_sec = flops / (1.e9 * runtime);
  std::cout << "LDLTranspose GFlops/sec: " << gflops_per_sec << std::endl;
}

}  // anonymous namespace

int main(int argc, char** argv) {
  specify::ArgumentParser parser(argc, argv);
  const Int matrix_size = parser.OptionalInput<Int>(
      "matrix_size", "The dimension of the matrix to factor.", 1000);
  const Int tile_size = parser.OptionalInput<Int>(
      "tile_size", "The tile size for multithreaded factorization.", 128);
  const Int block_size = parser.OptionalInput<Int>(
      "block_size", "The block_size for dense factorization.", 64);
  const Int num_rounds = parser.OptionalInput<Int>(
      "num_rounds", "The number of rounds of factorizations.", 2);
  if (!parser.OK()) {
    return 0;
  }

  BlasMatrix<Complex<double>> matrix;
  Buffer<Complex<double>> extra_buffer;

  for (Int round = 0; round < num_rounds; ++round) {
    InitializeMatrix(matrix_size, &matrix);
    RunCholeskyFactorization(tile_size, block_size, &matrix.view);

    InitializeMatrix(matrix_size, &matrix);
    RunLDLAdjointFactorization(tile_size, block_size, &matrix.view,
                               &extra_buffer);

    InitializeMatrix(matrix_size, &matrix);
    RunLDLTransposeFactorization(tile_size, block_size, &matrix.view,
                                 &extra_buffer);

    std::cout << std::endl;
  }

  return 0;
}
