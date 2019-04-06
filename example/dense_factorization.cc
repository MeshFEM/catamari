/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
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
  matrix->Resize(matrix_size, matrix_size, Field{0});
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

  int num_pivots;
#ifdef CATAMARI_OPENMP
  const int old_max_threads = catamari::GetMaxBlasThreads();
  catamari::SetNumBlasThreads(1);
  #pragma omp parallel
  #pragma omp single
  num_pivots = OpenMPLowerCholeskyFactorization(tile_size, block_size, matrix);
  catamari::SetNumBlasThreads(old_max_threads);
#else
  num_pivots = LowerCholeskyFactorization(block_size, matrix);
#endif

  if (num_pivots == matrix_size) {
    const double runtime = timer.Stop();
    const double flops =
        (is_complex ? 4 : 1) * std::pow(1. * matrix_size, 3.) / 3.;
    const double gflops_per_sec = flops / (1.e9 * runtime);
    std::cout << "Cholesky runtime: " << runtime << " seconds." << std::endl;
    std::cout << "Cholesky GFlops/sec: " << gflops_per_sec << std::endl;
  } else {
    std::cout << "Cholesky failed after " << num_pivots << " pivots."
              << std::endl;
  }
}

template <typename Field>
void RunLDLAdjointFactorization(Int tile_size, Int block_size,
                                BlasMatrixView<Field>* matrix,
                                Buffer<Field>* extra_buffer) {
  const bool is_complex = catamari::IsComplex<Field>::value;
  const Int matrix_size = matrix->height;
  quotient::Timer timer;
  timer.Start();

  int num_pivots;
#ifdef CATAMARI_OPENMP
  const int old_max_threads = catamari::GetMaxBlasThreads();
  catamari::SetNumBlasThreads(1);
  #pragma omp parallel
  #pragma omp single
  num_pivots = OpenMPLowerLDLAdjointFactorization(tile_size, block_size, matrix,
                                                  extra_buffer);
  catamari::SetNumBlasThreads(old_max_threads);
#else
  num_pivots = LowerLDLAdjointFactorization(block_size, matrix);
#endif

  if (num_pivots == matrix_size) {
    const double runtime = timer.Stop();
    const double flops =
        (is_complex ? 4 : 1) * std::pow(1. * matrix_size, 3.) / 3.;
    const double gflops_per_sec = flops / (1.e9 * runtime);
    std::cout << "LDL^H runtime: " << runtime << " seconds." << std::endl;
    std::cout << "LDL^H GFlops/sec: " << gflops_per_sec << std::endl;
  } else {
    std::cout << "LDL^H failed after " << num_pivots << " pivots." << std::endl;
  }
}

template <typename Field>
void RunLDLTransposeFactorization(Int tile_size, Int block_size,
                                  BlasMatrixView<Field>* matrix,
                                  Buffer<Field>* extra_buffer) {
  const bool is_complex = catamari::IsComplex<Field>::value;
  const Int matrix_size = matrix->height;
  quotient::Timer timer;
  timer.Start();

  int num_pivots;
#ifdef CATAMARI_OPENMP
  const int old_max_threads = catamari::GetMaxBlasThreads();
  catamari::SetNumBlasThreads(1);
  #pragma omp parallel
  #pragma omp single
  num_pivots = OpenMPLowerLDLTransposeFactorization(tile_size, block_size,
                                                    matrix, extra_buffer);
  catamari::SetNumBlasThreads(old_max_threads);
#else
  num_pivots = LowerLDLTransposeFactorization(block_size, matrix);
#endif

  if (num_pivots == matrix_size) {
    const double runtime = timer.Stop();
    const double flops =
        (is_complex ? 4 : 1) * std::pow(1. * matrix_size, 3.) / 3.;
    const double gflops_per_sec = flops / (1.e9 * runtime);
    std::cout << "LDL^T runtime: " << runtime << " seconds." << std::endl;
    std::cout << "LDL^T GFlops/sec: " << gflops_per_sec << std::endl;
  } else {
    std::cout << "LDL^T failed after " << num_pivots << " pivots." << std::endl;
  }
}

template <typename Field>
void TestFactorizations(Int matrix_size, Int num_rounds, Int tile_size,
                        Int block_size) {
  BlasMatrix<Field> matrix;
  Buffer<Field> extra_buffer;

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

  std::cout << "Testing for float:" << std::endl;
  TestFactorizations<float>(matrix_size, num_rounds, tile_size, block_size);
  std::cout << std::endl;

  std::cout << "Testing for double:" << std::endl;
  TestFactorizations<double>(matrix_size, num_rounds, tile_size, block_size);
  std::cout << std::endl;

  std::cout << "Testing for Complex<float>:" << std::endl;
  TestFactorizations<Complex<float>>(matrix_size, num_rounds, tile_size,
                                     block_size);
  std::cout << std::endl;

  std::cout << "Testing for Complex<double>:" << std::endl;
  TestFactorizations<Complex<double>>(matrix_size, num_rounds, tile_size,
                                      block_size);

  return 0;
}
