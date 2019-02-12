/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cmath>
#include <iostream>

#include "catamari/dense_factorizations.hpp"
#include "catamari/ldl.hpp"
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
void InitializeMatrix(Int matrix_size, BlasMatrix<Field>* matrix,
                      Buffer<Field>* buffer) {
  matrix->height = matrix->width = matrix->leading_dim = matrix_size;
  buffer->Resize(matrix->leading_dim * matrix->width);
  matrix->data = buffer->Data();

  for (Int j = 0; j < matrix_size; ++j) {
    matrix->Entry(j, j) = 1.;
    for (Int i = j + 1; i < matrix_size; ++i) {
      matrix->Entry(i, j) = -1. / (2 * matrix_size);
    }
  }
}

template <typename Field>
void SampleDPP(
    Int block_size, bool maximum_likelihood, BlasMatrix<Field>* matrix,
    std::mt19937* generator,
    std::uniform_real_distribution<ComplexBase<Field>>* uniform_dist) {
  const bool is_complex = catamari::IsComplex<Field>::value;
  const Int matrix_size = matrix->height;
  quotient::Timer timer;
  timer.Start();

  LowerFactorAndSampleDPP(block_size, maximum_likelihood, matrix, generator,
                          uniform_dist);

  const double runtime = timer.Stop();
  const double flops =
      (is_complex ? 4 : 1) * std::pow(1. * matrix_size, 3.) / 3.;
  const double gflops_per_sec = flops / (1.e9 * runtime);
  std::cout << "Sequential DPP GFlop/s: " << gflops_per_sec << std::endl;
}

#ifdef _OPENMP
template <typename Field>
void MultithreadedSampleDPP(
    Int tile_size, Int block_size, bool maximum_likelihood,
    BlasMatrix<Field>* matrix, std::mt19937* generator,
    std::uniform_real_distribution<ComplexBase<Field>>* uniform_dist,
    Buffer<Field>* extra_buffer) {
  const bool is_complex = catamari::IsComplex<Field>::value;
  const Int matrix_size = matrix->height;
  quotient::Timer timer;
  timer.Start();

  #pragma omp parallel
  #pragma omp single
  MultithreadedLowerFactorAndSampleDPP(tile_size, block_size,
                                       maximum_likelihood, matrix, generator,
                                       uniform_dist, extra_buffer);

  const double runtime = timer.Stop();
  const double flops =
      (is_complex ? 4 : 1) * std::pow(1. * matrix_size, 3.) / 3.;
  const double gflops_per_sec = flops / (1.e9 * runtime);
  std::cout << "Multithreaded DPP GFlop/s: " << gflops_per_sec << std::endl;
}
#endif  // ifdef _OPENMP

}  // anonymous namespace

int main(int argc, char** argv) {
  specify::ArgumentParser parser(argc, argv);
  const Int matrix_size = parser.OptionalInput<Int>(
      "matrix_size", "The dimension of the matrix to factor.", 1000);
#ifdef _OPENMP
  const Int tile_size = parser.OptionalInput<Int>(
      "tile_size", "The tile size for multithreaded factorization.", 128);
#endif  // ifdef _OPENMP
  const Int block_size = parser.OptionalInput<Int>(
      "block_size", "The block_size for dense factorization.", 64);
  const Int num_rounds = parser.OptionalInput<Int>(
      "num_rounds", "The number of rounds of factorizations.", 2);
  const unsigned int random_seed = parser.OptionalInput<unsigned int>(
      "random_seed", "The random seed for the DPP.", 17u);
  const bool maximum_likelihood = parser.OptionalInput<bool>(
      "maximum_likelihood", "Take a maximum likelihood DPP sample?", true);
  if (!parser.OK()) {
    return 0;
  }

  // TODO(Jack Poulson): Templatize so that there is no float/double redundancy.

  std::cout << "Single-precision:" << std::endl;
  {
    BlasMatrix<Complex<float>> matrix;
    Buffer<Complex<float>> buffer;
    Buffer<Complex<float>> extra_buffer(matrix_size * matrix_size);

    std::mt19937 generator(random_seed);
    std::uniform_real_distribution<float> uniform_dist(0., 1.);

    for (Int round = 0; round < num_rounds; ++round) {
#ifdef _OPENMP
      InitializeMatrix(matrix_size, &matrix, &buffer);
      MultithreadedSampleDPP(tile_size, block_size, maximum_likelihood, &matrix,
                             &generator, &uniform_dist, &extra_buffer);
#endif  // ifdef _OPENMP

      InitializeMatrix(matrix_size, &matrix, &buffer);
      SampleDPP(block_size, maximum_likelihood, &matrix, &generator,
                &uniform_dist);
    }
  }
  std::cout << std::endl;

  std::cout << "Double-precision:" << std::endl;
  {
    BlasMatrix<Complex<double>> matrix;
    Buffer<Complex<double>> buffer;
    Buffer<Complex<double>> extra_buffer(matrix_size * matrix_size);

    std::mt19937 generator(random_seed);
    std::uniform_real_distribution<double> uniform_dist(0., 1.);

    for (Int round = 0; round < num_rounds; ++round) {
#ifdef _OPENMP
      InitializeMatrix(matrix_size, &matrix, &buffer);
      MultithreadedSampleDPP(tile_size, block_size, maximum_likelihood, &matrix,
                             &generator, &uniform_dist, &extra_buffer);
#endif  // ifdef _OPENMP

      InitializeMatrix(matrix_size, &matrix, &buffer);
      SampleDPP(block_size, maximum_likelihood, &matrix, &generator,
                &uniform_dist);
    }
  }

  return 0;
}
