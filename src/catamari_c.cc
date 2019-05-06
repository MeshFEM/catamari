/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "catamari.h"

namespace {

std::mt19937 g_generator;

void VectorIntToC(const std::vector<catamari::Int>& vec,
                  CatamariBufferInt* vec_c) {
  CatamariBufferIntDestroy(vec_c);
  vec_c->size = vec.size();
  vec_c->data = new CatamariInt[vec_c->size];
  for (catamari::Int i = 0; i < vec_c->size; ++i) {
    vec_c->data[i] = vec[i];
  }
}

catamari::BlasMatrixView<float> BlasMatrixViewToCxx(
    CatamariBlasMatrixViewFloat* matrix) {
  catamari::BlasMatrixView<float> matrix_cxx;
  matrix_cxx.height = matrix->height;
  matrix_cxx.width = matrix->width;
  matrix_cxx.leading_dim = matrix->leading_dim;
  matrix_cxx.data = matrix->data;
  return matrix_cxx;
}

catamari::BlasMatrixView<double> BlasMatrixViewToCxx(
    CatamariBlasMatrixViewDouble* matrix) {
  catamari::BlasMatrixView<double> matrix_cxx;
  matrix_cxx.height = matrix->height;
  matrix_cxx.width = matrix->width;
  matrix_cxx.leading_dim = matrix->leading_dim;
  matrix_cxx.data = matrix->data;
  return matrix_cxx;
}

catamari::BlasMatrixView<catamari::Complex<float>> BlasMatrixViewToCxx(
    CatamariBlasMatrixViewComplexFloat* matrix) {
  catamari::BlasMatrixView<catamari::Complex<float>> matrix_cxx;
  matrix_cxx.height = matrix->height;
  matrix_cxx.width = matrix->width;
  matrix_cxx.leading_dim = matrix->leading_dim;
  matrix_cxx.data = reinterpret_cast<catamari::Complex<float>*>(matrix->data);
  return matrix_cxx;
}

catamari::BlasMatrixView<catamari::Complex<double>> BlasMatrixViewToCxx(
    CatamariBlasMatrixViewComplexDouble* matrix) {
  catamari::BlasMatrixView<catamari::Complex<double>> matrix_cxx;
  matrix_cxx.height = matrix->height;
  matrix_cxx.width = matrix->width;
  matrix_cxx.leading_dim = matrix->leading_dim;
  matrix_cxx.data = reinterpret_cast<catamari::Complex<double>*>(matrix->data);
  return matrix_cxx;
}

}  // anonymous namespace

void CatamariHas64BitInts(bool* has_64_bit_ints) {
  *has_64_bit_ints = sizeof(catamari::Int) == 8;
}

void CatamariHasOpenMP(bool* has_openmp) {
#ifdef CATAMARI_OPENMP
  *has_openmp = true;
#else
  *has_openmp = false;
#endif
}

void CatamariBufferIntInit(CatamariBufferInt* buffer) {
  buffer->size = 0;
  buffer->data = nullptr;
}

void CatamariBufferIntDestroy(CatamariBufferInt* buffer) {
  buffer->size = 0;
  if (buffer->data) {
    delete[] buffer->data;
    buffer->data = nullptr;
  }
}

void CatamariInitGenerator(unsigned int random_seed) {
  g_generator.seed(random_seed);
}

void CatamariSampleLowerHermitianDPPFloat(CatamariInt block_size,
                                          bool maximum_likelihood,
                                          CatamariBlasMatrixViewFloat* matrix,
                                          CatamariBufferInt* sample) {
  // Convert the view into C++.
  auto matrix_cxx = BlasMatrixViewToCxx(matrix);

  // Sample the DPP.
  const std::vector<catamari::Int> sample_cxx =
      catamari::SampleLowerHermitianDPP(block_size, maximum_likelihood,
                                        &matrix_cxx, &g_generator);

  // Return the sample in the C format.
  VectorIntToC(sample_cxx, sample);
}

void CatamariSampleLowerHermitianDPPDouble(CatamariInt block_size,
                                           bool maximum_likelihood,
                                           CatamariBlasMatrixViewDouble* matrix,
                                           CatamariBufferInt* sample) {
  // Convert the view into C++.
  auto matrix_cxx = BlasMatrixViewToCxx(matrix);

  // Sample the DPP.
  const std::vector<catamari::Int> sample_cxx =
      catamari::SampleLowerHermitianDPP(block_size, maximum_likelihood,
                                        &matrix_cxx, &g_generator);

  // Return the sample in the C format.
  VectorIntToC(sample_cxx, sample);
}

void CatamariSampleLowerHermitianDPPComplexFloat(
    CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexFloat* matrix, CatamariBufferInt* sample) {
  // Convert the view into C++.
  auto matrix_cxx = BlasMatrixViewToCxx(matrix);

  // Sample the DPP.
  const std::vector<catamari::Int> sample_cxx =
      catamari::SampleLowerHermitianDPP(block_size, maximum_likelihood,
                                        &matrix_cxx, &g_generator);

  // Return the sample in the C format.
  VectorIntToC(sample_cxx, sample);
}

void CatamariSampleLowerHermitianDPPComplexDouble(
    CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexDouble* matrix, CatamariBufferInt* sample) {
  // Convert the view into C++.
  auto matrix_cxx = BlasMatrixViewToCxx(matrix);

  // Sample the DPP.
  const std::vector<catamari::Int> sample_cxx =
      catamari::SampleLowerHermitianDPP(block_size, maximum_likelihood,
                                        &matrix_cxx, &g_generator);

  // Return the sample in the C format.
  VectorIntToC(sample_cxx, sample);
}

#ifdef CATAMARI_OPENMP
void CatamariOpenMPSampleLowerHermitianDPPFloat(
    CatamariInt tile_size, CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewFloat* matrix, CatamariBufferInt* sample) {
  // Convert the view into C++.
  auto matrix_cxx = BlasMatrixViewToCxx(matrix);

  // Sample the DPP.
  const catamari::Int height = matrix_cxx.height;
  catamari::Buffer<float> buffer(height * height);
  std::vector<catamari::Int> sample_cxx;
  #pragma omp parallel
  #pragma omp single
  sample_cxx = catamari::OpenMPSampleLowerHermitianDPP(
      tile_size, block_size, maximum_likelihood, &matrix_cxx, &g_generator,
      &buffer);

  // Return the sample in the C format.
  VectorIntToC(sample_cxx, sample);
}

void CatamariOpenMPSampleLowerHermitianDPPDouble(
    CatamariInt tile_size, CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewDouble* matrix, CatamariBufferInt* sample) {
  // Convert the view into C++.
  auto matrix_cxx = BlasMatrixViewToCxx(matrix);

  // Sample the DPP.
  const catamari::Int height = matrix_cxx.height;
  catamari::Buffer<double> buffer(height * height);
  std::vector<catamari::Int> sample_cxx;
  #pragma omp parallel
  #pragma omp single
  sample_cxx = catamari::OpenMPSampleLowerHermitianDPP(
      tile_size, block_size, maximum_likelihood, &matrix_cxx, &g_generator,
      &buffer);

  // Return the sample in the C format.
  VectorIntToC(sample_cxx, sample);
}

void CatamariOpenMPSampleLowerHermitianDPPComplexFloat(
    CatamariInt tile_size, CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexFloat* matrix, CatamariBufferInt* sample) {
  // Convert the view into C++.
  auto matrix_cxx = BlasMatrixViewToCxx(matrix);

  // Sample the DPP.
  const catamari::Int height = matrix_cxx.height;
  catamari::Buffer<catamari::Complex<float>> buffer(height * height);
  std::vector<catamari::Int> sample_cxx;
  #pragma omp parallel
  #pragma omp single
  sample_cxx = catamari::OpenMPSampleLowerHermitianDPP(
      tile_size, block_size, maximum_likelihood, &matrix_cxx, &g_generator,
      &buffer);

  // Return the sample in the C format.
  VectorIntToC(sample_cxx, sample);
}

void CatamariOpenMPSampleLowerHermitianDPPComplexDouble(
    CatamariInt tile_size, CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexDouble* matrix, CatamariBufferInt* sample) {
  // Convert the view into C++.
  auto matrix_cxx = BlasMatrixViewToCxx(matrix);

  // Sample the DPP.
  const catamari::Int height = matrix_cxx.height;
  catamari::Buffer<catamari::Complex<double>> buffer(height * height);
  std::vector<catamari::Int> sample_cxx;
  #pragma omp parallel
  #pragma omp single
  sample_cxx = catamari::OpenMPSampleLowerHermitianDPP(
      tile_size, block_size, maximum_likelihood, &matrix_cxx, &g_generator,
      &buffer);

  // Return the sample in the C format.
  VectorIntToC(sample_cxx, sample);
}
#endif  // ifdef CATAMARI_OPENMP

void CatamariSampleNonHermitianDPPFloat(CatamariInt block_size,
                                        bool maximum_likelihood,
                                        CatamariBlasMatrixViewFloat* matrix,
                                        CatamariBufferInt* sample) {
  // Convert the view into C++.
  auto matrix_cxx = BlasMatrixViewToCxx(matrix);

  // Sample the DPP.
  const std::vector<catamari::Int> sample_cxx = catamari::SampleNonHermitianDPP(
      block_size, maximum_likelihood, &matrix_cxx, &g_generator);

  // Return the sample in the C format.
  VectorIntToC(sample_cxx, sample);
}

void CatamariSampleNonHermitianDPPDouble(CatamariInt block_size,
                                         bool maximum_likelihood,
                                         CatamariBlasMatrixViewDouble* matrix,
                                         CatamariBufferInt* sample) {
  // Convert the view into C++.
  auto matrix_cxx = BlasMatrixViewToCxx(matrix);

  // Sample the DPP.
  const std::vector<catamari::Int> sample_cxx = catamari::SampleNonHermitianDPP(
      block_size, maximum_likelihood, &matrix_cxx, &g_generator);

  // Return the sample in the C format.
  VectorIntToC(sample_cxx, sample);
}

void CatamariSampleNonHermitianDPPComplexFloat(
    CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexFloat* matrix, CatamariBufferInt* sample) {
  // Convert the view into C++.
  auto matrix_cxx = BlasMatrixViewToCxx(matrix);

  // Sample the DPP.
  const std::vector<catamari::Int> sample_cxx = catamari::SampleNonHermitianDPP(
      block_size, maximum_likelihood, &matrix_cxx, &g_generator);

  // Return the sample in the C format.
  VectorIntToC(sample_cxx, sample);
}

void CatamariSampleNonHermitianDPPComplexDouble(
    CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexDouble* matrix, CatamariBufferInt* sample) {
  // Convert the view into C++.
  auto matrix_cxx = BlasMatrixViewToCxx(matrix);

  // Sample the DPP.
  const std::vector<catamari::Int> sample_cxx = catamari::SampleNonHermitianDPP(
      block_size, maximum_likelihood, &matrix_cxx, &g_generator);

  // Return the sample in the C format.
  VectorIntToC(sample_cxx, sample);
}

#ifdef CATAMARI_OPENMP
void CatamariOpenMPSampleNonHermitianDPPFloat(
    CatamariInt tile_size, CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewFloat* matrix, CatamariBufferInt* sample) {
  // Convert the view into C++.
  auto matrix_cxx = BlasMatrixViewToCxx(matrix);

  // Sample the DPP.
  std::vector<catamari::Int> sample_cxx;
  #pragma omp parallel
  #pragma omp single
  sample_cxx = catamari::OpenMPSampleNonHermitianDPP(
      tile_size, block_size, maximum_likelihood, &matrix_cxx, &g_generator);

  // Return the sample in the C format.
  VectorIntToC(sample_cxx, sample);
}

void CatamariOpenMPSampleNonHermitianDPPDouble(
    CatamariInt tile_size, CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewDouble* matrix, CatamariBufferInt* sample) {
  // Convert the view into C++.
  auto matrix_cxx = BlasMatrixViewToCxx(matrix);

  // Sample the DPP.
  std::vector<catamari::Int> sample_cxx;
  #pragma omp parallel
  #pragma omp single
  sample_cxx = catamari::OpenMPSampleNonHermitianDPP(
      tile_size, block_size, maximum_likelihood, &matrix_cxx, &g_generator);

  // Return the sample in the C format.
  VectorIntToC(sample_cxx, sample);
}

void CatamariOpenMPSampleNonHermitianDPPComplexFloat(
    CatamariInt tile_size, CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexFloat* matrix, CatamariBufferInt* sample) {
  // Convert the view into C++.
  auto matrix_cxx = BlasMatrixViewToCxx(matrix);

  // Sample the DPP.
  std::vector<catamari::Int> sample_cxx;
  #pragma omp parallel
  #pragma omp single
  sample_cxx = catamari::OpenMPSampleNonHermitianDPP(
      tile_size, block_size, maximum_likelihood, &matrix_cxx, &g_generator);

  // Return the sample in the C format.
  VectorIntToC(sample_cxx, sample);
}

void CatamariOpenMPSampleNonHermitianDPPComplexDouble(
    CatamariInt tile_size, CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexDouble* matrix, CatamariBufferInt* sample) {
  // Convert the view into C++.
  auto matrix_cxx = BlasMatrixViewToCxx(matrix);

  // Sample the DPP.
  std::vector<catamari::Int> sample_cxx;
  #pragma omp parallel
  #pragma omp single
  sample_cxx = catamari::OpenMPSampleNonHermitianDPP(
      tile_size, block_size, maximum_likelihood, &matrix_cxx, &g_generator);

  // Return the sample in the C format.
  VectorIntToC(sample_cxx, sample);
}
#endif  // ifdef CATAMARI_OPENMP

void CatamariDPPLogLikelihoodFloat(CatamariBlasMatrixViewFloat* matrix,
                                   float* log_likelihood) {
  auto matrix_cxx = BlasMatrixViewToCxx(matrix).ToConst();
  *log_likelihood = catamari::DPPLogLikelihood(matrix_cxx);
}

void CatamariDPPLogLikelihoodDouble(CatamariBlasMatrixViewDouble* matrix,
                                    double* log_likelihood) {
  auto matrix_cxx = BlasMatrixViewToCxx(matrix).ToConst();
  *log_likelihood = catamari::DPPLogLikelihood(matrix_cxx);
}

void CatamariDPPLogLikelihoodComplexFloat(
    CatamariBlasMatrixViewComplexFloat* matrix, float* log_likelihood) {
  auto matrix_cxx = BlasMatrixViewToCxx(matrix).ToConst();
  *log_likelihood = catamari::DPPLogLikelihood(matrix_cxx);
}

void CatamariDPPLogLikelihoodComplexDouble(
    CatamariBlasMatrixViewComplexDouble* matrix, double* log_likelihood) {
  auto matrix_cxx = BlasMatrixViewToCxx(matrix).ToConst();
  *log_likelihood = catamari::DPPLogLikelihood(matrix_cxx);
}

void CatamariSampleElementaryLowerHermitianDPPFloat(
    CatamariInt block_size, CatamariInt rank, bool maximum_likelihood,
    CatamariBlasMatrixViewFloat* matrix, CatamariBufferInt* sample) {
  // Convert the view into C++.
  auto matrix_cxx = BlasMatrixViewToCxx(matrix);

  // Sample the elementary DPP.
  const std::vector<catamari::Int> sample_cxx =
      catamari::SampleElementaryLowerHermitianDPP(
          block_size, rank, maximum_likelihood, &matrix_cxx, &g_generator);

  // Return the sample in the C format.
  VectorIntToC(sample_cxx, sample);
}

void CatamariSampleElementaryLowerHermitianDPPDouble(
    CatamariInt block_size, CatamariInt rank, bool maximum_likelihood,
    CatamariBlasMatrixViewDouble* matrix, CatamariBufferInt* sample) {
  // Convert the view into C++.
  auto matrix_cxx = BlasMatrixViewToCxx(matrix);

  // Sample the elementary DPP.
  const std::vector<catamari::Int> sample_cxx =
      catamari::SampleElementaryLowerHermitianDPP(
          block_size, rank, maximum_likelihood, &matrix_cxx, &g_generator);

  // Return the sample in the C format.
  VectorIntToC(sample_cxx, sample);
}

void CatamariSampleElementaryLowerHermitianDPPComplexFloat(
    CatamariInt block_size, CatamariInt rank, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexFloat* matrix, CatamariBufferInt* sample) {
  // Convert the view into C++.
  auto matrix_cxx = BlasMatrixViewToCxx(matrix);

  // Sample the elementary DPP.
  const std::vector<catamari::Int> sample_cxx =
      catamari::SampleElementaryLowerHermitianDPP(
          block_size, rank, maximum_likelihood, &matrix_cxx, &g_generator);

  // Return the sample in the C format.
  VectorIntToC(sample_cxx, sample);
}

void CatamariSampleElementaryLowerHermitianDPPComplexDouble(
    CatamariInt block_size, CatamariInt rank, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexDouble* matrix, CatamariBufferInt* sample) {
  // Convert the view into C++.
  auto matrix_cxx = BlasMatrixViewToCxx(matrix);

  // Sample the elementary DPP.
  const std::vector<catamari::Int> sample_cxx =
      catamari::SampleElementaryLowerHermitianDPP(
          block_size, rank, maximum_likelihood, &matrix_cxx, &g_generator);

  // Return the sample in the C format.
  VectorIntToC(sample_cxx, sample);
}

void CatamariElementaryDPPLogLikelihoodFloat(
    CatamariInt rank, CatamariBlasMatrixViewFloat* matrix,
    float* log_likelihood) {
  auto matrix_cxx = BlasMatrixViewToCxx(matrix).ToConst();
  *log_likelihood = catamari::ElementaryDPPLogLikelihood(rank, matrix_cxx);
}

void CatamariElementaryDPPLogLikelihoodDouble(
    CatamariInt rank, CatamariBlasMatrixViewDouble* matrix,
    double* log_likelihood) {
  auto matrix_cxx = BlasMatrixViewToCxx(matrix).ToConst();
  *log_likelihood = catamari::ElementaryDPPLogLikelihood(rank, matrix_cxx);
}

void CatamariElementaryDPPLogLikelihoodComplexFloat(
    CatamariInt rank, CatamariBlasMatrixViewComplexFloat* matrix,
    float* log_likelihood) {
  auto matrix_cxx = BlasMatrixViewToCxx(matrix).ToConst();
  *log_likelihood = catamari::ElementaryDPPLogLikelihood(rank, matrix_cxx);
}

void CatamariElementaryDPPLogLikelihoodComplexDouble(
    CatamariInt rank, CatamariBlasMatrixViewComplexDouble* matrix,
    double* log_likelihood) {
  auto matrix_cxx = BlasMatrixViewToCxx(matrix).ToConst();
  *log_likelihood = catamari::ElementaryDPPLogLikelihood(rank, matrix_cxx);
}
