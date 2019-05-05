/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_C_H_
#define CATAMARI_C_H_

#include "catamari.hpp"

#if defined _WIN32 || defined __CYGWIN__
#ifdef __GNUC__  // Compling with GNU on Windows.
#define CATAMARI_EXPORT __attribute__((dllexport))
#else
#define CATAMARI_EXPORT __declspec(dllexport)
#endif  // ifdef __GNUC__
#define CATAMARI_LOCAL
#else
#if __GNUC__ >= 4
#define CATAMARI_EXPORT __attribute__((visibility("default")))
#define CATAMARI_LOCAL __attribute__((visibility("hidden")))
#else
#define CATAMARI_EXPORT
#define CATAMARI_LOCAL
#endif  // __GNUC__ >= 4
#endif  // if defined _WIN32 || defined __CYGWIN__

#ifdef __cplusplus
extern "C" {
#endif  // ifdef __cplusplus

typedef catamari::Int CatamariInt;

struct CatamariBlasMatrixViewFloat {
  CatamariInt height;
  CatamariInt width;
  CatamariInt leading_dim;
  float* data;
};

struct CatamariBlasMatrixViewDouble {
  CatamariInt height;
  CatamariInt width;
  CatamariInt leading_dim;
  double* data;
};

struct CatamariBlasMatrixViewComplexFloat {
  CatamariInt height;
  CatamariInt width;
  CatamariInt leading_dim;
  BlasComplexFloat* data;
};

struct CatamariBlasMatrixViewComplexDouble {
  CatamariInt height;
  CatamariInt width;
  CatamariInt leading_dim;
  BlasComplexDouble* data;
};

struct CatamariBufferInt {
  CatamariInt size;
  CatamariInt* data;
};

CATAMARI_EXPORT void CatamariBufferIntInit(CatamariBufferInt* buffer);
CATAMARI_EXPORT void CatamariBufferIntDestroy(CatamariBufferInt* buffer);

CATAMARI_EXPORT void CatamariInitGenerator(unsigned int random_seed);

CATAMARI_EXPORT void CatamariSampleLowerHermitianDPPFloat(
    CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewFloat* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariSampleLowerHermitianDPPDouble(
    CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewDouble* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariSampleLowerHermitianDPPComplexFloat(
    CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexFloat* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariSampleLowerHermitianDPPComplexDouble(
    CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexDouble* matrix, CatamariBufferInt* sample);

#ifdef CATAMARI_OPENMP
CATAMARI_EXPORT void CatamariOpenMPSampleLowerHermitianDPPFloat(
    CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewFloat* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariOpenMPSampleLowerHermitianDPPDouble(
    CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewDouble* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariOpenMPSampleLowerHermitianDPPComplexFloat(
    CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexFloat* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariOpenMPSampleLowerHermitianDPPComplexDouble(
    CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexDouble* matrix, CatamariBufferInt* sample);
#endif  // ifdef CATAMARI_OPENMP

CATAMARI_EXPORT void CatamariSampleNonHermitianDPPFloat(
    CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewFloat* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariSampleNonHermitianDPPDouble(
    CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewDouble* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariSampleNonHermitianDPPComplexFloat(
    CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexFloat* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariSampleNonHermitianDPPComplexDouble(
    CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexDouble* matrix, CatamariBufferInt* sample);

#ifdef CATAMARI_OPENMP
CATAMARI_EXPORT void CatamariOpenMPSampleNonHermitianDPPFloat(
    CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewFloat* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariOpenMPSampleNonHermitianDPPDouble(
    CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewDouble* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariOpenMPSampleNonHermitianDPPComplexFloat(
    CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexFloat* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariOpenMPSampleNonHermitianDPPComplexDouble(
    CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexDouble* matrix, CatamariBufferInt* sample);
#endif  // ifdef CATAMARI_OPENMP

CATAMARI_EXPORT void CatamariDPPLogLikelihoodFloat(
    CatamariBlasMatrixViewFloat* matrix, float* log_likelihood);

CATAMARI_EXPORT void CatamariDPPLogLikelihoodDouble(
    CatamariBlasMatrixViewDouble* matrix, double* log_likelihood);

CATAMARI_EXPORT void CatamariDPPLogLikelihoodComplexFloat(
    CatamariBlasMatrixViewComplexFloat* matrix, float* log_likelihood);

CATAMARI_EXPORT void CatamariDPPLogLikelihoodComplexDouble(
    CatamariBlasMatrixViewComplexDouble* matrix, double* log_likelihood);

CATAMARI_EXPORT void CatamariSampleElementaryLowerHermitianDPPFloat(
    CatamariInt block_size, CatamariInt rank, bool maximum_likelihood,
    CatamariBlasMatrixViewFloat* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariSampleElementaryLowerHermitianDPPDouble(
    CatamariInt block_size, CatamariInt rank, bool maximum_likelihood,
    CatamariBlasMatrixViewDouble* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariSampleElementaryLowerHermitianDPPComplexFloat(
    CatamariInt block_size, CatamariInt rank, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexFloat* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariSampleElementaryLowerHermitianDPPComplexDouble(
    CatamariInt block_size, CatamariInt rank, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexDouble* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariElementaryDPPLogLikelihoodFloat(
    CatamariInt rank, CatamariBlasMatrixViewFloat* matrix,
    float* log_likelihood);

CATAMARI_EXPORT void CatamariElementaryDPPLogLikelihoodDouble(
    CatamariInt rank, CatamariBlasMatrixViewDouble* matrix,
    double* log_likelihood);

CATAMARI_EXPORT void CatamariElementaryDPPLogLikelihoodComplexFloat(
    CatamariInt rank, CatamariBlasMatrixViewComplexFloat* matrix,
    float* log_likelihood);

CATAMARI_EXPORT void CatamariElementaryDPPLogLikelihoodComplexDouble(
    CatamariInt rank, CatamariBlasMatrixViewComplexDouble* matrix,
    double* log_likelihood);

#ifdef __cplusplus
}  // extern "C"
#endif  // ifdef __cplusplus

#endif  // ifndef CATAMARI_C_H_
