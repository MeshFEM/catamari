/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_DPP_C_H_
#define CATAMARI_DENSE_DPP_C_H_

#include "catamari/integers.h"
#include "catamari/macros.h"

#include "catamari/blas_matrix_view.h"
#include "catamari/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // ifdef __cplusplus

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
    CatamariInt tile_size, CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewFloat* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariOpenMPSampleLowerHermitianDPPDouble(
    CatamariInt tile_size, CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewDouble* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariOpenMPSampleLowerHermitianDPPComplexFloat(
    CatamariInt tile_size, CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexFloat* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariOpenMPSampleLowerHermitianDPPComplexDouble(
    CatamariInt tile_size, CatamariInt block_size, bool maximum_likelihood,
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
    CatamariInt tile_size, CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewFloat* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariOpenMPSampleNonHermitianDPPDouble(
    CatamariInt tile_size, CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewDouble* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariOpenMPSampleNonHermitianDPPComplexFloat(
    CatamariInt tile_size, CatamariInt block_size, bool maximum_likelihood,
    CatamariBlasMatrixViewComplexFloat* matrix, CatamariBufferInt* sample);

CATAMARI_EXPORT void CatamariOpenMPSampleNonHermitianDPPComplexDouble(
    CatamariInt tile_size, CatamariInt block_size, bool maximum_likelihood,
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

#endif  // ifndef CATAMARI_DENSE_DPP_C_H_
