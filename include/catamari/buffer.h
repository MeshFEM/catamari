/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_BUFFER_C_H_
#define CATAMARI_BUFFER_C_H_

#include "catamari/blas.hpp"
#include "catamari/integers.hpp"

#ifdef __cplusplus
extern "C" {
#endif  // ifdef __cplusplus

struct CatamariBufferInt {
  CatamariInt size;
  CatamariInt* data;
};

struct CatamariBufferFloat {
  CatamariInt size;
  float* data;
};

struct CatamariBufferDouble {
  CatamariInt size;
  double* data;
};

struct CatamariBufferComplexFloat {
  CatamariInt size;
  BlasComplexFloat* data;
};

struct CatamariBufferComplexDouble {
  CatamariInt size;
  BlasComplexDouble* data;
};

CATAMARI_EXPORT void CatamariBufferIntInit(CatamariBufferInt* buffer);
CATAMARI_EXPORT void CatamariBufferIntDestroy(CatamariBufferInt* buffer);

CATAMARI_EXPORT void CatamariBufferFloatInit(CatamariBufferFloat* buffer);
CATAMARI_EXPORT void CatamariBufferFloatDestroy(CatamariBufferFloat* buffer);

CATAMARI_EXPORT void CatamariBufferDoubleInit(CatamariBufferDouble* buffer);
CATAMARI_EXPORT void CatamariBufferDoubleDestroy(CatamariBufferDouble* buffer);

CATAMARI_EXPORT void CatamariBufferComplexFloatInit(
    CatamariBufferComplexFloat* buffer);
CATAMARI_EXPORT void CatamariBufferComplexFloatDestroy(
    CatamariBufferComplexFloat* buffer);

CATAMARI_EXPORT void CatamariBufferComplexDoubleInit(
    CatamariBufferComplexDouble* buffer);
CATAMARI_EXPORT void CatamariBufferComplexDoubleDestroy(
    CatamariBufferComplexDouble* buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // ifdef __cplusplus

#endif  // ifndef CATAMARI_BUFFER_C_H_
