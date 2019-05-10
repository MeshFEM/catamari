/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_BLAS_MATRIX_VIEW_C_H_
#define CATAMARI_BLAS_MATRIX_VIEW_C_H_

#include "catamari/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // ifdef __cplusplus

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

#ifdef __cplusplus
}  // extern "C"
#endif  // ifdef __cplusplus

#endif  // ifndef CATAMARI_BLAS_MATRIX_VIEW_C_H_
