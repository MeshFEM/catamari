/*
 * Copyright (c) 2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "catamari.h"

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

void CatamariBufferFloatInit(CatamariBufferFloat* buffer) {
  buffer->size = 0;
  buffer->data = nullptr;
}

void CatamariBufferFloatDestroy(CatamariBufferFloat* buffer) {
  buffer->size = 0;
  if (buffer->data) {
    delete[] buffer->data;
    buffer->data = nullptr;
  }
}

void CatamariBufferDoubleInit(CatamariBufferDouble* buffer) {
  buffer->size = 0;
  buffer->data = nullptr;
}

void CatamariBufferDoubleDestroy(CatamariBufferDouble* buffer) {
  buffer->size = 0;
  if (buffer->data) {
    delete[] buffer->data;
    buffer->data = nullptr;
  }
}

void CatamariBufferComplexFloatInit(CatamariBufferComplexFloat* buffer) {
  buffer->size = 0;
  buffer->data = nullptr;
}

void CatamariBufferComplexFloatDestroy(CatamariBufferComplexFloat* buffer) {
  buffer->size = 0;
  if (buffer->data) {
    delete[] buffer->data;
    buffer->data = nullptr;
  }
}

void CatamariBufferComplexDoubleInit(CatamariBufferComplexDouble* buffer) {
  buffer->size = 0;
  buffer->data = nullptr;
}

void CatamariBufferComplexDoubleDestroy(CatamariBufferComplexDouble* buffer) {
  buffer->size = 0;
  if (buffer->data) {
    delete[] buffer->data;
    buffer->data = nullptr;
  }
}
