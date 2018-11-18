/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_MATRIX_MARKET_IMPL_H_
#define CATAMARI_MATRIX_MARKET_IMPL_H_

#include "catamari/matrix_market.hpp"

namespace catamari {

template<class Real>
bool ReadMatrixMarketArrayValue(
    const MatrixMarketDescription& description,
    std::ifstream& file,
    Real* value) {
  return quotient::ReadMatrixMarketArrayRealValue(description, file, value);
}

template<typename Real>
bool ReadMatrixMarketArrayValue(
    const MatrixMarketDescription& description,
    std::ifstream& file,
    std::complex<Real>* value) {
  Real real_value, imag_value;
  bool status = quotient::ReadMatrixMarketArrayComplexValue(
      description, file, &real_value, &imag_value);
  *value = std::complex<Real>{real_value, imag_value};
  return status;
}

template<class Real>
bool ReadMatrixMarketCoordinateEntry(
    const MatrixMarketDescription& description,
    std::ifstream& file,
    Int* row,
    Int* column,
    Real* value) {
  return quotient::ReadMatrixMarketCoordinateRealEntry(
      description, file, row, column, value);
}

template<class Real>
bool ReadMatrixMarketCoordinateEntry(
    const MatrixMarketDescription& description,
    std::ifstream& file,
    Int* row,
    Int* column,
    std::complex<Real>* value) {
  Real real_value, imag_value;
  bool status = quotient::ReadMatrixMarketCoordinateComplexEntry(
      description, file, row, column, &real_value, &imag_value);
  *value = std::complex<Real>{real_value, imag_value};
  return status;
}

} // namespace catamari

#endif // ifndef CATAMARI_MATRIX_MARKET_IMPL_H_
