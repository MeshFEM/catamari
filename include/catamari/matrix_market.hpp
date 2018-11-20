/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_MATRIX_MARKET_H_
#define CATAMARI_MATRIX_MARKET_H_

#include <complex>
#include <fstream>

#include "catamari/integers.hpp"
#include "quotient/matrix_market.hpp"

namespace catamari {

using quotient::MatrixMarketDescription;

// Reads a single floating-point value from a line of an array-format Matrix
// Market file.
//
// Returns true if successful.
template <class Real>
bool ReadMatrixMarketArrayValue(const MatrixMarketDescription& description,
                                std::ifstream& file, Real* value);

// Reads a single complex floating-point value from a line of an array-format
// Matrix Market file.
//
// Returns true if successful.
template <class Real>
bool ReadMatrixMarketArrayValue(const MatrixMarketDescription& description,
                                std::ifstream& file, std::complex<Real>* value);

// Reads a single real entry from a line of an coordinate-format Matrix Market
// file.
//
// Returns true if successful.
template <class Real>
bool ReadMatrixMarketCoordinateEntry(const MatrixMarketDescription& description,
                                     std::ifstream& file, Int* row, Int* column,
                                     Real* value);

// Reads a single complex entry from a line of an coordinate-format Matrix
// Market file.
//
// Returns true if successful.
template <class Real>
bool ReadMatrixMarketCoordinateEntry(const MatrixMarketDescription& description,
                                     std::ifstream& file, Int* row, Int* column,
                                     std::complex<Real>* value);

}  // namespace catamari

#include "catamari/matrix_market-impl.hpp"

#endif  // ifndef CATAMARI_MATRIX_MARKET_H_
