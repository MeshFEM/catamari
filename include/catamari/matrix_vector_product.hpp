/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_MATRIX_VECTOR_PRODUCT_H_
#define CATAMARI_MATRIX_VECTOR_PRODUCT_H_

#include "catamari/coordinate_matrix.hpp"

namespace catamari {

// vec1 := alpha matrix vec0 + beta vec1.
template <class Field>
void MatrixVectorProduct(const Field& alpha,
                         const CoordinateMatrix<Field>& matrix,
                         const std::vector<Field>& vec0, const Field& beta,
                         std::vector<Field>* vec1);

}  // namespace catamari

#include "catamari/matrix_vector_product-impl.hpp"

#endif  // ifndef CATAMARI_MATRIX_VECTOR_PRODUCT_H_
