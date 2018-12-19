/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_APPLY_SPARSE_H_
#define CATAMARI_APPLY_SPARSE_H_

#include "catamari/blas_matrix.hpp"
#include "catamari/coordinate_matrix.hpp"

namespace catamari {

// vec1 := alpha matrix vec0 + beta vec1.
template <class Field>
void ApplySparse(const Field& alpha,
                 const CoordinateMatrix<Field>& sparse_matrix,
                 const ConstBlasMatrix<Field>& input_matrix, const Field& beta,
                 BlasMatrix<Field>* result);

}  // namespace catamari

#include "catamari/apply_sparse-impl.hpp"

#endif  // ifndef CATAMARI_APPLY_SPARSE_H_
