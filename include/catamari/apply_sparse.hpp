/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_APPLY_SPARSE_H_
#define CATAMARI_APPLY_SPARSE_H_

#include "catamari/blas_matrix_view.hpp"
#include "catamari/coordinate_matrix.hpp"
#include "catamari/promote.hpp"

namespace catamari {

// vec1 := alpha matrix vec0 + beta vec1.
template <class Field>
void ApplySparse(const Field& alpha,
                 const CoordinateMatrix<Field>& sparse_matrix,
                 const ConstBlasMatrixView<Field>& input_matrix,
                 const Field& beta, BlasMatrixView<Field>* result);

// A version of
//
//   vec1 := alpha matrix vec0 + beta vec1.
//
// which uses (typically) higher-precision arithmetic.
template <class Field>
void ApplySparse(const Promote<Field>& alpha,
                 const CoordinateMatrix<Field>& sparse_matrix,
                 const ConstBlasMatrixView<Promote<Field>>& input_matrix,
                 const Promote<Field>& beta,
                 BlasMatrixView<Promote<Field>>* result);

// vec1 := alpha matrix^T vec0 + beta vec1.
template <class Field>
void ApplyTransposeSparse(const Field& alpha,
                          const CoordinateMatrix<Field>& sparse_matrix,
                          const ConstBlasMatrixView<Field>& input_matrix,
                          const Field& beta, BlasMatrixView<Field>* result);

// A version of
//
//   vec1 := alpha matrix^T vec0 + beta vec1.
//
// which uses (typically) higher-precision arithmetic.
template <class Field>
void ApplyTransposeSparse(
    const Promote<Field>& alpha, const CoordinateMatrix<Field>& sparse_matrix,
    const ConstBlasMatrixView<Promote<Field>>& input_matrix,
    const Promote<Field>& beta, BlasMatrixView<Promote<Field>>* result);

// vec1 := alpha matrix^H vec0 + beta vec1.
template <class Field>
void ApplyAdjointSparse(const Field& alpha,
                        const CoordinateMatrix<Field>& sparse_matrix,
                        const ConstBlasMatrixView<Field>& input_matrix,
                        const Field& beta, BlasMatrixView<Field>* result);

// A version of
//
//   vec1 := alpha matrix^H vec0 + beta vec1.
//
// which uses (typically) higher-precision arithmetic.
template <class Field>
void ApplyAdjointSparse(const Promote<Field>& alpha,
                        const CoordinateMatrix<Field>& sparse_matrix,
                        const ConstBlasMatrixView<Promote<Field>>& input_matrix,
                        const Promote<Field>& beta,
                        BlasMatrixView<Promote<Field>>* result);

}  // namespace catamari

#include "catamari/apply_sparse-impl.hpp"

#endif  // ifndef CATAMARI_APPLY_SPARSE_H_
