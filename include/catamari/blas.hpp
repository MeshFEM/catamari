/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_BLAS_H_
#define CATAMARI_BLAS_H_

#include "catamari/integers.hpp"

namespace catamari {

template <class Field>
void MatrixMultiplyTransposeNormal(
    Int output_height, Int output_width, Int contraction_size,
    const Field& alpha, const Field* left_matrix, Int left_leading_dim,
    const Field* right_matrix, Int right_leading_dim, const Field& beta,
    Field* output_matrix, Int output_leading_dim);

template <class Field>
void HermitianOuterProductTransposeLower(
    Int output_height, Int contraction_size,
    const Field& alpha, const Field* left_matrix, Int left_leading_dim,
    const Field& beta, Field* output_matrix, Int output_leading_dim);

template <class Field>
void MatrixMultiplyTransposeNormalLower(
    Int output_height, Int contraction_size,
    const Field& alpha, const Field* left_matrix, Int left_leading_dim,
    const Field* right_matrix, Int right_leading_dim, const Field& beta,
    Field* output_matrix, Int output_leading_dim);

template <class Field>
void ConjugateLowerTriangularSolves(
  Int height, Int width, const Field* triangular_matrix, Int triang_leading_dim,
  Field* matrix, Int leading_dim);

template <class Field>
void DiagonalTimesConjugateUnitLowerTriangularSolves(
  Int height, Int width, const Field* triangular_matrix,
  Int triangular_leading_dim, Field* matrix, Int leading_dim);

}  // namespace catamari

#include "catamari/blas-impl.hpp"

#endif  // ifndef CATAMARI_BLAS_H_
