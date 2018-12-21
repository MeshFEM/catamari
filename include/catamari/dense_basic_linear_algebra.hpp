/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_BASIC_LINEAR_ALGEBRA_H_
#define CATAMARI_DENSE_BASIC_LINEAR_ALGEBRA_H_

#include "catamari/blas_matrix.hpp"
#include "catamari/integers.hpp"

namespace catamari {

template <class Field>
void ConjugateMatrix(BlasMatrix<Field>* matrix);

template <class Field>
void MatrixVectorProduct(const Field& alpha,
                         const ConstBlasMatrix<Field>& matrix,
                         const Field* input_vector, Field* result);

template <class Field>
void ConjugateMatrixVectorProduct(const Field& alpha,
                                  const ConstBlasMatrix<Field>& matrix,
                                  const Field* input_vector, Field* result);

template <class Field>
void TransposeMatrixVectorProduct(const Field& alpha,
                                  const ConstBlasMatrix<Field>& matrix,
                                  const Field* input_vector, Field* result);

template <class Field>
void TriangularSolveLeftLower(const ConstBlasMatrix<Field>& triangular_matrix,
                              Field* vector);

template <class Field>
void TriangularSolveLeftLowerUnit(
    const ConstBlasMatrix<Field>& triangular_matrix, Field* vector);

template <class Field>
void TriangularSolveLeftLowerAdjoint(
    const ConstBlasMatrix<Field>& triangular_matrix, Field* vector);

template <class Field>
void TriangularSolveLeftLowerAdjointUnit(
    const ConstBlasMatrix<Field>& triangular_matrix, Field* vector);

template <class Field>
void MatrixMultiplyNormalNormal(const Field& alpha,
                                const ConstBlasMatrix<Field>& left_matrix,
                                const ConstBlasMatrix<Field>& right_matrix,
                                const Field& beta,
                                BlasMatrix<Field>* output_matrix);

template <class Field>
void MatrixMultiplyAdjointNormal(const Field& alpha,
                                 const ConstBlasMatrix<Field>& left_matrix,
                                 const ConstBlasMatrix<Field>& right_matrix,
                                 const Field& beta,
                                 BlasMatrix<Field>* output_matrix);

template <class Field>
void MatrixMultiplyTransposeNormal(const Field& alpha,
                                   const ConstBlasMatrix<Field>& left_matrix,
                                   const ConstBlasMatrix<Field>& right_matrix,
                                   const Field& beta,
                                   BlasMatrix<Field>* output_matrix);

template <class Field>
void LowerNormalHermitianOuterProduct(const Field& alpha,
                                      const ConstBlasMatrix<Field>& left_matrix,
                                      const Field& beta,
                                      BlasMatrix<Field>* output_matrix);

template <class Field>
void LowerTransposeHermitianOuterProduct(
    const Field& alpha, const ConstBlasMatrix<Field>& left_matrix,
    const Field& beta, BlasMatrix<Field>* output_matrix);

template <class Field>
void MatrixMultiplyLowerNormalTranspose(
    const Field& alpha, const ConstBlasMatrix<Field>& left_matrix,
    const ConstBlasMatrix<Field>& right_matrix, const Field& beta,
    BlasMatrix<Field>* output_matrix);

template <class Field>
void MatrixMultiplyLowerTransposeNormal(
    const Field& alpha, const ConstBlasMatrix<Field>& left_matrix,
    const ConstBlasMatrix<Field>& right_matrix, const Field& beta,
    BlasMatrix<Field>* output_matrix);

template <class Field>
void LeftLowerTriangularSolves(const ConstBlasMatrix<Field>& triangular_matrix,
                               BlasMatrix<Field>* matrix);

template <class Field>
void LeftLowerUnitTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix, BlasMatrix<Field>* matrix);

template <class Field>
void LeftLowerAdjointTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix, BlasMatrix<Field>* matrix);

template <class Field>
void LeftLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix, BlasMatrix<Field>* matrix);

template <class Field>
void RightLowerAdjointTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix, BlasMatrix<Field>* matrix);

template <class Field>
void RightLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix, BlasMatrix<Field>* matrix);

template <class Field>
void RightDiagonalTimesLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrix<Field>& triangular_matrix, BlasMatrix<Field>* matrix);

}  // namespace catamari

#include "catamari/dense_basic_linear_algebra-impl.hpp"

#endif  // ifndef CATAMARI_DENSE_BASIC_LINEAR_ALGEBRA_H_
