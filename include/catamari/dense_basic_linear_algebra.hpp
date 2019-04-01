/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_BASIC_LINEAR_ALGEBRA_H_
#define CATAMARI_DENSE_BASIC_LINEAR_ALGEBRA_H_

#include "catamari/blas_matrix_view.hpp"
#include "catamari/buffer.hpp"
#include "catamari/complex.hpp"
#include "catamari/integers.hpp"

namespace catamari {

// Conjugates a matrix in-place.
template <class Field>
void ConjugateMatrix(BlasMatrixView<Field>* matrix);

// Performs the operation 'result += alpha matrix input_vector', where
// 'input_vector' and 'result' are vectors (stored with unit stride).
template <class Field>
void MatrixVectorProduct(const Field& alpha,
                         const ConstBlasMatrixView<Field>& matrix,
                         const Field* input_vector, Field* result);

// Performs the operation 'result += alpha conj(matrix) input_vector', where
// 'input_vector' and 'result' are vectors (stored with unit stride).
template <class Field>
void ConjugateMatrixVectorProduct(const Field& alpha,
                                  const ConstBlasMatrixView<Field>& matrix,
                                  const Field* input_vector, Field* result);

// Performs the operation 'result += alpha matrix^T input_vector', where
// 'input_vector' and 'result' are vectors (stored with unit stride).
template <class Field>
void TransposeMatrixVectorProduct(const Field& alpha,
                                  const ConstBlasMatrixView<Field>& matrix,
                                  const Field* input_vector, Field* result);

// Updates 'vector := inv(triangular_matrix) vector', where 'vector' is a
// (unit-stride) vector and 'triangular_matrix' is assumed lower-triangular.
template <class Field>
void TriangularSolveLeftLower(
    const ConstBlasMatrixView<Field>& triangular_matrix, Field* vector);

// Updates 'vector := inv(triangular_matrix) vector', where 'vector' is a
// (unit-stride) vector and 'triangular_matrix' is assumed to be
// lower-triangular with unit diagonal.
template <class Field>
void TriangularSolveLeftLowerUnit(
    const ConstBlasMatrixView<Field>& triangular_matrix, Field* vector);

// Updates 'vector := inv(triangular_matrix)^H vector', where 'vector' is a
// (unit-stride) vector and 'triangular_matrix' is assumed lower-triangular.
template <class Field>
void TriangularSolveLeftLowerAdjoint(
    const ConstBlasMatrixView<Field>& triangular_matrix, Field* vector);

// Updates 'vector := inv(triangular_matrix)^H vector', where 'vector' is a
// (unit-stride) vector and 'triangular_matrix' is assumed lower-triangular
// with unit diagonal.
template <class Field>
void TriangularSolveLeftLowerAdjointUnit(
    const ConstBlasMatrixView<Field>& triangular_matrix, Field* vector);

// Updates
//
//   output_matrix := alpha left_matrix right_matrix + beta output_matrix
//
template <class Field>
void MatrixMultiplyNormalNormal(const Field& alpha,
                                const ConstBlasMatrixView<Field>& left_matrix,
                                const ConstBlasMatrixView<Field>& right_matrix,
                                const Field& beta,
                                BlasMatrixView<Field>* output_matrix);

// Updates
//
//   output_matrix := alpha left_matrix right_matrix^T + beta output_matrix
//
template <class Field>
void MatrixMultiplyNormalTranspose(
    const Field& alpha, const ConstBlasMatrixView<Field>& left_matrix,
    const ConstBlasMatrixView<Field>& right_matrix, const Field& beta,
    BlasMatrixView<Field>* output_matrix);

// Updates
//
//   output_matrix := alpha left_matrix right_matrix^H + beta output_matrix
//
template <class Field>
void MatrixMultiplyNormalAdjoint(const Field& alpha,
                                 const ConstBlasMatrixView<Field>& left_matrix,
                                 const ConstBlasMatrixView<Field>& right_matrix,
                                 const Field& beta,
                                 BlasMatrixView<Field>* output_matrix);

// Updates
//
//   output_matrix := alpha left_matrix^H right_matrix + beta output_matrix
//
template <class Field>
void MatrixMultiplyAdjointNormal(const Field& alpha,
                                 const ConstBlasMatrixView<Field>& left_matrix,
                                 const ConstBlasMatrixView<Field>& right_matrix,
                                 const Field& beta,
                                 BlasMatrixView<Field>* output_matrix);

// Updates
//
//   output_matrix := alpha left_matrix^T right_matrix + beta output_matrix
//
template <class Field>
void MatrixMultiplyTransposeNormal(
    const Field& alpha, const ConstBlasMatrixView<Field>& left_matrix,
    const ConstBlasMatrixView<Field>& right_matrix, const Field& beta,
    BlasMatrixView<Field>* output_matrix);

// Updates the lower triangular of
//
//   output_matrix := alpha left_matrix left_matrix^H + beta output_matrix.
//
template <class Field>
void LowerNormalHermitianOuterProduct(
    const ComplexBase<Field>& alpha,
    const ConstBlasMatrixView<Field>& left_matrix,
    const ComplexBase<Field>& beta, BlasMatrixView<Field>* output_matrix);

// Updates the lower triangular of
//
//   output_matrix := alpha left_matrix right_matrix^T + beta output_matrix.
//
template <class Field>
void MatrixMultiplyLowerNormalTranspose(
    const Field& alpha, const ConstBlasMatrixView<Field>& left_matrix,
    const ConstBlasMatrixView<Field>& right_matrix, const Field& beta,
    BlasMatrixView<Field>* output_matrix);

// Updates the lower triangular of
//
//   output_matrix := alpha left_matrix^T right_matrix + beta output_matrix.
//
template <class Field>
void MatrixMultiplyLowerTransposeNormal(
    const Field& alpha, const ConstBlasMatrixView<Field>& left_matrix,
    const ConstBlasMatrixView<Field>& right_matrix, const Field& beta,
    BlasMatrixView<Field>* output_matrix);

// Updates 'matrix := inv(triangular_matrix) matrix', where 'triangular_matrix'
// is assumed lower-triangular.
template <class Field>
void LeftLowerTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix);

// Updates 'matrix := inv(triangular_matrix) matrix', where 'triangular_matrix'
// is assumed lower-triangular with unit diagonal.
template <class Field>
void LeftLowerUnitTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix);

// Updates 'matrix := inv(triangular_matrix)^H matrix', where
// 'triangular_matrix' is assumed lower-triangular.
template <class Field>
void LeftLowerAdjointTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix);

// Updates 'matrix := inv(triangular_matrix)^H matrix', where
// 'triangular_matrix' is assumed lower-triangular with unit diagonal.
template <class Field>
void LeftLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix);

// Updates 'matrix := inv(triangular_matrix)^T matrix', where
// 'triangular_matrix' is assumed lower-triangular with unit diagonal.
template <class Field>
void LeftLowerTransposeUnitTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix);

// Updates 'matrix := matrix inv(triangular_matrix)^H', where
// 'triangular_matrix' is assumed lower-triangular.
template <class Field>
void RightLowerAdjointTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix);

// Updates 'matrix := matrix inv(triangular_matrix)^H', where
// 'triangular_matrix' is assumed lower-triangular.
template <class Field>
void RightLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix);

// Updates 'matrix := matrix inv(D triangular_matrix)^H', where
// 'triangular_matrix' is assumed lower-triangular with unit diagonal, and 'D'
// is the diagonal matrix stored in the diagonal entries of 'triangular_matrix'.
template <class Field>
void RightDiagonalTimesLowerAdjointUnitTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix);

// Updates 'matrix := matrix inv(triangular_matrix)^T', where
// 'triangular_matrix' is assumed lower-triangular.
template <class Field>
void RightLowerTransposeTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix);

// Updates 'matrix := matrix inv(triangular_matrix)^T', where
// 'triangular_matrix' is assumed lower-triangular with unit diagonal.
template <class Field>
void RightLowerTransposeUnitTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix);

// Updates 'matrix := matrix inv(D triangular_matrix)^T', where
// 'triangular_matrix' is assumed lower-triangular with unit diagonal, and 'D'
// is the diagonal matrix stored in the diagonal entries of 'triangular_matrix'.
template <class Field>
void RightDiagonalTimesLowerTransposeUnitTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix);

// Updates 'matrix := matrix inv(triangular_matrix)', where
// 'triangular_matrix' is assumed upper-triangular.
template <class Field>
void RightUpperTriangularSolves(
    const ConstBlasMatrixView<Field>& triangular_matrix,
    BlasMatrixView<Field>* matrix);

#ifdef CATAMARI_OPENMP
// Performs a multi-threaded MatrixMultiplyLowerNormalNormal using OpenMP tasks.
// In most circumstances, it should be surrounded by:
//     #pragma parallel
//     #pragma taskgroup
template <class Field>
void OpenMPMatrixMultiplyLowerNormalNormal(
    Int tile_size, const Field& alpha,
    const ConstBlasMatrixView<Field>& left_matrix,
    const ConstBlasMatrixView<Field>& right_matrix, const Field& beta,
    BlasMatrixView<Field>* output_matrix);

// Performs a multi-threaded lower-triangular GEMM using OpenMP tasks.
// In most circumstances, it should be surrounded by:
//     #pragma parallel
//     #pragma taskgroup
template <class Field>
void OpenMPLowerNormalHermitianOuterProduct(
    Int tile_size, const ComplexBase<Field>& alpha,
    const ConstBlasMatrixView<Field>& left_matrix,
    const ComplexBase<Field>& beta, BlasMatrixView<Field>* output_matrix);
#endif  // ifdef CATAMARI_OPENMP

// Applies a row permutation to a dense matrix.
template <class Field>
void Permute(const Buffer<Int>& permutation, BlasMatrixView<Field>* matrix);

}  // namespace catamari

#include "catamari/dense_basic_linear_algebra-impl.hpp"

#endif  // ifndef CATAMARI_DENSE_BASIC_LINEAR_ALGEBRA_H_
