/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_BLAS_IMPL_H_
#define CATAMARI_BLAS_IMPL_H_

#include "catamari/blas.hpp"

#ifdef CATAMARI_HAVE_OPENBLAS

// TODO(Jack Poulson): Attempt to support 64-bit BLAS when Int = long long int.
typedef int BlasInt;

#define CATAMARI_HAVE_BLAS
#define BLAS_SYMBOL(name) name ## _

#endif  // ifdef CATAMARI_HAVE_OPENBLAS

#ifdef CATAMARI_HAVE_BLAS
extern "C" {

void BLAS_SYMBOL(ssyr)(
    const char* uplo, const BlasInt* height, const float* alpha,
    const float* vector, const BlasInt* stride, float* matrix,
    const BlasInt* leading_dim);

void BLAS_SYMBOL(dsyr)(
    const char* uplo, const BlasInt* height, const double* alpha,
    const double* vector, const BlasInt* stride, double* matrix,
    const BlasInt* leading_dim);

void BLAS_SYMBOL(dtrsv)(
    const char* uplo, const char* trans, const char* diag,
    const BlasInt* height, const double* triangular_matrix,
    const BlasInt* triang_leading_dim, double* vector, const BlasInt* stride);

void BLAS_SYMBOL(strsv)(
    const char* uplo, const char* trans, const char* diag,
    const BlasInt* height, const float* triangular_matrix,
    const BlasInt* triang_leading_dim, float* vector, const BlasInt* stride);

void BLAS_SYMBOL(dgemm)(
    const char* trans_left, const char* trans_right,
    const BlasInt* output_height, const BlasInt* output_width,
    const BlasInt* contraction_size, const double* alpha,
    const double* left_matrix, const BlasInt* left_leading_dim,
    const double* right_matrix, const BlasInt* right_leading_dim,
    const double* beta, double* output_matrix,
    const BlasInt* output_leading_dim);

void BLAS_SYMBOL(sgemm)(
    const char* trans_left, const char* trans_right,
    const BlasInt* output_height, const BlasInt* output_width,
    const BlasInt* contraction_size, const float* alpha,
    const float* left_matrix, const BlasInt* left_leading_dim,
    const float* right_matrix, const BlasInt* right_leading_dim,
    const float* beta, float* output_matrix, const BlasInt* output_leading_dim);

void BLAS_SYMBOL(dsyrk)(
    const char* uplo, const char* trans, const BlasInt* height,
    const BlasInt* rank, const double* alpha, const double* factor,
    const BlasInt* factor_leading_dim, const double* beta, double* matrix,
    const BlasInt* leading_dim);

void BLAS_SYMBOL(ssyrk)(
    const char* uplo, const char* trans, const BlasInt* height,
    const BlasInt* rank, const float* alpha, const float* factor,
    const BlasInt* factor_leading_dim, const float* beta, float* matrix,
    const BlasInt* leading_dim);

void BLAS_SYMBOL(dtrsm)(
    const char* side, const char* uplo, const char* trans_triang,
    const char* diag, const BlasInt* height, const BlasInt* width,
    const double* alpha, const double* triang_matrix,
    const BlasInt* triang_leading_dim, double* matrix,
    const BlasInt* leading_dim);

void BLAS_SYMBOL(strsm)(
    const char* side, const char* uplo, const char* trans_triang,
    const char* diag, const BlasInt* height, const BlasInt* width,
    const float* alpha, const float* triang_matrix,
    const BlasInt* triang_leading_dim, float* matrix,
    const BlasInt* leading_dim);

}
#endif  // ifdef CATAMARI_HAVE_BLAS

namespace catamari {

template <class Field>
void TriangularSolveLeftLower(
    Int height, const Field* triangular_matrix, Int triang_leading_dim,
    Field* vector) {
  for (Int j = 0; j < height; ++j) {
    const Field* triang_column = &triangular_matrix[j * triang_leading_dim];
    vector[j] /= triang_column[j];
    const Field eta = vector[j];
    for (Int i = j + 1; i < height; ++i) {
      vector[i] -= triang_column[i] * eta;
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void TriangularSolveLeftLower(
    Int height, const float* triangular_matrix, Int triang_leading_dim,
    float* vector) {
  const char uplo = 'L';
  const char trans = 'N';
  const char diag = 'N';
  const BlasInt height_blas = height;
  const BlasInt triang_leading_dim_blas = triang_leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(strsv)(
      &uplo, &trans, &diag, &height_blas, triangular_matrix,
      &triang_leading_dim_blas, vector, &unit_stride_blas);
}

template <>
inline void TriangularSolveLeftLower(
    Int height, const double* triangular_matrix, Int triang_leading_dim,
    double* vector) {
  const char uplo = 'L';
  const char trans = 'N';
  const char diag = 'N';
  const BlasInt height_blas = height;
  const BlasInt triang_leading_dim_blas = triang_leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(dtrsv)(
      &uplo, &trans, &diag, &height_blas, triangular_matrix,
      &triang_leading_dim_blas, vector, &unit_stride_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void TriangularSolveLeftLowerUnit(
    Int height, const Field* triangular_matrix, Int triang_leading_dim,
    Field* vector) {
  for (Int j = 0; j < height; ++j) {
    const Field* triang_column = &triangular_matrix[j * triang_leading_dim];
    const Field eta = vector[j];
    for (Int i = j + 1; i < height; ++i) {
      vector[i] -= triang_column[i] * eta;
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void TriangularSolveLeftLowerUnit(
    Int height, const float* triangular_matrix, Int triang_leading_dim,
    float* vector) {
  const char uplo = 'L';
  const char trans = 'N';
  const char diag = 'U';
  const BlasInt height_blas = height;
  const BlasInt triang_leading_dim_blas = triang_leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(strsv)(
      &uplo, &trans, &diag, &height_blas, triangular_matrix,
      &triang_leading_dim_blas, vector, &unit_stride_blas);
}

template <>
inline void TriangularSolveLeftLowerUnit(
    Int height, const double* triangular_matrix, Int triang_leading_dim,
    double* vector) {
  const char uplo = 'L';
  const char trans = 'N';
  const char diag = 'U';
  const BlasInt height_blas = height;
  const BlasInt triang_leading_dim_blas = triang_leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(dtrsv)(
      &uplo, &trans, &diag, &height_blas, triangular_matrix,
      &triang_leading_dim_blas, vector, &unit_stride_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void TriangularSolveLeftLowerAdjoint(
    Int height, const Field* triangular_matrix, Int triang_leading_dim,
    Field* vector) {
  for (Int j = height - 1; j >= 0; --j) {
    const Field* triang_column = &triangular_matrix[j * triang_leading_dim];
    Field& eta = vector[j];
    for (Int i = j + 1; i < height; ++i) {
      eta -= Conjugate(triang_column[i]) * vector[i];
    }
    eta /= Conjugate(triang_column[j]);
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void TriangularSolveLeftLowerAdjoint(
    Int height, const float* triangular_matrix, Int triang_leading_dim,
    float* vector) {
  const char uplo = 'L';
  const char trans = 'T';
  const char diag = 'N';
  const BlasInt height_blas = height;
  const BlasInt triang_leading_dim_blas = triang_leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(strsv)(
      &uplo, &trans, &diag, &height_blas, triangular_matrix,
      &triang_leading_dim_blas, vector, &unit_stride_blas);
}

template <>
inline void TriangularSolveLeftLowerAdjoint(
    Int height, const double* triangular_matrix, Int triang_leading_dim,
    double* vector) {
  const char uplo = 'L';
  const char trans = 'T';
  const char diag = 'N';
  const BlasInt height_blas = height;
  const BlasInt triang_leading_dim_blas = triang_leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(dtrsv)(
      &uplo, &trans, &diag, &height_blas, triangular_matrix,
      &triang_leading_dim_blas, vector, &unit_stride_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void TriangularSolveLeftLowerAdjointUnit(
    Int height, const Field* triangular_matrix, Int triang_leading_dim,
    Field* vector) {
  for (Int j = height - 1; j >= 0; --j) {
    const Field* triang_column = &triangular_matrix[j * triang_leading_dim];
    Field& eta = vector[j];
    for (Int i = j + 1; i < height; ++i) {
      eta -= Conjugate(triang_column[i]) * vector[i];
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void TriangularSolveLeftLowerAdjointUnit(
    Int height, const float* triangular_matrix, Int triang_leading_dim,
    float* vector) {
  const char uplo = 'L';
  const char trans = 'T';
  const char diag = 'U';
  const BlasInt height_blas = height;
  const BlasInt triang_leading_dim_blas = triang_leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(strsv)(
      &uplo, &trans, &diag, &height_blas, triangular_matrix,
      &triang_leading_dim_blas, vector, &unit_stride_blas);
}

template <>
inline void TriangularSolveLeftLowerAdjointUnit(
    Int height, const double* triangular_matrix, Int triang_leading_dim,
    double* vector) {
  const char uplo = 'L';
  const char trans = 'T';
  const char diag = 'U';
  const BlasInt height_blas = height;
  const BlasInt triang_leading_dim_blas = triang_leading_dim;
  const BlasInt unit_stride_blas = 1;
  BLAS_SYMBOL(dtrsv)(
      &uplo, &trans, &diag, &height_blas, triangular_matrix,
      &triang_leading_dim_blas, vector, &unit_stride_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void MatrixMultiplyTransposeNormal(
    Int output_height, Int output_width, Int contraction_size,
    const Field& alpha, const Field* left_matrix, Int left_leading_dim,
    const Field* right_matrix, Int right_leading_dim, const Field& beta,
    Field* output_matrix, Int output_leading_dim) {
  for (Int j = 0; j < output_width; ++j) {
    for (Int i = 0; i < output_height; ++i) {
      Field& output_entry = output_matrix[i + j * output_leading_dim];
      output_entry *= beta;
      for (Int k = 0; k < contraction_size; ++k) {
        output_entry +=
            alpha * left_matrix[k + i * left_leading_dim] *
            right_matrix[k + j * right_leading_dim];
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void MatrixMultiplyTransposeNormal(
    Int output_height, Int output_width, Int contraction_size,
    const float& alpha, const float* left_matrix, Int left_leading_dim,
    const float* right_matrix, Int right_leading_dim, const float& beta,
    float* output_matrix, Int output_leading_dim) {
  const char trans_left = 'T';
  const char trans_right = 'N';
  const BlasInt output_height_blas = output_height;
  const BlasInt output_width_blas = output_width;
  const BlasInt contraction_size_blas = contraction_size;
  const BlasInt left_leading_dim_blas = left_leading_dim;
  const BlasInt right_leading_dim_blas = right_leading_dim;
  const BlasInt output_leading_dim_blas = output_leading_dim;
  BLAS_SYMBOL(sgemm)(
      &trans_left, &trans_right, &output_height_blas, &output_width_blas,
      &contraction_size_blas, &alpha, left_matrix, &left_leading_dim_blas,
      right_matrix, &right_leading_dim_blas, &beta, output_matrix,
      &output_leading_dim_blas);
}

template <>
inline void MatrixMultiplyTransposeNormal(
    Int output_height, Int output_width, Int contraction_size,
    const double& alpha, const double* left_matrix, Int left_leading_dim,
    const double* right_matrix, Int right_leading_dim, const double& beta,
    double* output_matrix, Int output_leading_dim) {
  const char trans_left = 'T';
  const char trans_right = 'N';
  const BlasInt output_height_blas = output_height;
  const BlasInt output_width_blas = output_width;
  const BlasInt contraction_size_blas = contraction_size;
  const BlasInt left_leading_dim_blas = left_leading_dim;
  const BlasInt right_leading_dim_blas = right_leading_dim;
  const BlasInt output_leading_dim_blas = output_leading_dim;
  BLAS_SYMBOL(dgemm)(
      &trans_left, &trans_right, &output_height_blas, &output_width_blas,
      &contraction_size_blas, &alpha, left_matrix, &left_leading_dim_blas,
      right_matrix, &right_leading_dim_blas, &beta, output_matrix,
      &output_leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void HermitianOuterProductTransposeLower(
    Int output_height, Int contraction_size,
    const Field& alpha, const Field* left_matrix, Int left_leading_dim,
    const Field& beta, Field* output_matrix, Int output_leading_dim) {
  for (Int j = 0; j < output_height; ++j) {
    for (Int i = j; i < output_height; ++i) {
      Field& output_entry = output_matrix[i + j * output_leading_dim];
      output_entry *= beta;
      for (Int k = 0; k < contraction_size; ++k) {
        output_entry +=
            alpha * left_matrix[k + i * left_leading_dim] *
            Conjugate(left_matrix[k + j * left_leading_dim]);
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void HermitianOuterProductTransposeLower(
    Int output_height, Int contraction_size,
    const float& alpha, const float* left_matrix, Int left_leading_dim,
    const float& beta, float* output_matrix, Int output_leading_dim) {
  const char uplo = 'L';
  const char trans = 'T';
  const BlasInt height_blas = output_height;
  const BlasInt rank_blas = contraction_size;
  const BlasInt factor_leading_dim_blas = left_leading_dim;
  const BlasInt leading_dim_blas = output_leading_dim;
  BLAS_SYMBOL(ssyrk)(
      &uplo, &trans, &height_blas, &rank_blas, &alpha, left_matrix,
      &factor_leading_dim_blas, &beta, output_matrix, &leading_dim_blas);
}

template <>
inline void HermitianOuterProductTransposeLower(
    Int output_height, Int contraction_size,
    const double& alpha, const double* left_matrix, Int left_leading_dim,
    const double& beta, double* output_matrix, Int output_leading_dim) {
  const char uplo = 'L';
  const char trans = 'T';
  const BlasInt height_blas = output_height;
  const BlasInt rank_blas = contraction_size;
  const BlasInt factor_leading_dim_blas = left_leading_dim;
  const BlasInt leading_dim_blas = output_leading_dim;
  BLAS_SYMBOL(dsyrk)(
      &uplo, &trans, &height_blas, &rank_blas, &alpha, left_matrix,
      &factor_leading_dim_blas, &beta, output_matrix, &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void MatrixMultiplyTransposeNormalLower(
    Int output_height, Int contraction_size,
    const Field& alpha, const Field* left_matrix, Int left_leading_dim,
    const Field* right_matrix, Int right_leading_dim, const Field& beta,
    Field* output_matrix, Int output_leading_dim) {
  for (Int j = 0; j < output_height; ++j) {
    for (Int i = j; i < output_height; ++i) {
      Field& output_entry = output_matrix[i + j * output_leading_dim];
      output_entry *= beta;
      for (Int k = 0; k < contraction_size; ++k) {
        output_entry +=
            alpha * left_matrix[k + i * left_leading_dim] *
            right_matrix[k + j * right_leading_dim];
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void MatrixMultiplyTransposeNormalLower(
    Int output_height, Int contraction_size,
    const float& alpha, const float* left_matrix, Int left_leading_dim,
    const float* right_matrix, Int right_leading_dim, const float& beta,
    float* output_matrix, Int output_leading_dim) {
  // TODO(Jack Poulson): Save a factor of two if we have MKL via GEMMT.
  MatrixMultiplyTransposeNormal(
      output_height, output_height, contraction_size, alpha, left_matrix,
      left_leading_dim, right_matrix, right_leading_dim, beta, output_matrix,
      output_leading_dim);

  // Explicitly zero out the strictly-upper triangle of C.
  for (Int j = 1; j < output_height; ++j) {
    for (Int i = 0; i < j; ++i) {
      output_matrix[i + j * output_leading_dim] = 0;
    }
  }
}

template <>
inline void MatrixMultiplyTransposeNormalLower(
    Int output_height, Int contraction_size,
    const double& alpha, const double* left_matrix, Int left_leading_dim,
    const double* right_matrix, Int right_leading_dim, const double& beta,
    double* output_matrix, Int output_leading_dim) {
  // TODO(Jack Poulson): Save a factor of two if we have MKL via GEMMT.
  MatrixMultiplyTransposeNormal(
      output_height, output_height, contraction_size, alpha, left_matrix,
      left_leading_dim, right_matrix, right_leading_dim, beta, output_matrix,
      output_leading_dim);

  // Explicitly zero out the strictly-upper triangle of C.
  for (Int j = 1; j < output_height; ++j) {
    for (Int i = 0; i < j; ++i) {
      output_matrix[i + j * output_leading_dim] = 0;
    }
  }
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void ConjugateLowerTriangularSolves(
  Int height, Int width, const Field* triangular_matrix, Int triang_leading_dim,
  Field* matrix, Int leading_dim) {
  for (Int j = 0; j < width; ++j) {
    Field* input_column = &matrix[j * leading_dim];

    // Interleave with a subsequent solve against D.
    for (Int i = 0; i < height; ++i) {
      const Field* l_column = &triangular_matrix[i * triang_leading_dim];
      input_column[i] /= Conjugate(l_column[i]);
      for (Int k = i + 1; k < height; ++k) {
        input_column[k] -= Conjugate(l_column[k]) * input_column[i];
      }
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void ConjugateLowerTriangularSolves(
  Int height, Int width, const float* triangular_matrix, Int triang_leading_dim,
  float* matrix, Int leading_dim) {
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'N';
  const char diag = 'N';
  const BlasInt height_blas = height;
  const BlasInt width_blas = width;
  const float alpha = 1.f;
  const BlasInt triang_leading_dim_blas = triang_leading_dim;
  const BlasInt leading_dim_blas = leading_dim;

  BLAS_SYMBOL(strsm)(
      &side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
      triangular_matrix, &triang_leading_dim_blas, matrix, &leading_dim_blas);
}

template <>
inline void ConjugateLowerTriangularSolves(
  Int height, Int width, const double* triangular_matrix,
  Int triang_leading_dim, double* matrix, Int leading_dim) {
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'N';
  const char diag = 'N';
  const BlasInt height_blas = height;
  const BlasInt width_blas = width;
  const double alpha = 1.f;
  const BlasInt triang_leading_dim_blas = triang_leading_dim;
  const BlasInt leading_dim_blas = leading_dim;

  BLAS_SYMBOL(dtrsm)(
      &side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
      triangular_matrix, &triang_leading_dim_blas, matrix, &leading_dim_blas);
}
#endif  // ifdef CATAMARI_HAVE_BLAS

template <class Field>
void DiagonalTimesConjugateUnitLowerTriangularSolves(
  Int height, Int width, const Field* triangular_matrix, Int triang_leading_dim,
  Field* matrix, Int leading_dim) {
  for (Int j = 0; j < width; ++j) {
    Field* input_column = &matrix[j * leading_dim];

    // Interleave with a subsequent solve against D.
    for (Int i = 0; i < height; ++i) {
      const Field* l_column = &triangular_matrix[i * triang_leading_dim];
      for (Int k = i + 1; k < height; ++k) {
        input_column[k] -= Conjugate(l_column[k]) * input_column[i];
      }
      input_column[i] /= l_column[i];
    }
  }
}

#ifdef CATAMARI_HAVE_BLAS
template <>
inline void DiagonalTimesConjugateUnitLowerTriangularSolves(
  Int height, Int width, const float* triangular_matrix, Int triang_leading_dim,
  float* matrix, Int leading_dim) {
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'N';
  const char diag = 'U';
  const BlasInt height_blas = height;
  const BlasInt width_blas = width;
  const float alpha = 1.f;
  const BlasInt triang_leading_dim_blas = triang_leading_dim;
  const BlasInt leading_dim_blas = leading_dim;

  BLAS_SYMBOL(strsm)(
      &side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
      triangular_matrix, &triang_leading_dim_blas, matrix, &leading_dim_blas);

  // Solve against the diagonal.
  for (Int j = 0; j < width; ++j) {
    for (Int i = 0; i < height; ++i) {
      matrix[i + j * leading_dim] /=
          triangular_matrix[i + i * triang_leading_dim];
    }
  }
}

template <>
inline void DiagonalTimesConjugateUnitLowerTriangularSolves(
  Int height, Int width, const double* triangular_matrix,
  Int triang_leading_dim,
  double* matrix, Int leading_dim) {
  const char side = 'L';
  const char uplo = 'L';
  const char trans_triang = 'N';
  const char diag = 'U';
  const BlasInt height_blas = height;
  const BlasInt width_blas = width;
  const double alpha = 1.;
  const BlasInt triang_leading_dim_blas = triang_leading_dim;
  const BlasInt leading_dim_blas = leading_dim;

  BLAS_SYMBOL(dtrsm)(
      &side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
      triangular_matrix, &triang_leading_dim_blas, matrix, &leading_dim_blas);

  // Solve against the diagonal.
  for (Int j = 0; j < width; ++j) {
    for (Int i = 0; i < height; ++i) {
      matrix[i + j * leading_dim] /=
          triangular_matrix[i + i * triang_leading_dim];
    }
  }
}
#endif  // ifdef CATAMARI_HAVE_BLAS

}  // namespace catamari

#endif  // ifndef CATAMARI_BLAS_IMPL_H_
