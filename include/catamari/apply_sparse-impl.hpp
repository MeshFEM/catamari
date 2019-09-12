/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_APPLY_SPARSE_IMPL_H_
#define CATAMARI_APPLY_SPARSE_IMPL_H_

#include "catamari/apply_sparse.hpp"
#include "catamari/integers.hpp"
#include "catamari/macros.hpp"

namespace catamari {

template <class Field>
void ApplySparse(const Field& alpha,
                 const CoordinateMatrix<Field>& sparse_matrix,
                 const ConstBlasMatrixView<Field>& input_matrix,
                 const Field& beta, BlasMatrixView<Field>* result) {
  const Int num_rows = sparse_matrix.NumRows();
  const Int num_rhs = input_matrix.width;
  CATAMARI_ASSERT(input_matrix.height == sparse_matrix.NumColumns(),
                  "input_matrix was the incorrect height.");
  CATAMARI_ASSERT(result->height == num_rows,
                  "result was the incorrect height.");
  CATAMARI_ASSERT(input_matrix.width == result->width,
                  "result was the incorrect width.");

  // vec1 += alpha matrix vec0 + beta vec1
  const Buffer<MatrixEntry<Field>>& entries = sparse_matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    for (Int j = 0; j < num_rhs; ++j) {
      result->Entry(row, j) *= beta;
    }

    const Int row_beg = sparse_matrix.RowEntryOffset(row);
    const Int row_end = sparse_matrix.RowEntryOffset(row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      CATAMARI_ASSERT(entry.row == row, "Invalid entry row index.");
      for (Int j = 0; j < num_rhs; ++j) {
        result->Entry(row, j) +=
            alpha * entry.value * input_matrix(entry.column, j);
      }
    }
  }
}

template <class Field>
void ApplySparse(const Promote<Field>& alpha,
                 const CoordinateMatrix<Field>& sparse_matrix,
                 const ConstBlasMatrixView<Promote<Field>>& input_matrix,
                 const Promote<Field>& beta,
                 BlasMatrixView<Promote<Field>>* result) {
  const Int num_rows = sparse_matrix.NumRows();
  const Int num_rhs = input_matrix.width;
  CATAMARI_ASSERT(input_matrix.height == sparse_matrix.NumColumns(),
                  "input_matrix was the incorrect height.");
  CATAMARI_ASSERT(result->height == num_rows,
                  "result was the incorrect height.");
  CATAMARI_ASSERT(input_matrix.width == result->width,
                  "result was the incorrect width.");

  // vec1 += alpha matrix vec0 + beta vec1
  const Buffer<MatrixEntry<Field>>& entries = sparse_matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    for (Int j = 0; j < num_rhs; ++j) {
      result->Entry(row, j) *= beta;
    }

    const Int row_beg = sparse_matrix.RowEntryOffset(row);
    const Int row_end = sparse_matrix.RowEntryOffset(row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      CATAMARI_ASSERT(entry.row == row, "Invalid entry row index.");
      for (Int j = 0; j < num_rhs; ++j) {
        result->Entry(row, j) += alpha * (input_matrix(entry.column, j) *
                                          Promote<Field>(entry.value));
      }
    }
  }
}

template <class Field>
void ApplyTransposeSparse(const Field& alpha,
                          const CoordinateMatrix<Field>& sparse_matrix,
                          const ConstBlasMatrixView<Field>& input_matrix,
                          const Field& beta, BlasMatrixView<Field>* result) {
  const Int num_rows = sparse_matrix.NumRows();
  const Int num_rhs = input_matrix.width;
  CATAMARI_ASSERT(input_matrix.height == num_rows,
                  "input_matrix was the incorrect height.");
  CATAMARI_ASSERT(result->height == sparse_matrix.NumColumns(),
                  "result was the incorrect height.");
  CATAMARI_ASSERT(input_matrix.width == result->width,
                  "result was the incorrect width.");

  // Scale the input by beta.
  if (beta != Field(1)) {
    for (Int j = 0; j < num_rhs; ++j) {
      for (Int i = 0; i < input_matrix.height; ++i) {
        result->Entry(i, j) *= beta;
      }
    }
  }

  // vec1 += alpha matrix vec0 + beta vec1
  const Buffer<MatrixEntry<Field>>& entries = sparse_matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int row_beg = sparse_matrix.RowEntryOffset(row);
    const Int row_end = sparse_matrix.RowEntryOffset(row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      CATAMARI_ASSERT(entry.row == row, "Invalid entry row index.");
      for (Int j = 0; j < num_rhs; ++j) {
        result->Entry(entry.column, j) +=
            alpha * entry.value * input_matrix(row, j);
      }
    }
  }
}

template <class Field>
void ApplyTransposeSparse(
    const Promote<Field>& alpha, const CoordinateMatrix<Field>& sparse_matrix,
    const ConstBlasMatrixView<Promote<Field>>& input_matrix,
    const Promote<Field>& beta, BlasMatrixView<Promote<Field>>* result) {
  const Int num_rows = sparse_matrix.NumRows();
  const Int num_rhs = input_matrix.width;
  CATAMARI_ASSERT(input_matrix.height == num_rows,
                  "input_matrix was the incorrect height.");
  CATAMARI_ASSERT(result->height == sparse_matrix.NumColumns(),
                  "result was the incorrect height.");
  CATAMARI_ASSERT(input_matrix.width == result->width,
                  "result was the incorrect width.");

  // Scale the input by beta.
  if (beta != Field(1)) {
    for (Int j = 0; j < num_rhs; ++j) {
      for (Int i = 0; i < input_matrix.height; ++i) {
        result->Entry(i, j) *= beta;
      }
    }
  }

  // vec1 += alpha matrix vec0 + beta vec1
  const Buffer<MatrixEntry<Field>>& entries = sparse_matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int row_beg = sparse_matrix.RowEntryOffset(row);
    const Int row_end = sparse_matrix.RowEntryOffset(row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      CATAMARI_ASSERT(entry.row == row, "Invalid entry row index.");
      for (Int j = 0; j < num_rhs; ++j) {
        result->Entry(entry.column, j) +=
            alpha * (input_matrix(row, j) * Promote<Field>(entry.value));
      }
    }
  }
}

template <class Field>
void ApplyAdjointSparse(const Field& alpha,
                        const CoordinateMatrix<Field>& sparse_matrix,
                        const ConstBlasMatrixView<Field>& input_matrix,
                        const Field& beta, BlasMatrixView<Field>* result) {
  const Int num_rows = sparse_matrix.NumRows();
  const Int num_rhs = input_matrix.width;
  CATAMARI_ASSERT(input_matrix.height == num_rows,
                  "input_matrix was the incorrect height.");
  CATAMARI_ASSERT(result->height == sparse_matrix.NumColumns(),
                  "result was the incorrect height.");
  CATAMARI_ASSERT(input_matrix.width == result->width,
                  "result was the incorrect width.");

  // Scale the input by beta.
  if (beta != Field(1)) {
    for (Int j = 0; j < num_rhs; ++j) {
      for (Int i = 0; i < input_matrix.height; ++i) {
        result->Entry(i, j) *= beta;
      }
    }
  }

  // vec1 += alpha matrix vec0 + beta vec1
  const Buffer<MatrixEntry<Field>>& entries = sparse_matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int row_beg = sparse_matrix.RowEntryOffset(row);
    const Int row_end = sparse_matrix.RowEntryOffset(row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      CATAMARI_ASSERT(entry.row == row, "Invalid entry row index.");
      const Field value = mantis::Conjugate(entry.value);
      for (Int j = 0; j < num_rhs; ++j) {
        result->Entry(entry.column, j) += alpha * value * input_matrix(row, j);
      }
    }
  }
}

template <class Field>
void ApplyAdjointSparse(const Promote<Field>& alpha,
                        const CoordinateMatrix<Field>& sparse_matrix,
                        const ConstBlasMatrixView<Promote<Field>>& input_matrix,
                        const Promote<Field>& beta,
                        BlasMatrixView<Promote<Field>>* result) {
  const Int num_rows = sparse_matrix.NumRows();
  const Int num_rhs = input_matrix.width;
  CATAMARI_ASSERT(input_matrix.height == num_rows,
                  "input_matrix was the incorrect height.");
  CATAMARI_ASSERT(result->height == sparse_matrix.NumColumns(),
                  "result was the incorrect height.");
  CATAMARI_ASSERT(input_matrix.width == result->width,
                  "result was the incorrect width.");

  // Scale the input by beta.
  if (beta != Field(1)) {
    for (Int j = 0; j < num_rhs; ++j) {
      for (Int i = 0; i < input_matrix.height; ++i) {
        result->Entry(i, j) *= beta;
      }
    }
  }

  // vec1 += alpha matrix vec0 + beta vec1
  const Buffer<MatrixEntry<Field>>& entries = sparse_matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int row_beg = sparse_matrix.RowEntryOffset(row);
    const Int row_end = sparse_matrix.RowEntryOffset(row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      CATAMARI_ASSERT(entry.row == row, "Invalid entry row index.");
      const Promote<Field> value = mantis::Conjugate(entry.value);
      for (Int j = 0; j < num_rhs; ++j) {
        result->Entry(entry.column, j) +=
            alpha * (input_matrix(row, j) * value);
      }
    }
  }
}

}  // namespace catamari

#endif  // ifndef CATAMARI_APPLY_SPARSE_IMPL_H_
