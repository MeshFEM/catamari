/*
 * Copyright (c) 2018-2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SCALAR_LDL_FACTORIZATION_SOLVE_IMPL_H_
#define CATAMARI_LDL_SCALAR_LDL_FACTORIZATION_SOLVE_IMPL_H_

#include <cmath>

#include "catamari/index_utils.hpp"
#include "catamari/ldl/scalar_ldl/scalar_utils.hpp"
#include "quotient/io_utils.hpp"

#include "catamari/ldl/scalar_ldl/factorization.hpp"

namespace catamari {
namespace scalar_ldl {

template <class Field>
void Factorization<Field>::Solve(
    BlasMatrixView<Field>* right_hand_sides) const {
  const bool have_permutation = !ordering.permutation.Empty();

  // Reorder the input into the relaxation permutation of the factorization.
  if (have_permutation) {
    Permute(ordering.permutation, right_hand_sides);
  }

  LowerTriangularSolve(right_hand_sides);
  DiagonalSolve(right_hand_sides);
  LowerTransposeTriangularSolve(right_hand_sides);

  // Reverse the factorization relxation permutation.
  if (have_permutation) {
    Permute(ordering.inverse_permutation, right_hand_sides);
  }
}

template <class Field>
void Factorization<Field>::LowerTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  const Int num_rhs = right_hand_sides->width;
  const LowerStructure& lower_structure = lower_factor.structure;
  const Int num_rows = lower_structure.column_offsets.Size() - 1;
  const bool is_cholesky = factorization_type == kCholeskyFactorization;

  CATAMARI_ASSERT(right_hand_sides->height == num_rows,
                  "matrix was an incorrect height.");

  for (Int column = 0; column < num_rows; ++column) {
    if (is_cholesky) {
      const Field delta = diagonal_factor.values[column];
      for (Int j = 0; j < num_rhs; ++j) {
        right_hand_sides->Entry(column, j) /= delta;
      }
    }

    const Int factor_column_beg = lower_structure.ColumnOffset(column);
    const Int factor_column_end = lower_structure.ColumnOffset(column + 1);
    for (Int j = 0; j < num_rhs; ++j) {
      const Field eta = right_hand_sides->Entry(column, j);
      for (Int index = factor_column_beg; index < factor_column_end; ++index) {
        const Int i = lower_structure.indices[index];
        const Field& value = lower_factor.values[index];
        right_hand_sides->Entry(i, j) -= value * eta;
      }
    }
  }
}

template <class Field>
void Factorization<Field>::DiagonalSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  if (factorization_type == kCholeskyFactorization) {
    return;
  }

  const Int num_rhs = right_hand_sides->width;
  const Int num_rows = diagonal_factor.values.Size();

  CATAMARI_ASSERT(right_hand_sides->height == num_rows,
                  "matrix was an incorrect height.");

  for (Int j = 0; j < num_rhs; ++j) {
    for (Int column = 0; column < num_rows; ++column) {
      right_hand_sides->Entry(column, j) /= diagonal_factor.values[column];
    }
  }
}

template <class Field>
void Factorization<Field>::LowerTransposeTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  const Int num_rhs = right_hand_sides->width;
  const LowerStructure& lower_structure = lower_factor.structure;
  const Int num_rows = lower_structure.column_offsets.Size() - 1;
  const bool is_cholesky = factorization_type == kCholeskyFactorization;
  const bool is_selfadjoint = factorization_type != kLDLTransposeFactorization;

  CATAMARI_ASSERT(right_hand_sides->height == num_rows,
                  "matrix was an incorrect height.");

  for (Int column = num_rows - 1; column >= 0; --column) {
    const Int factor_column_beg = lower_structure.ColumnOffset(column);
    const Int factor_column_end = lower_structure.ColumnOffset(column + 1);
    for (Int j = 0; j < num_rhs; ++j) {
      Field& eta = right_hand_sides->Entry(column, j);
      for (Int index = factor_column_beg; index < factor_column_end; ++index) {
        const Int i = lower_structure.indices[index];
        const Field& value = lower_factor.values[index];
        if (is_selfadjoint) {
          eta -= Conjugate(value) * right_hand_sides->Entry(i, j);
        } else {
          eta -= value * right_hand_sides->Entry(i, j);
        }
      }
    }

    if (is_cholesky) {
      const Field delta = diagonal_factor.values[column];
      for (Int j = 0; j < num_rhs; ++j) {
        right_hand_sides->Entry(column, j) /= delta;
      }
    }
  }
}

}  // namespace scalar_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_LDL_SCALAR_LDL_FACTORIZATION_SOLVE_IMPL_H_
