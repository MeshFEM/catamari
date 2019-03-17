/*
 * Copyright (c) 2018-2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SCALAR_LDL_FACTORIZATION_LEFT_LOOKING_IMPL_H_
#define CATAMARI_LDL_SCALAR_LDL_FACTORIZATION_LEFT_LOOKING_IMPL_H_

#include <cmath>

#include "catamari/index_utils.hpp"
#include "catamari/ldl/scalar_ldl/scalar_utils.hpp"
#include "quotient/io_utils.hpp"

#include "catamari/ldl/scalar_ldl/factorization.hpp"

namespace catamari {
namespace scalar_ldl {

template <class Field>
void Factorization<Field>::LeftLookingSetup(
    const CoordinateMatrix<Field>& matrix) CATAMARI_NOEXCEPT {
  // TODO(Jack Poulson): Decide if/when the following should be parallelized.
  // The main cost tradeoffs are the need for the creation of the children in
  // the supernodal assembly forest and the additional memory allocations.

  Buffer<Int> degrees;
  EliminationForestAndDegrees(matrix, ordering,
                              &ordering.assembly_forest.parents, &degrees);

  ordering.assembly_forest.FillFromParents();
  FillStructureIndices(matrix, ordering, ordering.assembly_forest, degrees,
                       &lower_factor.structure);

  FillNonzeros(matrix);
}

template <class Field>
LDLResult Factorization<Field>::LeftLooking(
    const CoordinateMatrix<Field>& matrix) CATAMARI_NOEXCEPT {
  typedef ComplexBase<Field> Real;
  const Int num_rows = matrix.NumRows();
  const Buffer<Int>& parents = ordering.assembly_forest.parents;
  const LowerStructure& lower_structure = lower_factor.structure;

  LeftLookingState state;
  state.column_update_ptrs.Resize(num_rows);
  state.row_structure.Resize(num_rows);
  state.pattern_flags.Resize(num_rows);

  LDLResult result;
  for (Int column = 0; column < num_rows; ++column) {
    state.pattern_flags[column] = column;
    state.column_update_ptrs[column] = lower_structure.ColumnOffset(column);

    // Compute the row pattern.
    const Int num_packed = ComputeRowPattern(matrix, ordering, parents, column,
                                             state.pattern_flags.Data(),
                                             state.row_structure.Data());

    // for j = find(L(column, :))
    //   L(column:n, column) -= L(column:n, j) * (d(j) * conj(L(column, j)))
    for (Int index = 0; index < num_packed; ++index) {
      const Int j = state.row_structure[index];
      CATAMARI_ASSERT(j < column, "Looking into upper triangle.");

      // Find L(column, j) in the j'th column.
      Int j_ptr = state.column_update_ptrs[j]++;
      const Int j_end = lower_structure.ColumnOffset(j + 1);
      CATAMARI_ASSERT(j_ptr != j_end, "Left column looking for L(column, j)");
      CATAMARI_ASSERT(lower_structure.indices[j_ptr] == column,
                      "Did not find L(column, j)");

      const Field lambda_k_j = lower_factor.values[j_ptr];
      Field eta;
      if (factorization_type == kCholeskyFactorization) {
        eta = Conjugate(lambda_k_j);
      } else if (factorization_type == kLDLAdjointFactorization) {
        eta = diagonal_factor.values[j] * Conjugate(lambda_k_j);
      } else {
        eta = diagonal_factor.values[j] * lambda_k_j;
      }

      // L(column, column) -= L(column, j) * eta.
      diagonal_factor.values[column] -= lambda_k_j * eta;
      ++j_ptr;

      // L(column+1:n, column) -= L(column+1:n, j) * eta.
      const Int column_beg = lower_structure.ColumnOffset(column);
      Int column_ptr = column_beg;
      for (; j_ptr != j_end; ++j_ptr) {
        const Int row = lower_structure.indices[j_ptr];
        CATAMARI_ASSERT(row >= column, "Row index was less than column.");

        // L(row, column) -= L(row, j) * eta.
        // Move the pointer for column 'column' to the equivalent index.
        //
        // TODO(Jack Poulson): Decide if this 'search kernel' (see Rothbert
        // and Gupta's "An Evaluation of Left-Looking, Right-Looking and
        // Multifrontal Approaches to Sparse Cholesky Factorization on
        // Hierarchical-Memory Machines") can be improved. Unfortunately,
        // binary search, e.g., via std::lower_bound between 'column_ptr' and
        // the end of the column, leads to a 3x slowdown of the factorization
        // of the bbmat matrix.
        while (lower_structure.indices[column_ptr] < row) {
          ++column_ptr;
        }
        CATAMARI_ASSERT(lower_structure.indices[column_ptr] == row,
                        "The column pattern did not contain the j pattern.");
        CATAMARI_ASSERT(column_ptr < lower_structure.column_offsets[column + 1],
                        "The column pointer left the column.");
        const Field update = lower_factor.values[j_ptr] * eta;
        lower_factor.values[column_ptr] -= update;
      }
    }

    // Early exit if solving would involve division by zero.
    Field pivot = diagonal_factor.values[column];
    if (factorization_type == kCholeskyFactorization) {
      if (RealPart(pivot) <= Real{0}) {
        return result;
      }
      pivot = std::sqrt(RealPart(pivot));
      diagonal_factor.values[column] = pivot;
    } else if (factorization_type == kLDLAdjointFactorization) {
      if (RealPart(pivot) == Real{0}) {
        return result;
      }
      pivot = RealPart(pivot);
      diagonal_factor.values[column] = pivot;
    } else {
      if (pivot == Field{0}) {
        return result;
      }
    }

    // L(column+1:n, column) /= d(column).
    {
      const Int column_beg = lower_structure.ColumnOffset(column);
      const Int column_end = lower_structure.ColumnOffset(column + 1);
      for (Int index = column_beg; index < column_end; ++index) {
        lower_factor.values[index] /= pivot;
      }
    }

    // Update the result structure.
    const Int degree = lower_structure.Degree(column);
    result.num_factorization_entries += 1 + degree;

    const double solve_flops = (IsComplex<Field>::value ? 6. : 1.) * degree;

    const double schur_complement_flops =
        (IsComplex<Field>::value ? 4. : 1.) * std::pow(1. * degree, 2.);

    result.num_subdiag_solve_flops += solve_flops;
    result.num_schur_complement_flops += schur_complement_flops;
    result.num_factorization_flops += solve_flops + schur_complement_flops;

    ++result.num_successful_pivots;
  }

  return result;
}

}  // namespace scalar_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_LDL_SCALAR_LDL_FACTORIZATION_LEFT_LOOKING_IMPL_H_
