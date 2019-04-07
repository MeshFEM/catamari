/*
 * Copyright (c) 2018-2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SCALAR_FACTORIZATION_UP_LOOKING_IMPL_H_
#define CATAMARI_SPARSE_LDL_SCALAR_FACTORIZATION_UP_LOOKING_IMPL_H_

#include <cmath>

#include "catamari/index_utils.hpp"
#include "catamari/sparse_ldl/scalar/scalar_utils.hpp"
#include "quotient/io_utils.hpp"

#include "catamari/sparse_ldl/scalar/factorization.hpp"

namespace catamari {
namespace scalar_ldl {

template <class Field>
void Factorization<Field>::UpLookingSetup(const CoordinateMatrix<Field>& matrix)
    CATAMARI_NOEXCEPT {
  // TODO(Jack Poulson): Decide if/when the following should be parallelized.
  // The main cost tradeoffs are the need for the creation of the children in
  // the supernodal assembly forest and the additional memory allocations.

  Buffer<Int> degrees;
  EliminationForestAndDegrees(matrix, ordering,
                              &ordering.assembly_forest.parents, &degrees);

  LowerStructure& lower_structure = lower_factor.structure;
  OffsetScan(degrees, &lower_structure.column_offsets);

  const Int num_rows = matrix.NumRows();
  diagonal_factor.values.Resize(num_rows);

  const Int num_entries = lower_structure.column_offsets.Back();
  lower_structure.indices.Resize(num_entries);
  lower_factor.values.Resize(num_entries);
}

template <class Field>
void Factorization<Field>::UpLookingRowUpdate(
    Int row, const Int* column_beg, const Int* column_end,
    Int* column_update_ptrs, Field* row_workspace) CATAMARI_NOEXCEPT {
  LowerStructure& lower_structure = lower_factor.structure;
  const bool is_cholesky = factorization_type == kCholeskyFactorization;
  const bool is_selfadjoint = factorization_type != kLDLTransposeFactorization;

  for (const Int* iter = column_beg; iter != column_end; ++iter) {
    const Int column = *iter;
    const Field pivot = is_selfadjoint
                            ? RealPart(diagonal_factor.values[column])
                            : diagonal_factor.values[column];

    // Load eta := L(row, column) * d(column) from the workspace.
    const Field eta =
        is_cholesky ? row_workspace[column] / pivot : row_workspace[column];
    row_workspace[column] = Field{0};

    // Update
    //
    //   L(row, I) -= (L(row, column) * d(column)) * conj(L(I, column)),
    //
    // where I is the set of already-formed entries in the structure of column
    // 'column' of L.
    //
    // Rothberg and Gupta refer to this as a 'scatter kernel' in:
    //   "An Evaluation of Left-Looking, Right-Looking and Multifrontal
    //   Approaches to Sparse Cholesky Factorization on Hierarchical-Memory
    //   Machines".
    const Int factor_column_beg = lower_structure.ColumnOffset(column);
    const Int factor_column_end = column_update_ptrs[column]++;
    for (Int index = factor_column_beg; index < factor_column_end; ++index) {
      // L(row, i) -= (L(row, column) * d(column)) * conj(L(i, column))
      const Int i = lower_structure.indices[index];
      const Field& value = lower_factor.values[index];
      if (is_selfadjoint) {
        row_workspace[i] -= eta * Conjugate(value);
      } else {
        row_workspace[i] -= eta * value;
      }
    }

    // Compute L(row, column) from eta = L(row, column) * d(column).
    const Field lambda = is_cholesky ? eta : eta / pivot;

    // Append L(row, column) into the structure of column 'column'.
    lower_structure.indices[factor_column_end] = row;
    lower_factor.values[factor_column_end] = lambda;

    // L(row, row) -= (L(row, column) * d(column)) * conj(L(row, column))
    if (is_selfadjoint) {
      diagonal_factor.values[row] -= eta * Conjugate(lambda);
    } else {
      diagonal_factor.values[row] -= eta * lambda;
    }
  }
}

template <class Field>
SparseLDLResult Factorization<Field>::UpLooking(
    const CoordinateMatrix<Field>& matrix) CATAMARI_NOEXCEPT {
  typedef ComplexBase<Field> Real;
  const Int num_rows = matrix.NumRows();
  const Buffer<Int>& parents = ordering.assembly_forest.parents;
  const LowerStructure& lower_structure = lower_factor.structure;

  UpLookingState<Field> state;
  state.column_update_ptrs.Resize(num_rows);
  state.pattern_flags.Resize(num_rows);
  state.row_structure.Resize(num_rows);
  state.row_workspace.Resize(num_rows);

  SparseLDLResult result;
  for (Int row = 0; row < num_rows; ++row) {
    state.pattern_flags[row] = row;
    state.column_update_ptrs[row] = lower_structure.ColumnOffset(row);

    // Compute the row pattern and scatter the row of the input matrix into
    // the workspace.
    const Int start = ComputeTopologicalRowPatternAndScatterNonzeros(
        matrix, ordering, parents, row, state.pattern_flags.Data(),
        state.row_structure.Data(), state.row_workspace.Data());

    // Pull the diagonal entry out of the workspace.
    diagonal_factor.values[row] = state.row_workspace[row];
    state.row_workspace[row] = Field{0};

    // Compute L(row, :) using a sparse triangular solve. In particular,
    //   L(row, :) := matrix(row, :) / L(0 : row - 1, 0 : row - 1)'.
    UpLookingRowUpdate(row, state.row_structure.Data() + start,
                       state.row_structure.Data() + num_rows,
                       state.column_update_ptrs.Data(),
                       state.row_workspace.Data());

    // Early exit if solving would involve division by zero.
    const Field pivot = diagonal_factor.values[row];
    if (factorization_type == kCholeskyFactorization) {
      if (RealPart(pivot) <= Real{0}) {
        return result;
      }
      diagonal_factor.values[row] = std::sqrt(RealPart(pivot));
    } else if (factorization_type == kLDLAdjointFactorization) {
      if (RealPart(pivot) == Real{0}) {
        return result;
      }
      diagonal_factor.values[row] = RealPart(pivot);
    } else {
      if (pivot == Field{0}) {
        return result;
      }
    }

    // Update the result structure.
    const Int degree = lower_structure.Degree(row);
    result.num_factorization_entries += 1 + degree;

    const double solve_flops = (IsComplex<Field>::value ? 6. : 1.) * degree;

    const double schur_complement_flops =
        (IsComplex<Field>::value ? 4. : 1.) * (1. * degree) * (1. * degree);

    result.num_subdiag_solve_flops += solve_flops;
    result.num_schur_complement_flops += schur_complement_flops;
    result.num_factorization_flops += solve_flops + schur_complement_flops;

    ++result.num_successful_pivots;
  }

  return result;
}

}  // namespace scalar_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_SPARSE_LDL_SCALAR_FACTORIZATION_UP_LOOKING_IMPL_H_
