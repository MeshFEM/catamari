/*
 * Copyright (c) 2018-2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SCALAR_LDL_FACTORIZATION_IMPL_H_
#define CATAMARI_LDL_SCALAR_LDL_FACTORIZATION_IMPL_H_

#include <cmath>

#include "catamari/index_utils.hpp"
#include "catamari/ldl/scalar_ldl/scalar_utils.hpp"
#include "quotient/io_utils.hpp"

#include "catamari/ldl/scalar_ldl/factorization.hpp"

namespace catamari {
namespace scalar_ldl {

template <class Field>
void Factorization<Field>::PrintLowerFactor(const std::string& label,
                                            std::ostream& os) const {
  const LowerStructure& lower_structure = lower_factor.structure;

  auto print_entry = [&](const Int& row, const Int& column,
                         const Field& value) {
    os << row << " " << column << " " << value << "\n";
  };

  os << label << ":\n";
  const Int num_columns = lower_structure.column_offsets.Size() - 1;
  for (Int column = 0; column < num_columns; ++column) {
    if (factorization_type == kCholeskyFactorization) {
      print_entry(column, column, diagonal_factor.values[column]);
    } else {
      print_entry(column, column, Field{1});
    }

    const Int column_beg = lower_structure.ColumnOffset(column);
    const Int column_end = lower_structure.ColumnOffset(column + 1);
    for (Int index = column_beg; index < column_end; ++index) {
      const Int row = lower_structure.indices[index];
      const Field& value = lower_factor.values[index];
      print_entry(row, column, value);
    }
  }
  os << std::endl;
}

template <class Field>
void Factorization<Field>::PrintDiagonalFactor(const std::string& label,
                                               std::ostream& os) const {
  if (factorization_type == kCholeskyFactorization) {
    // TODO(Jack Poulson): Print the identity.
    return;
  }
  quotient::PrintVector(diagonal_factor.values, label, os);
}

template <class Field>
void Factorization<Field>::FillNonzeros(const CoordinateMatrix<Field>& matrix) {
  LowerStructure& lower_structure = lower_factor.structure;
  const Int num_rows = matrix.NumRows();
  const Int num_entries = lower_structure.indices.Size();
  const bool have_permutation = !ordering.permutation.Empty();

  lower_factor.values.Resize(num_entries, Field{0});
  diagonal_factor.values.Resize(num_rows, Field{0});

  const Buffer<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int orig_row =
        have_permutation ? ordering.inverse_permutation[row] : row;
    const Int row_beg = matrix.RowEntryOffset(orig_row);
    const Int row_end = matrix.RowEntryOffset(orig_row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      Int column =
          have_permutation ? ordering.permutation[entry.column] : entry.column;

      if (column == row) {
        diagonal_factor.values[column] = entry.value;
      }
      if (column >= row) {
        if (have_permutation) {
          continue;
        } else {
          break;
        }
      }

      Int* iter = std::lower_bound(lower_structure.ColumnBeg(column),
                                   lower_structure.ColumnEnd(column), row);
      CATAMARI_ASSERT(iter != lower_structure.ColumnEnd(column),
                      "Exceeded column indices.");
      CATAMARI_ASSERT(*iter == row, "Did not find index.");
      const Int structure_index =
          std::distance(lower_structure.indices.Data(), iter);
      lower_factor.values[structure_index] = entry.value;
    }
  }
}

template <class Field>
void Factorization<Field>::LeftLookingSetup(
    const CoordinateMatrix<Field>& matrix) {
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
void Factorization<Field>::UpLookingSetup(
    const CoordinateMatrix<Field>& matrix) {
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
void Factorization<Field>::UpLookingRowUpdate(Int row, Int column,
                                              Int* column_update_ptrs,
                                              Field* row_workspace) {
  LowerStructure& lower_structure = lower_factor.structure;
  const bool is_cholesky = factorization_type == kCholeskyFactorization;
  const bool is_selfadjoint = factorization_type != kLDLTransposeFactorization;
  const Field pivot = is_selfadjoint ? RealPart(diagonal_factor.values[column])
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
    // Update L(row, i) -= (L(row, column) * d(column)) * conj(L(i, column)).
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

  // Update L(row, row) -= (L(row, column) * d(column)) * conj(L(row, column)).
  if (is_selfadjoint) {
    diagonal_factor.values[row] -= eta * Conjugate(lambda);
  } else {
    diagonal_factor.values[row] -= eta * lambda;
  }

  // Append L(row, column) into the structure of column 'column'.
  lower_structure.indices[factor_column_end] = row;
  lower_factor.values[factor_column_end] = lambda;
}

template <class Field>
LDLResult Factorization<Field>::LeftLooking(
    const CoordinateMatrix<Field>& matrix) {
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

template <class Field>
LDLResult Factorization<Field>::UpLooking(
    const CoordinateMatrix<Field>& matrix) {
  typedef ComplexBase<Field> Real;
  const Int num_rows = matrix.NumRows();
  const Buffer<Int>& parents = ordering.assembly_forest.parents;
  const LowerStructure& lower_structure = lower_factor.structure;

  UpLookingState state;
  state.column_update_ptrs.Resize(num_rows);
  state.pattern_flags.Resize(num_rows);
  state.row_structure.Resize(num_rows);
  state.row_workspace.Resize(num_rows);

  LDLResult result;
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
    for (Int index = start; index < num_rows; ++index) {
      const Int column = state.row_structure[index];
      UpLookingRowUpdate(row, column, state.column_update_ptrs.Data(),
                         state.row_workspace.Data());
    }

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

template <class Field>
LDLResult Factorization<Field>::Factor(const CoordinateMatrix<Field>& matrix,
                                       const SymmetricOrdering& manual_ordering,
                                       const Control& control) {
  ordering = manual_ordering;
  factorization_type = control.factorization_type;
  algorithm = control.algorithm;
  if (algorithm == kLeftLookingLDL) {
    LeftLookingSetup(matrix);
    return LeftLooking(matrix);
  } else {
    UpLookingSetup(matrix);
    return UpLooking(matrix);
  }
}

template <class Field>
LDLResult Factorization<Field>::RefactorWithFixedSparsityPattern(
    const CoordinateMatrix<Field>& matrix) {
  if (algorithm == kLeftLookingLDL) {
    FillNonzeros(matrix);
    return LeftLooking(matrix);
  } else {
    return UpLooking(matrix);
  }
}

}  // namespace scalar_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_LDL_SCALAR_LDL_FACTORIZATION_IMPL_H_
