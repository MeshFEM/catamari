/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_IMPL_H_
#define CATAMARI_LDL_IMPL_H_

#include "catamari/ldl.hpp"

namespace catamari {

template<class Field>
void PrintScalarLowerFactor(
    const ScalarLowerFactor<Field>& lower_factor, const std::string& label) {
  std::cout << label << ":\n";
  const Int num_columns = lower_factor.column_offsets.size() - 1;
  for (Int column = 0; column < num_columns; ++column) {
    const Int column_beg = lower_factor.column_offsets[column];
    const Int column_end = lower_factor.column_offsets[column + 1];
    for (Int index = column_beg; index < column_end; ++index) {
      const Int row = lower_factor.indices[index];
      const Field& value = lower_factor.values[index];
      std::cout << row << " " << column << " " << value << "\n";
    }
  }
  std::cout << std::endl;
}

template<class Field>
void PrintScalarDiagonalFactor(
    const ScalarDiagonalFactor<Field>& diagonal_factor,
    const std::string& label) {
  quotient::PrintVector(diagonal_factor.values, label);
}

template<class Field>
void MatrixVectorProduct(
    const Field& alpha,
    const CoordinateMatrix<Field>& matrix,
    const std::vector<Field>& vec0,
    const Field& beta,
    std::vector<Field>* vec1) {
  const Int num_rows = matrix.NumRows();
  CATAMARI_ASSERT(vec0.size() == matrix.NumColumns(),
      "vec0 was of the incorrect size.");
  CATAMARI_ASSERT(vec1->size() == num_rows,
      "vec1 was of the incorrect size.");

  // vec1 += alpha matrix vec0 + beta vec1
  const std::vector<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    (*vec1)[row] *= beta;
    const Int row_beg = matrix.RowEntryOffset(row);
    const Int row_end = matrix.RowEntryOffset(row + 1);
    for (Int index = row_beg; index < row_end; ++index) { 
      const MatrixEntry<Field>& entry = entries[index];
      CATAMARI_ASSERT(entry.row == row, "Invalid entry row index.");
      // vec1(row) += alpha matrix(row, column) vec0(column).
      (*vec1)[row] += alpha * entry.value * vec0[entry.column];
    }
  }
}

namespace scalar_ldl {

// Computes the elimination forest (via the 'parents' array) and sizes of the
// structures of a scalar (simplicial) LDL' factorization.
//
// Cf. Tim Davis's "LDL"'s symbolic factorization.
template<class Field>
void EliminationForestAndStructureSizes(
    const CoordinateMatrix<Field>& matrix,
    std::vector<Int>* parents,
    std::vector<Int>* structure_sizes) {
  const Int num_rows = matrix.NumRows();

  // Initialize all of the parent indices as unset.
  parents->resize(num_rows, -1);

  // Initialize each node's flag to its own index.
  std::vector<Int> flags(num_rows);
  std::iota(flags.begin(), flags.end(), 0);

  // Initialize the number of entries that will be stored into each column.
  structure_sizes->clear();
  structure_sizes->resize(num_rows, 0);

  const std::vector<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int row = 0; row < num_rows; ++row) {
    const Int row_beg = matrix.RowEntryOffset(row);
    const Int row_end = matrix.RowEntryOffset(row + 1);
    for (Int index = row_beg; index < row_end; ++index) {
      const MatrixEntry<Field>& entry = entries[index];
      Int column = entry.column;

      // We are traversing the strictly lower triangle and know that the
      // indices are sorted.
      if (column >= row) {
        break;
      }

      // Look for new entries in the pattern by walking up to the root of this
      // subtree of the elimination forest from index 'column'. Any unset
      // parent pointers can be filled in during the traversal, as the current
      // row index would then be the parent.
      while (flags[column] != row) {
        // Mark index 'column' as in the pattern of row 'row'.
        flags[column] = row;
        ++(*structure_sizes)[column];

        if ((*parents)[column] == -1) {
          // This is the first occurrence of 'column' in a row pattern.
          (*parents)[column] = row;
        }

        // Move up to the parent in this subtree of the elimination forest.
        // Moving to the parent will increase the index (but remain bounded
        // from above by 'row').
        column = (*parents)[column];
      }
    }
  }
}

// Sets up the data structures needed for a scalar up-looking LDL'
// factorization: the elimination forest and structure sizes are computed,
// and the memory for the factors is allocated.
template<class Field>
void UpLookingSetup(
    const CoordinateMatrix<Field>& matrix,
    ScalarLDLAnalysis* analysis,
    ScalarLowerFactor<Field>* unit_lower_factor,
    ScalarDiagonalFactor<Field>* diagonal_factor) {
  std::vector<Int> structure_sizes;
  scalar_ldl::EliminationForestAndStructureSizes(
      matrix, &analysis->parents, &structure_sizes);

  const Int num_rows = matrix.NumRows();
  unit_lower_factor->column_offsets.resize(num_rows + 1, 0);
  Int num_entries = 0;
  for (Int column = 0; column < num_rows; ++column) {
    unit_lower_factor->column_offsets[column] = num_entries;
    num_entries += structure_sizes[column];
  }
  unit_lower_factor->column_offsets[num_rows] = num_entries;
  unit_lower_factor->indices.resize(num_entries);
  unit_lower_factor->values.resize(num_entries);

  diagonal_factor->values.resize(num_rows); 
}

// Compute the nonzero pattern of L(row, :) in
// row_structure[start : num_rows - 1] and spread A(row, 0 : row - 1) into
// row_workspace.
template<class Field>
Int UpLookingRowPattern(
    const CoordinateMatrix<Field>& matrix,
    const std::vector<Int>& parents,
    Int row,
    Int* flags,
    Int* row_structure,
    Field* row_workspace) {
  Int start = matrix.NumRows();
  const Int row_beg = matrix.RowEntryOffset(row); 
  const Int row_end = matrix.RowEntryOffset(row + 1);
  const std::vector<MatrixEntry<Field>>& entries = matrix.Entries();
  for (Int index = row_beg; index < row_end; ++index) {
    const MatrixEntry<Field>& entry = entries[index];
    Int column = entry.column;
    if (column > row) {
      break;
    }

    // Scatter matrix(row, column) into row_workspace.
    row_workspace[column] = entry.value;

    // Walk up to the root of the current subtree of the elimination
    // forest, stopping if we encounter a member already marked as in the
    // row pattern.
    Int num_packed = 0;
    while (flags[column] != row) {
      // Place 'column' into the pattern of row 'row'.
      row_structure[num_packed++] = column;
      flags[column] = row;

      // Move up to the parent in this subtree of the elimination forest.
      column = parents[column];
    }

    // Pack this ancestral sequence into the back of the pattern.
    while (num_packed > 0) {
      row_structure[--start] = row_structure[--num_packed];
    }
  }
  return start;
}

// For each index 'i' in the structure of column 'column' of L formed so far:
//   L(row, i) -= (L(row, column) * d(column)) * conj(L(i, column)).
// L(row, row) is similarly updated, within d, then L(row, column) is finalized.
template<class Field>
void UpLookingRowUpdate(
    Int row,
    Int column,
    ScalarLowerFactor<Field>* unit_lower_factor,
    ScalarDiagonalFactor<Field>* diagonal_factor,
    Int* structure_sizes,
    Field* row_workspace) {
  // Load eta := L(row, column) * d(column) from the workspace.
  const Field eta = row_workspace[column];
  row_workspace[column] = Field{0};

  // Update
  //
  //   L(row, I) -= (L(row, column) * d(column)) * conj(L(I, column)),
  //
  // where I is the set of already-formed entries in the structure of column
  // 'column' of L.
  const Int column_structure_size = structure_sizes[column];
  const Int factor_column_beg = unit_lower_factor->column_offsets[column];
  const Int factor_column_end = factor_column_beg + column_structure_size;
  for (Int index = factor_column_beg; index < factor_column_end; ++index) {
    // Update L(row, i) -= (L(row, column) * d(column)) * conj(L(i, column)).
    const Int i = unit_lower_factor->indices[index];
    const Field& value = unit_lower_factor->values[index];
    row_workspace[i] -= eta * Conjugate(value);
  }

  // Compute L(row, column) from eta = L(row, column) * d(column).
  const Field lambda = eta / diagonal_factor->values[column];

  // Update L(row, row) -= (L(row, column) * d(column)) * conj(L(row, column)).
  diagonal_factor->values[row] -= eta * Conjugate(lambda);

  // Append L(row, column) into the structure of column 'column'.
  unit_lower_factor->indices[factor_column_end] = row;
  unit_lower_factor->values[factor_column_end] = lambda;
  ++structure_sizes[column];
}

} // namespace scalar_ldl

template<class Field>
void ScalarUpLookingCholeskySetup(
    const CoordinateMatrix<Field>& matrix,
    ScalarLDLAnalysis* analysis,
    ScalarLowerFactor<Field>* unit_lower_factor) {
  std::vector<Int> structure_sizes;
  scalar_ldl::EliminationForestAndStructureSizes(
      matrix, &analysis->parents, &structure_sizes);

  const Int num_rows = matrix.NumRows();
  unit_lower_factor->column_offsets.resize(num_rows + 1, 0);
  Int num_entries = 0;
  for (Int column = 0; column < num_rows; ++column) {
    unit_lower_factor->column_offsets[column] = num_entries;

    // We must also account for the diagonal entry.
    num_entries += structure_sizes[column] + 1;
  }
  unit_lower_factor->column_offsets[num_rows] = num_entries;
  unit_lower_factor->indices.resize(num_entries);
  unit_lower_factor->values.resize(num_entries);
}

// Cf. Tim Davis's up-looking "LDL" numerical factorization.
template<class Field>
Int ScalarUpLookingLDL(
    const CoordinateMatrix<Field>& matrix,
    ScalarLowerFactor<Field>* unit_lower_factor,
    ScalarDiagonalFactor<Field>* diagonal_factor) {
  const Int num_rows = matrix.NumRows();

  ScalarLDLAnalysis analysis;
  scalar_ldl::UpLookingSetup(
      matrix, &analysis, unit_lower_factor, diagonal_factor);

  // Initialize each node flag to its own index.
  std::vector<Int> flags(num_rows);
  std::iota(flags.begin(), flags.end(), 0);

  // Set up an integer workspace that could hold any row nonzero pattern.
  std::vector<Int> row_structure(num_rows);

  // Initialize the number of entries that have been packed into each column.
  std::vector<Int> structure_sizes(num_rows, 0);

  // Set up a workspace for performing a triangular solve against a row of the
  // input matrix.
  std::vector<Field> row_workspace(num_rows, Field{0});

  for (Int row = 0; row < num_rows; ++row) {
    // Compute the row pattern and scatter the row of the input matrix into
    // the workspace.
    const Int start = scalar_ldl::UpLookingRowPattern(
        matrix, analysis.parents, row, flags.data(), row_structure.data(),
        row_workspace.data());

    // Pull the diagonal entry out of the workspace.
    diagonal_factor->values[row] = row_workspace[row];
    row_workspace[row] = Field{0};

    // Compute L(row, :) using a sparse triangular solve. In particular,
    //   L(row, :) := matrix(row, :) / L(0 : row - 1, 0 : row - 1)'.
    for (Int index = start; index < num_rows; ++index) {
      const Int column = row_structure[index];
      scalar_ldl::UpLookingRowUpdate(
          row, column, unit_lower_factor, diagonal_factor,
          structure_sizes.data(), row_workspace.data());
    }

    // Early exit if solving would involve division by zero.
    if (diagonal_factor->values[row] == Field{0}) {
      return row;
    }
  }

  return num_rows;
}

template<class Field>
void LDLSolve(
    const ScalarLowerFactor<Field>& unit_lower_factor,
    const ScalarDiagonalFactor<Field>& diagonal_factor,
    std::vector<Field>* vector) {
  UnitLowerTriangularSolve(unit_lower_factor, vector);
  DiagonalSolve(diagonal_factor, vector);
  UnitLowerAdjointTriangularSolve(unit_lower_factor, vector);
}

template<class Field>
void UnitLowerTriangularSolve(
    const ScalarLowerFactor<Field>& unit_lower_factor,
    std::vector<Field>* vector) {
  const Int num_rows = unit_lower_factor.column_offsets.size() - 1;
  CATAMARI_ASSERT(vector->size() == num_rows,
      "Vector was of the incorrect size.");
  for (Int column = 0; column < num_rows; ++column) {
    const Int factor_column_beg = unit_lower_factor.column_offsets[column];
    const Int factor_column_end = unit_lower_factor.column_offsets[column + 1];
    for (Int index = factor_column_beg; index < factor_column_end; ++index) {
      const Int i = unit_lower_factor.indices[index];
      const Field& value = unit_lower_factor.values[index];
      (*vector)[i] -= value * (*vector)[column];
    }
  }
}

template<class Field>
void DiagonalSolve(
    const ScalarDiagonalFactor<Field>& diagonal_factor,
    std::vector<Field>* vector) {
  const Int num_rows = diagonal_factor.values.size();
  CATAMARI_ASSERT(vector->size() == num_rows,
      "Vector was of the incorrect size.");
  for (Int column = 0; column < num_rows; ++column) {
    (*vector)[column] /= diagonal_factor.values[column];
  }
}

template<class Field>
void UnitLowerAdjointTriangularSolve(
    const ScalarLowerFactor<Field>& unit_lower_factor,
    std::vector<Field>* vector) {
  const Int num_rows = unit_lower_factor.column_offsets.size() - 1;
  CATAMARI_ASSERT(vector->size() == num_rows,
      "Vector was of the incorrect size.");
  for (Int column = num_rows - 1; column >= 0; --column) { 
    const Int factor_column_beg = unit_lower_factor.column_offsets[column];
    const Int factor_column_end = unit_lower_factor.column_offsets[column + 1];
    for (Int index = factor_column_beg; index < factor_column_end; ++index) {
      const Int i = unit_lower_factor.indices[index];
      const Field& value = unit_lower_factor.values[index];
      (*vector)[column] -= Conjugate(value) * (*vector)[i];
    }
  }
}

} // namespace catamari

#endif // ifndef CATAMARI_LDL_IMPL_H_
