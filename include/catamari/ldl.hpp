/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_H_
#define CATAMARI_LDL_H_

#include "catamari/coordinate_matrix.hpp"
#include "catamari/integers.hpp"

namespace catamari {

// The metadata involved in the symbolic analysis for a scalar L D L'
// factorization.
struct ScalarLDLAnalysis {
  // A vector of length 'num_rows'; each entry is either the parent of the
  // corresponding vertex in the elimination forest or -1, denoting that said
  // vertex is a root in the elimination forest.
  std::vector<Int> parents;
};

// A column-major lower-triangular sparse matrix.
template<class Field>
struct ScalarLowerFactor {
  // A vector of length 'num_rows + 1'; each entry corresponds to the offset
  // in 'indices' and 'values' of the corresponding column of the matrix.
  std::vector<Int> column_offsets;

  // A vector of length 'num_entries'; each segment, column_offsets[j] to
  // column_offsets[j + 1] - 1, contains the row indices for the j'th column.
  std::vector<Int> indices;

  // A vector of length 'num_entries'; each segment, column_offsets[j] to
  // column_offsets[j + 1] - 1, contains the (typically nonzero) entries for
  // the j'th column.
  std::vector<Field> values;
};

// A diagonal matrix representing the 'D' in an L D L' factorization.
template<class Field>
struct ScalarDiagonalFactor {
  std::vector<Field> values;
};

// Pretty-prints a column-oriented sparse lower-triangular matrix.
template<class Field>
void PrintScalarLowerFactor(
    const ScalarLowerFactor<Field>& lower_factor, const std::string& label);

// Pretty-prints the diagonal factor of the L D L' factorization.
template<class Field>
void PrintScalarDiagonalFactor(
    const ScalarDiagonalFactor<Field>& diagonal_factor,
    const std::string& label);

// vec1 := alpha matrix vec0 + beta vec1.
template<class Field>
void MatrixVectorProduct(
    const Field& alpha,
    const CoordinateMatrix<Field>& matrix,
    const std::vector<Field>& vec0,
    const Field& beta,
    std::vector<Field>* vec1);

// Performs the symbolic analysis for the LDL' factorization.
template<class Field>
void ScalarLDLSetup(
    const CoordinateMatrix<Field>& matrix,
    ScalarLDLAnalysis* analysis,
    ScalarLowerFactor<Field>* unit_lower_factor,
    ScalarDiagonalFactor<Field>* diagonal_factor);

// Performs the LDL' factorization.
template<class Field>
Int ScalarLDLFactorization(
    const CoordinateMatrix<Field>& matrix,
    const ScalarLDLAnalysis& analysis,
    ScalarLowerFactor<Field>* unit_lower_factor,
    ScalarDiagonalFactor<Field>* diagonal_factor);

// Solve A x = b via the substitution (L D L') x = b and the sequence:
//   x := L' \ (D \ (L \ b)).
template<class Field>
void LDLSolve(
    const ScalarLowerFactor<Field>& unit_lower_factor,
    const ScalarDiagonalFactor<Field>& diagonal_factor,
    std::vector<Field>* vector);

// Solves L x = b using a unit-lower triangular matrix L.
template<class Field>
void UnitLowerTriangularSolve(
    const ScalarLowerFactor<Field>& unit_lower_factor,
    std::vector<Field>* vector);

// Solves D x = b using a diagonal matrix D.
template<class Field>
void DiagonalSolve(
    const ScalarDiagonalFactor<Field>& diagonal_factor,
    std::vector<Field>* vector);

// Solves L' x = b using a unit-lower triangular matrix L.
template<class Field>
void UnitLowerAdjointTriangularSolve(
    const ScalarLowerFactor<Field>& unit_lower_factor,
    std::vector<Field>* vector);

} // namespace catamari

#include "catamari/ldl-impl.hpp"

#endif // ifndef CATAMARI_LDL_H_
