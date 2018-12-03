/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_H_
#define CATAMARI_LDL_H_

#include <ostream>

#include "catamari/coordinate_matrix.hpp"
#include "catamari/integers.hpp"

namespace catamari {

enum LDLAlgorithm {
  // A left-looking LDL factorization. Cf. Section 4.8 of Tim Davis,
  // "Direct Methods for Sparse Linear Systems".
  kLeftLookingLDL,

  // An up-looking LDL factorization. Cf. Section 4.7 of Tim Davis,
  // "Direct Methods for Sparse Linear Systems".
  kUpLookingLDL,
};

// A column-major lower-triangular sparse matrix.
template <class Field>
struct LowerFactor {
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
template <class Field>
struct DiagonalFactor {
  std::vector<Field> values;
};

// A representation of a non-supernodal LDL' factorization.
template <class Field>
struct LDLFactorization {
  // The unit lower-triangular factor, L.
  LowerFactor<Field> lower_factor;

  // The diagonal factor, D.
  DiagonalFactor<Field> diagonal_factor;
};

// Pretty-prints a column-oriented sparse lower-triangular matrix.
template <class Field>
void PrintLowerFactor(const LowerFactor<Field>& lower_factor,
                      const std::string& label, std::ostream& os);

// Pretty-prints the diagonal factor of the L D L' factorization.
template <class Field>
void PrintDiagonalFactor(const DiagonalFactor<Field>& diagonal_factor,
                         const std::string& label, std::ostream& os);

// Performs a non-supernodal LDL' factorization in the natural ordering.
template <class Field>
Int LDL(const CoordinateMatrix<Field>& matrix, LDLAlgorithm algorithm,
        LDLFactorization<Field>* factorization);

// Solve A x = b via the substitution (L D L') x = b and the sequence:
//   x := L' \ (D \ (L \ b)).
template <class Field>
void LDLSolve(const LDLFactorization<Field>& factorization,
              std::vector<Field>* vector);

// Solves L x = b using a unit-lower triangular matrix L.
template <class Field>
void UnitLowerTriangularSolve(const LowerFactor<Field>& unit_lower_factor,
                              std::vector<Field>* vector);

// Solves D x = b using a diagonal matrix D.
template <class Field>
void DiagonalSolve(const DiagonalFactor<Field>& diagonal_factor,
                   std::vector<Field>* vector);

// Solves L' x = b using a unit-lower triangular matrix L.
template <class Field>
void UnitLowerAdjointTriangularSolve(
    const LowerFactor<Field>& unit_lower_factor, std::vector<Field>* vector);

}  // namespace catamari

#include "catamari/ldl-impl.hpp"

#endif  // ifndef CATAMARI_LDL_H_