/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SCALAR_LDL_H_
#define CATAMARI_SCALAR_LDL_H_

#include <ostream>

#include "catamari/blas_matrix.hpp"
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

// The nonzero patterns below the diagonal of the lower-triangular factor.
struct ScalarLowerStructure {
  // A vector of length 'num_rows + 1'; each entry corresponds to the offset
  // in 'indices' and 'values' of the corresponding column of the matrix.
  std::vector<Int> column_offsets;

  // A vector of length 'num_entries'; each segment, column_offsets[j] to
  // column_offsets[j + 1] - 1, contains the row indices for the j'th column.
  std::vector<Int> indices;
};

// A column-major lower-triangular sparse matrix.
template <class Field>
struct ScalarLowerFactor {
  // The nonzero structure of each column in the factor.
  ScalarLowerStructure structure;

  // A vector of length 'num_entries'; each segment, column_offsets[j] to
  // column_offsets[j + 1] - 1, contains the (typically nonzero) entries for
  // the j'th column.
  std::vector<Field> values;
};

// A diagonal matrix representing the 'D' in an L D L' factorization.
template <class Field>
struct ScalarDiagonalFactor {
  std::vector<Field> values;
};

enum SymmetricFactorizationType {
  // Computes lower-triangular L such that P A P' = L L'.
  kCholeskyFactorization,

  // Computes unit-lower triangular L and diagonal D such that P A P' = L D L'.
  kLDLAdjointFactorization,

  // Computes unit-lower triangular L and diagonal D such that P A P' = L D L^T.
  kLDLTransposeFactorization,
};

// A representation of a non-supernodal LDL' factorization.
template <class Field>
struct ScalarLDLFactorization {
  // Marks the type of factorization employed.
  SymmetricFactorizationType factorization_type;

  // The unit lower-triangular factor, L.
  ScalarLowerFactor<Field> lower_factor;

  // The diagonal factor, D.
  ScalarDiagonalFactor<Field> diagonal_factor;

  // If non-empty, the permutation mapping the original matrix ordering into the
  // factorization ordering.
  std::vector<Int> permutation;

  // If non-empty, the inverse of the permutation mapping the original matrix
  // ordering into the factorization ordering.
  std::vector<Int> inverse_permutation;
};

// Configuration options for non-supernodal LDL' factorization.
struct ScalarLDLControl {
  // The type of factorization to be performed.
  SymmetricFactorizationType factorization_type = kLDLAdjointFactorization;

  // The choice of either left-looking or up-looking LDL' factorization.
  LDLAlgorithm algorithm = kUpLookingLDL;
};

// Statistics from running an LDL' or Cholesky factorization.
struct LDLResult {
  // The number of successive legal pivots that were encountered during the
  // (attempted) factorization. If the factorization was successful, this will
  // equal the number of rows in the matrix.
  Int num_successful_pivots = 0;

  // The largest supernode size (after any relaxation).
  Int largest_supernode = 1;

  // The number of explicit entries in the factor.
  Int num_factorization_entries = 0;

  // The rough number of floating-point operations required for the
  // factorization.
  double num_factorization_flops = 0;
};

// Pretty-prints a column-oriented sparse lower-triangular matrix.
template <class Field>
void PrintLowerFactor(const ScalarLDLFactorization<Field>& factorization,
                      const std::string& label, std::ostream& os);

// Pretty-prints the diagonal factor of the L D L' factorization.
template <class Field>
void PrintDiagonalFactor(const ScalarLDLFactorization<Field>& factorization,
                         const std::string& label, std::ostream& os);

// Performs a non-supernodal LDL' factorization in the natural ordering.
template <class Field>
LDLResult LDL(const CoordinateMatrix<Field>& matrix,
              const ScalarLDLControl& control,
              ScalarLDLFactorization<Field>* factorization);

// Performs a non-supernodal LDL' factorization in a permuted ordering.
template <class Field>
LDLResult LDL(const CoordinateMatrix<Field>& matrix,
              const std::vector<Int>& permutation,
              const std::vector<Int>& inverse_permutation,
              const ScalarLDLControl& control,
              ScalarLDLFactorization<Field>* factorization);

// Solve (P A P') (P X) = (P B) via the substitution (L D L') (P X) = (P B) or
// (L D L^T) (P X) = (P B).
template <class Field>
void LDLSolve(const ScalarLDLFactorization<Field>& factorization,
              BlasMatrix<Field>* matrix);

// Solves L X = B using a unit-lower triangular matrix L.
template <class Field>
void LowerTriangularSolve(const ScalarLDLFactorization<Field>& factorization,
                          BlasMatrix<Field>* matrix);

// Solves D X = B using a diagonal matrix D.
template <class Field>
void DiagonalSolve(const ScalarLDLFactorization<Field>& factorization,
                   BlasMatrix<Field>* matrix);

// Solves L' X = B or L^T X = B using a unit-lower triangular matrix L.
template <class Field>
void LowerTransposeTriangularSolve(
    const ScalarLDLFactorization<Field>& factorization,
    BlasMatrix<Field>* matrix);

}  // namespace catamari

#include "catamari/scalar_ldl-impl.hpp"

#endif  // ifndef CATAMARI_SCALAR_LDL_H_
