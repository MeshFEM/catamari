/*
 * Copyright (c) 2018-2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SCALAR_LDL_FACTORIZATION_H_
#define CATAMARI_LDL_SCALAR_LDL_FACTORIZATION_H_

#include <ostream>

#include "catamari/blas_matrix_view.hpp"
#include "catamari/buffer.hpp"
#include "catamari/coordinate_matrix.hpp"
#include "catamari/integers.hpp"
#include "catamari/symmetric_ordering.hpp"

namespace catamari {

enum SymmetricFactorizationType {
  // Computes lower-triangular L such that P A P' = L L'.
  kCholeskyFactorization,

  // Computes unit-lower triangular L and diagonal D such that P A P' = L D L'.
  kLDLAdjointFactorization,

  // Computes unit-lower triangular L and diagonal D such that P A P' = L D L^T.
  kLDLTransposeFactorization,
};

enum LDLAlgorithm {
  // A left-looking LDL factorization. Cf. Section 4.8 of Tim Davis,
  // "Direct Methods for Sparse Linear Systems".
  kLeftLookingLDL,

  // An up-looking LDL factorization. Cf. Section 4.7 of Tim Davis,
  // "Direct Methods for Sparse Linear Systems".
  kUpLookingLDL,

  // A right-looking (multifrontal) factorization.
  kRightLookingLDL,
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

  // The rough number of flops required to factorize the diagonal blocks.
  //
  // In the case of complex factorizations, this is in terms of the number of
  // real flops.
  double num_diagonal_flops = 0;

  // The rough number of flops required to solve against the diagonal blocks
  // to update the subdiagonals.
  //
  // In the case of complex factorizations, this is in terms of the number of
  // real flops.
  double num_subdiag_solve_flops = 0;

  // The rough number of flops required to form the Schur complements.
  //
  // In the case of complex factorizations, this is in terms of the number of
  // real flops.
  double num_schur_complement_flops = 0;

  // The rough number of floating-point operations required for the
  // factorization.
  //
  // In the case of complex factorizations, this is in terms of the number of
  // real flops.
  double num_factorization_flops = 0;
};

namespace scalar_ldl {

// Configuration options for non-supernodal LDL' factorization.
struct Control {
  // The type of factorization to be performed.
  SymmetricFactorizationType factorization_type = kLDLAdjointFactorization;

  // The choice of either left-looking or up-looking LDL' factorization.
  // There is currently no scalar multifrontal support.
  LDLAlgorithm algorithm = kUpLookingLDL;
};

// The nonzero patterns below the diagonal of the lower-triangular factor.
struct LowerStructure {
  // A vector of length 'num_rows + 1'; each entry corresponds to the offset
  // in 'indices' and 'values' of the corresponding column of the matrix.
  Buffer<Int> column_offsets;

  // A vector of length 'num_entries'; each segment, column_offsets[j] to
  // column_offsets[j + 1] - 1, contains the row indices for the j'th column.
  Buffer<Int> indices;

  // Returns the starting index of a column's structure in 'indices'.
  Int ColumnOffset(Int column) const { return column_offsets[column]; }

  // Returns the external degree of the given column vertex.
  Int Degree(Int column) const {
    return ColumnOffset(column + 1) - ColumnOffset(column);
  }

  // Returns a pointer to a column's structure.
  Int* ColumnBeg(Int column) { return &indices[column_offsets[column]]; }

  // Returns an immutable pointer to a column's structure.
  const Int* ColumnBeg(Int column) const {
    return &indices[column_offsets[column]];
  }

  // Returns a pointer to the end of a column's structure.
  Int* ColumnEnd(Int column) { return &indices[column_offsets[column + 1]]; }

  // Returns an immutable pointer to the end of a column's structure.
  const Int* ColumnEnd(Int column) const {
    return &indices[column_offsets[column + 1]];
  }
};

// A column-major lower-triangular sparse matrix.
template <class Field>
struct LowerFactor {
  // The nonzero structure of each column in the factor.
  LowerStructure structure;

  // A vector of length 'num_entries'; each segment, column_offsets[j] to
  // column_offsets[j + 1] - 1, contains the (typically nonzero) entries for
  // the j'th column.
  Buffer<Field> values;
};

// A diagonal matrix representing the 'D' in an L D L' factorization.
template <class Field>
struct DiagonalFactor {
  Buffer<Field> values;
};

// A representation of a non-supernodal LDL' factorization.
template <class Field>
class Factorization {
 public:
  // Marks the type of factorization employed.
  SymmetricFactorizationType factorization_type;

  // The algorithm used for the factorization.
  LDLAlgorithm algorithm;

  // The unit lower-triangular factor, L.
  LowerFactor<Field> lower_factor;

  // The diagonal factor, D.
  DiagonalFactor<Field> diagonal_factor;

  // The reordering of the input matrix used for the factorization.
  SymmetricOrdering ordering;

  // Set the factor to the L L^T, L D L^T, or L D L' factorization of the given
  // permutation of the given matrix.
  LDLResult Factor(const CoordinateMatrix<Field>& matrix,
                   const SymmetricOrdering& ordering, const Control& control);

  // Factors a matrix which has the same sparsity pattern as the previously
  // factored matrix.
  LDLResult RefactorWithFixedSparsityPattern(
      const CoordinateMatrix<Field>& matrix);

  // Pretty-prints the diagonal matrix.
  void PrintDiagonalFactor(const std::string& label, std::ostream& os) const;

  // Pretty-prints the lower-triangular matrix.
  void PrintLowerFactor(const std::string& label, std::ostream& os) const;

  // Solves a linear system using the factorization.
  void Solve(BlasMatrixView<Field>* right_hand_sides) const;

  // Solves against the lower-triangular factor.
  void LowerTriangularSolve(BlasMatrixView<Field>* right_hand_sides) const;

  // Solves against the diagonal factor.
  void DiagonalSolve(BlasMatrixView<Field>* right_hand_sides) const;

  // Solves against the transpose (or adjoint) of the lower factor.
  void LowerTransposeTriangularSolve(
      BlasMatrixView<Field>* right_hand_sides) const;

 private:
  // The temporary state used by the left-looking factorization.
  struct LeftLookingState {
    // Since we will sequentially access each of the entries in each column of
    // L during the updates of the active column, we can avoid the need for
    // binary search by maintaining a separate counter for each column.
    Buffer<Int> column_update_ptrs;

    // An integer workspace for storing the indices in the current row pattern.
    Buffer<Int> row_structure;

    // A data structure for marking whether or not an index is in the pattern
    // of the active row of the lower-triangular factor.
    Buffer<Int> pattern_flags;
  };

  // The temporary state used by the up-looking factorization.
  struct UpLookingState {
    // An array for holding the active index to insert the new entry of each
    // column into.
    Buffer<Int> column_update_ptrs;

    // A data structure for marking whether or not an index is in the pattern
    // of the active row of the lower-triangular factor.
    Buffer<Int> pattern_flags;

    // Set up an integer workspace that could hold any row nonzero pattern.
    Buffer<Int> row_structure;

    // Set up a workspace for performing a triangular solve against a row of the
    // input matrix.
    Buffer<Field> row_workspace;
  };

  // Performs a non-supernodal left-looking LDL' factorization.
  // Cf. Section 4.8 of Tim Davis, "Direct Methods for Sparse Linear Systems".
  //
  // The basic high-level algorithm is of the form:
  //   for k = 1:n
  //     L(k, k) = sqrt(A(k, k) - L(k, 1:k-1) * L(k, 1:k-1)');
  //     L(k+1:n, k) = (A(k+1:n, k) - L(k+1:n, 1:k-1) * L(k, 1:k-1)') / L(k, k);
  //   end
  LDLResult LeftLooking(const CoordinateMatrix<Field>& matrix);

  // Performs a non-supernodal up-looking LDL' factorization.
  // Cf. Section 4.7 of Tim Davis, "Direct Methods for Sparse Linear Systems".
  LDLResult UpLooking(const CoordinateMatrix<Field>& matrix);

  // Fill the factorization with the nonzeros from the (permuted) input matrix.
  void FillNonzeros(const CoordinateMatrix<Field>& matrix);

  // Initializes for running a left-looking factorization.
  void LeftLookingSetup(const CoordinateMatrix<Field>& matrix);

  // Initializes for running an up-looking factorization.
  void UpLookingSetup(const CoordinateMatrix<Field>& matrix);

  // For each index 'i' in the structure of each column 'column' of L formed
  // so far:
  //   L(row, i) -= (L(row, column) * d(column)) * conj(L(i, column)).
  // L(row, row) is similarly updated, within d, then L(row, column) is
  // finalized.
  void UpLookingRowUpdate(Int row, const Int* column_beg, const Int* column_end,
                          Int* column_update_ptrs, Field* row_workspace);
};

}  // namespace scalar_ldl
}  // namespace catamari

#include "catamari/ldl/scalar_ldl/factorization-impl.hpp"

#endif  // ifndef CATAMARI_LDL_SCALAR_LDL_FACTORIZATION_H_
