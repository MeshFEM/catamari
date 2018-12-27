/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SUPERNODAL_LDL_SOLVE_IMPL_H_
#define CATAMARI_SUPERNODAL_LDL_SOLVE_IMPL_H_

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"

#include "catamari/supernodal_ldl/solve.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
void Solve(const Factorization<Field>& factorization,
           BlasMatrix<Field>* matrix) {
  const bool have_permutation = !factorization.permutation.empty();

  // Reorder the input into the permutation of the factorization.
  if (have_permutation) {
    Permute(factorization.permutation, matrix);
  }

  LowerTriangularSolve(factorization, matrix);
  DiagonalSolve(factorization, matrix);
  LowerTransposeTriangularSolve(factorization, matrix);

  // Reverse the factorization permutation.
  if (have_permutation) {
    Permute(factorization.inverse_permutation, matrix);
  }
}

template <class Field>
void LowerTriangularSolve(const Factorization<Field>& factorization,
                          BlasMatrix<Field>* matrix) {
  const Int num_rhs = matrix->width;
  const Int num_supernodes = factorization.supernode_sizes.size();
  const LowerFactor<Field>& lower_factor = *factorization.lower_factor;
  const DiagonalFactor<Field>& diagonal_factor = *factorization.diagonal_factor;
  const bool is_cholesky =
      factorization.factorization_type == kCholeskyFactorization;

  std::vector<Field> workspace(factorization.max_degree * num_rhs, Field{0});

  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const ConstBlasMatrix<Field> triangular_matrix =
        diagonal_factor.blocks[supernode];

    const Int supernode_size = factorization.supernode_sizes[supernode];
    const Int supernode_start = factorization.supernode_starts[supernode];
    BlasMatrix<Field> matrix_supernode =
        matrix->Submatrix(supernode_start, 0, supernode_size, num_rhs);

    // Solve against the diagonal block of the supernode.
    if (is_cholesky) {
      LeftLowerTriangularSolves(triangular_matrix, &matrix_supernode);
    } else {
      LeftLowerUnitTriangularSolves(triangular_matrix, &matrix_supernode);
    }

    const ConstBlasMatrix<Field> subdiagonal = lower_factor.blocks[supernode];
    if (!subdiagonal.height) {
      continue;
    }

    // Handle the external updates for this supernode.
    const Int* indices = lower_factor.Structure(supernode);
    if (supernode_size >=
        factorization.forward_solve_out_of_place_supernode_threshold) {
      // Perform an out-of-place GEMV.
      BlasMatrix<Field> work_matrix;
      work_matrix.height = subdiagonal.height;
      work_matrix.width = num_rhs;
      work_matrix.leading_dim = subdiagonal.height;
      work_matrix.data = workspace.data();

      // Store the updates in the workspace.
      MatrixMultiplyNormalNormal(Field{-1}, subdiagonal,
                                 matrix_supernode.ToConst(), Field{0},
                                 &work_matrix);

      // Accumulate the workspace into the solution matrix.
      for (Int j = 0; j < num_rhs; ++j) {
        for (Int i = 0; i < subdiagonal.height; ++i) {
          const Int row = indices[i];
          matrix->Entry(row, j) += work_matrix(i, j);
        }
      }
    } else {
      for (Int j = 0; j < num_rhs; ++j) {
        for (Int k = 0; k < supernode_size; ++k) {
          const Field& eta = matrix_supernode(k, j);
          for (Int i = 0; i < subdiagonal.height; ++i) {
            const Int row = indices[i];
            matrix->Entry(row, j) -= subdiagonal(i, k) * eta;
          }
        }
      }
    }
  }
}

template <class Field>
void DiagonalSolve(const Factorization<Field>& factorization,
                   BlasMatrix<Field>* matrix) {
  const Int num_rhs = matrix->width;
  const Int num_supernodes = factorization.supernode_sizes.size();
  const DiagonalFactor<Field>& diagonal_factor = *factorization.diagonal_factor;
  const bool is_cholesky =
      factorization.factorization_type == kCholeskyFactorization;
  if (is_cholesky) {
    // D is the identity.
    return;
  }

  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const ConstBlasMatrix<Field> diagonal_matrix =
        diagonal_factor.blocks[supernode];

    const Int supernode_size = factorization.supernode_sizes[supernode];
    const Int supernode_start = factorization.supernode_starts[supernode];
    BlasMatrix<Field> matrix_supernode =
        matrix->Submatrix(supernode_start, 0, supernode_size, num_rhs);

    // Handle the diagonal-block portion of the supernode.
    for (Int j = 0; j < num_rhs; ++j) {
      for (Int i = 0; i < supernode_size; ++i) {
        matrix_supernode(i, j) /= diagonal_matrix(i, i);
      }
    }
  }
}

template <class Field>
void LowerTransposeTriangularSolve(const Factorization<Field>& factorization,
                                   BlasMatrix<Field>* matrix) {
  const Int num_rhs = matrix->width;
  const Int num_supernodes = factorization.supernode_sizes.size();
  const LowerFactor<Field>& lower_factor = *factorization.lower_factor;
  const DiagonalFactor<Field>& diagonal_factor = *factorization.diagonal_factor;
  const SymmetricFactorizationType factorization_type =
      factorization.factorization_type;
  const bool is_selfadjoint = factorization_type != kLDLTransposeFactorization;

  std::vector<Field> packed_input_buf(factorization.max_degree * num_rhs);

  for (Int supernode = num_supernodes - 1; supernode >= 0; --supernode) {
    const Int supernode_size = factorization.supernode_sizes[supernode];
    const Int supernode_start = factorization.supernode_starts[supernode];
    const Int* indices = lower_factor.Structure(supernode);

    BlasMatrix<Field> matrix_supernode =
        matrix->Submatrix(supernode_start, 0, supernode_size, num_rhs);

    const ConstBlasMatrix<Field> subdiagonal = lower_factor.blocks[supernode];
    if (subdiagonal.height) {
      // Handle the external updates for this supernode.
      if (supernode_size >=
          factorization.backward_solve_out_of_place_supernode_threshold) {
        // Fill the work matrix.
        BlasMatrix<Field> work_matrix;
        work_matrix.height = subdiagonal.height;
        work_matrix.width = num_rhs;
        work_matrix.leading_dim = subdiagonal.height;
        work_matrix.data = packed_input_buf.data();
        for (Int j = 0; j < num_rhs; ++j) {
          for (Int i = 0; i < subdiagonal.height; ++i) {
            const Int row = indices[i];
            work_matrix(i, j) = matrix->Entry(row, j);
          }
        }

        if (is_selfadjoint) {
          MatrixMultiplyAdjointNormal(Field{-1}, subdiagonal,
                                      work_matrix.ToConst(), Field{1},
                                      &matrix_supernode);
        } else {
          MatrixMultiplyTransposeNormal(Field{-1}, subdiagonal,
                                        work_matrix.ToConst(), Field{1},
                                        &matrix_supernode);
        }
      } else {
        for (Int k = 0; k < supernode_size; ++k) {
          for (Int i = 0; i < subdiagonal.height; ++i) {
            const Int row = indices[i];
            for (Int j = 0; j < num_rhs; ++j) {
              if (is_selfadjoint) {
                matrix_supernode(k, j) -=
                    Conjugate(subdiagonal(i, k)) * matrix->Entry(row, j);
              } else {
                matrix_supernode(k, j) -=
                    subdiagonal(i, k) * matrix->Entry(row, j);
              }
            }
          }
        }
      }
    }

    // Solve against the diagonal block of this supernode.
    const ConstBlasMatrix<Field> triangular_matrix =
        diagonal_factor.blocks[supernode];
    if (factorization_type == kCholeskyFactorization) {
      LeftLowerAdjointTriangularSolves(triangular_matrix, &matrix_supernode);
    } else if (factorization_type == kLDLAdjointFactorization) {
      LeftLowerAdjointUnitTriangularSolves(triangular_matrix,
                                           &matrix_supernode);
    } else {
      LeftLowerTransposeUnitTriangularSolves(triangular_matrix,
                                             &matrix_supernode);
    }
  }
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_SUPERNODAL_LDL_SOLVE_IMPL_H_
