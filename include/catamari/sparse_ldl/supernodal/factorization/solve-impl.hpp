/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_SOLVE_IMPL_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_SOLVE_IMPL_H_

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"

#include "catamari/sparse_ldl/supernodal/factorization.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
void Factorization<Field>::Solve(
    BlasMatrixView<Field>* right_hand_sides) const {
  const bool have_permutation = !ordering_.permutation.Empty();
  // Reorder the input into the permutation of the factorization.
  if (have_permutation) {
    Permute(ordering_.permutation, right_hand_sides);
  }

#ifdef CATAMARI_OPENMP
  if (omp_get_max_threads() > 1) {
    const int old_max_threads = GetMaxBlasThreads();
    SetNumBlasThreads(1);

    #pragma omp parallel
    #pragma omp single
    {
      OpenMPLowerTriangularSolve(right_hand_sides);
      OpenMPDiagonalSolve(right_hand_sides);
      OpenMPLowerTransposeTriangularSolve(right_hand_sides);
    }

    SetNumBlasThreads(old_max_threads);
  } else {
    LowerTriangularSolve(right_hand_sides);
    DiagonalSolve(right_hand_sides);
    LowerTransposeTriangularSolve(right_hand_sides);
  }
#else
  LowerTriangularSolve(right_hand_sides);
  DiagonalSolve(right_hand_sides);
  LowerTransposeTriangularSolve(right_hand_sides);
#endif  // ifdef CATAMARI_OPENMP

  // Reverse the factorization permutation.
  if (have_permutation) {
    Permute(ordering_.inverse_permutation, right_hand_sides);
  }
}

template <class Field>
void Factorization<Field>::LowerSupernodalTrapezoidalSolve(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    Buffer<Field>* workspace) const {
  // Eliminate this supernode.
  const Int num_rhs = right_hand_sides->width;
  const bool is_cholesky =
      control_.factorization_type == kCholeskyFactorization;
  const ConstBlasMatrixView<Field> triangular_right_hand_sides =
      diagonal_factor_->blocks[supernode];

  const Int supernode_size = ordering_.supernode_sizes[supernode];
  const Int supernode_start = ordering_.supernode_offsets[supernode];
  BlasMatrixView<Field> right_hand_sides_supernode =
      right_hand_sides->Submatrix(supernode_start, 0, supernode_size, num_rhs);

  // Solve against the diagonal block of the supernode.
  if (is_cholesky) {
    LeftLowerTriangularSolves(triangular_right_hand_sides,
                              &right_hand_sides_supernode);
  } else {
    LeftLowerUnitTriangularSolves(triangular_right_hand_sides,
                                  &right_hand_sides_supernode);
  }

  const ConstBlasMatrixView<Field> subdiagonal =
      lower_factor_->blocks[supernode];
  if (!subdiagonal.height) {
    return;
  }

  // Handle the external updates for this supernode.
  const Int* indices = lower_factor_->StructureBeg(supernode);
  if (supernode_size >=
      control_.forward_solve_out_of_place_supernode_threshold) {
    // Perform an out-of-place GEMM.
    BlasMatrixView<Field> work_right_hand_sides;
    work_right_hand_sides.height = subdiagonal.height;
    work_right_hand_sides.width = num_rhs;
    work_right_hand_sides.leading_dim = subdiagonal.height;
    work_right_hand_sides.data = workspace->Data();

    // Store the updates in the workspace.
    MatrixMultiplyNormalNormal(Field{-1}, subdiagonal,
                               right_hand_sides_supernode.ToConst(), Field{0},
                               &work_right_hand_sides);

    // Accumulate the workspace into the solution right_hand_sides.
    for (Int j = 0; j < num_rhs; ++j) {
      for (Int i = 0; i < subdiagonal.height; ++i) {
        const Int row = indices[i];
        right_hand_sides->Entry(row, j) += work_right_hand_sides(i, j);
      }
    }
  } else {
    for (Int j = 0; j < num_rhs; ++j) {
      for (Int k = 0; k < supernode_size; ++k) {
        const Field& eta = right_hand_sides_supernode(k, j);
        for (Int i = 0; i < subdiagonal.height; ++i) {
          const Int row = indices[i];
          right_hand_sides->Entry(row, j) -= subdiagonal(i, k) * eta;
        }
      }
    }
  }
}

template <class Field>
void Factorization<Field>::LowerTriangularSolveRecursion(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    Buffer<Field>* workspace) const {
  // Recurse on this supernode's children.
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    LowerTriangularSolveRecursion(child, right_hand_sides, workspace);
  }

  // Perform this supernode's trapezoidal solve.
  LowerSupernodalTrapezoidalSolve(supernode, right_hand_sides, workspace);
}

template <class Field>
void Factorization<Field>::LowerTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  // Allocate the workspace.
  const Int workspace_size = max_degree_ * right_hand_sides->width;
  Buffer<Field> workspace(workspace_size, Field{0});

  // Recurse on each tree in the elimination forest.
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = ordering_.assembly_forest.roots[root_index];
    LowerTriangularSolveRecursion(root, right_hand_sides, &workspace);
  }
}

template <class Field>
void Factorization<Field>::DiagonalSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  const Int num_rhs = right_hand_sides->width;
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const bool is_cholesky =
      control_.factorization_type == kCholeskyFactorization;
  if (is_cholesky) {
    // D is the identity.
    return;
  }

  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const ConstBlasMatrixView<Field> diagonal_right_hand_sides =
        diagonal_factor_->blocks[supernode];

    const Int supernode_size = ordering_.supernode_sizes[supernode];
    const Int supernode_start = ordering_.supernode_offsets[supernode];
    BlasMatrixView<Field> right_hand_sides_supernode =
        right_hand_sides->Submatrix(supernode_start, 0, supernode_size,
                                    num_rhs);

    // Handle the diagonal-block portion of the supernode.
    for (Int j = 0; j < num_rhs; ++j) {
      for (Int i = 0; i < supernode_size; ++i) {
        right_hand_sides_supernode(i, j) /= diagonal_right_hand_sides(i, i);
      }
    }
  }
}

template <class Field>
void Factorization<Field>::LowerTransposeSupernodalTrapezoidalSolve(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    Buffer<Field>* packed_input_buf) const {
  const Int num_rhs = right_hand_sides->width;
  const bool is_selfadjoint =
      control_.factorization_type != kLDLTransposeFactorization;
  const Int supernode_size = ordering_.supernode_sizes[supernode];
  const Int supernode_start = ordering_.supernode_offsets[supernode];
  const Int* indices = lower_factor_->StructureBeg(supernode);

  BlasMatrixView<Field> right_hand_sides_supernode =
      right_hand_sides->Submatrix(supernode_start, 0, supernode_size, num_rhs);

  const ConstBlasMatrixView<Field> subdiagonal =
      lower_factor_->blocks[supernode];
  if (subdiagonal.height) {
    // Handle the external updates for this supernode.
    if (supernode_size >=
        control_.backward_solve_out_of_place_supernode_threshold) {
      // Fill the work right_hand_sides.
      BlasMatrixView<Field> work_right_hand_sides;
      work_right_hand_sides.height = subdiagonal.height;
      work_right_hand_sides.width = num_rhs;
      work_right_hand_sides.leading_dim = subdiagonal.height;
      work_right_hand_sides.data = packed_input_buf->Data();
      for (Int j = 0; j < num_rhs; ++j) {
        for (Int i = 0; i < subdiagonal.height; ++i) {
          const Int row = indices[i];
          work_right_hand_sides(i, j) = right_hand_sides->Entry(row, j);
        }
      }

      if (is_selfadjoint) {
        MatrixMultiplyAdjointNormal(Field{-1}, subdiagonal,
                                    work_right_hand_sides.ToConst(), Field{1},
                                    &right_hand_sides_supernode);
      } else {
        MatrixMultiplyTransposeNormal(Field{-1}, subdiagonal,
                                      work_right_hand_sides.ToConst(), Field{1},
                                      &right_hand_sides_supernode);
      }
    } else {
      for (Int k = 0; k < supernode_size; ++k) {
        for (Int i = 0; i < subdiagonal.height; ++i) {
          const Int row = indices[i];
          for (Int j = 0; j < num_rhs; ++j) {
            if (is_selfadjoint) {
              right_hand_sides_supernode(k, j) -=
                  Conjugate(subdiagonal(i, k)) *
                  right_hand_sides->Entry(row, j);
            } else {
              right_hand_sides_supernode(k, j) -=
                  subdiagonal(i, k) * right_hand_sides->Entry(row, j);
            }
          }
        }
      }
    }
  }

  // Solve against the diagonal block of this supernode.
  const ConstBlasMatrixView<Field> triangular_right_hand_sides =
      diagonal_factor_->blocks[supernode];
  if (control_.factorization_type == kCholeskyFactorization) {
    LeftLowerAdjointTriangularSolves(triangular_right_hand_sides,
                                     &right_hand_sides_supernode);
  } else if (control_.factorization_type == kLDLAdjointFactorization) {
    LeftLowerAdjointUnitTriangularSolves(triangular_right_hand_sides,
                                         &right_hand_sides_supernode);
  } else {
    LeftLowerTransposeUnitTriangularSolves(triangular_right_hand_sides,
                                           &right_hand_sides_supernode);
  }
}

template <class Field>
void Factorization<Field>::LowerTransposeTriangularSolveRecursion(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    Buffer<Field>* packed_input_buf) const {
  // Perform this supernode's trapezoidal solve.
  LowerTransposeSupernodalTrapezoidalSolve(supernode, right_hand_sides,
                                           packed_input_buf);

  // Recurse on this supernode's children.
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    LowerTransposeTriangularSolveRecursion(child, right_hand_sides,
                                           packed_input_buf);
  }
}

template <class Field>
void Factorization<Field>::LowerTransposeTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  // Allocate the workspace.
  const Int workspace_size = max_degree_ * right_hand_sides->width;
  Buffer<Field> packed_input_buf(workspace_size);

  // Recurse from each root of the elimination forest.
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = ordering_.assembly_forest.roots[root_index];
    LowerTransposeTriangularSolveRecursion(root, right_hand_sides,
                                           &packed_input_buf);
  }
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_SOLVE_IMPL_H_
