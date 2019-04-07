/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_SOLVE_OPENMP_IMPL_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_SOLVE_OPENMP_IMPL_H_
#ifdef CATAMARI_OPENMP

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"

#include "catamari/sparse_ldl/supernodal/factorization.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
void Factorization<Field>::OpenMPLowerSupernodalTrapezoidalSolve(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    RightLookingSharedState<Field>* shared_state) const {
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

  // Store the updates in the workspace.
  MatrixMultiplyNormalNormal(Field{-1}, subdiagonal,
                             right_hand_sides_supernode.ToConst(), Field{1},
                             &shared_state->schur_complements[supernode]);
}

template <class Field>
void Factorization<Field>::OpenMPLowerTriangularSolveRecursion(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    RightLookingSharedState<Field>* shared_state) const {
  // Recurse on this supernode's children.
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;
  #pragma omp taskgroup
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    #pragma omp task default(none) firstprivate(child) \
        shared(right_hand_sides, shared_state)
    OpenMPLowerTriangularSolveRecursion(child, right_hand_sides, shared_state);
  }

  const Int supernode_start = ordering_.supernode_sizes[supernode];
  const Int supernode_size = ordering_.supernode_offsets[supernode];
  const Int* main_indices = lower_factor_->StructureBeg(supernode);

  // Merge the child Schur complements into the parent.
  const Int degree = lower_factor_->blocks[supernode].height;
  const Int num_rhs = right_hand_sides->width;
  const Int workspace_size = degree * num_rhs;
  shared_state->schur_complement_buffers[supernode].Resize(workspace_size,
                                                           Field{0});
  BlasMatrixView<Field>& main_right_hand_sides =
      shared_state->schur_complements[supernode];
  main_right_hand_sides.height = degree;
  main_right_hand_sides.width = num_rhs;
  main_right_hand_sides.leading_dim = degree;
  main_right_hand_sides.data =
      shared_state->schur_complement_buffers[supernode].Data();
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    const Int* child_indices = lower_factor_->StructureBeg(child);
    BlasMatrixView<Field>& child_right_hand_sides =
        shared_state->schur_complements[child];
    const Int child_degree = child_right_hand_sides.height;

    Int i_rel = supernode_size;
    for (Int i = 0; i < child_degree; ++i) {
      const Int row = child_indices[i];
      if (row < supernode_start + supernode_size) {
        for (Int j = 0; j < num_rhs; ++j) {
          right_hand_sides->Entry(row, j) += child_right_hand_sides(i, j);
        }
      } else {
        while (main_indices[i_rel - supernode_size] != row) {
          ++i_rel;
        }
        const Int main_rel = i_rel - supernode_size;
        for (Int j = 0; j < num_rhs; ++j) {
          main_right_hand_sides(main_rel, j) += child_right_hand_sides(i, j);
        }
      }
    }

    shared_state->schur_complement_buffers[child].Clear();
    child_right_hand_sides.height = 0;
    child_right_hand_sides.width = 0;
    child_right_hand_sides.leading_dim = 0;
    child_right_hand_sides.data = nullptr;
  }

  // Perform this supernode's trapezoidal solve.
  OpenMPLowerSupernodalTrapezoidalSolve(supernode, right_hand_sides,
                                        shared_state);
}

template <class Field>
void Factorization<Field>::OpenMPLowerTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  // Set up the shared state.
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  RightLookingSharedState<Field> shared_state;
  shared_state.schur_complement_buffers.Resize(num_supernodes);
  shared_state.schur_complements.Resize(num_supernodes);

  // Recurse on each tree in the elimination forest.
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  #pragma omp taskgroup
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = ordering_.assembly_forest.roots[root_index];
    #pragma omp task default(none) firstprivate(root) \
        shared(right_hand_sides, shared_state)
    OpenMPLowerTriangularSolveRecursion(root, right_hand_sides, &shared_state);
  }
}

template <class Field>
void Factorization<Field>::OpenMPDiagonalSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  if (control_.factorization_type == kCholeskyFactorization) {
    // D is the identity.
    return;
  }

  const SymmetricOrdering* ordering_ptr = &ordering_;
  const DiagonalFactor<Field>* diagonal_factor_ptr = diagonal_factor_.get();

  const Int num_supernodes = ordering_.supernode_sizes.Size();
  #pragma omp taskgroup
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    #pragma omp task default(none) firstprivate(supernode) \
        shared(right_hand_sides, ordering_ptr, diagonal_factor_ptr)
    {
      const ConstBlasMatrixView<Field> diagonal_right_hand_sides =
          diagonal_factor_ptr->blocks[supernode];

      const Int num_rhs = right_hand_sides->width;
      const Int supernode_size = ordering_ptr->supernode_sizes[supernode];
      const Int supernode_start = ordering_ptr->supernode_offsets[supernode];
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
}

template <class Field>
void Factorization<Field>::OpenMPLowerTransposeTriangularSolveRecursion(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    Buffer<Buffer<Field>>* private_packed_input_bufs) const {
  // Perform this supernode's trapezoidal solve.
  const int thread = omp_get_thread_num();
  Buffer<Field>* packed_input_buf = &(*private_packed_input_bufs)[thread];
  LowerTransposeSupernodalTrapezoidalSolve(supernode, right_hand_sides,
                                           packed_input_buf);

  // Recurse on this supernode's children.
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;
  #pragma omp taskgroup
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    #pragma omp task default(none) firstprivate(child) \
        shared(right_hand_sides, private_packed_input_bufs)
    OpenMPLowerTransposeTriangularSolveRecursion(child, right_hand_sides,
                                                 private_packed_input_bufs);
  }
}

template <class Field>
void Factorization<Field>::OpenMPLowerTransposeTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  // Allocate each thread's workspace.
  const int max_threads = omp_get_max_threads();
  const Int workspace_size = max_degree_ * right_hand_sides->width;
  Buffer<Buffer<Field>> private_packed_input_bufs(max_threads);
  #pragma omp taskgroup
  for (int t = 0; t < max_threads; ++t) {
    #pragma omp task default(none) firstprivate(t, workspace_size) \
        shared(private_packed_input_bufs)
    private_packed_input_bufs[t].Resize(workspace_size);
  }

  // Recurse from each root of the elimination forest.
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  #pragma omp taskgroup
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = ordering_.assembly_forest.roots[root_index];
    #pragma omp task default(none) firstprivate(root) \
        shared(right_hand_sides, private_packed_input_bufs)
    OpenMPLowerTransposeTriangularSolveRecursion(root, right_hand_sides,
                                                 &private_packed_input_bufs);
  }
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifdef CATAMARI_OPENMP
#endif  // ifndef
        // CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_SOLVE_OPENMP_IMPL_H_
