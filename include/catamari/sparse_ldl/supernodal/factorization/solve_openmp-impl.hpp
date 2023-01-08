/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_SOLVE_OPENMP_IMPL_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_SOLVE_OPENMP_IMPL_H_

#include <algorithm>
#include <catamari/dense_basic_linear_algebra-impl.hpp>
#include <queue>
#include <stdexcept>

#include <tbb/task_group.h>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"

#include "catamari/sparse_ldl/supernodal/factorization.hpp"

#include "../../../../../../../src/lib/MeshFEM/GlobalBenchmark.hh"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
void Factorization<Field>::OpenMPLowerSupernodalTrapezoidalSolve(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    BlasMatrixView<Field>* supernode_schur_complement) const {
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
  if (control_.supernodal_pivoting) {
    const ConstBlasMatrixView<Int> permutation =
        SupernodePermutation(supernode);
    InversePermute(permutation, &right_hand_sides_supernode);
  }
  if (is_cholesky) {
    LeftLowerTriangularSolvesDynamicBLASDispatch(triangular_right_hand_sides,
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
  if (true) {
      MatrixMultiplyNormalNormal(Field{-1}, subdiagonal,
                                 right_hand_sides_supernode.ToConst(), Field{1},
                                 supernode_schur_complement);
  }
  else {
#if 0
    eigenMap(*supernode_schur_complement).noalias() -= eigenMap(subdiagonal) * eigenMap(right_hand_sides_supernode);
#else
    const Int output_height    = supernode_schur_complement->height;
    const Int output_width     = supernode_schur_complement->width;
    const Int contraction_size = subdiagonal.width;
    for (Int j = 0; j < output_width; ++j) {
        Field *out_col = supernode_schur_complement->Pointer(0, j);
        const Field *right_col = right_hand_sides_supernode.Pointer(0, j);
        for (Int i = 0; i < output_height; ++i) {
            Field output_entry = out_col[i];
            for (Int k = 0; k < contraction_size; ++k)
                output_entry -= subdiagonal(i, k) * right_col[k];
            out_col[i] = output_entry;
        }
    }
#endif
  }
}

template <class Field>
void Factorization<Field>::OpenMPLowerTriangularSolveRecursion(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    RightLookingSharedState<Field>* shared_state, int level) const {
  // Recurse on this supernode's children.
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];

  auto processChild = [&, shared_state, right_hand_sides, level](Int child_index) {
      const Int child = ordering_.assembly_forest.children[child_index];
      OpenMPLowerTriangularSolveRecursion(child, right_hand_sides, shared_state, level + 1);
  };

  if ((child_end - child_beg) > 1 && (level < 8)) { // JP: avoid excessively fine-grained parallelism; TODO: use flop-based threshold
      tbb::task_group group;
      for (Int child_index = child_beg; child_index < child_end - 1; ++child_index) {
          group.run([&processChild, child_index]() { processChild(child_index); });
      }
      processChild(child_end - 1);
      group.wait();
  }
  else {
      for (Int child_index = child_beg; child_index < child_end; ++child_index) {
          processChild(child_index);
      }
  }

  const Int supernode_start = ordering_.supernode_offsets[supernode];
  const Int* main_indices = lower_factor_->StructureBeg(supernode);

  // Merge the child Schur complements into the parent.
  const Int degree = lower_factor_->blocks[supernode].height;
  const Int num_rhs = right_hand_sides->width;

  BlasMatrixView<Field>& main_right_hand_sides = shared_state->schur_complements[supernode];
  // eigenMap(main_right_hand_sides).setZero(); // <----- this doesn't account for the stride/height mismatch in main_right_hand_sides!!!!!
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < main_right_hand_sides.height; ++i)
        main_right_hand_sides(i, j) = 0;
  }

  auto &ncdi = ordering_.assembly_forest.num_child_diag_indices;
  auto  &cri = ordering_.assembly_forest.child_rel_indices;

  for (Int child_index = child_beg; child_index < child_end; ++child_index) {
    const Int child = ordering_.assembly_forest.children[child_index];
    const Int* child_indices = lower_factor_->StructureBeg(child);
    BlasMatrixView<Field>& child_right_hand_sides = shared_state->schur_complements[child];
    const Int child_degree = child_right_hand_sides.height;

    const Int &num_child_diag_indices = ncdi[child];
    const Buffer<Int> &child_rel_indices = cri[child];

#if 1
    for (Int j = 0; j < num_rhs; ++j) {
        Field* crhs_col = child_right_hand_sides.Pointer(0, j);
        Field*  rhs_col = right_hand_sides->Pointer(0, j);
        Field* mrhs_col = main_right_hand_sides.Pointer(0, j);
        for (Int i = 0; i < num_child_diag_indices; ++i)
            rhs_col[child_indices[i]] += crhs_col[i];
        for (Int i = num_child_diag_indices; i < child_degree; ++i)
            mrhs_col[child_rel_indices[i]] += crhs_col[i];
    }
#else
    for (Int j = 0; j < num_rhs; ++j) {
        for (Int i = 0; i < num_child_diag_indices; ++i) {
            const Int row = child_indices[i];
            right_hand_sides->Entry(row, j) += child_right_hand_sides(i, j);
        }
        for (Int i = num_child_diag_indices, main_i = 0; i < child_degree; ++i) {
            const Int row = child_indices[i];
            while (main_indices[main_i] != row) ++main_i;
            main_right_hand_sides(main_i, j) += child_right_hand_sides(i, j);
        }
    }
#endif
  }

  // Perform this supernode's trapezoidal solve.
  OpenMPLowerSupernodalTrapezoidalSolve(supernode, right_hand_sides,
                                        &main_right_hand_sides);
}

template <class Field>
void Factorization<Field>::OpenMPLowerTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides,
    RightLookingSharedState<Field>* shared_state) const {
  BENCHMARK_SCOPED_TIMER_SECTION timer("OpenMPLowerTriangularSolve");

  // Recurse on each tree in the elimination forest.
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = ordering_.assembly_forest.roots[root_index];
    OpenMPLowerTriangularSolveRecursion(root, right_hand_sides, shared_state, 0);
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
  // TODO(JP): re-parallelize
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
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
    RightLookingSharedState<Field>* shared_state, int level, tbb::task_group &tg) const {
    // Perform this supernode's trapezoidal solve.
    // TODO(Jack Poulson): Add OpenMP support into the trapezoidal solve.

    // Do the work for this supernode.
    LowerTransposeSupernodalTrapezoidalSolve(supernode, right_hand_sides, shared_state->schur_complements[supernode]);

    auto processChild = [right_hand_sides, shared_state, level, &tg, this](Int child_index) {
        const Int child = ordering_.assembly_forest.children[child_index];
        OpenMPLowerTransposeTriangularSolveRecursion(child, right_hand_sides, shared_state, level + 1, tg);
    };

    // Tail recurse on this supernode's children.
    const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
    const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
    const Int numChildren = child_end - child_beg;
    if (numChildren <= 1 || level > 9) { // Avoid spawning unnecessary threads...
        for (Int child_index = child_beg; child_index < child_end; ++child_index)
            processChild(child_index);
        return;
    }

    for (Int child_index = child_beg; child_index < child_end - 1; ++child_index)
        tg.run([processChild, child_index]() { processChild(child_index); });
    processChild(child_end - 1);
}

template <class Field>
void Factorization<Field>::OpenMPLowerTransposeTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides,
    RightLookingSharedState<Field>* shared_state) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("OpenMPLowerTransposeTriangularSolve");

    const Int num_roots = ordering_.assembly_forest.roots.Size();
    if (num_roots == 0) return;

    // Tail recurse from each root of the elimination forest.
    tbb::task_group tg;
    for (Int root_index = 0; root_index < num_roots - 1; ++root_index) {
        tg.run([right_hand_sides, shared_state, root_index, &tg, this]() {
            OpenMPLowerTransposeTriangularSolveRecursion(ordering_.assembly_forest.roots[root_index], right_hand_sides, shared_state, 0, tg);
        });
    }
    OpenMPLowerTransposeTriangularSolveRecursion(ordering_.assembly_forest.roots[num_roots - 1], right_hand_sides, shared_state, 0, tg);
    tg.wait();
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef
        // CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_SOLVE_OPENMP_IMPL_H_
