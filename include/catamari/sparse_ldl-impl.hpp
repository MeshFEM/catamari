/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.  */
#ifndef CATAMARI_SPARSE_LDL_IMPL_H_
#define CATAMARI_SPARSE_LDL_IMPL_H_

#include <memory>

#include "catamari/apply_sparse.hpp"
#include "catamari/blas_matrix.hpp"
#include "catamari/flush_to_zero.hpp"

#include "catamari/sparse_ldl.hpp"

namespace catamari {

template <class Field>
SparseLDL<Field>::SparseLDL() {
  // Avoid the potential for order-of-magnitude performance degradation from
  // slow subnormal processing.
  EnableFlushToZero();
}

template <class Field>
SparseLDLResult SparseLDL<Field>::Factor(const CoordinateMatrix<Field>& matrix,
                                         const SparseLDLControl& control) {
  scalar_factorization.reset();
  supernodal_factorization.reset();

#ifdef CATAMARI_ENABLE_TIMERS
  quotient::Timer timer;
  timer.Start();
#endif
  std::unique_ptr<quotient::QuotientGraph> quotient_graph(
      new quotient::QuotientGraph(matrix.NumRows(), matrix.Entries(),
                                  control.md_control));
  const quotient::MinimumDegreeResult analysis =
      quotient::MinimumDegree(quotient_graph.get());
#ifdef QUOTIENT_ENABLE_TIMERS
  for (const std::pair<std::string, double>& time :
       quotient_graph->ComponentSeconds()) {
    std::cout << "  " << time.first << ": " << time.second << std::endl;
  }
#endif  // ifdef QUOTIENT_TIMERS
#ifdef CATAMARI_ENABLE_TIMERS
  std::cout << "Reordering time: " << timer.Stop() << " seconds." << std::endl;
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  CATAMARI_START_TIMER(timer);
  SymmetricOrdering ordering;
  quotient_graph->ComputePostorder(&ordering.inverse_permutation);
  quotient::InvertPermutation(ordering.inverse_permutation,
                              &ordering.permutation);
#ifdef CATAMARI_ENABLE_TIMERS
  std::cout << "Postorder and invert: " << timer.Stop() << " seconds."
            << std::endl;
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  bool use_supernodal;
  if (control.supernodal_strategy == kScalarFactorization) {
    use_supernodal = false;
  } else if (control.supernodal_strategy == kSupernodalFactorization) {
    use_supernodal = true;
  } else {
    const double intensity =
        analysis.num_cholesky_flops / analysis.num_cholesky_nonzeros;

    // TODO(Jack Poulson): Make these configurable.
    const double flop_threshold = 1e5;
    const double intensity_threshold = 40;

    use_supernodal = analysis.num_cholesky_flops >= flop_threshold &&
                     intensity >= intensity_threshold;
  }

  is_supernodal = use_supernodal;
  SparseLDLResult result;
  if (use_supernodal) {
    CATAMARI_START_TIMER(timer);
    Buffer<Int> member_to_supernode;
    quotient_graph->PermutedMemberToSupernode(ordering.inverse_permutation,
                                              &member_to_supernode);
    quotient_graph->PermutedAssemblyParents(ordering.permutation,
                                            member_to_supernode,
                                            &ordering.assembly_forest.parents);

    // TODO(Jack Poulson): Only compute the children and/or roots when needed.
    ordering.assembly_forest.FillFromParents();

    quotient_graph->PermutedSupernodeSizes(ordering.inverse_permutation,
                                           &ordering.supernode_sizes);
    OffsetScan(ordering.supernode_sizes, &ordering.supernode_offsets);
#ifdef CATAMARI_ENABLE_TIMERS
    std::cout << "Ordering postprocessing: " << timer.Stop() << " seconds."
              << std::endl;
#endif  // ifdef CATAMARI_ENABLE_TIMERS

    quotient_graph.release();
    supernodal_factorization.reset(new supernodal_ldl::Factorization<Field>);
    result = supernodal_factorization->Factor(matrix, ordering,
                                              control.supernodal_control);
  } else {
    quotient_graph.release();
    scalar_factorization.reset(new scalar_ldl::Factorization<Field>);
    result =
        scalar_factorization->Factor(matrix, ordering, control.scalar_control);
  }

  return result;
}

template <class Field>
SparseLDLResult SparseLDL<Field>::Factor(const CoordinateMatrix<Field>& matrix,
                                         const SymmetricOrdering& ordering,
                                         const SparseLDLControl& control) {
  scalar_factorization.reset();
  supernodal_factorization.reset();

  bool use_supernodal;
  if (control.supernodal_strategy == kScalarFactorization) {
    use_supernodal = false;
  } else if (control.supernodal_strategy == kSupernodalFactorization) {
    use_supernodal = true;
  } else {
    // TODO(Jack Poulson): Use a more intelligent means of selecting.
    // This routine should likely take in a flop count analysis.
    use_supernodal = true;
  }

  is_supernodal = use_supernodal;
  if (use_supernodal) {
    supernodal_factorization.reset(new supernodal_ldl::Factorization<Field>);
    return supernodal_factorization->Factor(matrix, ordering,
                                            control.supernodal_control);
  } else {
    scalar_factorization.reset(new scalar_ldl::Factorization<Field>);
    return scalar_factorization->Factor(matrix, ordering,
                                        control.scalar_control);
  }
}

template <class Field>
SparseLDLResult SparseLDL<Field>::RefactorWithFixedSparsityPattern(
    const CoordinateMatrix<Field>& matrix) {
  if (is_supernodal) {
    return supernodal_factorization->RefactorWithFixedSparsityPattern(matrix);
  } else {
    return scalar_factorization->RefactorWithFixedSparsityPattern(matrix);
  }
}

template <class Field>
void SparseLDL<Field>::Solve(BlasMatrixView<Field>* right_hand_sides) const {
  if (is_supernodal) {
    supernodal_factorization->Solve(right_hand_sides);
  } else {
    scalar_factorization->Solve(right_hand_sides);
  }
}

template <class Field>
Int SparseLDL<Field>::RefinedSolve(
    const CoordinateMatrix<Field>& matrix,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides) const {
  const Int num_rows = matrix.NumRows();
  const Int num_rhs = right_hand_sides->width;
  if (control.max_iters <= 0) {
    Solve(right_hand_sides);
    return 0;
  }

  const BlasMatrix<Field> rhs_orig = *right_hand_sides;
  Buffer<Real> rhs_orig_norms(num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    const ConstBlasMatrixView<Field> column =
        rhs_orig.view.Submatrix(0, j, num_rows, 1);
    rhs_orig_norms[j] = MaxNorm(column);
  }

  // Compute the initial guesses.
  BlasMatrix<Field> solution = rhs_orig;
  Solve(&solution.view);

  // image := matrix * solution
  BlasMatrix<Field> image;
  image.Resize(num_rows, num_rhs, Field{0});
  ApplySparse(Field{1}, matrix, solution.ConstView(), Field{1}, &image.view);

  // Compute the original residuals and their max norms.
  Buffer<Real> error_norms(num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    BlasMatrixView<Field> column =
        right_hand_sides->Submatrix(0, j, num_rows, 1);

    for (Int i = 0; i < num_rows; ++i) {
      column(i, 0) -= image(i, j);
    }

    error_norms[j] = MaxNorm(column.ToConst());
    if (control.verbose) {
      std::cout << "Original relative error " << j << ": "
                << error_norms[j] / rhs_orig_norms[j] << std::endl;
    }
  }

  // We will begin with each right-hand side being 'active' and deflate out
  // each that has converged (or diverged) during iterative refinement.
  Buffer<Int> active_indices(num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    active_indices[j] = j;
  }

  Int refine_iter = 0;
  BlasMatrix<Field> update, candidate_solution;
  while (true) {
    // Deflate any converged active right-hand sides.
    {
      const Int num_active = active_indices.Size();
      Int num_remaining = 0;
      for (Int j_active = 0; j_active < num_active; ++j_active) {
        const Int j = active_indices[j_active];
        const Real error_norm = error_norms[j];
        const Real rhs_orig_norm = rhs_orig_norms[j];
        const Real relative_error = error_norm / rhs_orig_norm;
        if (relative_error <= control.relative_tol) {
          if (control.verbose) {
            std::cout << "Relative error " << j << " (" << j_active
                      << "): " << relative_error
                      << " <= " << control.relative_tol << std::endl;
          }
        } else {
          active_indices[num_remaining++] = j;
        }
      }
      active_indices.Resize(num_remaining);
    }
    const Int num_active = active_indices.Size();
    if (!num_active) {
      break;
    }

    // update := inv(matrix) * right_hand_sides.
    update.Resize(num_rows, num_active);
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      const Int j = active_indices[j_active];
      for (Int i = 0; i < num_rows; ++i) {
        update(i, j_active) = right_hand_sides->Entry(i, j);
      }
    }
    Solve(&update.view);

    // candidate_solution = solution + update
    candidate_solution.Resize(num_rows, num_active);
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      const Int j = active_indices[j_active];
      for (Int i = 0; i < num_rows; ++i) {
        candidate_solution(i, j_active) = solution(i, j) + update(i, j_active);
      }
    }

    // Compute the image of the proposed solution.
    image.Resize(num_rows, num_active);
    ApplySparse(Field{1}, matrix, candidate_solution.ConstView(), Field{0},
                &image.view);

    // Overwrite the right_hand_sides matrix with the proposed residual:
    //   right_hand_sides := rhs_orig - image
    // Also, compute the max norms of each column.
    Int num_remaining = 0;
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      const Int j = active_indices[j_active];
      BlasMatrixView<Field> column =
          right_hand_sides->Submatrix(0, j, num_rows, 1);
      for (Int i = 0; i < num_rows; ++i) {
        column(i, 0) = rhs_orig(i, j) - image(i, j_active);
      }
      const Real new_error_norm = MaxNorm(column.ToConst());
      if (control.verbose) {
        std::cout << "Refined relative error " << j << ": "
                  << new_error_norm / rhs_orig_norms[j] << std::endl;
      }
      if (new_error_norm < error_norms[j]) {
        for (Int i = 0; i < num_rows; ++i) {
          solution(i, j) = candidate_solution(i, j_active);
        }
        error_norms[j] = new_error_norm;
        active_indices[num_remaining++] = j;
      } else if (control.verbose) {
        std::cout << "Right-hand side " << j << "(" << j_active << ") diverged."
                  << std::endl;
      }
    }
    active_indices.Resize(num_remaining);

    ++refine_iter;
    if (refine_iter >= control.max_iters || active_indices.Empty()) {
      break;
    }
  }

  // *right_hand_sides := solution
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      right_hand_sides->Entry(i, j) = solution(i, j);
    }
  }

  return refine_iter;
}

template <class Field>
void SparseLDL<Field>::LowerTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  if (is_supernodal) {
    supernodal_factorization->LowerTriangularSolve(right_hand_sides);
  } else {
    scalar_factorization->LowerTriangularSolve(right_hand_sides);
  }
}

template <class Field>
void SparseLDL<Field>::DiagonalSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  if (is_supernodal) {
    supernodal_factorization->DiagonalSolve(right_hand_sides);
  } else {
    scalar_factorization->DiagonalSolve(right_hand_sides);
  }
}

template <class Field>
void SparseLDL<Field>::LowerTransposeTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  if (is_supernodal) {
    supernodal_factorization->LowerTransposeTriangularSolve(right_hand_sides);
  } else {
    scalar_factorization->LowerTransposeTriangularSolve(right_hand_sides);
  }
}

template <class Field>
void SparseLDL<Field>::PrintLowerFactor(const std::string& label,
                                        std::ostream& os) const {
  if (is_supernodal) {
    supernodal_factorization->PrintLowerFactor(label, os);
  } else {
    scalar_factorization->PrintLowerFactor(label, os);
  }
}

template <class Field>
void SparseLDL<Field>::PrintDiagonalFactor(const std::string& label,
                                           std::ostream& os) const {
  if (is_supernodal) {
    supernodal_factorization->PrintDiagonalFactor(label, os);
  } else {
    scalar_factorization->PrintDiagonalFactor(label, os);
  }
}

}  // namespace catamari

#endif  // ifndef CATAMARI_SPARSE_LDL_IMPL_H_
