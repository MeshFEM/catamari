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
#include "catamari/equilibrate_symmetric_matrix.hpp"
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
SparseLDLResult<Field> SparseLDL<Field>::Factor(
    const CoordinateMatrix<Field>& matrix,
    const SparseLDLControl<Field>& control) {
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

  if (control.supernodal_strategy == kScalarFactorization) {
    is_supernodal = false;
  } else if (control.supernodal_strategy == kSupernodalFactorization) {
    is_supernodal = true;
  } else {
    const double intensity =
        analysis.num_cholesky_flops / analysis.num_cholesky_nonzeros;
    is_supernodal =
        analysis.num_cholesky_flops >= control.supernodal_flop_threshold &&
        intensity >= control.supernodal_intensity_threshold;
  }

  // Optionally equilibrate the matrix.
  const CoordinateMatrix<Field>* matrix_to_factor;
  CoordinateMatrix<Field> equilibrated_matrix;
  if (control.equilibrate) {
    equilibrated_matrix = matrix;
    EquilibrateSymmetricMatrix(&equilibrated_matrix, &equilibration_,
                               control.verbose);
    matrix_to_factor = &equilibrated_matrix;
    have_equilibration_ = true;
  } else {
    matrix_to_factor = &matrix;
    have_equilibration_ = false;
  }

  SparseLDLResult<Field> result;
  if (is_supernodal) {
    CATAMARI_START_TIMER(timer);

    // TODO(Jack Poulson): Modify quotient to combine these into single routine.
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
    result = supernodal_factorization->Factor(*matrix_to_factor, ordering,
                                              control.supernodal_control);
  } else {
    quotient_graph.release();
    scalar_factorization.reset(new scalar_ldl::Factorization<Field>);
    result = scalar_factorization->Factor(*matrix_to_factor, ordering,
                                          control.scalar_control);
  }

  return result;
}

template <class Field>
SparseLDLResult<Field> SparseLDL<Field>::Factor(
    const CoordinateMatrix<Field>& matrix, const SymmetricOrdering& ordering,
    const SparseLDLControl<Field>& control) {
  scalar_factorization.reset();
  supernodal_factorization.reset();

  if (control.supernodal_strategy == kScalarFactorization) {
    is_supernodal = false;
  } else if (control.supernodal_strategy == kSupernodalFactorization) {
    is_supernodal = true;
  } else {
    // TODO(Jack Poulson): Use a more intelligent means of selecting.
    // This routine should likely take in a flop count analysis.
    is_supernodal = true;
  }

  // Optionally equilibrate the matrix.
  const CoordinateMatrix<Field>* matrix_to_factor;
  CoordinateMatrix<Field> equilibrated_matrix;
  if (control.equilibrate) {
    equilibrated_matrix = matrix;
    EquilibrateSymmetricMatrix(&equilibrated_matrix, &equilibration_,
                               control.verbose);
    matrix_to_factor = &equilibrated_matrix;
    have_equilibration_ = true;
  } else {
    matrix_to_factor = &matrix;
    have_equilibration_ = false;
  }

  if (is_supernodal) {
    supernodal_factorization.reset(new supernodal_ldl::Factorization<Field>);
    return supernodal_factorization->Factor(*matrix_to_factor, ordering,
                                            control.supernodal_control);
  } else {
    scalar_factorization.reset(new scalar_ldl::Factorization<Field>);
    return scalar_factorization->Factor(*matrix_to_factor, ordering,
                                        control.scalar_control);
  }
}

template <class Field>
Int SparseLDL<Field>::NumRows() const {
  if (is_supernodal) {
    return supernodal_factorization->NumRows();
  } else {
    return scalar_factorization->NumRows();
  }
}

template <class Field>
const Buffer<Int>& SparseLDL<Field>::Permutation() const {
  if (is_supernodal) {
    return supernodal_factorization->Permutation();
  } else {
    return scalar_factorization->Permutation();
  }
}

template <class Field>
const Buffer<Int>& SparseLDL<Field>::InversePermutation() const {
  if (is_supernodal) {
    return supernodal_factorization->InversePermutation();
  } else {
    return scalar_factorization->InversePermutation();
  }
}

template <class Field>
void SparseLDL<Field>::DynamicRegularizationDiagonal(
    const SparseLDLResult<Field>& result,
    BlasMatrix<ComplexBase<Field>>* diagonal) const {
  typedef ComplexBase<Field> Real;
  const Int height = NumRows();
  diagonal->Resize(height, 1, Real(0));
  for (const auto& perturbation : result.dynamic_regularization) {
    const Int index = perturbation.first;
    const Real regularization = perturbation.second;
    diagonal->Entry(index) = regularization;
  }
}

template <class Field>
SparseLDLResult<Field> SparseLDL<Field>::RefactorWithFixedSparsityPattern(
    const CoordinateMatrix<Field>& matrix) {
  typedef ComplexBase<Field> Real;

  // Optionally equilibrate the matrix.
  const CoordinateMatrix<Field>* matrix_to_factor;
  CoordinateMatrix<Field> equilibrated_matrix;
  if (have_equilibration_) {
    const bool kVerboseEquil = false;
    equilibrated_matrix = matrix;
    EquilibrateSymmetricMatrix(&equilibrated_matrix, &equilibration_,
                               kVerboseEquil);
    matrix_to_factor = &equilibrated_matrix;
  } else {
    matrix_to_factor = &matrix;
  }

  SparseLDLResult<Field> result;
  if (is_supernodal) {
    result = supernodal_factorization->RefactorWithFixedSparsityPattern(
        *matrix_to_factor);
  } else {
    result = scalar_factorization->RefactorWithFixedSparsityPattern(
        *matrix_to_factor);
  }
  if (have_equilibration_) {
    // We factored inv(D) A inv(D), so the regularization needs to be wrapped
    // with D . D.
    for (std::pair<Int, Real>& reg : result.dynamic_regularization) {
      reg.second *= equilibration_(reg.first) * equilibration_(reg.first);
    }
  }
  return result;
}

template <class Field>
SparseLDLResult<Field> SparseLDL<Field>::RefactorWithFixedSparsityPattern(
    const CoordinateMatrix<Field>& matrix,
    const SparseLDLControl<Field>& control) {
  typedef ComplexBase<Field> Real;

  // TODO(Jack Poulson): Add sanity checks here that, for example, the algorithm
  // hasn't changed.

  // Optionally equilibrate the matrix.
  const CoordinateMatrix<Field>* matrix_to_factor;
  CoordinateMatrix<Field> equilibrated_matrix;
  if (have_equilibration_) {
    const bool kVerboseEquil = false;
    equilibrated_matrix = matrix;
    EquilibrateSymmetricMatrix(&equilibrated_matrix, &equilibration_,
                               kVerboseEquil);
    matrix_to_factor = &equilibrated_matrix;
  } else {
    matrix_to_factor = &matrix;
  }

  SparseLDLResult<Field> result;
  if (is_supernodal) {
    result = supernodal_factorization->RefactorWithFixedSparsityPattern(
        *matrix_to_factor, control.supernodal_control);
  } else {
    result = scalar_factorization->RefactorWithFixedSparsityPattern(
        *matrix_to_factor, control.scalar_control);
  }
  if (have_equilibration_) {
    // We factored inv(D) A inv(D), so the regularization needs to be wrapped
    // with D . D.
    for (std::pair<Int, Real>& reg : result.dynamic_regularization) {
      reg.second *= equilibration_(reg.first) * equilibration_(reg.first);
    }
  }
  return result;
}

template <class Field>
void SparseLDL<Field>::Solve(BlasMatrixView<Field>* right_hand_sides) const {
  if (have_equilibration_) {
    // Apply the inverse of the equilibration matrix.
    for (Int j = 0; j < right_hand_sides->width; ++j) {
      for (Int i = 0; i < right_hand_sides->height; ++i) {
        right_hand_sides->Entry(i) /= equilibration_(i);
      }
    }
  }
  if (is_supernodal) {
    supernodal_factorization->Solve(right_hand_sides);
  } else {
    scalar_factorization->Solve(right_hand_sides);
  }
  if (have_equilibration_) {
    // Apply the inverse of the equilibration matrix.
    for (Int j = 0; j < right_hand_sides->width; ++j) {
      for (Int i = 0; i < right_hand_sides->height; ++i) {
        right_hand_sides->Entry(i) /= equilibration_(i);
      }
    }
  }
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>> SparseLDL<Field>::RefinedSolveHelper(
    const CoordinateMatrix<Field>& matrix,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides) const {
  const Int num_rows = matrix.NumRows();
  const Int num_rhs = right_hand_sides->width;
  RefinedSolveStatus<Real> state;

  // Compute the original maximum norms.
  const BlasMatrix<Field> rhs_orig = *right_hand_sides;
  Buffer<Real> rhs_orig_norms(num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    const ConstBlasMatrixView<Field> column =
        rhs_orig.view.Submatrix(0, j, num_rows, 1);
    rhs_orig_norms[j] = MaxNorm(column);
  }

  // Compute the initial guesses.
  // TODO(Jack Poulson): Avoid solving against all zero right-hand sides.
  BlasMatrix<Field> solution = rhs_orig;
  Solve(&solution.view);

  // image := matrix * solution
  BlasMatrix<Field> image;
  image.Resize(num_rows, num_rhs, Field{0});
  ApplySparse(Field{1}, matrix, solution.ConstView(), Field{1}, &image.view);

  // Compute the original residuals and their max norms.
  //
  // We will begin with each nonzero right-hand side being 'active' and deflate
  // out each that has converged (or diverged) during iterative refinement.
  Int num_nonzero = 0;
  Buffer<Real> error_norms(num_rhs);
  Buffer<Int> active_indices(num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    BlasMatrixView<Field> column =
        right_hand_sides->Submatrix(0, j, num_rows, 1);
    if (rhs_orig_norms[j] == Real(0)) {
      if (control.verbose) {
        std::cout << "Right-hand side " << j << " was zero." << std::endl;
      }
      continue;
    }
    active_indices[num_nonzero++] = j;

    for (Int i = 0; i < num_rows; ++i) {
      column(i) -= image(i, j);
    }

    error_norms[j] = MaxNorm(column.ToConst());
    if (control.verbose) {
      const Real relative_error = error_norms[j] / rhs_orig_norms[j];
      std::cout << "Original relative error: " << j << ": " << relative_error
                << std::endl;
    }
  }
  active_indices.Resize(num_nonzero);

  state.num_iterations = 0;
  BlasMatrix<Field> update, candidate_solution;
  while (true) {
    // Deflate any converged active right-hand sides.
    {
      const Int num_active = active_indices.Size();
      Int num_remaining = 0;
      for (Int j_active = 0; j_active < num_active; ++j_active) {
        const Int j = active_indices[j_active];
        const Real relative_error = error_norms[j] / rhs_orig_norms[j];
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
        column(i) = rhs_orig(i, j) - image(i, j_active);
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

    ++state.num_iterations;
    if (state.num_iterations >= control.max_iters || active_indices.Empty()) {
      break;
    }
  }

  // *right_hand_sides := solution
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      right_hand_sides->Entry(i, j) = solution(i, j);
    }
  }

  // Compute the final maximum scaled residual max norm.
  state.residual_relative_max_norm = 0;
  for (Int j = 0; j < num_rhs; ++j) {
    const Real relative_error = rhs_orig_norms[j] == Real(0)
                                    ? Real(0)
                                    : error_norms[j] / rhs_orig_norms[j];
    state.residual_relative_max_norm =
        std::max(state.residual_relative_max_norm, relative_error);
  }

  return state;
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>>
SparseLDL<Field>::PromotedRefinedSolveHelper(
    const CoordinateMatrix<Field>& matrix,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides_lower) const {
  const Int num_rows = matrix.NumRows();
  const Int num_rhs = right_hand_sides_lower->width;
  RefinedSolveStatus<Real> state;

  // Compute the original maximum norms.
  const BlasMatrix<Field> rhs_orig_lower = *right_hand_sides_lower;
  Buffer<Real> rhs_orig_norms(num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    const ConstBlasMatrixView<Field> column =
        rhs_orig_lower.view.Submatrix(0, j, num_rows, 1);
    rhs_orig_norms[j] = MaxNorm(column);
  }

  // Compute the initial guesses.
  // TODO(Jack Poulson): Avoid solving against all zero right-hand sides.
  BlasMatrix<Field> solution_lower = rhs_orig_lower;
  Solve(&solution_lower.view);
  BlasMatrix<Promote<Field>> solution_higher(num_rows, num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      solution_higher(i, j) = solution_lower(i, j);
    }
  }

  // image := matrix * solution
  BlasMatrix<Promote<Field>> image_higher;
  image_higher.Resize(num_rows, num_rhs, Field{0});
  ApplySparse(Promote<Field>{1}, matrix, solution_higher.ConstView(),
              Promote<Field>{1}, &image_higher.view);

  // Convert the right-hand sides into higher precision.
  BlasMatrix<Promote<Field>> right_hand_sides_higher(num_rows, num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      right_hand_sides_higher(i, j) = right_hand_sides_lower->Entry(i, j);
    }
  }

  // Compute the original residuals and their max norms.
  //
  // We will begin with each nonzero right-hand side being 'active' and deflate
  // out each that has converged (or diverged) during iterative refinement.
  Int num_nonzero = 0;
  Buffer<Real> error_norms(num_rhs);
  Buffer<Int> active_indices(num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    BlasMatrixView<Promote<Field>> column =
        right_hand_sides_higher.Submatrix(0, j, num_rows, 1);
    if (rhs_orig_norms[j] == Real(0)) {
      if (control.verbose) {
        std::cout << "Right-hand side " << j << " was zero." << std::endl;
      }
      continue;
    }
    active_indices[num_nonzero++] = j;

    for (Int i = 0; i < num_rows; ++i) {
      column(i) -= image_higher(i, j);
    }

    error_norms[j] = MaxNorm(column.ToConst());
    if (control.verbose) {
      const Real relative_error = error_norms[j] / rhs_orig_norms[j];
      std::cout << "Original relative error: " << j << ": " << relative_error
                << std::endl;
    }
  }
  active_indices.Resize(num_nonzero);

  state.num_iterations = 0;
  BlasMatrix<Field> update_lower;
  BlasMatrix<Promote<Field>> update_higher, candidate_solution_higher;
  while (true) {
    // Deflate any converged active right-hand sides.
    {
      const Int num_active = active_indices.Size();
      Int num_remaining = 0;
      for (Int j_active = 0; j_active < num_active; ++j_active) {
        const Int j = active_indices[j_active];
        const Real relative_error = error_norms[j] / rhs_orig_norms[j];
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
    update_lower.Resize(num_rows, num_active);
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      const Int j = active_indices[j_active];
      for (Int i = 0; i < num_rows; ++i) {
        update_lower(i, j_active) = right_hand_sides_higher(i, j);
      }
    }
    Solve(&update_lower.view);
    update_higher.Resize(num_rows, num_active);
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      for (Int i = 0; i < num_rows; ++i) {
        update_higher(i, j_active) = update_lower(i, j_active);
      }
    }

    // candidate_solution = solution + update
    candidate_solution_higher.Resize(num_rows, num_active);
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      const Int j = active_indices[j_active];
      for (Int i = 0; i < num_rows; ++i) {
        candidate_solution_higher(i, j_active) =
            solution_higher(i, j) + update_higher(i, j_active);
      }
    }

    // Compute the image of the proposed solution.
    image_higher.Resize(num_rows, num_active);
    ApplySparse(Promote<Field>{1}, matrix,
                candidate_solution_higher.ConstView(), Promote<Field>{0},
                &image_higher.view);

    // Overwrite the right_hand_sides matrix with the proposed residual:
    //   right_hand_sides := rhs_orig - image
    // Also, compute the max norms of each column.
    Int num_remaining = 0;
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      const Int j = active_indices[j_active];
      BlasMatrixView<Promote<Field>> column =
          right_hand_sides_higher.Submatrix(0, j, num_rows, 1);
      for (Int i = 0; i < num_rows; ++i) {
        column(i) = -image_higher(i, j_active);
        column(i) += rhs_orig_lower(i, j);
      }
      const Real new_error_norm = MaxNorm(column.ToConst());
      if (control.verbose) {
        std::cout << "Refined relative error " << j << ": "
                  << new_error_norm / rhs_orig_norms[j] << std::endl;
      }
      if (new_error_norm < error_norms[j]) {
        for (Int i = 0; i < num_rows; ++i) {
          solution_higher(i, j) = candidate_solution_higher(i, j_active);
        }
        error_norms[j] = new_error_norm;
        active_indices[num_remaining++] = j;
      } else if (control.verbose) {
        std::cout << "Right-hand side " << j << "(" << j_active << ") diverged."
                  << std::endl;
      }
    }
    active_indices.Resize(num_remaining);

    ++state.num_iterations;
    if (state.num_iterations >= control.max_iters || active_indices.Empty()) {
      break;
    }
  }

  // *right_hand_sides := solution
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      right_hand_sides_lower->Entry(i, j) = solution_higher(i, j);
    }
  }

  // Compute the final maximum scaled residual max norm.
  state.residual_relative_max_norm = 0;
  for (Int j = 0; j < num_rhs; ++j) {
    const Real relative_error = rhs_orig_norms[j] == Real(0)
                                    ? Real(0)
                                    : error_norms[j] / rhs_orig_norms[j];
    state.residual_relative_max_norm =
        std::max(state.residual_relative_max_norm, relative_error);
  }

  return state;
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>> SparseLDL<Field>::RefinedSolve(
    const CoordinateMatrix<Field>& matrix,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides) const {
  if (control.promote) {
    return PromotedRefinedSolveHelper(matrix, control, right_hand_sides);
  } else {
    return RefinedSolveHelper(matrix, control, right_hand_sides);
  }
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>>
SparseLDL<Field>::DynamicallyRegularizedRefinedSolveHelper(
    const CoordinateMatrix<Field>& matrix, const SparseLDLResult<Field>& result,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides) const {
  const Int num_rows = matrix.NumRows();
  const Int num_rhs = right_hand_sides->width;
  RefinedSolveStatus<Real> state;

  // Compute the original maximum norms.
  const BlasMatrix<Field> rhs_orig = *right_hand_sides;
  Buffer<Real> rhs_orig_norms(num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    const ConstBlasMatrixView<Field> column =
        rhs_orig.view.Submatrix(0, j, num_rows, 1);
    rhs_orig_norms[j] = MaxNorm(column);
  }

  // Compute the initial guesses.
  // TODO(Jack Poulson): Avoid solving against all zero right-hand sides.
  BlasMatrix<Field> solution = rhs_orig;
  Solve(&solution.view);

  // image := (matrix + diagonal_reg) * solution
  BlasMatrix<Field> image;
  image.Resize(num_rows, num_rhs, Field{0});
  ApplySparse(Field{1}, matrix, solution.ConstView(), Field{1}, &image.view);
  for (const std::pair<Int, Real>& perturb : result.dynamic_regularization) {
    const Int i = perturb.first;
    const Real regularization = perturb.second;
    for (Int j = 0; j < num_rhs; ++j) {
      image(i, j) += regularization * solution(i, j);
    }
  }

  // Compute the original residuals and their max norms.
  //
  // We will begin with each nonzero right-hand side being 'active' and deflate
  // out each that has converged (or diverged) during iterative refinement.
  Int num_nonzero = 0;
  Buffer<Real> error_norms(num_rhs);
  Buffer<Int> active_indices(num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    BlasMatrixView<Field> column =
        right_hand_sides->Submatrix(0, j, num_rows, 1);
    if (rhs_orig_norms[j] == Real(0)) {
      if (control.verbose) {
        std::cout << "Right-hand side " << j << " was zero." << std::endl;
      }
      continue;
    }
    active_indices[num_nonzero++] = j;

    for (Int i = 0; i < num_rows; ++i) {
      column(i) -= image(i, j);
    }

    error_norms[j] = MaxNorm(column.ToConst());
    if (control.verbose) {
      const Real relative_error = error_norms[j] / rhs_orig_norms[j];
      std::cout << "Original relative error: " << j << ": " << relative_error
                << std::endl;
    }
  }
  active_indices.Resize(num_nonzero);

  state.num_iterations = 0;
  BlasMatrix<Field> update, candidate_solution;
  while (true) {
    // Deflate any converged active right-hand sides.
    {
      const Int num_active = active_indices.Size();
      Int num_remaining = 0;
      for (Int j_active = 0; j_active < num_active; ++j_active) {
        const Int j = active_indices[j_active];
        const Real relative_error = error_norms[j] / rhs_orig_norms[j];
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
    for (const std::pair<Int, Real>& perturb : result.dynamic_regularization) {
      const Int i = perturb.first;
      const Real regularization = perturb.second;
      for (Int j_active = 0; j_active < num_active; ++j_active) {
        image(i, j_active) += regularization * candidate_solution(i, j_active);
      }
    }

    // Overwrite the right_hand_sides matrix with the proposed residual:
    //   right_hand_sides := rhs_orig - image
    // Also, compute the max norms of each column.
    Int num_remaining = 0;
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      const Int j = active_indices[j_active];
      BlasMatrixView<Field> column =
          right_hand_sides->Submatrix(0, j, num_rows, 1);
      for (Int i = 0; i < num_rows; ++i) {
        column(i) = rhs_orig(i, j) - image(i, j_active);
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

    ++state.num_iterations;
    if (state.num_iterations >= control.max_iters || active_indices.Empty()) {
      break;
    }
  }

  // *right_hand_sides := solution
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      right_hand_sides->Entry(i, j) = solution(i, j);
    }
  }

  // Compute the final maximum scaled residual max norm.
  state.residual_relative_max_norm = 0;
  for (Int j = 0; j < num_rhs; ++j) {
    const Real relative_error = rhs_orig_norms[j] == Real(0)
                                    ? Real(0)
                                    : error_norms[j] / rhs_orig_norms[j];
    state.residual_relative_max_norm =
        std::max(state.residual_relative_max_norm, relative_error);
  }

  return state;
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>>
SparseLDL<Field>::PromotedDynamicallyRegularizedRefinedSolveHelper(
    const CoordinateMatrix<Field>& matrix, const SparseLDLResult<Field>& result,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides_lower) const {
  const Int num_rows = matrix.NumRows();
  const Int num_rhs = right_hand_sides_lower->width;
  RefinedSolveStatus<Real> state;

  // Compute the original maximum norms.
  const BlasMatrix<Field> rhs_orig_lower = *right_hand_sides_lower;
  Buffer<Real> rhs_orig_norms(num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    const ConstBlasMatrixView<Field> column =
        rhs_orig_lower.view.Submatrix(0, j, num_rows, 1);
    rhs_orig_norms[j] = MaxNorm(column);
  }

  // Compute the initial guesses.
  // TODO(Jack Poulson): Avoid solving against all zero right-hand sides.
  BlasMatrix<Field> solution_lower = rhs_orig_lower;
  Solve(&solution_lower.view);
  BlasMatrix<Promote<Field>> solution_higher(num_rows, num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      solution_higher(i, j) = solution_lower(i, j);
    }
  }

  // image := (matrix + diagonal_reg) * solution
  BlasMatrix<Promote<Field>> image_higher;
  image_higher.Resize(num_rows, num_rhs, Field{0});
  ApplySparse(Promote<Field>{1}, matrix, solution_higher.ConstView(),
              Promote<Field>{1}, &image_higher.view);
  for (const std::pair<Int, Real>& perturb : result.dynamic_regularization) {
    const Int i = perturb.first;
    const Promote<Real> regularization = perturb.second;
    for (Int j = 0; j < num_rhs; ++j) {
      image_higher(i, j) += regularization * solution_higher(i, j);
    }
  }

  // Convert the right-hand sides into higher precision.
  BlasMatrix<Promote<Field>> right_hand_sides_higher(num_rows, num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      right_hand_sides_higher(i, j) = right_hand_sides_lower->Entry(i, j);
    }
  }

  // Compute the original residuals and their max norms.
  //
  // We will begin with each nonzero right-hand side being 'active' and deflate
  // out each that has converged (or diverged) during iterative refinement.
  Int num_nonzero = 0;
  Buffer<Real> error_norms(num_rhs);
  Buffer<Int> active_indices(num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    BlasMatrixView<Promote<Field>> column =
        right_hand_sides_higher->Submatrix(0, j, num_rows, 1);
    if (rhs_orig_norms[j] == Real(0)) {
      if (control.verbose) {
        std::cout << "Right-hand side " << j << " was zero." << std::endl;
      }
      continue;
    }
    active_indices[num_nonzero++] = j;

    for (Int i = 0; i < num_rows; ++i) {
      column(i) -= image_higher(i, j);
    }

    error_norms[j] = MaxNorm(column.ToConst());
    if (control.verbose) {
      const Real relative_error = error_norms[j] / rhs_orig_norms[j];
      std::cout << "Original relative error: " << j << ": " << relative_error
                << std::endl;
    }
  }
  active_indices.Resize(num_nonzero);

  state.num_iterations = 0;
  BlasMatrix<Field> update_lower;
  BlasMatrix<Promote<Field>> update_higher, candidate_solution_higher;
  while (true) {
    // Deflate any converged active right-hand sides.
    {
      const Int num_active = active_indices.Size();
      Int num_remaining = 0;
      for (Int j_active = 0; j_active < num_active; ++j_active) {
        const Int j = active_indices[j_active];
        const Real relative_error = error_norms[j] / rhs_orig_norms[j];
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
    update_lower.Resize(num_rows, num_active);
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      const Int j = active_indices[j_active];
      for (Int i = 0; i < num_rows; ++i) {
        update_lower(i, j_active) = right_hand_sides_higher(i, j);
      }
    }
    Solve(&update_lower.view);
    update_higher.Resize(num_rows, num_active);
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      for (Int i = 0; i < num_rows; ++i) {
        update_higher(i, j_active) = update_lower(i, j_active);
      }
    }

    // candidate_solution = solution + update
    candidate_solution_higher.Resize(num_rows, num_active);
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      const Int j = active_indices[j_active];
      for (Int i = 0; i < num_rows; ++i) {
        candidate_solution_higher(i, j_active) =
            solution_higher(i, j) + update_higher(i, j_active);
      }
    }

    // Compute the image of the proposed solution.
    image_higher.Resize(num_rows, num_active);
    ApplySparse(Promote<Field>{1}, matrix,
                candidate_solution_higher.ConstView(), Promote<Field>{0},
                &image_higher.view);
    for (const std::pair<Int, Real>& perturb : result.dynamic_regularization) {
      const Int i = perturb.first;
      const Promote<Real> regularization = perturb.second;
      for (Int j_active = 0; j_active < num_active; ++j_active) {
        image_higher(i, j_active) +=
            regularization * candidate_solution_higher(i, j_active);
      }
    }

    // Overwrite the right_hand_sides matrix with the proposed residual:
    //   right_hand_sides := rhs_orig - image
    // Also, compute the max norms of each column.
    Int num_remaining = 0;
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      const Int j = active_indices[j_active];
      BlasMatrixView<Promote<Field>> column =
          right_hand_sides_higher.Submatrix(0, j, num_rows, 1);
      for (Int i = 0; i < num_rows; ++i) {
        column(i) = -image_higher(i, j_active);
        column(i) += rhs_orig_lower(i, j);
      }
      const Real new_error_norm = MaxNorm(column.ToConst());
      if (control.verbose) {
        std::cout << "Refined relative error " << j << ": "
                  << new_error_norm / rhs_orig_norms[j] << std::endl;
      }
      if (new_error_norm < error_norms[j]) {
        for (Int i = 0; i < num_rows; ++i) {
          solution_higher(i, j) = candidate_solution_higher(i, j_active);
        }
        error_norms[j] = new_error_norm;
        active_indices[num_remaining++] = j;
      } else if (control.verbose) {
        std::cout << "Right-hand side " << j << "(" << j_active << ") diverged."
                  << std::endl;
      }
    }
    active_indices.Resize(num_remaining);

    ++state.num_iterations;
    if (state.num_iterations >= control.max_iters || active_indices.Empty()) {
      break;
    }
  }

  // *right_hand_sides := solution
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      right_hand_sides_lower->Entry(i, j) = solution_higher(i, j);
    }
  }

  // Compute the final maximum scaled residual max norm.
  state.residual_relative_max_norm = 0;
  for (Int j = 0; j < num_rhs; ++j) {
    const Real relative_error = rhs_orig_norms[j] == Real(0)
                                    ? Real(0)
                                    : error_norms[j] / rhs_orig_norms[j];
    state.residual_relative_max_norm =
        std::max(state.residual_relative_max_norm, relative_error);
  }

  return state;
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>>
SparseLDL<Field>::DynamicallyRegularizedRefinedSolve(
    const CoordinateMatrix<Field>& matrix, const SparseLDLResult<Field>& result,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides) const {
  if (control.promote) {
    return PromotedDynamicallyRegularizedSolveHelper(matrix, result, control,
                                                     right_hand_sides);
  } else {
    return DynamicallyRegularizedSolveHelper(matrix, result, control,
                                             right_hand_sides);
  }
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>>
SparseLDL<Field>::DiagonallyScaledRefinedSolveHelper(
    const CoordinateMatrix<Field>& matrix,
    const ConstBlasMatrixView<Real>& scaling,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides) const {
  const Int num_rows = matrix.NumRows();
  const Int num_rhs = right_hand_sides->width;
  RefinedSolveStatus<Real> state;

  // Scale the right-hand side.
  for (Int j = 0; j < num_rhs; ++j) {
    BlasMatrixView<Field> column =
        right_hand_sides->Submatrix(0, j, num_rows, 1);
    for (Int i = 0; i < num_rows; ++i) {
      column(i) *= scaling(i);
    }
  }

  // Compute the original, scaled maximum norms.
  BlasMatrix<Field> rhs_orig = *right_hand_sides;
  Buffer<Real> rhs_orig_norms(num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    const ConstBlasMatrixView<Field> column =
        rhs_orig.view.Submatrix(0, j, num_rows, 1);
    rhs_orig_norms[j] = MaxNorm(column);
  }

  // Compute the initial guesses.
  // TODO(Jack Poulson): Avoid solving against all zero right-hand sides.
  BlasMatrix<Field> solution = rhs_orig;
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      solution(i, j) /= scaling(i);
    }
  }
  Solve(&solution.view);
  BlasMatrix<Field> scaled_solution = solution;
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      solution(i, j) /= scaling(i);
    }
  }

  // image := matrix * solution
  // TODO(Jack Poulson): Avoid unnecessary extra scalings.
  BlasMatrix<Field> image;
  image.Resize(num_rows, num_rhs, Field{0});
  ApplySparse(Field{1}, matrix, scaled_solution.ConstView(), Field{1},
              &image.view);
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      image(i, j) *= scaling(i);
    }
  }

  // Compute the original residuals and their max norms.
  //
  // We will begin with each nonzero right-hand side being 'active' and deflate
  // out each that has converged (or diverged) during iterative refinement.
  Int num_nonzero = 0;
  Buffer<Real> error_norms(num_rhs);
  Buffer<Int> active_indices(num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    BlasMatrixView<Field> column =
        right_hand_sides->Submatrix(0, j, num_rows, 1);
    if (rhs_orig_norms[j] == Real(0)) {
      if (control.verbose) {
        std::cout << "Right-hand side " << j << " was zero." << std::endl;
      }
      continue;
    }
    active_indices[num_nonzero++] = j;

    for (Int i = 0; i < num_rows; ++i) {
      column(i) -= image(i, j);
    }

    error_norms[j] = MaxNorm(column.ToConst());
    if (control.verbose) {
      const Real relative_error = error_norms[j] / rhs_orig_norms[j];
      std::cout << "Original relative error: " << j << ": " << relative_error
                << std::endl;
    }
  }
  active_indices.Resize(num_nonzero);

  state.num_iterations = 0;
  BlasMatrix<Field> update, candidate_solution;
  while (true) {
    // Deflate any converged active right-hand sides.
    {
      const Int num_active = active_indices.Size();
      Int num_remaining = 0;
      for (Int j_active = 0; j_active < num_active; ++j_active) {
        const Int j = active_indices[j_active];
        const Real relative_error = error_norms[j] / rhs_orig_norms[j];
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
        update(i, j_active) = right_hand_sides->Entry(i, j) / scaling(i);
      }
    }
    Solve(&update.view);
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      for (Int i = 0; i < num_rows; ++i) {
        update(i, j_active) /= scaling(i);
      }
    }

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
    scaled_solution = candidate_solution;
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      for (Int i = 0; i < num_rows; ++i) {
        scaled_solution(i, j_active) *= scaling(i);
      }
    }
    ApplySparse(Field{1}, matrix, scaled_solution.ConstView(), Field{0},
                &image.view);
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      for (Int i = 0; i < num_rows; ++i) {
        image(i, j_active) *= scaling(i);
      }
    }

    // Overwrite the right_hand_sides matrix with the proposed residual:
    //   right_hand_sides := rhs_orig - image
    // Also, compute the max norms of each column.
    Int num_remaining = 0;
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      const Int j = active_indices[j_active];
      BlasMatrixView<Field> column =
          right_hand_sides->Submatrix(0, j, num_rows, 1);
      for (Int i = 0; i < num_rows; ++i) {
        column(i) = rhs_orig(i, j) - image(i, j_active);
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

    ++state.num_iterations;
    if (state.num_iterations >= control.max_iters || active_indices.Empty()) {
      break;
    }
  }

  // *right_hand_sides := scaling * solution
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      right_hand_sides->Entry(i, j) = scaling(i) * solution(i, j);
    }
  }

  // Compute the final maximum scaled residual max norm.
  state.residual_relative_max_norm = 0;
  for (Int j = 0; j < num_rhs; ++j) {
    const Real relative_error = rhs_orig_norms[j] == Real(0)
                                    ? Real(0)
                                    : error_norms[j] / rhs_orig_norms[j];
    state.residual_relative_max_norm =
        std::max(state.residual_relative_max_norm, relative_error);
  }

  return state;
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>>
SparseLDL<Field>::PromotedDiagonallyScaledRefinedSolveHelper(
    const CoordinateMatrix<Field>& matrix,
    const ConstBlasMatrixView<Real>& scaling,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides_lower) const {
  const Int num_rows = matrix.NumRows();
  const Int num_rhs = right_hand_sides_lower->width;
  RefinedSolveStatus<Real> state;

  // Convert the right-hand sides into higher precision and scale them.
  BlasMatrix<Promote<Field>> right_hand_sides_higher(num_rows, num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      right_hand_sides_higher(i, j) =
          right_hand_sides_lower->Entry(i, j) * Promote<Real>(scaling(i));
    }
  }

  // Compute the original, scaled maximum norms.
  BlasMatrix<Field> rhs_orig_lower = *right_hand_sides_lower;
  Buffer<Real> rhs_orig_norms(num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    const ConstBlasMatrixView<Field> column =
        rhs_orig_lower.view.Submatrix(0, j, num_rows, 1);
    rhs_orig_norms[j] = MaxNorm(column);
  }

  // Compute the initial guesses.
  // TODO(Jack Poulson): Avoid solving against all zero right-hand sides.
  BlasMatrix<Field> solution_lower;
  BlasMatrix<Promote<Field>> solution_higher;
  solution_lower.Resize(num_rows, num_rhs);
  solution_higher.Resize(num_rows, num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      solution_higher(i, j) = rhs_orig_lower(i, j) / Promote<Real>(scaling(i));
    }
  }
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      solution_lower(i, j) = solution_higher(i, j);
    }
  }
  Solve(&solution_lower.view);
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      solution_higher(i, j) = solution_lower(i, j);
    }
  }
  BlasMatrix<Promote<Field>> scaled_solution_higher = solution_higher;
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      solution_higher(i, j) /= Promote<Field>(scaling(i));
    }
  }

  // image := matrix * solution
  // TODO(Jack Poulson): Avoid unnecessary extra scalings.
  BlasMatrix<Promote<Field>> image_higher;
  image_higher.Resize(num_rows, num_rhs, Field{0});
  ApplySparse(Promote<Field>{1}, matrix, scaled_solution_higher.ConstView(),
              Promote<Field>{1}, &image_higher.view);
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      image_higher(i, j) *= Promote<Real>(scaling(i));
    }
  }

  // Compute the original residuals and their max norms.
  //
  // We will begin with each nonzero right-hand side being 'active' and deflate
  // out each that has converged (or diverged) during iterative refinement.
  Int num_nonzero = 0;
  Buffer<Real> error_norms(num_rhs);
  Buffer<Int> active_indices(num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    BlasMatrixView<Promote<Field>> column =
        right_hand_sides_higher->Submatrix(0, j, num_rows, 1);
    if (rhs_orig_norms[j] == Real(0)) {
      if (control.verbose) {
        std::cout << "Right-hand side " << j << " was zero." << std::endl;
      }
      continue;
    }
    active_indices[num_nonzero++] = j;

    for (Int i = 0; i < num_rows; ++i) {
      column(i) -= image_higher(i, j);
    }

    error_norms[j] = MaxNorm(column.ToConst());
    if (control.verbose) {
      const Real relative_error = error_norms[j] / rhs_orig_norms[j];
      std::cout << "Original relative error: " << j << ": " << relative_error
                << std::endl;
    }
  }
  active_indices.Resize(num_nonzero);

  state.num_iterations = 0;
  BlasMatrix<Field> update_lower;
  BlasMatrix<Promote<Field>> update_higher, candidate_solution_higher;
  while (true) {
    // Deflate any converged active right-hand sides.
    {
      const Int num_active = active_indices.Size();
      Int num_remaining = 0;
      for (Int j_active = 0; j_active < num_active; ++j_active) {
        const Int j = active_indices[j_active];
        const Real relative_error = error_norms[j] / rhs_orig_norms[j];
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
    update_higher.Resize(num_rows, num_active);
    update_lower.Resize(num_rows, num_active);
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      const Int j = active_indices[j_active];
      for (Int i = 0; i < num_rows; ++i) {
        update_higher(i, j_active) =
            right_hand_sides_higher(i, j) / Promote<Real>(scaling(i));
      }
    }
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      for (Int i = 0; i < num_rows; ++i) {
        update_lower(i, j_active) = update_higher(i, j_active);
      }
    }
    Solve(&update_lower.view);
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      for (Int i = 0; i < num_rows; ++i) {
        update_higher(i, j_active) = update_lower(i, j_active);
      }
    }
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      for (Int i = 0; i < num_rows; ++i) {
        update_higher(i, j_active) /= Promote<Real>(scaling(i));
      }
    }

    // candidate_solution = solution + update
    candidate_solution_higher.Resize(num_rows, num_active);
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      const Int j = active_indices[j_active];
      for (Int i = 0; i < num_rows; ++i) {
        candidate_solution_higher(i, j_active) =
            solution_higher(i, j) + update_higher(i, j_active);
      }
    }

    // Compute the image of the proposed solution.
    image_higher.Resize(num_rows, num_active);
    scaled_solution_higher = candidate_solution_higher;
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      for (Int i = 0; i < num_rows; ++i) {
        scaled_solution_higher(i, j_active) *= Promote<Real>(scaling(i));
      }
    }
    ApplySparse(Promote<Field>{1}, matrix, scaled_solution_higher.ConstView(),
                Promote<Field>{0}, &image_higher.view);
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      for (Int i = 0; i < num_rows; ++i) {
        image_higher(i, j_active) *= Promote<Real>(scaling(i));
      }
    }

    // Overwrite the right_hand_sides matrix with the proposed residual:
    //   right_hand_sides := rhs_orig - image
    // Also, compute the max norms of each column.
    Int num_remaining = 0;
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      const Int j = active_indices[j_active];
      BlasMatrixView<Promote<Field>> column =
          right_hand_sides_higher.Submatrix(0, j, num_rows, 1);
      for (Int i = 0; i < num_rows; ++i) {
        column(i) = -image_higher(i, j_active);
        column(i) += rhs_orig_lower(i, j);
      }
      const Real new_error_norm = MaxNorm(column.ToConst());
      if (control.verbose) {
        std::cout << "Refined relative error " << j << ": "
                  << new_error_norm / rhs_orig_norms[j] << std::endl;
      }
      if (new_error_norm < error_norms[j]) {
        for (Int i = 0; i < num_rows; ++i) {
          solution_higher(i, j) = candidate_solution_higher(i, j_active);
        }
        error_norms[j] = new_error_norm;
        active_indices[num_remaining++] = j;
      } else if (control.verbose) {
        std::cout << "Right-hand side " << j << "(" << j_active << ") diverged."
                  << std::endl;
      }
    }
    active_indices.Resize(num_remaining);

    ++state.num_iterations;
    if (state.num_iterations >= control.max_iters || active_indices.Empty()) {
      break;
    }
  }

  // *right_hand_sides := scaling * solution
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      right_hand_sides_lower->Entry(i, j) =
          Promote<Real>(scaling(i)) * solution_higher(i, j);
    }
  }

  // Compute the final maximum scaled residual max norm.
  state.residual_relative_max_norm = 0;
  for (Int j = 0; j < num_rhs; ++j) {
    const Real relative_error = rhs_orig_norms[j] == Real(0)
                                    ? Real(0)
                                    : error_norms[j] / rhs_orig_norms[j];
    state.residual_relative_max_norm =
        std::max(state.residual_relative_max_norm, relative_error);
  }

  return state;
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>>
SparseLDL<Field>::DiagonallyScaledRefinedSolve(
    const CoordinateMatrix<Field>& matrix,
    const ConstBlasMatrixView<Real>& scaling,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides) const {
  if (control.promote) {
    return PromotedDiagonallyScaledRefinedSolveHelper(matrix, scaling, control,
                                                      right_hand_sides);
  } else {
    return DiagonallyScaledRefinedSolveHelper(matrix, scaling, control,
                                              right_hand_sides);
  }
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>>
SparseLDL<Field>::DiagonallyScaledDynamicallyRegularizedRefinedSolveHelper(
    const CoordinateMatrix<Field>& matrix, const SparseLDLResult<Field>& result,
    const ConstBlasMatrixView<Real>& scaling,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides) const {
  const Int num_rows = matrix.NumRows();
  const Int num_rhs = right_hand_sides->width;
  RefinedSolveStatus<Real> state;

  // Scale the right-hand side.
  for (Int j = 0; j < num_rhs; ++j) {
    BlasMatrixView<Field> column =
        right_hand_sides->Submatrix(0, j, num_rows, 1);
    for (Int i = 0; i < num_rows; ++i) {
      column(i) *= scaling(i);
    }
  }

  // Compute the original, scaled maximum norms.
  BlasMatrix<Field> rhs_orig = *right_hand_sides;
  Buffer<Real> rhs_orig_norms(num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    const ConstBlasMatrixView<Field> column =
        rhs_orig.view.Submatrix(0, j, num_rows, 1);
    rhs_orig_norms[j] = MaxNorm(column);
  }

  // Compute the initial guesses.
  // TODO(Jack Poulson): Avoid solving against all zero right-hand sides.
  BlasMatrix<Field> solution = rhs_orig;
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      solution(i, j) /= scaling(i);
    }
  }
  Solve(&solution.view);
  BlasMatrix<Field> scaled_solution = solution;
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      solution(i, j) /= scaling(i);
    }
  }

  // image := (matrix + diagonal_reg) * solution
  // TODO(Jack Poulson): Avoid unnecessary extra scalings.
  BlasMatrix<Field> image;
  image.Resize(num_rows, num_rhs, Field{0});
  ApplySparse(Field{1}, matrix, scaled_solution.ConstView(), Field{1},
              &image.view);
  for (const std::pair<Int, Real>& perturb : result.dynamic_regularization) {
    const Int i = perturb.first;
    const Real regularization = perturb.second;
    for (Int j = 0; j < num_rhs; ++j) {
      image(i, j) += regularization * scaled_solution(i, j);
    }
  }
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      image(i, j) *= scaling(i);
    }
  }

  // Compute the original residuals and their max norms.
  //
  // We will begin with each nonzero right-hand side being 'active' and deflate
  // out each that has converged (or diverged) during iterative refinement.
  Int num_nonzero = 0;
  Buffer<Real> error_norms(num_rhs);
  Buffer<Int> active_indices(num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    BlasMatrixView<Field> column =
        right_hand_sides->Submatrix(0, j, num_rows, 1);
    if (rhs_orig_norms[j] == Real(0)) {
      if (control.verbose) {
        std::cout << "Right-hand side " << j << " was zero." << std::endl;
      }
      continue;
    }
    active_indices[num_nonzero++] = j;

    for (Int i = 0; i < num_rows; ++i) {
      column(i) -= image(i, j);
    }

    error_norms[j] = MaxNorm(column.ToConst());
    if (control.verbose) {
      const Real relative_error = error_norms[j] / rhs_orig_norms[j];
      std::cout << "Original relative error: " << j << ": " << relative_error
                << std::endl;
    }
  }
  active_indices.Resize(num_nonzero);

  state.num_iterations = 0;
  BlasMatrix<Field> update, candidate_solution;
  while (true) {
    // Deflate any converged active right-hand sides.
    {
      const Int num_active = active_indices.Size();
      Int num_remaining = 0;
      for (Int j_active = 0; j_active < num_active; ++j_active) {
        const Int j = active_indices[j_active];
        const Real relative_error = error_norms[j] / rhs_orig_norms[j];
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
        update(i, j_active) = right_hand_sides->Entry(i, j) / scaling(i);
      }
    }
    Solve(&update.view);
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      for (Int i = 0; i < num_rows; ++i) {
        update(i, j_active) /= scaling(i);
      }
    }

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
    scaled_solution = candidate_solution;
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      for (Int i = 0; i < num_rows; ++i) {
        scaled_solution(i, j_active) *= scaling(i);
      }
    }
    ApplySparse(Field{1}, matrix, scaled_solution.ConstView(), Field{0},
                &image.view);
    for (const std::pair<Int, Real>& perturb : result.dynamic_regularization) {
      const Int i = perturb.first;
      const Real regularization = perturb.second;
      for (Int j_active = 0; j_active < num_active; ++j_active) {
        image(i, j_active) += regularization * scaled_solution(i, j_active);
      }
    }
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      for (Int i = 0; i < num_rows; ++i) {
        image(i, j_active) *= scaling(i);
      }
    }

    // Overwrite the right_hand_sides matrix with the proposed residual:
    //   right_hand_sides := rhs_orig - image
    // Also, compute the max norms of each column.
    Int num_remaining = 0;
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      const Int j = active_indices[j_active];
      BlasMatrixView<Field> column =
          right_hand_sides->Submatrix(0, j, num_rows, 1);
      for (Int i = 0; i < num_rows; ++i) {
        column(i) = rhs_orig(i, j) - image(i, j_active);
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

    ++state.num_iterations;
    if (state.num_iterations >= control.max_iters || active_indices.Empty()) {
      break;
    }
  }

  // *right_hand_sides := scaling * solution
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      right_hand_sides->Entry(i, j) = scaling(i) * solution(i, j);
    }
  }

  // Compute the final maximum scaled residual max norm.
  state.residual_relative_max_norm = 0;
  for (Int j = 0; j < num_rhs; ++j) {
    const Real relative_error = rhs_orig_norms[j] == Real(0)
                                    ? Real(0)
                                    : error_norms[j] / rhs_orig_norms[j];
    state.residual_relative_max_norm =
        std::max(state.residual_relative_max_norm, relative_error);
  }

  return state;
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>> SparseLDL<Field>::
    PromotedDiagonallyScaledDynamicallyRegularizedRefinedSolveHelper(
        const CoordinateMatrix<Field>& matrix,
        const SparseLDLResult<Field>& result,
        const ConstBlasMatrixView<Real>& scaling,
        const RefinedSolveControl<Real>& control,
        BlasMatrixView<Field>* right_hand_sides_lower) const {
  const Int num_rows = matrix.NumRows();
  const Int num_rhs = right_hand_sides_lower->width;
  RefinedSolveStatus<Real> state;

  // Convert the right-hand sides to higher precision and scale them.
  BlasMatrix<Promote<Field>> right_hand_sides_higher(num_rows, num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      right_hand_sides_higher(i, j) =
          right_hand_sides_lower->Entry(i, j) * Promote<Field>(scaling(i));
    }
  }

  // Compute the original, scaled maximum norms.
  BlasMatrix<Field> rhs_orig_lower = *right_hand_sides_lower;
  Buffer<Real> rhs_orig_norms(num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    const ConstBlasMatrixView<Field> column =
        rhs_orig_lower.view.Submatrix(0, j, num_rows, 1);
    rhs_orig_norms[j] = MaxNorm(column);
  }

  // Compute the initial guesses.
  // TODO(Jack Poulson): Avoid solving against all zero right-hand sides.
  BlasMatrix<Field> solution_lower;
  BlasMatrix<Promote<Field>> solution_higher;
  solution_lower.Resize(num_rows, num_rhs);
  solution_higher.Resize(num_rows, num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      solution_higher(i, j) = rhs_orig_lower(i, j) / Promote<Field>(scaling(i));
    }
  }
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      solution_lower(i, j) = solution_higher(i, j);
    }
  }
  Solve(&solution_lower.view);
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      solution_higher(i, j) = solution_lower(i, j);
    }
  }
  BlasMatrix<Promote<Field>> scaled_solution_higher = solution_higher;
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      solution_higher(i, j) /= Promote<Field>(scaling(i));
    }
  }

  // image := (matrix + diagonal_reg) * solution
  // TODO(Jack Poulson): Avoid unnecessary extra scalings.
  BlasMatrix<Promote<Field>> image_higher;
  image_higher.Resize(num_rows, num_rhs, Field{0});
  ApplySparse(Promote<Field>{1}, matrix, scaled_solution_higher.ConstView(),
              Promote<Field>{1}, &image_higher.view);
  for (const std::pair<Int, Real>& perturb : result.dynamic_regularization) {
    const Int i = perturb.first;
    const Promote<Real> regularization = perturb.second;
    for (Int j = 0; j < num_rhs; ++j) {
      image_higher(i, j) += regularization * scaled_solution_higher(i, j);
    }
  }
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      image_higher(i, j) *= Promote<Field>(scaling(i));
    }
  }

  // Compute the original residuals and their max norms.
  //
  // We will begin with each nonzero right-hand side being 'active' and deflate
  // out each that has converged (or diverged) during iterative refinement.
  Int num_nonzero = 0;
  Buffer<Real> error_norms(num_rhs);
  Buffer<Int> active_indices(num_rhs);
  for (Int j = 0; j < num_rhs; ++j) {
    BlasMatrixView<Promote<Field>> column =
        right_hand_sides_higher.Submatrix(0, j, num_rows, 1);
    if (rhs_orig_norms[j] == Real(0)) {
      if (control.verbose) {
        std::cout << "Right-hand side " << j << " was zero." << std::endl;
      }
      continue;
    }
    active_indices[num_nonzero++] = j;

    for (Int i = 0; i < num_rows; ++i) {
      column(i) -= image_higher(i, j);
    }

    error_norms[j] = MaxNorm(column.ToConst());
    if (control.verbose) {
      const Real relative_error = error_norms[j] / rhs_orig_norms[j];
      std::cout << "Original relative error: " << j << ": " << relative_error
                << std::endl;
    }
  }
  active_indices.Resize(num_nonzero);

  state.num_iterations = 0;
  BlasMatrix<Field> update_lower;
  BlasMatrix<Promote<Field>> update_higher, candidate_solution_higher;
  while (true) {
    // Deflate any converged active right-hand sides.
    {
      const Int num_active = active_indices.Size();
      Int num_remaining = 0;
      for (Int j_active = 0; j_active < num_active; ++j_active) {
        const Int j = active_indices[j_active];
        const Real relative_error = error_norms[j] / rhs_orig_norms[j];
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
    update_higher.Resize(num_rows, num_active);
    update_lower.Resize(num_rows, num_active);
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      const Int j = active_indices[j_active];
      for (Int i = 0; i < num_rows; ++i) {
        update_higher(i, j_active) =
            right_hand_sides_higher(i, j) / Promote<Field>(scaling(i));
      }
    }
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      for (Int i = 0; i < num_rows; ++i) {
        update_lower(i, j_active) = update_higher(i, j_active);
      }
    }
    Solve(&update_lower.view);
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      for (Int i = 0; i < num_rows; ++i) {
        update_higher(i, j_active) = update_lower(i, j_active);
      }
    }
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      for (Int i = 0; i < num_rows; ++i) {
        update_higher(i, j_active) /= Promote<Field>(scaling(i));
      }
    }

    // candidate_solution = solution + update
    candidate_solution_higher.Resize(num_rows, num_active);
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      const Int j = active_indices[j_active];
      for (Int i = 0; i < num_rows; ++i) {
        candidate_solution_higher(i, j_active) =
            solution_higher(i, j) + update_higher(i, j_active);
      }
    }

    // Compute the image of the proposed solution.
    image_higher.Resize(num_rows, num_active);
    scaled_solution_higher = candidate_solution_higher;
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      for (Int i = 0; i < num_rows; ++i) {
        scaled_solution_higher(i, j_active) *= Promote<Field>(scaling(i));
      }
    }
    ApplySparse(Promote<Field>{1}, matrix, scaled_solution_higher.ConstView(),
                Promote<Field>{0}, &image_higher.view);
    for (const std::pair<Int, Real>& perturb : result.dynamic_regularization) {
      const Int i = perturb.first;
      const Promote<Real> regularization = perturb.second;
      for (Int j_active = 0; j_active < num_active; ++j_active) {
        image_higher(i, j_active) +=
            regularization * scaled_solution_higher(i, j_active);
      }
    }
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      for (Int i = 0; i < num_rows; ++i) {
        image_higher(i, j_active) *= Promote<Real>(scaling(i));
      }
    }

    // Overwrite the right_hand_sides matrix with the proposed residual:
    //   right_hand_sides := rhs_orig - image
    // Also, compute the max norms of each column.
    Int num_remaining = 0;
    for (Int j_active = 0; j_active < num_active; ++j_active) {
      const Int j = active_indices[j_active];
      BlasMatrixView<Promote<Field>> column =
          right_hand_sides_higher.Submatrix(0, j, num_rows, 1);
      for (Int i = 0; i < num_rows; ++i) {
        column(i) = -image_higher(i, j_active);
        column(i) += rhs_orig_lower(i, j);
      }
      const Real new_error_norm = MaxNorm(column.ToConst());
      if (control.verbose) {
        std::cout << "Refined relative error " << j << ": "
                  << new_error_norm / rhs_orig_norms[j] << std::endl;
      }
      if (new_error_norm < error_norms[j]) {
        for (Int i = 0; i < num_rows; ++i) {
          solution_higher(i, j) = candidate_solution_higher(i, j_active);
        }
        error_norms[j] = new_error_norm;
        active_indices[num_remaining++] = j;
      } else if (control.verbose) {
        std::cout << "Right-hand side " << j << "(" << j_active << ") diverged."
                  << std::endl;
      }
    }
    active_indices.Resize(num_remaining);

    ++state.num_iterations;
    if (state.num_iterations >= control.max_iters || active_indices.Empty()) {
      break;
    }
  }

  // *right_hand_sides := scaling * solution
  for (Int j = 0; j < num_rhs; ++j) {
    for (Int i = 0; i < num_rows; ++i) {
      right_hand_sides_lower->Entry(i, j) =
          Promote<Real>(scaling(i)) * solution_higher(i, j);
    }
  }

  // Compute the final maximum scaled residual max norm.
  state.residual_relative_max_norm = 0;
  for (Int j = 0; j < num_rhs; ++j) {
    const Real relative_error = rhs_orig_norms[j] == Real(0)
                                    ? Real(0)
                                    : error_norms[j] / rhs_orig_norms[j];
    state.residual_relative_max_norm =
        std::max(state.residual_relative_max_norm, relative_error);
  }

  return state;
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>>
SparseLDL<Field>::DiagonallyScaledDynamicallyRegularizedRefinedSolve(
    const CoordinateMatrix<Field>& matrix, const SparseLDLResult<Field>& result,
    const ConstBlasMatrixView<Real>& scaling,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides) const {
  if (control.promote) {
    return PromotedDiagonallyScaledDynamicallyRegularizedRefinedSolveHelper(
        matrix, result, scaling, control, right_hand_sides);
  } else {
    return DiagonallyScaledDynamicallyRegularizedRefinedSolveHelper(
        matrix, result, scaling, control, right_hand_sides);
  }
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

template <class Field>
void MergeDynamicRegularizations(
    const Buffer<SparseLDLResult<Field>>& children_results,
    SparseLDLResult<Field>* result) {
  Int reg_offset = result->dynamic_regularization.size();
  Int num_regularizations = reg_offset;
  for (const SparseLDLResult<Field>& child_result : children_results) {
    num_regularizations += child_result.dynamic_regularization.size();
  }
  result->dynamic_regularization.resize(num_regularizations);
  for (const SparseLDLResult<Field>& child_result : children_results) {
    std::copy(child_result.dynamic_regularization.begin(),
              child_result.dynamic_regularization.end(),
              result->dynamic_regularization.begin() + reg_offset);
    reg_offset += child_result.dynamic_regularization.size();
  }
}

}  // namespace catamari

#endif  // ifndef CATAMARI_SPARSE_LDL_IMPL_H_
