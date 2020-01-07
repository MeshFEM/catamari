/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.  */
#ifndef CATAMARI_FGMRES_IMPL_H_
#define CATAMARI_FGMRES_IMPL_H_

#include <limits>

#include "catamari/givens_rotation.hpp"

#include "catamari/fgmres.hpp"

namespace catamari {

namespace fgmres {

template <class Field, class ApplyMatrix, class ApplyPreconditioner>
FGMRESStatus<ComplexBase<Field>> SingleSolve(
    const ApplyMatrix apply_matrix,
    const ApplyPreconditioner apply_preconditioner,
    const FGMRESControl<ComplexBase<Field>>& control,
    const ConstBlasMatrixView<Field>& right_hand_side,
    BlasMatrixView<Field>* solution) {
  typedef ComplexBase<Field> Real;
  CATAMARI_ASSERT(right_hand_side.width == 1, "Assumed a column vector.");
  const Int height = right_hand_side.height;
  const Int max_inner_iterations = control.max_inner_iterations;

  static const Real epsilon = std::numeric_limits<Real>::epsilon();
  const Real relative_tolerance =
      control.relative_tolerance_coefficient *
      std::pow(epsilon, control.relative_tolerance_exponent);

  // Begin with the initial guess of zero.
  for (Int i = 0; i < height; ++i) {
    solution->Entry(i) = Field(0);
  }

  // Form the image of the zero vector.
  BlasMatrix<Field> image(height, 1, Field(0));

  // Form the residual of the zero vector.
  BlasMatrix<Field> residual = right_hand_side;
  const Real original_residual_norm = EuclideanNorm(residual.ConstView());
  if (control.verbose) {
    std::cout << "|| b ||_2 = " << original_residual_norm << std::endl;
  }

  FGMRESStatus<Real> status;
  status.num_iterations = 0;
  status.num_outer_iterations = 0;

  // Early exit if the right-hand side is identically zero.
  if (original_residual_norm == Real(0)) {
    if (control.verbose) {
      std::cout << "Early-exiting FGMRES since || b ||_2 = 0." << std::endl;
    }
    status.relative_error = 0;
    return status;
  }

  bool converged = false;
  std::vector<GivensRotation<Field>> givens_rotations(max_inner_iterations);
  BlasMatrix<Field> arnoldi_matrix;
  BlasMatrix<Field> arnoldi_vectors;
  BlasMatrix<Field> preconditioned_vectors;
  BlasMatrix<Field> preconditioned_images;
  BlasMatrix<Field> projected_image;
  BlasMatrix<Field> projected_residual;
  BlasMatrix<Field> projected_solution;
  BlasMatrix<Field> initial_solution;
  BlasMatrix<Field> solution_image;
  BlasMatrix<Field> initial_solution_image;
  while (!converged) {
    if (control.verbose) {
      std::cout << "Starting outer FGMRES iteration "
                << status.num_outer_iterations << std::endl;
    }

    arnoldi_matrix.Resize(max_inner_iterations, max_inner_iterations, Field(0));
    arnoldi_vectors.Resize(height, max_inner_iterations, Field(0));
    preconditioned_vectors.Resize(height, max_inner_iterations, Field(0));
    preconditioned_images.Resize(height, max_inner_iterations, Field(0));

    // Set the initial guess of the outer iteration to the current estimate.
    initial_solution = *solution;
    if (status.num_outer_iterations > 0) {
      initial_solution_image = solution_image;
    } else {
      initial_solution_image = initial_solution;
    }

    // Compute the two-norm of the current residual vector.
    const Real outer_residual_norm = EuclideanNorm(residual.ConstView());

    // Store the normalization of the current residual as the first column
    // of the Arnoldi basis for the inner FGMRES iteration.
    BlasMatrixView<Field> v0 = arnoldi_vectors.Submatrix(0, 0, height, 1);
    for (Int i = 0; i < height; ++i) {
      v0(i) = residual(i) / outer_residual_norm;
    }

    // Initialize the projected residual as beta e_0.
    projected_residual.Resize(max_inner_iterations + 1, 1, Field(0));
    projected_residual(0) = outer_residual_norm;

    // Run the inner FGMRES sequence.
    Int inner_iteration_cap = max_inner_iterations;
    for (Int inner_iter = 0; inner_iter < inner_iteration_cap; ++inner_iter) {
      if (control.verbose) {
        std::cout << "  Starting inner FGMRES iteration " << inner_iter
                  << std::endl;
      }

      // Form the preconditioning of the most recent Arnoldi basis vector.
      const ConstBlasMatrixView<Field> arnoldi_vector =
          arnoldi_vectors.Submatrix(0, inner_iter, height, 1);
      BlasMatrixView<Field> preconditioned_vector =
          preconditioned_vectors.Submatrix(0, inner_iter, height, 1);
      for (Int i = 0; i < height; ++i) {
        preconditioned_vector(i) = arnoldi_vector(i);
      }
      apply_preconditioner(&preconditioned_vector);

      // Form the image of the preconditioning of the Arnoldi basis vector.
      BlasMatrixView<Field> preconditioned_image =
          preconditioned_images.Submatrix(0, inner_iter, height, 1);
      apply_matrix(Field(1), preconditioned_vector.ToConst(), Field(0),
                   &preconditioned_image);
      projected_image = preconditioned_image;

      // Run an Arnoldi step.
      for (Int i = 0; i <= inner_iter; ++i) {
        // H(i, inner_iter) = v_i' w, where 'w' is the projected image.
        const ConstBlasMatrixView<Field> vi =
            arnoldi_vectors.Submatrix(0, i, height, 1);
        Real arnoldi_component = Field(0);
        for (Int k = 0; k < height; ++k) {
          arnoldi_component += vi(k) * projected_image(k);
        }
        arnoldi_matrix(i, inner_iter) = arnoldi_component;

        // w := w - v_i H(i, inner_iter).
        for (Int k = 0; k < height; ++k) {
          projected_image(k) -= arnoldi_component * vi(k);
        }
      }
      const Real projected_image_norm =
          EuclideanNorm(projected_image.ConstView());
      if (!std::isfinite(projected_image_norm)) {
        if (control.verbose) {
          std::cout << "Non-finite projected image norm in FGMRES."
                    << std::endl;
        }
        // TODO(Jack Poulson): Handle this failure mode by returning the
        // current solution.
        break;
      }
      if (projected_image_norm == Real(0)) {
        inner_iteration_cap = inner_iter + 1;
      }
      if (inner_iter + 1 < inner_iteration_cap) {
        // v_{j + 1} := w / || w ||_2.
        BlasMatrixView<Field> new_arnoldi_vector =
            arnoldi_vectors.Submatrix(0, inner_iter + 1, height, 1);
        for (Int k = 0; k < height; ++k) {
          new_arnoldi_vector(k) = projected_image(k) / projected_image_norm;
        }
      }

      // Apply the existing Givens rotations to the new column of H.
      for (Int i = 0; i < inner_iter; ++i) {
        givens_rotations[i].Apply(&arnoldi_matrix(i, inner_iter),
                                  &arnoldi_matrix(i + 1, inner_iter));
      }

      // Generate a new Givens rotation.
      const Field eta_j_j = arnoldi_matrix(inner_iter, inner_iter);
      const Real eta_jp1_j = projected_image_norm;
      if (!std::isfinite(std::real(eta_j_j)) ||
          !std::isfinite(std::imag(eta_j_j))) {
        std::cout << "H(" << inner_iter << ", " << inner_iter
                  << ") was non-finite." << std::endl;
        // TODO(Jack Poulson): Handle this edge case.
        break;
      }
      if (!std::isfinite(eta_jp1_j)) {
        std::cout << "H(" << inner_iter + 1 << ", " << inner_iter << ") was "
                  << "non-finite." << std::endl;
        // TODO(Jack Poulson): Handle this edge case.
        break;
      }
      GivensRotation<Field>& new_rotation = givens_rotations[inner_iter];
      const Field combined_entry = new_rotation.Generate(eta_j_j, eta_jp1_j);
      if (!std::isfinite(std::real(combined_entry)) ||
          !std::isfinite(std::imag(combined_entry))) {
        std::cout << "Givens rotation generation produced non-finite combined "
                  << "entry." << std::endl;
        // TODO(Jack Poulson): Handle this edge case.
        break;
      }
      arnoldi_matrix(inner_iter, inner_iter) = combined_entry;

      // Apply the new Givens rotation to the projected residual.
      // HERE
      new_rotation.Apply(&projected_residual(inner_iter),
                         &projected_residual(inner_iter + 1));

      // Minimize the residual via a triangular solve to produce a vector, y.
      const ConstBlasMatrixView<Field> arnoldi_matrix_active =
          arnoldi_matrix.Submatrix(0, 0, inner_iter + 1, inner_iter + 1);
      projected_solution = projected_residual;
      TriangularSolveLeftUpper(arnoldi_matrix_active,
                               projected_solution.Data());

      // Set the new approximate solution via:
      //
      //   x := x_0 + Z_j y.
      //
      const ConstBlasMatrixView<Field> active_preconditioned_vectors =
          preconditioned_vectors.Submatrix(0, 0, height, inner_iter + 1);
      for (Int i = 0; i < height; ++i) {
        solution->Entry(i) = initial_solution(i);
      }
      MatrixVectorProduct(Field(1), active_preconditioned_vectors,
                          projected_solution.Data(), solution->Data());

      // Form the image of the solution without applying A, via:
      //
      //   A x = A x_0 + (A Z_j) y.
      //
      const ConstBlasMatrixView<Field> active_preconditioned_images =
          preconditioned_images.Submatrix(0, 0, height, inner_iter + 1);
      solution_image = initial_solution_image;
      MatrixVectorProduct(Field(1), active_preconditioned_images,
                          projected_solution.Data(), solution_image.Data());

      // Form the residual,
      //
      //   w := b - A x,
      //
      // and compute its two-norm.
      //
      for (Int i = 0; i < height; ++i) {
        residual(i) = right_hand_side(i) - solution_image(i);
      }
      const Real residual_norm = EuclideanNorm(residual.ConstView());
      const Real relative_residual_norm =
          residual_norm / original_residual_norm;
      status.relative_error = relative_residual_norm;
      ++status.num_iterations;

      // Check for convergence.
      if (status.relative_error <= relative_tolerance) {
        if (control.verbose) {
          std::cout << "FGMRES converged with relative_error = "
                    << status.relative_error << " < " << relative_tolerance
                    << std::endl;
        }
        converged = true;
        break;
      } else if (control.verbose) {
        std::cout << "  FGMRES inner iter finished with relative_error = "
                  << status.relative_error << std::endl;
      }

      if (status.num_iterations == control.max_iterations) {
        if (control.verbose) {
          std::cout << "FGMRES did not converge." << std::endl;
        }
        break;
      }
    }
    ++status.num_outer_iterations;
  }

  return status;
}

}  // namespace fgmres

template <class Field, class ApplyMatrix, class ApplyPreconditioner>
FGMRESStatus<ComplexBase<Field>> FGMRES(
    const ApplyMatrix apply_matrix,
    const ApplyPreconditioner apply_preconditioner,
    const FGMRESControl<ComplexBase<Field>>& control,
    const ConstBlasMatrixView<Field>& right_hand_sides,
    BlasMatrix<Field>* solutions) {
  typedef ComplexBase<Field> Real;
  const Int height = right_hand_sides.height;
  const Int num_rhs = right_hand_sides.width;
  solutions->Resize(height, num_rhs);

  FGMRESStatus<Real> status;
  status.relative_error = 0;
  status.num_iterations = 0;
  status.num_outer_iterations = 0;

  for (Int j = 0; j < num_rhs; ++j) {
    const ConstBlasMatrixView<Field> rhs =
        right_hand_sides.Submatrix(0, j, height, 1);
    BlasMatrixView<Field> solution = solutions->Submatrix(0, j, height, 1);
    auto single_status = fgmres::SingleSolve(apply_matrix, apply_preconditioner,
                                             control, rhs, &solution);

    status.relative_error =
        std::max(status.relative_error, single_status.relative_error);
    status.num_iterations =
        std::max(status.num_iterations, single_status.num_iterations);
    status.num_outer_iterations = std::max(status.num_outer_iterations,
                                           single_status.num_outer_iterations);
  }

  return status;
}

}  // namespace catamari

#endif  // ifndef CATAMARI_FGMRES_IMPL_H_
