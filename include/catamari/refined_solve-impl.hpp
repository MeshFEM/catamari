/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.  */
#ifndef CATAMARI_REFINED_SOLVE_IMPL_H_
#define CATAMARI_REFINED_SOLVE_IMPL_H_

#include "catamari/refined_solve.hpp"

namespace catamari {

namespace promote {

template <typename Field>
void HigherToLower(const ConstBlasMatrixView<Promote<Field>>& higher,
                   BlasMatrixView<Field>* lower) {
  for (Int j = 0; j < higher.width; ++j) {
    for (Int i = 0; i < higher.height; ++i) {
      lower->Entry(i, j) = higher(i, j);
    }
  }
}

template <typename Field>
void HigherToLower(const BlasMatrix<Promote<Field>>& higher,
                   BlasMatrix<Field>* lower) {
  lower->Resize(higher.Height(), higher.Width());
  for (Int j = 0; j < higher.Width(); ++j) {
    for (Int i = 0; i < higher.Height(); ++i) {
      lower->Entry(i, j) = higher(i, j);
    }
  }
}

template <typename Field>
void LowerToHigher(const ConstBlasMatrixView<Field>& lower,
                   BlasMatrixView<Promote<Field>>* higher) {
  for (Int j = 0; j < lower.width; ++j) {
    for (Int i = 0; i < lower.height; ++i) {
      higher->Entry(i, j) = lower(i, j);
    }
  }
}

template <typename Field>
void LowerToHigher(const ConstBlasMatrixView<Field>& lower,
                   BlasMatrix<Promote<Field>>* higher) {
  higher->Resize(lower.height, lower.width);
  for (Int j = 0; j < lower.width; ++j) {
    for (Int i = 0; i < lower.height; ++i) {
      higher->Entry(i, j) = lower(i, j);
    }
  }
}

template <typename Field>
void LowerToHigher(const BlasMatrix<Field>& lower,
                   BlasMatrix<Promote<Field>>* higher) {
  higher->Resize(lower.Height(), lower.Width());
  for (Int j = 0; j < lower.Width(); ++j) {
    for (Int i = 0; i < lower.Height(); ++i) {
      higher->Entry(i, j) = lower(i, j);
    }
  }
}

}  // namespace promote

template <class Field, class ApplyMatrix, class ApplyInverse>
RefinedSolveStatus<ComplexBase<Field>> RefinedSolve(
    const ApplyMatrix apply_matrix, const ApplyInverse apply_inverse,
    const RefinedSolveControl<ComplexBase<Field>>& control,
    BlasMatrixView<Field>* right_hand_sides) {
  typedef ComplexBase<Field> Real;
  const Int num_rows = right_hand_sides->height;
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
  apply_inverse(&solution.view);

  // image := matrix * solution
  BlasMatrix<Field> image;
  image.Resize(num_rows, num_rhs, Field{0});
  apply_matrix(Field{1}, solution.ConstView(), Field{1}, &image.view);

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
    apply_inverse(&update.view);

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
    apply_matrix(Field{1}, candidate_solution.ConstView(), Field{0},
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

template <class Field, class ApplyMatrix, class ApplyInverse>
RefinedSolveStatus<ComplexBase<Field>> PromotedRefinedSolve(
    const ApplyMatrix apply_matrix, const ApplyInverse apply_inverse,
    const RefinedSolveControl<ComplexBase<Field>>& control,
    BlasMatrixView<Field>* right_hand_sides_lower) {
  typedef ComplexBase<Field> Real;
  const Int num_rows = right_hand_sides_lower->height;
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
  apply_inverse(&solution_lower.view);
  BlasMatrix<Promote<Field>> solution_higher;
  promote::LowerToHigher(solution_lower, &solution_higher);

  // image := matrix * solution
  BlasMatrix<Promote<Field>> image_higher;
  image_higher.Resize(num_rows, num_rhs, Field{0});
  apply_matrix(Promote<Field>{1}, solution_higher.ConstView(),
               Promote<Field>{1}, &image_higher.view);

  // Convert the right-hand sides into higher precision.
  BlasMatrix<Promote<Field>> right_hand_sides_higher;
  promote::LowerToHigher(right_hand_sides_lower->ToConst(),
                         &right_hand_sides_higher);

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
    apply_inverse(&update_lower.view);
    promote::LowerToHigher(update_lower, &update_higher);

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
    apply_matrix(Promote<Field>{1}, candidate_solution_higher.ConstView(),
                 Promote<Field>{0}, &image_higher.view);

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
  promote::HigherToLower(solution_higher.ConstView(), right_hand_sides_lower);

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

}  // namespace catamari

#endif  // ifndef CATAMARI_REFINED_SOLVE_IMPL_H_
