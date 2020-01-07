/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#define CATCH_CONFIG_MAIN
#include <vector>
#include "catamari.hpp"
#include "catch2/catch.hpp"

TEST_CASE("FGMRES double", "[FGMRES double]") {
  const catamari::Int height = 5;

  // Build the sparse matrix
  //
  //
  //     |  8  -1  -3        13 |
  //     | -1   7  -2   -4      |
  // A = | -3  -2  -9           |,
  //     |     -4      -10      |
  //     | 13               -11 |
  //
  // which has a condition number of 2.2098. Also, construct a lambda function
  // for its application.
  catamari::CoordinateMatrix<double> matrix;
  matrix.Resize(height, height);
  matrix.AddEntries(std::vector<catamari::MatrixEntry<double>>{
      {0, 0, 8.},
      {0, 1, -1.},
      {0, 2, -3.},
      {0, 4, 13.},
      {1, 0, -1.},
      {1, 1, 7.},
      {1, 2, -2.},
      {1, 3, -4.},
      {2, 0, -3.},
      {2, 1, -2.},
      {2, 2, -9.},
      {3, 1, -4.},
      {3, 3, -10.},
      {4, 0, 13.},
      {4, 4, -11.},
  });
  auto apply_matrix =
      [&](double alpha, const catamari::ConstBlasMatrixView<double>& input,
          double beta, catamari::BlasMatrixView<double>* output) {
        catamari::ApplySparse(alpha, matrix, input, beta, output);
      };

  // Build a perturbed matrix, A + I, which has condition number 2.0552.
  catamari::CoordinateMatrix<double> perturbed_matrix = matrix;
  for (catamari::Int i = 0; i < height; ++i) {
    perturbed_matrix.AddEntry(i, i, 1.);
  }

  // Build a right-hand side vector, [1; 2; 3; 4; 5]. The corresponding solution
  // should be roughly:
  //
  //  |  0.242256 |
  //  | -0.020556 |
  //  | -0.409517 |.
  //  | -0.391778 |
  //  | -0.168243 |
  //
  catamari::BlasMatrix<double> right_hand_side(height, 1);
  for (catamari::Int i = 0; i < height; ++i) {
    right_hand_side(i) = i + 1.;
  }
  const double right_hand_side_norm =
      catamari::EuclideanNorm(right_hand_side.ConstView());

  // Build a preconditioner using the LDL' factorizatization of the perturbed
  // matrix.
  catamari::SparseLDLControl<double> ldl_control;
  ldl_control.SetFactorizationType(catamari::kLDLAdjointFactorization);
  catamari::SparseLDL<double> ldl;
  ldl.Factor(perturbed_matrix, ldl_control);
  auto apply_preconditioner =
      [&](catamari::BlasMatrixView<double>* right_hand_sides) {
        ldl.Solve(right_hand_sides);
      };

  // Approximately solve the linear system using FGMRES.
  catamari::FGMRESControl<double> fgmres_control;
  fgmres_control.max_inner_iterations = 2;
  fgmres_control.relative_tolerance_coefficient = 1e-12;
  fgmres_control.relative_tolerance_exponent = 0;
  catamari::BlasMatrix<double> fgmres_solution;
  const catamari::FGMRESStatus<double> fgmres_status =
      catamari::FGMRES(apply_matrix, apply_preconditioner, fgmres_control,
                       right_hand_side.ConstView(), &fgmres_solution);

  // Compute the residual from the FGMRES solution.
  catamari::BlasMatrix<double> residual = right_hand_side;
  catamari::ApplySparse(-1., matrix, fgmres_solution.ConstView(), 1.,
                        &residual.view);
  const double residual_norm = catamari::EuclideanNorm(residual.ConstView());
  const double relative_residual_norm = residual_norm / right_hand_side_norm;
  const double epsilon = std::numeric_limits<double>::epsilon();
  const double relative_tolerance =
      fgmres_control.relative_tolerance_coefficient *
      std::pow(epsilon, fgmres_control.relative_tolerance_exponent);
  REQUIRE(relative_residual_norm <= relative_tolerance);

  // With a restart parameter of 10, it should only take 5 iterations to reach
  // a tolerance of 2.5e-16. With a restart parameter of 2, we should converge
  // after 11 inner iterations.
  REQUIRE(fgmres_status.num_iterations == 11);
}
