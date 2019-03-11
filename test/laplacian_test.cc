/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#define CATCH_CONFIG_MAIN
#include <iostream>
#include "catamari/apply_sparse.hpp"
#include "catamari/blas_matrix.hpp"
#include "catamari/ldl.hpp"
#include "catamari/norms.hpp"
#include "catamari/unit_reach_nested_dissection.hpp"
#include "catch2/catch.hpp"

using catamari::BlasMatrix;
using catamari::ConstBlasMatrixView;
using catamari::Int;

namespace {

// Returns the Experiment statistics for a single Matrix Market input matrix.
void RunTest(Int num_x_elements, Int num_y_elements, bool analytical_ordering,
             const catamari::LDLControl& ldl_control) {
  typedef double Field;
  typedef catamari::ComplexBase<Field> Real;

  // Construct the scaled negative laplacian.
  catamari::CoordinateMatrix<Field> matrix;
  const Int num_rows = num_x_elements * num_y_elements;
  matrix.Resize(num_rows, num_rows);
  matrix.ReserveEntryAdditions(5 * num_rows);
  for (Int x = 0; x < num_x_elements; ++x) {
    for (Int y = 0; y < num_y_elements; ++y) {
      const Int index = x + y * num_x_elements;
      matrix.QueueEntryAddition(index, index, Field{4});
      if (x > 0) {
        matrix.QueueEntryAddition(index, index - 1, Field{-1});
      }
      if (x < num_x_elements - 1) {
        matrix.QueueEntryAddition(index, index + 1, Field{-1});
      }
      if (y > 0) {
        matrix.QueueEntryAddition(index, index - num_x_elements, Field{-1});
      }
      if (y < num_y_elements - 1) {
        matrix.QueueEntryAddition(index, index + num_x_elements, Field{-1});
      }
    }
  }
  matrix.FlushEntryQueues();

  // Construct the right-hand side and its norm.
  BlasMatrix<Field> right_hand_sides;
  right_hand_sides.Resize(num_rows, 1, Field{0});
  right_hand_sides(num_rows / 2, 0) = Field{1};
  const Real right_hand_side_norm =
      catamari::EuclideanNorm(right_hand_sides.ConstView());

  // Factor the matrix.
  catamari::LDLFactorization<Field> ldl_factorization;
  catamari::LDLResult result;
  if (analytical_ordering) {
    catamari::SymmetricOrdering ordering;
    catamari::UnitReachNestedDissection2D(num_x_elements - 1,
                                          num_y_elements - 1, &ordering);
    result = ldl_factorization.Factor(matrix, ordering, ldl_control);
  } else {
    result = ldl_factorization.Factor(matrix, ldl_control);
  }
  REQUIRE(result.num_successful_pivots == num_rows);

  // Solve a random linear system.
  BlasMatrix<Field> solution = right_hand_sides;
  ldl_factorization.Solve(&solution.view);

  // Compute the residual.
  BlasMatrix<Field> residual = right_hand_sides;
  catamari::ApplySparse(Field{-1}, matrix, solution.ConstView(), Field{1},
                        &residual.view);
  const Real residual_norm = catamari::EuclideanNorm(residual.ConstView());
  const Real relative_residual = residual_norm / right_hand_side_norm;
  REQUIRE(relative_residual < 1e-12);
}

}  // anonymous namespace

TEST_CASE("2D right Cholesky [analytical]", "2D right chol [analyt]") {
  const Int num_x_elements = 400;
  const Int num_y_elements = 400;
  const bool analytical_ordering = true;
  const int ldl_algorithm_int = 2;

  catamari::LDLControl ldl_control;
  ldl_control.SetFactorizationType(catamari::kCholeskyFactorization);
  ldl_control.scalar_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);
  ldl_control.supernodal_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);

  RunTest(num_x_elements, num_y_elements, analytical_ordering, ldl_control);
}

TEST_CASE("2D right adjoint [analytical]", "2D right adjoint [analyt]") {
  const Int num_x_elements = 400;
  const Int num_y_elements = 400;
  const bool analytical_ordering = true;
  const int ldl_algorithm_int = 2;

  catamari::LDLControl ldl_control;
  ldl_control.SetFactorizationType(catamari::kLDLAdjointFactorization);
  ldl_control.scalar_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);
  ldl_control.supernodal_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);

  RunTest(num_x_elements, num_y_elements, analytical_ordering, ldl_control);
}

TEST_CASE("2D right transpose [analytical]", "2D right transpose [analyt]") {
  const Int num_x_elements = 400;
  const Int num_y_elements = 400;
  const bool analytical_ordering = true;
  const int ldl_algorithm_int = 2;

  catamari::LDLControl ldl_control;
  ldl_control.SetFactorizationType(catamari::kLDLTransposeFactorization);
  ldl_control.scalar_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);
  ldl_control.supernodal_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);

  RunTest(num_x_elements, num_y_elements, analytical_ordering, ldl_control);
}

TEST_CASE("2D right Cholesky", "2D right chol") {
  const Int num_x_elements = 400;
  const Int num_y_elements = 400;
  const bool analytical_ordering = false;
  const int ldl_algorithm_int = 2;

  catamari::LDLControl ldl_control;
  ldl_control.SetFactorizationType(catamari::kCholeskyFactorization);
  ldl_control.scalar_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);
  ldl_control.supernodal_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);

  RunTest(num_x_elements, num_y_elements, analytical_ordering, ldl_control);
}

TEST_CASE("2D right adjoint", "2D right adjoint") {
  const Int num_x_elements = 400;
  const Int num_y_elements = 400;
  const bool analytical_ordering = false;
  const int ldl_algorithm_int = 2;

  catamari::LDLControl ldl_control;
  ldl_control.SetFactorizationType(catamari::kLDLAdjointFactorization);
  ldl_control.scalar_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);
  ldl_control.supernodal_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);

  RunTest(num_x_elements, num_y_elements, analytical_ordering, ldl_control);
}

TEST_CASE("2D right transpose", "2D right transpose") {
  const Int num_x_elements = 400;
  const Int num_y_elements = 400;
  const bool analytical_ordering = false;
  const int ldl_algorithm_int = 2;

  catamari::LDLControl ldl_control;
  ldl_control.SetFactorizationType(catamari::kLDLTransposeFactorization);
  ldl_control.scalar_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);
  ldl_control.supernodal_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);

  RunTest(num_x_elements, num_y_elements, analytical_ordering, ldl_control);
}

TEST_CASE("2D left Cholesky [analytical]", "2D left chol [analyt]") {
  const Int num_x_elements = 400;
  const Int num_y_elements = 400;
  const bool analytical_ordering = true;
  const int ldl_algorithm_int = 0;

  catamari::LDLControl ldl_control;
  ldl_control.SetFactorizationType(catamari::kCholeskyFactorization);
  ldl_control.scalar_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);
  ldl_control.supernodal_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);

  RunTest(num_x_elements, num_y_elements, analytical_ordering, ldl_control);
}

TEST_CASE("2D left adjoint [analytical]", "2D left adjoint [analyt]") {
  const Int num_x_elements = 400;
  const Int num_y_elements = 400;
  const bool analytical_ordering = true;
  const int ldl_algorithm_int = 0;

  catamari::LDLControl ldl_control;
  ldl_control.SetFactorizationType(catamari::kLDLAdjointFactorization);
  ldl_control.scalar_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);
  ldl_control.supernodal_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);

  RunTest(num_x_elements, num_y_elements, analytical_ordering, ldl_control);
}

TEST_CASE("2D left transpose [analytical]", "2D left transpose [analyt]") {
  const Int num_x_elements = 400;
  const Int num_y_elements = 400;
  const bool analytical_ordering = true;
  const int ldl_algorithm_int = 0;

  catamari::LDLControl ldl_control;
  ldl_control.SetFactorizationType(catamari::kLDLTransposeFactorization);
  ldl_control.scalar_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);
  ldl_control.supernodal_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);

  RunTest(num_x_elements, num_y_elements, analytical_ordering, ldl_control);
}

TEST_CASE("2D left Cholesky", "2D left chol") {
  const Int num_x_elements = 400;
  const Int num_y_elements = 400;
  const bool analytical_ordering = false;
  const int ldl_algorithm_int = 0;

  catamari::LDLControl ldl_control;
  ldl_control.SetFactorizationType(catamari::kCholeskyFactorization);
  ldl_control.scalar_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);
  ldl_control.supernodal_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);

  RunTest(num_x_elements, num_y_elements, analytical_ordering, ldl_control);
}

TEST_CASE("2D left adjoint", "2D left adjoint") {
  const Int num_x_elements = 400;
  const Int num_y_elements = 400;
  const bool analytical_ordering = false;
  const int ldl_algorithm_int = 0;

  catamari::LDLControl ldl_control;
  ldl_control.SetFactorizationType(catamari::kLDLAdjointFactorization);
  ldl_control.scalar_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);
  ldl_control.supernodal_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);

  RunTest(num_x_elements, num_y_elements, analytical_ordering, ldl_control);
}

TEST_CASE("2D left transpose", "2D left transpose") {
  const Int num_x_elements = 400;
  const Int num_y_elements = 400;
  const bool analytical_ordering = false;
  const int ldl_algorithm_int = 0;

  catamari::LDLControl ldl_control;
  ldl_control.SetFactorizationType(catamari::kLDLTransposeFactorization);
  ldl_control.scalar_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);
  ldl_control.supernodal_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);

  RunTest(num_x_elements, num_y_elements, analytical_ordering, ldl_control);
}
