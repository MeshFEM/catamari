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
#include "catamari/norms.hpp"
#include "catamari/sparse_ldl.hpp"
#include "catamari/unit_reach_nested_dissection.hpp"
#include "catch2/catch.hpp"

using catamari::BlasMatrix;
using catamari::Buffer;
using catamari::ConstBlasMatrixView;
using catamari::Int;

namespace {

// Returns the Experiment statistics for a single Matrix Market input matrix.
template <typename Field>
void RunTest(Int num_x_elements, Int num_y_elements,
             catamari::SymmetricFactorizationType factorization_type,
             catamari::LDLAlgorithm ldl_algorithm, bool analytical_ordering,
             bool dynamically_regularize, const Buffer<bool>& signatures,
             catamari::ComplexBase<Field> positive_threshold_exponent,
             catamari::ComplexBase<Field> negative_threshold_exponent) {
  typedef catamari::ComplexBase<Field> Real;

  catamari::DynamicRegularizationControl<Field> dynamic_reg_control;
  dynamic_reg_control.enabled = dynamically_regularize;
  dynamic_reg_control.signatures = signatures;
  dynamic_reg_control.positive_threshold_exponent = positive_threshold_exponent;
  dynamic_reg_control.negative_threshold_exponent = negative_threshold_exponent;

  catamari::SparseLDLControl<Field> ldl_control;
  ldl_control.SetFactorizationType(factorization_type);
  ldl_control.scalar_control.algorithm = ldl_algorithm;
  ldl_control.supernodal_control.algorithm = ldl_algorithm;
  ldl_control.SetDynamicRegularization(dynamic_reg_control);

  // Construct the scaled negative laplacian.
  catamari::CoordinateMatrix<Field> matrix;
  const Int num_rows = num_x_elements * num_y_elements;
  matrix.Resize(num_rows, num_rows);
  matrix.ReserveEntryAdditions(5 * num_rows);
  for (Int x = 0; x < num_x_elements; ++x) {
    for (Int y = 0; y < num_y_elements; ++y) {
      const Int index = x + y * num_x_elements;
      matrix.QueueEntryAddition(index, index, Field{4} / num_rows);
      if (x > 0) {
        matrix.QueueEntryAddition(index, index - 1, Field{-1} / num_rows);
      }
      if (x < num_x_elements - 1) {
        matrix.QueueEntryAddition(index, index + 1, Field{-1} / num_rows);
      }
      if (y > 0) {
        matrix.QueueEntryAddition(index, index - num_x_elements,
                                  Field{-1} / num_rows);
      }
      if (y < num_y_elements - 1) {
        matrix.QueueEntryAddition(index, index + num_x_elements,
                                  Field{-1} / num_rows);
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
  catamari::SparseLDL<Field> ldl;
  catamari::SparseLDLResult<Field> result;
  if (analytical_ordering) {
    catamari::SymmetricOrdering ordering;
    catamari::UnitReachNestedDissection2D(num_x_elements - 1,
                                          num_y_elements - 1, &ordering);
    result = ldl.Factor(matrix, ordering, ldl_control);
  } else {
    result = ldl.Factor(matrix, ldl_control);
  }
  REQUIRE(result.num_successful_pivots == num_rows);

  // Get the diagonal regularization vector.
  BlasMatrix<Real> diagonal_reg;
  ldl.DynamicRegularizationDiagonal(result, &diagonal_reg);

  // Solve a random linear system.
  BlasMatrix<Field> solution = right_hand_sides;
  ldl.Solve(&solution.view);

  // Compute the residual:
  //
  //   b  - (A + diagonal_reg) x,
  //
  // and its norm.
  BlasMatrix<Field> residual = right_hand_sides;
  catamari::ApplySparse(Field{-1}, matrix, solution.ConstView(), Field{1},
                        &residual.view);
  for (Int i = 0; i < num_rows; ++i) {
    residual(i) -= diagonal_reg(i) * solution(i);
  }
  const Real residual_norm = catamari::EuclideanNorm(residual.ConstView());
  const Real relative_residual = residual_norm / right_hand_side_norm;
  const Real tolerance = 100 * std::numeric_limits<Real>::epsilon();
  REQUIRE(relative_residual <= tolerance);
}

void RunTests(Int num_x_elements, Int num_y_elements,
              catamari::SymmetricFactorizationType factorization_type,
              catamari::LDLAlgorithm ldl_algorithm, bool analytical_ordering,
              bool dynamically_regularize, const Buffer<bool>& signatures,
              double positive_threshold_exponent,
              double negative_threshold_exponent) {
  RunTest<float>(num_x_elements, num_y_elements, factorization_type,
                 ldl_algorithm, analytical_ordering, dynamically_regularize,
                 signatures, positive_threshold_exponent,
                 negative_threshold_exponent);
  RunTest<double>(num_x_elements, num_y_elements, factorization_type,
                  ldl_algorithm, analytical_ordering, dynamically_regularize,
                  signatures, positive_threshold_exponent,
                  negative_threshold_exponent);
  RunTest<mantis::DoubleMantissa<float>>(
      num_x_elements, num_y_elements, factorization_type, ldl_algorithm,
      analytical_ordering, dynamically_regularize, signatures,
      positive_threshold_exponent, negative_threshold_exponent);
  RunTest<mantis::DoubleMantissa<double>>(
      num_x_elements, num_y_elements, factorization_type, ldl_algorithm,
      analytical_ordering, dynamically_regularize, signatures,
      positive_threshold_exponent, negative_threshold_exponent);
  RunTest<mantis::Complex<float>>(
      num_x_elements, num_y_elements, factorization_type, ldl_algorithm,
      analytical_ordering, dynamically_regularize, signatures,
      positive_threshold_exponent, negative_threshold_exponent);
  RunTest<mantis::Complex<double>>(
      num_x_elements, num_y_elements, factorization_type, ldl_algorithm,
      analytical_ordering, dynamically_regularize, signatures,
      positive_threshold_exponent, negative_threshold_exponent);
  RunTest<mantis::Complex<mantis::DoubleMantissa<float>>>(
      num_x_elements, num_y_elements, factorization_type, ldl_algorithm,
      analytical_ordering, dynamically_regularize, signatures,
      positive_threshold_exponent, negative_threshold_exponent);
  RunTest<mantis::Complex<mantis::DoubleMantissa<double>>>(
      num_x_elements, num_y_elements, factorization_type, ldl_algorithm,
      analytical_ordering, dynamically_regularize, signatures,
      positive_threshold_exponent, negative_threshold_exponent);
}

}  // anonymous namespace

TEST_CASE("2D right Cholesky [analytical]", "2D right chol [analyt]") {
  const Int num_x_elements = 80;
  const Int num_y_elements = 80;
  const catamari::SymmetricFactorizationType factorization_type =
      catamari::kCholeskyFactorization;
  const catamari::LDLAlgorithm ldl_algorithm = catamari::kRightLookingLDL;
  const bool analytical_ordering = true;
  const bool dynamically_regularize = false;
  Buffer<bool> signatures(num_x_elements * num_y_elements, true);
  const double positive_threshold_exponent = 1;
  const double negative_threshold_exponent = 1;

  RunTests(num_x_elements, num_y_elements, factorization_type, ldl_algorithm,
           analytical_ordering, dynamically_regularize, signatures,
           positive_threshold_exponent, negative_threshold_exponent);
}

TEST_CASE("2D right adjoint [analytical]", "2D right adjoint [analyt]") {
  const Int num_x_elements = 160;
  const Int num_y_elements = 160;
  const catamari::SymmetricFactorizationType factorization_type =
      catamari::kLDLAdjointFactorization;
  const catamari::LDLAlgorithm ldl_algorithm = catamari::kRightLookingLDL;
  const bool analytical_ordering = true;
  const bool dynamically_regularize = false;
  Buffer<bool> signatures(num_x_elements * num_y_elements, true);
  const double positive_threshold_exponent = 1;
  const double negative_threshold_exponent = 1;

  RunTests(num_x_elements, num_y_elements, factorization_type, ldl_algorithm,
           analytical_ordering, dynamically_regularize, signatures,
           positive_threshold_exponent, negative_threshold_exponent);
}

TEST_CASE("2D right transpose [analytical]", "2D right transpose [analyt]") {
  const Int num_x_elements = 80;
  const Int num_y_elements = 80;
  const catamari::SymmetricFactorizationType factorization_type =
      catamari::kLDLTransposeFactorization;
  const catamari::LDLAlgorithm ldl_algorithm = catamari::kRightLookingLDL;
  const bool analytical_ordering = true;
  const bool dynamically_regularize = false;
  Buffer<bool> signatures(num_x_elements * num_y_elements, true);
  const double positive_threshold_exponent = 1.;
  const double negative_threshold_exponent = 1.;

  RunTests(num_x_elements, num_y_elements, factorization_type, ldl_algorithm,
           analytical_ordering, dynamically_regularize, signatures,
           positive_threshold_exponent, negative_threshold_exponent);
}

TEST_CASE("2D right Cholesky", "2D right chol") {
  const Int num_x_elements = 80;
  const Int num_y_elements = 80;
  const catamari::SymmetricFactorizationType factorization_type =
      catamari::kCholeskyFactorization;
  const catamari::LDLAlgorithm ldl_algorithm = catamari::kRightLookingLDL;
  const bool analytical_ordering = false;
  const bool dynamically_regularize = false;
  Buffer<bool> signatures(num_x_elements * num_y_elements, true);
  const double positive_threshold_exponent = 1.;
  const double negative_threshold_exponent = 1.;

  RunTests(num_x_elements, num_y_elements, factorization_type, ldl_algorithm,
           analytical_ordering, dynamically_regularize, signatures,
           positive_threshold_exponent, negative_threshold_exponent);
}

TEST_CASE("2D right adjoint", "2D right adjoint") {
  const Int num_x_elements = 80;
  const Int num_y_elements = 80;
  const catamari::SymmetricFactorizationType factorization_type =
      catamari::kLDLAdjointFactorization;
  const catamari::LDLAlgorithm ldl_algorithm = catamari::kRightLookingLDL;
  const bool analytical_ordering = false;
  const bool dynamically_regularize = false;
  Buffer<bool> signatures(num_x_elements * num_y_elements, true);
  const double positive_threshold_exponent = 1.;
  const double negative_threshold_exponent = 1.;

  RunTests(num_x_elements, num_y_elements, factorization_type, ldl_algorithm,
           analytical_ordering, dynamically_regularize, signatures,
           positive_threshold_exponent, negative_threshold_exponent);
}

TEST_CASE("2D right adjoint pivoted", "2D right adjoint pivoted") {
  const Int num_x_elements = 80;
  const Int num_y_elements = 80;
  const catamari::SymmetricFactorizationType factorization_type =
      catamari::kLDLAdjointFactorization;
  const catamari::LDLAlgorithm ldl_algorithm = catamari::kRightLookingLDL;
  const bool analytical_ordering = false;
  const bool dynamically_regularize = false;
  Buffer<bool> signatures(num_x_elements * num_y_elements, true);
  const double positive_threshold_exponent = 1.;
  const double negative_threshold_exponent = 1.;

  RunTests(num_x_elements, num_y_elements, factorization_type, ldl_algorithm,
           analytical_ordering, dynamically_regularize, signatures,
           positive_threshold_exponent, negative_threshold_exponent);
}

TEST_CASE("2D right transpose", "2D right transpose") {
  const Int num_x_elements = 80;
  const Int num_y_elements = 80;
  const catamari::SymmetricFactorizationType factorization_type =
      catamari::kLDLTransposeFactorization;
  const catamari::LDLAlgorithm ldl_algorithm = catamari::kRightLookingLDL;
  const bool analytical_ordering = false;
  const bool dynamically_regularize = false;
  Buffer<bool> signatures(num_x_elements * num_y_elements, true);
  const double positive_threshold_exponent = 1.;
  const double negative_threshold_exponent = 1.;

  RunTests(num_x_elements, num_y_elements, factorization_type, ldl_algorithm,
           analytical_ordering, dynamically_regularize, signatures,
           positive_threshold_exponent, negative_threshold_exponent);
}

TEST_CASE("2D left Cholesky [analytical]", "2D left chol [analyt]") {
  const Int num_x_elements = 80;
  const Int num_y_elements = 80;
  const catamari::SymmetricFactorizationType factorization_type =
      catamari::kCholeskyFactorization;
  const catamari::LDLAlgorithm ldl_algorithm = catamari::kLeftLookingLDL;
  const bool analytical_ordering = true;
  const bool dynamically_regularize = false;
  Buffer<bool> signatures(num_x_elements * num_y_elements, true);
  const double positive_threshold_exponent = 1.;
  const double negative_threshold_exponent = 1.;

  RunTests(num_x_elements, num_y_elements, factorization_type, ldl_algorithm,
           analytical_ordering, dynamically_regularize, signatures,
           positive_threshold_exponent, negative_threshold_exponent);
}

TEST_CASE("2D left adjoint [analytical]", "2D left adjoint [analyt]") {
  const Int num_x_elements = 80;
  const Int num_y_elements = 80;
  const catamari::SymmetricFactorizationType factorization_type =
      catamari::kLDLAdjointFactorization;
  const catamari::LDLAlgorithm ldl_algorithm = catamari::kLeftLookingLDL;
  const bool analytical_ordering = true;
  const bool dynamically_regularize = false;
  Buffer<bool> signatures(num_x_elements * num_y_elements, true);
  const double positive_threshold_exponent = 1.;
  const double negative_threshold_exponent = 1.;

  RunTests(num_x_elements, num_y_elements, factorization_type, ldl_algorithm,
           analytical_ordering, dynamically_regularize, signatures,
           positive_threshold_exponent, negative_threshold_exponent);
}

TEST_CASE("2D left adjoint pivoted [analytical]",
          "2D left adjoint pivoted [analyt]") {
  const Int num_x_elements = 80;
  const Int num_y_elements = 80;
  const catamari::SymmetricFactorizationType factorization_type =
      catamari::kLDLAdjointFactorization;
  const catamari::LDLAlgorithm ldl_algorithm = catamari::kLeftLookingLDL;
  const bool analytical_ordering = true;
  const bool dynamically_regularize = false;
  Buffer<bool> signatures(num_x_elements * num_y_elements, true);
  const double positive_threshold_exponent = 1.;
  const double negative_threshold_exponent = 1.;

  RunTests(num_x_elements, num_y_elements, factorization_type, ldl_algorithm,
           analytical_ordering, dynamically_regularize, signatures,
           positive_threshold_exponent, negative_threshold_exponent);
}

TEST_CASE("2D left transpose [analytical]", "2D left transpose [analyt]") {
  const Int num_x_elements = 80;
  const Int num_y_elements = 80;
  const catamari::SymmetricFactorizationType factorization_type =
      catamari::kLDLTransposeFactorization;
  const catamari::LDLAlgorithm ldl_algorithm = catamari::kLeftLookingLDL;
  const bool analytical_ordering = true;
  const bool dynamically_regularize = false;
  Buffer<bool> signatures(num_x_elements * num_y_elements, true);
  const double positive_threshold_exponent = 1.;
  const double negative_threshold_exponent = 1.;

  RunTests(num_x_elements, num_y_elements, factorization_type, ldl_algorithm,
           analytical_ordering, dynamically_regularize, signatures,
           positive_threshold_exponent, negative_threshold_exponent);
}

TEST_CASE("2D left Cholesky", "2D left chol") {
  const Int num_x_elements = 80;
  const Int num_y_elements = 80;
  const catamari::SymmetricFactorizationType factorization_type =
      catamari::kCholeskyFactorization;
  const catamari::LDLAlgorithm ldl_algorithm = catamari::kLeftLookingLDL;
  const bool analytical_ordering = false;
  const bool dynamically_regularize = false;
  Buffer<bool> signatures(num_x_elements * num_y_elements, true);
  const double positive_threshold_exponent = 1.;
  const double negative_threshold_exponent = 1.;

  RunTests(num_x_elements, num_y_elements, factorization_type, ldl_algorithm,
           analytical_ordering, dynamically_regularize, signatures,
           positive_threshold_exponent, negative_threshold_exponent);
}

TEST_CASE("2D left adjoint", "2D left adjoint") {
  const Int num_x_elements = 80;
  const Int num_y_elements = 80;
  const catamari::SymmetricFactorizationType factorization_type =
      catamari::kLDLAdjointFactorization;
  const catamari::LDLAlgorithm ldl_algorithm = catamari::kLeftLookingLDL;
  const bool analytical_ordering = false;
  const bool dynamically_regularize = false;
  Buffer<bool> signatures(num_x_elements * num_y_elements, true);
  const double positive_threshold_exponent = 1.;
  const double negative_threshold_exponent = 1.;

  RunTests(num_x_elements, num_y_elements, factorization_type, ldl_algorithm,
           analytical_ordering, dynamically_regularize, signatures,
           positive_threshold_exponent, negative_threshold_exponent);
}

TEST_CASE("2D left transpose", "2D left transpose") {
  const Int num_x_elements = 80;
  const Int num_y_elements = 80;
  const catamari::SymmetricFactorizationType factorization_type =
      catamari::kLDLTransposeFactorization;
  const catamari::LDLAlgorithm ldl_algorithm = catamari::kLeftLookingLDL;
  const bool analytical_ordering = false;
  const bool dynamically_regularize = false;
  Buffer<bool> signatures(num_x_elements * num_y_elements, true);
  const double positive_threshold_exponent = 1.;
  const double negative_threshold_exponent = 1.;

  RunTests(num_x_elements, num_y_elements, factorization_type, ldl_algorithm,
           analytical_ordering, dynamically_regularize, signatures,
           positive_threshold_exponent, negative_threshold_exponent);
}
