/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#define CATCH_CONFIG_MAIN
#include <iostream>
#include <limits>

#include "catamari/apply_sparse.hpp"
#include "catamari/blas_matrix.hpp"
#include "catamari/norms.hpp"
#include "catamari/sparse_ldl.hpp"
#include "catch2/catch.hpp"

using catamari::BlasMatrix;
using catamari::Buffer;
using catamari::ConstBlasMatrixView;
using catamari::Int;

namespace {

// Returns the Experiment statistics for a single Matrix Market input matrix.
template <typename Field>
void RunTest(Int num_rows,
             catamari::SymmetricFactorizationType factorization_type,
             catamari::LDLAlgorithm ldl_algorithm, bool analytical_ordering,
             bool dynamically_regularize, const Buffer<bool>& signatures,
             catamari::ComplexBase<Field> positive_threshold_exponent,
             catamari::ComplexBase<Field> negative_threshold_exponent) {
  typedef catamari::ComplexBase<Field> Real;
  const Real kEpsilon = std::numeric_limits<Real>::epsilon();
  const Real large_value = 1;
  const Real small_value = kEpsilon / 2;

  catamari::DynamicRegularizationControl<Field> dynamic_reg_control;
  dynamic_reg_control.enabled = dynamically_regularize;
  dynamic_reg_control.signatures = signatures;
  dynamic_reg_control.positive_threshold_exponent = positive_threshold_exponent;
  dynamic_reg_control.negative_threshold_exponent = negative_threshold_exponent;

  catamari::SparseLDLControl<Field> ldl_control;
  ldl_control.SetFactorizationType(factorization_type);
  ldl_control.scalar_control.algorithm = ldl_algorithm;
  ldl_control.supernodal_control.algorithm = ldl_algorithm;
  ldl_control.supernodal_strategy = catamari::kSupernodalFactorization;
  ldl_control.SetDynamicRegularization(dynamic_reg_control);

  // Construct the scaled negative laplacian.
  catamari::CoordinateMatrix<Field> matrix;
  matrix.Resize(num_rows, num_rows);
  matrix.ReserveEntryAdditions(num_rows);
  for (Int index = 0; index < num_rows; ++index) {
    if (index % 2) {
      if (signatures[index]) {
        matrix.QueueEntryAddition(index, index, large_value);
      } else {
        matrix.QueueEntryAddition(index, index, -large_value);
      }
    } else {
      if (signatures[index]) {
        matrix.QueueEntryAddition(index, index, small_value);
      } else {
        matrix.QueueEntryAddition(index, index, -small_value);
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
  result = ldl.Factor(matrix, ldl_control);
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

void RunTests(Int num_rows,
              catamari::SymmetricFactorizationType factorization_type,
              catamari::LDLAlgorithm ldl_algorithm, bool analytical_ordering,
              bool dynamically_regularize, const Buffer<bool>& signatures,
              double positive_threshold_exponent,
              double negative_threshold_exponent) {
  RunTest<float>(num_rows, factorization_type, ldl_algorithm,
                 analytical_ordering, dynamically_regularize, signatures,
                 positive_threshold_exponent, negative_threshold_exponent);
  RunTest<double>(num_rows, factorization_type, ldl_algorithm,
                  analytical_ordering, dynamically_regularize, signatures,
                  positive_threshold_exponent, negative_threshold_exponent);
  RunTest<mantis::DoubleMantissa<float>>(
      num_rows, factorization_type, ldl_algorithm, analytical_ordering,
      dynamically_regularize, signatures, positive_threshold_exponent,
      negative_threshold_exponent);
  RunTest<mantis::DoubleMantissa<double>>(
      num_rows, factorization_type, ldl_algorithm, analytical_ordering,
      dynamically_regularize, signatures, positive_threshold_exponent,
      negative_threshold_exponent);
  RunTest<mantis::Complex<float>>(num_rows, factorization_type, ldl_algorithm,
                                  analytical_ordering, dynamically_regularize,
                                  signatures, positive_threshold_exponent,
                                  negative_threshold_exponent);
  RunTest<mantis::Complex<double>>(num_rows, factorization_type, ldl_algorithm,
                                   analytical_ordering, dynamically_regularize,
                                   signatures, positive_threshold_exponent,
                                   negative_threshold_exponent);
  RunTest<mantis::Complex<mantis::DoubleMantissa<float>>>(
      num_rows, factorization_type, ldl_algorithm, analytical_ordering,
      dynamically_regularize, signatures, positive_threshold_exponent,
      negative_threshold_exponent);
  RunTest<mantis::Complex<mantis::DoubleMantissa<double>>>(
      num_rows, factorization_type, ldl_algorithm, analytical_ordering,
      dynamically_regularize, signatures, positive_threshold_exponent,
      negative_threshold_exponent);
}

}  // anonymous namespace

TEST_CASE("2D right Cholesky", "2D right Cholesky") {
  const Int num_rows = 1000;
  const catamari::SymmetricFactorizationType factorization_type =
      catamari::kCholeskyFactorization;
  const catamari::LDLAlgorithm ldl_algorithm = catamari::kRightLookingLDL;
  const bool analytical_ordering = false;
  const bool dynamically_regularize = true;
  Buffer<bool> signatures(num_rows, true);
  const double positive_threshold_exponent = 0.5;
  const double negative_threshold_exponent = 0.5;

  RunTests(num_rows, factorization_type, ldl_algorithm, analytical_ordering,
           dynamically_regularize, signatures, positive_threshold_exponent,
           negative_threshold_exponent);
}

TEST_CASE("2D left Cholesky", "2D left Cholesky") {
  const Int num_rows = 1000;
  const catamari::SymmetricFactorizationType factorization_type =
      catamari::kCholeskyFactorization;
  const catamari::LDLAlgorithm ldl_algorithm = catamari::kLeftLookingLDL;
  const bool analytical_ordering = false;
  const bool dynamically_regularize = true;
  Buffer<bool> signatures(num_rows, true);
  const double positive_threshold_exponent = 0.5;
  const double negative_threshold_exponent = 0.5;

  RunTests(num_rows, factorization_type, ldl_algorithm, analytical_ordering,
           dynamically_regularize, signatures, positive_threshold_exponent,
           negative_threshold_exponent);
}

TEST_CASE("2D right adjoint", "2D right adjoint") {
  const Int num_rows = 1000;
  const catamari::SymmetricFactorizationType factorization_type =
      catamari::kLDLAdjointFactorization;
  const catamari::LDLAlgorithm ldl_algorithm = catamari::kRightLookingLDL;
  const bool analytical_ordering = false;
  const bool dynamically_regularize = true;
  Buffer<bool> signatures(num_rows);
  for (Int i = 0; i < num_rows; ++i) {
    if (i % 6) {
      signatures[i] = false;
    } else {
      signatures[i] = true;
    }
  }
  const double positive_threshold_exponent = 0.5;
  const double negative_threshold_exponent = 0.5;

  RunTests(num_rows, factorization_type, ldl_algorithm, analytical_ordering,
           dynamically_regularize, signatures, positive_threshold_exponent,
           negative_threshold_exponent);
}

TEST_CASE("2D left adjoint", "2D left adjoint") {
  const Int num_rows = 1000;
  const catamari::SymmetricFactorizationType factorization_type =
      catamari::kLDLAdjointFactorization;
  const catamari::LDLAlgorithm ldl_algorithm = catamari::kLeftLookingLDL;
  const bool analytical_ordering = false;
  const bool dynamically_regularize = true;
  Buffer<bool> signatures(num_rows);
  for (Int i = 0; i < num_rows; ++i) {
    if (i % 6) {
      signatures[i] = false;
    } else {
      signatures[i] = true;
    }
  }
  const double positive_threshold_exponent = 0.5;
  const double negative_threshold_exponent = 0.5;

  RunTests(num_rows, factorization_type, ldl_algorithm, analytical_ordering,
           dynamically_regularize, signatures, positive_threshold_exponent,
           negative_threshold_exponent);
}
