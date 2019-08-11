/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#define CATCH_CONFIG_MAIN
#include <iostream>
#include "catamari/blas_matrix.hpp"
#include "catamari/dense_factorizations.hpp"
#include "catamari/norms.hpp"
#include "catch2/catch.hpp"

using catamari::BlasMatrix;
using catamari::BlasMatrixView;
using catamari::Complex;
using catamari::ConstBlasMatrixView;
using catamari::Int;
using quotient::Buffer;

namespace {

template <typename Real>
void InitializeHPD(Int matrix_size, BlasMatrix<Real>* matrix) {
  matrix->Resize(matrix_size, matrix_size, Real{0});
  for (Int j = 0; j < matrix_size; ++j) {
    matrix->Entry(j, j) = 2 * matrix_size;
    for (Int i = j + 1; i < matrix_size; ++i) {
      matrix->Entry(i, j) = Real{-1} / i;
      matrix->Entry(j, i) = matrix->Entry(i, j);
    }
  }
}

template <typename Real>
void InitializeHPD(Int matrix_size, BlasMatrix<Complex<Real>>* matrix) {
  typedef Complex<Real> Field;
  matrix->Resize(matrix_size, matrix_size, Field{0});
  for (Int j = 0; j < matrix_size; ++j) {
    matrix->Entry(j, j) = 4 * matrix_size;
    for (Int i = j + 1; i < matrix_size; ++i) {
      matrix->Entry(i, j) = Complex<Real>{Real{-1} / i, Real{1}};
      matrix->Entry(j, i) = catamari::Conjugate(matrix->Entry(i, j));
    }
  }
}

template <typename Real>
void InitializeHermitian(Int matrix_size, BlasMatrix<Real>* matrix) {
  matrix->Resize(matrix_size, matrix_size, Real{0});
  for (Int j = 0; j < matrix_size; ++j) {
    matrix->Entry(j, j) = 2 * matrix_size;
    for (Int i = j + 1; i < matrix_size; ++i) {
      matrix->Entry(i, j) = Real{-1} / i;
      matrix->Entry(j, i) = matrix->Entry(i, j);
    }
  }
}

template <typename Real>
void InitializeHermitian(Int matrix_size, BlasMatrix<Complex<Real>>* matrix) {
  typedef Complex<Real> Field;
  matrix->Resize(matrix_size, matrix_size, Field{0});
  for (Int j = 0; j < matrix_size; ++j) {
    matrix->Entry(j, j) = 4 * matrix_size;
    for (Int i = j + 1; i < matrix_size; ++i) {
      matrix->Entry(i, j) = Complex<Real>{Real{-1} / i, Real{1}};
      matrix->Entry(j, i) = catamari::Conjugate(matrix->Entry(i, j));
    }
  }
}

template <typename Real>
void InitializeSymmetric(Int matrix_size, BlasMatrix<Real>* matrix) {
  matrix->Resize(matrix_size, matrix_size, Real{0});
  for (Int j = 0; j < matrix_size; ++j) {
    matrix->Entry(j, j) = 2 * matrix_size;
    for (Int i = j + 1; i < matrix_size; ++i) {
      matrix->Entry(i, j) = Real{-1} / i;
      matrix->Entry(j, i) = matrix->Entry(i, j);
    }
  }
}

template <typename Real>
void InitializeSymmetric(Int matrix_size, BlasMatrix<Complex<Real>>* matrix) {
  typedef Complex<Real> Field;
  matrix->Resize(matrix_size, matrix_size, Field{0});
  for (Int j = 0; j < matrix_size; ++j) {
    matrix->Entry(j, j) = 4 * matrix_size;
    for (Int i = j + 1; i < matrix_size; ++i) {
      matrix->Entry(i, j) = Complex<Real>{Real{-1} / i, Real{1}};
      matrix->Entry(j, i) = matrix->Entry(i, j);
    }
  }
}

template <typename Field>
void RunCholeskyFactorization(Int tile_size, Int block_size,
                              BlasMatrix<Field>* matrix) {
  typedef catamari::ComplexBase<Field> Real;
  const BlasMatrix<Field> matrix_copy = *matrix;

  int num_pivots;
#ifdef CATAMARI_OPENMP
  const int old_max_threads = catamari::GetMaxBlasThreads();
  catamari::SetNumBlasThreads(1);
  #pragma omp parallel
  #pragma omp single
  num_pivots =
      OpenMPLowerCholeskyFactorization(tile_size, block_size, &matrix->view);
  catamari::SetNumBlasThreads(old_max_threads);
#else
  num_pivots = LowerCholeskyFactorization(block_size, &matrix->view);
#endif

  const Int matrix_size = matrix->view.height;
  REQUIRE(num_pivots == matrix_size);

  BlasMatrix<Field> right_hand_side;
  right_hand_side.Resize(matrix_size, 1, Field{0});
  right_hand_side(matrix_size / 2, 0) = Field{1};
  const Real right_hand_side_norm =
      catamari::EuclideanNorm(right_hand_side.ConstView());

  BlasMatrix<Field> solution = right_hand_side;
  catamari::LeftLowerTriangularSolves(matrix->ConstView(), &solution.view);
  catamari::LeftLowerAdjointTriangularSolves(matrix->ConstView(),
                                             &solution.view);

  BlasMatrix<Field> residual = right_hand_side;
  catamari::MatrixMultiplyNormalNormal(Field{-1}, matrix_copy.ConstView(),
                                       solution.ConstView(), Field{1},
                                       &residual.view);
  const Real residual_norm = catamari::EuclideanNorm(residual.ConstView());
  const Real relative_residual = residual_norm / right_hand_side_norm;
  const Real tolerance = 100 * std::numeric_limits<Real>::epsilon();
  REQUIRE(relative_residual <= tolerance);
}

template <typename Field>
void RunLDLAdjointFactorization(Int tile_size, Int block_size,
                                BlasMatrix<Field>* matrix,
                                Buffer<Field>* extra_buffer) {
  typedef catamari::ComplexBase<Field> Real;
  const BlasMatrix<Field> matrix_copy = *matrix;

  int num_pivots;
#ifdef CATAMARI_OPENMP
  const int old_max_threads = catamari::GetMaxBlasThreads();
  catamari::SetNumBlasThreads(1);
  #pragma omp parallel
  #pragma omp single
  num_pivots = OpenMPLDLAdjointFactorization(tile_size, block_size,
                                             &matrix->view, extra_buffer);
  catamari::SetNumBlasThreads(old_max_threads);
#else
  num_pivots = LDLAdjointFactorization(block_size, &matrix->view);
#endif

  const Int matrix_size = matrix->view.height;
  REQUIRE(num_pivots == matrix_size);

  BlasMatrix<Field> right_hand_side;
  right_hand_side.Resize(matrix_size, 1, Field{0});
  right_hand_side(matrix_size / 2, 0) = Field{1};
  const Real right_hand_side_norm =
      catamari::EuclideanNorm(right_hand_side.ConstView());

  BlasMatrix<Field> solution = right_hand_side;
  catamari::LeftLowerUnitTriangularSolves(matrix->ConstView(), &solution.view);
  for (Int i = 0; i < matrix_size; ++i) {
    solution(i, 0) /= matrix->Entry(i, i);
  }
  catamari::LeftLowerAdjointUnitTriangularSolves(matrix->ConstView(),
                                                 &solution.view);

  BlasMatrix<Field> residual = right_hand_side;
  catamari::MatrixMultiplyNormalNormal(Field{-1}, matrix_copy.ConstView(),
                                       solution.ConstView(), Field{1},
                                       &residual.view);
  const Real residual_norm = catamari::EuclideanNorm(residual.ConstView());
  const Real relative_residual = residual_norm / right_hand_side_norm;
  const Real tolerance = 100 * std::numeric_limits<Real>::epsilon();
  REQUIRE(relative_residual <= tolerance);
}

template <typename Field>
void RunPivotedLDLAdjointFactorization(Int tile_size, Int block_size,
                                       BlasMatrix<Field>* matrix,
                                       Buffer<Field>* extra_buffer) {
  typedef catamari::ComplexBase<Field> Real;
  const BlasMatrix<Field> matrix_copy = *matrix;
  const Int matrix_size = matrix->view.height;
  BlasMatrix<Int> permutation(matrix_size, 1);

  int num_pivots;
#ifdef CATAMARI_OPENMP
  const int old_max_threads = catamari::GetMaxBlasThreads();
  catamari::SetNumBlasThreads(1);
  #pragma omp parallel
  #pragma omp single
  num_pivots = OpenMPPivotedLDLAdjointFactorization(
      tile_size, block_size, &matrix->view, &permutation.view);
  catamari::SetNumBlasThreads(old_max_threads);
#else
  num_pivots = PivotedLDLAdjointFactorization(block_size, &matrix->view,
                                              &permutation.view);
#endif
  REQUIRE(num_pivots == matrix_size);

  BlasMatrix<Field> right_hand_side;
  right_hand_side.Resize(matrix_size, 1, Field{0});
  right_hand_side(matrix_size / 2, 0) = Field{1};
  const Real right_hand_side_norm =
      catamari::EuclideanNorm(right_hand_side.ConstView());

  BlasMatrix<Field> solution = right_hand_side;
  catamari::InversePermute(permutation.ConstView(), &solution.view);
  catamari::LeftLowerUnitTriangularSolves(matrix->ConstView(), &solution.view);
  for (Int i = 0; i < matrix_size; ++i) {
    solution(i, 0) /= matrix->Entry(i, i);
  }
  catamari::LeftLowerAdjointUnitTriangularSolves(matrix->ConstView(),
                                                 &solution.view);
  catamari::Permute(permutation.ConstView(), &solution.view);

  BlasMatrix<Field> residual = right_hand_side;
  catamari::MatrixMultiplyNormalNormal(Field{-1}, matrix_copy.ConstView(),
                                       solution.ConstView(), Field{1},
                                       &residual.view);
  const Real residual_norm = catamari::EuclideanNorm(residual.ConstView());
  const Real relative_residual = residual_norm / right_hand_side_norm;
  const Real tolerance = 100 * std::numeric_limits<Real>::epsilon();
  REQUIRE(relative_residual <= tolerance);
}

template <typename Field>
void RunLDLTransposeFactorization(Int tile_size, Int block_size,
                                  BlasMatrix<Field>* matrix,
                                  Buffer<Field>* extra_buffer) {
  typedef catamari::ComplexBase<Field> Real;
  const BlasMatrix<Field> matrix_copy = *matrix;

  int num_pivots;
#ifdef CATAMARI_OPENMP
  const int old_max_threads = catamari::GetMaxBlasThreads();
  catamari::SetNumBlasThreads(1);
  #pragma omp parallel
  #pragma omp single
  num_pivots = OpenMPLDLTransposeFactorization(tile_size, block_size,
                                               &matrix->view, extra_buffer);
  catamari::SetNumBlasThreads(old_max_threads);
#else
  num_pivots = LDLTransposeFactorization(block_size, &matrix->view);
#endif

  const Int matrix_size = matrix->view.height;
  REQUIRE(num_pivots == matrix_size);

  BlasMatrix<Field> right_hand_side;
  right_hand_side.Resize(matrix_size, 1, Field{0});
  right_hand_side(matrix_size / 2, 0) = Field{1};
  const Real right_hand_side_norm =
      catamari::EuclideanNorm(right_hand_side.ConstView());

  BlasMatrix<Field> solution = right_hand_side;
  catamari::LeftLowerUnitTriangularSolves(matrix->ConstView(), &solution.view);
  for (Int i = 0; i < matrix_size; ++i) {
    solution(i, 0) /= matrix->Entry(i, i);
  }
  catamari::LeftLowerTransposeUnitTriangularSolves(matrix->ConstView(),
                                                   &solution.view);

  BlasMatrix<Field> residual = right_hand_side;
  catamari::MatrixMultiplyNormalNormal(Field{-1}, matrix_copy.ConstView(),
                                       solution.ConstView(), Field{1},
                                       &residual.view);
  const Real residual_norm = catamari::EuclideanNorm(residual.ConstView());
  const Real relative_residual = residual_norm / right_hand_side_norm;
  const Real tolerance = 100 * std::numeric_limits<Real>::epsilon();
  REQUIRE(relative_residual <= tolerance);
}

}  // anonymous namespace

TEST_CASE("Double", "Double") {
  const Int matrix_size = 300;
  const Int tile_size = 128;
  const Int block_size = 64;
  const Int num_rounds = 2;

  BlasMatrix<double> matrix;
  Buffer<double> extra_buffer;

  for (Int round = 0; round < num_rounds; ++round) {
    InitializeHPD(matrix_size, &matrix);
    RunCholeskyFactorization(tile_size, block_size, &matrix);

    InitializeHermitian(matrix_size, &matrix);
    RunLDLAdjointFactorization(tile_size, block_size, &matrix, &extra_buffer);

    InitializeSymmetric(matrix_size, &matrix);
    RunLDLTransposeFactorization(tile_size, block_size, &matrix, &extra_buffer);

    InitializeHermitian(matrix_size, &matrix);
    RunPivotedLDLAdjointFactorization(tile_size, block_size, &matrix,
                                      &extra_buffer);
  }
}

TEST_CASE("DoubleFloat", "DoubleFloat") {
  const Int matrix_size = 300;
  const Int tile_size = 128;
  const Int block_size = 64;
  const Int num_rounds = 2;

  BlasMatrix<mantis::DoubleMantissa<float>> matrix;
  Buffer<mantis::DoubleMantissa<float>> extra_buffer;

  for (Int round = 0; round < num_rounds; ++round) {
    InitializeHPD(matrix_size, &matrix);
    RunCholeskyFactorization(tile_size, block_size, &matrix);

    InitializeHermitian(matrix_size, &matrix);
    RunLDLAdjointFactorization(tile_size, block_size, &matrix, &extra_buffer);

    InitializeSymmetric(matrix_size, &matrix);
    RunLDLTransposeFactorization(tile_size, block_size, &matrix, &extra_buffer);

    InitializeHermitian(matrix_size, &matrix);
    RunPivotedLDLAdjointFactorization(tile_size, block_size, &matrix,
                                      &extra_buffer);
  }
}

TEST_CASE("DoubleDouble", "DoubleDouble") {
  const Int matrix_size = 300;
  const Int tile_size = 128;
  const Int block_size = 64;
  const Int num_rounds = 2;

  BlasMatrix<mantis::DoubleMantissa<double>> matrix;
  Buffer<mantis::DoubleMantissa<double>> extra_buffer;

  for (Int round = 0; round < num_rounds; ++round) {
    InitializeHPD(matrix_size, &matrix);
    RunCholeskyFactorization(tile_size, block_size, &matrix);

    InitializeHermitian(matrix_size, &matrix);
    RunLDLAdjointFactorization(tile_size, block_size, &matrix, &extra_buffer);

    InitializeSymmetric(matrix_size, &matrix);
    RunLDLTransposeFactorization(tile_size, block_size, &matrix, &extra_buffer);

    InitializeHermitian(matrix_size, &matrix);
    RunPivotedLDLAdjointFactorization(tile_size, block_size, &matrix,
                                      &extra_buffer);
  }
}

TEST_CASE("ComplexDouble", "ComplexDouble") {
  const Int matrix_size = 300;
  const Int tile_size = 128;
  const Int block_size = 64;
  const Int num_rounds = 2;

  BlasMatrix<Complex<double>> matrix;
  Buffer<Complex<double>> extra_buffer;

  for (Int round = 0; round < num_rounds; ++round) {
    InitializeHPD(matrix_size, &matrix);
    RunCholeskyFactorization(tile_size, block_size, &matrix);

    InitializeHermitian(matrix_size, &matrix);
    RunLDLAdjointFactorization(tile_size, block_size, &matrix, &extra_buffer);

    InitializeSymmetric(matrix_size, &matrix);
    RunLDLTransposeFactorization(tile_size, block_size, &matrix, &extra_buffer);

    InitializeHermitian(matrix_size, &matrix);
    RunPivotedLDLAdjointFactorization(tile_size, block_size, &matrix,
                                      &extra_buffer);
  }
}

TEST_CASE("ComplexDoubleFloat", "ComplexDoubleFloat") {
  const Int matrix_size = 300;
  const Int tile_size = 128;
  const Int block_size = 64;
  const Int num_rounds = 2;

  BlasMatrix<Complex<mantis::DoubleMantissa<float>>> matrix;
  Buffer<Complex<mantis::DoubleMantissa<float>>> extra_buffer;

  for (Int round = 0; round < num_rounds; ++round) {
    InitializeHPD(matrix_size, &matrix);
    RunCholeskyFactorization(tile_size, block_size, &matrix);

    InitializeHermitian(matrix_size, &matrix);
    RunLDLAdjointFactorization(tile_size, block_size, &matrix, &extra_buffer);

    InitializeSymmetric(matrix_size, &matrix);
    RunLDLTransposeFactorization(tile_size, block_size, &matrix, &extra_buffer);

    InitializeHermitian(matrix_size, &matrix);
    RunPivotedLDLAdjointFactorization(tile_size, block_size, &matrix,
                                      &extra_buffer);
  }
}

TEST_CASE("ComplexDoubleDouble", "ComplexDoubleDouble") {
  const Int matrix_size = 300;
  const Int tile_size = 128;
  const Int block_size = 64;
  const Int num_rounds = 2;

  BlasMatrix<Complex<mantis::DoubleMantissa<double>>> matrix;
  Buffer<Complex<mantis::DoubleMantissa<double>>> extra_buffer;

  for (Int round = 0; round < num_rounds; ++round) {
    InitializeHPD(matrix_size, &matrix);
    RunCholeskyFactorization(tile_size, block_size, &matrix);

    InitializeHermitian(matrix_size, &matrix);
    RunLDLAdjointFactorization(tile_size, block_size, &matrix, &extra_buffer);

    InitializeSymmetric(matrix_size, &matrix);
    RunLDLTransposeFactorization(tile_size, block_size, &matrix, &extra_buffer);

    InitializeHermitian(matrix_size, &matrix);
    RunPivotedLDLAdjointFactorization(tile_size, block_size, &matrix,
                                      &extra_buffer);
  }
}
