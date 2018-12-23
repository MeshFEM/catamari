/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// This drive is a simple implementation of a 2D Helmholtz equation in the
// unit box, [0, 1]^2, with Perfectly Matched Layer absorbing boundary
// conditions on all sides. The discretization is over rectangles with bilinear,
// Lagrangian basis functions based at the corner points (Q4 elements). The
// bilinear form is integrated over each element using a simple tensor product
// of three-point 1D Gaussian quadratures.
//
// A good reference for the weak formulation of the 2D Helmholtz equation with
// PML is:
//
//   Erkki Heikkola, Tuomo Rossi, and Jari Toivanen,
//   "Fast solvers for the Helmholtz equation with a perfectly matched layer /
//   an absorbing boundary condition", 2002.
//
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include "catamari/apply_sparse.hpp"
#include "catamari/ldl.hpp"
#include "quotient/minimum_degree.hpp"
#include "specify.hpp"

using catamari::BlasMatrix;
using catamari::Complex;
using catamari::ComplexBase;
using catamari::Conjugate;
using catamari::ConstBlasMatrix;
using catamari::Int;

namespace {

// Fills 'points' and 'weights' with the transformed evaluation points and
// weights for the interval [a, b].
template <class Real>
void ThirdOrderGaussianPointsAndWeights(const Real& a, const Real& b,
                                        std::vector<Real>* points,
                                        std::vector<Real>* weights) {
  const Real scale = (b - a) / 2;

  // TODO(Jack Poulson): Avoid continually recreating these values by storing
  // them in a templated structure which we only construct once and pass in.
  const std::vector<Real> orig_points{-std::sqrt(Real{3}) / std::sqrt(Real{5}),
                                      Real{0},
                                      std::sqrt(Real{3}) / std::sqrt(Real{5})};

  const std::vector<Real> orig_weights{Real(5) / Real(9), Real(8) / Real(9),
                                       Real(5) / Real(9)};

  points->resize(3);
  weights->resize(3);
  for (int i = 0; i < 3; ++i) {
    (*points)[i] = a + scale * (1 + orig_points[i]);
    (*weights)[i] = scale * orig_weights[i];
  }
}

// A representation of an arbitrary axis-aligned rectangle,
//
//     [x_beg, x_end] x [y_beg, y_end].
//
template <typename Real>
struct Rectangle {
  Real x_beg;
  Real x_end;
  Real y_beg;
  Real y_end;
};

// A cache for the tensor-product quadrature points over an arbitrary rectangle.
template <typename Real>
struct RectangleQuadrature {
  std::vector<Real> x_points;
  std::vector<Real> x_weights;

  std::vector<Real> y_points;
  std::vector<Real> y_weights;

  RectangleQuadrature(const Rectangle<Real>& extent) {
    ThirdOrderGaussianPointsAndWeights(extent.x_beg, extent.x_end, &x_points,
                                       &x_weights);
    ThirdOrderGaussianPointsAndWeights(extent.y_beg, extent.y_end, &y_points,
                                       &y_weights);
  }
};

// A point in the 2D domain (i.e., [0, 1]^2).
template <typename Real>
struct Point {
  Real x;
  Real y;
};

// This currently uses third-order Gaussian quadrature. The basis functions
// over [-1, 1]^2 are:
//
//   \psi_{0, 0}(x, y) = (1 - x) (1 - y) / 4,
//   \psi_{0, 1}(x, y) = (1 - x) (1 + y) / 4,
//   \psi_{1, 0}(x, y) = (1 + x) (1 - y) / 4,
//   \psi_{1, 1}(x, y) = (1 + x) (1 + y) / 4.
//
// The gradients are:
//
//   grad \psi_{0, 0}(x, y) = [-(1 - y), -(1 - x)]^T / 4,
//   grad \psi_{0, 1}(x, y) = [-(1 + y),  (1 - x)]^T / 4,
//   grad \psi_{1, 0}(x, y) = [ (1 - y), -(1 + x)]^T / 4,
//   grad \psi_{1, 1}(x, y) = [ (1 + y),  (1 + x)]^T / 4.
//
// Over an element [x_beg, x_end] x [y_beg, y_end], where we denote the lengths
// by L_x and L_y, we have
//
//   \psi_{0, 0}(x, y) = (x_end - x) (y_end - y) / (L_x L_y),
//   \psi_{0, 1}(x, y) = (x_end - x) (y - y_beg) / (L_x L_y),
//   \psi_{1, 0}(x, y) = (x - x_beg) (y_end - y) / (L_x L_y),
//   \psi_{1, 1}(x, y) = (x - x_beg) (y - y_beg) / (L_x L_y).
//
// The gradients are:
//
//   grad \psi_{0, 0}(x, y) = [-(y_end - y), -(x_end - x)]^T / (L_x L_y),
//   grad \psi_{0, 1}(x, y) = [-(y - y_beg),  (x_end - x)]^T / (L_x L_y),
//   grad \psi_{1, 0}(x, y) = [ (y_end - y), -(x - x_beg)]^T / (L_x L_y),
//   grad \psi_{1, 1}(x, y) = [ (y - y_beg),  (x - x_beg)]^T / (L_x L_y).
//
template <class Real>
class HelmholtzWithPMLQ4Element {
 public:
  // The term -div (A grad u) involves a two-tensor A : Omega -> C^{2 x 2}.
  // The (i, j) index of the result is thus a function:
  //
  //   A_{i, j} : Omega -> C,  i, j in {0, 1}.
  //
  // Since our weight tensor is diagonal, only we need only provide
  //
  //   A_{i, i} : Omega -> C,  i in {0, 1}.
  //
  typedef std::function<Complex<Real>(int i, const Point<Real>&)>
      DiagonalWeightTensor;

  // The term -s u involves a scalar function s : Omega -> C, which we refer
  // to as the diagonal shift function.
  typedef std::function<Complex<Real>(const Point<Real>&)>
      DiagonalShiftFunction;

  // The constructor for the Q4 element.
  HelmholtzWithPMLQ4Element(const DiagonalWeightTensor weight_tensor,
                            const DiagonalShiftFunction shift_function)
      : weight_tensor_(weight_tensor), shift_function_(shift_function) {}

  // Returns \psi_{i, j} evaluated at the given point.
  Real Basis(int i, int j, const Rectangle<Real>& extent,
             const Point<Real>& point) const {
    CATAMARI_ASSERT(i == 0 || i == 1, "Invalid choice of i basis index.");
    CATAMARI_ASSERT(j == 0 || j == 1, "Invalid choice of j basis index.");
    const Real scale =
        (extent.x_end - extent.x_beg) * (extent.y_end - extent.y_beg);
    if (i == 0 && j == 0) {
      return (extent.x_end - point.x) * (extent.y_end - point.y) / scale;
    } else if (i == 0 && j == 1) {
      return (extent.x_end - point.x) * (point.y - extent.y_beg) / scale;
    } else if (i == 1 && j == 0) {
      return (point.x - extent.x_beg) * (extent.y_end - point.y) / scale;
    } else {
      return (point.x - extent.x_beg) * (point.y - extent.y_beg) / scale;
    }
  }

  // Returns index 'k' of the gradient of \psi_{i, j} evaluated at the given
  // point.
  Real BasisGradient(int i, int j, int k, const Rectangle<Real>& extent,
                     const Point<Real>& point) const {
    CATAMARI_ASSERT(i == 0 || i == 1, "Invalid choice of i basis index.");
    CATAMARI_ASSERT(j == 0 || j == 1, "Invalid choice of j basis index.");
    CATAMARI_ASSERT(k == 0 || k == 1, "Invalid choice of gradient index.");
    const Real scale =
        (extent.x_end - extent.x_beg) * (extent.y_end - extent.y_beg);
    if (i == 0 && j == 0) {
      if (k == 0) {
        return -(extent.y_end - point.y) / scale;
      } else {
        return -(extent.x_end - point.x) / scale;
      }
    } else if (i == 0 && j == 1) {
      if (k == 0) {
        return -(point.y - extent.y_beg) / scale;
      } else {
        return (extent.x_end - point.x) / scale;
      }
    } else if (i == 1 && j == 0) {
      if (k == 0) {
        return (extent.y_end - point.y) / scale;
      } else {
        return -(point.x - extent.x_beg) / scale;
      }
    } else {
      if (k == 0) {
        return (point.y - extent.y_beg) / scale;
      } else {
        return (point.x - extent.x_beg) / scale;
      }
    }
  }

  // Returns the evaluation of the density of the bilinear form over an element
  // [x_beg, x_end] x [y_beg, y_end] at a point (x, y) with test function
  // v = \phi_{i_test, j_test} and trial function u = \psi_{i_trial, j_trial}.
  // That is,
  //
  //   a_density(u, v) = (grad v)' (A grad u) - s u conj(v).
  //
  Complex<Real> BilinearDensity(int i_test, int j_test, int i_trial,
                                int j_trial, const Rectangle<Real>& extent,
                                const Point<Real>& point) const {
    Complex<Real> result = 0;

    // Add in the (grad v)' (A grad u) contribution. Recall that A is diagonal.
    for (int k = 0; k <= 1; ++k) {
      const Real test_grad_entry =
          BasisGradient(i_test, j_test, k, extent, point);
      const Real trial_grad_entry =
          BasisGradient(i_trial, j_trial, k, extent, point);
      const Complex<Real> weight_entry = weight_tensor_(k, point);
      // We explicitly call 'Conjugate' even though the basis functions are
      // real.
      result += Conjugate(test_grad_entry) * (weight_entry * trial_grad_entry);
    }

    // Add in the -s u conj(v) contribution.
    const Real test_entry = Basis(i_test, j_test, extent, point);
    const Real trial_entry = Basis(i_trial, j_trial, extent, point);
    const Complex<Real> diagonal_shift = shift_function_(point);
    // Again, we explicitly call 'Conjugate' even though the basis functions
    // are real.
    result -= diagonal_shift * trial_entry * Conjugate(test_entry);

    return result;
  }

  // Use a tensor product of third-order Gaussian quadrature to integrate the
  // bilinear form over the quadrilateral element.
  Complex<Real> Bilinear(int i_test, int j_test, int i_trial, int j_trial,
                         const Rectangle<Real>& extent,
                         const RectangleQuadrature<Real>& quadrature) const {
    Complex<Real> result = 0;
    for (int i = 0; i < 3; ++i) {
      const Real& x = quadrature.x_points[i];
      const Real& x_weight = quadrature.x_weights[i];
      for (int j = 0; j < 3; ++j) {
        const Real& y = quadrature.y_points[j];
        const Real& y_weight = quadrature.y_weights[j];
        const Point<Real> point{x, y};
        result +=
            x_weight * y_weight *
            BilinearDensity(i_test, j_test, i_trial, j_trial, extent, point);
      }
    }
    return result;
  }

 private:
  // The diagonal symmetric tensor field A : Omega -> C^{2 x 2} in the
  // Helmholtz equation
  //
  //   -div (A grad u) - s u = f.
  //
  const DiagonalWeightTensor weight_tensor_;

  // The scalar function s : Omega -> C in
  //
  //   -div (A grad u) - s u = f.
  //
  const DiagonalShiftFunction shift_function_;
};

// Generates a Q4 discretization of the 2D Helmholtz equation over [0, 1]^2
// with inserted PML.
template <typename Real>
std::unique_ptr<catamari::CoordinateMatrix<Complex<Real>>> HelmholtzWithPML(
    const Real& omega, Int num_x_elements, Int num_y_elements,
    const Real& pml_scale, const Real& pml_exponent, Int num_pml_elements) {
  const Real h_x = Real{1} / num_x_elements;
  const Real h_y = Real{1} / num_y_elements;

  std::unique_ptr<catamari::CoordinateMatrix<Complex<Real>>> matrix(
      new catamari::CoordinateMatrix<Complex<Real>>);

  const Real x_pml_width = num_pml_elements * h_x;
  const Real y_pml_width = num_pml_elements * h_y;

  // The frequency-scaled horizontal PML profile function.
  const auto sigma_x = [&](const Real& x) {
    Real result;
    const Real first_pml_end = x_pml_width;
    const Real last_pml_beg = Real{1} - x_pml_width;
    if (x < first_pml_end) {
      const Real pml_rel_depth = (first_pml_end - x) / x_pml_width;
      result = pml_scale * std::pow(pml_rel_depth, pml_exponent);
    } else if (x > last_pml_beg) {
      const Real pml_rel_depth = (x - last_pml_beg) / x_pml_width;
      result = pml_scale * std::pow(pml_rel_depth, pml_exponent);
    } else {
      result = 0;
    }
    return result;
  };

  // The frequency-scaled vertical PML profile function.
  const auto sigma_y = [&](const Real& y) {
    Real result;
    const Real first_pml_end = y_pml_width;
    const Real last_pml_beg = Real{1} - y_pml_width;
    if (y < first_pml_end) {
      const Real pml_rel_depth = (first_pml_end - y) / y_pml_width;
      result = pml_scale * std::pow(pml_rel_depth, pml_exponent);
    } else if (y > last_pml_beg) {
      const Real pml_rel_depth = (y - last_pml_beg) / y_pml_width;
      result = pml_scale * std::pow(pml_rel_depth, pml_exponent);
    } else {
      result = 0;
    }
    return result;
  };

  // The differential from the horizontal tangent space to the horizontal PML
  // tangent space.
  const auto gamma_x = [&](const Real& x) {
    return Complex<Real>{Real{1}, sigma_x(x) / omega};
  };

  // The differential from the vertical tangent space to the vertical PML
  // tangent space.
  const auto gamma_y = [&](const Real& y) {
    return Complex<Real>{Real{1}, sigma_y(y) / omega};
  };

  // The product of the horizontal and vertical PML differentials.
  const auto gamma = [&](const Point<Real>& point) {
    return gamma_x(point.x) * gamma_y(point.y);
  };

  // The sound speed of the material.
  // TODO(Jack Poulson): Expose a family of material profiles.
  const bool checkerboard = false;
  const auto speed = [&](const Point<Real>& point) {
    if (checkerboard) {
      // We divide [0, 1]^2 into a 10 x 10 checker-board so that tile (i, j)
      // has the 'white' sound speed of 1 if i + j == 0 (mod 2) or 10 otherwise.
      const Real white_speed = 1;
      const Real black_speed = 10;
      const Int i = std::floor(10. * point.x);
      const Int j = std::floor(10. * point.y);
      const bool white = ((i + j) % 2) == 0;
      if (white) {
        return white_speed;
      } else {
        return black_speed;
      }
    } else {
      return Real{1};
    }
  };

  // The diagonal 'A' tensor in the equation:
  //
  //    -div (A grad u) - (omega / c)^2 gamma u = f.
  //
  std::function<Complex<Real>(int i, const Point<Real>& point)> weight_tensor =
      [&](int i, const Point<Real>& point) {
        Complex<Real> result;
        if (i == 0) {
          const Complex<Real> gamma_k = gamma_x(point.x);
          result = gamma(point) / (gamma_k * gamma_k);
        } else {
          const Complex<Real> gamma_k = gamma_y(point.y);
          result = gamma(point) / (gamma_k * gamma_k);
        }
        return result;
      };

  // The '(omega / c)^2 gamma' in the equation:
  //
  //    -div (A grad u) - (omega / c)^2 gamma u = f.
  //
  std::function<Complex<Real>(const Point<Real>& point)> diagonal_shift =
      [&](const Point<Real>& point) {
        const Real local_speed = speed(point);
        return std::pow(omega / local_speed, Real{2}) * gamma(point);
      };

  const HelmholtzWithPMLQ4Element<Real> element(weight_tensor, diagonal_shift);

  const Int num_rows = (num_x_elements + 1) * (num_y_elements + 1);
  matrix->Resize(num_rows, num_rows);
  matrix->ReserveEntryAdditions(16. * num_x_elements * num_y_elements);
  for (Int x_element = 0; x_element < num_x_elements; ++x_element) {
    const Real x_beg = x_element * h_x;
    const Real x_end = x_beg + h_x;
    for (Int y_element = 0; y_element < num_y_elements; ++y_element) {
      const Real y_beg = y_element * h_y;
      const Real y_end = y_beg + h_y;
      const Rectangle<Real> extent{x_beg, x_end, y_beg, y_end};
      const RectangleQuadrature<Real> quadrature(extent);

      // The index of the bottom-left entry of the quadrilateral.
      const Int offset = x_element + y_element * (num_x_elements + 1);

      // Iterate over all of the interactions in the element.
      for (Int i_test = 0; i_test <= 1; ++i_test) {
        for (Int j_test = 0; j_test <= 1; ++j_test) {
          const Int row = offset + i_test + j_test * (num_x_elements + 1);
          for (Int i_trial = 0; i_trial <= 1; ++i_trial) {
            for (Int j_trial = 0; j_trial <= 1; ++j_trial) {
              const Int column =
                  offset + i_trial + j_trial * (num_x_elements + 1);
              const Complex<Real> value = element.Bilinear(
                  i_test, j_test, i_trial, j_trial, extent, quadrature);
              matrix->QueueEntryAddition(row, column, value);
            }
          }
        }
      }
    }
  }
  matrix->FlushEntryQueues();

  return matrix;
}

template <typename Real>
ConstBlasMatrix<Complex<Real>> GenerateRightHandSide(
    Int num_x_elements, Int num_y_elements,
    std::vector<Complex<Real>>* buffer) {
  typedef Complex<Real> Field;
  const Int num_rows = (num_x_elements + 1) * (num_y_elements + 1);

  BlasMatrix<Field> right_hand_side;
  right_hand_side.height = num_rows;
  right_hand_side.width = 1;
  right_hand_side.leading_dim = std::max(num_rows, Int(1));
  buffer->clear();
  buffer->resize(right_hand_side.leading_dim * right_hand_side.width);
  right_hand_side.data = buffer->data();

  // Generate a point source in the center of the domain.
  std::fill(buffer->begin(), buffer->end(), Complex<Real>{0});
  const Int middle_row =
      (num_x_elements / 2) + (num_y_elements / 2) * (num_x_elements + 1);
  right_hand_side(middle_row, 0) = Complex<Real>{1};

  return right_hand_side.ToConst();
}

// A list of properties to measure from a sparse LDL factorization / solve.
struct Experiment {
  // The number of seconds it took to construct the FEM matrix.
  double construction_seconds = 0;

  // The number of (structural) nonzeros in the associated Cholesky factor.
  Int num_nonzeros = 0;

  // The number of floating-point operations required for a standard Cholesky
  // factorization using the returned ordering.
  double num_flops = 0;

  // The number of seconds that elapsed during the factorization.
  double factorization_seconds = 0;

  // The number of seconds that elapsed during the solve.
  double solve_seconds = 0;
};

// Pretty prints the Experiment structure.
void PrintExperiment(const Experiment& experiment) {
  std::cout << "  construction_seconds:  " << experiment.construction_seconds
            << "\n";
  std::cout << "  num_nonzeros:          " << experiment.num_nonzeros << "\n";
  std::cout << "  num_flops:             " << experiment.num_flops << "\n";
  std::cout << "  factorization_seconds: " << experiment.factorization_seconds
            << "\n";
  std::cout << "  solve_seconds:         " << experiment.solve_seconds << "\n";
  std::cout << std::endl;
}

// Returns the Frobenius norm of a complex vector.
// NOTE: Due to the direct accumulation of the squared norm, this algorithm is
// unstable. But it suffices for example purposes.
template <typename Real>
Real EuclideanNorm(const ConstBlasMatrix<catamari::Complex<Real>>& matrix) {
  Real squared_norm{0};
  const Int height = matrix.height;
  const Int width = matrix.width;
  for (Int j = 0; j < width; ++j) {
    for (Int i = 0; i < height; ++i) {
      squared_norm += std::norm(matrix(i, j));
    }
  }
  return std::sqrt(squared_norm);
}

template <typename Field>
BlasMatrix<Field> CopyMatrix(const ConstBlasMatrix<Field>& matrix,
                             std::vector<Field>* buffer) {
  BlasMatrix<Field> matrix_copy;
  matrix_copy.height = matrix.height;
  matrix_copy.width = matrix.width;
  matrix_copy.leading_dim = std::max(matrix.height, Int(1));
  buffer->clear();
  buffer->resize(matrix_copy.leading_dim * matrix_copy.width);
  matrix_copy.data = buffer->data();
  for (Int j = 0; j < matrix.width; ++j) {
    for (Int i = 0; i < matrix.height; ++i) {
      matrix_copy(i, j) = matrix(i, j);
    }
  }
  return matrix_copy;
}

// Returns the Experiment statistics for a single Matrix Market input matrix.
Experiment RunTest(const double& omega, Int num_x_elements, Int num_y_elements,
                   const double& pml_scale, const double& pml_exponent,
                   int num_pml_elements,
                   const catamari::LDLControl& ldl_control,
                   bool print_progress) {
  typedef Complex<double> Field;
  typedef catamari::ComplexBase<Field> Real;
  Experiment experiment;
  quotient::Timer timer;

  // Read the matrix from file.
  timer.Start();
  std::unique_ptr<catamari::CoordinateMatrix<Field>> matrix =
      HelmholtzWithPML<Real>(omega, num_x_elements, num_y_elements, pml_scale,
                             pml_exponent, num_pml_elements);
  experiment.construction_seconds = timer.Stop();
  if (!matrix) {
    return experiment;
  }
  const Int num_rows = matrix->NumRows();

  // Factor the matrix.
  if (print_progress) {
    std::cout << "  Running factorization..." << std::endl;
  }
  timer.Start();
  catamari::LDLFactorization<Field> ldl_factorization;
  const catamari::LDLResult result =
      catamari::LDL(*matrix, ldl_control, &ldl_factorization);
  experiment.factorization_seconds = timer.Stop();
  if (result.num_successful_pivots < num_rows) {
    std::cout << "  Failed factorization after " << result.num_successful_pivots
              << " pivots." << std::endl;
    return experiment;
  }
  experiment.num_nonzeros = result.num_factorization_entries;
  experiment.num_flops = result.num_factorization_flops;

  // Generate an arbitrary right-hand side.
  std::vector<Field> right_hand_side_buffer;
  const ConstBlasMatrix<Complex<Real>> right_hand_side =
      GenerateRightHandSide<Real>(num_x_elements, num_y_elements,
                                  &right_hand_side_buffer);
  const Real right_hand_side_norm = EuclideanNorm(right_hand_side);
  if (print_progress) {
    std::cout << "  || b ||_F = " << right_hand_side_norm << std::endl;
  }

  // Solve a random linear system.
  if (print_progress) {
    std::cout << "  Running solve..." << std::endl;
  }
  std::vector<Field> solution_buffer;
  BlasMatrix<Field> solution = CopyMatrix(right_hand_side, &solution_buffer);
  timer.Start();
  catamari::LDLSolve(ldl_factorization, &solution);
  experiment.solve_seconds = timer.Stop();

  // Print the solution.
  std::cout << "X: \n";
  for (Int row = 0; row < num_rows; ++row) {
    const Complex<Real> entry = solution(row, 0);
    std::cout << entry.real() << " + " << entry.imag() << "i\n";
  }
  std::cout << std::endl;

  // Compute the residual.
  std::vector<Field> residual_buffer;
  BlasMatrix<Field> residual = CopyMatrix(right_hand_side, &residual_buffer);
  catamari::ApplySparse(Field{-1}, *matrix, solution.ToConst(), Field{1},
                        &residual);
  const Real residual_norm = EuclideanNorm(residual.ToConst());
  std::cout << "  || B - A X ||_F / || B ||_F = "
            << residual_norm / right_hand_side_norm << std::endl;

  return experiment;
}

}  // anonymous namespace

int main(int argc, char** argv) {
  specify::ArgumentParser parser(argc, argv);
  const double omega = parser.OptionalInput<double>(
      "omega", "The angular frequency of the Helmholtz problem.", 120.);
  const Int num_x_elements = parser.OptionalInput<Int>(
      "num_x_elements", "The number of elements in the x direction.", 200);
  const Int num_y_elements = parser.OptionalInput<Int>(
      "num_y_elements", "The number of elements in the y direction.", 200);
  const double pml_scale = parser.OptionalInput<double>(
      "pml_scale", "The scaling factor of the PML profile.", 20.);
  const double pml_exponent = parser.OptionalInput<double>(
      "pml_exponent", "The exponent of the PML profile.", 3.);
  const Int num_pml_elements = parser.OptionalInput<Int>(
      "num_pml_elements", "The number of elements the PML should span.", 20);
  const int degree_type_int =
      parser.OptionalInput<int>("degree_type_int",
                                "The degree approximation type.\n"
                                "0:exact, 1:Amestoy, 2:Ashcraft, 3:Gilbert",
                                1);
  const bool aggressive_absorption = parser.OptionalInput<bool>(
      "aggressive_absorption", "Eliminate elements with aggressive absorption?",
      true);
  const Int min_dense_threshold = parser.OptionalInput<Int>(
      "min_dense_threshold",
      "Lower-bound on non-diagonal nonzeros for a row to be dense. The actual "
      "threshold will be: "
      "max(min_dense_threshold, dense_sqrt_multiple * sqrt(n))",
      16);
  const float dense_sqrt_multiple = parser.OptionalInput<float>(
      "dense_sqrt_multiple",
      "The multiplier on the square-root of the number of vertices for "
      "determining if a row is dense. The actual threshold will be: "
      "max(min_dense_threshold, dense_sqrt_multiple * sqrt(n))",
      10.f);
  const int supernodal_strategy_int =
      parser.OptionalInput<int>("supernodal_strategy_int",
                                "The SupernodalStrategy int.\n"
                                "0:scalar, 1:supernodal, 2:adaptive",
                                2);
  const bool relax_supernodes = parser.OptionalInput<bool>(
      "relax_supernodes", "Relax the supernodes?", true);
  const Int allowable_supernode_zeros =
      parser.OptionalInput<Int>("allowable_supernode_zeros",
                                "Number of zeros allowed in relaxations.", 128);
  const float allowable_supernode_zero_ratio = parser.OptionalInput<float>(
      "allowable_supernode_zero_ratio",
      "Ratio of explicit zeros allowed in a relaxed supernode.", 0.01f);
  const int ldl_algorithm_int =
      parser.OptionalInput<int>("ldl_algorithm_int",
                                "The LDL algorithm type.\n"
                                "0:left-looking, 1:up-looking",
                                1);
  const bool print_progress = parser.OptionalInput<bool>(
      "print_progress", "Print the progress of the experiments?", false);
#ifdef _OPENMP
  const int num_omp_threads = parser.OptionalInput<int>(
      "num_omp_threads",
      "The desired number of OpenMP threads. Uses default if <= 0.", 1);
#endif
  if (!parser.OK()) {
    return 0;
  }

#ifdef _OPENMP
  if (num_omp_threads > 0) {
    const int max_omp_threads = omp_get_max_threads();
    omp_set_num_threads(num_omp_threads);
    std::cout << "Will use " << num_omp_threads << " of " << max_omp_threads
              << " OpenMP threads." << std::endl;
  } else {
    std::cout << "Will use all " << omp_get_max_threads() << " OpenMP threads."
              << std::endl;
  }
#endif

  catamari::LDLControl ldl_control;
  ldl_control.md_control.degree_type =
      static_cast<quotient::DegreeType>(degree_type_int);
  ldl_control.md_control.aggressive_absorption = aggressive_absorption;
  ldl_control.md_control.min_dense_threshold = min_dense_threshold;
  ldl_control.md_control.dense_sqrt_multiple = dense_sqrt_multiple;
  ldl_control.supernodal_strategy =
      static_cast<catamari::SupernodalStrategy>(supernodal_strategy_int);
  ldl_control.scalar_control.factorization_type =
      catamari::kLDLTransposeFactorization;
  ldl_control.scalar_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);
  ldl_control.supernodal_control.factorization_type =
      catamari::kLDLTransposeFactorization;
  ldl_control.supernodal_control.relaxation_control.relax_supernodes =
      relax_supernodes;
  ldl_control.supernodal_control.relaxation_control.allowable_supernode_zeros =
      allowable_supernode_zeros;
  ldl_control.supernodal_control.relaxation_control
      .allowable_supernode_zero_ratio = allowable_supernode_zero_ratio;

  const Experiment experiment =
      RunTest(omega, num_x_elements, num_y_elements, pml_scale, pml_exponent,
              num_pml_elements, ldl_control, print_progress);
  PrintExperiment(experiment);

  return 0;
}
