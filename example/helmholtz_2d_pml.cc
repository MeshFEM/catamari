/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// This driver is a simple implementation of a 2D Helmholtz equation in the
// unit box, [0, 1]^2, with Perfectly Matched Layer absorbing boundary
// conditions on all sides. The discretization is over rectangles with bilinear,
// Lagrangian basis functions based at the corner points. The bilinear form is
// integrated over each element using a simple tensor product of three-point 1D
// Gaussian quadratures.
//
// A good reference for the weak formulation of the Helmholtz equation with PML
// is:
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
#include "catamari/unit_reach_nested_dissection.hpp"
#include "quotient/minimum_degree.hpp"
#include "specify.hpp"

using catamari::BlasMatrix;
using catamari::Buffer;
using catamari::Complex;
using catamari::ComplexBase;
using catamari::Conjugate;
using catamari::ConstBlasMatrix;
using catamari::Int;

namespace {

// TODO(Jack Poulson): Move this into the official library.
template <typename Field>
struct Matrix {
  BlasMatrix<Field> blas_matrix;

  Buffer<Field> data;

  Matrix() {
    blas_matrix.height = 0;
    blas_matrix.width = 0;
    blas_matrix.leading_dim = 0;
    blas_matrix.data = nullptr;
  }

  Matrix(const Matrix<Field>& matrix) {
    const Int height = matrix.blas_matrix.height;
    const Int width = matrix.blas_matrix.width;

    data.Resize(height * width);
    blas_matrix.height = height;
    blas_matrix.width = width;
    blas_matrix.leading_dim = height;
    blas_matrix.data = data.Data();

    // Copy each individual column so that the leading dimension does not
    // impact the copy time.
    for (Int j = 0; j < width; ++j) {
      std::copy(&matrix(0, j), &matrix(height, j), &blas_matrix(0, j));
    }
  }

  Matrix<Field>& operator=(const Matrix<Field>& matrix) {
    if (this != &matrix) {
      const Int height = matrix.blas_matrix.height;
      const Int width = matrix.blas_matrix.width;

      data.Resize(height * width);
      blas_matrix.height = height;
      blas_matrix.width = width;
      blas_matrix.leading_dim = height;
      blas_matrix.data = data.Data();

      // Copy each individual column so that the leading dimension does not
      // impact the copy time.
      for (Int j = 0; j < width; ++j) {
        std::copy(&matrix(0, j), &matrix(height, j), &blas_matrix(0, j));
      }
    }
    return *this;
  }

  void Resize(const Int& height, const Int& width) {
    if (height == blas_matrix.height && width == blas_matrix.width) {
      return;
    }
    data.Resize(height * width);
    blas_matrix.height = height;
    blas_matrix.width = width;
    blas_matrix.leading_dim = height;  // TODO(Jack Poulson): Handle 0 case.
    blas_matrix.data = data.Data();
  }

  void Resize(const Int& height, const Int& width, const Field& value) {
    data.Resize(height * width, value);
    blas_matrix.height = height;
    blas_matrix.width = width;
    blas_matrix.leading_dim = height;  // TODO(Jack Poulson): Handle 0 case.
    blas_matrix.data = data.Data();
  }

  Field& operator()(Int row, Int column) { return blas_matrix(row, column); }

  const Field& operator()(Int row, Int column) const {
    return blas_matrix(row, column);
  }

  Field& Entry(Int row, Int column) { return blas_matrix(row, column); }

  const Field& Entry(Int row, Int column) const {
    return blas_matrix(row, column);
  }
};

// A point in the 2D domain (i.e., [0, 1]^2).
template <typename Real>
struct Point {
  Real x;
  Real y;
};

enum SpeedProfile {
  kFreeSpace,
  kConvergingLens,
  kWaveGuide,
};

// The pointwise acoustic speed.
template <typename Real>
class Speed {
 public:
  Speed(SpeedProfile profile) : profile_(profile) {}

  Real FreeSpace(const Point<Real>& point) const { return Real{1}; }

  Real ConvergingLens(const Point<Real>& point) const {
    const Real x_center = Real{1} / Real{2};
    const Real y_center = Real{1} / Real{2};
    const Real center_speed = 0.675;
    const Real max_speed = 1.325;
    const Real variance = 0.01;
    const Real dist_squared = (point.x - x_center) * (point.x - x_center) +
                              (point.y - y_center) * (point.y - y_center);
    const Real gaussian_scale = max_speed - center_speed;
    return max_speed -
           gaussian_scale * std::exp(-dist_squared / (2 * variance));
  }

  Real WaveGuide(const Point<Real>& point) const {
    const Real x_center = Real{1} / Real{2};
    const Real center_speed = 0.675;
    const Real max_speed = 1.325;
    const Real variance = 0.01;
    const Real dist_squared = (point.x - x_center) * (point.x - x_center);
    const Real gaussian_scale = max_speed - center_speed;
    return max_speed -
           gaussian_scale * std::exp(-dist_squared / (2 * variance));
  }

  Real operator()(const Point<Real>& point) const {
    if (profile_ == kFreeSpace) {
      return FreeSpace(point);
    } else if (profile_ == kConvergingLens) {
      return ConvergingLens(point);
    } else if (profile_ == kWaveGuide) {
      return WaveGuide(point);
    } else {
      return 1;
    }
  }

 private:
  const SpeedProfile profile_;
};

// This currently uses third-order Gaussian quadrature. The basis functions
// over [-1, 1]^2 are:
//
//   psi_{0, 0}(x, y) = (1 - x) (1 - y) / 2^2,
//   psi_{0, 1}(x, y) = (1 - x) (1 + y) / 2^2,
//   psi_{1, 0}(x, y) = (1 + x) (1 - y) / 2^2,
//   psi_{1, 1}(x, y) = (1 + x) (1 + y) / 2^2.
//
// Over an element [x_beg, x_end] x [y_beg, y_end], where we denote the lengths
// by L_x and L_y, and their product by A = L_x L_y,
//
//   psi_{0, 0}(x, y) = (x_end - x) (y_end - y) / A,
//   psi_{0, 1}(x, y) = (x_end - x) (y - y_beg) / A,
//   psi_{1, 0}(x, y) = (x - x_beg) (y_end - y) / A,
//   psi_{1, 1}(x, y) = (x - x_beg) (y - y_beg) / A.
//
// More compactly, if we define:
//
//   psi_{x, 0} = (x_end - x) / L_x,  psi_{x, 1} = (x - x_beg) / L_x,
//   psi_{y, 0} = (y_end - y) / L_y,  psi_{y, 1} = (y - y_beg) / L_y,
//
// then we have
//
//   psi_I = psi_{x, I_x} psi_{y, I_y},
//
// and
//
//   grad_l psi_I = (prod_{alpha != l} psi_{alpha, I_alpha}) grad_l psi_alpha.
//
template <class Real>
class HelmholtzWithPMLQ4 {
 public:
  // A representation of an arbitrary axis-aligned box,
  //
  //     [x_beg, x_end] x [y_beg, y_end].
  //
  struct Box {
    Real x_beg;
    Real x_end;
    Real y_beg;
    Real y_end;
  };

  // The differential mapping the trivial tangent space of the real line into
  // the tangent space of the complex-stretched domain.
  class PMLDifferential {
   public:
    PMLDifferential(const Real& omega, const Real& pml_scale,
                    const Real& pml_exponent, const Real& pml_width)
        : omega_(omega),
          pml_scale_(pml_scale),
          pml_exponent_(pml_exponent),
          pml_width_(pml_width) {}

    Complex<Real> operator()(const Real& x) const {
      Real profile;
      const Real first_pml_end = pml_width_;
      const Real last_pml_beg = 1 - pml_width_;
      if (x < first_pml_end) {
        const Real pml_rel_depth = (first_pml_end - x) / pml_width_;
        profile = pml_scale_ * std::pow(pml_rel_depth, pml_exponent_);
      } else if (x > last_pml_beg) {
        const Real pml_rel_depth = (x - last_pml_beg) / pml_width_;
        profile = pml_scale_ * std::pow(pml_rel_depth, pml_exponent_);
      } else {
        profile = 0;
      }

      return Complex<Real>(Real{1}, profile / omega_);
    }

   private:
    const Real omega_;
    const Real pml_scale_;
    const Real pml_exponent_;
    const Real pml_width_;
  };

  // Returns \psi_{i, j} evaluated at the given point.
  Real Basis(int i, int j, const Box& extent, const Point<Real>& point) const {
    CATAMARI_ASSERT(i == 0 || i == 1, "Invalid choice of i basis index.");
    CATAMARI_ASSERT(j == 0 || j == 1, "Invalid choice of j basis index.");
    const Real area =
        (extent.x_end - extent.x_beg) * (extent.y_end - extent.y_beg);

    Real product = Real{1} / area;

    if (i == 0) {
      product *= extent.x_end - point.x;
    } else {
      product *= point.x - extent.x_beg;
    }

    if (j == 0) {
      product *= extent.y_end - point.y;
    } else {
      product *= point.y - extent.y_beg;
    }

    return product;
  }

  // Returns index 'l' of the gradient of psi_{i, j} evaluated at the given
  // point.
  Real BasisGradient(int i, int j, int l, const Box& extent,
                     const Point<Real>& point) const {
    CATAMARI_ASSERT(i == 0 || i == 1, "Invalid choice of i basis index.");
    CATAMARI_ASSERT(j == 0 || j == 1, "Invalid choice of j basis index.");
    CATAMARI_ASSERT(l == 0 || l == 1, "Invalid choice of gradient index.");
    const Real area =
        (extent.x_end - extent.x_beg) * (extent.y_end - extent.y_beg);

    Real product = Real{1} / area;

    // Either multiply by the (rescaled) psi_{x, i} or its gradient.
    if (l == 0) {
      if (i == 0) {
        product *= -1;
      }
    } else {
      if (i == 0) {
        product *= extent.x_end - point.x;
      } else {
        product *= point.x - extent.x_beg;
      }
    }

    // Either multiply by the (rescaled) psi_{y, j} or its gradient.
    if (l == 1) {
      if (j == 0) {
        product *= -1;
      }
    } else {
      if (j == 0) {
        product *= extent.y_end - point.y;
      } else {
        product *= point.y - extent.y_beg;
      }
    }

    return product;
  }

  // Fills 'points' and 'weights' with the transformed evaluation points and
  // weights for the interval [a, b].
  static void ThirdOrderGaussianPointsAndWeights(const Real& a, const Real& b,
                                                 Real* points, Real* weights) {
    const Real scale = (b - a) / 2;

    static const Real orig_points[] = {-std::sqrt(Real{3}) / std::sqrt(Real{5}),
                                       Real{0},
                                       std::sqrt(Real{3}) / std::sqrt(Real{5})};

    static const Real orig_weights[] = {Real(5) / Real(9), Real(8) / Real(9),
                                        Real(5) / Real(9)};

    for (int i = 0; i < 3; ++i) {
      points[i] = a + scale * (1 + orig_points[i]);
      weights[i] = scale * orig_weights[i];
    }
  }

  // The constructor for the Q4 elements.
  HelmholtzWithPMLQ4(Int num_x_elements, Int num_y_elements, const Real& omega,
                     const Real& pml_scale, const Real& pml_exponent,
                     Int num_pml_elements, const Speed<Real>& speed)
      : num_x_elements_(num_x_elements),
        num_y_elements_(num_y_elements),
        element_x_size_(Real{1} / num_x_elements),
        element_y_size_(Real{1} / num_y_elements),
        omega_(omega),
        speed_(speed) {
    const Int num_dimensions = 2;
    const Int quadrature_1d_order = 3;
    const Int num_quadrature_points = quadrature_1d_order * quadrature_1d_order;
    const Int num_basis_functions = 4;
    const Real x_pml_width = num_pml_elements * element_x_size_;
    const Real y_pml_width = num_pml_elements * element_y_size_;

    // Exploit the fact that the elements are translations of each other to
    // precompute the basis and gradient evaluations.
    const Box extent{0, element_x_size_, 0, element_y_size_};

    ThirdOrderGaussianPointsAndWeights(extent.x_beg, extent.x_end,
                                       quadrature_x_points_,
                                       quadrature_x_weights_);
    ThirdOrderGaussianPointsAndWeights(extent.y_beg, extent.y_end,
                                       quadrature_y_points_,
                                       quadrature_y_weights_);

    // The differentials from the real axes to the PML Profile tangent spaces.
    const PMLDifferential gamma_x(omega, pml_scale, pml_exponent, x_pml_width);
    const PMLDifferential gamma_y(omega, pml_scale, pml_exponent, y_pml_width);

    pml_x_points_.Resize(num_x_elements * quadrature_1d_order);
    for (Int x_element = 0; x_element < num_x_elements; ++x_element) {
      const Int x_offset = x_element * quadrature_1d_order;
      const Real x_beg = x_element * element_x_size_;
      for (Int i = 0; i < quadrature_1d_order; ++i) {
        const Real& x_point = quadrature_x_points_[i];
        pml_x_points_[x_offset + i] = gamma_x(x_beg + x_point);
      }
    }

    pml_y_points_.Resize(num_y_elements * quadrature_1d_order);
    for (Int y_element = 0; y_element < num_y_elements; ++y_element) {
      const Int y_offset = y_element * quadrature_1d_order;
      const Real y_beg = y_element * element_y_size_;
      for (Int i = 0; i < quadrature_1d_order; ++i) {
        const Real& y_point = quadrature_y_points_[i];
        pml_y_points_[y_offset + i] = gamma_y(y_beg + y_point);
      }
    }

    // Store the quadrature weights over the tensor product grid.
    quadrature_weights_.Resize(num_quadrature_points);
    for (int y_quad = 0; y_quad < quadrature_1d_order; ++y_quad) {
      const Real& y_weight = quadrature_y_weights_[y_quad];
      for (int x_quad = 0; x_quad < quadrature_1d_order; ++x_quad) {
        const Real& x_weight = quadrature_x_weights_[x_quad];
        const int row = x_quad + y_quad * quadrature_1d_order;
        quadrature_weights_[row] = x_weight * y_weight;
      }
    }

    // Store the evaluations of the basis functions.
    basis_evals_.Resize(num_quadrature_points, num_basis_functions);
    for (int y_quad = 0; y_quad < quadrature_1d_order; ++y_quad) {
      const Real& y_point = quadrature_y_points_[y_quad];
      for (int x_quad = 0; x_quad < quadrature_1d_order; ++x_quad) {
        const Real& x_point = quadrature_x_points_[x_quad];
        const Point<Real> point{x_point, y_point};
        const int row = x_quad + y_quad * quadrature_1d_order;
        for (int j = 0; j <= 1; ++j) {
          for (int i = 0; i <= 1; ++i) {
            const int column = i + j * 2;
            basis_evals_(row, column) = Basis(i, j, extent, point);
          }
        }
      }
    }

    // Store the evaluations of the basis function gradients.
    basis_grad_evals_.Resize(num_quadrature_points,
                             num_basis_functions * num_dimensions);
    for (int y_quad = 0; y_quad < quadrature_1d_order; ++y_quad) {
      const Real& y_point = quadrature_y_points_[y_quad];
      for (int x_quad = 0; x_quad < quadrature_1d_order; ++x_quad) {
        const Real& x_point = quadrature_x_points_[x_quad];
        const Point<Real> point{x_point, y_point};
        const int row = x_quad + y_quad * quadrature_1d_order;
        for (int l = 0; l < num_dimensions; ++l) {
          for (int j = 0; j <= 1; ++j) {
            for (int i = 0; i <= 1; ++i) {
              const int column = i + j * 2 + l * num_basis_functions;
              basis_grad_evals_(row, column) =
                  BasisGradient(i, j, l, extent, point);
            }
          }
        }
      }
    }

    // Initialize the weight tensor evaluation matrix.
    gradient_evals_.Resize(num_quadrature_points, num_dimensions);

    // Initialize the diagonal shift evaluation vector.
    scalar_evals_.Resize(num_quadrature_points);
  }

  // Form all of the matrix updates for a particular element.
  void ElementBilinearForms(Int x_element, Int y_element,
                            BlasMatrix<Complex<Real>>* element_updates) const {
    const int quadrature_1d_order = 3;
    const int num_dimensions = 2;
    const int num_basis_functions = 4;
    const Int x_offset = x_element * quadrature_1d_order;
    const Int y_offset = y_element * quadrature_1d_order;
    const Real x_beg = x_element * element_x_size_;
    const Real y_beg = y_element * element_y_size_;

    // Evaluate the weight tensor over the element.
    for (int l = 0; l < num_dimensions; ++l) {
      for (int j = 0; j < quadrature_1d_order; ++j) {
        const Complex<Real>& gamma_y = pml_y_points_[y_offset + j];
        for (int i = 0; i < quadrature_1d_order; ++i) {
          const Complex<Real>& gamma_x = pml_x_points_[x_offset + i];
          const int quadrature_index = i + j * quadrature_1d_order;

          const Complex<Real> gamma_product = gamma_x * gamma_y;
          if (l == 0) {
            gradient_evals_(quadrature_index, l) =
                gamma_product / (gamma_x * gamma_x);
          } else if (l == 1) {
            gradient_evals_(quadrature_index, l) =
                gamma_product / (gamma_y * gamma_y);
          }
        }
      }
    }

    // Evaluate the diagonal shifts over the element.
    for (int j = 0; j < quadrature_1d_order; ++j) {
      const Real y = y_beg + quadrature_y_points_[j];
      const Complex<Real>& gamma_y = pml_y_points_[y_offset + j];
      for (int i = 0; i < quadrature_1d_order; ++i) {
        const Real x = x_beg + quadrature_x_points_[i];
        const Complex<Real>& gamma_x = pml_x_points_[x_offset + i];

        const Point<Real> point{x, y};
        const Complex<Real> gamma_product = gamma_x * gamma_y;
        const int quadrature_index = i + j * quadrature_1d_order;

        const Real rel_omega = omega_ / speed_(point);
        scalar_evals_[quadrature_index] = rel_omega * rel_omega * gamma_product;
      }
    }

    // Compute the element updates.
    // TODO(Jack Poulson): Add OpenMP task parallelism.
    for (int j_test = 0; j_test <= 1; ++j_test) {
      for (int i_test = 0; i_test <= 1; ++i_test) {
        const int element_row = i_test + j_test * 2;
        for (int j_trial = 0; j_trial <= 1; ++j_trial) {
          for (int i_trial = 0; i_trial <= 1; ++i_trial) {
            const int element_column = i_trial + j_trial * 2;

            Complex<Real> result = 0;
            for (int j = 0; j < quadrature_1d_order; ++j) {
              for (int i = 0; i < quadrature_1d_order; ++i) {
                const int quadrature_index = i + j * quadrature_1d_order;
                Complex<Real> update = 0;

                // Add in the (grad v)' (A grad u) contribution. Recall
                // that A is diagonal.
                for (int l = 0; l < num_dimensions; ++l) {
                  const Real test_grad_entry = basis_grad_evals_(
                      quadrature_index, element_row + num_basis_functions * l);
                  const Real trial_grad_entry = basis_grad_evals_(
                      quadrature_index,
                      element_column + num_basis_functions * l);
                  const Complex<Real> weight_entry =
                      gradient_evals_(quadrature_index, l);
                  // We explicitly call 'Conjugate' even though the basis
                  // functions are real.
                  update += Conjugate(test_grad_entry) *
                            (weight_entry * trial_grad_entry);
                }

                // Add in the -s u conj(v) contribution.
                // Again, we explicitly call 'Conjugate' even though the
                // basis functions are real.
                const Real test_entry =
                    basis_evals_(quadrature_index, element_row);
                const Real trial_entry =
                    basis_evals_(quadrature_index, element_column);
                const Complex<Real> diagonal_shift =
                    scalar_evals_[quadrature_index];
                update -= diagonal_shift * trial_entry * Conjugate(test_entry);

                result += quadrature_weights_[quadrature_index] * update;
              }
            }
            element_updates->Entry(element_row, element_column) = result;
          }
        }
      }
    }
  }

  // Form all of the right-hand side updates for a single element.
  template <class RightHandSideFunction>
  void ElementRightHandSide(Int x_element, Int y_element,
                            const RightHandSideFunction& rhs_function,
                            Buffer<Complex<Real>>* element_updates) const {
    const Real x_beg = x_element * element_x_size_;
    const Real y_beg = y_element * element_y_size_;
    const int quadrature_1d_order = 3;
    const int num_basis_functions = 4;

    // Evaluate the right-hand side over the element.
    for (int j = 0; j < quadrature_1d_order; ++j) {
      const Real y = y_beg + quadrature_y_points_[j];
      for (int i = 0; i < quadrature_1d_order; ++i) {
        const Real x = x_beg + quadrature_x_points_[i];
        const Point<Real> point{x, y};
        const int quadrature_index = i + j * quadrature_1d_order;
        scalar_evals_[quadrature_index] = rhs_function(point);
      }
    }

    // Compute the element updates.
    // TODO(Jack Poulson): Add OpenMP task parallelism.
    element_updates->Resize(num_basis_functions);
    for (int j_test = 0; j_test <= 1; ++j_test) {
      for (int i_test = 0; i_test <= 1; ++i_test) {
        const int element_row = i_test + j_test * 2;
        Complex<Real> result = 0;
        for (int j = 0; j < quadrature_1d_order; ++j) {
          for (int i = 0; i < quadrature_1d_order; ++i) {
            const int quadrature_index = i + j * quadrature_1d_order;

            // Add in the f conj(v) contribution.
            // Again, we explicitly call 'Conjugate' even though the
            // basis functions are real.
            const Real test_entry = basis_evals_(quadrature_index, element_row);
            const Complex<Real> rhs_value = scalar_evals_[quadrature_index];
            result += quadrature_weights_[quadrature_index] * rhs_value *
                      Conjugate(test_entry);
          }
        }

        (*element_updates)[element_row] = result;
      }
    }
  }

 private:
  // The number of elements in the x direction.
  const Int num_x_elements_;

  // The number of elements in the y direction.
  const Int num_y_elements_;

  // The x-length of each rectangular element.
  const Real element_x_size_;

  // The y-length of each rectangular element.
  const Real element_y_size_;

  // The angular frequency of the harmonic forcing function.
  const Real omega_;

  // The sound speed over the domain.
  const Speed<Real> speed_;

  // Evaluations of the PML profile over the quadrature points in the x
  // direction.
  Buffer<Complex<Real>> pml_x_points_;

  // Evaluations of the PML profile over the quadrature points in the y
  // direction.
  Buffer<Complex<Real>> pml_y_points_;

  // The locations of quadrature points in each of the two dimensions.
  Real quadrature_x_points_[3];
  Real quadrature_y_points_[3];

  // The weights of quadrature points in each of the two dimensions.
  Real quadrature_x_weights_[3];
  Real quadrature_y_weights_[3];

  Buffer<Real> quadrature_weights_;

  Matrix<Real> basis_evals_;

  Matrix<Real> basis_grad_evals_;

  mutable Matrix<Complex<Real>> gradient_evals_;

  mutable Buffer<Complex<Real>> scalar_evals_;
};

// Generates a Q4 discretization of the 2D Helmholtz equation over [0, 1]^2
// with inserted PML.
template <typename Real>
void HelmholtzWithPML(SpeedProfile profile, const Real& omega,
                      Int num_x_elements, Int num_y_elements,
                      const Real& pml_scale, const Real& pml_exponent,
                      Int num_pml_elements, const Point<Real>& source_point,
                      const Real& source_stddev,
                      catamari::CoordinateMatrix<Complex<Real>>* matrix,
                      Matrix<Complex<Real>>* right_hand_sides) {
  const Speed<Real> speed(profile);

  const HelmholtzWithPMLQ4<Real> discretization(num_x_elements, num_y_elements,
                                                omega, pml_scale, pml_exponent,
                                                num_pml_elements, speed);

  const Int num_element_members = 16;
  Buffer<Complex<Real>> element_update_buffer(num_element_members);
  BlasMatrix<Complex<Real>> element_updates;
  element_updates.height = 4;
  element_updates.width = 4;
  element_updates.leading_dim = 4;
  element_updates.data = element_update_buffer.Data();

  // TODO(Jack Poulson): Decide how to parallelize this formation.
  const Int num_rows = (num_x_elements + 1) * (num_y_elements + 1);
  const Int y_stride = num_x_elements + 1;
  matrix->Resize(num_rows, num_rows);
  const Int queue_upper_bound =
      num_element_members * num_x_elements * num_y_elements;
  matrix->ReserveEntryAdditions(queue_upper_bound);
  for (Int y_element = 0; y_element < num_y_elements; ++y_element) {
    for (Int x_element = 0; x_element < num_x_elements; ++x_element) {
      // Form the batch of updates.
      discretization.ElementBilinearForms(x_element, y_element,
                                          &element_updates);

      // Insert the updates into the matrix.
      const Int offset = x_element + y_element * y_stride;

      for (int j_test = 0; j_test <= 1; ++j_test) {
        for (int i_test = 0; i_test <= 1; ++i_test) {
          const int element_row = i_test + j_test * 2;
          const Int row = offset + i_test + j_test * y_stride;
          for (int j_trial = 0; j_trial <= 1; ++j_trial) {
            for (int i_trial = 0; i_trial <= 1; ++i_trial) {
              const int element_column = i_trial + j_trial * 2;
              const Int column = offset + i_trial + j_trial * y_stride;
              matrix->QueueEntryAddition(
                  row, column, element_updates(element_row, element_column));
            }
          }
        }
      }
    }
  }
  matrix->FlushEntryQueues();

  // Form the right-hand side.
  right_hand_sides->Resize(num_rows, 1, Complex<Real>{0});

  const std::function<Complex<Real>(const Point<Real>&)> point_source =
      [&](const Point<Real>& point) {
        const Real scale = 1000;
        const Real variance = source_stddev * source_stddev;

        const Real x_diff = point.x - source_point.x;
        const Real y_diff = point.y - source_point.y;
        const Real dist_squared = x_diff * x_diff + y_diff * y_diff;
        const Real gaussian = scale * std::exp(-dist_squared / (2 * variance));
        return Complex<Real>(gaussian);
      };

  Buffer<Complex<Real>> element_right_hand_side(4);
  for (Int y_element = 0; y_element < num_y_elements; ++y_element) {
    for (Int x_element = 0; x_element < num_x_elements; ++x_element) {
      // Form the batch of updates.
      discretization.ElementRightHandSide(x_element, y_element, point_source,
                                          &element_right_hand_side);

      // Insert the updates into the matrix.
      const Int offset = x_element + y_element * y_stride;

      for (int j_test = 0; j_test <= 1; ++j_test) {
        for (int i_test = 0; i_test <= 1; ++i_test) {
          const int element_row = i_test + j_test * 2;
          const Int row = offset + i_test + j_test * y_stride;
          right_hand_sides->Entry(row, 0) +=
              element_right_hand_side[element_row];
        }
      }
    }
  }
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

// Returns the Experiment statistics for a single Matrix Market input matrix.
Experiment RunTest(SpeedProfile profile, const double& omega,
                   Int num_x_elements, Int num_y_elements,
                   const double& pml_scale, const double& pml_exponent,
                   int num_pml_elements, const Point<double>& source_point,
                   const double& source_stddev, bool analytical_ordering,
                   const catamari::LDLControl& ldl_control,
                   bool print_progress) {
  typedef Complex<double> Field;
  typedef catamari::ComplexBase<Field> Real;
  Experiment experiment;
  quotient::Timer timer;

  // Construct the problem.
  timer.Start();
  Matrix<Field> right_hand_sides;
  catamari::CoordinateMatrix<Field> matrix;
  HelmholtzWithPML<Real>(profile, omega, num_x_elements, num_y_elements,
                         pml_scale, pml_exponent, num_pml_elements,
                         source_point, source_stddev, &matrix,
                         &right_hand_sides);
  experiment.construction_seconds = timer.Stop();
  const Int num_rows = matrix.NumRows();
  const Real right_hand_side_norm =
      EuclideanNorm(right_hand_sides.blas_matrix.ToConst());
  if (print_progress) {
    std::cout << "  || b ||_F = " << right_hand_side_norm << std::endl;
  }

  // Factor the matrix.
  if (print_progress) {
    std::cout << "  Running factorization..." << std::endl;
  }
  timer.Start();
  catamari::LDLFactorization<Field> ldl_factorization;
  catamari::LDLResult result;
  if (analytical_ordering) {
    catamari::SymmetricOrdering ordering;
    catamari::UnitReachNestedDissection2D(num_x_elements, num_y_elements,
                                          &ordering);
    result = ldl_factorization.Factor(matrix, ordering, ldl_control);
  } else {
    result = ldl_factorization.Factor(matrix, ldl_control);
  }
  experiment.factorization_seconds = timer.Stop();
  if (result.num_successful_pivots < num_rows) {
    std::cout << "  Failed factorization after " << result.num_successful_pivots
              << " pivots." << std::endl;
    return experiment;
  }
  experiment.num_nonzeros = result.num_factorization_entries;
  experiment.num_flops = result.num_factorization_flops;

  // Solve a random linear system.
  if (print_progress) {
    std::cout << "  Running solve..." << std::endl;
  }
  Matrix<Field> solution = right_hand_sides;
  timer.Start();
  ldl_factorization.Solve(&solution.blas_matrix);
  experiment.solve_seconds = timer.Stop();

  if (print_progress) {
    // Print the solution.
    std::cout << "X: \n";
    for (Int row = 0; row < num_rows; ++row) {
      const Complex<Real> entry = solution(row, 0);
      std::cout << entry.real() << " + " << entry.imag() << "i\n";
    }
    std::cout << std::endl;
  }

  // Compute the residual.
  Matrix<Field> residual = right_hand_sides;
  catamari::ApplySparse(Field{-1}, matrix, solution.blas_matrix.ToConst(),
                        Field{1}, &residual.blas_matrix);
  const Real residual_norm = EuclideanNorm(residual.blas_matrix.ToConst());
  std::cout << "  || B - A X ||_F / || B ||_F = "
            << residual_norm / right_hand_side_norm << std::endl;

  return experiment;
}

}  // anonymous namespace

int main(int argc, char** argv) {
  specify::ArgumentParser parser(argc, argv);
  const int speed_profile_int =
      parser.OptionalInput<int>("speed_profile_int",
                                "The sound speed model to use:\n"
                                "0:free space, 1:converging lens, 2:wave guide",
                                1);
  const double omega = parser.OptionalInput<double>(
      "omega", "The angular frequency of the Helmholtz problem.", 471.2);
  const Int num_x_elements = parser.OptionalInput<Int>(
      "num_x_elements", "The number of elements in the x direction.", 600);
  const Int num_y_elements = parser.OptionalInput<Int>(
      "num_y_elements", "The number of elements in the y direction.", 600);
  const double pml_scale = parser.OptionalInput<double>(
      "pml_scale", "The scaling factor of the PML profile.", 500.);
  const double pml_exponent = parser.OptionalInput<double>(
      "pml_exponent", "The exponent of the PML profile.", 3.);
  const Int num_pml_elements = parser.OptionalInput<Int>(
      "num_pml_elements", "The number of elements the PML should span.", 10);
  const double source_x = parser.OptionalInput<double>(
      "source_x", "The x location of the point source.", 0.5);
  const double source_y = parser.OptionalInput<double>(
      "source_y", "The y location of the point source.", 0.125);
  const double source_stddev = parser.OptionalInput<double>(
      "source_stddev", "The standard deviation of the point source.", 1e-3);
  const bool analytical_ordering = parser.OptionalInput<bool>(
      "analytical_ordering", "Use an analytical reordering?", true);
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
                                "0:left-looking, 1:up-looking, 2:right-looking",
                                2);
  const bool print_progress = parser.OptionalInput<bool>(
      "print_progress", "Print the progress of the experiments?", false);
  if (!parser.OK()) {
    return 0;
  }

  const SpeedProfile profile = static_cast<SpeedProfile>(speed_profile_int);

  const Point<double> source_point{source_x, source_y};

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
  ldl_control.supernodal_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);
  ldl_control.supernodal_control.relaxation_control.relax_supernodes =
      relax_supernodes;
  ldl_control.supernodal_control.relaxation_control.allowable_supernode_zeros =
      allowable_supernode_zeros;
  ldl_control.supernodal_control.relaxation_control
      .allowable_supernode_zero_ratio = allowable_supernode_zero_ratio;

  const Experiment experiment =
      RunTest(profile, omega, num_x_elements, num_y_elements, pml_scale,
              pml_exponent, num_pml_elements, source_point, source_stddev,
              analytical_ordering, ldl_control, print_progress);
  PrintExperiment(experiment);

  return 0;
}
