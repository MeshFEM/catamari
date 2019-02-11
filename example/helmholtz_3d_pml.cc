/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// This driver is a simple implementation of a 3D Helmholtz equation in the
// unit box, [0, 1]^3, with Perfectly Matched Layer absorbing boundary
// conditions on all sides. The discretization is over boxes with trilinear,
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
using catamari::Complex;
using catamari::ComplexBase;
using catamari::Conjugate;
using catamari::ConstBlasMatrix;
using catamari::Int;

namespace {

// A point in the 3D domain (i.e., [0, 1]^3).
template <typename Real>
struct Point {
  Real x;
  Real y;
  Real z;
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
    const Real z_center = Real{1} / Real{2};
    const Real center_speed = 0.675;
    const Real max_speed = 1.325;
    const Real variance = 0.01;
    const Real dist_squared = (point.x - x_center) * (point.x - x_center) +
                              (point.y - y_center) * (point.y - y_center) +
                              (point.z - z_center) * (point.z - z_center);
    const Real gaussian_scale = max_speed - center_speed;
    return max_speed -
           gaussian_scale * std::exp(-dist_squared / (2 * variance));
  }

  Real WaveGuide(const Point<Real>& point) const {
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
// over [-1, 1]^3 are:
//
//   psi_{0, 0, 0}(x, y, z) = (1 - x) (1 - y) (1 - z) / 2^3,
//   psi_{0, 0, 1}(x, y, z) = (1 - x) (1 - y) (1 + z) / 2^3,
//   psi_{0, 1, 0}(x, y, z) = (1 - x) (1 + y) (1 - z) / 2^3,
//   psi_{0, 1, 1}(x, y, z) = (1 - x) (1 + y) (1 + z) / 2^3,
//   psi_{1, 0, 0}(x, y, z) = (1 + x) (1 - y) (1 - z) / 2^3,
//   psi_{1, 0, 1}(x, y, z) = (1 + x) (1 - y) (1 + z) / 2^3,
//   psi_{1, 1, 0}(x, y, z) = (1 + x) (1 + y) (1 - z) / 2^3.
//   psi_{1, 1, 1}(x, y, z) = (1 + x) (1 + y) (1 + z) / 2^3.
//
// Over an element [x_beg, x_end] x [y_beg, y_end] x [z_beg, z_end], where we
// denote the lengths by L_x, L_y, and L_z, and their product by
// V = L_x L_y L_z,
//
//   psi_{0, 0, 0}(x, y, z) = (x_end - x) (y_end - y) (z_end - z) / V,
//   psi_{0, 0, 1}(x, y, z) = (x_end - x) (y_end - y) (z - z_beg) / V,
//   psi_{0, 1, 0}(x, y, z) = (x_end - x) (y - y_beg) (z_end - z) / V,
//   psi_{0, 1, 1}(x, y, z) = (x_end - x) (y - y_beg) (z - z_beg) / V,
//   psi_{1, 0, 0}(x, y, z) = (x - x_beg) (y_end - y) (z_end - z) / V,
//   psi_{1, 0, 1}(x, y, z) = (x - x_beg) (y_end - y) (z - z_beg) / V,
//   psi_{1, 1, 0}(x, y, z) = (x - x_beg) (y - y_beg) (z_end - z) / V.
//   psi_{1, 1, 1}(x, y, z) = (x - x_beg) (y - y_beg) (z - z_beg) / V.
//
// More compactly, if we define:
//
//   psi_{x, 0} = (x_end - x) / L_x,  psi_{x, 1} = (x - x_beg) / L_x,
//   psi_{y, 0} = (y_end - y) / L_y,  psi_{y, 1} = (y - y_beg) / L_y,
//   psi_{z, 0} = (z_end - z) / L_z,  psi_{z, 1} = (z - z_beg) / L_z,
//
// then we have
//
//   psi_I = psi_{x, I_x} psi_{y, I_y} psi_{z, I_z},
//
// and
//
//   grad_l psi_I = (prod_{alpha != l} psi_{alpha, I_alpha}) grad_l psi_alpha.
//
template <class Real>
class HelmholtzWithPMLTrilinearHexahedra {
 public:
  // A representation of an arbitrary axis-aligned box,
  //
  //     [x_beg, x_end] x [y_beg, y_end] x [z_beg, z_end].
  //
  struct Box {
    Real x_beg;
    Real x_end;
    Real y_beg;
    Real y_end;
    Real z_beg;
    Real z_end;
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

  // Returns \psi_{i, j, k} evaluated at the given point.
  Real Basis(int i, int j, int k, const Box& extent,
             const Point<Real>& point) const {
    CATAMARI_ASSERT(i == 0 || i == 1, "Invalid choice of i basis index.");
    CATAMARI_ASSERT(j == 0 || j == 1, "Invalid choice of j basis index.");
    CATAMARI_ASSERT(k == 0 || k == 1, "Invalid choice of k basis index.");
    const Real volume = (extent.x_end - extent.x_beg) *
                        (extent.y_end - extent.y_beg) *
                        (extent.z_end - extent.z_beg);

    Real product = Real{1} / volume;

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

    if (k == 0) {
      product *= extent.z_end - point.z;
    } else {
      product *= point.z - extent.z_beg;
    }

    return product;
  }

  // Returns index 'l' of the gradient of psi_{i, j, k} evaluated at the given
  // point.
  Real BasisGradient(int i, int j, int k, int l, const Box& extent,
                     const Point<Real>& point) const {
    CATAMARI_ASSERT(i == 0 || i == 1, "Invalid choice of i basis index.");
    CATAMARI_ASSERT(j == 0 || j == 1, "Invalid choice of j basis index.");
    CATAMARI_ASSERT(k == 0 || k == 1, "Invalid choice of k basis index.");
    CATAMARI_ASSERT(l == 0 || l == 1 || l == 2,
                    "Invalid choice of gradient index.");
    const Real volume = (extent.x_end - extent.x_beg) *
                        (extent.y_end - extent.y_beg) *
                        (extent.z_end - extent.z_beg);

    Real product = Real{1} / volume;

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

    // Either multiply by the (rescaled) psi_{z, k} or its gradient.
    if (l == 2) {
      if (k == 0) {
        product *= -1;
      }
    } else {
      if (k == 0) {
        product *= extent.z_end - point.z;
      } else {
        product *= point.z - extent.z_beg;
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

  // The constructor for the trilinear hexahedron.
  HelmholtzWithPMLTrilinearHexahedra(Int num_x_elements, Int num_y_elements,
                                     Int num_z_elements, const Real& omega,
                                     const Real& pml_scale,
                                     const Real& pml_exponent,
                                     Int num_pml_elements,
                                     const Speed<Real>& speed)
      : num_x_elements_(num_x_elements),
        num_y_elements_(num_y_elements),
        num_z_elements_(num_z_elements),
        element_x_size_(Real{1} / num_x_elements),
        element_y_size_(Real{1} / num_y_elements),
        element_z_size_(Real{1} / num_z_elements),
        omega_(omega),
        speed_(speed) {
    const Int num_dimensions = 3;
    const Int quadrature_1d_order = 3;
    const Int num_quadrature_points =
        quadrature_1d_order * quadrature_1d_order * quadrature_1d_order;
    const Int num_basis_functions = 8;
    const Real x_pml_width = num_pml_elements * element_x_size_;
    const Real y_pml_width = num_pml_elements * element_y_size_;
    const Real z_pml_width = num_pml_elements * element_z_size_;

    // Exploit the fact that the elements are translations of each other to
    // precompute the basis and gradient evaluations.
    const Box extent{0, element_x_size_, 0, element_y_size_,
                     0, element_z_size_};

    ThirdOrderGaussianPointsAndWeights(extent.x_beg, extent.x_end,
                                       quadrature_x_points_,
                                       quadrature_x_weights_);
    ThirdOrderGaussianPointsAndWeights(extent.y_beg, extent.y_end,
                                       quadrature_y_points_,
                                       quadrature_y_weights_);
    ThirdOrderGaussianPointsAndWeights(extent.z_beg, extent.z_end,
                                       quadrature_z_points_,
                                       quadrature_z_weights_);

    // The differentials from the real axes to the PML Profile tangent spaces.
    const PMLDifferential gamma_x(omega, pml_scale, pml_exponent, x_pml_width);
    const PMLDifferential gamma_y(omega, pml_scale, pml_exponent, y_pml_width);
    const PMLDifferential gamma_z(omega, pml_scale, pml_exponent, z_pml_width);

    pml_x_points_.resize(num_x_elements * quadrature_1d_order);
    for (Int x_element = 0; x_element < num_x_elements; ++x_element) {
      const Int x_offset = x_element * quadrature_1d_order;
      const Real x_beg = x_element * element_x_size_;
      for (Int i = 0; i < quadrature_1d_order; ++i) {
        const Real& x_point = quadrature_x_points_[i];
        pml_x_points_[x_offset + i] = gamma_x(x_beg + x_point);
      }
    }

    pml_y_points_.resize(num_y_elements * quadrature_1d_order);
    for (Int y_element = 0; y_element < num_y_elements; ++y_element) {
      const Int y_offset = y_element * quadrature_1d_order;
      const Real y_beg = y_element * element_y_size_;
      for (Int i = 0; i < quadrature_1d_order; ++i) {
        const Real& y_point = quadrature_y_points_[i];
        pml_y_points_[y_offset + i] = gamma_y(y_beg + y_point);
      }
    }

    pml_z_points_.resize(num_z_elements * quadrature_1d_order);
    for (Int z_element = 0; z_element < num_z_elements; ++z_element) {
      const Int z_offset = z_element * quadrature_1d_order;
      const Real z_beg = z_element * element_z_size_;
      for (Int i = 0; i < quadrature_1d_order; ++i) {
        const Real& z_point = quadrature_z_points_[i];
        pml_z_points_[z_offset + i] = gamma_z(z_beg + z_point);
      }
    }

    // Store the quadrature weights over the tensor product grid.
    quadrature_weights_buffer_.resize(num_quadrature_points);
    quadrature_weights_.height = num_quadrature_points;
    quadrature_weights_.width = 1;
    quadrature_weights_.leading_dim = num_quadrature_points;
    quadrature_weights_.data = quadrature_weights_buffer_.data();
    for (int z_quad = 0; z_quad < quadrature_1d_order; ++z_quad) {
      const Real& z_weight = quadrature_z_weights_[z_quad];
      for (int y_quad = 0; y_quad < quadrature_1d_order; ++y_quad) {
        const Real& y_weight = quadrature_y_weights_[y_quad];
        for (int x_quad = 0; x_quad < quadrature_1d_order; ++x_quad) {
          const Real& x_weight = quadrature_x_weights_[x_quad];
          const int row = x_quad + y_quad * quadrature_1d_order +
                          z_quad * quadrature_1d_order * quadrature_1d_order;
          quadrature_weights_(row, 0) = x_weight * y_weight * z_weight;
        }
      }
    }

    // Store the evaluations of the basis functions.
    basis_evals_buffer_.resize(num_quadrature_points * num_basis_functions);
    basis_evals_.height = num_quadrature_points;
    basis_evals_.width = num_basis_functions;
    basis_evals_.leading_dim = num_quadrature_points;
    basis_evals_.data = basis_evals_buffer_.data();
    for (int z_quad = 0; z_quad < quadrature_1d_order; ++z_quad) {
      const Real& z_point = quadrature_z_points_[z_quad];
      for (int y_quad = 0; y_quad < quadrature_1d_order; ++y_quad) {
        const Real& y_point = quadrature_y_points_[y_quad];
        for (int x_quad = 0; x_quad < quadrature_1d_order; ++x_quad) {
          const Real& x_point = quadrature_x_points_[x_quad];
          const Point<Real> point{x_point, y_point, z_point};
          const int row = x_quad + y_quad * quadrature_1d_order +
                          z_quad * quadrature_1d_order * quadrature_1d_order;
          for (int k = 0; k <= 1; ++k) {
            for (int j = 0; j <= 1; ++j) {
              for (int i = 0; i <= 1; ++i) {
                const int column = i + j * 2 + k * 4;
                basis_evals_(row, column) = Basis(i, j, k, extent, point);
              }
            }
          }
        }
      }
    }

    // Store the evaluations of the basis function gradients.
    basis_grad_evals_buffer_.resize(num_quadrature_points *
                                    num_basis_functions * num_dimensions);
    basis_grad_evals_.height = num_quadrature_points;
    basis_grad_evals_.width = num_basis_functions * num_dimensions;
    basis_grad_evals_.leading_dim = num_quadrature_points;
    basis_grad_evals_.data = basis_grad_evals_buffer_.data();
    for (int z_quad = 0; z_quad < quadrature_1d_order; ++z_quad) {
      const Real& z_point = quadrature_z_points_[z_quad];
      for (int y_quad = 0; y_quad < quadrature_1d_order; ++y_quad) {
        const Real& y_point = quadrature_y_points_[y_quad];
        for (int x_quad = 0; x_quad < quadrature_1d_order; ++x_quad) {
          const Real& x_point = quadrature_x_points_[x_quad];
          const Point<Real> point{x_point, y_point, z_point};
          const int row = x_quad + y_quad * quadrature_1d_order +
                          z_quad * quadrature_1d_order * quadrature_1d_order;
          for (int l = 0; l < num_dimensions; ++l) {
            for (int k = 0; k <= 1; ++k) {
              for (int j = 0; j <= 1; ++j) {
                for (int i = 0; i <= 1; ++i) {
                  const int column =
                      i + j * 2 + k * 4 + l * num_basis_functions;
                  basis_grad_evals_(row, column) =
                      BasisGradient(i, j, k, l, extent, point);
                }
              }
            }
          }
        }
      }
    }

    // Initialize the weight tensor evaluation matrix.
    gradient_evals_buffer_.resize(num_quadrature_points * num_dimensions);
    gradient_evals_.height = num_quadrature_points;
    gradient_evals_.width = num_dimensions;
    gradient_evals_.leading_dim = num_quadrature_points;
    gradient_evals_.data = gradient_evals_buffer_.data();

    // Initialize the scalar evaluation vector.
    scalar_evals_buffer_.resize(num_quadrature_points);
    scalar_evals_.height = num_quadrature_points;
    scalar_evals_.width = 1;
    scalar_evals_.leading_dim = num_quadrature_points;
    scalar_evals_.data = scalar_evals_buffer_.data();
  }

  // Form all of the matrix updates for a particular element.
  void ElementBilinearForms(Int x_element, Int y_element, Int z_element,
                            BlasMatrix<Complex<Real>>* element_updates) const {
    const int quadrature_1d_order = 3;
    const int num_dimensions = 3;
    const int num_basis_functions = 8;
    const Int x_offset = x_element * quadrature_1d_order;
    const Int y_offset = y_element * quadrature_1d_order;
    const Int z_offset = z_element * quadrature_1d_order;
    const Real x_beg = x_element * element_x_size_;
    const Real y_beg = y_element * element_y_size_;
    const Real z_beg = z_element * element_z_size_;

    // Evaluate the weight tensor over the element.
    for (int l = 0; l < num_dimensions; ++l) {
      for (int k = 0; k < quadrature_1d_order; ++k) {
        const Complex<Real>& gamma_z = pml_z_points_[z_offset + k];
        for (int j = 0; j < quadrature_1d_order; ++j) {
          const Complex<Real>& gamma_y = pml_y_points_[y_offset + j];
          for (int i = 0; i < quadrature_1d_order; ++i) {
            const Complex<Real>& gamma_x = pml_x_points_[x_offset + i];
            const int quadrature_index =
                i + j * quadrature_1d_order +
                k * quadrature_1d_order * quadrature_1d_order;

            const Complex<Real> gamma_product = gamma_x * gamma_y * gamma_z;
            if (l == 0) {
              gradient_evals_(quadrature_index, l) =
                  gamma_product / (gamma_x * gamma_x);
            } else if (l == 1) {
              gradient_evals_(quadrature_index, l) =
                  gamma_product / (gamma_y * gamma_y);
            } else {
              gradient_evals_(quadrature_index, l) =
                  gamma_product / (gamma_z * gamma_z);
            }
          }
        }
      }
    }

    // Evaluate the diagonal shifts over the element.
    for (int k = 0; k < quadrature_1d_order; ++k) {
      const Real z = z_beg + quadrature_z_points_[k];
      const Complex<Real>& gamma_z = pml_z_points_[z_offset + k];
      for (int j = 0; j < quadrature_1d_order; ++j) {
        const Real y = y_beg + quadrature_y_points_[j];
        const Complex<Real>& gamma_y = pml_y_points_[y_offset + j];
        for (int i = 0; i < quadrature_1d_order; ++i) {
          const Real x = x_beg + quadrature_x_points_[i];
          const Complex<Real>& gamma_x = pml_x_points_[x_offset + i];

          const Point<Real> point{x, y, z};
          const Complex<Real> gamma_product = gamma_x * gamma_y * gamma_z;
          const int quadrature_index =
              i + j * quadrature_1d_order + k * quadrature_1d_order;

          const Real rel_omega = omega_ / speed_(point);
          scalar_evals_(quadrature_index, 0) =
              rel_omega * rel_omega * gamma_product;
        }
      }
    }

    // Compute the element updates.
    for (int k_test = 0; k_test <= 1; ++k_test) {
      for (int j_test = 0; j_test <= 1; ++j_test) {
        for (int i_test = 0; i_test <= 1; ++i_test) {
          const int element_row = i_test + j_test * 2 + k_test * 4;
          for (int k_trial = 0; k_trial <= 1; ++k_trial) {
            for (int j_trial = 0; j_trial <= 1; ++j_trial) {
              for (int i_trial = 0; i_trial <= 1; ++i_trial) {
                const int element_column = i_trial + j_trial * 2 + k_trial * 4;

                Complex<Real> result = 0;
                for (int k = 0; k < quadrature_1d_order; ++k) {
                  for (int j = 0; j < quadrature_1d_order; ++j) {
                    for (int i = 0; i < quadrature_1d_order; ++i) {
                      const int quadrature_index =
                          i + j * quadrature_1d_order +
                          k * quadrature_1d_order * quadrature_1d_order;
                      Complex<Real> update = 0;

                      // Add in the (grad v)' (A grad u) contribution. Recall
                      // that A is diagonal.
                      for (int l = 0; l < num_dimensions; ++l) {
                        const Real test_grad_entry = basis_grad_evals_(
                            quadrature_index,
                            element_row + num_basis_functions * l);
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
                          scalar_evals_(quadrature_index, 0);
                      update -=
                          diagonal_shift * trial_entry * Conjugate(test_entry);

                      result +=
                          quadrature_weights_(quadrature_index, 0) * update;
                    }
                  }
                }

                element_updates->Entry(element_row, element_column) = result;
              }
            }
          }
        }
      }
    }
  }

  // Form all of the right-hand side updates for a single element.
  template <class RightHandSideFunction>
  void ElementRightHandSide(Int x_element, Int y_element, Int z_element,
                            const RightHandSideFunction& rhs_function,
                            std::vector<Complex<Real>>* element_updates) const {
    const Real x_beg = x_element * element_x_size_;
    const Real y_beg = y_element * element_y_size_;
    const Real z_beg = z_element * element_z_size_;
    const int quadrature_1d_order = 3;
    const int num_basis_functions = 8;

    // Evaluate the right-hand side over the element.
    for (int k = 0; k < quadrature_1d_order; ++k) {
      const Real z = z_beg + quadrature_z_points_[k];
      for (int j = 0; j < quadrature_1d_order; ++j) {
        const Real y = y_beg + quadrature_y_points_[j];
        for (int i = 0; i < quadrature_1d_order; ++i) {
          const Real x = x_beg + quadrature_x_points_[i];
          const Point<Real> point{x, y, z};
          const int quadrature_index =
              i + j * quadrature_1d_order +
              k * quadrature_1d_order * quadrature_1d_order;
          scalar_evals_(quadrature_index, 0) = rhs_function(point);
        }
      }
    }

    // Compute the element updates.
    element_updates->resize(num_basis_functions);
    for (int k_test = 0; k_test <= 1; ++k_test) {
      for (int j_test = 0; j_test <= 1; ++j_test) {
        for (int i_test = 0; i_test <= 1; ++i_test) {
          const int element_row = i_test + j_test * 2 + k_test * 4;
          Complex<Real> result = 0;
          for (int k = 0; k < quadrature_1d_order; ++k) {
            for (int j = 0; j < quadrature_1d_order; ++j) {
              for (int i = 0; i < quadrature_1d_order; ++i) {
                const int quadrature_index =
                    i + j * quadrature_1d_order +
                    k * quadrature_1d_order * quadrature_1d_order;

                // Add in the f conj(v) contribution.
                // Again, we explicitly call 'Conjugate' even though the
                // basis functions are real.
                const Real test_entry =
                    basis_evals_(quadrature_index, element_row);
                const Complex<Real> rhs_value =
                    scalar_evals_(quadrature_index, 0);
                result += quadrature_weights_(quadrature_index, 0) * rhs_value *
                          Conjugate(test_entry);
              }
            }
          }
          (*element_updates)[element_row] = result;
        }
      }
    }
  }

 private:
  // The number of elements in the x direction.
  const Int num_x_elements_;

  // The number of elements in the y direction.
  const Int num_y_elements_;

  // The number of elements in the z direction.
  const Int num_z_elements_;

  // The x-length of each box element.
  const Real element_x_size_;

  // The y-length of each box element.
  const Real element_y_size_;

  // The z-length of each box element.
  const Real element_z_size_;

  // The angular frequency of the harmonic forcing function.
  const Real omega_;

  // The sound speed over the domain.
  const Speed<Real> speed_;

  // Evaluations of the PML profile over the quadrature points in the x
  // direction.
  std::vector<Complex<Real>> pml_x_points_;

  // Evaluations of the PML profile over the quadrature points in the y
  // direction.
  std::vector<Complex<Real>> pml_y_points_;

  // Evaluations of the PML profile over the quadrature points in the z
  // direction.
  std::vector<Complex<Real>> pml_z_points_;

  Real quadrature_x_points_[3];
  Real quadrature_y_points_[3];
  Real quadrature_z_points_[3];

  Real quadrature_x_weights_[3];
  Real quadrature_y_weights_[3];
  Real quadrature_z_weights_[3];

  std::vector<Real> quadrature_weights_buffer_;
  BlasMatrix<Real> quadrature_weights_;

  std::vector<Real> basis_evals_buffer_;
  BlasMatrix<Real> basis_evals_;

  std::vector<Real> basis_grad_evals_buffer_;
  BlasMatrix<Real> basis_grad_evals_;

  mutable std::vector<Complex<Real>> gradient_evals_buffer_;
  mutable BlasMatrix<Complex<Real>> gradient_evals_;

  mutable std::vector<Complex<Real>> scalar_evals_buffer_;
  mutable BlasMatrix<Complex<Real>> scalar_evals_;
};

// Generates a trilinear hexahedral discretization of the 3D Helmholtz equation
// over [0, 1]^3 with inserted PML.
template <typename Real>
void HelmholtzWithPML(SpeedProfile profile, const Real& omega,
                      Int num_x_elements, Int num_y_elements,
                      Int num_z_elements, const Real& pml_scale,
                      const Real& pml_exponent, Int num_pml_elements,
                      const Point<Real>& source_point,
                      const Real& source_stddev,
                      catamari::CoordinateMatrix<Complex<Real>>* matrix,
                      BlasMatrix<Complex<Real>>* right_hand_sides,
                      std::vector<Complex<Real>>* right_hand_side_buffer) {
  const Speed<Real> speed(profile);

  const HelmholtzWithPMLTrilinearHexahedra<Real> discretization(
      num_x_elements, num_y_elements, num_z_elements, omega, pml_scale,
      pml_exponent, num_pml_elements, speed);

  const Int num_element_members = 64;
  std::vector<Complex<Real>> element_update_buffer(num_element_members);
  BlasMatrix<Complex<Real>> element_updates;
  element_updates.height = 8;
  element_updates.width = 8;
  element_updates.leading_dim = 8;
  element_updates.data = element_update_buffer.data();

  const Int num_rows =
      (num_x_elements + 1) * (num_y_elements + 1) * (num_z_elements + 1);
  const Int y_stride = num_x_elements + 1;
  const Int z_stride = y_stride * (num_y_elements + 1);

  // Form the FEM matrix.
  matrix->Resize(num_rows, num_rows);
  const Int queue_upper_bound =
      num_element_members * num_x_elements * num_y_elements * num_z_elements;
  matrix->ReserveEntryAdditions(queue_upper_bound);
  for (Int z_element = 0; z_element < num_z_elements; ++z_element) {
    for (Int y_element = 0; y_element < num_y_elements; ++y_element) {
      for (Int x_element = 0; x_element < num_x_elements; ++x_element) {
        // Form the batch of updates.
        discretization.ElementBilinearForms(x_element, y_element, z_element,
                                            &element_updates);

        // Insert the updates into the matrix.
        const Int offset =
            x_element + y_element * y_stride + z_element * z_stride;

        for (int k_test = 0; k_test <= 1; ++k_test) {
          for (int j_test = 0; j_test <= 1; ++j_test) {
            for (int i_test = 0; i_test <= 1; ++i_test) {
              const int element_row = i_test + j_test * 2 + k_test * 4;
              const Int row =
                  offset + i_test + j_test * y_stride + k_test * z_stride;
              for (int k_trial = 0; k_trial <= 1; ++k_trial) {
                for (int j_trial = 0; j_trial <= 1; ++j_trial) {
                  for (int i_trial = 0; i_trial <= 1; ++i_trial) {
                    const int element_column =
                        i_trial + j_trial * 2 + k_trial * 4;
                    const Int column = offset + i_trial + j_trial * y_stride +
                                       k_trial * z_stride;
                    matrix->QueueEntryAddition(
                        row, column,
                        element_updates(element_row, element_column));
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  matrix->FlushEntryQueues();

  // Form the right-hand side.
  right_hand_side_buffer->clear();
  right_hand_side_buffer->resize(num_rows, Complex<Real>{0});
  right_hand_sides->height = num_rows;
  right_hand_sides->width = 1;
  right_hand_sides->leading_dim = num_rows;
  right_hand_sides->data = right_hand_side_buffer->data();

  const std::function<Complex<Real>(const Point<Real>&)> point_source =
      [&](const Point<Real>& point) {
        const Real scale = 1000;

        const Real variance = source_stddev * source_stddev;
        const Real x_diff = point.x - source_point.x;
        const Real y_diff = point.y - source_point.y;
        const Real z_diff = point.z - source_point.z;
        const Real dist_squared =
            x_diff * x_diff + y_diff * y_diff + z_diff * z_diff;
        const Real gaussian = scale * std::exp(-dist_squared / (2 * variance));
        return Complex<Real>(gaussian);
      };

  std::vector<Complex<Real>> element_right_hand_side(8);
  for (Int z_element = 0; z_element < num_z_elements; ++z_element) {
    for (Int y_element = 0; y_element < num_y_elements; ++y_element) {
      for (Int x_element = 0; x_element < num_x_elements; ++x_element) {
        // Form the batch of updates.
        discretization.ElementRightHandSide(x_element, y_element, z_element,
                                            point_source,
                                            &element_right_hand_side);

        // Insert the updates into the matrix.
        const Int offset =
            x_element + y_element * y_stride + z_element * z_stride;

        for (int k_test = 0; k_test <= 1; ++k_test) {
          for (int j_test = 0; j_test <= 1; ++j_test) {
            for (int i_test = 0; i_test <= 1; ++i_test) {
              const int element_row = i_test + j_test * 2 + k_test * 4;
              const Int row =
                  offset + i_test + j_test * y_stride + k_test * z_stride;
              right_hand_sides->Entry(row, 0) +=
                  element_right_hand_side[element_row];
            }
          }
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
Experiment RunTest(SpeedProfile profile, const double& omega,
                   Int num_x_elements, Int num_y_elements, Int num_z_elements,
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
  std::vector<Field> right_hand_side_buffer;
  BlasMatrix<Field> right_hand_sides;
  catamari::CoordinateMatrix<Field> matrix;
  HelmholtzWithPML<Real>(profile, omega, num_x_elements, num_y_elements,
                         num_z_elements, pml_scale, pml_exponent,
                         num_pml_elements, source_point, source_stddev, &matrix,
                         &right_hand_sides, &right_hand_side_buffer);
  experiment.construction_seconds = timer.Stop();
  const Int num_rows = matrix.NumRows();
  const Real right_hand_side_norm = EuclideanNorm(right_hand_sides.ToConst());
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
    catamari::UnitReachNestedDissection3D(num_x_elements, num_y_elements,
                                          num_z_elements, &ordering);
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
  std::vector<Field> solution_buffer;
  BlasMatrix<Field> solution =
      CopyMatrix(right_hand_sides.ToConst(), &solution_buffer);
  timer.Start();
  ldl_factorization.Solve(&solution);
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
  std::vector<Field> residual_buffer;
  BlasMatrix<Field> residual =
      CopyMatrix(right_hand_sides.ToConst(), &residual_buffer);
  catamari::ApplySparse(Field{-1}, matrix, solution.ToConst(), Field{1},
                        &residual);
  const Real residual_norm = EuclideanNorm(residual.ToConst());
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
      "omega", "The angular frequency of the Helmholtz problem.", 31.4);
  const Int num_x_elements = parser.OptionalInput<Int>(
      "num_x_elements", "The number of elements in the x direction.", 40);
  const Int num_y_elements = parser.OptionalInput<Int>(
      "num_y_elements", "The number of elements in the y direction.", 40);
  const Int num_z_elements = parser.OptionalInput<Int>(
      "num_z_elements", "The number of elements in the z direction.", 40);
  const double pml_scale = parser.OptionalInput<double>(
      "pml_scale", "The scaling factor of the PML profile.", 100.);
  const double pml_exponent = parser.OptionalInput<double>(
      "pml_exponent", "The exponent of the PML profile.", 3.);
  const Int num_pml_elements = parser.OptionalInput<Int>(
      "num_pml_elements", "The number of elements the PML should span.", 8);
  const double source_x = parser.OptionalInput<double>(
      "source_x", "The x location of the point source.", 0.5);
  const double source_y = parser.OptionalInput<double>(
      "source_y", "The y location of the point source.", 0.5);
  const double source_z = parser.OptionalInput<double>(
      "source_z", "The z location of the point source.", 0.5);
  const double source_stddev = parser.OptionalInput<double>(
      "source_stddev", "The standard deviation of the point source.", 1e-2);
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
  const Int block_size = parser.OptionalInput<Int>(
      "block_size", "The dense algorithmic block size.", 64);
#ifdef _OPENMP
  const Int factor_tile_size = parser.OptionalInput<Int>(
      "factor_tile_size", "The multithreaded factorization tile size.", 128);
  const Int outer_product_tile_size = parser.OptionalInput<Int>(
      "outer_product_tile_size", "The multithreaded outer-product tile size.",
      240);
  const Int merge_grain_size = parser.OptionalInput<Int>(
      "merge_grain_size", "The number of columns to merge at once.", 500);
  const Int sort_grain_size = parser.OptionalInput<Int>(
      "sort_grain_size", "The number of columns to sort at once.", 200);
#endif  // ifdef _OPENMP
  const bool print_progress = parser.OptionalInput<bool>(
      "print_progress", "Print the progress of the experiments?", false);
  if (!parser.OK()) {
    return 0;
  }

  const SpeedProfile profile = static_cast<SpeedProfile>(speed_profile_int);

  const Point<double> source_point{source_x, source_y, source_z};

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
  ldl_control.supernodal_control.block_size = block_size;
#ifdef _OPENMP
  ldl_control.supernodal_control.factor_tile_size = factor_tile_size;
  ldl_control.supernodal_control.outer_product_tile_size =
      outer_product_tile_size;
  ldl_control.supernodal_control.merge_grain_size = merge_grain_size;
  ldl_control.supernodal_control.sort_grain_size = sort_grain_size;
#endif  // ifdef _OPENMP
  ldl_control.supernodal_control.relaxation_control.relax_supernodes =
      relax_supernodes;
  ldl_control.supernodal_control.relaxation_control.allowable_supernode_zeros =
      allowable_supernode_zeros;
  ldl_control.supernodal_control.relaxation_control
      .allowable_supernode_zero_ratio = allowable_supernode_zero_ratio;

  const Experiment experiment =
      RunTest(profile, omega, num_x_elements, num_y_elements, num_z_elements,
              pml_scale, pml_exponent, num_pml_elements, source_point,
              source_stddev, analytical_ordering, ldl_control, print_progress);
  PrintExperiment(experiment);

  return 0;
}
