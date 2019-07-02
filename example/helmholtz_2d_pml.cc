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
#include <functional>
#include <iostream>

#include "catamari/apply_sparse.hpp"
#include "catamari/blas_matrix.hpp"
#include "catamari/norms.hpp"
#include "catamari/sparse_ldl.hpp"
#include "catamari/unit_reach_nested_dissection.hpp"
#include "specify.hpp"

using catamari::BlasMatrix;
using catamari::BlasMatrixView;
using catamari::Buffer;
using catamari::Complex;
using catamari::ComplexBase;
using catamari::Conjugate;
using catamari::ConstBlasMatrixView;
using catamari::Int;

namespace {

// A point in the 2D domain.
template <typename Real>
struct Point {
  Real x;
  Real y;
};

// A Gaussian point source in the 2D domain.
template <typename Real>
struct GaussianSource {
  Point<Real> point;
  Real scale;
  Real stddev;
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

  // The dimension of the domain.
  constexpr static Int kDimension = 2;

  // The number of quadrature points per dimension.
  constexpr static Int kQuadratureOrder = 3;

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
    const Int num_quadrature_points = kQuadratureOrder * kQuadratureOrder;
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

    pml_x_points_.Resize(num_x_elements * kQuadratureOrder);
    for (Int x_element = 0; x_element < num_x_elements; ++x_element) {
      const Int x_offset = x_element * kQuadratureOrder;
      const Real x_beg = x_element * element_x_size_;
      for (Int i = 0; i < kQuadratureOrder; ++i) {
        const Real& x_point = quadrature_x_points_[i];
        pml_x_points_[x_offset + i] = gamma_x(x_beg + x_point);
      }
    }

    pml_y_points_.Resize(num_y_elements * kQuadratureOrder);
    for (Int y_element = 0; y_element < num_y_elements; ++y_element) {
      const Int y_offset = y_element * kQuadratureOrder;
      const Real y_beg = y_element * element_y_size_;
      for (Int i = 0; i < kQuadratureOrder; ++i) {
        const Real& y_point = quadrature_y_points_[i];
        pml_y_points_[y_offset + i] = gamma_y(y_beg + y_point);
      }
    }

    // Store the quadrature weights over the tensor product grid.
    quadrature_weights_.Resize(num_quadrature_points);
    for (int y_quad = 0; y_quad < kQuadratureOrder; ++y_quad) {
      const Real& y_weight = quadrature_y_weights_[y_quad];
      for (int x_quad = 0; x_quad < kQuadratureOrder; ++x_quad) {
        const Real& x_weight = quadrature_x_weights_[x_quad];
        const int row = x_quad + y_quad * kQuadratureOrder;
        quadrature_weights_[row] = x_weight * y_weight;
      }
    }

    // Store the evaluations of the basis functions.
    basis_evals_.Resize(num_quadrature_points, num_basis_functions);
    for (int y_quad = 0; y_quad < kQuadratureOrder; ++y_quad) {
      const Real& y_point = quadrature_y_points_[y_quad];
      for (int x_quad = 0; x_quad < kQuadratureOrder; ++x_quad) {
        const Real& x_point = quadrature_x_points_[x_quad];
        const Point<Real> point{x_point, y_point};
        const int row = x_quad + y_quad * kQuadratureOrder;
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
                             num_basis_functions * kDimension);
    for (int y_quad = 0; y_quad < kQuadratureOrder; ++y_quad) {
      const Real& y_point = quadrature_y_points_[y_quad];
      for (int x_quad = 0; x_quad < kQuadratureOrder; ++x_quad) {
        const Real& x_point = quadrature_x_points_[x_quad];
        const Point<Real> point{x_point, y_point};
        const int row = x_quad + y_quad * kQuadratureOrder;
        for (int l = 0; l < kDimension; ++l) {
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
    gradient_evals_.Resize(num_quadrature_points, kDimension);

    // Initialize the diagonal shift evaluation vector.
    scalar_evals_.Resize(num_quadrature_points);
  }

  // Form all of the matrix updates for a particular element.
  void ElementBilinearForms(
      Int x_element, Int y_element,
      BlasMatrixView<Complex<Real>>* element_updates) const {
    const int num_basis_functions = 4;
    const Int x_offset = x_element * kQuadratureOrder;
    const Int y_offset = y_element * kQuadratureOrder;
    const Real x_beg = x_element * element_x_size_;
    const Real y_beg = y_element * element_y_size_;

    // Evaluate the weight tensor over the element.
    for (int l = 0; l < kDimension; ++l) {
      for (int j = 0; j < kQuadratureOrder; ++j) {
        const Complex<Real>& gamma_y = pml_y_points_[y_offset + j];
        for (int i = 0; i < kQuadratureOrder; ++i) {
          const Complex<Real>& gamma_x = pml_x_points_[x_offset + i];
          const int quadrature_index = i + j * kQuadratureOrder;

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
    for (int j = 0; j < kQuadratureOrder; ++j) {
      const Real y = y_beg + quadrature_y_points_[j];
      const Complex<Real>& gamma_y = pml_y_points_[y_offset + j];
      for (int i = 0; i < kQuadratureOrder; ++i) {
        const Real x = x_beg + quadrature_x_points_[i];
        const Complex<Real>& gamma_x = pml_x_points_[x_offset + i];

        const Point<Real> point{x, y};
        const Complex<Real> gamma_product = gamma_x * gamma_y;
        const int quadrature_index = i + j * kQuadratureOrder;

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
            for (int j = 0; j < kQuadratureOrder; ++j) {
              for (int i = 0; i < kQuadratureOrder; ++i) {
                const int quadrature_index = i + j * kQuadratureOrder;
                Complex<Real> update = 0;

                // Add in the (grad v)' (A grad u) contribution. Recall
                // that A is diagonal.
                for (int l = 0; l < kDimension; ++l) {
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
    const int num_basis_functions = 4;

    // Evaluate the right-hand side over the element.
    for (int j = 0; j < kQuadratureOrder; ++j) {
      const Real y = y_beg + quadrature_y_points_[j];
      for (int i = 0; i < kQuadratureOrder; ++i) {
        const Real x = x_beg + quadrature_x_points_[i];
        const Point<Real> point{x, y};
        const int quadrature_index = i + j * kQuadratureOrder;
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
        for (int j = 0; j < kQuadratureOrder; ++j) {
          for (int i = 0; i < kQuadratureOrder; ++i) {
            const int quadrature_index = i + j * kQuadratureOrder;

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
  Real quadrature_x_points_[kQuadratureOrder];
  Real quadrature_y_points_[kQuadratureOrder];

  // The weights of quadrature points in each of the two dimensions.
  Real quadrature_x_weights_[kQuadratureOrder];
  Real quadrature_y_weights_[kQuadratureOrder];

  Buffer<Real> quadrature_weights_;

  BlasMatrix<Real> basis_evals_;

  BlasMatrix<Real> basis_grad_evals_;

  mutable BlasMatrix<Complex<Real>> gradient_evals_;

  mutable Buffer<Complex<Real>> scalar_evals_;
};

template <class Real>
constexpr Int HelmholtzWithPMLQ4<Real>::kDimension;

template <class Real>
constexpr Int HelmholtzWithPMLQ4<Real>::kQuadratureOrder;

// Generates a Q4 discretization of the 2D Helmholtz equation over [0, 1]^2
// with inserted PML.
template <typename Real>
void HelmholtzWithPML(SpeedProfile profile, const Real& omega,
                      Int num_x_elements, Int num_y_elements,
                      const Real& pml_scale, const Real& pml_exponent,
                      Int num_pml_elements,
                      const Buffer<GaussianSource<Real>>& sources,
                      catamari::CoordinateMatrix<Complex<Real>>* matrix,
                      BlasMatrix<Complex<Real>>* right_hand_sides) {
  const Speed<Real> speed(profile);

  const HelmholtzWithPMLQ4<Real> discretization(num_x_elements, num_y_elements,
                                                omega, pml_scale, pml_exponent,
                                                num_pml_elements, speed);

  const Int num_element_members = 16;
  BlasMatrix<Complex<Real>> element_updates;
  element_updates.Resize(4, 4);

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
                                          &element_updates.view);

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

  // Form the right-hand sides.
  const Int num_sources = sources.Size();
  right_hand_sides->Resize(num_rows, num_sources, Complex<Real>{0});
  Buffer<Complex<Real>> element_right_hand_side(4);
  for (Int s = 0; s < num_sources; ++s) {
    const GaussianSource<Real>& source = sources[s];
    const std::function<Complex<Real>(const Point<Real>&)> point_source =
        [&](const Point<Real>& point) {
          const Real variance = source.stddev * source.stddev;
          const Real x_diff = point.x - source.point.x;
          const Real y_diff = point.y - source.point.y;
          const Real dist_squared = x_diff * x_diff + y_diff * y_diff;
          const Real gaussian =
              source.scale * std::exp(-dist_squared / (2 * variance));
          return Complex<Real>(gaussian);
        };

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
            right_hand_sides->Entry(row, s) +=
                element_right_hand_side[element_row];
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

  // The rough number of floating-point operations required to factor the
  // supernodal diagonal blocks.
  double num_diagonal_flops = 0;

  // The rough number of floating-point operations required to solve against the
  // diagonal blocks to update the subdiagonals.
  double num_subdiag_solve_flops = 0;

  // The rough number of floating-point operations required to form the Schur
  // complements.
  double num_schur_complement_flops = 0;

  // The number of floating-point operations required for a standard Cholesky
  // factorization using the returned ordering.
  double num_flops = 0;

  // The number of seconds that elapsed during the factorization.
  double factorization_seconds = 0;

  // The number of seconds that elapsed during the solve.
  double solve_seconds = 0;

  // The number of seconds that elapsed during the refined solve.
  double refined_solve_seconds = 0;
};

// Pretty prints the Experiment structure.
void PrintExperiment(const Experiment& experiment) {
  const double factorization_gflops_per_sec =
      experiment.num_flops / (1.e9 * experiment.factorization_seconds);

  std::cout << "  construction_seconds:       "
            << experiment.construction_seconds << "\n"
            << "  num_nonzeros:               " << experiment.num_nonzeros
            << "\n"
            << "  num_diagonal_flops:         " << experiment.num_diagonal_flops
            << "\n"
            << "  num_subdiag_solve_flops:    "
            << experiment.num_subdiag_solve_flops << "\n"
            << "  num_schur_complement_flops: "
            << experiment.num_schur_complement_flops << "\n"
            << "  num_flops:                  " << experiment.num_flops << "\n"
            << "  factorization_seconds:      "
            << experiment.factorization_seconds << "\n"
            << "  factorization gflops/sec:   " << factorization_gflops_per_sec
            << "\n"
            << "  solve_seconds:              " << experiment.solve_seconds
            << "\n"
            << "  refined_solve_seconds:      "
            << experiment.refined_solve_seconds << "\n"
            << std::endl;
}

template <typename Real>
Experiment SolveModel(SpeedProfile profile, Real omega, Int num_x_elements,
                      Int num_y_elements, Real pml_scale, Real pml_exponent,
                      int num_pml_elements,
                      const Buffer<GaussianSource<double>>& double_sources,
                      bool analytical_ordering,
                      const catamari::SparseLDLControl& ldl_control,
                      bool print_progress) {
  typedef Complex<Real> Field;
  Experiment experiment;
  quotient::Timer timer;

  Buffer<GaussianSource<Real>> sources(double_sources.Size());
  for (unsigned index = 0; index < sources.Size(); ++index) {
    sources[index].point.x = double_sources[index].point.x;
    sources[index].point.y = double_sources[index].point.y;
    sources[index].scale = double_sources[index].scale;
    sources[index].stddev = double_sources[index].stddev;
  }

  // Construct the problem.
  timer.Start();
  BlasMatrix<Field> right_hand_sides;
  catamari::CoordinateMatrix<Field> matrix;
  HelmholtzWithPML<Real>(profile, omega, num_x_elements, num_y_elements,
                         pml_scale, pml_exponent, num_pml_elements, sources,
                         &matrix, &right_hand_sides);
  experiment.construction_seconds = timer.Stop();
  const Int num_rows = matrix.NumRows();
  const Real right_hand_side_norm =
      catamari::EuclideanNorm(right_hand_sides.ConstView());
  if (print_progress) {
    std::cout << "  || B ||_F = " << right_hand_side_norm << std::endl;
  }

  // Factor the matrix.
  if (print_progress) {
    std::cout << "  Running factorization..." << std::endl;
  }
  timer.Start();
  catamari::SparseLDL<Field> ldl;
  catamari::SparseLDLResult result;
  if (analytical_ordering) {
    catamari::SymmetricOrdering ordering;
    catamari::UnitReachNestedDissection2D(num_x_elements, num_y_elements,
                                          &ordering);
    result = ldl.Factor(matrix, ordering, ldl_control);
  } else {
    result = ldl.Factor(matrix, ldl_control);
  }
  experiment.factorization_seconds = timer.Stop();
  if (result.num_successful_pivots < num_rows) {
    std::cout << "  Failed factorization after " << result.num_successful_pivots
              << " pivots." << std::endl;
    return experiment;
  }
  experiment.num_nonzeros = result.num_factorization_entries;
  experiment.num_diagonal_flops = result.num_diagonal_flops;
  experiment.num_subdiag_solve_flops = result.num_subdiag_solve_flops;
  experiment.num_schur_complement_flops = result.num_schur_complement_flops;
  experiment.num_flops = result.num_factorization_flops;

  // Solve the linear systems.
  {
    if (print_progress) {
      std::cout << "  Running solve..." << std::endl;
    }
    BlasMatrix<Field> solution = right_hand_sides;
    timer.Start();
    ldl.Solve(&solution.view);
    experiment.solve_seconds = timer.Stop();
    if (print_progress) {
      catamari::Print(solution, "X", std::cout);
    }

    // Compute the residual.
    BlasMatrix<Field> residual = right_hand_sides;
    catamari::ApplySparse(Field{-1}, matrix, solution.ConstView(), Field{1},
                          &residual.view);
    const Real residual_norm = catamari::EuclideanNorm(residual.ConstView());
    std::cout << "  || B - A X ||_F / || B ||_F = "
              << residual_norm / right_hand_side_norm << std::endl;
  }

  // Solve the linear system with iterative refinement.
  {
    // TODO(Jack Poulson): Make these parameters configurable.
    catamari::RefinedSolveControl<Real> refined_solve_control;
    refined_solve_control.verbose = true;

    if (print_progress) {
      std::cout << "  Running iteratively-refined solve..." << std::endl;
    }
    BlasMatrix<Field> solution = right_hand_sides;
    timer.Start();
    ldl.RefinedSolve(matrix, refined_solve_control, &solution.view);
    experiment.refined_solve_seconds = timer.Stop();
    if (print_progress) {
      catamari::Print(solution, "XRefined", std::cout);
    }

    // Compute the residual.
    BlasMatrix<Field> residual = right_hand_sides;
    catamari::ApplySparse(Field{-1}, matrix, solution.ConstView(), Field{1},
                          &residual.view);
    const Real residual_norm = catamari::EuclideanNorm(residual.ConstView());
    std::cout << "  Refined || B - A X ||_F / || B ||_F = "
              << residual_norm / right_hand_side_norm << std::endl;
  }
  PrintExperiment(experiment);

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
      "num_x_elements", "The number of elements in the x direction.", 300);
  const Int num_y_elements = parser.OptionalInput<Int>(
      "num_y_elements", "The number of elements in the y direction.", 300);
  const double pml_scale = parser.OptionalInput<double>(
      "pml_scale", "The scaling factor of the PML profile.", 500.);
  const double pml_exponent = parser.OptionalInput<double>(
      "pml_exponent", "The exponent of the PML profile.", 3.);
  const Int num_pml_elements = parser.OptionalInput<Int>(
      "num_pml_elements", "The number of elements the PML should span.", 10);
  const double source_x0 = parser.OptionalInput<double>(
      "source_x0", "The x location of the first point source.", 0.5);
  const double source_y0 = parser.OptionalInput<double>(
      "source_y0", "The y location of the first point source.", 0.125);
  const double source_scale0 = parser.OptionalInput<double>(
      "source_scale0", "The amplitude of the first point source.", 1000.);
  const double source_stddev0 = parser.OptionalInput<double>(
      "source_stddev0", "The standard deviation of the first point source.",
      1e-3);
  const double source_x1 = parser.OptionalInput<double>(
      "source_x1", "The x location of the second point source.", 0.4);
  const double source_y1 = parser.OptionalInput<double>(
      "source_y1", "The y location of the second point source.", 0.4);
  const double source_scale1 = parser.OptionalInput<double>(
      "source_scale1", "The amplitude of the second point source.", 1000.);
  const double source_stddev1 = parser.OptionalInput<double>(
      "source_stddev1", "The standard deviation of the second point source.",
      1e-3);
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
                                "0:left-looking, 1:up-looking, 2:right-looking,"
                                "3:adaptive",
                                3);
  const bool print_progress = parser.OptionalInput<bool>(
      "print_progress", "Print the progress of the experiments?", false);
  if (!parser.OK()) {
    return 0;
  }

  const SpeedProfile profile = static_cast<SpeedProfile>(speed_profile_int);

  const Buffer<GaussianSource<double>> sources{
      GaussianSource<double>{Point<double>{source_x0, source_y0}, source_scale0,
                             source_stddev0},
      GaussianSource<double>{Point<double>{source_x1, source_y1}, source_scale1,
                             source_stddev1},
  };

  catamari::SparseLDLControl ldl_control;
  ldl_control.SetFactorizationType(catamari::kLDLTransposeFactorization);
  ldl_control.supernodal_strategy =
      static_cast<catamari::SupernodalStrategy>(supernodal_strategy_int);

  // Set the minimum degree control options.
  {
    auto& md_control = ldl_control.md_control;
    md_control.degree_type = static_cast<quotient::DegreeType>(degree_type_int);
    md_control.aggressive_absorption = aggressive_absorption;
    md_control.min_dense_threshold = min_dense_threshold;
    md_control.dense_sqrt_multiple = dense_sqrt_multiple;
  }

  // Set the scalar control options.
  ldl_control.scalar_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);

  // Set the supernodal control options.
  {
    auto& sn_control = ldl_control.supernodal_control;
    sn_control.algorithm =
        static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);
    sn_control.relaxation_control.relax_supernodes = relax_supernodes;
    sn_control.relaxation_control.allowable_supernode_zeros =
        allowable_supernode_zeros;
    sn_control.relaxation_control.allowable_supernode_zero_ratio =
        allowable_supernode_zero_ratio;
  }

  std::cout << "Solving with double-precision..." << std::endl;
  SolveModel<double>(profile, omega, num_x_elements, num_y_elements, pml_scale,
                     pml_exponent, num_pml_elements, sources,
                     analytical_ordering, ldl_control, print_progress);

  std::cout << "Solving with double-double-precision..." << std::endl;
  SolveModel<mantis::DoubleMantissa<double>>(
      profile, omega, num_x_elements, num_y_elements, pml_scale, pml_exponent,
      num_pml_elements, sources, analytical_ordering, ldl_control,
      print_progress);

  return 0;
}
