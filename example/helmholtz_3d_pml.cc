/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// This drive is a simple implementation of a 3D Helmholtz equation in the
// unit box, [0, 1]^3, with Perfectly Matched Layer absorbing boundary
// conditions on all sides. The discretization is over boxs with trilinear,
// Lagrangian basis functions based at the corner points. The trilinear form is
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

// A representation of an arbitrary axis-aligned box,
//
//     [x_beg, x_end] x [y_beg, y_end] x [z_beg, z_end].
//
template <typename Real>
struct Box {
  Real x_beg;
  Real x_end;
  Real y_beg;
  Real y_end;
  Real z_beg;
  Real z_end;
};

// A cache for the tensor-product quadrature points over an arbitrary box.
template <typename Real>
struct BoxQuadrature {
  Real x_points[3];
  Real x_weights[3];

  Real y_points[3];
  Real y_weights[3];

  Real z_points[3];
  Real z_weights[3];

  BoxQuadrature(const Box<Real>& extent) {
    ThirdOrderGaussianPointsAndWeights(extent.x_beg, extent.x_end, x_points,
                                       x_weights);
    ThirdOrderGaussianPointsAndWeights(extent.y_beg, extent.y_end, y_points,
                                       y_weights);
    ThirdOrderGaussianPointsAndWeights(extent.z_beg, extent.z_end, z_points,
                                       z_weights);
  }
};

// A point in the 3D domain (i.e., [0, 1]^3).
template <typename Real>
struct Point {
  Real x;
  Real y;
  Real z;
};

// The profile of a single direction's PML as a function of a principal axis.
// NOTE: The majority of the runtime appears to be spent in this function.
template <typename Real>
class PMLProfile {
 public:
  PMLProfile(
      const Real& pml_scale, const Real& pml_exponent, const Real& pml_width) :
      pml_scale_(pml_scale), pml_exponent_(pml_exponent), pml_width_(pml_width)
  {}

  Real operator()(const Real& x) const {
    Real result;
    const Real first_pml_end = pml_width_;
    const Real last_pml_beg = 1 - pml_width_;
    if (x < first_pml_end) {
      const Real pml_rel_depth = (first_pml_end - x) / pml_width_;
      result = pml_scale_ * std::pow(pml_rel_depth, pml_exponent_);
    } else if (x > last_pml_beg) {
      const Real pml_rel_depth = (x - last_pml_beg) / pml_width_;
      result = pml_scale_ * std::pow(pml_rel_depth, pml_exponent_);
    } else {
      result = 0;
    }
    return result;
  }

 private:
  const Real pml_scale_;
  const Real pml_exponent_;
  const Real pml_width_;
};

// The differential mapping the trivial tangent space of the real line into
// the tangent space of the complex-stretched domain.
template <typename Real>
class PMLDifferential {
 public:
  PMLDifferential(
      const Real& omega, const PMLProfile<Real>& profile) :
  omega_(omega), profile_(profile) {}

  Complex<Real> operator()(const Real& x) const {
    return Complex<Real>(Real{1}, profile_(x) / omega_);
  }

 private:
  const Real omega_;
  const PMLProfile<Real> profile_;
};

// The pointwise acoustic speed.
template <typename Real>
class Speed {
 public:
  Speed() {}

  Real operator()(const Point<Real>& point) const {
    return Real{1};
  }
};

// The diagonal 'A' tensor in the equation:
//
//    -div (A grad u) - (omega / c)^2 gamma u = f.
//
template <typename Real>
class WeightTensor {
 public:
  WeightTensor(const PMLDifferential<Real>& gamma_x,
               const PMLDifferential<Real>& gamma_y,
               const PMLDifferential<Real>& gamma_z) :
      gamma_x_(gamma_x), gamma_y_(gamma_y), gamma_z_(gamma_z) {}

  Complex<Real> operator()(int i, const Point<Real>& point) const {
    const Complex<Real> gamma_x = gamma_x_(point.x);
    const Complex<Real> gamma_y = gamma_y_(point.y);
    const Complex<Real> gamma_z = gamma_z_(point.z);
    Complex<Real> result = gamma_x * gamma_y * gamma_z;
    Complex<Real> gamma_k;
    if (i == 0) {
      return result / (gamma_x * gamma_x);
    } else if (i == 1) {
      return result / (gamma_y * gamma_y);
    } else {
      return result / (gamma_z * gamma_z);
    }
  }

 private:
  const PMLDifferential<Real> gamma_x_;
  const PMLDifferential<Real> gamma_y_;
  const PMLDifferential<Real> gamma_z_;
};

// The '(omega / c)^2 gamma' in the equation:
//
//    -div (A grad u) - (omega / c)^2 gamma u = f.
//
template <typename Real>
class DiagonalShift {
 public:
  DiagonalShift(const Real& omega, const Speed<Real>& speed,
                const PMLDifferential<Real>& gamma_x,
                const PMLDifferential<Real>& gamma_y,
                const PMLDifferential<Real>& gamma_z) :
      omega_(omega), speed_(speed), gamma_x_(gamma_x), gamma_y_(gamma_y),
      gamma_z_(gamma_z) {}

  Complex<Real> operator()(const Point<Real>& point) const {
    const Real rel_omega = omega_ / speed_(point);
    const Complex<Real> gamma =
        gamma_x_(point.x) * gamma_y_(point.y) * gamma_z_(point.z);
    return rel_omega * rel_omega * gamma;
  }

 private:
  const Real omega_;
  const Speed<Real> speed_;
  const PMLDifferential<Real> gamma_x_;
  const PMLDifferential<Real> gamma_y_;
  const PMLDifferential<Real> gamma_z_;
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
class HelmholtzWithPMLTrilinearHexahedron {
 public:
  // The term -div (A grad u) involves a two-tensor A : Omega -> C^{3 x 3}.
  // The (i, j, k) index of the result is thus a function:
  //
  //   A_{i, j, k} : Omega -> C,  i, j, k in {0, 1}.
  //
  // Since our weight tensor is diagonal, only we need only provide
  //
  //   A_{i, i} : Omega -> C,  i in {0, 1, 2}.
  //
  typedef WeightTensor<Real> DiagonalWeightTensor; 

  // The term -s u involves a scalar function s : Omega -> C, which we refer
  // to as the diagonal shift function.
  typedef DiagonalShift<Real> DiagonalShiftFunction;

  // The constructor for the trilinear hexahedron.
  HelmholtzWithPMLTrilinearHexahedron(
      const DiagonalWeightTensor weight_tensor,
      const DiagonalShiftFunction shift_function)
      : weight_tensor_(weight_tensor), shift_function_(shift_function) {}

  // Returns \psi_{i, j, k} evaluated at the given point.
  Real Basis(int i, int j, int k, const Box<Real>& extent,
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
  Real BasisGradient(int i, int j, int k, int l, const Box<Real>& extent,
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

  // Returns the evaluation of the density of the bilinear form over an element
  // [x_beg, x_end] x [y_beg, y_end] x [z_beg, z_end] at a point (x, y, z) with
  // test function v = \phi_{i_test, j_test, k_test} and trial function
  // u = \psi_{i_trial, j_trial, k_trial}. That is,
  //
  //   a_density(u, v) = (grad v)' (A grad u) - s u conj(v).
  //
  Complex<Real> BilinearFormDensity(int i_test, int j_test, int k_test,
                                    int i_trial, int j_trial, int k_trial,
                                    const Box<Real>& extent,
                                    const Point<Real>& point) const {
    Complex<Real> result = 0;

    // Add in the (grad v)' (A grad u) contribution. Recall that A is diagonal.
    for (int l = 0; l < 3; ++l) {
      const Real test_grad_entry =
          BasisGradient(i_test, j_test, k_test, l, extent, point);
      const Real trial_grad_entry =
          BasisGradient(i_trial, j_trial, k_trial, l, extent, point);
      const Complex<Real> weight_entry = weight_tensor_(l, point);
      // We explicitly call 'Conjugate' even though the basis functions are
      // real.
      result += Conjugate(test_grad_entry) * (weight_entry * trial_grad_entry);
    }

    // Add in the -s u conj(v) contribution.
    const Real test_entry = Basis(i_test, j_test, k_test, extent, point);
    const Real trial_entry = Basis(i_trial, j_trial, k_trial, extent, point);
    const Complex<Real> diagonal_shift = shift_function_(point);
    // Again, we explicitly call 'Conjugate' even though the basis functions
    // are real.
    result -= diagonal_shift * trial_entry * Conjugate(test_entry);

    return result;
  }

  // Use a tensor product of third-order Gaussian quadrature to integrate the
  // trilinear form over the hexahedral element.
  Complex<Real> BilinearForm(int i_test, int j_test, int k_test, int i_trial,
                             int j_trial, int k_trial, const Box<Real>& extent,
                             const BoxQuadrature<Real>& quadrature) const {
    Complex<Real> result = 0;
    for (int i = 0; i < 3; ++i) {
      const Real& x = quadrature.x_points[i];
      const Real& x_weight = quadrature.x_weights[i];
      for (int j = 0; j < 3; ++j) {
        const Real& y = quadrature.y_points[j];
        const Real& y_weight = quadrature.y_weights[j];
        for (int k = 0; k < 3; ++k) {
          const Real& z = quadrature.z_points[k];
          const Real& z_weight = quadrature.z_weights[k];
          const Point<Real> point{x, y, z};
          result += x_weight * y_weight * z_weight *
              BilinearFormDensity(i_test, j_test, k_test, i_trial, j_trial,
                                  k_trial, extent, point);
        }
      }
    }
    return result;
  }

 private:
  // The diagonal symmetric tensor field A : Omega -> C^{3 x 3} in the
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

// Generates a trilinear hexahedral discretization of the 2D Helmholtz equation
// over [0, 1]^3 with inserted PML.
template <typename Real>
std::unique_ptr<catamari::CoordinateMatrix<Complex<Real>>> HelmholtzWithPML(
    const Real& omega, Int num_x_elements, Int num_y_elements,
    Int num_z_elements, const Real& pml_scale, const Real& pml_exponent,
    Int num_pml_elements) {
  const Real h_x = Real{1} / num_x_elements;
  const Real h_y = Real{1} / num_y_elements;
  const Real h_z = Real{1} / num_z_elements;

  std::unique_ptr<catamari::CoordinateMatrix<Complex<Real>>> matrix(
      new catamari::CoordinateMatrix<Complex<Real>>);

  const Real x_pml_width = num_pml_elements * h_x;
  const Real y_pml_width = num_pml_elements * h_y;
  const Real z_pml_width = num_pml_elements * h_z;

  // The frequency-scaled horizontal PML profile function.
  const PMLProfile<Real> sigma_x(pml_scale, pml_exponent, x_pml_width);
  const PMLProfile<Real> sigma_y(pml_scale, pml_exponent, y_pml_width);
  const PMLProfile<Real> sigma_z(pml_scale, pml_exponent, z_pml_width);

  // The differentials from the real axes to the PML Profile tangent spaces.
  const PMLDifferential<Real> gamma_x(omega, sigma_x);
  const PMLDifferential<Real> gamma_y(omega, sigma_y);
  const PMLDifferential<Real> gamma_z(omega, sigma_z);

  const Speed<Real> speed;

  const WeightTensor<Real> weight_tensor(gamma_x, gamma_y, gamma_z);

  const DiagonalShift<Real> diagonal_shift(
      omega, speed, gamma_x, gamma_y, gamma_z);

  const HelmholtzWithPMLTrilinearHexahedron<Real> element(weight_tensor,
                                                          diagonal_shift);

  const Int num_rows =
      (num_x_elements + 1) * (num_y_elements + 1) * (num_z_elements + 1);
  const Int y_stride = num_x_elements + 1;
  const Int z_stride = y_stride * (num_y_elements + 1);
  matrix->Resize(num_rows, num_rows);
  const Int queue_upper_bound =
      64 * num_x_elements * num_y_elements * num_z_elements;
  matrix->ReserveEntryAdditions(queue_upper_bound);
  for (Int x_element = 0; x_element < num_x_elements; ++x_element) {
    const Real x_beg = x_element * h_x;
    const Real x_end = x_beg + h_x;
    for (Int y_element = 0; y_element < num_y_elements; ++y_element) {
      const Real y_beg = y_element * h_y;
      const Real y_end = y_beg + h_y;
      for (Int z_element = 0; z_element < num_z_elements; ++z_element) {
        const Real z_beg = z_element * h_z;
        const Real z_end = z_beg + h_z;
        const Box<Real> extent{x_beg, x_end, y_beg, y_end, z_beg, z_end};
        const BoxQuadrature<Real> quadrature(extent);

        // The index of the bottom-left entry of the hexahedron.
        const Int offset =
            x_element + y_element * y_stride + z_element * z_stride;

        // Iterate over all of the interactions in the element.
        for (Int i_test = 0; i_test <= 1; ++i_test) {
          for (Int j_test = 0; j_test <= 1; ++j_test) {
            for (Int k_test = 0; k_test <= 1; ++k_test) {
              const Int row =
                  offset + i_test + j_test * y_stride + k_test * z_stride;
              for (Int i_trial = 0; i_trial <= 1; ++i_trial) {
                for (Int j_trial = 0; j_trial <= 1; ++j_trial) {
                  for (Int k_trial = 0; k_trial <= 1; ++k_trial) {
                    const Int column = offset + i_trial + j_trial * y_stride +
                                       k_trial * z_stride;
                    const Complex<Real> value =
                        element.BilinearForm(i_test, j_test, k_test, i_trial,
                                             j_trial, k_trial, extent,
                                             quadrature);
                    matrix->QueueEntryAddition(row, column, value);
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

  return matrix;
}

template <typename Real>
ConstBlasMatrix<Complex<Real>> GenerateRightHandSide(
    Int num_x_elements, Int num_y_elements, Int num_z_elements,
    std::vector<Complex<Real>>* buffer) {
  typedef Complex<Real> Field;
  const Int num_rows =
      (num_x_elements + 1) * (num_y_elements + 1) * (num_z_elements + 1);

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
      (num_x_elements / 2) + (num_y_elements / 2) * (num_x_elements + 1) +
      (num_z_elements / 2) * (num_x_elements + 1) * (num_y_elements + 1);
  right_hand_side(middle_row, 0) = Complex<Real>{1};

  return right_hand_side.ToConst();
}

struct IntegerBox {
  Int x_beg;
  Int x_end;
  Int y_beg;
  Int y_end;
  Int z_beg;
  Int z_end;
};

inline void AnalyticalOrderingRecursion(Int num_x_elements, Int num_y_elements,
                                        Int num_z_elements, Int offset,
                                        const IntegerBox& box,
                                        std::vector<Int>* inverse_permutation) {
  const Int y_stride = num_x_elements + 1;
  const Int z_stride = y_stride * (num_y_elements + 1);
  const Int min_cut_size = 5;

  // Determine which dimension to cut (if any).
  const Int x_size = box.x_end - box.x_beg;
  const Int y_size = box.y_end - box.y_beg;
  const Int z_size = box.z_end - box.z_beg;
  const Int max_size = std::max(std::max(x_size, y_size), z_size);
  CATAMARI_ASSERT(max_size > 0, "The maximum size was non-positive.");
  if (max_size < min_cut_size) {
    // Do not split.
    for (Int z = box.z_beg; z < box.z_end; ++z) {
      for (Int y = box.y_beg; y < box.y_end; ++y) {
        for (Int x = box.x_beg; x < box.x_end; ++x) {
          (*inverse_permutation)[offset++] = x + y * y_stride + z * z_stride;
        }
      }
    }
    return;
  }

  if (x_size == max_size) {
    // Cut the x dimension.
    const Int x_cut = box.x_beg + (box.x_end - box.x_beg) / 2;
    const IntegerBox left_box{box.x_beg, x_cut,     box.y_beg,
                              box.y_end, box.z_beg, box.z_end};
    const IntegerBox right_box{x_cut + 1, box.x_end, box.y_beg,
                               box.y_end, box.z_beg, box.z_end};
    const Int left_offset = offset;
    const Int right_offset =
        left_offset + (left_box.x_end - left_box.x_beg) *
                          (left_box.y_end - left_box.y_beg) *
                          (left_box.z_end - left_box.z_beg);
    const Int cut_offset =
        right_offset + (right_box.x_end - right_box.x_beg) *
                           (right_box.y_end - right_box.y_beg) *
                           (right_box.z_end - right_box.z_beg);

    // Fill the left child.
    AnalyticalOrderingRecursion(num_x_elements, num_y_elements, num_z_elements,
                                left_offset, left_box, inverse_permutation);

    // Fill the right child.
    AnalyticalOrderingRecursion(num_x_elements, num_y_elements, num_z_elements,
                                right_offset, right_box, inverse_permutation);

    // Fill the separator.
    offset = cut_offset;
    for (Int z = box.z_beg; z < box.z_end; ++z) {
      for (Int y = box.y_beg; y < box.y_end; ++y) {
        (*inverse_permutation)[offset++] = x_cut + y * y_stride + z * z_stride;
      }
    }
  } else if (y_size == max_size) {
    // Cut the y dimension.
    const Int y_cut = box.y_beg + (box.y_end - box.y_beg) / 2;
    const IntegerBox left_box{box.x_beg, box.x_end, box.y_beg,
                              y_cut,     box.z_beg, box.z_end};
    const IntegerBox right_box{box.x_beg, box.x_end, y_cut + 1,
                               box.y_end, box.z_beg, box.z_end};
    const Int left_offset = offset;
    const Int right_offset =
        left_offset + (left_box.x_end - left_box.x_beg) *
                          (left_box.y_end - left_box.y_beg) *
                          (left_box.z_end - left_box.z_beg);
    const Int cut_offset =
        right_offset + (right_box.x_end - right_box.x_beg) *
                           (right_box.y_end - right_box.y_beg) *
                           (right_box.z_end - right_box.z_beg);

    // Fill the left child.
    AnalyticalOrderingRecursion(num_x_elements, num_y_elements, num_z_elements,
                                left_offset, left_box, inverse_permutation);

    // Fill the right child.
    AnalyticalOrderingRecursion(num_x_elements, num_y_elements, num_z_elements,
                                right_offset, right_box, inverse_permutation);

    // Fill the separator.
    offset = cut_offset;
    for (Int z = box.z_beg; z < box.z_end; ++z) {
      for (Int x = box.x_beg; x < box.x_end; ++x) {
        (*inverse_permutation)[offset++] = x + y_cut * y_stride + z * z_stride;
      }
    }
  } else {
    // Cut the z dimension.
    const Int z_cut = box.z_beg + (box.z_end - box.z_beg) / 2;
    const IntegerBox left_box{box.x_beg, box.x_end, box.y_beg,
                              box.y_end, box.z_beg, z_cut};
    const IntegerBox right_box{box.x_beg, box.x_end, box.y_beg,
                               box.y_end, z_cut + 1, box.z_end};
    const Int left_offset = offset;
    const Int right_offset =
        left_offset + (left_box.x_end - left_box.x_beg) *
                          (left_box.y_end - left_box.y_beg) *
                          (left_box.z_end - left_box.z_beg);
    const Int cut_offset =
        right_offset + (right_box.x_end - right_box.x_beg) *
                           (right_box.y_end - right_box.y_beg) *
                           (right_box.z_end - right_box.z_beg);

    // Fill the left child.
    AnalyticalOrderingRecursion(num_x_elements, num_y_elements, num_z_elements,
                                left_offset, left_box, inverse_permutation);

    // Fill the right child.
    AnalyticalOrderingRecursion(num_x_elements, num_y_elements, num_z_elements,
                                right_offset, right_box, inverse_permutation);

    // Fill the separator.
    offset = cut_offset;
    for (Int y = box.y_beg; y < box.y_end; ++y) {
      for (Int x = box.x_beg; x < box.x_end; ++x) {
        (*inverse_permutation)[offset++] = x + y * y_stride + z_cut * z_stride;
      }
    }
  }
}

// Because of the connectivity of the trilinear hexahedra, we can impose an
// analytical nested-dissection ordering with a width of 1.
inline void AnalyticalOrdering(Int num_x_elements, Int num_y_elements,
                               Int num_z_elements,
                               std::vector<Int>* permutation,
                               std::vector<Int>* inverse_permutation) {
  const Int num_rows =
      (num_x_elements + 1) * (num_y_elements + 1) * (num_z_elements + 1);
  permutation->resize(num_rows);
  inverse_permutation->resize(num_rows);

  Int offset = 0;
  IntegerBox box{0, num_x_elements + 1, 0, num_y_elements + 1,
                 0, num_z_elements + 1};
  AnalyticalOrderingRecursion(num_x_elements, num_y_elements, num_z_elements,
                              offset, box, inverse_permutation);

  // Invert the inverse permutation.
  for (Int row = 0; row < num_rows; ++row) {
    (*permutation)[(*inverse_permutation)[row]] = row;
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
Experiment RunTest(const double& omega, Int num_x_elements, Int num_y_elements,
                   Int num_z_elements, const double& pml_scale,
                   const double& pml_exponent, int num_pml_elements,
                   bool analytical_ordering,
                   const catamari::LDLControl& ldl_control,
                   bool print_progress) {
  typedef Complex<double> Field;
  typedef catamari::ComplexBase<Field> Real;
  Experiment experiment;
  quotient::Timer timer;

  // Read the matrix from file.
  timer.Start();
  std::unique_ptr<catamari::CoordinateMatrix<Field>> matrix =
      HelmholtzWithPML<Real>(omega, num_x_elements, num_y_elements,
                             num_z_elements, pml_scale, pml_exponent,
                             num_pml_elements);
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
  catamari::LDLResult result;
  if (analytical_ordering) {
    std::vector<Int> permutation, inverse_permutation;
    AnalyticalOrdering(num_x_elements, num_y_elements, num_z_elements,
                       &permutation, &inverse_permutation);
    result = catamari::LDL(*matrix, permutation, inverse_permutation,
                           ldl_control, &ldl_factorization);
  } else {
    result = catamari::LDL(*matrix, ldl_control, &ldl_factorization);
  }
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
                                  num_z_elements, &right_hand_side_buffer);
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
      "omega", "The angular frequency of the Helmholtz problem.", 20.);
  const Int num_x_elements = parser.OptionalInput<Int>(
      "num_x_elements", "The number of elements in the x direction.", 40);
  const Int num_y_elements = parser.OptionalInput<Int>(
      "num_y_elements", "The number of elements in the y direction.", 40);
  const Int num_z_elements = parser.OptionalInput<Int>(
      "num_z_elements", "The number of elements in the z direction.", 40);
  const double pml_scale = parser.OptionalInput<double>(
      "pml_scale", "The scaling factor of the PML profile.", 20.);
  const double pml_exponent = parser.OptionalInput<double>(
      "pml_exponent", "The exponent of the PML profile.", 3.);
  const Int num_pml_elements = parser.OptionalInput<Int>(
      "num_pml_elements", "The number of elements the PML should span.", 10);
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
      RunTest(omega, num_x_elements, num_y_elements, num_z_elements, pml_scale,
              pml_exponent, num_pml_elements, analytical_ordering, ldl_control,
              print_progress);
  PrintExperiment(experiment);

  return 0;
}
