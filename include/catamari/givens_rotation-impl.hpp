/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.  */
#ifndef CATAMARI_GIVENS_ROTATION_IMPL_H_
#define CATAMARI_GIVENS_ROTATION_IMPL_H_

#include <limits>

#include "catamari/givens_rotation.hpp"

namespace catamari {

template <typename Field>
ComplexBase<Field> GivensRotation<Field>::MinimumSafeToInvert() {
  static const Real epsilon = std::numeric_limits<Real>::epsilon();
  static const Real min = std::numeric_limits<Real>::min();
  static const Real inv_max = Real(1) / std::numeric_limits<Real>::max();
  static const Real min_safe_to_invert =
      inv_max > min ? inv_max * (1 + epsilon) : min;
  return min_safe_to_invert;
}

template <typename Field>
ComplexBase<Field> GivensRotation<Field>::MinimumSafeToSquare() {
  static const Real radix = std::numeric_limits<Real>::radix;
  static const Real epsilon = std::numeric_limits<Real>::epsilon();
  static const Real min_safe_to_invert = MinimumSafeToInvert();
  static const Real min_safe_to_square =
      std::pow(radix, std::round(std::log(min_safe_to_invert / epsilon) /
                                 (2 * std::log(radix))));
  return min_safe_to_square;
}

template <typename Field>
ComplexBase<Field> GivensRotation<Field>::MaxAbs(const Field& alpha) {
  return std::max(std::abs(std::real(alpha)), std::abs(std::imag(alpha)));
}

template <typename Field>
ComplexBase<Field> GivensRotation<Field>::SafeNorm(const Real& alpha,
                                                   const Real& beta) {
  const Real alpha_abs = std::abs(alpha);
  const Real beta_abs = std::abs(beta);
  const Real min_abs = std::min(alpha_abs, beta_abs);
  const Real max_abs = std::max(alpha_abs, beta_abs);

  if (min_abs == Real(0)) {
    return max_abs;
  } else {
    const Real ratio = min_abs / max_abs;
    return max_abs * std::sqrt(1 + ratio * ratio);
  }
}

template <typename Field>
ComplexBase<Field> GivensRotation<Field>::SafeAbs(const Field& alpha) {
  if (std::imag(alpha) == Real(0)) {
    return std::abs(std::real(alpha));
  }
  return SafeNorm(std::real(alpha), std::imag(alpha));
}

template <typename Field>
ComplexBase<Field> GivensRotation<Field>::GenerateOverReals(const Real& alpha,
                                                            const Real& beta) {
  if (beta == Real(0)) {
    // Use the identity matrix as our Givens rotation.
    cosine = 1;
    sine = 0;
    return alpha;
  } else if (alpha == Real(0)) {
    // Use the swap matrix as our Givens rotation.
    cosine = 0;
    sine = 1;
    return beta;
  }

  static const Real min_safe_to_square = MinimumSafeToSquare();
  static const Real max_safe_to_square = Real(1) / min_safe_to_square;

  // Ensure that the maximum of |alpha| and |beta| is in the interval
  //
  //   0 < [min_safe_to_square, max_safe_to_square].
  //
  // We keep track of the number of steps of upscaling or downscaling we
  // must use to postprocess our scaled result.
  Real scale = std::max(std::abs(alpha), std::abs(beta));
  Real alpha_scaled = alpha;
  Real beta_scaled = beta;
  Int rescale_counter = 0;
  while (scale > max_safe_to_square) {
    alpha_scaled *= min_safe_to_square;
    beta_scaled *= min_safe_to_square;
    scale *= min_safe_to_square;
    ++rescale_counter;
  }
  if (rescale_counter == 0 && scale < min_safe_to_square) {
    if (beta == Real(0) || !std::isfinite(std::abs(beta))) {
      // Use the identity matrix as our Givens rotation.
      cosine = 1;
      sine = 0;
      return alpha;
    }
    while (scale < min_safe_to_square) {
      alpha_scaled *= max_safe_to_square;
      beta_scaled *= max_safe_to_square;
      scale *= max_safe_to_square;
      --rescale_counter;
    }
  }

  // Compute the norm of the scaled input vector.
  Real combined_entry =
      std::sqrt(alpha_scaled * alpha_scaled + beta_scaled * beta_scaled);

  // Compute the initial proposal for the Givens rotation parameters.
  // By definition, their combined norm will be one.
  cosine = alpha_scaled / combined_entry;
  sine = beta_scaled / combined_entry;

  // Rescale the nonzero result of the proposed Givens rotation.
  for (; rescale_counter > 0; --rescale_counter) {
    combined_entry *= max_safe_to_square;
  }
  for (; rescale_counter < 0; ++rescale_counter) {
    combined_entry *= min_safe_to_square;
  }

  if (std::abs(alpha) > std::abs(beta) && cosine < Real(0)) {
    // Negate the Givens rotation and the nonzero entry of the result.
    cosine = -cosine;
    sine = -sine;
    combined_entry = -combined_entry;
  }

  return combined_entry;
}

template <typename Field>
Field GivensRotation<Field>::Generate(const Field& alpha, const Field& beta) {
  if (std::imag(alpha) == Real(0) && std::imag(beta) == Real(0)) {
    return GenerateOverReals(std::real(alpha), std::real(beta));
  }

  static const Real min_safe_to_invert = MinimumSafeToInvert();
  static const Real min_safe_to_square = MinimumSafeToSquare();
  static const Real max_safe_to_square = Real(1) / min_safe_to_square;

  // Ensure that the maximum of |alpha| and |beta| is in the interval
  //
  //   0 < [min_safe_to_square, max_safe_to_square]
  //
  // and keep track of the number of postprocessing steps that need to be
  // performed.
  Real scale = std::max(MaxAbs(alpha), MaxAbs(beta));
  Field alpha_scaled = alpha;
  Field beta_scaled = beta;
  Int rescale_counter = 0;
  while (scale > max_safe_to_square) {
    alpha_scaled *= min_safe_to_square;
    beta_scaled *= min_safe_to_square;
    scale *= min_safe_to_square;
    ++rescale_counter;
  }
  if (rescale_counter == 0 && scale < min_safe_to_square) {
    if (beta == Field(0) || !std::isfinite(std::abs(beta))) {
      // Use the identity matrix as our Givens rotation.
      cosine = 1;
      sine = 0;
      return alpha;
    }
    while (scale < min_safe_to_square) {
      alpha_scaled *= max_safe_to_square;
      beta_scaled *= max_safe_to_square;
      scale *= max_safe_to_square;
      --rescale_counter;
    }
  }

  const Real alpha_scaled_squared_norm =
      std::real(alpha_scaled) * std::real(alpha_scaled) +
      std::imag(alpha_scaled) * std::imag(alpha_scaled);
  const Real beta_scaled_squared_norm =
      std::real(beta_scaled) * std::real(beta_scaled) +
      std::imag(beta_scaled) * std::imag(beta_scaled);
  if (alpha_scaled_squared_norm <=
      std::max(beta_scaled_squared_norm, Real(1)) * min_safe_to_invert) {
    // This branch handles the exceptional case where alpha is very small
    // -- either in an absolute sense or relative to beta.

    if (alpha == Field(0)) {
      // Use a swap matrix with a phase.
      cosine = 0;
      sine = Conjugate(beta_scaled) / SafeAbs(beta_scaled);
      return SafeAbs(beta);
    }

    const Real alpha_scaled_abs = SafeAbs(alpha_scaled);
    const Real beta_scaled_abs = std::sqrt(beta_scaled_squared_norm);

    cosine = alpha_scaled_abs / beta_scaled_abs;

    Field phase;
    if (MaxAbs(alpha) > Real(1)) {
      phase = alpha / SafeAbs(alpha);
    } else {
      const Field delta = max_safe_to_square * alpha;
      phase = delta / SafeAbs(delta);
    }

    sine = phase * (Conjugate(beta_scaled) / beta_scaled_abs);

    return cosine * alpha + sine * beta;
  } else {
    // This is the usual branch where alpha is not excessively small.
    const Real tau = std::sqrt(Real(1) + beta_scaled_squared_norm /
                                             alpha_scaled_squared_norm);

    Field combined_entry = tau * alpha_scaled;
    cosine = Real(1) / tau;
    const Real delta = alpha_scaled_squared_norm + beta_scaled_squared_norm;
    sine = (combined_entry / delta) * Conjugate(beta_scaled);

    // Rescale the nonzero result of the proposed Givens rotation.
    for (; rescale_counter > 0; --rescale_counter) {
      combined_entry *= max_safe_to_square;
    }
    for (; rescale_counter < 0; ++rescale_counter) {
      combined_entry *= min_safe_to_square;
    }

    return combined_entry;
  }
}

template <typename Field>
void GivensRotation<Field>::Apply(Field* alpha, Field* beta) const {
  const Field alpha_new = cosine * (*alpha) + sine * (*beta);
  const Field beta_new = -Conjugate(sine) * (*alpha) + cosine * (*beta);
  *alpha = alpha_new;
  *beta = beta_new;
}

}  // namespace catamari

#endif  // ifndef CATAMARI_GIVENS_ROTATION_IMPL_H_
