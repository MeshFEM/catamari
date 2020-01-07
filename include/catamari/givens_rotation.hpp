/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.  */
#ifndef CATAMARI_GIVENS_ROTATION_H_
#define CATAMARI_GIVENS_ROTATION_H_

#include "catamari/complex.hpp"

namespace catamari {

// A Givens rotation is defined as the unitary matrix
//
//   givens(c, s) = |      c,    s |,
//                  | -conj(s),  c |
//
// where c is real-valued and s is generally complex-valued, such that
// c^2 + |s|^2 = 1. In the real-valued case, there exists an angle, theta in
// [0, 2 pi), such that c = cos(theta) and s = sin(theta).
//
// Since the determinant of a Givens rotation is
//
//   c^2 + |s|^2 = 1,
//
// we see that Givens rotations are a subset of SU(2), which would only be
// covered if we allowed c to take on complex values and we used the
// generalized parameterization
//
//   |     c,        s    |.
//   | -conj(s),  conj(c) |
//
// We typically define a Givens rotation so that it "zeroes out" the bottom
// entry of a given length-two vector via the action:
//
//   |     c,     s | | alpha | = | rho |.
//   | -conj(s),  c | |  beta |   |  0  |
//
template <typename Field>
struct GivensRotation {
  typedef ComplexBase<Field> Real;

  // The cosine component of the Givens rotation.
  Real cosine;

  // The sine component of the Givens rotation.
  Field sine;

  // Returns the minimum positive value of tye 'Real' which is safe to invert
  // without overflow.
  static Real MinimumSafeToInvert();

  // Returns the minimum positive value of type 'Real' which is safe to square
  // without underflow. The formula follows that of LAPACK Working Note 148:
  //
  //   base^{round((1 / 2) log_{base}(safe_min / epsilon))}.
  //
  static Real MinimumSafeToSquare();

  // Returns the maximum absolute value of the real and imaginary components of
  // a (generally) complex value.
  static Real MaxAbs(const Field& alpha);

  // Safely computes the two-norm of two real numbers.
  static Real SafeNorm(const Real& alpha, const Real& beta);

  // Safely computes the two-norm of the real and imaginary components of a
  // (generally) complex number.
  static Real SafeAbs(const Field& alpha);

  // Defines this Givens rotation, and the resulting combined_entry, via the
  // given length-two vector, [alpha; beta], so that
  //
  //   givens(c, s) | alpha | = | combined_entry |,
  //                |  beta |   |        0       |
  //
  // where alpha and beta are both real-valued.
  //
  // Please see LAPACK Working Note 148,
  //
  //   D. Bindel, J. Demmel, W. Kahan, and O. Marques,
  //   "On computing Givens rotations reliably and efficiently",
  //   http://www.netlib.org/lapack/lawnspdf/lawn148.pdf
  //
  // which resulted in the LAPACK routines {s,d,c,z}lartg. But note
  // that the LAPACK implementations slightly differ from said working
  // note in that they round z^2 to the nearest radix rather than z
  // (which results in a different result with float, but the same with
  // double).
  //
  Real GenerateOverReals(const Real& alpha, const Real& beta);

  // Defines this Givens rotation, and the resulting combined_entry, via the
  // given length-two vector, [alpha; beta], so that
  //
  //   givens(c, s) | alpha | = | combined_entry |.
  //                |  beta |   |        0       |
  //
  // Again, please see LAPACK Working Note 148.
  //
  Field Generate(const Field& alpha, const Field& beta);

  // Applies this Givens rotation to a length-two vector via:
  //
  //   | alpha | := givens(c, s) | alpha |.
  //   |  beta |                 |  beta |
  //
  void Apply(Field* alpha, Field* beta) const;
};

}  // namespace catamari

#include "catamari/givens_rotation-impl.hpp"

#endif  // ifndef CATAMARI_GIVENS_ROTATION_H_
