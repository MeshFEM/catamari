/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_COMPLEX_H_
#define CATAMARI_COMPLEX_H_

#include <complex>
#include <type_traits>

#include "catamari/enable_if.hpp"
#include "catamari/macros.hpp"

namespace catamari {

// An extension of std::complex beyond float and double.
template <class Real>
class Complex {
 public:
  // The real and imaginary components of the complex variable.
  Real real, imag;
};

// A specialization of Complex to an underlying real type of 'float'.
template <>
class Complex<float> : public std::complex<float> {
 public:
  // The underlying real type of the complex class.
  typedef float RealType;

  // Imports of the std::complex operators.
  using std::complex<RealType>::operator=;
  using std::complex<RealType>::operator-=;
  using std::complex<RealType>::operator+=;
  using std::complex<RealType>::operator*=;
  using std::complex<RealType>::operator/=;

  // The default copy constructor.
  Complex();

  // A copy constructor from a std::complex variable.
  Complex(const std::complex<RealType>& input);

  // A copy constructor from a real variable.
  template <class RealInputType>
  Complex(const RealInputType& input);

  // A copy constructor from real and imaginary parts.
  template <class RealInputType, class ImagInputType>
  Complex(const RealInputType& real, const ImagInputType& imag);

  // A copy constructor from a Complex variable.
  template <class RealInputType>
  Complex(const Complex<RealInputType>& input);
};

// A specialization of Complex to an underlying real type of 'double'.
template <>
class Complex<double> : public std::complex<double> {
 public:
  // The underlying real type of the complex class.
  typedef double RealType;

  // Imports of the std::complex operators.
  using std::complex<RealType>::operator=;
  using std::complex<RealType>::operator-=;
  using std::complex<RealType>::operator+=;
  using std::complex<RealType>::operator*=;
  using std::complex<RealType>::operator/=;

  // The default copy constructor.
  Complex();

  // A copy constructor from a std::complex variable.
  Complex(const std::complex<RealType>& input);

  // A copy constructor from a real variable.
  template <class RealInputType>
  Complex(const RealInputType& input);

  // A copy constructor from real and imaginary parts.
  template <class RealInputType, class ImagInputType>
  Complex(const RealInputType& real, const ImagInputType& imag);

  // A copy constructor from a Complex variable.
  template <class RealInputType>
  Complex(const Complex<RealInputType>& input);
};

namespace complex_base {

template <class Real>
struct ComplexBaseHelper {
  typedef Real type;
};

template <class Real>
struct ComplexBaseHelper<Complex<Real>> {
  typedef Real type;
};

}  // namespace complex_base

// Returns the type of the base field of a real or complex scalar. For example:
//   ComplexBase<double> == double
//   ComplexBase<Complex<double>> == double.
template <class Field>
using ComplexBase = typename complex_base::ComplexBaseHelper<Field>::type;

// Encodes whether or not a given type is complex. For example,
//   IsComplex<double>::value == false
//   IsComplex<Complex<double>>::value == true
template <class Real>
struct IsComplex {
  static constexpr bool value = false;
};

template <class Real>
struct IsComplex<Complex<Real>> {
  static constexpr bool value = true;
};

// Encodes whether or not a given type is real. For example,
//   IsComplex<double>::value == true
//   IsComplex<Complex<double>>::value == false
template <class Field>
struct IsReal {
  static constexpr bool value = !IsComplex<Field>::value;
};

template <typename Condition, class T = void>
using EnableIf = typename std::enable_if<Condition::value, T>::type;

template <typename Condition, class T = void>
using DisableIf = typename std::enable_if<!Condition::value, T>::type;

// Returns the negation of a complex value.
template <class Real>
Complex<Real> operator-(const Complex<Real>& value);

// Returns the sum of two values.
template <class Real>
Complex<Real> operator+(const Complex<Real>& a, const Complex<Real>& b);
template <class Real>
Complex<Real> operator+(const Complex<Real>& a, const Real& b);
template <class Real>
Complex<Real> operator+(const Real& a, const Complex<Real>& b);

// Returns the difference of two values.
template <class Real>
Complex<Real> operator-(const Complex<Real>& a, const Complex<Real>& b);
template <class Real>
Complex<Real> operator-(const Complex<Real>& a, const Real& b);
template <class Real>
Complex<Real> operator-(const Real& a, const Complex<Real>& b);

// Returns the product of two values.
template <class Real>
Complex<Real> operator*(const Complex<Real>& a, const Complex<Real>& b);
template <class Real>
Complex<Real> operator*(const Complex<Real>& a, const Real& b);
template <class Real>
Complex<Real> operator*(const Real& a, const Complex<Real>& b);

// Returns the ratio of two values.
template <class Real>
Complex<Real> operator/(const Complex<Real>& a, const Complex<Real>& b);
template <class Real>
Complex<Real> operator/(const Complex<Real>& a, const Real& b);
template <class Real>
Complex<Real> operator/(const Real& a, const Complex<Real>& b);

// Returns the real part of a real scalar.
template <class Real>
Real RealPart(const Real& value) CATAMARI_NOEXCEPT;

// Returns the real part of a complex scalar.
template <class Real>
Real RealPart(const Complex<Real>& value) CATAMARI_NOEXCEPT;

// Returns the imaginary part of a real scalar (zero).
template <class Real>
Real ImagPart(const Real& value) CATAMARI_NOEXCEPT;

// Returns the imaginary part of a complex scalar.
template <class Real>
Real ImagPart(const Complex<Real>& value) CATAMARI_NOEXCEPT;

// Returns the complex-conjugate of a real value (the value itself).
template <class Real>
Real Conjugate(const Real& value) CATAMARI_NOEXCEPT;

// Returns the complex-conjugate of a complex value.
template <class Real>
Complex<Real> Conjugate(const Complex<Real>& value) CATAMARI_NOEXCEPT;

}  // namespace catamari

#include "catamari/complex-impl.hpp"

#endif  // ifndef CATAMARI_COMPLEX_H_
