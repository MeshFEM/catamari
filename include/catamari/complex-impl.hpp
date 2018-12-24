/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_COMPLEX_IMPL_H_
#define CATAMARI_COMPLEX_IMPL_H_

#include "catamari/complex.hpp"

namespace catamari {

Complex<float>::Complex() : std::complex<float>() {}

Complex<double>::Complex() : std::complex<double>() {}

Complex<float>::Complex(const Complex<float>& input)
    : std::complex<float>(input.real(), input.imag()) {}

Complex<double>::Complex(const Complex<double>& input)
    : std::complex<double>(input.real(), input.imag()) {}

Complex<float>::Complex(const std::complex<float>& input)
    : std::complex<float>(input) {}

Complex<double>::Complex(const std::complex<double>& input)
    : std::complex<double>(input) {}

template <class RealInputType>
Complex<float>::Complex(const RealInputType& input)
    : std::complex<float>(static_cast<float>(input)) {}

template <class RealInputType>
Complex<double>::Complex(const RealInputType& input)
    : std::complex<double>(static_cast<double>(input)) {}

template <class RealInputType>
Complex<float>::Complex(const Complex<RealInputType>& input)
    : std::complex<float>(static_cast<float>(input.real()),
                          static_cast<float>(input.imag())) {}

template <class RealInputType>
Complex<double>::Complex(const Complex<RealInputType>& input)
    : std::complex<double>(static_cast<double>(input.real()),
                           static_cast<double>(input.imag())) {}

template <class RealInputType, class ImagInputType>
Complex<float>::Complex(const RealInputType& real, const ImagInputType& imag)
    : std::complex<float>(static_cast<float>(real), static_cast<float>(imag)) {}

template <class RealInputType, class ImagInputType>
Complex<double>::Complex(const RealInputType& real, const ImagInputType& imag)
    : std::complex<double>(static_cast<double>(real),
                           static_cast<double>(imag)) {}

template <class Real>
Complex<Real> operator-(const Complex<Real>& value) {
  const std::complex<Real>& value_std =
      static_cast<const std::complex<Real>&>(value);
  return -value_std;
}

template <class Real>
Complex<Real> operator+(const Complex<Real>& a, const Complex<Real>& b) {
  const std::complex<Real>& a_std = static_cast<const std::complex<Real>&>(a);
  const std::complex<Real>& b_std = static_cast<const std::complex<Real>&>(b);
  return a_std + b_std;
}

template <class Real>
Complex<Real> operator+(const Complex<Real>& a, const Real& b) {
  const std::complex<Real>& a_std = static_cast<const std::complex<Real>&>(a);
  return a_std + b;
}

template <class Real>
Complex<Real> operator+(const Real& a, const Complex<Real>& b) {
  const std::complex<Real>& b_std = static_cast<const std::complex<Real>&>(b);
  return a + b_std;
}

template <class Real>
Complex<Real> operator-(const Complex<Real>& a, const Complex<Real>& b) {
  const std::complex<Real>& a_std = static_cast<const std::complex<Real>&>(a);
  const std::complex<Real>& b_std = static_cast<const std::complex<Real>&>(b);
  return a_std - b_std;
}

template <class Real>
Complex<Real> operator-(const Complex<Real>& a, const Real& b) {
  const std::complex<Real>& a_std = static_cast<const std::complex<Real>&>(a);
  return a_std - b;
}

template <class Real>
Complex<Real> operator-(const Real& a, const Complex<Real>& b) {
  const std::complex<Real>& b_std = static_cast<const std::complex<Real>&>(b);
  return a - b_std;
}

template <class Real>
Complex<Real> operator*(const Complex<Real>& a, const Complex<Real>& b) {
  const std::complex<Real>& a_std = static_cast<const std::complex<Real>&>(a);
  const std::complex<Real>& b_std = static_cast<const std::complex<Real>&>(b);
  return a_std * b_std;
}

template <class Real>
Complex<Real> operator*(const Complex<Real>& a, const Real& b) {
  const std::complex<Real>& a_std = static_cast<const std::complex<Real>&>(a);
  return a_std * b;
}

template <class Real>
Complex<Real> operator*(const Real& a, const Complex<Real>& b) {
  const std::complex<Real>& b_std = static_cast<const std::complex<Real>&>(b);
  return a * b_std;
}

template <class Real>
Complex<Real> operator/(const Complex<Real>& a, const Complex<Real>& b) {
  const std::complex<Real>& a_std = static_cast<const std::complex<Real>&>(a);
  const std::complex<Real>& b_std = static_cast<const std::complex<Real>&>(b);
  return a_std / b_std;
}

template <class Real>
Complex<Real> operator/(const Complex<Real>& a, const Real& b) {
  const std::complex<Real>& a_std = static_cast<const std::complex<Real>&>(a);
  return a_std / b;
}

template <class Real>
Complex<Real> operator/(const Real& a, const Complex<Real>& b) {
  const std::complex<Real>& b_std = static_cast<const std::complex<Real>&>(b);
  return a / b_std;
}

template <class Real>
Real RealPart(const Real& value) CATAMARI_NOEXCEPT {
  return value;
}

template <class Real>
Real RealPart(const Complex<Real>& value) CATAMARI_NOEXCEPT {
  return value.real();
}

template <class Real>
Real ImagPart(const Real& value) CATAMARI_NOEXCEPT {
  return 0;
}

template <class Real>
Real ImagPart(const Complex<Real>& value) CATAMARI_NOEXCEPT {
  return value.imag();
}

template <class Real>
Real Conjugate(const Real& value) CATAMARI_NOEXCEPT {
  return value;
}

template <class Real>
Complex<Real> Conjugate(const Complex<Real>& value) CATAMARI_NOEXCEPT {
  return Complex<Real>{value.real(), -value.imag()};
}

}  // namespace catamari

#endif  // ifndef CATAMARI_COMPLEX_IMPL_H_
