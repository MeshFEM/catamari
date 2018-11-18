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

#include "catamari/enable_if.hpp"
#include "catamari/macros.hpp"

namespace catamari {

template<typename Real>
using Complex = std::complex<Real>;

namespace complex_base {

template<class Real>
struct ComplexBaseHelper { typedef Real type; };

template<class Real>
struct ComplexBaseHelper<Complex<Real>> { typedef Real type; };

} // namespace complex_base

// Returns the type of the base field of a real or complex scalar. For example:
//   ComplexBase<double> == double
//   ComplexBase<Complex<double>> == double.
template<typename Field>
using ComplexBase = typename complex_base::ComplexBaseHelper<Field>::type;

// Encodes whether or not a given type is complex. For example,
//   IsComplex<double>::value == false
//   IsComplex<Complex<double>>::value == true
template<class Real>
struct IsComplex { static constexpr bool value = false; };

template<class Real>
struct IsComplex<Complex<Real>> { static constexpr bool value = true; };

// Returns the real part of a real scalar.
template<class Real>
Real RealPart(const Real& value) CATAMARI_NOEXCEPT;

// Returns the real part of a complex scalar.
template<class Real>
Real RealPart(const Complex<Real>& value) CATAMARI_NOEXCEPT;

// Returns the imaginary part of a real scalar (zero).
template<class Real>
Real ImagPart(const Real& value) CATAMARI_NOEXCEPT;

// Returns the imaginary part of a complex scalar.
template<class Real>
Real ImagPart(const Complex<Real>& value) CATAMARI_NOEXCEPT;

// Returns the complex-conjugate of a real value (the value itself).
template<class Real>
Real Conjugate(const Real& value) CATAMARI_NOEXCEPT;

// Returns the complex-conjugate of a complex value.
template<class Real>
Complex<Real> Conjugate(const Complex<Real>& value) CATAMARI_NOEXCEPT;

} // namespace catamari

#include "catamari/complex-impl.hpp"

#endif // ifndef CATAMARI_COMPLEX_H_
