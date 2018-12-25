/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#define CATCH_CONFIG_MAIN
#include <vector>
#include "catamari.hpp"
#include "catch2/catch.hpp"

TEST_CASE("IsComplex", "[IsComplex]") {
  REQUIRE(catamari::IsComplex<int>::value == false);
  REQUIRE(catamari::IsComplex<float>::value == false);
  REQUIRE(catamari::IsComplex<double>::value == false);
  REQUIRE(catamari::IsComplex<catamari::Complex<float>>::value == true);
  REQUIRE(catamari::IsComplex<catamari::Complex<double>>::value == true);
}

TEST_CASE("RealPart", "[RealPart]") {
  REQUIRE(catamari::RealPart(17.) == 17.);
  REQUIRE(catamari::ImagPart(17.) == 0.);
  REQUIRE(catamari::RealPart(catamari::Complex<double>{17., 18.}) == 17.);
  REQUIRE(catamari::ImagPart(catamari::Complex<double>{17., 18.}) == 18.);
}
