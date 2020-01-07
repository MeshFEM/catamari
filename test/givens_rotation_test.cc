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

TEST_CASE("TrivialGivens float", "[TrivialGivens float]") {
  catamari::GivensRotation<float> rotation;

  const float alpha = 3.f;
  const float beta = 4.f;
  const float combined_entry = rotation.Generate(alpha, beta);

  const float combined_entry_target = 5.f;
  const float combined_entry_deviation =
      std::abs(combined_entry - combined_entry_target) /
      std::abs(combined_entry_target);
  REQUIRE(combined_entry_deviation <= 1e-5f);

  float rotated_alpha = alpha;
  float rotated_beta = beta;
  rotation.Apply(&rotated_alpha, &rotated_beta);
  const float rotated_alpha_deviation =
      std::abs(rotated_alpha - combined_entry_target) /
      std::abs(combined_entry_target);
  REQUIRE(rotated_alpha_deviation <= 1e-5f);
  const float rotated_beta_deviation = std::abs(rotated_beta) / std::abs(beta);
  REQUIRE(rotated_beta_deviation <= 1e-5f);
}

TEST_CASE("SmallGivens float", "[SmallGivens float]") {
  catamari::GivensRotation<float> rotation;

  const float alpha = 3e-30f;
  const float beta = 4e-30f;
  const float combined_entry = rotation.Generate(alpha, beta);

  const float combined_entry_target = 5e-30f;
  const float combined_entry_deviation =
      std::abs(combined_entry - combined_entry_target) /
      std::abs(combined_entry_target);
  REQUIRE(combined_entry_deviation <= 1e-5f);

  float rotated_alpha = alpha;
  float rotated_beta = beta;
  rotation.Apply(&rotated_alpha, &rotated_beta);
  const float rotated_alpha_deviation =
      std::abs(rotated_alpha - combined_entry_target) /
      std::abs(combined_entry_target);
  REQUIRE(rotated_alpha_deviation <= 1e-5f);
  const float rotated_beta_deviation = std::abs(rotated_beta) / std::abs(beta);
  REQUIRE(rotated_beta_deviation <= 1e-5f);
}

TEST_CASE("LargeGivens float", "[LargeGivens float]") {
  catamari::GivensRotation<float> rotation;

  const float alpha = 3e30f;
  const float beta = 4e30f;
  const float combined_entry = rotation.Generate(alpha, beta);

  const float combined_entry_target = 5e30f;
  const float combined_entry_deviation =
      std::abs(combined_entry - combined_entry_target) /
      std::abs(combined_entry_target);
  REQUIRE(combined_entry_deviation <= 1e-5f);

  float rotated_alpha = alpha;
  float rotated_beta = beta;
  rotation.Apply(&rotated_alpha, &rotated_beta);
  const float rotated_alpha_deviation =
      std::abs(rotated_alpha - combined_entry_target) /
      std::abs(combined_entry_target);
  REQUIRE(rotated_alpha_deviation <= 1e-5f);
  const float rotated_beta_deviation = std::abs(rotated_beta) / std::abs(beta);
  REQUIRE(rotated_beta_deviation <= 1e-5f);
}

TEST_CASE("TrivialGivens Complex<float>", "[TrivialGivens Complex<float>]") {
  catamari::GivensRotation<catamari::Complex<float>> rotation;

  const catamari::Complex<float> alpha(0.f, 3.f);
  const catamari::Complex<float> beta(4.f);
  const catamari::Complex<float> combined_entry =
      rotation.Generate(alpha, beta);

  const catamari::Complex<float> combined_entry_target(0.f, 5.f);
  const float combined_entry_deviation =
      std::abs(combined_entry - combined_entry_target) /
      std::abs(combined_entry_target);
  REQUIRE(combined_entry_deviation <= 1e-5f);

  catamari::Complex<float> rotated_alpha = alpha;
  catamari::Complex<float> rotated_beta = beta;
  rotation.Apply(&rotated_alpha, &rotated_beta);
  const float rotated_alpha_deviation =
      std::abs(rotated_alpha - combined_entry_target) /
      std::abs(combined_entry_target);
  REQUIRE(rotated_alpha_deviation <= 1e-5f);
  const float rotated_beta_deviation = std::abs(rotated_beta) / std::abs(beta);
  REQUIRE(rotated_beta_deviation <= 1e-5f);
}

TEST_CASE("SmallGivens Complex<float>", "[SmallGivens Complex<float>]") {
  catamari::GivensRotation<catamari::Complex<float>> rotation;

  const catamari::Complex<float> alpha(0.f, 3e-30f);
  const catamari::Complex<float> beta(4e-30f);
  const catamari::Complex<float> combined_entry =
      rotation.Generate(alpha, beta);

  const catamari::Complex<float> combined_entry_target(0.f, 5e-30f);
  const float combined_entry_deviation =
      std::abs(combined_entry - combined_entry_target) /
      std::abs(combined_entry_target);
  REQUIRE(combined_entry_deviation <= 1e-5f);

  catamari::Complex<float> rotated_alpha = alpha;
  catamari::Complex<float> rotated_beta = beta;
  rotation.Apply(&rotated_alpha, &rotated_beta);
  const float rotated_alpha_deviation =
      std::abs(rotated_alpha - combined_entry_target) /
      std::abs(combined_entry_target);
  REQUIRE(rotated_alpha_deviation <= 1e-5f);
  const float rotated_beta_deviation = std::abs(rotated_beta) / std::abs(beta);
  REQUIRE(rotated_beta_deviation <= 1e-5f);
}

TEST_CASE("SmallAlphaGivens Complex<float>", "[SmallGivens Complex<float>]") {
  catamari::GivensRotation<catamari::Complex<float>> rotation;

  const catamari::Complex<float> alpha(0.f, 3e-30f);
  const catamari::Complex<float> beta(0.f, 4.f);
  const catamari::Complex<float> combined_entry =
      rotation.Generate(alpha, beta);

  const catamari::Complex<float> combined_entry_target(0.f, 4.f);
  const float combined_entry_deviation =
      std::abs(combined_entry - combined_entry_target) /
      std::abs(combined_entry_target);
  REQUIRE(combined_entry_deviation <= 1e-5f);

  catamari::Complex<float> rotated_alpha = alpha;
  catamari::Complex<float> rotated_beta = beta;
  rotation.Apply(&rotated_alpha, &rotated_beta);
  const float rotated_alpha_deviation =
      std::abs(rotated_alpha - combined_entry_target) /
      std::abs(combined_entry_target);
  REQUIRE(rotated_alpha_deviation <= 1e-5f);
  const float rotated_beta_deviation = std::abs(rotated_beta) / std::abs(beta);
  REQUIRE(rotated_beta_deviation <= 1e-5f);
}

TEST_CASE("ZeroAlphaGivens Complex<float>", "[SmallGivens Complex<float>]") {
  catamari::GivensRotation<catamari::Complex<float>> rotation;

  const catamari::Complex<float> alpha(0.f);
  const catamari::Complex<float> beta(0.f, 4.f);
  const catamari::Complex<float> combined_entry =
      rotation.Generate(alpha, beta);

  const catamari::Complex<float> combined_entry_target(4.f);
  const float combined_entry_deviation =
      std::abs(combined_entry - combined_entry_target) /
      std::abs(combined_entry_target);
  REQUIRE(combined_entry_deviation <= 1e-5f);

  catamari::Complex<float> rotated_alpha = alpha;
  catamari::Complex<float> rotated_beta = beta;
  rotation.Apply(&rotated_alpha, &rotated_beta);
  const float rotated_alpha_deviation =
      std::abs(rotated_alpha - combined_entry_target) /
      std::abs(combined_entry_target);
  REQUIRE(rotated_alpha_deviation <= 1e-5f);
  const float rotated_beta_deviation = std::abs(rotated_beta) / std::abs(beta);
  REQUIRE(rotated_beta_deviation <= 1e-5f);
}

TEST_CASE("LargeGivens Complex<float>", "[LargeGivens Complex<float>]") {
  catamari::GivensRotation<catamari::Complex<float>> rotation;

  const catamari::Complex<float> alpha(0.f, 3e30f);
  const catamari::Complex<float> beta(4e30f);
  const catamari::Complex<float> combined_entry =
      rotation.Generate(alpha, beta);

  const catamari::Complex<float> combined_entry_target(0.f, 5e30f);
  const float combined_entry_deviation =
      std::abs(combined_entry - combined_entry_target) /
      std::abs(combined_entry_target);
  REQUIRE(combined_entry_deviation <= 1e-5f);

  catamari::Complex<float> rotated_alpha = alpha;
  catamari::Complex<float> rotated_beta = beta;
  rotation.Apply(&rotated_alpha, &rotated_beta);
  const float rotated_alpha_deviation =
      std::abs(rotated_alpha - combined_entry_target) /
      std::abs(combined_entry_target);
  REQUIRE(rotated_alpha_deviation <= 1e-5f);
  const float rotated_beta_deviation = std::abs(rotated_beta) / std::abs(beta);
  REQUIRE(rotated_beta_deviation <= 1e-5f);
}
