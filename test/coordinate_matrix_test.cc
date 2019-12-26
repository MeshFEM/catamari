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

TEST_CASE("Basic", "[Basic]") {
  catamari::CoordinateMatrix<float> matrix;

  matrix.Resize(5, 5);
  matrix.ReserveEntryAdditions(6);
  matrix.QueueEntryAdditions(std::vector<catamari::MatrixEntry<float>>{
      {3, 4, 1.f},
      {2, 3, 2.f},
      {2, 0, -1.f},
      {4, 2, -2.f},
      {4, 4, 3.f},
      {3, 2, 4.f},
  });
  matrix.FlushEntryQueues();
  const catamari::Buffer<catamari::MatrixEntry<float>>& entries =
      matrix.Entries();
  const catamari::Buffer<catamari::MatrixEntry<float>> expected_entries{
      {2, 0, -1.f}, {2, 3, 2.f},  {3, 2, 4.f},
      {3, 4, 1.f},  {4, 2, -2.f}, {4, 4, 3.f},
  };
  REQUIRE(entries == expected_entries);
  REQUIRE(matrix.Value(0, 0) == 0.f);
  REQUIRE(matrix.Value(0, 1) == 0.f);
  REQUIRE(matrix.Value(0, 2) == 0.f);
  REQUIRE(matrix.Value(0, 3) == 0.f);
  REQUIRE(matrix.Value(0, 4) == 0.f);
  REQUIRE(matrix.Value(1, 0) == 0.f);
  REQUIRE(matrix.Value(1, 1) == 0.f);
  REQUIRE(matrix.Value(2, 0) == -1.f);
  REQUIRE(matrix.Value(2, 1) == 0.f);
  REQUIRE(matrix.Value(2, 2) == 0.f);
  REQUIRE(matrix.Value(2, 3) == 2.f);
  REQUIRE(matrix.Value(2, 4) == 0.f);
  REQUIRE(matrix.Value(3, 0) == 0.f);
  REQUIRE(matrix.Value(3, 3) == 0.f);
  REQUIRE(matrix.Value(3, 4) == 1.f);
  REQUIRE(matrix.Value(3, 5) == 0.f);
  REQUIRE(matrix.Value(4, 4) == 3.f);

  matrix.ReserveEntryRemovals(2);
  matrix.QueueEntryRemoval(2, 3);
  matrix.QueueEntryRemoval(0, 4);
  matrix.FlushEntryQueues();
  const catamari::Buffer<catamari::MatrixEntry<float>> update0_entries{
      {2, 0, -1.f}, {3, 2, 4.f}, {3, 4, 1.f}, {4, 2, -2.f}, {4, 4, 3.f},
  };
  REQUIRE(entries == update0_entries);

  matrix.ReserveEntryAdditions(5);
  for (quotient::Int i = 0; i < 5; ++i) {
    matrix.QueueEntryAddition(i, i, 10.f);
  }
  matrix.FlushEntryQueues();
  const catamari::Buffer<catamari::MatrixEntry<float>> update1_entries{
      {0, 0, 10.f}, {1, 1, 10.f}, {2, 0, -1.f}, {2, 2, 10.f}, {3, 2, 4.f},
      {3, 3, 10.f}, {3, 4, 1.f},  {4, 2, -2.f}, {4, 4, 13.f},
  };
  REQUIRE(entries == update1_entries);

  matrix.ReplaceEntry(2, 0, 3.7f);
  matrix.ReplaceEntry(3, 3, -9.f);
  const catamari::Buffer<catamari::MatrixEntry<float>> update2_entries{
      {0, 0, 10.f}, {1, 1, 10.f}, {2, 0, 3.7f}, {2, 2, 10.f}, {3, 2, 4.f},
      {3, 3, -9.f}, {3, 4, 1.f},  {4, 2, -2.f}, {4, 4, 13.f},
  };
  REQUIRE(entries == update2_entries);

  matrix.AddToEntry(3, 2, 1.f);
  matrix.AddToEntry(4, 4, -2.f);
  const catamari::Buffer<catamari::MatrixEntry<float>> update3_entries{
      {0, 0, 10.f}, {1, 1, 10.f}, {2, 0, 3.7f}, {2, 2, 10.f}, {3, 2, 5.f},
      {3, 3, -9.f}, {3, 4, 1.f},  {4, 2, -2.f}, {4, 4, 11.f},
  };
  REQUIRE(entries == update3_entries);
}
