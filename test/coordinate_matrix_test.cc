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
#include "catch.hpp"

TEST_CASE("Basic", "[Basic]") {
  catamari::CoordinateMatrix<float> matrix;
  matrix.Resize(5, 5);
  matrix.ReserveEntryAdditions(6);
  matrix.QueueEntryAddition(3, 4, 1.f);
  matrix.QueueEntryAddition(2, 3, 2.f); 
  matrix.QueueEntryAddition(2, 0, -1.f);
  matrix.QueueEntryAddition(4, 2, -2.f);
  matrix.QueueEntryAddition(4, 4, 3.f);
  matrix.QueueEntryAddition(3, 2, 4.f);
  matrix.FlushEntryQueues();
  const std::vector<catamari::MatrixEntry<float>>& entries =
      matrix.Entries();

  const std::vector<catamari::MatrixEntry<float>> expected_entries{
      {2, 0, -1.f},
      {2, 3, 2.f},
      {3, 2, 4.f},
      {3, 4, 1.f},
      {4, 2, -2.f},
      {4, 4, 3.f},
  };

  REQUIRE(entries == expected_entries);

  matrix.ReserveEntryRemovals(2);
  matrix.QueueEntryRemoval(2, 3);
  matrix.QueueEntryRemoval(0, 4);
  matrix.FlushEntryQueues();

  const std::vector<catamari::MatrixEntry<float>> new_expected_entries{
      {2, 0, -1.f},
      {3, 2, 4.f},
      {3, 4, 1.f},
      {4, 2, -2.f},
      {4, 4, 3.f},
  };

  REQUIRE(entries == new_expected_entries);
}
