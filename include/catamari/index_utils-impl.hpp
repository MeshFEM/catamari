/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_INDEX_UTILS_IMPL_H_
#define CATAMARI_INDEX_UTILS_IMPL_H_

#include "catamari/index_utils.hpp"

namespace catamari {

inline void OffsetScan(const std::vector<Int>& sizes,
                       std::vector<Int>* offsets) {
  const Int num_indices = sizes.size();
  offsets->resize(num_indices + 1);

  Int offset = 0;
  for (Int index = 0; index < num_indices; ++index) {
    (*offsets)[index] = offset;
    offset += sizes[index];
  }
  (*offsets)[num_indices] = offset;
}

inline void InvertPermutation(const std::vector<Int>& permutation,
                              std::vector<Int>* inverse_permutation) {
  const Int num_indices = permutation.size();
  inverse_permutation->clear();
  inverse_permutation->resize(num_indices);
  for (Int index = 0; index < num_indices; ++index) {
    (*inverse_permutation)[permutation[index]] = index;
  }
}

}  // namespace catamari

#endif  // ifndef CATAMARI_INDEX_UTILS_IMPL_H_
