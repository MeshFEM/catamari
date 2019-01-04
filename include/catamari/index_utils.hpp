/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_INDEX_UTILS_H_
#define CATAMARI_INDEX_UTILS_H_

#include <vector>

#include "catamari/blas_matrix.hpp"
#include "catamari/integers.hpp"

namespace catamari {

// Fills 'offsets' with a length 'num_indices + 1' array whose i'th index is
// the sum of the sizes whose indices are less than i.
void OffsetScan(const std::vector<Int>& sizes, std::vector<Int>* offsets);

}  // namespace catamari

#include "catamari/index_utils-impl.hpp"

#endif  // ifndef CATAMARI_INDEX_UTILS_H_
