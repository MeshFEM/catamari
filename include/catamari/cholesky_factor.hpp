/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_CHOLESKY_FACTOR_H_
#define CATAMARI_CHOLESKY_FACTOR_H_

#include "catamari/coordinate_matrix.hpp"
#include "catamari/integers.hpp"

namespace catamari {

template<class Field>
struct CholeskyFactor {
  // A vector of length 'num_supernodes' whose i'th value is the cardinality of
  // the i'th supernode.
  std::vector<Int> supernode_sizes;

  // A vector of length 'num_supernodes' whose i'th value is the principal
  // member of the i'th supernode.
  std::vector<Int> supernode_principals;

  // A vector of length 'num_vertices' whose j'th value is the index of the
  // supernode containing node j.
  // NOTE: The index of the supernode is *not* the same as its principal member.
  std::vector<Int> supernode_index;

  // The nonzero indices of the lower-triangular Cholesky factor for each
  // supernode.
  std::vector<std::vector<Int>> structures;

  // The nonzeros in each column of the lower-triangular Cholesky factor.
  std::vector<std::vector<Field>> nonzeros;
};

/*
template<class Field>
void InitializeCholeskyFactor(
    const CoordinateMatrix<Field>& matrix,
    const quotient::MinimumDegreeResult& analysis,
    CholeskyFactor<Field>* factor) {
}
*/

} // namespace catamari

#endif // ifndef CATAMARI_CHOLESKY_FACTOR_H_
