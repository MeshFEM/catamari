/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SUPERNODAL_LDL_IO_H_
#define CATAMARI_SUPERNODAL_LDL_IO_H_

#include <ostream>

namespace catamari {
namespace supernodal_ldl {

// Prints the diagonal factor of the LDL' factorization.
template <class Field>
void PrintDiagonalFactor(const Factorization<Field>& factorization,
                         const std::string& label, std::ostream& os);

// Prints the unit-diagonal lower-triangular factor of the LDL' factorization.
template <class Field>
void PrintLowerFactor(const Factorization<Field>& factorization,
                      const std::string& label, std::ostream& os);

}  // namespace supernodal_ldl
}  // namespace catamari

#include "catamari/supernodal_ldl/io-impl.hpp"

#endif  // ifndef CATAMARI_SUPERNODAL_LDL_IO_H_
