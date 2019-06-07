/*
 * Copyright (c) 2018-2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SCALAR_FACTORIZATION_IO_IMPL_H_
#define CATAMARI_SPARSE_LDL_SCALAR_FACTORIZATION_IO_IMPL_H_

#include <cmath>

#include "catamari/index_utils.hpp"
#include "catamari/sparse_ldl/scalar/scalar_utils.hpp"
#include "quotient/io_utils.hpp"

#include "catamari/sparse_ldl/scalar/factorization.hpp"

namespace catamari {
namespace scalar_ldl {

template <class Field>
void Factorization<Field>::PrintLowerFactor(const std::string& label,
                                            std::ostream& os) const {
  const LowerStructure& lower_structure = lower_factor.structure;

  auto print_entry = [&](const Int& row, const Int& column,
                         const Field& value) {
    os << row << " " << column << " " << value << "\n";
  };

  os << label << ":\n";
  const Int num_columns = lower_structure.column_offsets.Size() - 1;
  for (Int column = 0; column < num_columns; ++column) {
    if (factorization_type == kCholeskyFactorization) {
      print_entry(column, column, diagonal_factor.values[column]);
    } else {
      print_entry(column, column, Field{1});
    }

    const Int column_beg = lower_structure.ColumnOffset(column);
    const Int column_end = lower_structure.ColumnOffset(column + 1);
    for (Int index = column_beg; index < column_end; ++index) {
      const Int row = lower_structure.indices[index];
      const Field& value = lower_factor.values[index];
      print_entry(row, column, value);
    }
  }
  os << std::endl;
}

template <class Field>
void Factorization<Field>::PrintDiagonalFactor(const std::string& label,
                                               std::ostream& os) const {
  if (factorization_type == kCholeskyFactorization) {
    // TODO(Jack Poulson): Print the identity.
    return;
  }
  quotient::Print(diagonal_factor.values, label, os);
}

}  // namespace scalar_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_SPARSE_LDL_SCALAR_FACTORIZATION_IO_IMPL_H_
