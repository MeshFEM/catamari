/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SUPERNODAL_LDL_IO_IMPL_H_
#define CATAMARI_LDL_SUPERNODAL_LDL_IO_IMPL_H_

#include "catamari/ldl/supernodal_ldl/io.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
void PrintDiagonalFactor(const Factorization<Field>& factorization,
                         const std::string& label, std::ostream& os) {
  const DiagonalFactor<Field>& diag_factor = *factorization.diagonal_factor;
  if (factorization.factorization_type == kCholeskyFactorization) {
    // TODO(Jack Poulson): Print the identity.
    return;
  }

  os << label << ": \n";
  const Int num_supernodes = factorization.supernode_sizes.size();
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const ConstBlasMatrix<Field>& diag_matrix = diag_factor.blocks[supernode];
    for (Int j = 0; j < diag_matrix.height; ++j) {
      os << diag_matrix(j, j) << " ";
    }
  }
  os << std::endl;
}

template <class Field>
void PrintLowerFactor(const Factorization<Field>& factorization,
                      const std::string& label, std::ostream& os) {
  const LowerFactor<Field>& lower_factor = *factorization.lower_factor;
  const DiagonalFactor<Field>& diag_factor = *factorization.diagonal_factor;
  const bool is_cholesky =
      factorization.factorization_type == kCholeskyFactorization;

  auto print_entry = [&](const Int& row, const Int& column,
                         const Field& value) {
    os << row << " " << column << " " << value << "\n";
  };

  os << label << ": \n";
  const Int num_supernodes = factorization.supernode_sizes.size();
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int supernode_start = factorization.supernode_starts[supernode];
    const Int* indices = lower_factor.Structure(supernode);

    const ConstBlasMatrix<Field>& diag_matrix = diag_factor.blocks[supernode];
    const ConstBlasMatrix<Field>& lower_matrix = lower_factor.blocks[supernode];

    for (Int j = 0; j < diag_matrix.height; ++j) {
      const Int column = supernode_start + j;

      // Print the portion in the diagonal block.
      if (is_cholesky) {
        print_entry(column, column, diag_matrix(j, j));
      } else {
        print_entry(column, column, Field{1});
      }
      for (Int k = j + 1; k < diag_matrix.height; ++k) {
        const Int row = supernode_start + k;
        print_entry(row, column, diag_matrix(k, j));
      }

      // Print the portion below the diagonal block.
      for (Int i = 0; i < lower_matrix.height; ++i) {
        const Int row = indices[i];
        print_entry(row, column, lower_matrix(i, j));
      }
    }
  }
  os << std::endl;
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef CATAMARI_LDL_SUPERNODAL_LDL_IO_IMPL_H_
