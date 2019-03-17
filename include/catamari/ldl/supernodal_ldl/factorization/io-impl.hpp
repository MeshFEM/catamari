/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_IO_IMPL_H_
#define CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_IO_IMPL_H_

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"

#include "catamari/ldl/supernodal_ldl/factorization.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
void Factorization<Field>::PrintDiagonalFactor(const std::string& label,
                                               std::ostream& os) const {
  os << label << ": \n";
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const ConstBlasMatrixView<Field>& diag_matrix =
        diagonal_factor_->blocks[supernode];
    if (control_.factorization_type == kCholeskyFactorization) {
      for (Int j = 0; j < diag_matrix.height; ++j) {
        os << "1 ";
      }
    } else {
      for (Int j = 0; j < diag_matrix.height; ++j) {
        os << diag_matrix(j, j) << " ";
      }
    }
  }
  os << std::endl;
}

template <class Field>
void Factorization<Field>::PrintLowerFactor(const std::string& label,
                                            std::ostream& os) const {
  const bool is_cholesky =
      control_.factorization_type == kCholeskyFactorization;

  auto print_entry = [&](const Int& row, const Int& column,
                         const Field& value) {
    os << row << " " << column << " " << value << "\n";
  };

  os << label << ": \n";
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int supernode_start = ordering_.supernode_offsets[supernode];
    const Int* indices = lower_factor_->StructureBeg(supernode);

    const ConstBlasMatrixView<Field>& diag_matrix =
        diagonal_factor_->blocks[supernode];
    const ConstBlasMatrixView<Field>& lower_matrix =
        lower_factor_->blocks[supernode];

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

#endif  // ifndef CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_IO_IMPL_H_
