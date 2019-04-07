/*
 * Copyright (c) 2018-2019 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_HERMITIAN_DPP_SCALAR_IMPL_H_
#define CATAMARI_SPARSE_HERMITIAN_DPP_SCALAR_IMPL_H_

#include <algorithm>

#include "catamari/sparse_hermitian_dpp/scalar.hpp"

namespace catamari {

template <class Field>
ScalarHermitianDPP<Field>::ScalarHermitianDPP(
    const CoordinateMatrix<Field>& matrix, const SymmetricOrdering& ordering,
    const ScalarHermitianDPPControl& control)
    : matrix_(matrix), ordering_(ordering), control_(control) {
  UpLookingSetup();
}

template <class Field>
void ScalarHermitianDPP<Field>::UpLookingSetup() CATAMARI_NOEXCEPT {
  // TODO(Jack Poulson): Decide if/when the following should be parallelized.
  // The main cost tradeoffs are the need for the creation of the children in
  // the supernodal assembly forest and the additional memory allocations.
  lower_factor_.reset(new scalar_ldl::LowerFactor<Field>);
  diagonal_factor_.reset(new scalar_ldl::DiagonalFactor<Real>);

  Buffer<Int> degrees;
  scalar_ldl::EliminationForestAndDegrees(
      matrix_, ordering_, &ordering_.assembly_forest.parents, &degrees);

  scalar_ldl::LowerStructure& lower_structure = lower_factor_->structure;
  OffsetScan(degrees, &lower_structure.column_offsets);

  const Int num_rows = matrix_.NumRows();
  diagonal_factor_->values.Resize(num_rows);

  const Int num_entries = lower_structure.column_offsets.Back();
  lower_structure.indices.Resize(num_entries);
  lower_factor_->values.Resize(num_entries);
}

template <class Field>
void ScalarHermitianDPP<Field>::UpLookingRowUpdate(
    Int row, const Int* column_beg, const Int* column_end,
    Int* column_update_ptrs, Field* row_workspace) const CATAMARI_NOEXCEPT {
  scalar_ldl::LowerStructure& lower_structure = lower_factor_->structure;
  for (const Int* iter = column_beg; iter != column_end; ++iter) {
    const Int column = *iter;
    const Real pivot = diagonal_factor_->values[column];

    // Load eta := L(row, column) * d(column) from the workspace.
    const Field eta = row_workspace[column];
    row_workspace[column] = Field{0};

    // Update
    //
    //   L(row, I) -= (L(row, column) * d(column)) * conj(L(I, column)),
    //
    // where I is the set of already-formed entries in the structure of column
    // 'column' of L.
    //
    // Rothberg and Gupta refer to this as a 'scatter kernel' in:
    //   "An Evaluation of Left-Looking, Right-Looking and Multifrontal
    //   Approaches to Sparse Cholesky Factorization on Hierarchical-Memory
    //   Machines".
    const Int factor_column_beg = lower_structure.ColumnOffset(column);
    const Int factor_column_end = column_update_ptrs[column]++;
    for (Int index = factor_column_beg; index < factor_column_end; ++index) {
      // L(row, i) -= (L(row, column) * d(column)) * conj(L(i, column))
      const Int i = lower_structure.indices[index];
      const Field& value = lower_factor_->values[index];
      row_workspace[i] -= eta * Conjugate(value);
    }

    // Compute L(row, column) from eta = L(row, column) * d(column).
    const Field lambda = eta / pivot;

    // Append L(row, column) into the structure of column 'column'.
    lower_structure.indices[factor_column_end] = row;
    lower_factor_->values[factor_column_end] = lambda;

    // L(row, row) -= (L(row, column) * d(column)) * conj(L(row, column))
    diagonal_factor_->values[row] -= RealPart(eta * Conjugate(lambda));
  }
}

template <class Field>
std::vector<Int> ScalarHermitianDPP<Field>::UpLookingSample(
    bool maximum_likelihood) const {
  const Int num_rows = matrix_.NumRows();
  const Buffer<Int>& parents = ordering_.assembly_forest.parents;
  const scalar_ldl::LowerStructure& lower_structure = lower_factor_->structure;

  std::random_device random_device;
  UpLookingState state;
  scalar_ldl::UpLookingState<Field>& ldl_state = state.ldl_state;
  ldl_state.column_update_ptrs.Resize(num_rows);
  ldl_state.pattern_flags.Resize(num_rows);
  ldl_state.row_structure.Resize(num_rows);
  ldl_state.row_workspace.Resize(num_rows);
  state.generator.seed(random_device());

  std::uniform_real_distribution<Real> uniform_dist{Real{0}, Real{1}};

  std::vector<Int> sample;
  sample.reserve(num_rows);
  for (Int row = 0; row < num_rows; ++row) {
    ldl_state.pattern_flags[row] = row;
    ldl_state.column_update_ptrs[row] = lower_structure.ColumnOffset(row);

    // Compute the row pattern and scatter the row of the input matrix into
    // the workspace.
    const Int start =
        scalar_ldl::ComputeTopologicalRowPatternAndScatterNonzeros(
            matrix_, ordering_, parents, row, ldl_state.pattern_flags.Data(),
            ldl_state.row_structure.Data(), ldl_state.row_workspace.Data());

    // Pull the diagonal entry out of the workspace.
    diagonal_factor_->values[row] = ldl_state.row_workspace[row];
    ldl_state.row_workspace[row] = Field{0};

    // Compute L(row, :) using a sparse triangular solve. In particular,
    //   L(row, :) := matrix(row, :) / L(0 : row - 1, 0 : row - 1)'.
    UpLookingRowUpdate(row, ldl_state.row_structure.Data() + start,
                       ldl_state.row_structure.Data() + num_rows,
                       ldl_state.column_update_ptrs.Data(),
                       ldl_state.row_workspace.Data());

    // Early exit if solving would involve division by zero.
    Real pivot = diagonal_factor_->values[row];
    const bool keep_index = maximum_likelihood
                                ? pivot >= Real(1) / Real(2)
                                : uniform_dist(state.generator) <= pivot;
    if (keep_index) {
      sample.push_back(row);
    } else {
      pivot -= Real(1);
    }
    diagonal_factor_->values[row] = pivot;
  }

  std::sort(sample.begin(), sample.end());

  return sample;
}

template <class Field>
std::vector<Int> ScalarHermitianDPP<Field>::Sample(
    bool maximum_likelihood) const {
  return UpLookingSample(maximum_likelihood);
}

template <typename Field>
ComplexBase<Field> ScalarHermitianDPP<Field>::LogLikelihood() const {
  typedef ComplexBase<Field> Real;
  const Int num_values = diagonal_factor_->values.Size();

  Real log_likelihood = 0;
  for (Int j = 0; j < num_values; ++j) {
    log_likelihood += std::log(std::abs(diagonal_factor_->values[j]));
  }

  return log_likelihood;
}

}  // namespace catamari

#endif  // ifndef CATAMARI_SPARSE_HERMITIAN_DPP_SCALAR_IMPL_H_
