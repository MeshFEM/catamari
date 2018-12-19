/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SUPERNODAL_DPP_IMPL_H_
#define CATAMARI_SUPERNODAL_DPP_IMPL_H_

#include "catamari/supernodal_dpp.hpp"

namespace catamari {

template <class Field>
SupernodalDPP<Field>::SupernodalDPP(const CoordinateMatrix<Field>& matrix,
                                    const std::vector<Int>& permutation,
                                    const std::vector<Int>& inverse_permutation,
                                    const SupernodalDPPControl& control,
                                    unsigned int random_seed)
    : matrix_(matrix),
      permutation_(permutation),
      inverse_permutation_(inverse_permutation),
      control_(control),
      generator_(random_seed),
      unit_uniform_(ComplexBase<Field>{0}, ComplexBase<Field>{1}) {
  FormSupernodes();
  FormStructure();
}

template <class Field>
void SupernodalDPP<Field>::FormSupernodes() {
  std::vector<Int> orig_parents, orig_degrees;
  ldl::EliminationForestAndDegrees(matrix_, permutation_, inverse_permutation_,
                                   &orig_parents, &orig_degrees);

  std::vector<Int> orig_supernode_sizes;
  ScalarLowerStructure scalar_structure;
  supernodal_ldl::FormFundamentalSupernodes(
      matrix_, permutation_, inverse_permutation_, orig_parents, orig_degrees,
      &orig_supernode_sizes, &scalar_structure);

  std::vector<Int> orig_supernode_starts;
  ldl::OffsetScan(orig_supernode_sizes, &orig_supernode_starts);

  std::vector<Int> orig_member_to_index;
  supernodal_ldl::MemberToIndex(matrix_.NumRows(), orig_supernode_starts,
                                &orig_member_to_index);

  std::vector<Int> orig_supernode_degrees;
  supernodal_ldl::SupernodalDegrees(matrix_, permutation_, inverse_permutation_,
                                    orig_supernode_sizes, orig_supernode_starts,
                                    orig_member_to_index, orig_parents,
                                    &orig_supernode_degrees);

  const Int num_orig_supernodes = orig_supernode_sizes.size();
  std::vector<Int> orig_supernode_parents;
  supernodal_ldl::ConvertFromScalarToSupernodalEliminationForest(
      num_orig_supernodes, orig_parents, orig_member_to_index,
      &orig_supernode_parents);

  if (control_.relaxation_control.relax_supernodes) {
    supernodal_ldl::RelaxSupernodes(
        orig_parents, orig_supernode_sizes, orig_supernode_starts,
        orig_supernode_parents, orig_supernode_degrees, orig_member_to_index,
        scalar_structure, control_.relaxation_control, &permutation_,
        &inverse_permutation_, &parents_, &supernode_parents_,
        &supernode_degrees_, &supernode_sizes_, &supernode_starts_,
        &supernode_member_to_index_);
  } else {
    parents_ = orig_parents;
    supernode_parents_ = orig_supernode_parents;
    supernode_degrees_ = orig_supernode_degrees;
    supernode_sizes_ = orig_supernode_sizes;
    supernode_starts_ = orig_supernode_starts;
    supernode_member_to_index_ = orig_member_to_index;
  }
}

template <class Field>
void SupernodalDPP<Field>::FormStructure() {
  const Int num_supernodes = supernode_sizes_.size();
  CATAMARI_ASSERT(static_cast<Int>(supernode_degrees_.size()) == num_supernodes,
                  "Invalid supernode degrees size.");

  // Set up the column offsets and allocate space (initializing the values of
  // the unit-lower and diagonal and all zeros).
  lower_factor_.index_offsets.resize(num_supernodes + 1);
  lower_factor_.value_offsets.resize(num_supernodes + 1);
  diagonal_factor_.value_offsets.resize(num_supernodes + 1);
  Int degree_sum = 0;
  Int num_entries = 0;
  Int num_diagonal_entries = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    lower_factor_.index_offsets[supernode] = degree_sum;
    lower_factor_.value_offsets[supernode] = num_entries;
    diagonal_factor_.value_offsets[supernode] = num_diagonal_entries;

    const Int degree = supernode_degrees_[supernode];
    const Int supernode_size = supernode_sizes_[supernode];
    degree_sum += degree;
    num_entries += degree * supernode_size;
    num_diagonal_entries += supernode_size * supernode_size;
  }
  lower_factor_.index_offsets[num_supernodes] = degree_sum;
  lower_factor_.value_offsets[num_supernodes] = num_entries;
  diagonal_factor_.value_offsets[num_supernodes] = num_diagonal_entries;

  lower_factor_.indices.resize(degree_sum);
  lower_factor_.values.resize(num_entries);
  diagonal_factor_.values.resize(num_diagonal_entries);

  supernodal_ldl::FillStructureIndices(
      matrix_, permutation_, inverse_permutation_, parents_, supernode_sizes_,
      supernode_member_to_index_, &lower_factor_);
  supernodal_ldl::FillIntersections(supernode_sizes_,
                                    supernode_member_to_index_, &lower_factor_);
}

template <class Field>
std::vector<Int> SupernodalDPP<Field>::Sample() const {
  return LeftLookingSample();
}

template <class Field>
std::vector<Int> SupernodalDPP<Field>::LeftLookingSample() const {
  const Int num_rows = supernode_starts_.back();
  const Int num_supernodes = supernode_sizes_.size();
  const bool is_cholesky = false;

  std::fill(lower_factor_.values.begin(), lower_factor_.values.end(), Field{0});
  std::fill(diagonal_factor_.values.begin(), diagonal_factor_.values.end(),
            Field{0});

  supernodal_ldl::FillNonzeros(matrix_, permutation_, inverse_permutation_,
                               supernode_starts_, supernode_sizes_,
                               supernode_member_to_index_, &lower_factor_,
                               &diagonal_factor_);

  std::vector<Int> sample;
  sample.reserve(num_rows);

  // Set up a buffer for supernodal updates.
  Int max_supernode_size = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    max_supernode_size =
        std::max(max_supernode_size, supernode_sizes_[supernode]);
  }
  std::vector<Field> scaled_adjoint_buffer(
      max_supernode_size * max_supernode_size, Field{0});
  std::vector<Field> update_buffer(
      max_supernode_size * (max_supernode_size - 1), Field{0});

  // A data structure for marking whether or not a supernode is in the pattern
  // of the active row of the lower-triangular factor.
  std::vector<Int> pattern_flags(num_supernodes);

  // An integer workspace for storing the supernodes in the current row
  // pattern.
  std::vector<Int> row_structure(num_supernodes);

  // Since we will sequentially access each of the entries in each block column
  // of  L during the updates of a supernode, we can avoid the need for binary
  // search by maintaining a separate counter for each supernode.
  std::vector<Int> intersect_ptrs(num_supernodes);
  std::vector<Int> index_ptrs(num_supernodes);
  std::vector<Int> value_ptrs(num_supernodes);

  for (Int main_supernode = 0; main_supernode < num_supernodes;
       ++main_supernode) {
    const Int main_supernode_size = supernode_sizes_[main_supernode];
    const Int main_supernode_start = supernode_starts_[main_supernode];

    pattern_flags[main_supernode] = main_supernode;

    intersect_ptrs[main_supernode] =
        lower_factor_.intersect_size_offsets[main_supernode];
    index_ptrs[main_supernode] = lower_factor_.index_offsets[main_supernode];
    value_ptrs[main_supernode] = lower_factor_.value_offsets[main_supernode];

    // Compute the supernodal row pattern.
    const Int num_packed = supernodal_ldl::ComputeRowPattern(
        matrix_, permutation_, inverse_permutation_, supernode_sizes_,
        supernode_starts_, supernode_member_to_index_, supernode_parents_,
        main_supernode, pattern_flags.data(), row_structure.data());

    // for J = find(L(K, :))
    //   L(K:n, K) -= L(K:n, J) * (D(J, J) * L(K, J)')
    for (Int index = 0; index < num_packed; ++index) {
      const Int descendant_supernode = row_structure[index];
      CATAMARI_ASSERT(descendant_supernode < main_supernode,
                      "Looking into upper triangle.");
      const Int descendant_supernode_size =
          supernode_sizes_[descendant_supernode];

      const Int descendant_main_intersect_size_beg =
          intersect_ptrs[descendant_supernode];
      const Int descendant_main_index_beg = index_ptrs[descendant_supernode];
      const Int descendant_main_value_beg = value_ptrs[descendant_supernode];
      const Int descendant_main_intersect_size =
          lower_factor_.intersect_sizes[descendant_main_intersect_size_beg];

      supernodal_ldl::FormScaledAdjoint(
          is_cholesky, supernode_sizes_, lower_factor_, diagonal_factor_,
          descendant_supernode, descendant_main_intersect_size,
          descendant_main_value_beg, scaled_adjoint_buffer.data());

      supernodal_ldl::UpdateDiagonalBlock(
          is_cholesky, supernode_starts_, supernode_sizes_, lower_factor_,
          main_supernode, descendant_supernode, descendant_main_intersect_size,
          descendant_main_index_beg, descendant_main_value_beg,
          scaled_adjoint_buffer.data(), &diagonal_factor_,
          update_buffer.data());

      intersect_ptrs[descendant_supernode]++;
      index_ptrs[descendant_supernode] += descendant_main_intersect_size;
      value_ptrs[descendant_supernode] +=
          descendant_main_intersect_size * descendant_supernode_size;

      // L(KNext:n, K) -= L(KNext:n, J) * (D(J, J) * L(K, J)')
      //                = L(KNext:n, J) * Z(J, K).
      Int descendant_active_intersect_size_beg =
          intersect_ptrs[descendant_supernode];
      Int descendant_active_index_beg = index_ptrs[descendant_supernode];
      Int descendant_active_value_beg = value_ptrs[descendant_supernode];
      Int main_active_intersect_size_beg =
          lower_factor_.intersect_size_offsets[main_supernode];
      Int main_active_index_beg = lower_factor_.index_offsets[main_supernode];
      Int main_active_value_beg = lower_factor_.value_offsets[main_supernode];
      const Int descendant_index_guard =
          lower_factor_.index_offsets[descendant_supernode + 1];
      while (descendant_active_index_beg != descendant_index_guard) {
        const Int descendant_active_intersect_size =
            lower_factor_.intersect_sizes[descendant_active_intersect_size_beg];
        supernodal_ldl::UpdateSubdiagonalBlock(
            supernode_starts_, supernode_sizes_, supernode_member_to_index_,
            main_supernode, descendant_supernode,
            descendant_main_intersect_size, descendant_main_index_beg,
            descendant_active_intersect_size, descendant_active_index_beg,
            descendant_active_value_beg, scaled_adjoint_buffer.data(),
            &lower_factor_, &main_active_intersect_size_beg,
            &main_active_index_beg, &main_active_value_beg,
            update_buffer.data());

        ++descendant_active_intersect_size_beg;
        descendant_active_index_beg += descendant_active_intersect_size;
        descendant_active_value_beg +=
            descendant_active_intersect_size * descendant_supernode_size;
      }
    }

    // Replace with a dense factorization/sampler.
    {
      BlasMatrix<Field> diagonal_block;
      diagonal_block.height = main_supernode_size;
      diagonal_block.width = main_supernode_size;
      diagonal_block.leading_dim = main_supernode_size;
      diagonal_block.data =
          &diagonal_factor_
               .values[diagonal_factor_.value_offsets[main_supernode]];

      const std::vector<Int> supernode_sample =
          LowerFactorAndSampleDPP(&diagonal_block, &generator_, &unit_uniform_);
      for (const Int& index : supernode_sample) {
        const Int orig_row = main_supernode_start + index;
        if (inverse_permutation_.empty()) {
          sample.push_back(orig_row);
        } else {
          sample.push_back(inverse_permutation_[orig_row]);
        }
      }
    }

    supernodal_ldl::SolveAgainstDiagonalBlock(is_cholesky, main_supernode,
                                              supernode_sizes_,
                                              diagonal_factor_, &lower_factor_);
  }

  std::sort(sample.begin(), sample.end());

  return sample;
}

}  // namespace catamari

#endif  // ifndef CATAMARI_SUPERNODAL_DPP_IMPL_H_
