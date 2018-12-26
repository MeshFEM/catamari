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
  CATAMARI_ASSERT(supernode_degrees_.size() == supernode_sizes_.size(),
                  "Invalid supernode degrees size.");

  lower_factor_.reset(
      new SupernodalLowerFactor<Field>(supernode_sizes_, supernode_degrees_));
  diagonal_factor_.reset(new SupernodalDiagonalFactor<Field>(supernode_sizes_));

  supernodal_ldl::FillStructureIndices(
      matrix_, permutation_, inverse_permutation_, parents_, supernode_sizes_,
      supernode_member_to_index_, lower_factor_.get());
}

template <class Field>
std::vector<Int> SupernodalDPP<Field>::Sample(bool maximum_likelihood) const {
  return LeftLookingSample(maximum_likelihood);
}

template <class Field>
std::vector<Int> SupernodalDPP<Field>::LeftLookingSample(
    bool maximum_likelihood) const {
  const Int num_rows = supernode_starts_.back();
  const Int num_supernodes = supernode_sizes_.size();
  const SymmetricFactorizationType factorization_type =
      kLDLAdjointFactorization;

  // Reset the lower factor to all zeros.
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    BlasMatrix<Field>& matrix = lower_factor_->blocks[supernode];
    std::fill(matrix.data, matrix.data + matrix.leading_dim * matrix.width,
              Field{0});
  }

  // Reset the diagonal factor to all zeros.
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    BlasMatrix<Field>& matrix = diagonal_factor_->blocks[supernode];
    std::fill(matrix.data, matrix.data + matrix.leading_dim * matrix.width,
              Field{0});
  }

  // Initialize the factors with the input matrix.
  supernodal_ldl::FillNonzeros(matrix_, permutation_, inverse_permutation_,
                               supernode_starts_, supernode_sizes_,
                               supernode_member_to_index_, lower_factor_.get(),
                               diagonal_factor_.get());

  std::vector<Int> sample;
  sample.reserve(num_rows);

  // Set up a buffer for supernodal updates.
  Int max_supernode_size = 0;
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    max_supernode_size = std::max(max_supernode_size,
                                  diagonal_factor_->blocks[supernode].height);
  }
  std::vector<Field> scaled_transpose_buffer(
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
  std::vector<const Int*> intersect_ptrs(num_supernodes);
  std::vector<Int> rel_rows(num_supernodes);

  for (Int main_supernode = 0; main_supernode < num_supernodes;
       ++main_supernode) {
    BlasMatrix<Field>& main_diagonal_block =
        diagonal_factor_->blocks[main_supernode];
    BlasMatrix<Field>& main_lower_block = lower_factor_->blocks[main_supernode];
    const Int main_supernode_start = supernode_starts_[main_supernode];

    pattern_flags[main_supernode] = main_supernode;

    intersect_ptrs[main_supernode] =
        lower_factor_->IntersectionSizes(main_supernode);
    rel_rows[main_supernode] = 0;

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
      const ConstBlasMatrix<Field>& descendant_lower_block =
          lower_factor_->blocks[descendant_supernode];
      const Int descendant_degree = descendant_lower_block.height;
      const Int descendant_supernode_size = descendant_lower_block.width;

      const Int descendant_main_intersect_size =
          *intersect_ptrs[descendant_supernode];

      const Int descendant_main_rel_row = rel_rows[descendant_supernode];
      ConstBlasMatrix<Field> descendant_main_matrix =
          descendant_lower_block.Submatrix(descendant_main_rel_row, 0,
                                           descendant_main_intersect_size,
                                           descendant_supernode_size);

      BlasMatrix<Field> scaled_transpose;
      scaled_transpose.height = descendant_supernode_size;
      scaled_transpose.width = descendant_main_intersect_size;
      scaled_transpose.leading_dim = descendant_supernode_size;
      scaled_transpose.data = scaled_transpose_buffer.data();

      supernodal_ldl::FormScaledTranspose(
          factorization_type,
          diagonal_factor_->blocks[descendant_supernode].ToConst(),
          descendant_main_matrix, &scaled_transpose);

      BlasMatrix<Field> update_matrix;
      update_matrix.height = descendant_main_intersect_size;
      update_matrix.width = descendant_main_intersect_size;
      update_matrix.leading_dim = descendant_main_intersect_size;
      update_matrix.data = update_buffer.data();

      supernodal_ldl::UpdateDiagonalBlock(
          factorization_type, supernode_starts_, *lower_factor_, main_supernode,
          descendant_supernode, descendant_main_rel_row, descendant_main_matrix,
          scaled_transpose.ToConst(), &main_diagonal_block, &update_matrix);

      intersect_ptrs[descendant_supernode]++;
      rel_rows[descendant_supernode] += descendant_main_intersect_size;

      // L(KNext:n, K) -= L(KNext:n, J) * (D(J, J) * L(K, J)')
      //                = L(KNext:n, J) * Z(J, K).
      const Int* descendant_active_intersect_size_beg =
          intersect_ptrs[descendant_supernode];
      Int descendant_active_rel_row = rel_rows[descendant_supernode];
      const Int* main_active_intersect_sizes =
          lower_factor_->IntersectionSizes(main_supernode);
      Int main_active_rel_row = 0;
      while (descendant_active_rel_row != descendant_degree) {
        const Int descendant_active_intersect_size =
            *descendant_active_intersect_size_beg;

        ConstBlasMatrix<Field> descendant_active_matrix =
            descendant_lower_block.Submatrix(descendant_active_rel_row, 0,
                                             descendant_active_intersect_size,
                                             descendant_supernode_size);

        // The width of the update matrix and pointer are already correct.
        update_matrix.height = descendant_active_intersect_size;
        update_matrix.leading_dim = descendant_active_intersect_size;

        supernodal_ldl::SeekForMainActiveRelativeRow(
            main_supernode, descendant_supernode, descendant_active_rel_row,
            supernode_member_to_index_, *lower_factor_, &main_active_rel_row,
            &main_active_intersect_sizes);
        const Int main_active_intersect_size = *main_active_intersect_sizes;

        supernodal_ldl::UpdateSubdiagonalBlock(
            main_supernode, descendant_supernode, main_active_rel_row,
            descendant_main_rel_row, descendant_active_rel_row,
            main_active_intersect_size, supernode_starts_,
            supernode_member_to_index_, scaled_transpose.ToConst(),
            descendant_active_matrix, lower_factor_.get(), &update_matrix);

        ++descendant_active_intersect_size_beg;
        descendant_active_rel_row += descendant_active_intersect_size;
      }
    }

    // Sample and factor the diagonal block.
    {
      const std::vector<Int> supernode_sample =
          LowerFactorAndSampleDPP(maximum_likelihood, &main_diagonal_block,
                                  &generator_, &unit_uniform_);
      for (const Int& index : supernode_sample) {
        const Int orig_row = main_supernode_start + index;
        if (inverse_permutation_.empty()) {
          sample.push_back(orig_row);
        } else {
          sample.push_back(inverse_permutation_[orig_row]);
        }
      }
    }

    supernodal_ldl::SolveAgainstDiagonalBlock(
        factorization_type, main_diagonal_block.ToConst(), &main_lower_block);
  }

  std::sort(sample.begin(), sample.end());

  return sample;
}

}  // namespace catamari

#endif  // ifndef CATAMARI_SUPERNODAL_DPP_IMPL_H_
