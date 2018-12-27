/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SUPERNODAL_LDL_FORM_SUPERNODES_H_
#define CATAMARI_LDL_SUPERNODAL_LDL_FORM_SUPERNODES_H_

#include "catamari/ldl/scalar_ldl.hpp"
#include "catamari/ldl/supernodal_ldl/factorization.hpp"

namespace catamari {

struct SupernodalRelaxationControl {
  // If true, relaxed supernodes are created in a manner similar to the
  // suggestion from:
  //   Ashcraft and Grime, "The impact of relaxed supernode partitions on the
  //   multifrontal method", 1989.
  bool relax_supernodes = false;

  // The allowable number of explicit zeros in any relaxed supernode.
  Int allowable_supernode_zeros = 128;

  // The allowable ratio of explicit zeros in any relaxed supernode. This
  // should be interpreted as an *alternative* allowance for a supernode merge.
  // If the number of explicit zeros that would be introduced is less than or
  // equal to 'allowable_supernode_zeros', *or* the ratio of explicit zeros to
  // nonzeros is bounded by 'allowable_supernode_zero_ratio', then the merge
  // can procede.
  float allowable_supernode_zero_ratio = 0.01;
};

namespace supernodal_ldl {

// A data structure for representing whether or not a child supernode can be
// merged with its parent and, if so, how many explicit zeros would be in the
// combined supernode's block column.
struct MergableStatus {
  // Whether or not the supernode can be merged.
  bool mergable;

  // How many explicit zeros would be stored in the merged supernode.
  Int num_merged_zeros;
};

// Fills 'member_to_index' with a length 'num_rows' array whose i'th index
// is the index of the supernode containing column 'i'.
void MemberToIndex(Int num_rows, const std::vector<Int>& supernode_starts,
                   std::vector<Int>* member_to_index);

// Builds a packed set of child links for an elimination forest given the
// parent links (with root nodes having their parent set to -1).
void EliminationForestFromParents(const std::vector<Int>& parents,
                                  std::vector<Int>* children,
                                  std::vector<Int>* child_offsets);

// Builds an elimination forest over the supernodes from an elimination forest
// over the nodes.
void ConvertFromScalarToSupernodalEliminationForest(
    Int num_supernodes, const std::vector<Int>& parents,
    const std::vector<Int>& member_to_index,
    std::vector<Int>* supernode_parents);

// Checks that a valid set of supernodes has been provided by explicitly
// computing each row pattern and ensuring that each intersects entire
// supernodes.
template <class Field>
bool ValidFundamentalSupernodes(const CoordinateMatrix<Field>& matrix,
                                const std::vector<Int>& permutation,
                                const std::vector<Int>& inverse_permutation,
                                const std::vector<Int>& supernode_sizes);

// Compute an unrelaxed supernodal partition using the existing ordering.
// We require that supernodes have dense diagonal blocks and equal structures
// below the diagonal block.
template <class Field>
void FormFundamentalSupernodes(const CoordinateMatrix<Field>& matrix,
                               const std::vector<Int>& permutation,
                               const std::vector<Int>& inverse_permutation,
                               const std::vector<Int>& parents,
                               const std::vector<Int>& degrees,
                               std::vector<Int>* supernode_sizes,
                               ScalarLowerStructure* scalar_structure);

// Returns whether or not the child supernode can be merged into its parent by
// counting the number of explicit zeros that would be introduced by the
// merge.
//
// Consider the possibility of merging a child supernode '0' with parent
// supernode '1', which would result in an expanded supernode of the form:
//
//    -------------
//   | L00  |      |
//   |------|------|
//   | L10  | L11  |
//   ---------------
//   |      |      |
//   | L20  | L21  |
//   |      |      |
//    -------------
//
// Because it is assumed that supernode 1 is the parent of supernode 0,
// the only places explicit nonzeros can be introduced are in:
//   * L10: if supernode 0 was not fully-connected to supernode 1.
//   * L20: if supernode 0's structure (with supernode 1 removed) does not
//     contain every member of the structure of supernode 1.
//
// Counting the number of explicit zeros that would be introduced is thus
// simply a matter of counting these mismatches.
//
// The reason that downstream explicit zeros would not be introduced is the
// same as the reason explicit zeros are not introduced into L21; supernode
// 1 is the parent of supernode 0.
//
MergableStatus MergableSupernode(Int child_tail, Int parent_tail,
                                 Int child_size, Int parent_size,
                                 Int num_child_explicit_zeros,
                                 Int num_parent_explicit_zeros,
                                 const std::vector<Int>& orig_member_to_index,
                                 const ScalarLowerStructure& scalar_structure,
                                 const SupernodalRelaxationControl& control);

void MergeChildren(Int parent, const std::vector<Int>& orig_supernode_starts,
                   const std::vector<Int>& orig_supernode_sizes,
                   const std::vector<Int>& orig_member_to_index,
                   const std::vector<Int>& children,
                   const std::vector<Int>& child_offsets,
                   const ScalarLowerStructure& scalar_structure,
                   const SupernodalRelaxationControl& control,
                   std::vector<Int>* supernode_sizes,
                   std::vector<Int>* num_explicit_zeros,
                   std::vector<Int>* last_merged_child,
                   std::vector<Int>* merge_parents);

// Walk up the tree in the original postordering, merging supernodes as we
// progress. The 'relaxed_permutation' and 'relaxed_inverse_permutation'
// variables are also inputs.
void RelaxSupernodes(const std::vector<Int>& orig_parents,
                     const std::vector<Int>& orig_supernode_sizes,
                     const std::vector<Int>& orig_supernode_starts,
                     const std::vector<Int>& orig_supernode_parents,
                     const std::vector<Int>& orig_supernode_degrees,
                     const std::vector<Int>& orig_member_to_index,
                     const ScalarLowerStructure& scalar_structure,
                     const SupernodalRelaxationControl& control,
                     std::vector<Int>* relaxed_permutation,
                     std::vector<Int>* relaxed_inverse_permutation,
                     std::vector<Int>* relaxed_parents,
                     std::vector<Int>* relaxed_supernode_parents,
                     std::vector<Int>* relaxed_supernode_degrees,
                     std::vector<Int>* relaxed_supernode_sizes,
                     std::vector<Int>* relaxed_supernode_starts,
                     std::vector<Int>* relaxed_supernode_member_to_index);

// Computes the sizes of the structures of a supernodal LDL' factorization.
template <class Field>
void SupernodalDegrees(const CoordinateMatrix<Field>& matrix,
                       const std::vector<Int>& permutation,
                       const std::vector<Int>& inverse_permutation,
                       const std::vector<Int>& supernode_sizes,
                       const std::vector<Int>& supernode_starts,
                       const std::vector<Int>& member_to_index,
                       const std::vector<Int>& parents,
                       std::vector<Int>* supernode_degrees);

// Form the (possibly relaxed) supernodes for the factorization.
template <class Field>
void FormSupernodes(const CoordinateMatrix<Field>& matrix,
                    const SupernodalRelaxationControl& control,
                    std::vector<Int>* parents,
                    std::vector<Int>* supernode_degrees,
                    std::vector<Int>* supernode_parents,
                    Factorization<Field>* factorization);

}  // namespace supernodal_ldl
}  // namespace catamari

#include "catamari/ldl/supernodal_ldl/form_supernodes-impl.hpp"

#endif  // ifndef CATAMARI_LDL_SUPERNODAL_LDL_FORM_SUPERNODES_H_
