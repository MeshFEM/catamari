/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DPP_IMPL_H_
#define CATAMARI_DPP_IMPL_H_

#include "catamari/dpp.hpp"

namespace catamari {

template <class Field>
DPP<Field>::DPP(const CoordinateMatrix<Field>& matrix,
                const DPPControl& control) {
  std::unique_ptr<quotient::QuotientGraph> quotient_graph(
      new quotient::QuotientGraph(matrix.NumRows(), matrix.Entries(),
                                  control.md_control));
  const quotient::MinimumDegreeResult analysis =
      quotient::MinimumDegree(quotient_graph.get());

  SymmetricOrdering ordering;
  quotient_graph->ComputePostorder(&ordering.inverse_permutation);
  quotient::InvertPermutation(ordering.inverse_permutation,
                              &ordering.permutation);

  quotient_graph->PermutedSupernodeSizes(ordering.inverse_permutation,
                                         &ordering.supernode_sizes);
  OffsetScan(ordering.supernode_sizes, &ordering.supernode_offsets);

  Buffer<Int> member_to_supernode;
  quotient_graph->PermutedMemberToSupernode(ordering.inverse_permutation,
                                            &member_to_supernode);
  quotient_graph->PermutedAssemblyParents(ordering.permutation,
                                          member_to_supernode,
                                          &ordering.assembly_forest.parents);
  ordering.assembly_forest.FillFromParents();

  quotient_graph.release();
  supernodal_dpp_.reset(
      new SupernodalDPP<Field>(matrix, ordering, control.supernodal_control));
}

template <class Field>
DPP<Field>::DPP(const CoordinateMatrix<Field>& matrix,
                const SymmetricOrdering& ordering, const DPPControl& control) {
  supernodal_dpp_.reset(
      new SupernodalDPP<Field>(matrix, ordering, control.supernodal_control));
}

template <class Field>
std::vector<Int> DPP<Field>::Sample(bool maximum_likelihood) const {
  return supernodal_dpp_->Sample(maximum_likelihood);
}

}  // namespace catamari

#endif  // ifndef CATAMARI_DPP_IMPL_H_
