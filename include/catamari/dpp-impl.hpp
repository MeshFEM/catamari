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
  std::unique_ptr<quotient::CoordinateGraph> graph = matrix.CoordinateGraph();
  const quotient::MinimumDegreeResult analysis =
      quotient::MinimumDegree(*graph, control.md_control);

  SymmetricOrdering ordering;
  ordering.permutation = analysis.permutation;
  ordering.inverse_permutation = analysis.inverse_permutation;
  ordering.supernode_sizes = analysis.permuted_supernode_sizes;
  ordering.assembly_forest.parents = analysis.permuted_assembly_parents;
  quotient::ChildrenFromParents(ordering.assembly_forest.parents,
                                &ordering.assembly_forest.children,
                                &ordering.assembly_forest.child_offsets);

  OffsetScan(ordering.supernode_sizes, &ordering.supernode_offsets);

  // Fill the list of supernodal ordering roots.
  // TODO(Jack Poulson): Encapsulate this into a utility function.
  {
    const Int num_supernodes = ordering.supernode_sizes.Size();

    Int num_roots = 0;
    for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
      if (ordering.assembly_forest.parents[supernode] < 0) {
        ++num_roots;
      }
    }
    ordering.assembly_forest.roots.Resize(num_roots);
    num_roots = 0;
    for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
      if (ordering.assembly_forest.parents[supernode] < 0) {
        ordering.assembly_forest.roots[num_roots++] = supernode;
      }
    }
  }

  supernodal_dpp_.reset(
      new SupernodalDPP<Field>(matrix, ordering, control.supernodal_control));
}

template <class Field>
std::vector<Int> DPP<Field>::Sample(bool maximum_likelihood) const {
  return supernodal_dpp_->Sample(maximum_likelihood);
}

}  // namespace catamari

#endif  // ifndef CATAMARI_DPP_IMPL_H_
