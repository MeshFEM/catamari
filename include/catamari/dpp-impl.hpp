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
  scalar_dpp_.reset();
  supernodal_dpp_.reset();

  std::unique_ptr<quotient::QuotientGraph> quotient_graph(
      new quotient::QuotientGraph(matrix.NumRows(), matrix.Entries(),
                                  control.md_control));
  const quotient::MinimumDegreeResult analysis =
      quotient::MinimumDegree(quotient_graph.get());

  SymmetricOrdering ordering;
  quotient_graph->ComputePostorder(&ordering.inverse_permutation);
  quotient::InvertPermutation(ordering.inverse_permutation,
                              &ordering.permutation);

  bool use_supernodal;
  if (control.supernodal_strategy == kScalarFactorization) {
    use_supernodal = false;
  } else if (control.supernodal_strategy == kSupernodalFactorization) {
    use_supernodal = true;
  } else {
    const double intensity =
        analysis.num_cholesky_flops / analysis.num_cholesky_nonzeros;

    // TODO(Jack Poulson): Make these configurable.
    const double flop_threshold = 1e5;
    const double intensity_threshold = 40;

    use_supernodal = analysis.num_cholesky_flops >= flop_threshold &&
                     intensity >= intensity_threshold;
  }

  is_supernodal_ = use_supernodal;
  if (use_supernodal) {
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
  } else {
    quotient_graph.release();
    scalar_dpp_.reset(
        new ScalarDPP<Field>(matrix, ordering, control.scalar_control));
  }
}

template <class Field>
DPP<Field>::DPP(const CoordinateMatrix<Field>& matrix,
                const SymmetricOrdering& ordering, const DPPControl& control) {
  scalar_dpp_.reset();
  supernodal_dpp_.reset();

  bool use_supernodal;
  if (control.supernodal_strategy == kScalarFactorization) {
    use_supernodal = false;
  } else if (control.supernodal_strategy == kSupernodalFactorization) {
    use_supernodal = true;
  } else {
    // TODO(Jack Poulson): Use a more intelligent means of selecting.
    // This routine should likely take in a flop count analysis.
    use_supernodal = true;
  }

  is_supernodal_ = use_supernodal;
  if (use_supernodal) {
    supernodal_dpp_.reset(
        new SupernodalDPP<Field>(matrix, ordering, control.supernodal_control));
  } else {
    scalar_dpp_.reset(
        new ScalarDPP<Field>(matrix, ordering, control.scalar_control));
  }
}

template <class Field>
std::vector<Int> DPP<Field>::Sample(bool maximum_likelihood) const {
  if (is_supernodal_) {
    return supernodal_dpp_->Sample(maximum_likelihood);
  } else {
    return scalar_dpp_->Sample(maximum_likelihood);
  }
}

}  // namespace catamari

#endif  // ifndef CATAMARI_DPP_IMPL_H_
