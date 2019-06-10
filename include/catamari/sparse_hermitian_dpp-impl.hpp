/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_HERMITIAN_DPP_IMPL_H_
#define CATAMARI_SPARSE_HERMITIAN_DPP_IMPL_H_

#include "catamari/flush_to_zero.hpp"

#include "catamari/sparse_hermitian_dpp.hpp"

namespace catamari {

template <class Field>
SparseHermitianDPP<Field>::SparseHermitianDPP(
    const CoordinateMatrix<Field>& matrix,
    const SparseHermitianDPPControl& control) {
  // Avoid the potential for order-of-magnitude performance degradation from
  // slow subnormal processing.
  EnableFlushToZero();

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
    supernodal_dpp_.reset(new SupernodalHermitianDPP<Field>(
        matrix, ordering, control.supernodal_control));
  } else {
    quotient_graph.release();
    scalar_dpp_.reset(new ScalarHermitianDPP<Field>(matrix, ordering,
                                                    control.scalar_control));
  }
}

template <class Field>
SparseHermitianDPP<Field>::SparseHermitianDPP(
    const CoordinateMatrix<Field>& matrix, const SymmetricOrdering& ordering,
    const SparseHermitianDPPControl& control) {
  // Avoid the potential for order-of-magnitude performance degradation from
  // slow subnormal processing.
  EnableFlushToZero();

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
    supernodal_dpp_.reset(new SupernodalHermitianDPP<Field>(
        matrix, ordering, control.supernodal_control));
  } else {
    scalar_dpp_.reset(new ScalarHermitianDPP<Field>(matrix, ordering,
                                                    control.scalar_control));
  }
}

template <class Field>
std::vector<Int> SparseHermitianDPP<Field>::Sample(
    bool maximum_likelihood) const {
  if (is_supernodal_) {
    return supernodal_dpp_->Sample(maximum_likelihood);
  } else {
    return scalar_dpp_->Sample(maximum_likelihood);
  }
}

template <class Field>
ComplexBase<Field> SparseHermitianDPP<Field>::LogLikelihood() const {
  if (is_supernodal_) {
    return supernodal_dpp_->LogLikelihood();
  } else {
    return scalar_dpp_->LogLikelihood();
  }
}

}  // namespace catamari

#endif  // ifndef CATAMARI_SPARSE_HERMITIAN_DPP_IMPL_H_
