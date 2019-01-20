/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_IMPL_H_
#define CATAMARI_LDL_IMPL_H_

#include <memory>

#include "catamari/ldl.hpp"

namespace catamari {

template <class Field>
LDLResult LDLFactorization<Field>::Factor(const CoordinateMatrix<Field>& matrix,
                                          const LDLControl& control) {
  std::unique_ptr<quotient::CoordinateGraph> graph = matrix.CoordinateGraph();
  const quotient::MinimumDegreeResult analysis =
      quotient::MinimumDegree(*graph, control.md_control);
  graph.reset();

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
    const Int num_supernodes = ordering.supernode_sizes.size();

    Int num_roots = 0;
    for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
      if (ordering.assembly_forest.parents[supernode] < 0) {
        ++num_roots;
      }
    }
    ordering.assembly_forest.roots.clear();
    ordering.assembly_forest.roots.reserve(num_roots);
    for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
      if (ordering.assembly_forest.parents[supernode] < 0) {
        ordering.assembly_forest.roots.push_back(supernode);
      }
    }
  }

  bool use_supernodal;
  if (control.supernodal_strategy == kScalarFactorization) {
    use_supernodal = false;
  } else if (control.supernodal_strategy == kSupernodalFactorization) {
    use_supernodal = true;
  } else {
    // TODO(Jack Poulson): Use a more intelligent means of selecting based upon
    // the floating-point count.
    use_supernodal = true;
  }

  is_supernodal = use_supernodal;
  if (use_supernodal) {
    supernodal_factorization.reset(new supernodal_ldl::Factorization<Field>);
    return supernodal_factorization->Factor(matrix, ordering,
                                            control.supernodal_control);
  } else {
    scalar_factorization.reset(new scalar_ldl::Factorization<Field>);
    return scalar_factorization->Factor(matrix, ordering,
                                        control.scalar_control);
  }
}

template <class Field>
LDLResult LDLFactorization<Field>::Factor(const CoordinateMatrix<Field>& matrix,
                                          const SymmetricOrdering& ordering,
                                          const LDLControl& control) {
  bool use_supernodal;
  if (control.supernodal_strategy == kScalarFactorization) {
    use_supernodal = false;
  } else if (control.supernodal_strategy == kSupernodalFactorization) {
    use_supernodal = true;
  } else {
    // TODO(Jack Poulson): Use a more intelligent means of selecting.
    use_supernodal = true;
  }

  is_supernodal = use_supernodal;
  if (use_supernodal) {
    supernodal_factorization.reset(new supernodal_ldl::Factorization<Field>);
    return supernodal_factorization->Factor(matrix, ordering,
                                            control.supernodal_control);
  } else {
    scalar_factorization.reset(new scalar_ldl::Factorization<Field>);
    return scalar_factorization->Factor(matrix, ordering,
                                        control.scalar_control);
  }
}

template <class Field>
void LDLFactorization<Field>::Solve(BlasMatrix<Field>* matrix) const {
  if (is_supernodal) {
    supernodal_factorization->Solve(matrix);
  } else {
    scalar_factorization->Solve(matrix);
  }
}

template <class Field>
void LDLFactorization<Field>::LowerTriangularSolve(
    BlasMatrix<Field>* matrix) const {
  if (is_supernodal) {
    supernodal_factorization->LowerTriangularSolve(matrix);
  } else {
    scalar_factorization->LowerTriangularSolve(matrix);
  }
}

template <class Field>
void LDLFactorization<Field>::DiagonalSolve(BlasMatrix<Field>* matrix) const {
  if (is_supernodal) {
    supernodal_factorization->DiagonalSolve(matrix);
  } else {
    scalar_factorization->DiagonalSolve(matrix);
  }
}

template <class Field>
void LDLFactorization<Field>::LowerTransposeTriangularSolve(
    BlasMatrix<Field>* matrix) const {
  if (is_supernodal) {
    supernodal_factorization->LowerTransposeTriangularSolve(matrix);
  } else {
    scalar_factorization->LowerTransposeTriangularSolve(matrix);
  }
}

template <class Field>
void LDLFactorization<Field>::PrintLowerFactor(const std::string& label,
                                               std::ostream& os) const {
  if (is_supernodal) {
    supernodal_factorization->PrintLowerFactor(label, os);
  } else {
    scalar_factorization->PrintLowerFactor(label, os);
  }
}

template <class Field>
void LDLFactorization<Field>::PrintDiagonalFactor(const std::string& label,
                                                  std::ostream& os) const {
  if (is_supernodal) {
    supernodal_factorization->PrintDiagonalFactor(label, os);
  } else {
    scalar_factorization->PrintDiagonalFactor(label, os);
  }
}

}  // namespace catamari

#endif  // ifndef CATAMARI_LDL_IMPL_H_
