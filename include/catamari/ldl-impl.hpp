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

#include "catamari/apply_sparse.hpp"
#include "catamari/blas_matrix.hpp"
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
    const Int num_supernodes = ordering.supernode_sizes.Size();

    Int num_roots = 0;
    for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
      if (ordering.assembly_forest.parents[supernode] < 0) {
        ++num_roots;
      }
    }
    ordering.assembly_forest.roots.Resize(num_roots);
    Int counter = 0;
    for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
      if (ordering.assembly_forest.parents[supernode] < 0) {
        ordering.assembly_forest.roots[counter++] = supernode;
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
void LDLFactorization<Field>::Solve(
    BlasMatrixView<Field>* right_hand_sides) const {
  if (is_supernodal) {
    supernodal_factorization->Solve(right_hand_sides);
  } else {
    scalar_factorization->Solve(right_hand_sides);
  }
}

template <class Field>
Int LDLFactorization<Field>::RefinedSolve(
    const CoordinateMatrix<Field>& matrix, ComplexBase<Field> relative_tol,
    Int max_refine_iters, bool verbose,
    BlasMatrixView<Field>* right_hand_sides) const {
  typedef ComplexBase<Field> Real;
  const Int num_rows = matrix.NumRows();

  if (right_hand_sides->width != 1) {
    std::cerr << "Only single right-hand sides are currently supported."
              << std::endl;
    Solve(right_hand_sides);
    return 0;
  }
  if (max_refine_iters <= 0) {
    Solve(right_hand_sides);
    return 0;
  }

  const BlasMatrix<Field> b_orig = *right_hand_sides;
  const Real b_norm = MaxNorm(b_orig.ConstView());

  // Compute the initial guess
  BlasMatrix<Field> x = b_orig;
  Solve(&x.view);

  BlasMatrix<Field> dx, x_cand, y;
  y.Resize(x.view.height, 1, Field{0});
  ApplySparse(Field{1}, matrix, x.ConstView(), Field{1}, &y.view);

  // b -= y
  for (Int i = 0; i < num_rows; ++i) {
    right_hand_sides->Entry(i, 0) -= y(i, 0);
  }

  Real error_norm = MaxNorm(right_hand_sides->ToConst());
  if (verbose) {
    std::cout << "Original relative error: " << error_norm << std::endl;
  }

  Int refine_iter = 0;
  while (true) {
    const Real relative_error = error_norm / b_norm;
    if (relative_error <= relative_tol) {
      if (verbose) {
        std::cout << "Relative error: " << relative_error
                  << " <= " << relative_tol << std::endl;
      }
      break;
    }

    // Compute the proposed update to the solution.
    dx = *right_hand_sides;
    Solve(&dx.view);
    x_cand = x;

    // x_cand += dx
    for (Int i = 0; i < num_rows; ++i) {
      x_cand(i, 0) += dx(i, 0);
    }

    // Check the new residual.
    ApplySparse(Field{1}, matrix, x_cand.ConstView(), Field{0}, &y.view);

    // *right_hand_sides := b_orig - y
    for (Int i = 0; i < num_rows; ++i) {
      right_hand_sides->Entry(i, 0) = b_orig(i, 0) - y(i, 0);
    }
    const Real new_error_norm = MaxNorm(right_hand_sides->ToConst());
    if (verbose) {
      std::cout << "Refined relative error: " << new_error_norm / b_norm
                << std::endl;
    }

    if (new_error_norm < error_norm) {
      x = x_cand;
    } else {
      break;
    }

    error_norm = new_error_norm;
    ++refine_iter;
    if (refine_iter >= max_refine_iters) {
      break;
    }
  }

  // *right_hand_sides := x
  for (Int i = 0; i < num_rows; ++i) {
    right_hand_sides->Entry(i, 0) = x(i, 0);
  }

  return refine_iter;
}

template <class Field>
void LDLFactorization<Field>::LowerTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  if (is_supernodal) {
    supernodal_factorization->LowerTriangularSolve(right_hand_sides);
  } else {
    scalar_factorization->LowerTriangularSolve(right_hand_sides);
  }
}

template <class Field>
void LDLFactorization<Field>::DiagonalSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  if (is_supernodal) {
    supernodal_factorization->DiagonalSolve(right_hand_sides);
  } else {
    scalar_factorization->DiagonalSolve(right_hand_sides);
  }
}

template <class Field>
void LDLFactorization<Field>::LowerTransposeTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  if (is_supernodal) {
    supernodal_factorization->LowerTransposeTriangularSolve(right_hand_sides);
  } else {
    scalar_factorization->LowerTransposeTriangularSolve(right_hand_sides);
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
