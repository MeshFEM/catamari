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
LDLResult LDL(const CoordinateMatrix<Field>& matrix, const LDLControl& control,
              LDLFactorization<Field>* factorization) {
  std::unique_ptr<quotient::CoordinateGraph> graph = matrix.CoordinateGraph();
  const quotient::MinimumDegreeResult analysis =
      quotient::MinimumDegree(*graph, control.md_control);
  graph.reset();
  const std::vector<Int> permutation = analysis.Permutation();

  const Int num_rows = permutation.size();
  std::vector<Int> inverse_permutation(num_rows);
  for (Int row = 0; row < num_rows; ++row) {
    inverse_permutation[permutation[row]] = row;
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

  factorization->is_supernodal = use_supernodal;
  if (use_supernodal) {
    factorization->supernodal_factorization.reset(
        new supernodal_ldl::Factorization<Field>);
    return LDL(matrix, permutation, inverse_permutation,
               control.supernodal_control,
               factorization->supernodal_factorization.get());
  } else {
    factorization->scalar_factorization.reset(
        new ScalarLDLFactorization<Field>);
    return LDL(matrix, permutation, inverse_permutation, control.scalar_control,
               factorization->scalar_factorization.get());
  }
}

template <class Field>
LDLResult LDL(const CoordinateMatrix<Field>& matrix,
              const std::vector<Int>& permutation,
              const std::vector<Int>& inverse_permutation,
              const LDLControl& control,
              LDLFactorization<Field>* factorization) {
  bool use_supernodal;
  if (control.supernodal_strategy == kScalarFactorization) {
    use_supernodal = false;
  } else if (control.supernodal_strategy == kSupernodalFactorization) {
    use_supernodal = true;
  } else {
    // TODO(Jack Poulson): Use a more intelligent means of selecting.
    use_supernodal = true;
  }

  factorization->is_supernodal = use_supernodal;
  if (use_supernodal) {
    factorization->supernodal_factorization.reset(
        new supernodal_ldl::Factorization<Field>);
    return LDL(matrix, permutation, inverse_permutation,
               control.supernodal_control,
               factorization->supernodal_factorization.get());
  } else {
    factorization->scalar_factorization.reset(
        new ScalarLDLFactorization<Field>);
    return LDL(matrix, permutation, inverse_permutation, control.scalar_control,
               factorization->scalar_factorization.get());
  }
}

template <class Field>
void LDLSolve(const LDLFactorization<Field>& factorization,
              BlasMatrix<Field>* matrix) {
  if (factorization.is_supernodal) {
    supernodal_ldl::Solve(
        *factorization.supernodal_factorization.get(), matrix);
  } else {
    LDLSolve(*factorization.scalar_factorization.get(), matrix);
  }
}

template <class Field>
void LowerTriangularSolve(const LDLFactorization<Field>& factorization,
                          BlasMatrix<Field>* matrix) {
  if (factorization.is_supernodal) {
    supernodal_ldl::LowerTriangularSolve(
        *factorization.supernodal_factorization.get(), matrix);
  } else {
    LowerTriangularSolve(*factorization.scalar_factorization.get(), matrix);
  }
}

template <class Field>
void DiagonalSolve(const LDLFactorization<Field>& factorization,
                   BlasMatrix<Field>* matrix) {
  if (factorization.is_supernodal) {
    supernodal_ldl::DiagonalSolve(
        *factorization.supernodal_factorization.get(), matrix);
  } else {
    DiagonalSolve(*factorization.scalar_factorization.get(), matrix);
  }
}

template <class Field>
void LowerTransposeTriangularSolve(const LDLFactorization<Field>& factorization,
                                   BlasMatrix<Field>* matrix) {
  if (factorization.is_supernodal) {
    supernodal_ldl::LowerTransposeTriangularSolve(
        *factorization.supernodal_factorization.get(),
                                matrix);
  } else {
    LowerTransposeTriangularSolve(*factorization.scalar_factorization.get(),
                                  matrix);
  }
}

template <class Field>
void PrintLowerFactor(const LDLFactorization<Field>& factorization,
                      const std::string& label, std::ostream& os) {
  if (factorization.is_supernodal) {
    supernodal_ldl::PrintLowerFactor(
        *factorization.supernodal_factorization.get(), label, os);
  } else {
    PrintLowerFactor(*factorization.scalar_factorization.get(), label, os);
  }
}

template <class Field>
void PrintDiagonalFactor(const LDLFactorization<Field>& factorization,
                         const std::string& label, std::ostream& os) {
  if (factorization.is_supernodal) {
    supernodal_ldl::PrintDiagonalFactor(
        *factorization.supernodal_factorization.get(), label, os);
  } else {
    PrintDiagonalFactor(*factorization.scalar_factorization.get(), label, os);
  }
}

}  // namespace catamari

#endif  // ifndef CATAMARI_LDL_IMPL_H_
