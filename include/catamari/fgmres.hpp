/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.  */
#ifndef CATAMARI_FGMRES_H_
#define CATAMARI_FGMRES_H_

#include "catamari/blas_matrix.hpp"

namespace catamari {

// The configuration parameters for Flexible GMRES, as defined in
//
//   Youcef Saad, "A flexible inner-outer preconditioned GMRES algorithm",
//   SIAM Journal on Scientific Computing, 14(2), pp. 461--469, 1993.
//   DOI: https://epubs.siam.org/doi/10.1137/0914028
//
template <typename Real>
struct FGMRESControl {
  // The maximum number of iterations before restarting FGMRES.
  Int max_inner_iterations = 10;

  // The maximum number of total inner iterations before stopping FGMRES.
  Int max_iterations = 100;

  // We define the '(two-norm) relative error' of an approximate solution 'x'
  // to 'A x = b' as:
  //
  //   || b - A x ||_2 / || b ||_2.
  //
  // Our target tolerance for the relative error of our solution takes the form:
  //
  //   coefficient * epsilon^exponent,
  //
  // where 'epsilon' is the precision of type 'Real'. We stop iterating if this
  // tolerance has been reached.
  Real relative_tolerance_coefficient = Real(1);
  Real relative_tolerance_exponent = Real(0.5);

  // If true, progress information is printed.
  bool verbose = false;
};

// The summary of a run of FGMRES on a set of right-hand sides.
template <typename Real>
struct FGMRESStatus {
  // The maximum two-norm relative error,
  //
  //   max_j || b_j - A x_j ||_2 / || b_j ||_2,
  //
  // over the solved right-hand sides.
  Real relative_error;

  // The maximum number of FGMRES iterations performed for any of the
  // right-hand sides.
  Int num_iterations;

  // The maximum number of outer FGMRES iterations performed for any of the
  // right-hand sides.
  Int num_outer_iterations;
};

template <class Field, class ApplyMatrix, class ApplyPreconditioner>
FGMRESStatus<ComplexBase<Field>> FGMRES(
    const ApplyMatrix apply_matrix,
    const ApplyPreconditioner apply_preconditioner,
    const FGMRESControl<ComplexBase<Field>>& control,
    const ConstBlasMatrixView<Field>& right_hand_sides,
    BlasMatrix<Field>* solutions);

}  // namespace catamari

#include "catamari/fgmres-impl.hpp"

#endif  // ifndef CATAMARI_FGMRES_H_
