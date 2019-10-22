/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.  */
#ifndef CATAMARI_REFINED_SOLVE_H_
#define CATAMARI_REFINED_SOLVE_H_

#include "catamari/blas_matrix.hpp"

namespace catamari {

namespace promote {

template <typename Field>
void HigherToLower(const ConstBlasMatrixView<Promote<Field>>& higher,
                   BlasMatrixView<Field>* lower);

template <typename Field>
void HigherToLower(const BlasMatrix<Promote<Field>>& higher,
                   BlasMatrix<Field>* lower);

template <typename Field>
void LowerToHigher(const ConstBlasMatrixView<Field>& lower,
                   BlasMatrixView<Promote<Field>>* higher);

template <typename Field>
void LowerToHigher(const ConstBlasMatrixView<Field>& lower,
                   BlasMatrix<Promote<Field>>* higher);

template <typename Field>
void LowerToHigher(const BlasMatrix<Field>& lower,
                   BlasMatrix<Promote<Field>>* higher);

}  // namespace promote

template <class Field, class ApplyMatrix, class ApplyInverse>
RefinedSolveStatus<ComplexBase<Field>> RefinedSolve(
    const ApplyMatrix apply_matrix, const ApplyInverse apply_inverse,
    const RefinedSolveControl<ComplexBase<Field>>& control,
    BlasMatrixView<Field>* right_hand_sides);

template <class Field, class ApplyMatrix, class ApplyInverse>
RefinedSolveStatus<ComplexBase<Field>> PromotedRefinedSolve(
    const ApplyMatrix apply_matrix, const ApplyInverse apply_inverse,
    const RefinedSolveControl<ComplexBase<Field>>& control,
    BlasMatrixView<Field>* right_hand_sides_lower);

}  // namespace catamari

#include "catamari/refined_solve-impl.hpp"

#endif  // ifndef CATAMARI_REFINED_SOLVE_H_
