/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.  */
#ifndef CATAMARI_SPARSE_LDL_IMPL_H_
#define CATAMARI_SPARSE_LDL_IMPL_H_

#include <memory>

#include "catamari/apply_sparse.hpp"
#include "catamari/blas_matrix.hpp"
#include "catamari/equilibrate_symmetric_matrix.hpp"
#include "catamari/flush_to_zero.hpp"
#include "catamari/refined_solve.hpp"

#include "catamari/sparse_ldl.hpp"

namespace catamari {

template <class Field>
SparseLDL<Field>::SparseLDL() {
  // Avoid the potential for order-of-magnitude performance degradation from
  // slow subnormal processing.
  EnableFlushToZero();
}

template <class Field>
SparseLDLResult<Field> SparseLDL<Field>::Factor(
    const CoordinateMatrix<Field>& matrix,
    const SparseLDLControl<Field>& control) {
  scalar_factorization.reset();
  supernodal_factorization.reset();

#ifdef CATAMARI_ENABLE_TIMERS
  quotient::Timer timer;
  timer.Start();
#endif
  std::unique_ptr<quotient::QuotientGraph> quotient_graph(
      new quotient::QuotientGraph(matrix.NumRows(), matrix.Entries(),
                                  control.md_control));
  const quotient::MinimumDegreeResult analysis =
      quotient::MinimumDegree(quotient_graph.get());
#ifdef QUOTIENT_ENABLE_TIMERS
  for (const std::pair<std::string, double>& time :
       quotient_graph->ComponentSeconds()) {
    std::cout << "  " << time.first << ": " << time.second << std::endl;
  }
#endif  // ifdef QUOTIENT_TIMERS
#ifdef CATAMARI_ENABLE_TIMERS
  std::cout << "Reordering time: " << timer.Stop() << " seconds." << std::endl;
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  CATAMARI_START_TIMER(timer);
  SymmetricOrdering ordering;
  quotient_graph->ComputePostorder(&ordering.inverse_permutation);
  quotient::InvertPermutation(ordering.inverse_permutation,
                              &ordering.permutation);
#ifdef CATAMARI_ENABLE_TIMERS
  std::cout << "Postorder and invert: " << timer.Stop() << " seconds."
            << std::endl;
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  if (control.supernodal_strategy == kScalarFactorization) {
    is_supernodal = false;
  } else if (control.supernodal_strategy == kSupernodalFactorization) {
    is_supernodal = true;
  } else {
    const double intensity =
        analysis.num_cholesky_flops / analysis.num_cholesky_nonzeros;
    is_supernodal =
        analysis.num_cholesky_flops >= control.supernodal_flop_threshold &&
        intensity >= control.supernodal_intensity_threshold;
  }

  // Optionally equilibrate the matrix.
  const CoordinateMatrix<Field>* matrix_to_factor;
  CoordinateMatrix<Field> equilibrated_matrix;
  if (control.equilibrate) {
    equilibrated_matrix = matrix;
    EquilibrateSymmetricMatrix(&equilibrated_matrix, &equilibration_,
                               control.verbose);
    matrix_to_factor = &equilibrated_matrix;
    have_equilibration_ = true;
  } else {
    matrix_to_factor = &matrix;
    have_equilibration_ = false;
  }

  SparseLDLResult<Field> result;
  if (is_supernodal) {
    CATAMARI_START_TIMER(timer);

    // TODO(Jack Poulson): Modify quotient to combine these into single routine.
    Buffer<Int> member_to_supernode;
    quotient_graph->PermutedMemberToSupernode(ordering.inverse_permutation,
                                              &member_to_supernode);
    quotient_graph->PermutedAssemblyParents(ordering.permutation,
                                            member_to_supernode,
                                            &ordering.assembly_forest.parents);

    // TODO(Jack Poulson): Only compute the children and/or roots when needed.
    ordering.assembly_forest.FillFromParents();

    quotient_graph->PermutedSupernodeSizes(ordering.inverse_permutation,
                                           &ordering.supernode_sizes);
    OffsetScan(ordering.supernode_sizes, &ordering.supernode_offsets);
#ifdef CATAMARI_ENABLE_TIMERS
    std::cout << "Ordering postprocessing: " << timer.Stop() << " seconds."
              << std::endl;
#endif  // ifdef CATAMARI_ENABLE_TIMERS

    quotient_graph.release();
    supernodal_factorization.reset(new supernodal_ldl::Factorization<Field>);
    result = supernodal_factorization->Factor(*matrix_to_factor, ordering,
                                              control.supernodal_control);
  } else {
    quotient_graph.release();
    scalar_factorization.reset(new scalar_ldl::Factorization<Field>);
    result = scalar_factorization->Factor(*matrix_to_factor, ordering,
                                          control.scalar_control);
  }

  return result;
}

template <class Field>
SparseLDLResult<Field> SparseLDL<Field>::Factor(
    const CoordinateMatrix<Field>& matrix, const SymmetricOrdering& ordering,
    const SparseLDLControl<Field>& control) {
  scalar_factorization.reset();
  supernodal_factorization.reset();

  if (control.supernodal_strategy == kScalarFactorization) {
    is_supernodal = false;
  } else if (control.supernodal_strategy == kSupernodalFactorization) {
    is_supernodal = true;
  } else {
    // TODO(Jack Poulson): Use a more intelligent means of selecting.
    // This routine should likely take in a flop count analysis.
    is_supernodal = true;
  }

  // Optionally equilibrate the matrix.
  const CoordinateMatrix<Field>* matrix_to_factor;
  CoordinateMatrix<Field> equilibrated_matrix;
  if (control.equilibrate) {
    equilibrated_matrix = matrix;
    EquilibrateSymmetricMatrix(&equilibrated_matrix, &equilibration_,
                               control.verbose);
    matrix_to_factor = &equilibrated_matrix;
    have_equilibration_ = true;
  } else {
    matrix_to_factor = &matrix;
    have_equilibration_ = false;
  }

  SparseLDLResult<Field> result;
  if (is_supernodal) {
    supernodal_factorization.reset(new supernodal_ldl::Factorization<Field>);
    result = supernodal_factorization->Factor(*matrix_to_factor, ordering,
                                              control.supernodal_control);
  } else {
    scalar_factorization.reset(new scalar_ldl::Factorization<Field>);
    result = scalar_factorization->Factor(*matrix_to_factor, ordering,
                                          control.scalar_control);
  }
  if (have_equilibration_) {
    // We factored inv(D) A inv(D), so the regularization needs to be wrapped
    // with D . D.
    for (std::pair<Int, Real>& reg : result.dynamic_regularization) {
      reg.second *= equilibration_(reg.first) * equilibration_(reg.first);
    }
  }
  return result;
}

template <class Field>
Int SparseLDL<Field>::NumRows() const {
  if (is_supernodal) {
    return supernodal_factorization->NumRows();
  } else {
    return scalar_factorization->NumRows();
  }
}

template <class Field>
const Buffer<Int>& SparseLDL<Field>::Permutation() const {
  if (is_supernodal) {
    return supernodal_factorization->Permutation();
  } else {
    return scalar_factorization->Permutation();
  }
}

template <class Field>
const Buffer<Int>& SparseLDL<Field>::InversePermutation() const {
  if (is_supernodal) {
    return supernodal_factorization->InversePermutation();
  } else {
    return scalar_factorization->InversePermutation();
  }
}

template <class Field>
void SparseLDL<Field>::DynamicRegularizationDiagonal(
    const SparseLDLResult<Field>& result,
    BlasMatrix<ComplexBase<Field>>* diagonal) const {
  typedef ComplexBase<Field> Real;
  const Int height = NumRows();
  diagonal->Resize(height, 1, Real(0));
  for (const auto& perturbation : result.dynamic_regularization) {
    const Int index = perturbation.first;
    const Real regularization = perturbation.second;
    diagonal->Entry(index) = regularization;
  }
}

template <class Field>
SparseLDLResult<Field> SparseLDL<Field>::RefactorWithFixedSparsityPattern(
    const CoordinateMatrix<Field>& matrix) {
  typedef ComplexBase<Field> Real;

  // Optionally equilibrate the matrix.
  const CoordinateMatrix<Field>* matrix_to_factor;
  CoordinateMatrix<Field> equilibrated_matrix;
  if (have_equilibration_) {
    const bool kVerboseEquil = false;
    equilibrated_matrix = matrix;
    EquilibrateSymmetricMatrix(&equilibrated_matrix, &equilibration_,
                               kVerboseEquil);
    matrix_to_factor = &equilibrated_matrix;
  } else {
    matrix_to_factor = &matrix;
  }

  SparseLDLResult<Field> result;
  if (is_supernodal) {
    result = supernodal_factorization->RefactorWithFixedSparsityPattern(
        *matrix_to_factor);
  } else {
    result = scalar_factorization->RefactorWithFixedSparsityPattern(
        *matrix_to_factor);
  }
  if (have_equilibration_) {
    // We factored inv(D) A inv(D), so the regularization needs to be wrapped
    // with D . D.
    for (std::pair<Int, Real>& reg : result.dynamic_regularization) {
      reg.second *= equilibration_(reg.first) * equilibration_(reg.first);
    }
  }
  return result;
}

template <class Field>
SparseLDLResult<Field> SparseLDL<Field>::RefactorWithFixedSparsityPattern(
    const CoordinateMatrix<Field>& matrix,
    const SparseLDLControl<Field>& control) {
  typedef ComplexBase<Field> Real;

  // TODO(Jack Poulson): Add sanity checks here that, for example, the algorithm
  // hasn't changed.

  // Optionally equilibrate the matrix.
  const CoordinateMatrix<Field>* matrix_to_factor;
  CoordinateMatrix<Field> equilibrated_matrix;
  if (have_equilibration_) {
    const bool kVerboseEquil = false;
    equilibrated_matrix = matrix;
    EquilibrateSymmetricMatrix(&equilibrated_matrix, &equilibration_,
                               kVerboseEquil);
    matrix_to_factor = &equilibrated_matrix;
  } else {
    matrix_to_factor = &matrix;
  }

  SparseLDLResult<Field> result;
  if (is_supernodal) {
    result = supernodal_factorization->RefactorWithFixedSparsityPattern(
        *matrix_to_factor, control.supernodal_control);
  } else {
    result = scalar_factorization->RefactorWithFixedSparsityPattern(
        *matrix_to_factor, control.scalar_control);
  }
  if (have_equilibration_) {
    // We factored inv(D) A inv(D), so the regularization needs to be wrapped
    // with D . D.
    for (std::pair<Int, Real>& reg : result.dynamic_regularization) {
      reg.second *= equilibration_(reg.first) * equilibration_(reg.first);
    }
  }
  return result;
}

template <class Field>
void SparseLDL<Field>::Solve(BlasMatrixView<Field>* right_hand_sides) const {
  if (have_equilibration_) {
    // Apply the inverse of the equilibration matrix.
    for (Int j = 0; j < right_hand_sides->width; ++j) {
      for (Int i = 0; i < right_hand_sides->height; ++i) {
        right_hand_sides->Entry(i) /= equilibration_(i);
      }
    }
  }
  if (is_supernodal) {
    supernodal_factorization->Solve(right_hand_sides);
  } else {
    scalar_factorization->Solve(right_hand_sides);
  }
  if (have_equilibration_) {
    // Apply the inverse of the equilibration matrix.
    for (Int j = 0; j < right_hand_sides->width; ++j) {
      for (Int i = 0; i < right_hand_sides->height; ++i) {
        right_hand_sides->Entry(i) /= equilibration_(i);
      }
    }
  }
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>> SparseLDL<Field>::RefinedSolveHelper(
    const CoordinateMatrix<Field>& matrix,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides) const {
  auto apply_matrix = [&](Field alpha, const ConstBlasMatrixView<Field>& input,
                          Field beta, BlasMatrixView<Field>* output) {
    ApplySparse(alpha, matrix, input, beta, output);
  };

  auto apply_inverse = [&](BlasMatrixView<Field>* input) { Solve(input); };

  return catamari::RefinedSolve(apply_matrix, apply_inverse, control,
                                right_hand_sides);
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>>
SparseLDL<Field>::PromotedRefinedSolveHelper(
    const CoordinateMatrix<Field>& matrix,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides_lower) const {
  auto apply_matrix = [&](Promote<Field> alpha,
                          const ConstBlasMatrixView<Promote<Field>>& input,
                          Promote<Field> beta,
                          BlasMatrixView<Promote<Field>>* output) {
    ApplySparse(alpha, matrix, input, beta, output);
  };

  auto apply_inverse = [&](BlasMatrixView<Field>* input) { Solve(input); };

  return catamari::PromotedRefinedSolve(apply_matrix, apply_inverse, control,
                                        right_hand_sides_lower);
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>> SparseLDL<Field>::RefinedSolve(
    const CoordinateMatrix<Field>& matrix,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides) const {
  if (control.promote) {
    return PromotedRefinedSolveHelper(matrix, control, right_hand_sides);
  } else {
    return RefinedSolveHelper(matrix, control, right_hand_sides);
  }
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>>
SparseLDL<Field>::DynamicallyRegularizedRefinedSolveHelper(
    const CoordinateMatrix<Field>& matrix, const SparseLDLResult<Field>& result,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides) const {
  auto apply_matrix = [&](Field alpha, const ConstBlasMatrixView<Field>& input,
                          Field beta, BlasMatrixView<Field>* output) {
    ApplySparse(alpha, matrix, input, beta, output);
    for (const std::pair<Int, Real>& perturb : result.dynamic_regularization) {
      const Int i = perturb.first;
      const Real regularization = perturb.second;
      for (Int j = 0; j < output->width; ++j) {
        output->Entry(i, j) += alpha * regularization * input(i, j);
      }
    }
  };

  auto apply_inverse = [&](BlasMatrixView<Field>* input) { Solve(input); };

  return catamari::RefinedSolve(apply_matrix, apply_inverse, control,
                                right_hand_sides);
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>>
SparseLDL<Field>::PromotedDynamicallyRegularizedRefinedSolveHelper(
    const CoordinateMatrix<Field>& matrix, const SparseLDLResult<Field>& result,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides_lower) const {
  auto apply_matrix = [&](Promote<Field> alpha,
                          const ConstBlasMatrixView<Promote<Field>>& input,
                          Promote<Field> beta,
                          BlasMatrixView<Promote<Field>>* output) {
    ApplySparse(alpha, matrix, input, beta, output);
    for (const std::pair<Int, Real>& perturb : result.dynamic_regularization) {
      const Int i = perturb.first;
      const Promote<Real> regularization = perturb.second;
      for (Int j = 0; j < output->width; ++j) {
        output->Entry(i, j) += alpha * regularization * input(i, j);
      }
    }
  };

  auto apply_inverse = [&](BlasMatrixView<Field>* input) { Solve(input); };

  return catamari::PromotedRefinedSolve(apply_matrix, apply_inverse, control,
                                        right_hand_sides_lower);
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>>
SparseLDL<Field>::DynamicallyRegularizedRefinedSolve(
    const CoordinateMatrix<Field>& matrix, const SparseLDLResult<Field>& result,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides) const {
  if (control.promote) {
    return PromotedDynamicallyRegularizedRefinedSolveHelper(
        matrix, result, control, right_hand_sides);
  } else {
    return DynamicallyRegularizedRefinedSolveHelper(matrix, result, control,
                                                    right_hand_sides);
  }
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>>
SparseLDL<Field>::DiagonallyScaledRefinedSolveHelper(
    const CoordinateMatrix<Field>& matrix,
    const ConstBlasMatrixView<Real>& scaling,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides) const {
  BlasMatrix<Field> scaled_input;
  auto apply_matrix = [&](Field alpha, const ConstBlasMatrixView<Field>& input,
                          Field beta, BlasMatrixView<Field>* output) {
    scaled_input.Resize(input.height, input.width);
    for (Int j = 0; j < input.width; ++j) {
      for (Int i = 0; i < input.height; ++i) {
        scaled_input(i, j) = input(i, j) * scaling(i);
      }
    }
    ApplySparse(alpha, matrix, scaled_input.ConstView(), beta, output);
    for (Int j = 0; j < output->width; ++j) {
      for (Int i = 0; i < output->height; ++i) {
        output->Entry(i, j) *= scaling(i);
      }
    }
  };

  auto apply_inverse = [&](BlasMatrixView<Field>* input) {
    for (Int j = 0; j < input->width; ++j) {
      for (Int i = 0; i < input->height; ++i) {
        input->Entry(i, j) /= scaling(i);
      }
    }
    Solve(input);
    for (Int j = 0; j < input->width; ++j) {
      for (Int i = 0; i < input->height; ++i) {
        input->Entry(i, j) /= scaling(i);
      }
    }
  };

  // Scale the right-hand side.
  for (Int j = 0; j < right_hand_sides->width; ++j) {
    for (Int i = 0; i < right_hand_sides->height; ++i) {
      right_hand_sides->Entry(i, j) *= scaling(i);
    }
  }

  auto state = catamari::RefinedSolve(apply_matrix, apply_inverse, control,
                                      right_hand_sides);

  // *right_hand_sides := scaling * solution
  for (Int j = 0; j < right_hand_sides->width; ++j) {
    for (Int i = 0; i < right_hand_sides->height; ++i) {
      right_hand_sides->Entry(i, j) *= scaling(i);
    }
  }

  return state;
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>>
SparseLDL<Field>::PromotedDiagonallyScaledRefinedSolveHelper(
    const CoordinateMatrix<Field>& matrix,
    const ConstBlasMatrixView<Real>& scaling,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides_lower) const {
  BlasMatrix<Promote<Field>> scaled_input;
  auto apply_matrix = [&](Promote<Field> alpha,
                          const ConstBlasMatrixView<Promote<Field>>& input,
                          Promote<Field> beta,
                          BlasMatrixView<Promote<Field>>* output) {
    scaled_input.Resize(input.height, input.width);
    for (Int j = 0; j < input.width; ++j) {
      for (Int i = 0; i < input.height; ++i) {
        scaled_input(i, j) = input(i, j) * Promote<Real>(scaling(i));
      }
    }
    ApplySparse(alpha, matrix, scaled_input.ConstView(), beta, output);
    for (Int j = 0; j < output->width; ++j) {
      for (Int i = 0; i < output->height; ++i) {
        output->Entry(i, j) *= Promote<Real>(scaling(i));
      }
    }
  };

  auto apply_inverse = [&](BlasMatrixView<Field>* input) {
    for (Int j = 0; j < input->width; ++j) {
      for (Int i = 0; i < input->height; ++i) {
        input->Entry(i, j) /= scaling(i);
      }
    }
    Solve(input);
    for (Int j = 0; j < input->width; ++j) {
      for (Int i = 0; i < input->height; ++i) {
        input->Entry(i, j) /= scaling(i);
      }
    }
  };

  // Scale the right-hand side.
  for (Int j = 0; j < right_hand_sides_lower->width; ++j) {
    for (Int i = 0; i < right_hand_sides_lower->height; ++i) {
      right_hand_sides_lower->Entry(i, j) *= scaling(i);
    }
  }

  auto state = catamari::PromotedRefinedSolve(apply_matrix, apply_inverse,
                                              control, right_hand_sides_lower);

  // *right_hand_sides := scaling * solution
  for (Int j = 0; j < right_hand_sides_lower->width; ++j) {
    for (Int i = 0; i < right_hand_sides_lower->height; ++i) {
      right_hand_sides_lower->Entry(i, j) *= scaling(i);
    }
  }

  return state;
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>>
SparseLDL<Field>::DiagonallyScaledRefinedSolve(
    const CoordinateMatrix<Field>& matrix,
    const ConstBlasMatrixView<Real>& scaling,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides) const {
  if (control.promote) {
    return PromotedDiagonallyScaledRefinedSolveHelper(matrix, scaling, control,
                                                      right_hand_sides);
  } else {
    return DiagonallyScaledRefinedSolveHelper(matrix, scaling, control,
                                              right_hand_sides);
  }
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>>
SparseLDL<Field>::DiagonallyScaledDynamicallyRegularizedRefinedSolveHelper(
    const CoordinateMatrix<Field>& matrix, const SparseLDLResult<Field>& result,
    const ConstBlasMatrixView<Real>& scaling,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides) const {
  BlasMatrix<Field> scaled_input;
  auto apply_matrix = [&](Field alpha, const ConstBlasMatrixView<Field>& input,
                          Field beta, BlasMatrixView<Field>* output) {
    scaled_input.Resize(input.height, input.width);
    for (Int j = 0; j < input.width; ++j) {
      for (Int i = 0; i < input.height; ++i) {
        scaled_input(i, j) = input(i, j) * scaling(i);
      }
    }
    ApplySparse(alpha, matrix, scaled_input.ConstView(), beta, output);
    for (const std::pair<Int, Real>& perturb : result.dynamic_regularization) {
      const Int i = perturb.first;
      const Real regularization = perturb.second;
      for (Int j = 0; j < output->width; ++j) {
        output->Entry(i, j) += alpha * regularization * scaled_input(i, j);
      }
    }
    for (Int j = 0; j < output->width; ++j) {
      for (Int i = 0; i < output->height; ++i) {
        output->Entry(i, j) *= scaling(i);
      }
    }
  };

  auto apply_inverse = [&](BlasMatrixView<Field>* input) {
    for (Int j = 0; j < input->width; ++j) {
      for (Int i = 0; i < input->height; ++i) {
        input->Entry(i, j) /= scaling(i);
      }
    }
    Solve(input);
    for (Int j = 0; j < input->width; ++j) {
      for (Int i = 0; i < input->height; ++i) {
        input->Entry(i, j) /= scaling(i);
      }
    }
  };

  // Scale the right-hand side.
  for (Int j = 0; j < right_hand_sides->width; ++j) {
    for (Int i = 0; i < right_hand_sides->height; ++i) {
      right_hand_sides->Entry(i, j) *= scaling(i);
    }
  }

  auto state = catamari::RefinedSolve(apply_matrix, apply_inverse, control,
                                      right_hand_sides);

  // *right_hand_sides := scaling * solution
  for (Int j = 0; j < right_hand_sides->width; ++j) {
    for (Int i = 0; i < right_hand_sides->height; ++i) {
      right_hand_sides->Entry(i, j) *= scaling(i);
    }
  }

  return state;
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>> SparseLDL<Field>::
    PromotedDiagonallyScaledDynamicallyRegularizedRefinedSolveHelper(
        const CoordinateMatrix<Field>& matrix,
        const SparseLDLResult<Field>& result,
        const ConstBlasMatrixView<Real>& scaling,
        const RefinedSolveControl<Real>& control,
        BlasMatrixView<Field>* right_hand_sides_lower) const {
  BlasMatrix<Promote<Field>> scaled_input;
  auto apply_matrix = [&](Promote<Field> alpha,
                          const ConstBlasMatrixView<Promote<Field>>& input,
                          Promote<Field> beta,
                          BlasMatrixView<Promote<Field>>* output) {
    scaled_input.Resize(input.height, input.width);
    for (Int j = 0; j < input.width; ++j) {
      for (Int i = 0; i < input.height; ++i) {
        scaled_input(i, j) = input(i, j) * Promote<Real>(scaling(i));
      }
    }
    ApplySparse(alpha, matrix, scaled_input.ConstView(), beta, output);
    for (const std::pair<Int, Real>& perturb : result.dynamic_regularization) {
      const Int i = perturb.first;
      const Promote<Real> regularization = perturb.second;
      for (Int j = 0; j < output->width; ++j) {
        output->Entry(i, j) += alpha * regularization * scaled_input(i, j);
      }
    }
    for (Int j = 0; j < output->width; ++j) {
      for (Int i = 0; i < output->height; ++i) {
        output->Entry(i, j) *= Promote<Real>(scaling(i));
      }
    }
  };

  auto apply_inverse = [&](BlasMatrixView<Field>* input) {
    for (Int j = 0; j < input->width; ++j) {
      for (Int i = 0; i < input->height; ++i) {
        input->Entry(i, j) /= scaling(i);
      }
    }
    Solve(input);
    for (Int j = 0; j < input->width; ++j) {
      for (Int i = 0; i < input->height; ++i) {
        input->Entry(i, j) /= scaling(i);
      }
    }
  };

  // Scale the right-hand side.
  for (Int j = 0; j < right_hand_sides_lower->width; ++j) {
    for (Int i = 0; i < right_hand_sides_lower->height; ++i) {
      right_hand_sides_lower->Entry(i, j) *= scaling(i);
    }
  }

  auto state = catamari::PromotedRefinedSolve(apply_matrix, apply_inverse,
                                              control, right_hand_sides_lower);

  // *right_hand_sides := scaling * solution
  for (Int j = 0; j < right_hand_sides_lower->width; ++j) {
    for (Int i = 0; i < right_hand_sides_lower->height; ++i) {
      right_hand_sides_lower->Entry(i, j) *= scaling(i);
    }
  }

  return state;
}

template <class Field>
RefinedSolveStatus<ComplexBase<Field>>
SparseLDL<Field>::DiagonallyScaledDynamicallyRegularizedRefinedSolve(
    const CoordinateMatrix<Field>& matrix, const SparseLDLResult<Field>& result,
    const ConstBlasMatrixView<Real>& scaling,
    const RefinedSolveControl<Real>& control,
    BlasMatrixView<Field>* right_hand_sides) const {
  if (control.promote) {
    return PromotedDiagonallyScaledDynamicallyRegularizedRefinedSolveHelper(
        matrix, result, scaling, control, right_hand_sides);
  } else {
    return DiagonallyScaledDynamicallyRegularizedRefinedSolveHelper(
        matrix, result, scaling, control, right_hand_sides);
  }
}

template <class Field>
void SparseLDL<Field>::LowerTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  if (is_supernodal) {
    supernodal_factorization->LowerTriangularSolve(right_hand_sides);
  } else {
    scalar_factorization->LowerTriangularSolve(right_hand_sides);
  }
}

template <class Field>
void SparseLDL<Field>::DiagonalSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  if (is_supernodal) {
    supernodal_factorization->DiagonalSolve(right_hand_sides);
  } else {
    scalar_factorization->DiagonalSolve(right_hand_sides);
  }
}

template <class Field>
void SparseLDL<Field>::LowerTransposeTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  if (is_supernodal) {
    supernodal_factorization->LowerTransposeTriangularSolve(right_hand_sides);
  } else {
    scalar_factorization->LowerTransposeTriangularSolve(right_hand_sides);
  }
}

template <class Field>
void SparseLDL<Field>::PrintLowerFactor(const std::string& label,
                                        std::ostream& os) const {
  if (is_supernodal) {
    supernodal_factorization->PrintLowerFactor(label, os);
  } else {
    scalar_factorization->PrintLowerFactor(label, os);
  }
}

template <class Field>
void SparseLDL<Field>::PrintDiagonalFactor(const std::string& label,
                                           std::ostream& os) const {
  if (is_supernodal) {
    supernodal_factorization->PrintDiagonalFactor(label, os);
  } else {
    scalar_factorization->PrintDiagonalFactor(label, os);
  }
}

template <class Field>
void MergeDynamicRegularizations(
    const Buffer<SparseLDLResult<Field>>& children_results,
    SparseLDLResult<Field>* result) {
  Int reg_offset = result->dynamic_regularization.size();
  Int num_regularizations = reg_offset;
  for (const SparseLDLResult<Field>& child_result : children_results) {
    num_regularizations += child_result.dynamic_regularization.size();
  }
  result->dynamic_regularization.resize(num_regularizations);
  for (const SparseLDLResult<Field>& child_result : children_results) {
    std::copy(child_result.dynamic_regularization.begin(),
              child_result.dynamic_regularization.end(),
              result->dynamic_regularization.begin() + reg_offset);
    reg_offset += child_result.dynamic_regularization.size();
  }
}

}  // namespace catamari

#endif  // ifndef CATAMARI_SPARSE_LDL_IMPL_H_
