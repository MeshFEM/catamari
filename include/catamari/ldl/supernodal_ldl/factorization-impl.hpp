/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_IMPL_H_
#define CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_IMPL_H_

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"

#include "catamari/ldl/supernodal_ldl/factorization.hpp"

namespace catamari {
namespace supernodal_ldl {

template <class Field>
void Factorization<Field>::IncorporateSupernodeIntoLDLResult(
    Int supernode_size, Int degree, LDLResult* result) {
  // Finish updating the result structure.
  result->largest_supernode =
      std::max(result->largest_supernode, supernode_size);
  result->num_factorization_entries +=
      (supernode_size * (supernode_size + 1)) / 2 + supernode_size * degree;

  // Compute the number of flops for factoring the diagonal block.
  const double diagonal_flops = (IsComplex<Field>::value ? 4. : 1.) *
                                    std::pow(1. * supernode_size, 3.) / 3. +
                                std::pow(1. * supernode_size, 2.) / 2.;

  // Compute the number of flops to update the subdiagonal block.
  const double solve_flops = (IsComplex<Field>::value ? 4. : 1.) * degree *
                             std::pow(1. * supernode_size, 2.);

  // Compute the number of flops for forming the Schur complement.
  const double schur_complement_flops = (IsComplex<Field>::value ? 4. : 1.) *
                                        supernode_size *
                                        std::pow(1. * degree, 2.);

  result->num_diagonal_flops += diagonal_flops;
  result->num_subdiag_solve_flops += solve_flops;
  result->num_schur_complement_flops += schur_complement_flops;
  result->num_factorization_flops +=
      diagonal_flops + solve_flops + schur_complement_flops;
}

template <class Field>
void Factorization<Field>::MergeContribution(const LDLResult& contribution,
                                             LDLResult* result) {
  result->num_successful_pivots += contribution.num_successful_pivots;
  result->largest_supernode =
      std::max(result->largest_supernode, contribution.largest_supernode);
  result->num_factorization_entries += contribution.num_factorization_entries;

  result->num_diagonal_flops += contribution.num_diagonal_flops;
  result->num_subdiag_solve_flops += contribution.num_subdiag_solve_flops;
  result->num_schur_complement_flops += contribution.num_schur_complement_flops;
  result->num_factorization_flops += contribution.num_factorization_flops;
}

template <class Field>
void Factorization<Field>::FormSupernodes(const CoordinateMatrix<Field>& matrix,
                                          AssemblyForest* forest,
                                          Buffer<Int>* supernode_degrees) {
  // Greedily compute a supernodal partition using the original ordering.
  AssemblyForest orig_scalar_forest;
  SymmetricOrdering fund_ordering;
  fund_ordering.permutation = ordering_.permutation;
  fund_ordering.inverse_permutation = ordering_.inverse_permutation;
  scalar_ldl::LowerStructure scalar_structure;
  FormFundamentalSupernodes(matrix, ordering_, &orig_scalar_forest,
                            &fund_ordering.supernode_sizes, &scalar_structure);
  OffsetScan(fund_ordering.supernode_sizes, &fund_ordering.supernode_offsets);
  CATAMARI_ASSERT(fund_ordering.supernode_offsets.Back() == matrix.NumRows(),
                  "Supernodes did not sum to the matrix size.");

  Buffer<Int> fund_member_to_index;
  MemberToIndex(matrix.NumRows(), fund_ordering.supernode_offsets,
                &fund_member_to_index);

  const Int num_fund_supernodes = fund_ordering.supernode_sizes.Size();
  ConvertFromScalarToSupernodalEliminationForest(
      num_fund_supernodes, orig_scalar_forest.parents, fund_member_to_index,
      &fund_ordering.assembly_forest.parents);
  fund_ordering.assembly_forest.FillFromParents();

  Buffer<Int> fund_supernode_degrees;
  SupernodalDegrees(matrix, fund_ordering, orig_scalar_forest,
                    fund_member_to_index, &fund_supernode_degrees);

  const SupernodalRelaxationControl& relax_control =
      control_.relaxation_control;
  if (relax_control.relax_supernodes) {
    RelaxSupernodes(
        orig_scalar_forest.parents, fund_ordering.supernode_sizes,
        fund_ordering.supernode_offsets, fund_ordering.assembly_forest.parents,
        fund_supernode_degrees, fund_member_to_index, scalar_structure,
        relax_control, &ordering_.permutation, &ordering_.inverse_permutation,
        &forest->parents, &ordering_.assembly_forest.parents, supernode_degrees,
        &ordering_.supernode_sizes, &ordering_.supernode_offsets,
        &supernode_member_to_index_);
    forest->FillFromParents();
    ordering_.assembly_forest.FillFromParents();
  } else {
    *forest = orig_scalar_forest;

    ordering_.supernode_sizes = fund_ordering.supernode_sizes;
    ordering_.supernode_offsets = fund_ordering.supernode_offsets;
    ordering_.assembly_forest.parents = fund_ordering.assembly_forest.parents;
    ordering_.assembly_forest.FillFromParents();

    supernode_member_to_index_ = fund_member_to_index;
    *supernode_degrees = fund_supernode_degrees;
  }
}

template <class Field>
void Factorization<Field>::InitializeFactors(
    const CoordinateMatrix<Field>& matrix, const AssemblyForest& forest,
    const Buffer<Int>& supernode_degrees) {
  lower_factor_.reset(
      new LowerFactor<Field>(ordering_.supernode_sizes, supernode_degrees));
  diagonal_factor_.reset(new DiagonalFactor<Field>(ordering_.supernode_sizes));

  CATAMARI_ASSERT(supernode_degrees.Size() == ordering_.supernode_sizes.Size(),
                  "Invalid supernode degrees size.");

  // Store the largest supernode size of the factorization.
  max_supernode_size_ = *std::max_element(ordering_.supernode_sizes.begin(),
                                          ordering_.supernode_sizes.end());

  // Store the largest degree of the factorization for use in the solve phase.
  max_degree_ =
      *std::max_element(supernode_degrees.begin(), supernode_degrees.end());

  // Compute the maximum number of entries below the diagonal block of a
  // supernode.
  max_lower_block_size_ = 0;
  const Int num_supernodes = supernode_degrees.Size();
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int lower_block_size =
        supernode_degrees[supernode] * ordering_.supernode_sizes[supernode];
    max_lower_block_size_ = std::max(max_lower_block_size_, lower_block_size);
  }

  FillStructureIndices(matrix, ordering_, forest, supernode_member_to_index_,
                       lower_factor_.get());
  if (control_.algorithm == kLeftLookingLDL) {
    lower_factor_->FillIntersectionSizes(ordering_.supernode_sizes,
                                         supernode_member_to_index_);
  }

  FillZeros(ordering_, lower_factor_.get(), diagonal_factor_.get());
  FillNonzeros(matrix, ordering_, supernode_member_to_index_,
               lower_factor_.get(), diagonal_factor_.get());
}

template <class Field>
void Factorization<Field>::LeftLookingSupernodeUpdate(
    Int main_supernode, const CoordinateMatrix<Field>& matrix,
    LeftLookingSharedState* shared_state, PrivateState<Field>* private_state) {
  BlasMatrixView<Field>& main_diagonal_block =
      diagonal_factor_->blocks[main_supernode];
  BlasMatrixView<Field>& main_lower_block =
      lower_factor_->blocks[main_supernode];
  const Int main_supernode_size = main_lower_block.width;

  private_state->pattern_flags[main_supernode] = main_supernode;

  shared_state->rel_rows[main_supernode] = 0;
  shared_state->intersect_ptrs[main_supernode] =
      lower_factor_->IntersectionSizesBeg(main_supernode);

  // Compute the supernodal row pattern.
  const Int num_packed = ComputeRowPattern(
      matrix, ordering_.permutation, ordering_.inverse_permutation,
      ordering_.supernode_sizes, ordering_.supernode_offsets,
      supernode_member_to_index_, ordering_.assembly_forest.parents,
      main_supernode, private_state->pattern_flags.Data(),
      private_state->row_structure.Data());

  // for J = find(L(K, :))
  //   L(K:n, K) -= L(K:n, J) * (D(J, J) * L(K, J)')
  for (Int index = 0; index < num_packed; ++index) {
    const Int descendant_supernode = private_state->row_structure[index];
    CATAMARI_ASSERT(descendant_supernode < main_supernode,
                    "Looking into upper triangle");
    const ConstBlasMatrixView<Field>& descendant_lower_block =
        lower_factor_->blocks[descendant_supernode];
    const Int descendant_degree = descendant_lower_block.height;
    const Int descendant_supernode_size = descendant_lower_block.width;

    const Int descendant_main_rel_row =
        shared_state->rel_rows[descendant_supernode];
    const Int descendant_main_intersect_size =
        *shared_state->intersect_ptrs[descendant_supernode];

    const ConstBlasMatrixView<Field> descendant_main_matrix =
        descendant_lower_block.Submatrix(descendant_main_rel_row, 0,
                                         descendant_main_intersect_size,
                                         descendant_supernode_size);

    BlasMatrixView<Field> scaled_transpose;
    scaled_transpose.height = descendant_supernode_size;
    scaled_transpose.width = descendant_main_intersect_size;
    scaled_transpose.leading_dim = descendant_supernode_size;
    scaled_transpose.data = private_state->scaled_transpose_buffer.Data();

    FormScaledTranspose(
        control_.factorization_type,
        diagonal_factor_->blocks[descendant_supernode].ToConst(),
        descendant_main_matrix, &scaled_transpose);

    BlasMatrixView<Field> workspace_matrix;
    workspace_matrix.height = descendant_main_intersect_size;
    workspace_matrix.width = descendant_main_intersect_size;
    workspace_matrix.leading_dim = descendant_main_intersect_size;
    workspace_matrix.data = private_state->workspace_buffer.Data();

    UpdateDiagonalBlock(
        control_.factorization_type, ordering_.supernode_offsets,
        *lower_factor_, main_supernode, descendant_supernode,
        descendant_main_rel_row, descendant_main_matrix,
        scaled_transpose.ToConst(), &main_diagonal_block, &workspace_matrix);

    shared_state->intersect_ptrs[descendant_supernode]++;
    shared_state->rel_rows[descendant_supernode] +=
        descendant_main_intersect_size;

    // L(KNext:n, K) -= L(KNext:n, J) * (D(J, J) * L(K, J)')
    //                = L(KNext:n, J) * Z(J, K).
    const Int* descendant_active_intersect_size_beg =
        shared_state->intersect_ptrs[descendant_supernode];
    Int descendant_active_rel_row =
        shared_state->rel_rows[descendant_supernode];
    const Int* main_active_intersect_sizes =
        lower_factor_->IntersectionSizesBeg(main_supernode);
    Int main_active_rel_row = 0;
    while (descendant_active_rel_row != descendant_degree) {
      const Int descendant_active_intersect_size =
          *descendant_active_intersect_size_beg;

      SeekForMainActiveRelativeRow(
          main_supernode, descendant_supernode, descendant_active_rel_row,
          supernode_member_to_index_, *lower_factor_, &main_active_rel_row,
          &main_active_intersect_sizes);
      const Int main_active_intersect_size = *main_active_intersect_sizes;

      const ConstBlasMatrixView<Field> descendant_active_matrix =
          descendant_lower_block.Submatrix(descendant_active_rel_row, 0,
                                           descendant_active_intersect_size,
                                           descendant_supernode_size);

      // The width of the workspace matrix and pointer are already correct.
      workspace_matrix.height = descendant_active_intersect_size;
      workspace_matrix.leading_dim = descendant_active_intersect_size;

      BlasMatrixView<Field> main_active_block = main_lower_block.Submatrix(
          main_active_rel_row, 0, main_active_intersect_size,
          main_supernode_size);

      UpdateSubdiagonalBlock(
          main_supernode, descendant_supernode, main_active_rel_row,
          descendant_main_rel_row, descendant_active_rel_row,
          ordering_.supernode_offsets, supernode_member_to_index_,
          scaled_transpose.ToConst(), descendant_active_matrix, *lower_factor_,
          &main_active_block, &workspace_matrix);

      ++descendant_active_intersect_size_beg;
      descendant_active_rel_row += descendant_active_intersect_size;
    }
  }
}

template <class Field>
bool Factorization<Field>::LeftLookingSupernodeFinalize(Int main_supernode,
                                                        LDLResult* result) {
  BlasMatrixView<Field>& main_diagonal_block =
      diagonal_factor_->blocks[main_supernode];
  BlasMatrixView<Field>& main_lower_block =
      lower_factor_->blocks[main_supernode];
  const Int main_degree = main_lower_block.height;
  const Int main_supernode_size = main_lower_block.width;

  const Int num_supernode_pivots = FactorDiagonalBlock(
      control_.block_size, control_.factorization_type, &main_diagonal_block);
  result->num_successful_pivots += num_supernode_pivots;
  if (num_supernode_pivots < main_supernode_size) {
    return false;
  }
  IncorporateSupernodeIntoLDLResult(main_supernode_size, main_degree, result);
  if (!main_degree) {
    return true;
  }

  CATAMARI_ASSERT(main_supernode_size > 0, "Supernode size was non-positive.");
  SolveAgainstDiagonalBlock(control_.factorization_type,
                            main_diagonal_block.ToConst(), &main_lower_block);

  return true;
}

template <class Field>
void Factorization<Field>::InitialFactorizationSetup(
    const CoordinateMatrix<Field>& matrix) {
  AssemblyForest forest;
  Buffer<Int> supernode_degrees;
  FormSupernodes(matrix, &forest, &supernode_degrees);
  InitializeFactors(matrix, forest, supernode_degrees);
}

template <class Field>
LDLResult Factorization<Field>::LeftLooking(
    const CoordinateMatrix<Field>& matrix) {
#ifdef CATAMARI_OPENMP
  if (omp_get_max_threads() > 1) {
    return OpenMPLeftLooking(matrix);
  }
#endif
  const Int num_supernodes = ordering_.supernode_sizes.Size();

  LeftLookingSharedState shared_state;
  shared_state.rel_rows.Resize(num_supernodes);
  shared_state.intersect_ptrs.Resize(num_supernodes);

  PrivateState<Field> private_state;
  private_state.row_structure.Resize(num_supernodes);
  private_state.pattern_flags.Resize(num_supernodes);
  private_state.scaled_transpose_buffer.Resize(
      max_supernode_size_ * max_supernode_size_, Field{0});
  private_state.workspace_buffer.Resize(
      max_supernode_size_ * (max_supernode_size_ - 1), Field{0});

  LDLResult result;

  // Note that any postordering of the supernodal elimination forest suffices.
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    LeftLookingSupernodeUpdate(supernode, matrix, &shared_state,
                               &private_state);
    const bool succeeded = LeftLookingSupernodeFinalize(supernode, &result);
    if (!succeeded) {
      return result;
    }
  }

  return result;
}

template <class Field>
bool Factorization<Field>::RightLookingSupernodeFinalize(
    Int supernode, RightLookingSharedState<Field>* shared_state,
    PrivateState<Field>* private_state, LDLResult* result) {
  typedef ComplexBase<Field> Real;
  BlasMatrixView<Field>& diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrixView<Field>& lower_block = lower_factor_->blocks[supernode];
  const Int degree = lower_block.height;
  const Int supernode_size = lower_block.width;

  // Initialize this supernode's Schur complement as the zero matrix.
  Buffer<Field>& schur_complement_buffer =
      shared_state->schur_complement_buffers[supernode];
  BlasMatrixView<Field>& schur_complement =
      shared_state->schur_complements[supernode];
  schur_complement_buffer.Resize(degree * degree, Field{0});
  schur_complement.height = degree;
  schur_complement.width = degree;
  schur_complement.leading_dim = degree;
  schur_complement.data = schur_complement_buffer.Data();

  MergeChildSchurComplements(supernode, ordering_, lower_factor_.get(),
                             diagonal_factor_.get(), shared_state);

  const Int num_supernode_pivots = FactorDiagonalBlock(
      control_.block_size, control_.factorization_type, &diagonal_block);
  result->num_successful_pivots += num_supernode_pivots;
  if (num_supernode_pivots < supernode_size) {
    return false;
  }
  IncorporateSupernodeIntoLDLResult(supernode_size, degree, result);

  if (!degree) {
    // We can early exit.
    return true;
  }

  CATAMARI_ASSERT(supernode_size > 0, "Supernode size was non-positive.");
  SolveAgainstDiagonalBlock(control_.factorization_type,
                            diagonal_block.ToConst(), &lower_block);

  if (control_.factorization_type == kCholeskyFactorization) {
    LowerNormalHermitianOuterProduct(Real{-1}, lower_block.ToConst(), Real{1},
                                     &schur_complement);
  } else {
    BlasMatrixView<Field> scaled_transpose;
    scaled_transpose.height = supernode_size;
    scaled_transpose.width = degree;
    scaled_transpose.leading_dim = supernode_size;
    scaled_transpose.data = private_state->scaled_transpose_buffer.Data();
    FormScaledTranspose(control_.factorization_type, diagonal_block.ToConst(),
                        lower_block.ToConst(), &scaled_transpose);
    MatrixMultiplyLowerNormalNormal(Field{-1}, lower_block.ToConst(),
                                    scaled_transpose.ToConst(), Field{1},
                                    &schur_complement);
  }

  return true;
}

template <class Field>
bool Factorization<Field>::RightLookingSubtree(
    Int supernode, const CoordinateMatrix<Field>& matrix,
    RightLookingSharedState<Field>* shared_state,
    PrivateState<Field>* private_state, LDLResult* result) {
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;

  Buffer<int> successes(num_children);
  Buffer<LDLResult> result_contributions(num_children);

  // Recurse on the children.
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    CATAMARI_ASSERT(ordering_.assembly_forest.parents[child] == supernode,
                    "Incorrect child index");
    LDLResult& result_contribution = result_contributions[child_index];
    successes[child_index] = RightLookingSubtree(
        child, matrix, shared_state, private_state, &result_contribution);
  }

  // Merge the children's results (stopping if a failure is detected).
  bool succeeded = true;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    if (!successes[child_index]) {
      succeeded = false;
      break;
    }
    MergeContribution(result_contributions[child_index], result);
  }

  if (succeeded) {
    succeeded = RightLookingSupernodeFinalize(supernode, shared_state,
                                              private_state, result);
  } else {
    // Clear the child fronts.
    for (Int child_index = 0; child_index < num_children; ++child_index) {
      const Int child =
          ordering_.assembly_forest.children[child_beg + child_index];
      Buffer<Field>& child_schur_complement_buffer =
          shared_state->schur_complement_buffers[child];
      BlasMatrixView<Field>& child_schur_complement =
          shared_state->schur_complements[child];
      child_schur_complement.height = 0;
      child_schur_complement.width = 0;
      child_schur_complement.data = nullptr;
      child_schur_complement_buffer.Clear();
    }
  }

  return succeeded;
}

template <class Field>
LDLResult Factorization<Field>::RightLooking(
    const CoordinateMatrix<Field>& matrix) {
#ifdef CATAMARI_OPENMP
  if (omp_get_max_threads() > 1) {
    return OpenMPRightLooking(matrix);
  }
#endif
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const Int num_roots = ordering_.assembly_forest.roots.Size();

  RightLookingSharedState<Field> shared_state;
  shared_state.schur_complement_buffers.Resize(num_supernodes);
  shared_state.schur_complements.Resize(num_supernodes);

  PrivateState<Field> private_state;
  if (control_.factorization_type != kCholeskyFactorization) {
    private_state.scaled_transpose_buffer.Resize(max_lower_block_size_);
  }

  LDLResult result;

  Buffer<int> successes(num_roots);
  Buffer<LDLResult> result_contributions(num_roots);

  // Merge the child results (stopping if a failure is detected).
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = ordering_.assembly_forest.roots[root_index];
    LDLResult& result_contribution = result_contributions[root_index];
    successes[root_index] = RightLookingSubtree(
        root, matrix, &shared_state, &private_state, &result_contribution);
  }

  // Merge the child results (stopping if a failure is detected).
  for (Int index = 0; index < num_roots; ++index) {
    if (!successes[index]) {
      break;
    }
    MergeContribution(result_contributions[index], &result);
  }

  return result;
}

template <class Field>
bool Factorization<Field>::LeftLookingSubtree(
    Int supernode, const CoordinateMatrix<Field>& matrix,
    LeftLookingSharedState* shared_state, PrivateState<Field>* private_state,
    LDLResult* result) {
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;

  Buffer<int> successes(num_children);
  Buffer<LDLResult> result_contributions(num_children);

  // Recurse on the children.
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    CATAMARI_ASSERT(ordering_.assembly_forest.parents[child] == supernode,
                    "Incorrect child index");

    LDLResult& result_contribution = result_contributions[child_index];
    successes[child_index] = LeftLookingSubtree(
        child, matrix, shared_state, private_state, &result_contribution);
  }

  // Merge the child results (stopping if a failure is detected).
  bool succeeded = true;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    if (!successes[child_index]) {
      succeeded = false;
      break;
    }
    MergeContribution(result_contributions[child_index], result);
  }

  if (succeeded) {
    LeftLookingSupernodeUpdate(supernode, matrix, shared_state, private_state);
    succeeded = LeftLookingSupernodeFinalize(supernode, result);
  }

  return succeeded;
}

template <class Field>
LDLResult Factorization<Field>::Factor(const CoordinateMatrix<Field>& matrix,
                                       const SymmetricOrdering& manual_ordering,
                                       const Control& control) {
  control_ = control;
  ordering_ = manual_ordering;

#ifdef CATAMARI_OPENMP
  if (omp_get_max_threads() > 1) {
    #pragma omp parallel
    #pragma omp single
    OpenMPInitialFactorizationSetup(matrix);
  } else {
    InitialFactorizationSetup(matrix);
  }
#else
  InitialFactorizationSetup(matrix);
#endif

  if (control_.algorithm == kLeftLookingLDL) {
    return LeftLooking(matrix);
  } else {
    return RightLooking(matrix);
  }
}

template <class Field>
LDLResult Factorization<Field>::RefactorWithFixedSparsityPattern(
    const CoordinateMatrix<Field>& matrix) {
// TODO(Jack Poulson): Check that the previous factorization had an identical
// sparsity pattern.

#ifdef CATAMARI_OPENMP
  if (omp_get_max_threads() > 1) {
    #pragma omp parallel
    #pragma omp single
    {
      OpenMPFillZeros(ordering_, lower_factor_.get(), diagonal_factor_.get());
      OpenMPFillNonzeros(matrix, ordering_, supernode_member_to_index_,
                         lower_factor_.get(), diagonal_factor_.get());
    }
  } else {
    FillZeros(ordering_, lower_factor_.get(), diagonal_factor_.get());
    FillNonzeros(matrix, ordering_, supernode_member_to_index_,
                 lower_factor_.get(), diagonal_factor_.get());
  }
#else
  FillZeros(ordering_, lower_factor_.get(), diagonal_factor_.get());
  FillNonzeros(matrix, ordering_, supernode_member_to_index_,
               lower_factor_.get(), diagonal_factor_.get());
#endif

  if (control_.algorithm == kLeftLookingLDL) {
    return LeftLooking(matrix);
  } else {
    return RightLooking(matrix);
  }
}

template <class Field>
void Factorization<Field>::Solve(
    BlasMatrixView<Field>* right_hand_sides) const {
  const bool have_permutation = !ordering_.permutation.Empty();
  // Reorder the input into the permutation of the factorization.
  if (have_permutation) {
    Permute(ordering_.permutation, right_hand_sides);
  }

#ifdef CATAMARI_OPENMP
  if (omp_get_max_threads() > 1) {
    const int old_max_threads = GetMaxBlasThreads();
    SetNumBlasThreads(1);

    #pragma omp parallel
    #pragma omp single
    {
      OpenMPLowerTriangularSolve(right_hand_sides);
      OpenMPDiagonalSolve(right_hand_sides);
      OpenMPLowerTransposeTriangularSolve(right_hand_sides);
    }

    SetNumBlasThreads(old_max_threads);
  } else {
    LowerTriangularSolve(right_hand_sides);
    DiagonalSolve(right_hand_sides);
    LowerTransposeTriangularSolve(right_hand_sides);
  }
#else
  LowerTriangularSolve(right_hand_sides);
  DiagonalSolve(right_hand_sides);
  LowerTransposeTriangularSolve(right_hand_sides);
#endif  // ifdef CATAMARI_OPENMP

  // Reverse the factorization permutation.
  if (have_permutation) {
    Permute(ordering_.inverse_permutation, right_hand_sides);
  }
}

template <class Field>
void Factorization<Field>::LowerSupernodalTrapezoidalSolve(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    Buffer<Field>* workspace) const {
  // Eliminate this supernode.
  const Int num_rhs = right_hand_sides->width;
  const bool is_cholesky =
      control_.factorization_type == kCholeskyFactorization;
  const ConstBlasMatrixView<Field> triangular_right_hand_sides =
      diagonal_factor_->blocks[supernode];

  const Int supernode_size = ordering_.supernode_sizes[supernode];
  const Int supernode_start = ordering_.supernode_offsets[supernode];
  BlasMatrixView<Field> right_hand_sides_supernode =
      right_hand_sides->Submatrix(supernode_start, 0, supernode_size, num_rhs);

  // Solve against the diagonal block of the supernode.
  if (is_cholesky) {
    LeftLowerTriangularSolves(triangular_right_hand_sides,
                              &right_hand_sides_supernode);
  } else {
    LeftLowerUnitTriangularSolves(triangular_right_hand_sides,
                                  &right_hand_sides_supernode);
  }

  const ConstBlasMatrixView<Field> subdiagonal =
      lower_factor_->blocks[supernode];
  if (!subdiagonal.height) {
    return;
  }

  // Handle the external updates for this supernode.
  const Int* indices = lower_factor_->StructureBeg(supernode);
  if (supernode_size >=
      control_.forward_solve_out_of_place_supernode_threshold) {
    // Perform an out-of-place GEMM.
    BlasMatrixView<Field> work_right_hand_sides;
    work_right_hand_sides.height = subdiagonal.height;
    work_right_hand_sides.width = num_rhs;
    work_right_hand_sides.leading_dim = subdiagonal.height;
    work_right_hand_sides.data = workspace->Data();

    // Store the updates in the workspace.
    MatrixMultiplyNormalNormal(Field{-1}, subdiagonal,
                               right_hand_sides_supernode.ToConst(), Field{0},
                               &work_right_hand_sides);

    // Accumulate the workspace into the solution right_hand_sides.
    for (Int j = 0; j < num_rhs; ++j) {
      for (Int i = 0; i < subdiagonal.height; ++i) {
        const Int row = indices[i];
        right_hand_sides->Entry(row, j) += work_right_hand_sides(i, j);
      }
    }
  } else {
    for (Int j = 0; j < num_rhs; ++j) {
      for (Int k = 0; k < supernode_size; ++k) {
        const Field& eta = right_hand_sides_supernode(k, j);
        for (Int i = 0; i < subdiagonal.height; ++i) {
          const Int row = indices[i];
          right_hand_sides->Entry(row, j) -= subdiagonal(i, k) * eta;
        }
      }
    }
  }
}

template <class Field>
void Factorization<Field>::LowerTriangularSolveRecursion(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    Buffer<Field>* workspace) const {
  // Recurse on this supernode's children.
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    LowerTriangularSolveRecursion(child, right_hand_sides, workspace);
  }

  // Perform this supernode's trapezoidal solve.
  LowerSupernodalTrapezoidalSolve(supernode, right_hand_sides, workspace);
}

template <class Field>
void Factorization<Field>::LowerTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  // Allocate the workspace.
  const Int workspace_size = max_degree_ * right_hand_sides->width;
  Buffer<Field> workspace(workspace_size, Field{0});

  // Recurse on each tree in the elimination forest.
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = ordering_.assembly_forest.roots[root_index];
    LowerTriangularSolveRecursion(root, right_hand_sides, &workspace);
  }
}

template <class Field>
void Factorization<Field>::DiagonalSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  const Int num_rhs = right_hand_sides->width;
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const bool is_cholesky =
      control_.factorization_type == kCholeskyFactorization;
  if (is_cholesky) {
    // D is the identity.
    return;
  }

  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const ConstBlasMatrixView<Field> diagonal_right_hand_sides =
        diagonal_factor_->blocks[supernode];

    const Int supernode_size = ordering_.supernode_sizes[supernode];
    const Int supernode_start = ordering_.supernode_offsets[supernode];
    BlasMatrixView<Field> right_hand_sides_supernode =
        right_hand_sides->Submatrix(supernode_start, 0, supernode_size,
                                    num_rhs);

    // Handle the diagonal-block portion of the supernode.
    for (Int j = 0; j < num_rhs; ++j) {
      for (Int i = 0; i < supernode_size; ++i) {
        right_hand_sides_supernode(i, j) /= diagonal_right_hand_sides(i, i);
      }
    }
  }
}

template <class Field>
void Factorization<Field>::LowerTransposeSupernodalTrapezoidalSolve(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    Buffer<Field>* packed_input_buf) const {
  const Int num_rhs = right_hand_sides->width;
  const bool is_selfadjoint =
      control_.factorization_type != kLDLTransposeFactorization;
  const Int supernode_size = ordering_.supernode_sizes[supernode];
  const Int supernode_start = ordering_.supernode_offsets[supernode];
  const Int* indices = lower_factor_->StructureBeg(supernode);

  BlasMatrixView<Field> right_hand_sides_supernode =
      right_hand_sides->Submatrix(supernode_start, 0, supernode_size, num_rhs);

  const ConstBlasMatrixView<Field> subdiagonal =
      lower_factor_->blocks[supernode];
  if (subdiagonal.height) {
    // Handle the external updates for this supernode.
    if (supernode_size >=
        control_.backward_solve_out_of_place_supernode_threshold) {
      // Fill the work right_hand_sides.
      BlasMatrixView<Field> work_right_hand_sides;
      work_right_hand_sides.height = subdiagonal.height;
      work_right_hand_sides.width = num_rhs;
      work_right_hand_sides.leading_dim = subdiagonal.height;
      work_right_hand_sides.data = packed_input_buf->Data();
      for (Int j = 0; j < num_rhs; ++j) {
        for (Int i = 0; i < subdiagonal.height; ++i) {
          const Int row = indices[i];
          work_right_hand_sides(i, j) = right_hand_sides->Entry(row, j);
        }
      }

      if (is_selfadjoint) {
        MatrixMultiplyAdjointNormal(Field{-1}, subdiagonal,
                                    work_right_hand_sides.ToConst(), Field{1},
                                    &right_hand_sides_supernode);
      } else {
        MatrixMultiplyTransposeNormal(Field{-1}, subdiagonal,
                                      work_right_hand_sides.ToConst(), Field{1},
                                      &right_hand_sides_supernode);
      }
    } else {
      for (Int k = 0; k < supernode_size; ++k) {
        for (Int i = 0; i < subdiagonal.height; ++i) {
          const Int row = indices[i];
          for (Int j = 0; j < num_rhs; ++j) {
            if (is_selfadjoint) {
              right_hand_sides_supernode(k, j) -=
                  Conjugate(subdiagonal(i, k)) *
                  right_hand_sides->Entry(row, j);
            } else {
              right_hand_sides_supernode(k, j) -=
                  subdiagonal(i, k) * right_hand_sides->Entry(row, j);
            }
          }
        }
      }
    }
  }

  // Solve against the diagonal block of this supernode.
  const ConstBlasMatrixView<Field> triangular_right_hand_sides =
      diagonal_factor_->blocks[supernode];
  if (control_.factorization_type == kCholeskyFactorization) {
    LeftLowerAdjointTriangularSolves(triangular_right_hand_sides,
                                     &right_hand_sides_supernode);
  } else if (control_.factorization_type == kLDLAdjointFactorization) {
    LeftLowerAdjointUnitTriangularSolves(triangular_right_hand_sides,
                                         &right_hand_sides_supernode);
  } else {
    LeftLowerTransposeUnitTriangularSolves(triangular_right_hand_sides,
                                           &right_hand_sides_supernode);
  }
}

template <class Field>
void Factorization<Field>::LowerTransposeTriangularSolveRecursion(
    Int supernode, BlasMatrixView<Field>* right_hand_sides,
    Buffer<Field>* packed_input_buf) const {
  // Perform this supernode's trapezoidal solve.
  LowerTransposeSupernodalTrapezoidalSolve(supernode, right_hand_sides,
                                           packed_input_buf);

  // Recurse on this supernode's children.
  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;
  for (Int child_index = 0; child_index < num_children; ++child_index) {
    const Int child =
        ordering_.assembly_forest.children[child_beg + child_index];
    LowerTransposeTriangularSolveRecursion(child, right_hand_sides,
                                           packed_input_buf);
  }
}

template <class Field>
void Factorization<Field>::LowerTransposeTriangularSolve(
    BlasMatrixView<Field>* right_hand_sides) const {
  // Allocate the workspace.
  const Int workspace_size = max_degree_ * right_hand_sides->width;
  Buffer<Field> packed_input_buf(workspace_size);

  // Recurse from each root of the elimination forest.
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  for (Int root_index = 0; root_index < num_roots; ++root_index) {
    const Int root = ordering_.assembly_forest.roots[root_index];
    LowerTransposeTriangularSolveRecursion(root, right_hand_sides,
                                           &packed_input_buf);
  }
}

template <class Field>
void Factorization<Field>::PrintDiagonalFactor(const std::string& label,
                                               std::ostream& os) const {
  os << label << ": \n";
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const ConstBlasMatrixView<Field>& diag_matrix =
        diagonal_factor_->blocks[supernode];
    if (control_.factorization_type == kCholeskyFactorization) {
      for (Int j = 0; j < diag_matrix.height; ++j) {
        os << "1 ";
      }
    } else {
      for (Int j = 0; j < diag_matrix.height; ++j) {
        os << diag_matrix(j, j) << " ";
      }
    }
  }
  os << std::endl;
}

template <class Field>
void Factorization<Field>::PrintLowerFactor(const std::string& label,
                                            std::ostream& os) const {
  const bool is_cholesky =
      control_.factorization_type == kCholeskyFactorization;

  auto print_entry = [&](const Int& row, const Int& column,
                         const Field& value) {
    os << row << " " << column << " " << value << "\n";
  };

  os << label << ": \n";
  const Int num_supernodes = ordering_.supernode_sizes.Size();
  for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
    const Int supernode_start = ordering_.supernode_offsets[supernode];
    const Int* indices = lower_factor_->StructureBeg(supernode);

    const ConstBlasMatrixView<Field>& diag_matrix =
        diagonal_factor_->blocks[supernode];
    const ConstBlasMatrixView<Field>& lower_matrix =
        lower_factor_->blocks[supernode];

    for (Int j = 0; j < diag_matrix.height; ++j) {
      const Int column = supernode_start + j;

      // Print the portion in the diagonal block.
      if (is_cholesky) {
        print_entry(column, column, diag_matrix(j, j));
      } else {
        print_entry(column, column, Field{1});
      }
      for (Int k = j + 1; k < diag_matrix.height; ++k) {
        const Int row = supernode_start + k;
        print_entry(row, column, diag_matrix(k, j));
      }

      // Print the portion below the diagonal block.
      for (Int i = 0; i < lower_matrix.height; ++i) {
        const Int row = indices[i];
        print_entry(row, column, lower_matrix(i, j));
      }
    }
  }
  os << std::endl;
}

}  // namespace supernodal_ldl
}  // namespace catamari

#include "catamari/ldl/supernodal_ldl/factorization/openmp-impl.hpp"

#endif  // ifndef CATAMARI_LDL_SUPERNODAL_LDL_FACTORIZATION_IMPL_H_
