/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_RIGHT_LOOKING_OPENMP_IMPL_H_
#define CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_RIGHT_LOOKING_OPENMP_IMPL_H_

#include <algorithm>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/dense_factorizations.hpp"
#include "catamari/io_utils.hpp"

#include "catamari/sparse_ldl/supernodal/factorization.hpp"

#include "../../../../../../../src/lib/MeshFEM/GlobalBenchmark.hh"
#include "../../../../../../../src/lib/MeshFEM/ParallelVectorOps.hh"

#define FINEGRAINED_PARALLELISM 0

namespace catamari {
namespace supernodal_ldl {

template <class Field>
bool Factorization<Field>::OpenMPRightLookingSupernodeFinalize(
    Int supernode, const DynamicRegularizationParams<Field>& dynamic_reg_params,
    RightLookingSharedState<Field>* shared_state,
    Buffer<PrivateState<Field>>* private_states,
    SparseLDLResult<Field>* result) {
  typedef ComplexBase<Field> Real;
  BlasMatrixView<Field> diagonal_block = diagonal_factor_->blocks[supernode];
  BlasMatrixView<Field> lower_block = lower_factor_->blocks[supernode];
  const Int degree = lower_block.height;
  const Int supernode_size = lower_block.width;
  const bool has_children = ordering_.assembly_forest.child_offsets[supernode + 1] > ordering_.assembly_forest.child_offsets[supernode];

  Int num_supernode_pivots;
  if (control_.supernodal_pivoting) {
    // TODO(Jack Poulson): Add support for OpenMP supernodal pivoting.
    BlasMatrixView<Int> permutation = SupernodePermutation(supernode);
    num_supernode_pivots = PivotedFactorDiagonalBlock(
        control_.block_size, control_.factorization_type, &diagonal_block,
        &permutation);
    result->num_successful_pivots += num_supernode_pivots;
  } else {
#if FINEGRAINED_PARALLELISM
    // TODO(Jack Poulson): Preallocate this buffer.
    Buffer<Field> multithreaded_buffer;
    // Buffer is not for Cholesky...
    if (control_.factorization_type != kCholeskyFactorization)
        multithreaded_buffer.Resize(supernode_size * supernode_size);
#endif
    {
#if FINEGRAINED_PARALLELISM
      #pragma omp taskgroup
      num_supernode_pivots = OpenMPFactorDiagonalBlock(
          control_.factor_tile_size, control_.block_size,
          control_.factorization_type, dynamic_reg_params, &diagonal_block,
          &multithreaded_buffer, &result->dynamic_regularization);
#else
      num_supernode_pivots = FactorDiagonalBlock(
          control_.block_size,
          control_.factorization_type, dynamic_reg_params, &diagonal_block,
          &result->dynamic_regularization);
#endif
      result->num_successful_pivots += num_supernode_pivots;

    }
  }
  if (num_supernode_pivots < supernode_size) {
    return false;
  }
  IncorporateSupernodeIntoLDLResult(supernode_size, degree, result);

  if (!degree) {
    // We can early exit.
    return true;
  }

  CATAMARI_ASSERT(supernode_size > 0, "Supernode size was non-positive.");
  if (control_.supernodal_pivoting) {
    // Solve against P^T from the right, which is the same as applying P
    // from the right, which is the same as applying P^T to each row.
    const ConstBlasMatrixView<Int> permutation =
        SupernodePermutation(supernode);
    InversePermuteColumns(permutation, &lower_block);
  }

#if 1
  SolveAgainstDiagonalBlock(control_.factorization_type,
                            diagonal_block.ToConst(), &lower_block);
#else
  // TODO: try constructing and using the *transpose* of `lower_block`;
  // this seems like it would be more efficient (e.g., it lends itself to
  // a more efficient naive implementations of dsyrk).
  // Solve against the lower-triangular matrix L(K, K)' from the right.
  Eigen::Matrix<Field, Eigen::Dynamic, Eigen::Dynamic> lower_block_transpose = Eigen::Map<const Eigen::Matrix<Field, Eigen::Dynamic, Eigen::Dynamic>>(lower_block.data, lower_block.height, lower_block.width).transpose();
  {
      const char side = 'L';
      const char uplo = 'L';
      const char trans_triang = 'N';
      const char diag = 'N';
      const BlasInt height_blas = lower_block_transpose.rows();
      const BlasInt width_blas = lower_block_transpose.cols();
      const double alpha = 1;
      const BlasInt triang_leading_dim_blas = diagonal_block.leading_dim;
      const BlasInt leading_dim_blas = height_blas;

      BLAS_SYMBOL(dtrsm)
      (&side, &uplo, &trans_triang, &diag, &height_blas, &width_blas, &alpha,
       (const double *) diagonal_block.data, &triang_leading_dim_blas, (double *) lower_block_transpose.data(),
       &leading_dim_blas);
  }
#endif

  if (control_.factorization_type == kCholeskyFactorization) {
    BlasMatrixView<Field>& schur_complement = shared_state->schur_complements[supernode];
#if 1
    LowerNormalHermitianOuterProductDynamicBLASDispatch(
                                     Real{-1}, lower_block.ToConst(),
                                     has_children ? Real{1} : Real{0}, &schur_complement);
#else
    LowerNormalHermitianOuterProductDynamicBLASDispatch(
                                     Real{-1}, lower_block_transpose,
                                     has_children ? Real{1} : Real{0}, &schur_complement);
#endif
  } else {
    const int thread = tbb::this_task_arena::current_thread_index(); // TODO(Julian Panetta): switch to thread-local storage
    PrivateState<Field> &private_state = (*private_states)[thread];
    BlasMatrixView<Field> scaled_transpose;
    scaled_transpose.height = supernode_size;
    scaled_transpose.width = degree;
    scaled_transpose.leading_dim = supernode_size;
    scaled_transpose.data = private_state.scaled_transpose_buffer.Data();

#if 0 // These parallelizations don't seem to make a huge difference
    #pragma omp taskgroup
    OpenMPFormScaledTranspose(
        control_.outer_product_tile_size, control_.factorization_type,
        diagonal_block.ToConst(), lower_block.ToConst(), &scaled_transpose);
#else
    FormScaledTranspose(
        control_.factorization_type,
        diagonal_block.ToConst(), lower_block.ToConst(), &scaled_transpose);
#endif
    BlasMatrixView<Field>& schur_complement = shared_state->schur_complements[supernode];

#if 0 // These parallelizations don't seem to make a huge difference
    // Note: does its own #paragma omp taskgroup...
    OpenMPMatrixMultiplyLowerNormalNormal(
        control_.outer_product_tile_size, Field{-1}, lower_block.ToConst(),
        scaled_transpose.ToConst(), Field{1}, &schur_complement);
#else
    MatrixMultiplyLowerNormalNormal(
        Field{-1}, lower_block.ToConst(),
        scaled_transpose.ToConst(), Field{1}, &schur_complement);
#endif
  }

  return true;
}

template <class Field>
void MergeChildSchurComplement(Int supernode, Int child,
                               const SymmetricOrdering& ordering,
                               const LowerFactor<Field> *lower_factor,
                               const BlasMatrixView<Field> &child_schur_complement,
                               BlasMatrixView<Field> lower_block,
                               BlasMatrixView<Field> diagonal_block,
                               BlasMatrixView<Field> schur_complement,
                               Factorization<Field> &ldl,
                               bool first_merge) {
    const Int child_degree = child_schur_complement.height;
    const Int sno = ordering.supernode_offsets[supernode];
    populateChildToParentMap(supernode, child, child_degree, ordering, lower_factor);

    // Number of child rows/cols that map to the parent's diagonal block.
    const Int num_child_diag_indices = ordering.assembly_forest.num_child_diag_indices[child];

    // Locations of child's rows/cols relative to the parent front's upper-left corner
    const Buffer<Int> &child_rel_indices = ordering.assembly_forest.child_rel_indices[child];

    const Int supernode_size = ordering.supernode_sizes[supernode];

    if (first_merge) {
        // Initialize each of the supernode's columns of the factor
        // and merge in the first child's Schur complement.
        for (Int j = 0, cj = 0; j < supernode_size; ++j) {
            ldl.InitializeFactorColumn(sno + j, j, diagonal_block);
            Field* factor_column = diagonal_block.Pointer(0, j);

            if (child_rel_indices[cj] != j) continue;

            const Field* child_column = child_schur_complement.Pointer(0, cj);
            factor_column[j] += child_column[cj]; // diagonal entry
            for (Int i = cj + 1; i < child_degree; ++i)
                factor_column[child_rel_indices[i]] += child_column[i];
            ++cj;
        }

#if 1 // This version seems faster...
        eigenMap(schur_complement).setZero();
        for (Int j = num_child_diag_indices; j < child_degree; ++j) {
            const Field* child_column = child_schur_complement.Pointer(0, j);
            // Get pointer to the (conceptual) full parent front column, of which schur_complement is the bottom part.
            // Note: parent front's upper-left corner is (-supernode_size, -supernode_size) relative to this block...
            Field* schur_column = schur_complement.Pointer(-supernode_size, child_rel_indices[j] - supernode_size);
            for (Int i = j; i < child_degree; ++i)
                schur_column[child_rel_indices[i]] = child_column[i];
        }
#else
        const Int sc_size = schur_complement.width;
        // Clear and contribute into the bottom-right block of the front.
        for (Int j = 0, cj = num_child_diag_indices; j < sc_size; ++j) {
            Int front_j = j + supernode_size;
            Field *schur_column = schur_complement.Pointer(-supernode_size, j);
            using VMap = Eigen::Map<Eigen::Matrix<Field, Eigen::Dynamic, 1>>;
            VMap(schur_complement.Pointer(0, j), sc_size).setZero();

            if (child_rel_indices[cj] != front_j) continue;
            const Field* child_column = child_schur_complement.Pointer(0, cj);
            for (Int i = cj; i < child_degree; ++i)
                schur_column[child_rel_indices[i]] += child_column[i];
            ++cj;
        }
#endif
    }
    else {
        // Add the child Schur complement into this supernode's front.
        for (Int j = 0; j < num_child_diag_indices; ++j) {
            const Field* child_column = child_schur_complement.Pointer(0, j);
            Field* factor_column = diagonal_block.Pointer(0, child_rel_indices[j]);
            for (Int i = j; i < child_degree; ++i)
                factor_column[child_rel_indices[i]] += child_column[i];
        }

        // Contribute into the bottom-right block of the front.
        for (Int j = num_child_diag_indices; j < child_degree; ++j) {
            const Field* child_column = child_schur_complement.Pointer(0, j);
            Field* schur_column = schur_complement.Pointer(-supernode_size, child_rel_indices[j] - supernode_size);
            for (Int i = j; i < child_degree; ++i)
                schur_column[child_rel_indices[i]] += child_column[i];
        }
    }
}

template <class Field>
void MergeChildSchurComplements(Int supernode, Factorization<Field> &ldl,
                                const Buffer<BlasMatrixView<Field>> &schur_complements) {
    using VMap = Eigen::Map<Eigen::Matrix<Field, Eigen::Dynamic, 1>>;

    const auto &o = ldl.ordering_;
    const auto &af = o.assembly_forest;
    const Int child_beg = af.child_offsets[supernode];
    const Int child_end = af.child_offsets[supernode + 1];
    const Int num_children = child_end - child_beg;
    const Int sno = o.supernode_offsets[supernode];

    // Output destination buffers
    BlasMatrixView<Field> lower_block      = ldl.lower_factor_->blocks[supernode];
    BlasMatrixView<Field> diagonal_block   = ldl.diagonal_factor_->blocks[supernode];
    BlasMatrixView<Field> schur_complement = schur_complements[supernode];

    for (Int child_index = child_beg; child_index < child_end; ++child_index) {
        const Int child = af.children[child_index];
        const Int child_degree = schur_complements[child].height;
        populateChildToParentMap(supernode, child, child_degree, o, ldl.lower_factor_.get());
    }

    const Int supernode_size = o.supernode_sizes[supernode];
    std::vector<size_t> child_j(num_children); // pointer into the child columns
    const Int factor_height = diagonal_block.Height() + lower_block.Height();

    // tbb::parallel_for(tbb::blocked_range<catamari::Int>(0, supernode_size), [&](const tbb::blocked_range<catamari::Int> &r) {
    //   for (Int j = r.begin(); j < r.end(); ++j) {

    for (Int j = 0; j < supernode_size; ++j) {
        ldl.InitializeFactorColumn(sno + j, j, diagonal_block);
        Field* factor_column = diagonal_block.Pointer(0, j);
        for (Int ci = 0; ci < num_children; ++ci) {
            Int cj = child_j[ci];

            const Int child = af.children[child_beg + ci];
            const Int num_child_diag_indices = af.num_child_diag_indices[child];
            const Buffer<Int> &child_rel_indices = af.child_rel_indices[child];

            // Note: if we merge columns in parallel, we cannot use the `cj` pointer
            // and instead have to search for the current column `j` in child_rel_indices.
            //      auto end = child_rel_indices.begin() + num_child_diag_indices;
            //      auto it = std::find(child_rel_indices.begin(), end, j);
            //      if (it == end) continue;
            //      const Int cj = std::distance(child_rel_indices.begin(), it);
            if (child_rel_indices[cj] != j) continue;

            const BlasMatrixView<Field> &child_schur_complement = schur_complements[child];
            const Int child_degree = child_schur_complement.height;
            const Field* child_column = child_schur_complement.Pointer(0, cj);

            factor_column[j] += child_column[cj]; // diagonal entry
            for (Int i = cj + 1; i < child_degree; ++i)
                factor_column[child_rel_indices[i]] += child_column[i];

            child_j[ci] = ++cj;
        }
    } // });

    const Int sc_size = schur_complement.width;
    // tbb::parallel_for(tbb::blocked_range<catamari::Int>(0, sc_size), [&](const tbb::blocked_range<catamari::Int> &r) {
    //     for (Int j = r.begin(); j < r.end(); ++j) {
    for (Int j = 0; j < sc_size; ++j) {
        Int front_j = j + supernode_size;
        Field *schur_column = schur_complement.Pointer(-supernode_size, j);
        VMap(schur_complement.Pointer(0, j), sc_size).setZero();

        for (Int ci = 0; ci < num_children; ++ci) {
            Int cj = child_j[ci];

            const Int child = af.children[child_beg + ci];
            const Int num_child_diag_indices = af.num_child_diag_indices[child];
            const Buffer<Int> &child_rel_indices = af.child_rel_indices[child];

            // auto it = std::lower_bound(child_rel_indices.begin() + num_child_diag_indices, child_rel_indices.end(), front_j);
            // if (it == child_rel_indices.end() || *it != front_j) continue;
            // Int cj = std::distance(child_rel_indices.begin(), it);

            if (child_rel_indices[cj] != front_j) continue;

            const BlasMatrixView<Field> &child_schur_complement = schur_complements[child];
            const Int child_degree = child_schur_complement.height;

            const Field* child_column = child_schur_complement.Pointer(0, cj);
            for (Int i = cj; i < child_degree; ++i)
                schur_column[child_rel_indices[i]] += child_column[i];

            child_j[ci] = ++cj;
        }
    }//  });
}

template <class Field>
bool Factorization<Field>::OpenMPRightLookingSubtree(
    Int supernode, const CoordinateMatrix<Field>& matrix,
    const DynamicRegularizationParams<Field>& dynamic_reg_params,
    const Buffer<double>& work_estimates, double min_parallel_work,
    RightLookingSharedState<Field>* shared_state,
    Buffer<PrivateState<Field>>* private_states,
    SparseLDLResult<Field>* result) {
  const double work_estimate = work_estimates[supernode];

  const Int child_beg = ordering_.assembly_forest.child_offsets[supernode];
  const Int child_end = ordering_.assembly_forest.child_offsets[supernode + 1];
  const Int num_children = child_end - child_beg;

  bool fail = false;

  const bool parallel = (work_estimate >= min_parallel_work) && (num_children > 1);

#if ALLOCATE_SCHUR_COMPLEMENT_OTF
  // Allocate a single buffer to hold all the children's Schur complements
  // (or just the single largest one in the non-parallel case, where the buffer is reused).
  Eigen::Matrix<Field, Eigen::Dynamic, 1> child_schur_complement_buffer;
  if (num_children > 0) {
      Int total_size = 0;
      for (Int child_index = child_beg; child_index < child_end; ++child_index) {
          const Int child = ordering_.assembly_forest.children[child_index];
          const Int degree = lower_factor_->blocks[child].height;
          if (parallel) total_size += degree * degree;
          else          total_size = std::max(total_size, degree * degree);
      }
      child_schur_complement_buffer.resize(total_size);

      Int offset = 0;
      for (Int child_index = child_beg; child_index < child_end; ++child_index) {
          const Int child = ordering_.assembly_forest.children[child_index];
          const Int degree = lower_factor_->blocks[child].height;
          BlasMatrixView<Field>& child_schur_complement = shared_state->schur_complements[child];
          child_schur_complement.height = degree;
          child_schur_complement.width = degree;
          child_schur_complement.leading_dim = degree;
          child_schur_complement.data = child_schur_complement_buffer.data() + offset;
          if (parallel) offset += degree * degree; // re-use memory in the single-threaded case (offset === 0)
      }
  }
#endif

  // Clear this supernode's factor columns and load matrix entries into them.
  auto init = [&](){
    BlasMatrixView<Field> diagonal_block = diagonal_factor_->blocks[supernode];
    const Int sno = ordering_.supernode_offsets[supernode];
    const Int supernode_size = ordering_.supernode_sizes[supernode];
    for (Int j = 0; j < supernode_size; ++j)
        InitializeFactorColumn(sno + j, j, diagonal_block);
  };

  // std::vector<Int> sorted_children(num_children);
  // for (Int child_index = 0; child_index < num_children; ++child_index)
  //     sorted_children[child_index] = ordering_.assembly_forest.children[child_beg + child_index];
  // std::sort(sorted_children.begin(), sorted_children.end(),
  //           [&](Int a, Int b) { return work_estimates[a] > work_estimates[b]; });

  Buffer<SparseLDLResult<Field>> result_contributions(num_children);
  auto process_child = [&, supernode, min_parallel_work, shared_state, private_states](Int child, Int child_index) {
      const Int child_offset = ordering_.supernode_offsets[child];
      DynamicRegularizationParams<Field> subparams = dynamic_reg_params;
      subparams.offset = child_offset;
      bool success = OpenMPRightLookingSubtree(
              child, matrix, subparams, work_estimates, min_parallel_work,
              shared_state, private_states, &result_contributions[child_index]);
      if (!success) fail = true;
  };

  if (!parallel) {
      // Output destination buffers
      BlasMatrixView<Field> lower_block      = lower_factor_->blocks[supernode];
      BlasMatrixView<Field> diagonal_block   = diagonal_factor_->blocks[supernode];
      BlasMatrixView<Field> schur_complement = shared_state->schur_complements[supernode];

      if (num_children == 0) init();

      for (Int child_index = 0; child_index < num_children; ++child_index) {
          const Int child = ordering_.assembly_forest.children[child_beg + child_index]; // sorted_children[child_index];

          process_child(child, child_index);
          // Stop early if a child failed to finalize.
          if (fail) return false;

          MergeChildSchurComplement(supernode, child, ordering_,
                  lower_factor_.get(), shared_state->schur_complements[child],
                  lower_block, diagonal_block, schur_complement, *this, /* first_merge = */ child_index == 0);
      }
  }
  else {
      tbb::task_group tg;
      for (Int child_index = 0; child_index < num_children - 1; ++child_index) {
          const Int child = ordering_.assembly_forest.children[child_beg + child_index]; // sorted_children[child_index];
          tg.run([&process_child, child, child_index]() { process_child(child, child_index); });
      }
      // tg.run([&init]() { init(); });
      process_child(ordering_.assembly_forest.children[child_end - 1], num_children - 1);
      // init();
      // process_child(sorted_children[num_children - 1], num_children - 1);
      tg.wait();

      // Stop early if a child failed to finalize.
      if (fail) return false;

      MergeChildSchurComplements(supernode, *this, shared_state->schur_complements);
  }

  for (Int child_index = 0; child_index < num_children; ++child_index)
      MergeContribution(result_contributions[child_index], result);
  if (dynamic_reg_params.enabled) MergeDynamicRegularizations(result_contributions, result);

  return OpenMPRightLookingSupernodeFinalize(supernode, dynamic_reg_params, shared_state, private_states, result);
}

template <class Field>
SparseLDLResult<Field> Factorization<Field>::OpenMPRightLooking(
    const CoordinateMatrix<Field>& matrix) {
  BENCHMARK_SCOPED_TIMER_SECTION timer("OpenMPRightLooking");
  typedef ComplexBase<Field> Real;

  const Int num_supernodes = ordering_.supernode_sizes.Size();
  const Int num_roots = ordering_.assembly_forest.roots.Size();
  // {
  //   // Histogram of child count.
  //   std::vector<size_t> child_count_statistics;
  //   auto record = [&](size_t count) {
  //       if (count >= child_count_statistics.size()) child_count_statistics.resize(count + 1);
  //       ++child_count_statistics[count];
  //   };
  //   const auto &af = ordering_.assembly_forest;
  //   for (size_t supernode = 0; supernode < num_supernodes; ++supernode) {
  //       const Int num_children = af.child_offsets[supernode + 1] - af.child_offsets[supernode];
  //       record(num_children);

  //   }
  //   for (size_t i = 0; i < child_count_statistics.size(); ++i) {
  //       std::cout << "child count " << i << ": " << child_count_statistics[i] << std::endl;
  //   }
  // }

  // {
  //     // Histogram of the "run lengths" of contiguous indices.
  //     std::vector<size_t> supernode_run_length_statistics;
  //     auto record = [&](size_t run_len) {
  //         if (run_len >= supernode_run_length_statistics.size()) supernode_run_length_statistics.resize(run_len + 1);
  //         supernode_run_length_statistics[run_len]++;
  //     };

  //     for (size_t supernode = 0; supernode < num_supernodes; ++supernode) {
  //         const Int *indices = lower_factor_->StructureBeg(supernode);
  //         const size_t num_indices = lower_factor_->StructureEnd(supernode) - indices;
  //         if (num_indices == 0) continue;
  //         size_t run_len = 1;
  //         for (size_t i = 1; i < num_indices; ++i) {
  //             if (indices[i] == indices[i - 1] + 1) ++run_len;
  //             else { record(run_len); run_len = 1; }
  //         }
  //         record(run_len);
  //     }

  //     for (size_t i = 0; i < supernode_run_length_statistics.size(); ++i) {
  //         std::cout << "run_len " << i << ": " << supernode_run_length_statistics[i] << std::endl;
  //     }
  // }

  // Set up the base state of the dynamic regularization parameters. We only
  // need to update the offset for each child.
  static const Real kEpsilon = std::numeric_limits<Real>::epsilon();
  DynamicRegularizationParams<Field> dynamic_reg_params;
  dynamic_reg_params.enabled = control_.dynamic_regularization.enabled;
  dynamic_reg_params.positive_threshold = std::pow(kEpsilon, control_.dynamic_regularization.positive_threshold_exponent);
  dynamic_reg_params.negative_threshold = std::pow(kEpsilon, control_.dynamic_regularization.negative_threshold_exponent);
  if (control_.dynamic_regularization.relative) {
    const Real matrix_max_norm = MaxNorm(matrix);
    dynamic_reg_params.positive_threshold *= matrix_max_norm;
    dynamic_reg_params.negative_threshold *= matrix_max_norm;
  }
  dynamic_reg_params.signatures = &control_.dynamic_regularization.signatures;
  dynamic_reg_params.inverse_permutation = ordering_.inverse_permutation.Empty()
                                               ? nullptr
                                               : &ordering_.inverse_permutation;

  // const Int max_threads = omp_get_max_threads();
  const Int max_threads = get_max_num_tbb_threads();
  Buffer<PrivateState<Field>> private_states(max_threads);
  if (control_.factorization_type != kCholeskyFactorization) {
    const Int workspace_size = max_lower_block_size_;
    for (int t = 0; t < max_threads; ++t) {
      private_states[t].scaled_transpose_buffer.Resize(workspace_size);
    }
  }

  // Compute flop-count estimates so that we may prioritize the expensive
  // tasks before the cheaper ones.
  Buffer<double> &work_estimates = work_estimates_;
  double &total_work = total_work_;
  if (work_estimates.Size() != num_supernodes) {
      work_estimates.Resize(num_supernodes);
      for (const Int& root : ordering_.assembly_forest.roots) {
          FillSubtreeWorkEstimates(root, ordering_.assembly_forest, *lower_factor_,
                  &work_estimates);
      }

      total_work = std::accumulate(work_estimates.begin(), work_estimates.end(), 0.);
  }

  const double min_parallel_ratio_work = (total_work * control_.parallel_ratio_threshold) / max_threads;
  const double min_parallel_work = std::max(std::max(control_.min_parallel_threshold, min_parallel_ratio_work),
                                            max_threads < 2 ? std::numeric_limits<double>::infinity() : 0); // Forbid parallel execution

  // Allocate the map from child structures to parent fronts.
  auto &ncdi   = ordering_.assembly_forest.num_child_diag_indices;
  auto &cri    = ordering_.assembly_forest.child_rel_indices;
  if ( cri.Size() != num_supernodes) {
      cri.Resize(num_supernodes);
      ncdi.Resize(num_supernodes);
  }

  RightLookingSharedState<Field> &shared_state = shared_state_;
  if (shared_state.schur_complements.Size() != num_supernodes) {
      shared_state.schur_complements.Resize(num_supernodes);
#if ALLOCATE_SCHUR_COMPLEMENT_OTF
      shared_state.schur_complement_buffers.Resize(num_supernodes);
#else
      {
          BENCHMARK_SCOPED_TIMER_SECTION atimer("Allocate buffers");

          Int total_size = 0;
          for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
              const Int degree = lower_factor_->blocks[supernode].height;
              total_size += degree * degree;
          }

          // std::cout << "Allocating buffer of size " << total_size << "(" << (8.0 * total_size / 1024. / 1024.) << "MB)" << std::endl;
          shared_state.schur_complement_buffers.Resize(1);
          Buffer<Field> &workspace_buffer = shared_state.schur_complement_buffers[0];
          workspace_buffer.Resize(total_size);
          Int offset = 0;
          for (Int supernode = 0; supernode < num_supernodes; ++supernode) {
              const Int degree = lower_factor_->blocks[supernode].height;
              const Int workspace_size = degree * degree;

              auto &supernode_rhs = shared_state.schur_complements[supernode];
              supernode_rhs.height = degree;
              supernode_rhs.width = degree;
              supernode_rhs.leading_dim = degree;
              supernode_rhs.data = workspace_buffer.Data() + offset;
              offset += workspace_size;
          }
      }
#endif
  }

#ifdef CATAMARI_ENABLE_TIMERS
  shared_state.inclusive_timers.Resize(num_supernodes);
  shared_state.exclusive_timers.Resize(num_supernodes);
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  SparseLDLResult<Field> result;

  bool fail = false;

  Buffer<SparseLDLResult<Field>> result_contributions(num_roots);

#if ALLOCATE_SCHUR_COMPLEMENT_OTF
  // Allocate a single buffer to hold all roots' Schur complements.
  Buffer<Field> root_schur_complement_buffer;
  {
      Int total_size = 0;
      for (Int root_index = 0; root_index < num_roots; ++root_index) {
          const Int root = ordering_.assembly_forest.roots[root_index];
          const Int degree = lower_factor_->blocks[root].height;
          total_size += degree * degree;
      }
      root_schur_complement_buffer.Resize(total_size);

      Int offset = 0;
      for (Int root_index = 0; root_index < num_roots; ++root_index) {
          const Int root = ordering_.assembly_forest.roots[root_index];
          const Int degree = lower_factor_->blocks[root].height;
          BlasMatrixView<Field>& root_schur_complement = shared_state.schur_complements[root];
          root_schur_complement.height = degree;
          root_schur_complement.width = degree;
          root_schur_complement.leading_dim = degree;
          root_schur_complement.data = root_schur_complement_buffer.Data() + offset;
          offset += degree * degree;
      }
  }
#endif

  auto process_root = [&, min_parallel_work](Int root_index) {
      const Int root = ordering_.assembly_forest.roots[root_index];
      DynamicRegularizationParams<Field> subparams = dynamic_reg_params;
      subparams.offset = ordering_.supernode_offsets[root];
      bool success = OpenMPRightLookingSubtree(
              root, matrix, subparams, work_estimates, min_parallel_work,
              &shared_state, &private_states, &result_contributions[root_index]);
      if (!success) fail = true;
  };

  // const int old_max_threads = GetMaxBlasThreads();
  const bool parallel = (max_threads > 1) && (total_work >= min_parallel_work);
  // if (parallel) SetNumBlasThreads(2);

  // Recurse on each tree in the elimination forest.
  if (total_work < min_parallel_work || num_roots <= 1) {
      for (Int root_index = 0; root_index < num_roots; ++root_index) {
          process_root(root_index);
          if (fail) break;
      }
  }
  else {
      tbb::task_group tg;
      for (Int root_index = 0; root_index < num_roots - 1; ++root_index) {
          tg.run([&process_root, root_index]() { process_root(root_index); });
      }
      process_root(num_roots - 1);
      tg.wait();
  }

  // if (parallel) SetNumBlasThreads(old_max_threads);

  bool succeeded = !fail;
  if (succeeded) {
    for (Int index = 0; index < num_roots; ++index)
        MergeContribution(result_contributions[index], &result);
    if (dynamic_reg_params.enabled)
        MergeDynamicRegularizations(result_contributions, &result);
  }

#ifdef CATAMARI_ENABLE_TIMERS
  TruncatedForestTimersToDot(
      control_.inclusive_timings_filename, shared_state.inclusive_timers,
      ordering_.assembly_forest, control_.max_timing_levels,
      control_.avoid_timing_isolated_roots);
  TruncatedForestTimersToDot(
      control_.exclusive_timings_filename, shared_state.exclusive_timers,
      ordering_.assembly_forest, control_.max_timing_levels,
      control_.avoid_timing_isolated_roots);
#endif  // ifdef CATAMARI_ENABLE_TIMERS

  return result;
}

}  // namespace supernodal_ldl
}  // namespace catamari

#endif  // ifndef
// CATAMARI_SPARSE_LDL_SUPERNODAL_FACTORIZATION_RIGHT_LOOKING_OPENMP_IMPL_H_
