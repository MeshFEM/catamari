/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <iostream>
#include <vector>

#include "catamari/apply_sparse.hpp"
#include "catamari/blas_matrix.hpp"
#include "catamari/norms.hpp"
#include "catamari/sparse_ldl.hpp"
#include "specify.hpp"

using catamari::BlasMatrix;
using catamari::BlasMatrixView;
using catamari::Buffer;
using catamari::ConstBlasMatrixView;
using catamari::Int;

// A list of properties to measure from a sparse LDL factorization / solve.
struct Experiment {
  // The number of (structural) nonzeros in the associated Cholesky factor.
  Int num_nonzeros = 0;

  // The rough number of floating-point operations required to factor the
  // supernodal diagonal blocks.
  double num_diagonal_flops = 0;

  // The rough number of floating-point operations required to solve against the
  // diagonal blocks to update the subdiagonals.
  double num_subdiag_solve_flops = 0;

  // The rough number of floating-point operations required to form the Schur
  // complements.
  double num_schur_complement_flops = 0;

  // The number of floating-point operations required for a standard Cholesky
  // factorization using the returned ordering.
  double num_flops = 0;

  // The number of seconds that elapsed during the factorization.
  double factorization_seconds = 0;

  // The number of seconds that elapsed during the solve.
  double solve_seconds = 0;
};

// Pretty prints the Experiment structure.
void PrintExperiment(const Experiment& experiment, const std::string& title) {
  const double factorization_gflops_per_sec =
      experiment.num_flops / (1.e9 * experiment.factorization_seconds);

  std::cout << title << ": \n"
            << "  num_nonzeros:               " << experiment.num_nonzeros
            << "\n"
            << "  num_diagonal_flops:         " << experiment.num_diagonal_flops
            << "\n"
            << "  num_subdiag_solve_flops:    "
            << experiment.num_subdiag_solve_flops << "\n"
            << "  num_schur_complement_flops: "
            << experiment.num_schur_complement_flops << "\n"
            << "  num_flops:                  " << experiment.num_flops << "\n"
            << "  factorization_seconds:      "
            << experiment.factorization_seconds << "\n"
            << "  factorization gflops/sec:   " << factorization_gflops_per_sec
            << "\n"
            << "  solve_seconds:              " << experiment.solve_seconds
            << "\n"
            << "\n"
            << std::endl;
}

// Overwrites a matrix A with A + A' or A + A^T.
template <typename Field>
void MakeSymmetric(catamari::CoordinateMatrix<Field>* matrix, bool hermitian) {
  matrix->ReserveEntryAdditions(matrix->NumEntries());
  for (const catamari::MatrixEntry<Field>& entry : matrix->Entries()) {
    if (hermitian) {
      matrix->QueueEntryAddition(entry.column, entry.row,
                                 catamari::Conjugate(entry.value));
    } else {
      matrix->QueueEntryAddition(entry.column, entry.row, entry.value);
    }
  }
  matrix->FlushEntryQueues();
}

// Adds diagonal_shift to each entry of the diagonal.
template <typename Field>
void AddDiagonalShift(Field diagonal_shift,
                      catamari::CoordinateMatrix<Field>* matrix) {
  const Int num_rows = matrix->NumRows();
  matrix->ReserveEntryAdditions(num_rows);
  for (Int i = 0; i < num_rows; ++i) {
    matrix->QueueEntryAddition(i, i, diagonal_shift);
  }
  matrix->FlushEntryQueues();
}

// Returns the densest row index and the number of entries in the row.
template <typename Field>
std::pair<Int, Int> DensestRow(
    const catamari::CoordinateMatrix<Field>& matrix) {
  Int densest_row_size = 0;
  Int densest_row_index = -1;
  for (Int i = 0; i < matrix.NumRows(); ++i) {
    if (matrix.NumRowNonzeros(i) > densest_row_size) {
      densest_row_size = matrix.NumRowNonzeros(i);
      densest_row_index = i;
    }
  }
  return std::make_pair(densest_row_size, densest_row_index);
}

// With a sufficiently large choice of 'diagonal_shift', this routine returns
// a symmetric positive-definite sparse matrix corresponding to the Matrix
// Market example living in the given file.
template <typename Field>
std::unique_ptr<catamari::CoordinateMatrix<Field>> LoadMatrix(
    const std::string& filename, bool skip_explicit_zeros,
    quotient::EntryMask mask, bool force_symmetry, bool hermitian,
    const Field& diagonal_shift, bool print_progress) {
  if (print_progress) {
    std::cout << "Reading CoordinateGraph from " << filename << "..."
              << std::endl;
  }
  std::unique_ptr<catamari::CoordinateMatrix<Field>> matrix =
      catamari::CoordinateMatrix<Field>::FromMatrixMarket(
          filename, skip_explicit_zeros, mask);
  if (!matrix) {
    std::cerr << "Could not open " << filename << "." << std::endl;
    return matrix;
  }

  // Force symmetry since many of the examples are not.
  if (force_symmetry) {
    if (print_progress) {
      std::cout << "Enforcing matrix Hermiticity..." << std::endl;
    }
    MakeSymmetric(matrix.get(), hermitian);
  }

  // Adding the diagonal shift.
  AddDiagonalShift(diagonal_shift, matrix.get());

  if (print_progress) {
    std::cout << "Matrix had " << matrix->NumRows() << " rows and "
              << matrix->NumEntries() << " entries." << std::endl;

    std::pair<Int, Int> densest_row_info = DensestRow(*matrix);
    std::cout << "Densest row is index " << densest_row_info.first << " with "
              << densest_row_info.second << " connections." << std::endl;
  }
  return matrix;
}

template <typename Field>
void GenerateRightHandSide(Int num_rows, BlasMatrix<Field>* right_hand_side) {
  right_hand_side->Resize(num_rows, 1);
  for (Int row = 0; row < num_rows; ++row) {
    right_hand_side->Entry(row, 0) = row % 5;
  }
}

// Returns the Experiment statistics for a single Matrix Market input matrix.
Experiment RunMatrixMarketTest(const std::string& filename,
                               bool skip_explicit_zeros,
                               quotient::EntryMask mask, bool force_symmetry,
                               double diagonal_shift,
                               const catamari::SparseLDLControl& ldl_control,
                               bool print_progress) {
  typedef double Field;
  typedef catamari::ComplexBase<Field> BaseField;
  Experiment experiment;

  // Read the matrix from file.
  const bool hermitian = ldl_control.supernodal_control.factorization_type !=
                         catamari::kLDLTransposeFactorization;
  std::unique_ptr<catamari::CoordinateMatrix<Field>> matrix =
      LoadMatrix(filename, skip_explicit_zeros, mask, force_symmetry, hermitian,
                 diagonal_shift, print_progress);
  if (!matrix) {
    return experiment;
  }
  const Int num_rows = matrix->NumRows();

  // Factor the matrix.
  if (print_progress) {
    std::cout << "  Running factorization..." << std::endl;
  }
  quotient::Timer factorization_timer;
  factorization_timer.Start();
  catamari::SparseLDL<Field> ldl;
  const catamari::SparseLDLResult result = ldl.Factor(*matrix, ldl_control);
  experiment.factorization_seconds = factorization_timer.Stop();
  if (result.num_successful_pivots < num_rows) {
    std::cout << "  Failed factorization after " << result.num_successful_pivots
              << " pivots." << std::endl;
    return experiment;
  }
  experiment.num_nonzeros = result.num_factorization_entries;
  experiment.num_diagonal_flops = result.num_diagonal_flops;
  experiment.num_subdiag_solve_flops = result.num_subdiag_solve_flops;
  experiment.num_schur_complement_flops = result.num_schur_complement_flops;
  experiment.num_flops = result.num_factorization_flops;

  // Generate an arbitrary right-hand side.
  BlasMatrix<Field> right_hand_side;
  GenerateRightHandSide(num_rows, &right_hand_side);
  const BaseField right_hand_side_norm =
      catamari::EuclideanNorm(right_hand_side.ConstView());
  if (print_progress) {
    std::cout << "  || b ||_F = " << right_hand_side_norm << std::endl;
  }

  // Solve a random linear system.
  if (print_progress) {
    std::cout << "  Running solve..." << std::endl;
  }
  BlasMatrix<Field> solution = right_hand_side;
  quotient::Timer solve_timer;
  solve_timer.Start();
  ldl.Solve(&solution.view);
  experiment.solve_seconds = solve_timer.Stop();

  // Compute the residual.
  BlasMatrix<Field> residual = right_hand_side;
  catamari::ApplySparse(Field{-1}, *matrix, solution.ConstView(), Field{1},
                        &residual.view);
  const BaseField residual_norm = catamari::EuclideanNorm(residual.ConstView());
  std::cout << "  || B - A X ||_F / || B ||_F = "
            << residual_norm / right_hand_side_norm << std::endl;

  return experiment;
}

// Returns a map from the identifying string of each test matrix from a custom
// suite of SPD Matrix Market matrices.
std::unordered_map<std::string, Experiment> RunCustomTests(
    const std::string& matrix_market_directory, bool skip_explicit_zeros,
    quotient::EntryMask mask, double diagonal_shift,
    const catamari::SparseLDLControl& ldl_control, bool print_progress) {
  // TODO(Jack Poulson): Document why each of the commented out matrices was
  // unable to be factored. In most, if not all, cases, this seems to be due
  // to excessive memory usage. But std::bad_alloc errors are not being
  // properly propagated/caught.
  const std::vector<std::string> matrix_names{
      // "Queen_4147",
      // "audikw_1",
      // "Serena",
      // "Geo_1438",
      "Hook_1498", "bone010", "ldoor", "boneS10",
      // "Emilia_923",
      "PFlow_742", "inline_1", "nd24k",
      // "Fault_639",
      // "StocF-1465",
      "bundle_adj", "msdoor", "af_shell7", "af_shell8", "af_shell4",
      "af_shell3", "af_3_k101",
      // ...
      "ted_B", "ted_B_unscaled", "bodyy6", "bodyy5", "aft01", "bodyy4",
      "bcsstk15", "crystm01", "nasa4704", "LF10000",
      // ...
      "mesh3e1", "bcsstm09", "bcsstm08", "nos1", "bcsstm19", "bcsstk22",
      "bcsstk03", "nos4", "bcsstm20", "bcsstm06", "bcsstk01", "mesh1em6",
      "mesh1em1", "mesh1e1",
  };
  const bool force_symmetry = true;

  std::unordered_map<std::string, Experiment> experiments;
  for (const std::string& matrix_name : matrix_names) {
    const std::string filename =
        matrix_market_directory + "/" + matrix_name + ".mtx";
    try {
      experiments[matrix_name] = RunMatrixMarketTest(
          filename, skip_explicit_zeros, mask, force_symmetry, diagonal_shift,
          ldl_control, print_progress);
      PrintExperiment(experiments[matrix_name], matrix_name);
    } catch (std::exception& error) {
      std::cerr << "Caught exception: " << error.what() << std::endl;
    }
  }

  return experiments;
}

int main(int argc, char** argv) {
  typedef double Field;
  typedef catamari::ComplexBase<Field> BaseField;

  specify::ArgumentParser parser(argc, argv);
  const std::string filename = parser.OptionalInput<std::string>(
      "filename", "The location of a Matrix Market file.", "");
  const bool skip_explicit_zeros = parser.OptionalInput<bool>(
      "skip_explicit_zeros", "Skip explicitly zero entries?", true);
  const int entry_mask_int =
      parser.OptionalInput<int>("entry_mask_int",
                                "The quotient::EntryMask integer.\n"
                                "0:full, 1:lower-triangle, 2:upper-triangle",
                                0);
  const bool allow_supernodes = parser.OptionalInput<bool>(
      "allow_supernodes", "Allow Minimum Degree supernodes?", true);
  const int degree_type_int =
      parser.OptionalInput<int>("degree_type_int",
                                "The degree approximation type.\n"
                                "0:exact, 1:Amestoy, 2:Ashcraft, 3:Gilbert",
                                1);
  const bool aggressive_absorption = parser.OptionalInput<bool>(
      "aggressive_absorption", "Eliminate elements with aggressive absorption?",
      true);
  const Int min_dense_threshold = parser.OptionalInput<Int>(
      "min_dense_threshold",
      "Lower-bound on non-diagonal nonzeros for a row to be dense. The actual "
      "threshold will be: "
      "max(min_dense_threshold, dense_sqrt_multiple * sqrt(n))",
      16);
  const float dense_sqrt_multiple = parser.OptionalInput<float>(
      "dense_sqrt_multiple",
      "The multiplier on the square-root of the number of vertices for "
      "determining if a row is dense. The actual threshold will be: "
      "max(min_dense_threshold, dense_sqrt_multiple * sqrt(n))",
      10.f);
  const bool force_symmetry = parser.OptionalInput<bool>(
      "force_symmetry", "Use the nonzero pattern of A + A'?", true);
  const int factorization_type_int =
      parser.OptionalInput<int>("factorization_type_int",
                                "Type of the factorization.\n"
                                "0:Cholesky, 1:LDL', 2:LDL^T",
                                1);
  const int supernodal_strategy_int =
      parser.OptionalInput<int>("supernodal_strategy_int",
                                "The SupernodalStrategy int.\n"
                                "0:scalar, 1:supernodal, 2:adaptive",
                                2);
  const bool relax_supernodes = parser.OptionalInput<bool>(
      "relax_supernodes", "Relax the supernodes?", true);
  const Int allowable_supernode_zeros =
      parser.OptionalInput<Int>("allowable_supernode_zeros",
                                "Number of zeros allowed in relaxations.", 128);
  const float allowable_supernode_zero_ratio = parser.OptionalInput<float>(
      "allowable_supernode_zero_ratio",
      "Ratio of explicit zeros allowed in a relaxed supernode.", 0.01f);
  const double diagonal_shift = parser.OptionalInput<BaseField>(
      "diagonal_shift", "The value to add to the diagonal.", 1e6);
  const int ldl_algorithm_int =
      parser.OptionalInput<int>("ldl_algorithm_int",
                                "The LDL algorithm type.\n"
                                "0:left-looking, 1:up-looking, 2:right-looking",
                                2);
  const bool print_progress = parser.OptionalInput<bool>(
      "print_progress", "Print the progress of the experiments?", false);
  const std::string matrix_market_directory = parser.OptionalInput<std::string>(
      "matrix_market_directory",
      "The directory where the custom matrix market .tar.gz's were unpacked",
      "");
  if (!parser.OK()) {
    return 0;
  }
  if (filename.empty() && matrix_market_directory.empty()) {
    std::cerr << "One of 'filename' or 'matrix_market_directory' must be "
                 "specified.\n"
              << std::endl;
    parser.PrintReport();
    return 0;
  }

  const quotient::EntryMask mask =
      static_cast<quotient::EntryMask>(entry_mask_int);

  catamari::SparseLDLControl ldl_control;
  ldl_control.SetFactorizationType(
      static_cast<catamari::SymmetricFactorizationType>(
          factorization_type_int));
  ldl_control.supernodal_strategy =
      static_cast<catamari::SupernodalStrategy>(supernodal_strategy_int);

  // Set the minimum degree control options.
  {
    auto& md_control = ldl_control.md_control;
    md_control.allow_supernodes = allow_supernodes;
    md_control.degree_type = static_cast<quotient::DegreeType>(degree_type_int);
    md_control.aggressive_absorption = aggressive_absorption;
    md_control.min_dense_threshold = min_dense_threshold;
    md_control.dense_sqrt_multiple = dense_sqrt_multiple;
  }

  // Set the scalar control options.
  ldl_control.scalar_control.algorithm =
      static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);

  // Set the supernodal control options.
  {
    auto& sn_control = ldl_control.supernodal_control;
    sn_control.algorithm =
        static_cast<catamari::LDLAlgorithm>(ldl_algorithm_int);
    sn_control.relaxation_control.relax_supernodes = relax_supernodes;
    sn_control.relaxation_control.allowable_supernode_zeros =
        allowable_supernode_zeros;
    sn_control.relaxation_control.allowable_supernode_zero_ratio =
        allowable_supernode_zero_ratio;
  }

  if (!matrix_market_directory.empty()) {
    const std::unordered_map<std::string, Experiment> experiments =
        RunCustomTests(matrix_market_directory, skip_explicit_zeros, mask,
                       diagonal_shift, ldl_control, print_progress);
    for (const std::pair<std::string, Experiment>& pairing : experiments) {
      PrintExperiment(pairing.second, pairing.first);
    }
  } else {
    const Experiment experiment =
        RunMatrixMarketTest(filename, skip_explicit_zeros, mask, force_symmetry,
                            diagonal_shift, ldl_control, print_progress);
    PrintExperiment(experiment, filename);
  }

  return 0;
}
