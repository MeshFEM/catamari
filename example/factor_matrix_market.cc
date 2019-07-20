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

// With a sufficiently large choice of 'relative_shift', this routine returns
// a symmetric positive-definite sparse matrix corresponding to the Matrix
// Market example living in the given file.
template <typename Field>
std::unique_ptr<catamari::CoordinateMatrix<Field>> LoadMatrix(
    const std::string& filename, bool skip_explicit_zeros,
    quotient::EntryMask mask, bool force_symmetry, bool hermitian,
    const Field& relative_shift, double drop_tolerance, bool print_progress) {
  typedef catamari::ComplexBase<Field> Real;

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

  std::cout << "Before enforcing symmetry: " << matrix->NumEntries()
            << " entries." << std::endl;

  // Force symmetry since many of the examples are not.
  if (force_symmetry) {
    if (print_progress) {
      std::cout << "Enforcing matrix Hermiticity..." << std::endl;
    }
    MakeSymmetric(matrix.get(), hermitian);

    std::cout << "After enforcing symmetry: " << matrix->NumEntries()
              << " entries." << std::endl;
  }

  // Rescale the maximum norm to one.
  {
    const Real max_norm = catamari::MaxNorm(*matrix);
    Int num_to_drop = 0;
    for (auto& entry : matrix->Entries()) {
      entry.value /= max_norm;
      if (std::abs(entry.value) <= drop_tolerance) {
        std::cout << "  Will drop entry (" << entry.row << ", " << entry.column
                  << ") with abs " << std::abs(entry.value) << std::endl;
        ++num_to_drop;
      }
    }
    if (num_to_drop > 0) {
      for (const auto& entry : matrix->Entries()) {
        if (std::abs(entry.value) <= drop_tolerance) {
          matrix->QueueEntryRemoval(entry.row, entry.column);
        }
      }
      matrix->FlushEntryQueues();
    }
  }

  // Adding the diagonal shift.
  const Real frob_norm = catamari::EuclideanNorm(*matrix);
  AddDiagonalShift(relative_shift * frob_norm, matrix.get());

  if (print_progress) {
    Real min_abs = std::numeric_limits<Real>::max();
    Real max_abs = 0;
    for (auto& entry : matrix->Entries()) {
      min_abs = std::min(min_abs, std::abs(entry.value));
      max_abs = std::max(max_abs, std::abs(entry.value));
    }

    std::cout << "|A| is " << matrix->NumRows() << " x " << matrix->NumColumns()
              << "with " << matrix->NumEntries() << " entries, with values in ["
              << min_abs << ", " << max_abs << "]" << std::endl;

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
                               double relative_shift, double drop_tolerance,
                               const catamari::SparseLDLControl& ldl_control,
                               bool print_progress) {
  typedef double Field;
  typedef catamari::ComplexBase<Field> Real;
  Experiment experiment;

  // Read the matrix from file.
  const bool hermitian = ldl_control.supernodal_control.factorization_type !=
                         catamari::kLDLTransposeFactorization;
  std::unique_ptr<catamari::CoordinateMatrix<Field>> matrix =
      LoadMatrix(filename, skip_explicit_zeros, mask, force_symmetry, hermitian,
                 relative_shift, drop_tolerance, print_progress);
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
  const Real right_hand_side_norm =
      catamari::EuclideanNorm(right_hand_side.ConstView());
  if (print_progress) {
    std::cout << "  || b ||_F = " << right_hand_side_norm << std::endl;
  }

  // Solve a random linear system.
  if (print_progress) {
    std::cout << "  Running solve..." << std::endl;
  }
  // TODO(Jack Poulson): Make these parameters configurable.
  catamari::RefinedSolveControl<Real> refined_solve_control;
  refined_solve_control.verbose = true;
  BlasMatrix<Field> solution = right_hand_side;
  quotient::Timer solve_timer;
  solve_timer.Start();
  catamari::RefinedSolveStatus<Real> refined_solve_state =
      ldl.RefinedSolve(*matrix, refined_solve_control, &solution.view);
  experiment.solve_seconds = solve_timer.Stop();

  // Compute the residual.
  BlasMatrix<Field> residual = right_hand_side;
  catamari::ApplySparse(Field{-1}, *matrix, solution.ConstView(), Field{1},
                        &residual.view);
  const Real residual_norm = catamari::EuclideanNorm(residual.ConstView());
  std::cout << "  || B - A X ||_F / || B ||_F = "
            << residual_norm / right_hand_side_norm << std::endl;
  std::cout << "  refined_solve_state.num_iterations: "
            << refined_solve_state.num_iterations << "\n"
            << "  refined_solve_state.residual_relative_max_norm: "
            << refined_solve_state.residual_relative_max_norm << std::endl;

  return experiment;
}

// Returns a map from the identifying string of each test matrix from a custom
// suite of SPD Matrix Market matrices.
std::unordered_map<std::string, Experiment> RunCustomTests(
    const std::string& matrix_market_directory, bool skip_explicit_zeros,
    quotient::EntryMask mask, double relative_shift, double drop_tolerance,
    const catamari::SparseLDLControl& ldl_control, bool print_progress) {
  // The list of matrices tested in:
  //
  //   J.D. Hogg, J.K. Reid, and J.A. Scott,
  //   "Design of a multicore sparse Cholesky factorization using DAGs",
  //   SIAM J. Sci. Comput., Vol. 32, No. 6, pp. 3627--3649.
  //
  const std::vector<std::string> matrix_names{
      "tmt_sym",
      "thermal2",
      "gearbox",
      "m_t1",
      "pwtk",
      "pkustk13",
      "crankseg_1",
      "cfd2",
      "thread",
      "shipsec8",
      "shipsec1",
      "crankseg_2",
      "fcondp2",
      "af_shell3",
      "troll",
      "G3_circuit",
      "bmwcra_1",
      "halfb",
      "2cubes_sphere",
      "ldoor",
      "ship_003",
      "fullb",
      "inline_1",
      "pkustk14",
      "apache2",
      "F1",
      "boneS10",
      "nd12k",
      "Trefethen_20000",
      "nd24k",
      "bone010",
      "audikw_1",
  };
  const bool force_symmetry = true;

  std::unordered_map<std::string, Experiment> experiments;
  for (const std::string& matrix_name : matrix_names) {
    const std::string filename =
        matrix_market_directory + "/" + matrix_name + ".mtx";
    std::cout << "Testing " << filename << std::endl;
    try {
      experiments[matrix_name] = RunMatrixMarketTest(
          filename, skip_explicit_zeros, mask, force_symmetry, relative_shift,
          drop_tolerance, ldl_control, print_progress);
      PrintExperiment(experiments[matrix_name], matrix_name);
    } catch (std::exception& error) {
      std::cerr << "Caught exception: " << error.what() << std::endl;
    }
  }

  return experiments;
}

int main(int argc, char** argv) {
  typedef double Field;
  typedef catamari::ComplexBase<Field> Real;

  specify::ArgumentParser parser(argc, argv);
  const std::string filename = parser.OptionalInput<std::string>(
      "filename", "The location of a Matrix Market file.", "");
  const bool skip_explicit_zeros = parser.OptionalInput<bool>(
      "skip_explicit_zeros", "Skip explicitly zero entries?", false);
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
  const bool mass_elimination = parser.OptionalInput<bool>(
      "mass_elimination", "Perform element 'mass elimination' absorption?",
      true);
  const bool push_pivot_into_front = parser.OptionalInput<bool>(
      "push_pivot_into_front",
      "When constructing element lists in AMD, put pivot in front?", true);
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
                                0);
  const int supernodal_strategy_int =
      parser.OptionalInput<int>("supernodal_strategy_int",
                                "The SupernodalStrategy int.\n"
                                "0:scalar, 1:supernodal, 2:adaptive",
                                2);
  const bool relax_supernodes = parser.OptionalInput<bool>(
      "relax_supernodes", "Relax the supernodes?", true);
  const double relative_shift = parser.OptionalInput<Real>(
      "relative_shift", "We add || A ||_F * relative_shift to the diagonal",
      2.1);
  const double drop_tolerance = parser.OptionalInput<Real>(
      "drop_tolerance", "We drop entries with absolute value <= this.", -1.);
  const int ldl_algorithm_int =
      parser.OptionalInput<int>("ldl_algorithm_int",
                                "The LDL algorithm type.\n"
                                "0:left-looking, 1:up-looking, 2:right-looking,"
                                "3:adaptive",
                                3);
  const Int block_size = parser.OptionalInput<Int>(
      "block_size", "The dense algorithmic block size.", 64);
#ifdef CATAMARI_OPENMP
  const Int factor_tile_size = parser.OptionalInput<Int>(
      "factor_tile_size", "The multithreaded factorization tile size.", 128);
  const Int outer_product_tile_size = parser.OptionalInput<Int>(
      "outer_product_tile_size", "The multithreaded outer-product tile size.",
      240);
  const Int merge_grain_size = parser.OptionalInput<Int>(
      "merge_grain_size", "The number of columns to merge at once.", 500);
  const Int sort_grain_size = parser.OptionalInput<Int>(
      "sort_grain_size", "The number of columns to sort at once.", 200);
#endif  // ifdef CATAMARI_OPENMP
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
    md_control.mass_elimination = mass_elimination;
    md_control.push_pivot_into_front = push_pivot_into_front;
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
    sn_control.block_size = block_size;
#ifdef CATAMARI_OPENMP
    sn_control.factor_tile_size = factor_tile_size;
    sn_control.outer_product_tile_size = outer_product_tile_size;
    sn_control.merge_grain_size = merge_grain_size;
    sn_control.sort_grain_size = sort_grain_size;
#endif  // ifdef CATAMARI_OPENMP
    sn_control.relaxation_control.relax_supernodes = relax_supernodes;
  }

  if (!matrix_market_directory.empty()) {
    const std::unordered_map<std::string, Experiment> experiments =
        RunCustomTests(matrix_market_directory, skip_explicit_zeros, mask,
                       relative_shift, drop_tolerance, ldl_control,
                       print_progress);
    for (const std::pair<std::string, Experiment>& pairing : experiments) {
      PrintExperiment(pairing.second, pairing.first);
    }
  } else {
    const Experiment experiment = RunMatrixMarketTest(
        filename, skip_explicit_zeros, mask, force_symmetry, relative_shift,
        drop_tolerance, ldl_control, print_progress);
    PrintExperiment(experiment, filename);
  }

  return 0;
}
