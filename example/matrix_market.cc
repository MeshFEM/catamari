/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>
#include "catamari.hpp"
#include "specify.hpp"

using catamari::Int;

// A list of properties to measure from a sparse LDL factorization / solve.
struct Experiment {
  // The number of (structural) nonzeros in the associated Cholesky factor.
  Int num_nonzeros;

  // The number of floating-point operations required for a standard Cholesky
  // factorization using the returned ordering.
  double num_flops;

  // The number of seconds that elapsed during the analysis.
  double analysis_seconds;

  // The number of seconds that elapsed during the factorization.
  double factorization_seconds;

  // The number of seconds that elapsed during the solve.
  double solve_seconds;
};

// Pretty prints the Experiment structure.
void PrintExperiment(
    const Experiment& experiment, const std::string& label) {
  std::cout << label << ":\n";
  std::cout << "  num_nonzeros:          " << experiment.num_nonzeros << "\n";
  std::cout << "  num_flops:             " << experiment.num_flops << "\n";
  std::cout << "  analysis_seconds:      "
            << experiment.analysis_seconds << "\n";
  std::cout << "  factorization_seconds: "
            << experiment.factorization_seconds << "\n";
  std::cout << "  solve_seconds:         " << experiment.solve_seconds << "\n";
  std::cout << std::endl;
}

template<typename Real>
Real EuclideanNorm(const std::vector<Real>& vector) {
  Real squared_norm{0};
  const Int num_rows = vector.size();
  for (Int i = 0; i < num_rows; ++i) {
    squared_norm += vector[i] * vector[i];
  }
  return std::sqrt(squared_norm);
}

template<typename Real>
Real EuclideanNorm(const std::vector<catamari::Complex<Real>>& vector) {
  Real squared_norm{0};
  const Int num_rows = vector.size();
  for (Int i = 0; i < num_rows; ++i) {
    squared_norm += std::norm(vector[i]);
  }
  return std::sqrt(squared_norm);
}

// Returns the Experiment statistics for a single Matrix Market input matrix.
Experiment RunMatrixMarketTest(
    const std::string& filename,
    bool skip_explicit_zeros,
    quotient::EntryMask mask,
    const quotient::MinimumDegreeControl& control,
    bool force_symmetry,
    double diagonal_shift,
    bool print_progress,
    bool write_permuted_matrix) {
  typedef double Field;
  typedef catamari::ComplexBase<Field> BaseField;

  if (print_progress) {
    std::cout << "Reading CoordinateGraph from " << filename << "..."
              << std::endl;
  }
  std::unique_ptr<catamari::CoordinateMatrix<Field>> matrix =
      catamari::CoordinateMatrix<Field>::FromMatrixMarket(
          filename, skip_explicit_zeros, mask);
  Experiment experiment;
  if (!matrix) {
    std::cerr << "Could not open " << filename << "." << std::endl;
    return experiment;
  }

  // Force symmetry since many of the examples are not. We form the nonzero
  // pattern of A + A'.
  if (force_symmetry) {
    if (print_progress) {
      std::cout << "Enforcing matrix Hermiticity..." << std::endl;
    }
    matrix->ReserveEntryAdditions(matrix->NumEntries());
    for (const catamari::MatrixEntry<Field>& entry :
        matrix->Entries()) {
      matrix->QueueEntryAddition(
          entry.column, entry.row, catamari::Conjugate(entry.value));
    }
    matrix->FlushEntryQueues();
  }

  // Adding the diagonal shift.
  const Int num_rows = matrix->NumRows();
  matrix->ReserveEntryAdditions(num_rows);
  for (Int i = 0; i < num_rows; ++i) {
    matrix->QueueEntryAddition(i, i, diagonal_shift);
  }
  matrix->FlushEntryQueues();

  if (print_progress) {
    std::cout << "Matrix had " << matrix->NumRows() << " rows and "
              << matrix->NumEntries() << " entries." << std::endl;

    Int densest_row_size = 0;
    Int densest_row_index = -1;
    for (Int i = 0; i < matrix->NumRows(); ++i ) {
      if (matrix->NumRowNonzeros(i) > densest_row_size) {
        densest_row_size = matrix->NumRowNonzeros(i);
        densest_row_index = i;
      }
    }
    std::cout << "Densest row is index " << densest_row_index << " with "
              << densest_row_size << " connections." << std::endl;
  }

  // Produce a graph from the loaded matrix.
  std::unique_ptr<quotient::CoordinateGraph> graph = matrix->CoordinateGraph();

  if (print_progress) {
    std::cout << "  Running analysis..." << std::endl;
  }
  quotient::Timer timer;
  timer.Start();
  const quotient::MinimumDegreeResult analysis =
      quotient::MinimumDegree(*graph, control);
  experiment.analysis_seconds = timer.Stop();
  experiment.num_nonzeros = analysis.num_cholesky_nonzeros;
  experiment.num_flops = analysis.num_cholesky_flops;
  if (print_progress) {
    std::cout << "  Finished analysis in " << experiment.analysis_seconds
              << " seconds. There were " << experiment.num_nonzeros
              << " nonzeros." << std::endl;
  }
#ifdef QUOTIENT_ENABLE_TIMERS
  for (const std::pair<std::string, double>& pairing :
      analysis.elapsed_seconds) {
    std::cout << "    " << pairing.first << ": " << pairing.second
              << " seconds." << std::endl;
  }
#endif

  // Form the permuted matrix.
  const std::vector<Int> permutation = analysis.Permutation();
  catamari::CoordinateMatrix<Field> permuted_matrix;
  permuted_matrix.Resize(matrix->NumRows(), matrix->NumColumns());
  permuted_matrix.ReserveEntryAdditions(matrix->NumEntries());
  for (const catamari::MatrixEntry<Field>& entry : matrix->Entries()) {
    permuted_matrix.QueueEntryAddition(
        permutation[entry.row], permutation[entry.column], entry.value);
  }
  permuted_matrix.FlushEntryQueues();
  if (write_permuted_matrix) {
    const std::string new_filename = filename + "-perm.mtx";
    permuted_matrix.ToMatrixMarket(new_filename);
  }

  // Factor the permuted matrix.
  if (print_progress) {
    std::cout << "  Running factorization..." << std::endl;
  }
  quotient::Timer factorization_timer;
  factorization_timer.Start();
  catamari::ScalarLDLAnalysis ldl_analysis;
  catamari::ScalarLowerFactor<Field> ldl_lower_factor;
  catamari::ScalarDiagonalFactor<Field> ldl_diagonal_factor;
  catamari::ScalarLDLSetup(
      permuted_matrix, &ldl_analysis, &ldl_lower_factor, &ldl_diagonal_factor);
  if (print_progress) {
    std::cout << "    Finished setup..." << std::endl;
  }
  const Int num_pivots = catamari::ScalarLDLFactorization(
      permuted_matrix, ldl_analysis, &ldl_lower_factor, &ldl_diagonal_factor);
  experiment.factorization_seconds = factorization_timer.Stop();
  if (num_pivots < num_rows) {
    std::cout << "  Failed factorization after " << num_pivots << " pivots."
              << std::endl;
    return experiment;
  }

  // Solve a random linear system.
  if (print_progress) {
    std::cout << "  Running solve..." << std::endl;
  }

  std::vector<Field> right_hand_side(num_rows, 0);
  for (Int row = 0; row < num_rows; ++row) {
    right_hand_side[row] = row % 5;
  }
  const BaseField right_hand_side_norm = EuclideanNorm(right_hand_side);
  if (print_progress) {
    std::cout << "  || b ||_F = " << right_hand_side_norm << std::endl;
  }
  auto solution = right_hand_side;
  quotient::Timer solve_timer;
  solve_timer.Start();
  catamari::LDLSolve(ldl_lower_factor, ldl_diagonal_factor, &solution);
  experiment.solve_seconds = solve_timer.Stop();

  // Compute the residual.
  auto residual = right_hand_side;
  catamari::MatrixVectorProduct(
      Field{-1}, permuted_matrix, solution, Field{1}, &residual);
  const BaseField residual_norm = EuclideanNorm(residual);
  std::cout << "  || b - A x ||_F / || b ||_F = "
            << residual_norm / right_hand_side_norm << std::endl;

  return experiment;
}

// Returns a map from the identifying string of each test matrix from the
// Amestoy/Davis/Duff Approximate Minimum Degree reordering 1996 paper meant
// to loosely reproduce Fig. 2.
//
// It is worth noting that the LHR34 results from the paper appear to be
// incorrect, as the results shown in
//
//   https://www.cise.ufl.edu/research/sparse/matrices/Mallya/lhr34.html
//
// agree with the results observed from this code's implementation.
//
std::unordered_map<std::string, Experiment> RunADD96Tests(
    const std::string& matrix_market_directory,
    bool skip_explicit_zeros,
    quotient::EntryMask mask,
    const quotient::MinimumDegreeControl& control,
    double diagonal_shift,
    bool print_progress,
    bool write_permuted_matrix) {
  const std::vector<std::string> kMatrixNames{
      "appu",
      "bbmat",
      "bcsstk30",
      "bcsstk31",
      "bcsstk32",
      "bcsstk33",
      "crystk02",
      "crystk03",
      "ct20stif",
      "ex11",
      "ex19",
      "ex40",
      "finan512",
      "lhr34",
      "lhr71",
      "nasasrb",
      "olafu",
      "orani678",
      "psmigr_1",
      "raefsky1",
      "raefsky3",
      "raefsky4",
      "rim",
      "venkat01",
      "wang3",
      "wang4",
  };
  const bool force_symmetry = true;

  std::unordered_map<std::string, Experiment> experiments;
  for (const std::string& matrix_name : kMatrixNames) {
    const std::string filename = matrix_market_directory + "/" + matrix_name +
        "/" + matrix_name + ".mtx";
    experiments[matrix_name] = RunMatrixMarketTest(
        filename,
        skip_explicit_zeros,
        mask,
        control,
        force_symmetry,
        diagonal_shift,
        print_progress,
        write_permuted_matrix);
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
  const int entry_mask_int = parser.OptionalInput<int>(
      "entry_mask_int",
      "The quotient::EntryMask integer.\n"
      "0:full, 1:lower-triangle, 2:upper-triangle",
      0);
  const int degree_type_int = parser.OptionalInput<int>(
      "degree_type_int",
      "The degree approximation type.\n"
      "0:exact, 1:Amestoy, 2:Ashcraft, 3:Gilbert",
      1);
  const bool allow_supernodes = parser.OptionalInput<bool>(
      "allow_supernodes",
      "Allow variables to be merged into supernodes?",
      true);
  const bool aggressive_absorption = parser.OptionalInput<bool>(
      "aggressive_absorption",
      "Eliminate elements with aggressive absorption?",
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
      "force_symmetry",
      "Use the nonzero pattern of A + A'?",
      true);
  const double diagonal_shift = parser.OptionalInput<BaseField>(
      "diagonal_shift",
      "The value to add to the diagonal.",
      1e6);
  const bool print_progress = parser.OptionalInput<bool>(
      "print_progress",
      "Print the progress of the experiments?",
      false);
  const bool write_permuted_matrix = parser.OptionalInput<bool>(
      "write_permuted_matrix",
      "Write the permuted matrix to file?",
      false);
  const std::string matrix_market_directory = parser.OptionalInput<std::string>(
      "matrix_market_directory",
      "The directory where the ADD96 matrix market .tar.gz's were unpacked",
      "");
  const bool randomly_seed = parser.OptionalInput<bool>(
      "randomly_seed",
      "Randomly seed the pseudo-random number generator?",
      false);
#ifdef _OPENMP
  const int num_omp_threads = parser.OptionalInput<int>(
      "num_omp_threads",
      "The desired number of OpenMP threads. Uses default if <= 0.",
      1);
#endif
  if (!parser.OK()) {
    return 0;
  }
  if (filename.empty() && matrix_market_directory.empty()) {
    std::cerr << "One of 'filename' or 'matrix_market_directory' must be "
                 "specified.\n" << std::endl;
    parser.PrintReport();
    return 0;
  }

  const quotient::EntryMask mask = static_cast<quotient::EntryMask>(
      entry_mask_int);

#ifdef _OPENMP
  if (num_omp_threads > 0) {
    const int max_omp_threads = omp_get_max_threads();
    omp_set_num_threads(num_omp_threads);
    std::cout << "Will use " << num_omp_threads << " of " << max_omp_threads
              << " OpenMP threads." << std::endl;
  } else {
    std::cout << "Will use all " << omp_get_max_threads() << " OpenMP threads."
              << std::endl;
  }
#endif

  quotient::MinimumDegreeControl control;
  control.degree_type = static_cast<quotient::DegreeType>(degree_type_int);
  control.allow_supernodes = allow_supernodes;
  control.aggressive_absorption = aggressive_absorption;
  control.min_dense_threshold = min_dense_threshold;
  control.dense_sqrt_multiple = dense_sqrt_multiple;

  if (randomly_seed) {
    // Seed the random number generator based upon the current time.
    const unsigned srand_seed = std::time(0);
    std::cout << "Seeding std::srand with " << srand_seed << std::endl;
    std::srand(srand_seed);
  }

  if (!matrix_market_directory.empty()) {
    const std::unordered_map<std::string, Experiment> experiments =
        RunADD96Tests(
            matrix_market_directory,
            skip_explicit_zeros,
            mask,
            control,
            diagonal_shift,
            print_progress,
            write_permuted_matrix);
    for (const std::pair<std::string, Experiment>& pairing : experiments) {
      PrintExperiment(pairing.second, pairing.first);
    }
  } else {
    const Experiment experiment = RunMatrixMarketTest(
        filename,
        skip_explicit_zeros,
        mask,
        control,
        force_symmetry,
        diagonal_shift,
        print_progress,
        write_permuted_matrix);
    PrintExperiment(experiment, filename);
  }

  return 0;
}
