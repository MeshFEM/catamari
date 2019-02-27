/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <iostream>
#include <vector>

#include "catamari/dpp.hpp"
#include "catamari/unit_reach_nested_dissection.hpp"
#include "quotient/io_utils.hpp"
#include "specify.hpp"

using catamari::Int;

// A list of properties to measure from DPP sampling.
struct Experiment {
  // The number of seconds that elapsed during the object construction.
  double init_seconds = 0;

  // The number of seconds that elapsed during the sample.
  double sample_seconds = 0;
};

// Pretty prints the Experiment structure.
void PrintExperiment(const Experiment& experiment) {
  std::cout << "  init_seconds: " << experiment.init_seconds << "\n";
  std::cout << "  sample_seconds: " << experiment.sample_seconds << "\n";
  std::cout << std::endl;
}

// Returns the Frobenius norm of a real sparse matrix.
// NOTE: Due to the direct accumulation of the squared norm, this algorithm is
// unstable. But it suffices for example purposes.
template <typename Real>
Real EuclideanNorm(const catamari::CoordinateMatrix<Real>& matrix) {
  Real squared_norm{0};
  for (const catamari::MatrixEntry<Real>& entry : matrix.Entries()) {
    squared_norm += entry.value * entry.value;
  }
  return std::sqrt(squared_norm);
}

// Returns the Frobenius norm of a complex sparse matrix.
// NOTE: Due to the direct accumulation of the squared norm, this algorithm is
// unstable. But it suffices for example purposes.
template <typename Real>
Real EuclideanNorm(
    const catamari::CoordinateMatrix<catamari::Complex<Real>>& matrix) {
  Real squared_norm{0};
  for (const catamari::MatrixEntry<catamari::Complex<Real>>& entry :
       matrix.Entries()) {
    squared_norm += std::norm(entry.value);
  }
  return std::sqrt(squared_norm);
}

template <typename Field>
void Rescale(const catamari::ComplexBase<Field>& scale,
             catamari::CoordinateMatrix<Field>* matrix) {
  for (catamari::MatrixEntry<Field>& entry : matrix->Entries()) {
    entry.value *= scale;
  }
}

template <typename Field>
std::unique_ptr<catamari::CoordinateMatrix<Field>> Shifted2DNegativeLaplacian(
    Int x_size, Int y_size, const Field& diagonal_shift,
    const catamari::ComplexBase<Field>& scale, bool print_progress) {
  typedef catamari::ComplexBase<Field> Real;

  std::unique_ptr<catamari::CoordinateMatrix<Field>> matrix;
  matrix.reset(new catamari::CoordinateMatrix<Field>);

  const Int num_rows = x_size * y_size;
  matrix->Resize(num_rows, num_rows);
  matrix->ReserveEntryAdditions(5 * num_rows);
  for (Int x = 0; x < x_size; ++x) {
    for (Int y = 0; y < y_size; ++y) {
      const Int row = x + y * x_size;
      if (y > 0) {
        const Int down_row = x + (y - 1) * x_size;
        matrix->QueueEntryAddition(row, down_row, Field{-1});
      }
      if (x > 0) {
        const Int left_row = (x - 1) + y * x_size;
        matrix->QueueEntryAddition(row, left_row, Field{-1});
      }
      matrix->QueueEntryAddition(row, row, Field{5} + diagonal_shift);
      if (x < x_size - 1) {
        const Int right_row = (x + 1) + y * x_size;
        matrix->QueueEntryAddition(row, right_row, Field{-1});
      }
      if (y < y_size - 1) {
        const Int up_row = x + (y + 1) * x_size;
        matrix->QueueEntryAddition(row, up_row, Field{-1});
      }
    }
  }
  matrix->FlushEntryQueues();

  const Real pi = std::acos(Real{-1});
  CATAMARI_ASSERT(diagonal_shift >= Real{0}, "Shift was assumed non-negative.");
  const Real two_norm =
      4 * std::pow(std::sin((pi * x_size) / (2 * x_size)), Real{2}) +
      4 * std::pow(std::sin((pi * y_size) / (2 * y_size)), Real{2}) +
      diagonal_shift;
  Rescale(scale / two_norm, matrix.get());

  if (print_progress) {
    std::cout << "Matrix had " << matrix->NumRows() << " rows and "
              << matrix->NumEntries() << " entries." << std::endl;
  }
  return matrix;
}

// Prints the 2D sampling on-screen.
inline void AsciiDisplaySample(Int x_size, Int y_size,
                               const std::vector<Int>& sample,
                               char missing_char, char sampled_char) {
  const Int num_samples = sample.size();

  Int sample_ptr = 0;
  for (Int i = 0; i < y_size; ++i) {
    for (Int j = 0; j < x_size; ++j) {
      const Int index = j + i * x_size;
      while (sample_ptr < num_samples && sample[sample_ptr] < index) {
        ++sample_ptr;
      }
      if (sample_ptr < num_samples && sample[sample_ptr] == index) {
        std::cout << sampled_char;
      } else {
        std::cout << missing_char;
      }
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
  std::cout << "Sample density: " << (1. * sample.size()) / (x_size * y_size)
            << std::endl;
}

// Returns the Experiment statistics for a single Matrix Market input matrix.
Experiment RunShifted2DNegativeLaplacianTest(
    Int x_size, Int y_size, double diagonal_shift, double scale,
    bool maximum_likelihood, bool analytical_ordering, bool print_sample,
    bool ascii_display, char missing_char, char sampled_char, Int num_samples,
    const catamari::DPPControl& dpp_control, bool print_progress) {
  Experiment experiment;

  // Read the matrix from file.
  std::unique_ptr<catamari::CoordinateMatrix<double>> matrix =
      Shifted2DNegativeLaplacian(x_size, y_size, diagonal_shift, scale,
                                 print_progress);

  // Create the DPP object.
  if (print_progress) {
    std::cout << "  Initializing DPP..." << std::endl;
  }
  quotient::Timer init_timer;
  init_timer.Start();
  std::unique_ptr<catamari::DPP<double>> dpp;
  if (analytical_ordering) {
    catamari::SymmetricOrdering ordering;
    catamari::UnitReachNestedDissection2D(x_size - 1, y_size - 1, &ordering);
    dpp.reset(new catamari::DPP<double>(*matrix, ordering, dpp_control));
  } else {
    dpp.reset(new catamari::DPP<double>(*matrix, dpp_control));
  }
  experiment.init_seconds = init_timer.Stop();

  // Sample the matrix.
  if (print_progress) {
    std::cout << "  Sampling DPP..." << std::endl;
  }
  quotient::Timer sample_timer;
  std::vector<Int> sample;
  for (Int run = 0; run < num_samples; ++run) {
    sample_timer.Start();
    sample = dpp->Sample(maximum_likelihood);
    const double sample_seconds = sample_timer.Stop();
    std::cout << "  sample took " << sample_seconds << " seconds." << std::endl;
    if (print_sample) {
      quotient::PrintVector(sample, "sample", std::cout);
    }
    if (ascii_display) {
      AsciiDisplaySample(x_size, y_size, sample, missing_char, sampled_char);
    }
  }
  experiment.sample_seconds = sample_timer.TotalSeconds() / num_samples;

  return experiment;
}

int main(int argc, char** argv) {
  specify::ArgumentParser parser(argc, argv);
  const Int x_size = parser.OptionalInput<Int>(
      "x_size", "The x dimension of the 2D Laplacian", 80);
  const Int y_size = parser.OptionalInput<Int>(
      "y_size", "The y dimension of the 2D Laplacian", 80);
  const double diagonal_shift = parser.OptionalInput<double>(
      "diagonal_shift", "The value to add to the diagonal.", 0.);
  const double scale = parser.OptionalInput<double>(
      "scale", "The two-norm, in [0, 1], of the shifted negative Laplacian.",
      0.75);
  const bool maximum_likelihood = parser.OptionalInput<bool>(
      "maximum_likelihood", "Make the maximum-likelihood decisions?", true);
  const bool analytical_ordering = parser.OptionalInput<bool>(
      "analytical_ordering", "Use analytical nested dissection?", true);
  const bool print_sample = parser.OptionalInput<bool>(
      "print_sample", "Print each DPP sample?", false);
  const bool ascii_display = parser.OptionalInput<bool>(
      "ascii_display", "Display sample in ASCII?", true);
  const char missing_char = parser.OptionalInput<char>(
      "missing_char", "ASCII display for missing index.", ' ');
  const char sampled_char = parser.OptionalInput<char>(
      "sampled_char", "ASCII display for sampled index.", 'x');
  const bool relax_supernodes = parser.OptionalInput<bool>(
      "relax_supernodes", "Relax the supernodes?", true);
  const Int allowable_supernode_zeros =
      parser.OptionalInput<Int>("allowable_supernode_zeros",
                                "Number of zeros allowed in relaxations.", 128);
  const float allowable_supernode_zero_ratio = parser.OptionalInput<float>(
      "allowable_supernode_zero_ratio",
      "Ratio of explicit zeros allowed in a relaxed supernode.", 0.01f);
  const Int num_samples =
      parser.OptionalInput<Int>("num_samples", "The number of DPP samples.", 1);
  const bool print_progress = parser.OptionalInput<bool>(
      "print_progress", "Print the progress of the experiments?", false);
  if (!parser.OK()) {
    return 0;
  }

  catamari::DPPControl dpp_control;
  dpp_control.supernodal_control.relaxation_control.relax_supernodes =
      relax_supernodes;
  dpp_control.supernodal_control.relaxation_control.allowable_supernode_zeros =
      allowable_supernode_zeros;
  dpp_control.supernodal_control.relaxation_control
      .allowable_supernode_zero_ratio = allowable_supernode_zero_ratio;

  const Experiment experiment = RunShifted2DNegativeLaplacianTest(
      x_size, y_size, diagonal_shift, scale, maximum_likelihood,
      analytical_ordering, print_sample, ascii_display, missing_char,
      sampled_char, num_samples, dpp_control, print_progress);
  PrintExperiment(experiment);

  return 0;
}
