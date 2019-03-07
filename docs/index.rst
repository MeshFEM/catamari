.. catamari documentation master file, created by
   sphinx-quickstart on Mon Mar  4 10:29:06 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Catamari's development documentation
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

`catamari <https://hodgestar.com/catamari/>`_ is a `C++14 <https://en.wikipedia.org/wiki/C%2B%2B14>`_, header-only implementations of sequential and
`DAG-scheduled <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`_, real
and complex, supernodal sparse-direct `Cholesky <https://en.wikipedia.org/wiki/Cholesky_decomposition>`_, LDL^T, and LDL^H factorizations. It similarly
contains sequential and DAG-scheduled, dense and sparse-direct, real and
complex, `Determinantal Point Process <https://en.wikipedia.org/wiki/Determinantal_point_process>`_ sampling through modified LDL^H factorizations.

Quickstart
----------

Sparse-direct solver
^^^^^^^^^^^^^^^^^^^^
Usage of the sparse-direct solver through the
:samp:`catamari::CoordinateMatrix` template class is fairly straight-forward:

.. code-block:: cpp

  #include "catamari.hpp"
  // Build a real or complex symmetric input matrix.
  //
  // Alternatively, one could use
  // catamari::CoordinateMatrix<Field>::FromMatrixMarket to read the matrix from
  // a Matrix Market file (e.g., from the Davis sparse matrix collection). But
  // keep in mind that one often needs to enforce explicit symmetry.
  catamari::CoordinateMatrix<double> matrix;
  matrix.Resize(num_rows, num_rows);
  matrix.ReserveEntryAdditions(num_entries_upper_bound);
  // Queue updates of entries in the sparse matrix using commands of the form:
  //   matrix.QueueEdgeAddition(row, column, value);
  matrix.FlushEntryQueues();

  // Fill the options for the factorization.
  catamari::LDLControl ldl_control;
  // The options for the factorization type are:
  //   * catamari::kCholeskyFactorization,
  //   * catamari::kLDLAdjointFactorization,
  //   * catamari::kLDLTransposeFactorization.
  ldl_control.SetFactorizationType(catamari::kCholeskyFactorization);

  // Factor the matrix.
  catamari::LDLFactorization<double> factorization;
  const catamari::LDLResult result = factorization.Factor(matrix, ldl_control);

  // Solve a linear system using the factorization.
  catamari::BlasMatrix<double> right_hand_sides;
  right_hand_sides.Resize(num_rows, num_rhs);
  // The (i, j) entry of the right-hand side can easily be read or modified, e.g.:
  //   right_hand_sides(i, j) = 1.;
  factorization.Solve(&right_hand_sides.view);

  // Alternatively, one can solve using iterative-refinement, e.g., using:
  catamari::RefinedSolveControl<double> refined_solve_control;
  refined_solve_control.relative_tol = 1e-15;
  refined_solve_control.max_iters = 3;
  refined_solve_control.verbose = true;
  factorization.RefinedSolve(
      matrix, refined_solve_control, &right_hand_sides.view);

One can also browse the `example/ <https://gitlab.com/hodge_star/catamari/tree/master/example>`_ folder for complete examples (e.g., for `solving 3D Helmholtz equations <https://gitlab.com/hodge_star/catamari/blob/master/example/helmholtz_3d_pml.cc>`_ with PML boundary conditions discretized using trilinear hexahedral elements using a complex LDL^T factorization).

Determinantal Point Process sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dense DPP sampling
""""""""""""""""""
A dense DPP can be sampled from its kernel matrix (in a sequential manner,
perhaps using multithreaded BLAS calls) using the routine
:samp:`catamari::LowerFactorAndSampleDPP`:

.. code-block:: cpp

  #include "catamari.hpp"
  catamari::BlasMatrix<catamari::Complex<double>> matrix;
  matrix.Resize(num_rows, num_rows);
  // Fill the matrix with calls of the form: matrix(i, j) = value;

  std::random_device random_device;
  std::mt19937 generator(random_device());
  const catamari::Int block_size = 64;
  const bool maximum_likelihood = false;
  const int num_samples = 10;
  std::vector<std::vector<catamari::Int>> samples(num_samples);
  for (int sample_index = 0; sample_index < num_samples; ++sample_index) {
    auto matrix_copy = matrix;
    samples[sample_index] = catamari::LowerFactorAndSampleDPP(
        block_size, maximum_likelihood, &matrix_copy, &generator);
  }

The DPP can be sampled using a DAG-scheduler by instead calling
:samp:`catamari::OpenMPLowerFactorAndSampleSPP`:

.. code-block:: cpp

  #include "catamari.hpp"
  catamari::BlasMatrix<catamari::Complex<double>> matrix;
  matrix.Resize(num_rows, num_rows);
  // Fill the matrix with calls of the form: matrix(i, j) = value;

  // Ensure that the DAG-scheduled routine will use single-threaded BLAS calls.
  const int old_max_threads = catamari::GetMaxBlasThreads();
  catamari::SetNumBlasThreads(1);

  std::random_device random_device;
  std::mt19937 generator(random_device());
  const catamari::Int block_size = 64;
  const catamari::Int tile_size = 128;
  const bool maximum_likelihood = false;
  const int num_samples = 10;
  std::vector<std::vector<catamari::Int>> samples(num_samples);
  for (int sample_index = 0; sample_index < num_samples; ++sample_index) {
    auto matrix_copy = matrix;
    #pragma omp parallel
    #pragma omp single
    samples[sample_index] = catamari::OpenMPLowerFactorAndSampleDPP(
        tile_size, block_size, maximum_likelihood, &matrix_copy, &generator);
  }

  // Revert to the original number of BLAS threads.
  catamari::SetNumBlasThreads(old_max_threads);

An example of calling each of these routines can be found in
`example/dense_dpp.cc <https://gitlab.com/hodge_star/catamari/blob/master/example/dense_dpp.cc>`_. A more interest example, which builds and samples from a
dense DPP that uniformly samples spanning trees over a 2D grid graph, is given
in `example/uniform_spanning_tree.cc <https://gitlab.com/hodge_star/catamari/blob/master/example/uniform_spanning_tree.cc>`_.

Sparse DPP sampling
"""""""""""""""""""
Usage of catamari's sparse-direct DPP sampler via
:samp:`catamari::CoordinateMatrix` is similar to usage of the library's
sparse-direct solver.

.. code-block:: cpp

  #include "catamari.hpp"
  // Build a real or complex symmetric input matrix.
  //
  // Alternatively, one could use
  // catamari::CoordinateMatrix<Field>::FromMatrixMarket to read the matrix from
  // a Matrix Market file (e.g., from the Davis sparse matrix collection). But
  // keep in mind that one often needs to enforce explicit symmetry.
  catamari::CoordinateMatrix<double> matrix;
  matrix.Resize(num_rows, num_rows);
  matrix.ReserveEntryAdditions(num_entries_upper_bound);
  // Queue updates of entries in the sparse matrix using commands of the form:
  //   matrix.QueueEdgeAddition(row, column, value);
  matrix.FlushEntryQueues();

  // Construct the sampler.
  catamari::DPPControl dpp_control;
  catamari::DPP<double> dpp(matrix, dpp_control);

  // Extract samples (which can either be maximum-likelihood or not).
  const bool maximum_likelihood = false;
  std::vector<std::vector<catamari::Int>> samples;
  for (int sample_index = 0; sample_index < num_samples; ++sample_index) {
    samples[sample_index] = dpp.Sample(maximum_likelihood);
  }

A full example of sampling a DPP from a scaled negative 2D Laplacian is given at
`example/dpp_shifted_2d_negative_laplacian.cc <https://gitlab.com/hodge_star/catamari/blob/master/example/dpp_shifted_2d_negative_laplacian.cc>`_.
