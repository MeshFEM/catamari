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

  // [...]

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
  const Int max_refine_iters = 3;
  const double relative_tol = 1e-15;
  const bool verbose = true;
  factorization.RefinedSolve(
      matrix, relative_tol, max_refine_iters, verbose, &right_hand_sides.view);

One can also browse the `example/ <https://gitlab.com/hodge_star/catamari/tree/master/example>`_ folder for complete examples (e.g., for `solving 3D Helmholtz equations <https://gitlab.com/hodge_star/catamari/blob/master/example/helmholtz_3d_pml.cc>`_ with PML boundary conditions discretized using trilinear hexahedral elements using a complex LDL^T factorization).

Determinantal Point Process sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dense DPP sampling
""""""""""""""""""
TODO

Sparse DPP sampling
"""""""""""""""""""
Usage of catamari's sparse-direct DPP sampler via `catamari::CoordinateMatrix`
is similar to usage of the library's sparse-direct solver.

.. code-block:: cpp

  #include "catamari.hpp"

  // [...] 

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
