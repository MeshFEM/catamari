.. catamari documentation master file, created by
   sphinx-quickstart on Mon Mar  4 10:29:06 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Quickstart
----------
The primary functionality of the library consists of dual implementations of
a statically-ordered sparse-direct solver for symmetric/Hermitian matrices and
a sparse-direct Determinantal Point Process sampler. Both support real and
complex matrices and have sequential and multithreaded (via OpenMP's task
scheduler) implementations built on top of custom dense kernels.

This brief quickstart guide walks through building the examples and tests using
the `meson <https://mesonbuild.com>`_ build system and provides short overviews
of the major functionality.

Building the examples and tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
While `catamari <https://hodgestar.com/catamari/>`_ is a header-only library,
there are a number of configuration options (e.g., which BLAS library to use,
and whether OpenMP is enabled) which are handled through preprocessor directives
and compiler/linker options determined during a configuration stage. Catamari
uses `meson <https://mesonbuild.com>`_, a modern alternative to
`CMake <https://cmake.org/>`_, for this configuration.

One might start with a debug (the default :samp:`buildtype` in
`meson <https://mesonbuild.com>`_). Assuming that the
`Ninja build system <https://ninja-build.org>`_ was installed alongside
meson (it typically is), one can configure, build, and test catamari with its
default options via:

.. code-block:: bash

  mkdir build-debug/
  meson build-debug
  cd build-debug
  ninja
  ninja test

A release version can be built by specifying the :samp:`buildtype` option as
:samp:`release`:

.. code-block:: bash

  mkdir build-release/
  meson build-release -Dbuild-type=release
  cd build-release
  ninja
  ninja test

OpenMP task parallelism will be used by default if support for OpenMP was
detected; shared-memory parallelism can be disabled with the
:samp:`-Ddisable_openmp=true` configuration option.

And `Intel MKL <https://software.intel.com/en-us/mkl>`_ support can be enabled
by configuring with :samp:`-Dmkl_lib=/PATH/TO/MKL/LIB`.


Symmetric and Hermitian direct linear solvers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Catamari's linear system solvers are targeted to the class of matrices which
can be (reasonably stably) factored with either Cholesky, :math:`LDL^T`, or
:math:`LDL^H` factorizations, where :math:`D` is diagonal and :math:`L` is
unit lower-triangular. This class is a strict (but large) subset of symmetric
and Hermitian systems that contains Hermitian Quasi-Definite [GeorgeEtAl-2006]_
and a complex-symmetric matrices with positive-definite real and imaginary
components [Higham-1998]_.

Dense factorizations
""""""""""""""""""""
Beyond their intrinsic usefulness, high-performance dense factorizations are a
core component of supernodal sparse-direct solvers. Catamari therefore provides
sequential and multithreaded (via OpenMP's task scheduler) implementations of
dense Cholesky, :math:`LDL^T`, and :math:`LDL^H` factorizations (as one might
infer, for both real and complex scalars).

Sequential (perhaps using multithreaded BLAS calls) dense Cholesky
factorizations can be easily performed using a call to 
:samp:`catamari::LowerCholeskyFactorization` on a
:samp:`catamari::BlasMatrixView`, which is essentially a pointer and dense
matrix metadata. One can either manually handle the memory allocation and
manipulation of a :samp:`catamari::BlasMatrixView` or have it automatically
maintained as a member of a :samp:`catamari::BlasMatrix`, which handles
resource allocation and destruction.

.. code-block:: cpp

  #include "catamari.hpp"
  // Build a dense Hermitian positive-definite matrix.
  catamari::BlasMatrix<catamari::Complex<double>> matrix;
  matrix.Resize(num_rows, num_rows);
  // Fill the matrix using commands of the form:
  //   matrix(row, column) = value;
  
  // Perform the sequential, dense Cholesky factorization using a
  // user-determined algorithmic blocksize.
  const catamari::Int block_size = 64;
  catamari::LowerCholeskyFactorization(block_size, &matrix.view);

Multithreaded dense Cholesky factorization can similarly be performed with a
call to :samp:`catamari::OpenMPLowerCholeskyFactorization`, though care must be
taken to avoid thread oversubscription by ensuring that only a single thread is
used for each BLAS call. Each OpenMP routine in Catamari assumes that it is
within a :samp:`#pragma omp single` section of an :samp:`#pragma omp parallel`
region.

.. code-block:: cpp

  #include "catamari.hpp"
  // Build a dense Hermitian positive-definite matrix.
  catamari::BlasMatrix<catamari::Complex<double>> matrix;
  matrix.Resize(num_rows, num_rows);
  // Fill the matrix using commands of the form:
  //   matrix(row, column) = value;

  // Avoid BLAS thread oversubscription.
  const int old_max_threads = catamari::GetMaxBlasThreads();
  catamari::SetNumBlasThreads(1);
  
  // Perform the sequential, dense Cholesky factorization using a
  // user-determined algorithmic blocksize.
  const catamari::Int tile_size = 128;
  const catamari::Int block_size = 64;
  #pragma omp parallel
  #pragma omp single
  catamari::OpenMPLowerCholeskyFactorization(
      tile_size, block_size, &matrix.view);

  // Restore the number of BLAS threads.
  catamari::SetNumBlasThreads(old_max_threads);

Real and complex :math:`LDL^T` and :math:`LDL^H` can be executed with nearly
identical code by instead calling
:samp:`catamari::LowerLDLTransposeFactorization`, 
:samp:`catamari::OpenMPLowerLDLTransposeFactorization`, 
:samp:`catamari::LowerLDLAdjointFactorization`,  or
:samp:`catamari::OpenMPLowerLDLAdjointFactorization`.

Please see
`example/dense_factorization.cc <https://gitlab.com/hodge_star/catamari/blob/master/example/dense_factorization.cc>`_
for full examples of using the sequential and multithreaded dense factorizations.

Sparse-direct solver
""""""""""""""""""""
Usage of catamari's sparse-direct solver through the
:samp:`catamari::CoordinateMatrix` template class is fairly straight-forward
and has an identical interface in sequential and multithreaded contexts
(the multithreaded solver is called if more the maximum number of OpenMP threads
is detected as greater than one).

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

One can also browse the
`example/ <https://gitlab.com/hodge_star/catamari/tree/master/example>`_ folder
for complete examples (e.g., for
`solving 3D Helmholtz equations <https://gitlab.com/hodge_star/catamari/blob/master/example/helmholtz_3d_pml.cc>`_
with PML boundary conditions discretized using trilinear hexahedral elements
using a complex :math:`LDL^T` factorization).

Determinantal Point Process sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Catamari's
`Determinantal Point Process <https://en.wikipedia.org/wiki/Determinantal_point_process>`_
samplers all operate directly on the *marginal kernel matrix*: if
:math:`P` is a determinantal point process  with a ground set of cardinality
:math:`n` (so that we may identify the ground set with indices
:math:`\mathcal{Y} = [0, ..., n)`, the probability of a subset
:math:`A \subseteq \mathcal{Y}` being in a random sample
:math:`\mathbf{Y} \subseteq 2^\mathcal{Y}` is given by

.. math::

   P[A \subseteq \mathbf{Y}] = \text{det}(K_A), 

where :math:`K_A` is the :math:`|A| \times |A|` restriction of the row and
column indices of the marginal kernel matrix :math:`K` to :math:`A`.

The eigenvalues of a marginal kernel matrix are restricted to live in
:math:`[0, 1]` (ensuring that all minors are valid probabilities). And, in
the vast majority of cases (Cf. [Soshnikov-2000]_), including all of those relevant to this library,
marginal kernel matrices are assumed to be Hermitian. Thus, we will henceforth
assume all marginal kernel matrices Hermitian Positive Semi-Definite with
two-norm bounded from above by 1.

Essentially all of the high-performance techniques for performing a dense or
sparse-direct :math:`LDL^H` factorization can be carried over to directly
sampling a DPP from its marginal kernel matrix by exploiting the relationship
between Schur complements and conditional DPP sampling (by sequentially flipping
a Bernoulli coin based upon the value of each pivot to determine whether its
corresponding index should be in the sample).

TODO(Jack Poulson): Explain the mathematics behind this relationship.


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

The DPP can be sampled using OpenMP's DAG-scheduler by instead calling
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

.. [GeorgeEtAl-2006] Alan George, K.H. Irkamov, and A.B. Kucherov, Some properties of symmetric quasi-definite matrices, SIAM J. Matrix Anal. Appl., 21(4), pp. 1318--1323, 2006. DOI: https://epubs.siam.org/doi/10.1137/S0895479897329400

.. [Higham-1998] Nicholas J. Higham, Factorizing complex symmetric matrices with positive definite real and imaginary parts, Mathematics of Computation, 64(224), pp. 1591--1599, 1998. URL: https://www.ams.org/journals/mcom/1998-67-224/S0025-5718-98-00978-8/S0025-5718-98-00978-8.pdf

.. [Soshnikov-2000] A. Soshnikov, Determinantal random point fields. Russian Math. Surveys, 2000, 55 (5), 923–975. URL: https://arxiv.org/abs/math/0002099
