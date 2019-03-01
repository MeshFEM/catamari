#
# Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
project(
  'catamari',
  'cpp',
  version : '0.1',
  license : 'MPLv2.0',
  meson_version : '>=0.46.0',
  default_options : ['cpp_std=c++14'])
include_dir = include_directories('include')
cxx = meson.get_compiler('cpp')

# A list of dependencies for building the library.
deps = []

# A list of dependencies for building the unit tests.
test_deps = []

# A list of dependencies for building the example drivers.
example_deps = []

# A list of C++ command-line arguments.
cxx_args = []

# Handle fast (Fortran-like) std::complex arithmetic.
if cxx.get_id() == 'gcc'
  message('Using gcc')
  # Complex arithmetic is greatly accelerated by adopting the Fortran
  # division and NaN-checking conventions.
  cxx_args += '-fcx-fortran-rules'
endif

if get_option('disable_tcmalloc')
  have_tcmalloc = false
else
  tcmalloc_dep = cxx.find_library('tcmalloc_minimal')
  have_tcmalloc = true
  deps += tcmalloc_dep
  cxx_args += '-DHAVE_TCMALLOC'
endif

# For controlling whether OpenMP support is enabled.
have_openmp = false
trust_omp_version = true
# Older versions of clang, e.g., 3.8.0-2ubuntu4, falsely report support for
# OpenMP 4.0 despite numerous issues (e.g., with task data sharing attributes).
# As of clang 8.0, only OpenMP 3.1 is supported. We therefore block OpenMP
# if the version of clang is less than 9.0.
if cxx.get_id() == 'clang' and cxx.version() < '9.0'
  message('Ignoring Clang\'s reported OpenMP version')
  trust_omp_version = false
endif
# Older versions of gcc, e.g., 5.4.0-6ubuntu1~16.04.11, falsely report support
# for OpenMP 4.0 despite bugs in task dependencies. However, version
# 5.5.0-12ubuntu1 appears to have fixed this issue.
if cxx.get_id() == 'gcc' and cxx.version() < '5.5'
  message('Ignoring GCC\'s reported OpenMP version')
  trust_omp_version = false
endif
if trust_omp_version
  if get_option('enable_openmp')
    openmp_dep = dependency('openmp', version : '>=4.0')
    have_openmp = true
    deps += openmp_dep
  elif not get_option('disable_openmp')
    openmp_dep = dependency('openmp', version : '>=4.0', required : false)
    if openmp_dep.found()
      have_openmp = true
      deps += openmp_dep
    endif
  endif
  if have_openmp
    cxx_args += '-DCATAMARI_OPENMP'
  endif
endif

# Simple BLAS/LAPACK detection.
if not get_option('disable_blas')
  have_blas = false
  have_mkl = false
  have_openblas = false

  if not get_option('disable_mkl')
    mkl_lib_dir = get_option('mkl_lib')
    if mkl_lib_dir == ''
      mkl_dirs = []
    else
      mkl_dirs = [mkl_lib_dir,]
    endif
    mkl_lib = cxx.find_library('libmkl_rt', dirs : mkl_dirs, required : false)
    if mkl_lib.found()
      if have_openmp
        mkl_intel_thread_lib = cxx.find_library('libmkl_intel_thread', \
            dirs : mkl_dirs, required : false)
        iomp5_lib = cxx.find_library('iomp5', dirs : mkl_dirs, required : false)
        if mkl_intel_thread_lib.found() and iomp5_lib.found()
          have_mkl = true
          deps += mkl_lib
          deps += mkl_intel_thread_lib
          deps += iomp5_lib
        else
          message('Found libmkl_rt but not mkl_intel_thread and iomp5')
          have_mkl = false
        endif
      else
        have_mkl = true
        deps += mkl_lib
      endif
      if have_mkl
        message('Found MKL')
        have_blas = true
        cxx_args += '-DCATAMARI_HAVE_MKL'
        cxx_args += '-DCATAMARI_HAVE_BLAS'
        cxx_args += '-DCATAMARI_HAVE_LAPACK'
      endif
    endif
  endif

  if not have_blas and not get_option('disable_openblas')
    openblas_lib = cxx.find_library('libopenblas', required : false)
    if openblas_lib.found()
      message('Found OpenBLAS')
      have_blas = true
      have_openblas = true
      cxx_args += '-DCATAMARI_HAVE_OPENBLAS'
      cxx_args += '-DCATAMARI_HAVE_BLAS'
      deps += openblas_lib

      lapack_lib = cxx.find_library('liblapack', required : false)
      if lapack_lib.found()
        message('Found LAPACK')
        cxx_args += '-DCATAMARI_HAVE_LAPACK'
        deps += lapack_lib
      endif
    endif
  endif
endif

# Test if quotient is installed.
found_quotient = cxx.has_header('quotient.hpp')
if not found_quotient
  quotient_proj = subproject('quotient')
  quotient_dep = quotient_proj.get_variable('quotient_dep')
  deps += quotient_dep
endif

# Test if specify is installed.
found_specify = cxx.has_header('specify.hpp')
if not found_specify
  specify_proj = subproject('specify')
  specify_dep = specify_proj.get_variable('specify_dep')
  example_deps += specify_dep
endif

# Add Catch for C++ unit tests.
test_deps += subproject('catch2').get_variable('catch2_dep')

# For controlling whether 32-bit or 64-bit integers are used. By default,
# 64-bit integers are used.
if get_option('use_64bit')
  cxx_args += '-DQUOTIENT_USE_64BIT'
endif

# For controlling the tradeoff between quality and speed for the supervariable
# hash function. It seems to typically be preferable to prefer the cheapest-
# possible hash.
if get_option('strong_hashes')
  cxx_args += '-DQUOTIENT_STRONG_HASHES'
endif

# For controlling whether QuotientGraph includes timers for the various stages
# of minimum-degree analysis.
if get_option('enable_timers')
  cxx_args += '-DQUOTIENT_ENABLE_TIMERS'
  cxx_args += '-DCATAMARI_ENABLE_TIMERS'
endif

# For controlling whether (expensive) debugging checks are performed.
if get_option('enable_debug')
  cxx_args += '-DQUOTIENT_DEBUG'
  cxx_args += '-DCATAMARI_DEBUG'
endif

# Tests for the Complex class.
complex_test_exe = executable(
    'complex_test',
    ['test/complex_test.cc', 'include/catamari.hpp'],
    include_directories : include_dir,
    dependencies : deps + test_deps,
    cpp_args : cxx_args)
test('Complex tests', complex_test_exe)

# Tests for the CoordinateMatrix class.
coordinate_matrix_test_exe = executable(
    'coordinate_matrix_test',
    ['test/coordinate_matrix_test.cc', 'include/catamari.hpp'],
    include_directories : include_dir,
    dependencies : deps + test_deps,
    cpp_args : cxx_args)
test('CoordinateMatrix tests', coordinate_matrix_test_exe)

# An test of factoring and solving against a 2D negative Laplacian.
laplacian_test_exe = executable(
    'laplacian_test',
    ['test/laplacian_test.cc', 'include/catamari.hpp'],
    include_directories : include_dir,
    dependencies : deps + test_deps,
    cpp_args : cxx_args)
test('Laplacian tests', laplacian_test_exe)

# A performance test for multithreaded dense outer products.
dense_outer_product_exe = executable(
    'dense_outer_product',
    ['example/dense_outer_product.cc', 'include/catamari.hpp'],
    include_directories : include_dir,
    dependencies : deps + example_deps,
    cpp_args : cxx_args)

# A performance test for dense DPP sampling.
dense_dpp_exe = executable(
    'dense_dpp',
    ['example/dense_dpp.cc', 'include/catamari.hpp'],
    include_directories : include_dir,
    dependencies : deps + example_deps,
    cpp_args : cxx_args)

# A performance test for factoring a dense matrix.
dense_factorization_exe = executable(
    'dense_factorization',
    ['example/dense_factorization.cc', 'include/catamari.hpp'],
    include_directories : include_dir,
    dependencies : deps + example_deps,
    cpp_args : cxx_args)

# A utility for factoring a Matrix Market input.
factor_matrix_market_exe = executable(
    'factor_matrix_market',
    ['example/factor_matrix_market.cc', 'include/catamari.hpp'],
    include_directories : include_dir,
    dependencies : deps + example_deps,
    cpp_args : cxx_args)

# An example of factoring a 2D Helmholtz discretization.
helmholtz_2d_pml_exe = executable(
    'helmholtz_2d_pml',
    ['example/helmholtz_2d_pml.cc', 'include/catamari.hpp'],
    include_directories : include_dir,
    dependencies : deps + example_deps,
    cpp_args : cxx_args)

# An example of factoring a 3D Helmholtz discretization.
helmholtz_3d_pml_exe = executable(
    'helmholtz_3d_pml',
    ['example/helmholtz_3d_pml.cc', 'include/catamari.hpp'],
    include_directories : include_dir,
    dependencies : deps + example_deps,
    cpp_args : cxx_args)

# A utility for sampling a DPP from a Matrix Market input.
dpp_matrix_market_exe = executable(
    'dpp_matrix_market',
    ['example/dpp_matrix_market.cc', 'include/catamari.hpp'],
    include_directories : include_dir,
    dependencies : deps + example_deps,
    cpp_args : cxx_args)

# A utility for sampling a DPP from a 2D Laplacian.
dpp_shifted_2d_negative_laplacian_exe = executable(
    'dpp_shifted_2d_negative_laplacian',
    ['example/dpp_shifted_2d_negative_laplacian.cc', 'include/catamari.hpp'],
    include_directories : include_dir,
    dependencies : deps + example_deps,
    cpp_args : cxx_args)

# For using catamari as a subproject.
catamari_dep = declare_dependency(include_directories : include_dir)

# Install include/catamari.hpp and include/catamari/
install_subdir('include', install_dir : '.')
