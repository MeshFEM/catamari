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

deps = []
test_deps = []
cxx_args = []

if cxx.get_id() == 'gcc'
  message('Using gcc')
  # Complex arithmetic is greatly accelerated by adopting the Fortran
  # division and NaN-checking conventions.
  cxx_args += '-fcx-fortran-rules'
endif

# Add Catch
test_deps += subproject('catch2').get_variable('catch2_dep')

# Simple BLAS/LAPACK detection.
if not get_option('disable-blas')
  have_blas = false
  have_mkl = false
  have_openblas = false

  if not get_option('disable-mkl')
    mkl_lib = cxx.find_library('libmkl_rt', required : false)
    if mkl_lib.found()
      message('Found MKL')
      have_blas = true
      have_mkl = true
      cxx_args += '-DCATAMARI_HAVE_MKL'
      cxx_args += '-DCATAMARI_HAVE_LAPACK'
      deps += mkl_lib
    endif 
  endif

  if not have_blas and not get_option('disable-openblas')
    openblas_lib = cxx.find_library('libopenblas', required : false)
    if openblas_lib.found()
      message('Found OpenBLAS')
      have_blas = true
      have_openblas = true
      cxx_args += '-DCATAMARI_HAVE_OPENBLAS'
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

# For controlling whether 32-bit or 64-bit integers are used. By default,
# 64-bit integers are used.
if get_option('use-64bit')
  cxx_args += '-DQUOTIENT_USE_64BIT'
endif

# For controlling the tradeoff between quality and speed for the supervariable
# hash function. It seems to typically be preferable to prefer the cheapest-
# possible hash.
if get_option('strong-hashes')
  cxx_args += '-DQUOTIENT_STRONG_HASHES'
endif

# For controlling whether QuotientGraph includes timers for the various stages
# of minimum-degree analysis.
if get_option('enable-timers')
  cxx_args += '-DQUOTIENT_ENABLE_TIMERS'
  cxx_args += '-DCATAMARI_ENABLE_TIMERS'
endif

# For controlling whether (expensive) debugging checks are performed.
if get_option('enable-debug')
  cxx_args += '-DQUOTIENT_DEBUG'
  cxx_args += '-DCATAMARI_DEBUG'
endif

# For controlling whether (still not implemented) OpenMP support is enabled.
if get_option('enable-openmp')
  openmp_dep = dependency('openmp', version : '>=3.0')
  deps += openmp_dep
elif get_option('disable-openmp')
  openmp_dep = false
else
  openmp_dep = dependency('openmp', version : '>=3.0', required : false)
  deps += openmp_dep
endif

# Tests for the Complex class.
complex_test_exe = executable(
    'complex_test',
    ['test/complex_test.cc', 'include/catamari.hpp'],
    include_directories : include_dir,
    dependencies : deps + test_deps,
    cpp_args : cxx_args)

# Tests for the CoordinateMatrix class.
coordinate_matrix_test_exe = executable(
    'coordinate_matrix_test',
    ['test/coordinate_matrix_test.cc', 'include/catamari.hpp'],
    include_directories : include_dir,
    dependencies : deps + test_deps,
    cpp_args : cxx_args)

# A utility for factoring a Matrix Market input.
factor_matrix_market_exe = executable(
    'factor_matrix_market',
    ['example/factor_matrix_market.cc', 'include/catamari.hpp'],
    include_directories : include_dir,
    dependencies : deps,
    cpp_args : cxx_args)

# An example of factoring a 2D Helmholtz discretization.
helmholtz_2d_pml_exe = executable(
    'helmholtz_2d_pml',
    ['example/helmholtz_2d_pml.cc', 'include/catamari.hpp'],
    include_directories : include_dir,
    dependencies : deps,
    cpp_args : cxx_args)

# An example of factoring a 3D Helmholtz discretization.
helmholtz_3d_pml_exe = executable(
    'helmholtz_3d_pml',
    ['example/helmholtz_3d_pml.cc', 'include/catamari.hpp'],
    include_directories : include_dir,
    dependencies : deps,
    cpp_args : cxx_args)

# A utility for sampling a DPP from a Matrix Market input.
dpp_matrix_market_exe = executable(
    'dpp_matrix_market',
    ['example/dpp_matrix_market.cc', 'include/catamari.hpp'],
    include_directories : include_dir,
    dependencies : deps,
    cpp_args : cxx_args)

# A utility for sampling a DPP from a 2D Laplacian.
dpp_shifted_2d_negative_laplacian_exe = executable(
    'dpp_shifted_2d_negative_laplacian',
    ['example/dpp_shifted_2d_negative_laplacian.cc', 'include/catamari.hpp'],
    include_directories : include_dir,
    dependencies : deps,
    cpp_args : cxx_args)

# For using catamari as a subproject.
catamari_dep = declare_dependency(include_directories : include_dir)

# Install include/catamari.hpp and include/catamari/
install_subdir('include', install_dir : '.')
