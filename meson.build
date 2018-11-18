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

# Test if quotient is installed.
found_quotient = cxx.has_header('quotient.hpp')
if not found_quotient
  quotient_proj = subproject('quotient')
  quotient_dep = quotient_proj.get_variable('quotient_dep')
  deps += quotient_dep
endif

# Tests for the Complex class.
complex_test_exe = executable(
    'complex_test',
    ['test/complex_test.cc', 'include/catamari.hpp'],
    include_directories : include_dir,
    dependencies : deps)

# Tests for the CoordinateMatrix class.
coordinate_matrix_test_exe = executable(
    'coordinate_matrix_test',
    ['test/coordinate_matrix_test.cc', 'include/catamari.hpp'],
    include_directories : include_dir,
    dependencies : deps)
