#!/bin/bash

HEADERS=`find include/ -print | grep "\.h"`
EXAMPLES=`find example/ -print | grep "\.cc"`
TESTS=`find test/ -print | grep "\.cc"`
SOURCE_FILES="${HEADERS} ${EXAMPLES} ${TESTS}"

# Replace "#pragma omp" with "//#pragma omp" to avoid poor formatting.
sed -i 's/#pragma omp/\/\/#pragma omp/g' ${SOURCE_FILES}

# Run clang-format on the modified files.
clang-format -i -style=file ${SOURCE_FILES}

# Undo the "#pragma omp" comments.
sed -i 's/\/\/ *#pragma omp/#pragma omp/g' ${SOURCE_FILES}

