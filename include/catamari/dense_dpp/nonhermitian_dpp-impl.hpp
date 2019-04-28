/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_DPP_NONHERMITIAN_DPP_IMPL_H_
#define CATAMARI_DENSE_DPP_NONHERMITIAN_DPP_IMPL_H_

#include <cmath>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/lapack.hpp"

#include "catamari/dense_dpp.hpp"

namespace catamari {

template <class Field>
std::vector<Int> UnblockedSampleNonHermitianDPP(bool maximum_likelihood,
                                                BlasMatrixView<Field>* matrix,
                                                std::mt19937* generator) {
  typedef ComplexBase<Field> Real;
  const Int height = matrix->height;
  CATAMARI_ASSERT(height == matrix->width, "Can only sample square kernels.");
  std::vector<Int> sample;
  sample.reserve(height);

  std::uniform_real_distribution<Real> uniform_dist{Real{0}, Real{1}};

#ifdef CATAMARI_DEBUG
  const Real tolerance = 10 * std::numeric_limits<Real>::epsilon();
#endif  // ifdef CATAMARI_DEBUG

  for (Int i = 0; i < height; ++i) {
    Real delta = RealPart(matrix->Entry(i, i));
    CATAMARI_ASSERT(
        delta >= -tolerance && delta <= Real{1} + tolerance,
        "Diagonal value was outside of [0, 1]: " + std::to_string(delta));
    CATAMARI_ASSERT(std::abs(ImagPart(matrix->Entry(i, i))) <= tolerance,
                    "Imaginary part of diagonal was " +
                        std::to_string(ImagPart(matrix->Entry(i, i))));
    const bool keep_index = maximum_likelihood
                                ? delta >= Real(1) / Real(2)
                                : uniform_dist(*generator) <= delta;
    if (keep_index) {
      sample.push_back(i);
    } else {
      delta -= Real{1};
      matrix->Entry(i, i) = delta;
    }

    // Solve for the remainder of the i'th column of L.
    for (Int k = i + 1; k < height; ++k) {
      matrix->Entry(k, i) /= delta;
    }

    // Perform the rank-one update.
    for (Int j = i + 1; j < height; ++j) {
      const Field eta = matrix->Entry(i, j);
      for (Int k = i + 1; k < height; ++k) {
        const Field gamma = matrix->Entry(k, i);
        matrix->Entry(k, j) -= gamma * eta;
      }
    }
  }
  return sample;
}

template <class Field>
std::vector<Int> BlockedSampleNonHermitianDPP(Int block_size,
                                              bool maximum_likelihood,
                                              BlasMatrixView<Field>* matrix,
                                              std::mt19937* generator) {
  const Int height = matrix->height;

  std::vector<Int> sample;
  sample.reserve(height);

  for (Int i = 0; i < height; i += block_size) {
    const Int bsize = std::min(height - i, block_size);

    // Overwrite the diagonal block with its LDL' factorization.
    BlasMatrixView<Field> diagonal_block =
        matrix->Submatrix(i, i, bsize, bsize);
    std::vector<Int> block_sample = UnblockedSampleNonHermitianDPP(
        maximum_likelihood, &diagonal_block, generator);
    for (const Int& index : block_sample) {
      sample.push_back(i + index);
    }
    if (height == i + bsize) {
      break;
    }

    // Solve for the remainder of the block column of L.
    BlasMatrixView<Field> subdiagonal =
        matrix->Submatrix(i + bsize, i, height - (i + bsize), bsize);
    RightUpperTriangularSolves(diagonal_block.ToConst(), &subdiagonal);

    // Solve for the remainder of the block row of U.
    BlasMatrixView<Field> superdiagonal =
        matrix->Submatrix(i, i + bsize, bsize, height - (i + bsize));
    LeftLowerUnitTriangularSolves(diagonal_block.ToConst(), &superdiagonal);

    // Perform the rank-bsize update.
    BlasMatrixView<Field> submatrix = matrix->Submatrix(
        i + bsize, i + bsize, height - (i + bsize), height - (i + bsize));
    MatrixMultiplyNormalNormal(Field{-1}, subdiagonal.ToConst(),
                               superdiagonal.ToConst(), Field{1}, &submatrix);
  }
  return sample;
}

template <class Field>
std::vector<Int> SampleNonHermitianDPP(Int block_size, bool maximum_likelihood,
                                       BlasMatrixView<Field>* matrix,
                                       std::mt19937* generator) {
  return BlockedSampleNonHermitianDPP(block_size, maximum_likelihood, matrix,
                                      generator);
}

}  // namespace catamari

#endif  // ifndef CATAMARI_DENSE_DPP_NONHERMITIAN_DPP_IMPL_H_
