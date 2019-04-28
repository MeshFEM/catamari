/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_DPP_HERMITIAN_DPP_IMPL_H_
#define CATAMARI_DENSE_DPP_HERMITIAN_DPP_IMPL_H_

#include <cmath>

#include "catamari/dense_basic_linear_algebra.hpp"
#include "catamari/lapack.hpp"

#include "catamari/dense_dpp.hpp"

namespace catamari {

template <class Field>
std::vector<Int> UnblockedSampleLowerHermitianDPP(bool maximum_likelihood,
                                                  BlasMatrixView<Field>* matrix,
                                                  std::mt19937* generator) {
  typedef ComplexBase<Field> Real;
  const Int height = matrix->height;
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
      const Field eta = delta * Conjugate(matrix->Entry(j, i));
      for (Int k = j; k < height; ++k) {
        const Field& lambda_left = matrix->Entry(k, i);
        matrix->Entry(k, j) -= lambda_left * eta;
      }
    }
  }
  return sample;
}

template <class Field>
std::vector<Int> BlockedSampleLowerHermitianDPP(Int block_size,
                                                bool maximum_likelihood,
                                                BlasMatrixView<Field>* matrix,
                                                std::mt19937* generator) {
  const Int height = matrix->height;

  std::vector<Int> sample;
  sample.reserve(height);

  Buffer<Field> buffer(std::max(height - block_size, Int(0)) * block_size);
  BlasMatrixView<Field> factor;
  factor.data = buffer.Data();

  for (Int i = 0; i < height; i += block_size) {
    const Int bsize = std::min(height - i, block_size);

    // Overwrite the diagonal block with its LDL' factorization.
    BlasMatrixView<Field> diagonal_block =
        matrix->Submatrix(i, i, bsize, bsize);
    std::vector<Int> block_sample = UnblockedSampleLowerHermitianDPP(
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
    RightLowerAdjointUnitTriangularSolves(diagonal_block.ToConst(),
                                          &subdiagonal);

    // Copy the conjugate of the current factor.
    factor.height = subdiagonal.height;
    factor.width = subdiagonal.width;
    factor.leading_dim = subdiagonal.height;
    for (Int j = 0; j < subdiagonal.width; ++j) {
      for (Int k = 0; k < subdiagonal.height; ++k) {
        factor(k, j) = Conjugate(subdiagonal(k, j));
      }
    }

    // Solve against the diagonal.
    for (Int j = 0; j < subdiagonal.width; ++j) {
      const ComplexBase<Field> delta = RealPart(diagonal_block(j, j));
      for (Int k = 0; k < subdiagonal.height; ++k) {
        subdiagonal(k, j) /= delta;
      }
    }

    // Perform the Hermitian rank-bsize update.
    BlasMatrixView<Field> submatrix = matrix->Submatrix(
        i + bsize, i + bsize, height - (i + bsize), height - (i + bsize));
    MatrixMultiplyLowerNormalTranspose(Field{-1}, subdiagonal.ToConst(),
                                       factor.ToConst(), Field{1}, &submatrix);
  }
  return sample;
}

template <class Field>
std::vector<Int> SampleLowerHermitianDPP(Int block_size,
                                         bool maximum_likelihood,
                                         BlasMatrixView<Field>* matrix,
                                         std::mt19937* generator) {
  return BlockedSampleLowerHermitianDPP(block_size, maximum_likelihood, matrix,
                                        generator);
}

}  // namespace catamari

#endif  // ifndef CATAMARI_DENSE_DPP_HERMITIAN_DPP_IMPL_H_
