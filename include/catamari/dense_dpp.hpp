/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_DPP_H_
#define CATAMARI_DENSE_DPP_H_

#include <random>

#include "catamari/blas_matrix_view.hpp"
#include "catamari/buffer.hpp"
#include "catamari/complex.hpp"
#include "catamari/dense_factorizations.hpp"
#include "catamari/integers.hpp"

namespace catamari {

// Returns a sample from the Determinantal Point Process implied by the
// marginal kernel matrix (i.e., a Hermitian matrix with all eigenvalues
// contained in [0, 1]). The input matrix is overwritten with the associated
// L D L^H factorization of the modified kernel (each diagonal pivot with a
// failed coin-flip is decremented by one).
template <class Field>
std::vector<Int> SampleLowerHermitianDPP(Int block_size,
                                         bool maximum_likelihood,
                                         BlasMatrixView<Field>* matrix,
                                         std::mt19937* generator);

#ifdef CATAMARI_OPENMP
template <class Field>
std::vector<Int> OpenMPSampleLowerHermitianDPP(Int tile_size, Int block_size,
                                               bool maximum_likelihood,
                                               BlasMatrixView<Field>* matrix,
                                               std::mt19937* generator,
                                               std::vector<Field>* buffer);
#endif  // ifdef CATAMARI_OPENMP

// Returns a sample from the elementary Determinantal Point Process implied by
// the marginal kernel matrix (i.e., a Hermitian matrix with all eigenvalues
// contained in {0, 1}). The input matrix is overwritten with the associated
// L D L^H factorization of the modified kernel.
//
// Choosing the most performance block size can be nontrivial: if the block size
// is greater than or equal to the rank, then the factorization reduces to a
// lazy, level-2 BLAS based approach with complexity O(n rank^2), whereas, for
// smaller block sizes, the complexity becomes O(n^2 rank). When n / rank is
// large, which is where the elementary sampler performs best, the lazy
// approach should be preferred.
template <class Field>
std::vector<Int> SampleElementaryLowerHermitianDPP(
    Int block_size, Int rank, bool maximum_likelihood,
    BlasMatrixView<Field>* matrix, std::mt19937* generator);
// TODO(Jack Poulson): Implement an OpenMP version of the elementary sampler.

// Returns a sample from the Determinantal Point Process implied by a
// *non-Hermitian* marginal kernel matrix: a real or complex matrix with real
// diagonal which satisfies [1]
//
//     (-1)^{|J|} det(K - I_J) >= 0 for all J \subseteq [n].
//
// The input matrix is overwritten with the associated L U factorization of the
// modified kernel (each diagonal pivot with a failed coin-flip is decremented
// by one).
//
// [1] Brunel, Learning Signed Determinantal Point Processes through the
//     Principal Minor Assignment Problem.
//
template <class Field>
std::vector<Int> SampleNonHermitianDPP(Int block_size, bool maximum_likelihood,
                                       BlasMatrixView<Field>* matrix,
                                       std::mt19937* generator);

#ifdef CATAMARI_OPENMP
template <class Field>
std::vector<Int> OpenMPSampleNonHermitianDPP(Int tile_size, Int block_size,
                                             bool maximum_likelihood,
                                             BlasMatrixView<Field>* matrix,
                                             std::mt19937* generator);
#endif  // ifdef CATAMARI_OPENMP

// Returns the log-likelihood of a general DPP sample based upon the product of
// the (real part of the) diagonal of the factored result.
template <typename Field>
ComplexBase<Field> DPPLogLikelihood(const ConstBlasMatrixView<Field>& matrix);

// Returns the log-likelihood of an elementary DPP sample of the given rank
// based upon
//
//   P[Y \subseteq \mathbb{Y}] = P[Y = \mathbb{Y}] = det(K_Y) =
//       det(L_Y)^2 = prod(diag(L_Y))^2,
//
// where K_Y is the restriction of the marginal to the sample Y. For full-rank
// samples of a determinantal projection process, the marginal determinants
// are equivalent to the likelihood of the sample.
template <typename Field>
ComplexBase<Field> ElementaryDPPLogLikelihood(
    Int rank, const ConstBlasMatrixView<Field>& matrix);

}  // namespace catamari

#include "catamari/dense_dpp/dpp_log_likelihood-impl.hpp"
#include "catamari/dense_dpp/elementary_hermitian_dpp-impl.hpp"
#include "catamari/dense_dpp/hermitian_dpp-impl.hpp"
#include "catamari/dense_dpp/hermitian_dpp_openmp-impl.hpp"
#include "catamari/dense_dpp/nonhermitian_dpp-impl.hpp"
#include "catamari/dense_dpp/nonhermitian_dpp_openmp-impl.hpp"

#endif  // ifndef CATAMARI_DENSE_DPP_H_
