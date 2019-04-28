/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_FACTORIZATIONS_H_
#define CATAMARI_DENSE_FACTORIZATIONS_H_

#include <random>

#include "catamari/blas_matrix_view.hpp"
#include "catamari/buffer.hpp"
#include "catamari/complex.hpp"
#include "catamari/integers.hpp"

namespace catamari {

// Attempts to overwrite the lower triangle of an (implicitly) Hermitian
// Positive-Definite matrix with its lower-triangular Cholesky factor. The
// return value is the number of successful pivots; the factorization was
// successful if and only if the return value is the height of the input
// matrix.
template <class Field>
Int LowerCholeskyFactorization(Int block_size, BlasMatrixView<Field>* matrix);

#ifdef CATAMARI_OPENMP
template <class Field>
Int OpenMPLowerCholeskyFactorization(Int tile_size, Int block_size,
                                     BlasMatrixView<Field>* matrix);
#endif  // ifdef CATAMARI_OPENMP

// Attempts to overwrite the lower triangle of an (implicitly) Hermitian
// matrix with its L D L^H factorization, where L is lower-triangular with
// unit diagonal and D is diagonal (D is stored in place of the implicit
// unit diagonal). The return value is the number of successful pivots; the
// factorization was successful if and only if the return value is the height
// of the input matrix.
template <class Field>
Int LowerLDLAdjointFactorization(Int block_size, BlasMatrixView<Field>* matrix);

#ifdef CATAMARI_OPENMP
template <class Field>
Int OpenMPLowerLDLAdjointFactorization(Int tile_size, Int block_size,
                                       BlasMatrixView<Field>* matrix,
                                       Buffer<Field>* buffer);
#endif  // ifdef CATAMARI_OPENMP

// Attempts to overwrite the lower triangle of an (implicitly) symmetric
// matrix with its L D L^T factorization, where L is lower-triangular with
// unit diagonal and D is diagonal (D is stored in place of the implicit
// unit diagonal). The return value is the number of successful pivots; the
// factorization was successful if and only if the return value is the height
// of the input matrix.
template <class Field>
Int LowerLDLTransposeFactorization(Int block_size,
                                   BlasMatrixView<Field>* matrix);

#ifdef CATAMARI_OPENMP
template <class Field>
Int OpenMPLowerLDLTransposeFactorization(Int tile_size, Int block_size,
                                         BlasMatrixView<Field>* matrix,
                                         Buffer<Field>* buffer);
#endif  // ifdef CATAMARI_OPENMP

}  // namespace catamari

#include "catamari/dense_factorizations/cholesky-impl.hpp"
#include "catamari/dense_factorizations/cholesky_openmp-impl.hpp"
#include "catamari/dense_factorizations/ldl_adjoint-impl.hpp"
#include "catamari/dense_factorizations/ldl_adjoint_openmp-impl.hpp"
#include "catamari/dense_factorizations/ldl_transpose-impl.hpp"
#include "catamari/dense_factorizations/ldl_transpose_openmp-impl.hpp"

#endif  // ifndef CATAMARI_DENSE_FACTORIZATIONS_H_
