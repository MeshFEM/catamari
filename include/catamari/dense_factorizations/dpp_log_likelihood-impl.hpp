/*
 * Copyright (c) 2018 Jack Poulson <jack@hodgestar.com>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CATAMARI_DENSE_FACTORIZATIONS_DPP_LOG_LIKELIHOOD_IMPL_H_
#define CATAMARI_DENSE_FACTORIZATIONS_DPP_LOG_LIKELIHOOD_IMPL_H_

#include <cmath>

#include "catamari/dense_factorizations.hpp"

namespace catamari {

template <typename Field>
ComplexBase<Field> DPPLogLikelihood(const BlasMatrixView<Field>& matrix) {
  typedef ComplexBase<Field> Real;
  const Int matrix_size = matrix.height;
  Real log_likelihood = 0;
  for (Int i = 0; i < matrix_size; ++i) {
    Real entry = quotient::RealPart(matrix(i, i));
    if (entry > Real(0)) {
      log_likelihood += std::log(entry);
    } else if (entry < Real(0)) {
      log_likelihood += std::log(-entry);
    } else {
      std::cerr << "Had an exactly zero diagonal entry of the result."
                << std::endl;
    }
  }
  return log_likelihood;
}

}  // namespace catamari

#endif  // ifndef CATAMARI_DENSE_FACTORIZATIONS_DPP_LOG_LIKELIHOOD_IMPL_H_
