// JP: support, e.g., Apple's accelerate framework.
#ifndef CATAMARI_BLAS_GENERIC_H_
#define CATAMARI_BLAS_GENERIC_H_

#ifndef BLAS_SYMBOL
#define BLAS_SYMBOL(name) name
#endif

// TODO(Jack Poulson): Attempt to support 64-bit BLAS when Int = long long int.
typedef int BlasInt;
typedef std::complex<float> BlasComplexFloat;
typedef std::complex<double> BlasComplexDouble;

#endif  // ifndef CATAMARI_BLAS_GENERIC_H_
