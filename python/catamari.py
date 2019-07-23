# A preliminary Python interface to Catamari's dense DPP samplers.
import ctypes
import numpy as np
import time

from ctypes import (c_bool, c_int, c_float, c_double, c_longlong, c_void_p,
                    POINTER)

memory_view_from_memory = ctypes.pythonapi.PyMemoryView_FromMemory
memory_view_from_memory.restype = ctypes.py_object

# We need to ensure that libcatamari_c is in LD_LIBRARY_PATH.
# TODO(Jack Poulson): Use the correct extension.
lib = ctypes.cdll.LoadLibrary('libcatamari_c.so')


def Have64BitInts():
    """Queries whether or not Catamari was configured with 64-bit integers."""
    has_64_bit_ints = c_bool()
    lib.CatamariHas64BitInts.argtypes = [POINTER(c_bool)]
    lib.CatamariHas64BitInts(ctypes.pointer(has_64_bit_ints))
    return has_64_bit_ints.value


def HaveOpenMP():
    """Queries whether or not Catamari was compiled with OpenMP support."""
    has_openmp = c_bool()
    lib.CatamariHasOpenMP.argtypes = [POINTER(c_bool)]
    lib.CatamariHasOpenMP(ctypes.pointer(has_openmp))
    return has_openmp.value


# Cache whether Catamari was compiled with 64-bit integers.
have_64_bit_ints = Have64BitInts()

# Cache whether Catamari was compiled with OpenMP support.
have_openmp = HaveOpenMP()

# Choose the appropriate ctypes and numpy types for catamari::Int.
catamari_int = c_longlong if have_64_bit_ints else c_int
catamari_np_int = np.int64 if have_64_bit_ints else np.int32


class BufferInt(ctypes.Structure):
    """An equivalent of the C structure CatamariBufferInt."""
    _fields_ = [('size', catamari_int), ('data', POINTER(catamari_int))]
    lib.CatamariBufferIntInit.argtypes = [c_void_p]
    lib.CatamariBufferIntDestroy.argtypes = [c_void_p]

    def __init__(self):
        lib.CatamariBufferIntInit(ctypes.byref(self))

    def __del__(self):
        lib.CatamariBufferIntDestroy(ctypes.byref(self))


class ComplexFloat(ctypes.Structure):
    """An equivalent of a C struct rep'ing a complex number as two floats."""
    _fields_ = [('real', c_float), ('imag', c_float)]

    def __init__(self, real_value=0, imag_value=0):
        """Constructs a complex number from two real values."""
        real = c_float()
        imag = c_float()
        if type(real_value) is ComplexFloat or type(
                real_value) is ComplexDouble:
            real = real_value.real
            imag = real_value.imag
        else:
            real = real_value
            imag = imag_value
        super(ComplexFloat, self).__init__(real, imag)


class ComplexDouble(ctypes.Structure):
    """An equivalent of a C struct rep'ing a complex number as two doubles."""
    _fields_ = [('real', c_double), ('imag', c_double)]

    def __init__(self, real_value=0, imag_value=0):
        """Constructs a complex number from two real values."""
        real = c_double()
        imag = c_double()
        if type(real_value) is ComplexFloat or type(
                real_value) is ComplexDouble:
            real = real_value.real
            imag = real_value.imag
        else:
            real = real_value
            imag = imag_value
        super(ComplexDouble, self).__init__(real, imag)


def numpy_dtype_to_ctypes(dtype):
    """Converts a numpy.dtype into our corresponding ctypes value."""
    if dtype == catamari_np_int:
        return catamari_int
    elif dtype == np.float32:
        return c_float
    elif dtype == np.float64:
        return c_double
    elif dtype == np.complex64:
        return ComplexFloat
    elif dtype == np.complex128:
        return ComplexDouble
    else:
        raise ValueError('Invalid numpy datatype')


class BlasMatrixView(object):
    """An analogue of catamari::BlasMatrixView<Field>."""

    class ViewInt(ctypes.Structure):
        """An equivalent of CatamariBlasMatrixViewInt."""
        _fields_ = [('height', catamari_int), ('width', catamari_int),
                    ('leading_dim', catamari_int),
                    ('data', POINTER(catamari_int))]

    class ViewFloat(ctypes.Structure):
        """An equivalent of CatamariBlasMatrixViewFloat."""
        _fields_ = [('height', catamari_int), ('width', catamari_int),
                    ('leading_dim', catamari_int), ('data', POINTER(c_float))]

    class ViewDouble(ctypes.Structure):
        """An equivalent of CatamariBlasMatrixViewDouble."""
        _fields_ = [('height', catamari_int), ('width', catamari_int),
                    ('leading_dim', catamari_int), ('data', POINTER(c_double))]

    class ViewComplexFloat(ctypes.Structure):
        """An equivalent of CatamariBlasMatrixViewComplexFloat."""
        _fields_ = [('height', catamari_int), ('width', catamari_int),
                    ('leading_dim', catamari_int),
                    ('data', POINTER(ComplexFloat))]

    class ViewComplexDouble(ctypes.Structure):
        """An equivalent of CatamariBlasMatrixViewComplexDouble."""
        _fields_ = [('height', catamari_int), ('width', catamari_int),
                    ('leading_dim', catamari_int),
                    ('data', POINTER(ComplexDouble))]

    def __init__(self, dtype):
        self.dtype = dtype
        if dtype == catamari_np_int:
            self.view = BlasMatrixView.ViewInt()
        elif dtype == np.float32:
            self.view = BlasMatrixView.ViewFloat()
        elif dtype == np.float64:
            self.view = BlasMatrixView.ViewDouble()
        elif dtype == np.complex64:
            self.view = BlasMatrixView.ViewComplexFloat()
        elif dtype == np.complex128:
            self.view = BlasMatrixView.ViewComplexDouble()
        else:
            raise ValueError('Invalid datatype for BlasMatrixView')

    @classmethod
    def from_numpy(cls, matrix):
        """Converts a numpy matrix into a BlasMatrixView."""
        if len(matrix.shape) != 2:
            raise ValueError(
                'Expected a matrix input for BlasMatrixView.from_numpy')
        mat_f = matrix if matrix.flags.f_contiguous else np.asfortranarray(
            matrix)

        ctypes_dtype = numpy_dtype_to_ctypes(mat_f.dtype)

        view = BlasMatrixView(mat_f.dtype)
        view.view.height = mat_f.shape[0]
        view.view.width = mat_f.shape[1]
        view.view.leading_dim = mat_f.shape[0]
        view.view.data = mat_f.ctypes.data_as(POINTER(ctypes_dtype))

        return view


def SampleLowerHermitianDPP(matrix_view,
                            maximum_likelihood=False,
                            use_openmp=False,
                            block_size=64,
                            tile_size=256):
    """Samples a Hermitian DPP from its marginal kernel.

    Args:
      matrix_view (BlasMatrixView): The view of the copy of the marginal kernel
          matrix that is to be probabilistically factored in-place.
      maximum_likelihood (ctypes.c_bool): Whether a maximum-likelihood sample
          is desired.
      use_openmp (boolean): Whether OpenMP should be used.
      block_size (catamari_int): The algorithmic block size of the
        factorization.
      tile_size (catamari_int): The OpenMP task tile size for the factorization.

    Returns:
      A numpy array containing the sample. 
    """
    sample = BufferInt()
    if use_openmp and not have_openmp:
        print('Tried to use OpenMP despite not having support.')
    if use_openmp and have_openmp:
        if matrix_view.dtype == np.float32:
            lib.CatamariOpenMPSampleLowerHermitianDPPFloat.argtypes = [
                catamari_int, catamari_int, c_bool,
                POINTER(BlasMatrixView.ViewFloat),
                POINTER(BufferInt)
            ]
            lib.CatamariOpenMPSampleLowerHermitianDPPFloat(
                catamari_int(tile_size), catamari_int(block_size),
                c_bool(maximum_likelihood), ctypes.byref(matrix_view.view),
                ctypes.byref(sample))
        elif matrix_view.dtype == np.float64:
            lib.CatamariOpenMPSampleLowerHermitianDPPDouble.argtypes = [
                catamari_int, catamari_int, c_bool,
                POINTER(BlasMatrixView.ViewDouble),
                POINTER(BufferInt)
            ]
            lib.CatamariOpenMPSampleLowerHermitianDPPDouble(
                catamari_int(tile_size), catamari_int(block_size),
                c_bool(maximum_likelihood), ctypes.byref(matrix_view.view),
                ctypes.byref(sample))
        elif matrix_view.dtype == np.complex64:
            lib.CatamariOpenMPSampleLowerHermitianDPPComplexFloat.argtypes = [
                catamari_int, catamari_int, c_bool,
                POINTER(BlasMatrixView.ViewComplexFloat),
                POINTER(BufferInt)
            ]
            lib.CatamariOpenMPSampleLowerHermitianDPPComplexFloat(
                catamari_int(tile_size), catamari_int(block_size),
                c_bool(maximum_likelihood), ctypes.byref(matrix_view.view),
                ctypes.byref(sample))
        elif matrix_view.dtype == np.complex128:
            lib.CatamariOpenMPSampleLowerHermitianDPPComplexDouble.argtypes = [
                catamari_int, catamari_int, c_bool,
                POINTER(BlasMatrixView.ViewComplexDouble),
                POINTER(BufferInt)
            ]
            lib.CatamariOpenMPSampleLowerHermitianDPPComplexDouble(
                catamari_int(tile_size), catamari_int(block_size),
                c_bool(maximum_likelihood), ctypes.byref(matrix_view.view),
                ctypes.byref(sample))
        else:
            raise ValueError('Invalid np.dtype for SampleLowerHermitianDPP.')
    else:
        if matrix_view.dtype == np.float32:
            lib.CatamariSampleLowerHermitianDPPFloat.argtypes = [
                catamari_int, c_bool,
                POINTER(BlasMatrixView.ViewFloat),
                POINTER(BufferInt)
            ]
            lib.CatamariSampleLowerHermitianDPPFloat(
                catamari_int(block_size), c_bool(maximum_likelihood),
                ctypes.byref(matrix_view.view), ctypes.byref(sample))
        elif matrix_view.dtype == np.float64:
            lib.CatamariSampleLowerHermitianDPPDouble.argtypes = [
                catamari_int, c_bool,
                POINTER(BlasMatrixView.ViewDouble),
                POINTER(BufferInt)
            ]
            lib.CatamariSampleLowerHermitianDPPDouble(
                catamari_int(block_size), c_bool(maximum_likelihood),
                ctypes.byref(matrix_view.view), ctypes.byref(sample))
        elif matrix_view.dtype == np.complex64:
            lib.CatamariSampleLowerHermitianDPPComplexFloat.argtypes = [
                catamari_int, c_bool,
                POINTER(BlasMatrixView.ViewComplexFloat),
                POINTER(BufferInt)
            ]
            lib.CatamariSampleLowerHermitianDPPComplexFloat(
                catamari_int(block_size), c_bool(maximum_likelihood),
                ctypes.byref(matrix_view.view), ctypes.byref(sample))
        elif matrix_view.dtype == np.complex128:
            lib.CatamariSampleLowerHermitianDPPComplexDouble.argtypes = [
                catamari_int, c_bool,
                POINTER(BlasMatrixView.ViewComplexDouble),
                POINTER(BufferInt)
            ]
            lib.CatamariSampleLowerHermitianDPPComplexDouble(
                catamari_int(block_size), c_bool(maximum_likelihood),
                ctypes.byref(matrix_view.view), ctypes.byref(sample))
        else:
            raise ValueError('Invalid np.dtype for SampleLowerHermitianDPP.')

    sample_memory_view = memory_view_from_memory(
        sample.data,
        ctypes.sizeof(catamari_int) * sample.size)
    return np.frombuffer(sample_memory_view, catamari_int).copy()


def SampleNonHermitianDPP(matrix_view,
                          maximum_likelihood=False,
                          use_openmp=False,
                          block_size=64,
                          tile_size=256):
    """Samples a non-Hermitian DPP from its marginal kernel.

    Args:
      matrix_view (BlasMatrixView): The view of the copy of the marginal kernel
          matrix that is to be probabilistically factored in-place.
      maximum_likelihood (ctypes.c_bool): Whether a maximum-likelihood sample
          is desired.
      use_openmp (boolean): Whether OpenMP should be used.
      block_size (catamari_int): The algorithmic block size of the
          factorization.
      tile_size (catamari_int): The OpenMP task tile size for the factorization.

    Returns:
      A numpy array containing the sample. 
    """
    sample = BufferInt()
    if use_openmp and not have_openmp:
        print('Tried to use OpenMP despite not having support.')
    if use_openmp and have_openmp:
        if matrix_view.dtype == np.float32:
            lib.CatamariOpenMPSampleNonHermitianDPPFloat.argtypes = [
                catamari_int, catamari_int, c_bool,
                POINTER(BlasMatrixView.ViewFloat),
                POINTER(BufferInt)
            ]
            lib.CatamariOpenMPSampleNonHermitianDPPFloat(
                catamari_int(tile_size), catamari_int(block_size),
                c_bool(maximum_likelihood), ctypes.byref(matrix_view.view),
                ctypes.byref(sample))
        elif matrix_view.dtype == np.float64:
            lib.CatamariOpenMPSampleNonHermitianDPPDouble.argtypes = [
                catamari_int, catamari_int, c_bool,
                POINTER(BlasMatrixView.ViewDouble),
                POINTER(BufferInt)
            ]
            lib.CatamariOpenMPSampleNonHermitianDPPDouble(
                catamari_int(tile_size), catamari_int(block_size),
                c_bool(maximum_likelihood), ctypes.byref(matrix_view.view),
                ctypes.byref(sample))
        elif matrix_view.dtype == np.complex64:
            lib.CatamariOpenMPSampleNonHermitianDPPComplexFloat.argtypes = [
                catamari_int, catamari_int, c_bool,
                POINTER(BlasMatrixView.ViewComplexFloat),
                POINTER(BufferInt)
            ]
            lib.CatamariOpenMPSampleNonHermitianDPPComplexFloat(
                catamari_int(tile_size), catamari_int(block_size),
                c_bool(maximum_likelihood), ctypes.byref(matrix_view.view),
                ctypes.byref(sample))
        elif matrix_view.dtype == np.complex128:
            lib.CatamariOpenMPSampleNonHermitianDPPComplexDouble.argtypes = [
                catamari_int, catamari_int, c_bool,
                POINTER(BlasMatrixView.ViewComplexDouble),
                POINTER(BufferInt)
            ]
            lib.CatamariOpenMPSampleNonHermitianDPPComplexDouble(
                catamari_int(tile_size), catamari_int(block_size),
                c_bool(maximum_likelihood), ctypes.byref(matrix_view.view),
                ctypes.byref(sample))
        else:
            raise ValueError('Invalid np.dtype for SampleLowerHermitianDPP.')
    else:
        if matrix_view.dtype == np.float32:
            lib.CatamariSampleNonHermitianDPPFloat.argtypes = [
                catamari_int, c_bool,
                POINTER(BlasMatrixView.ViewFloat),
                POINTER(BufferInt)
            ]
            lib.CatamariSampleNonHermitianDPPFloat(
                catamari_int(block_size), c_bool(maximum_likelihood),
                ctypes.byref(matrix_view.view), ctypes.byref(sample))
        elif matrix_view.dtype == np.float64:
            lib.CatamariSampleNonHermitianDPPDouble.argtypes = [
                catamari_int, c_bool,
                POINTER(BlasMatrixView.ViewDouble),
                POINTER(BufferInt)
            ]
            lib.CatamariSampleNonHermitianDPPDouble(
                catamari_int(block_size), c_bool(maximum_likelihood),
                ctypes.byref(matrix_view.view), ctypes.byref(sample))
        elif matrix_view.dtype == np.complex64:
            lib.CatamariSampleNonHermitianDPPComplexFloat.argtypes = [
                catamari_int, c_bool,
                POINTER(BlasMatrixView.ViewComplexFloat),
                POINTER(BufferInt)
            ]
            lib.CatamariSampleNonHermitianDPPComplexFloat(
                catamari_int(block_size), c_bool(maximum_likelihood),
                ctypes.byref(matrix_view.view), ctypes.byref(sample))
        elif matrix_view.dtype == np.complex128:
            lib.CatamariSampleNonHermitianDPPComplexDouble.argtypes = [
                catamari_int, c_bool,
                POINTER(BlasMatrixView.ViewComplexDouble),
                POINTER(BufferInt)
            ]
            lib.CatamariSampleNonHermitianDPPComplexDouble(
                catamari_int(block_size), c_bool(maximum_likelihood),
                ctypes.byref(matrix_view.view), ctypes.byref(sample))
        else:
            raise ValueError('Invalid np.dtype for SampleLowerHermitianDPP.')

    sample_memory_view = memory_view_from_memory(
        sample.data,
        ctypes.sizeof(catamari_int) * sample.size)
    return np.frombuffer(sample_memory_view, catamari_int).copy()


def DPPLogLikelihood(matrix_view):
    """Returns the log-likelihood of a sample given its factored kernel.

    Args:
      matrix_view (BlasMatrixView): The view of the factored marginal kernel.

    Returns:
      The log-likelihood of the sample.
    """
    if matrix_view.dtype == np.float32:
        log_likelihood = c_float()
        lib.CatamariDPPLogLikelihoodFloat.argtypes = [
            POINTER(BlasMatrixView.ViewFloat),
            POINTER(c_float)
        ]
        lib.CatamariDPPLogLikelihoodFloat(
            ctypes.byref(matrix_view.view), ctypes.pointer(log_likelihood))
        return log_likelihood
    elif matrix_view.dtype == np.float64:
        log_likelihood = c_double()
        lib.CatamariDPPLogLikelihoodDouble.argtypes = [
            POINTER(BlasMatrixView.ViewDouble),
            POINTER(c_double)
        ]
        lib.CatamariDPPLogLikelihoodDouble(
            ctypes.byref(matrix_view.view), ctypes.pointer(log_likelihood))
        return log_likelihood
    elif matrix_view.dtype == np.complex64:
        log_likelihood = c_float()
        lib.CatamariDPPLogLikelihoodComplexFloat.argtypes = [
            POINTER(BlasMatrixView.ViewComplexFloat),
            POINTER(c_float)
        ]
        lib.CatamariDPPLogLikelihoodComplexFloat(
            ctypes.byref(matrix_view.view), ctypes.pointer(log_likelihood))
        return log_likelihood
    elif matrix_view.dtype == np.complex128:
        log_likelihood = c_double()
        lib.CatamariDPPLogLikelihoodComplexDouble.argtypes = [
            POINTER(BlasMatrixView.ViewComplexDouble),
            POINTER(c_double)
        ]
        lib.CatamariDPPLogLikelihoodComplexDouble(
            ctypes.byref(matrix_view.view), ctypes.pointer(log_likelihood))
        return log_likelihood
    else:
        raise ValueError('Invalid np.dtype for DPPLogLikelihood.')


if __name__ == '__main__':
    num_rows = 10000
    identity = np.identity(num_rows, dtype=np.complex128)
    identity_view = BlasMatrixView.from_numpy(identity)

    sample_start = time.time()
    block_size = 64
    tile_size = 256
    sample = SampleLowerHermitianDPP(
        identity_view,
        use_openmp=False,
        block_size=block_size,
        tile_size=tile_size)
    sample_stop = time.time()
    print(sample)
    print(sample.shape)

    log_likelihood = DPPLogLikelihood(identity_view)
    print(log_likelihood)

    print('Sampling took: {} seconds.'.format(sample_stop - sample_start))
