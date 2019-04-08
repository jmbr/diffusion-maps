"""Minimalistic interface to the NVIDIA cuSPARSE library.

"""

__all__ = ['cusparseCreate', 'cusparseDestroy', 'cusparseGetVersion',
           'cusparseCreateMatDescr', 'cusparseDestroyMatDescr',
           'cusparseDcsrmv']


import ctypes
import ctypes.util
from enum import IntEnum

from numba.cuda.cudadrv.devicearray import DeviceNDArray


class cusparseContext(ctypes.Structure):
    pass


cusparseHandle_t = ctypes.POINTER(cusparseContext)


class cusparseMatDescr(ctypes.Structure):
    pass


cusparseMatDescr_t = ctypes.POINTER(cusparseMatDescr)


libcusparse = ctypes.cdll.LoadLibrary(ctypes.util.find_library('cusparse'))

libcusparse.cusparseCreate.restype = int
libcusparse.cusparseCreate.argtypes = [ctypes.POINTER(cusparseHandle_t)]

libcusparse.cusparseDestroy.restype = int
libcusparse.cusparseDestroy.argtypes = [cusparseHandle_t]

libcusparse.cusparseGetVersion.restype = int
libcusparse.cusparseGetVersion.argtypes = [cusparseHandle_t, ctypes.c_void_p]

libcusparse.cusparseCreateMatDescr.restype = int
libcusparse.cusparseCreateMatDescr.argtypes = [ctypes.POINTER(cusparseMatDescr_t)]

libcusparse.cusparseDestroyMatDescr.restype = int
libcusparse.cusparseDestroyMatDescr.argtypes = [cusparseMatDescr_t]

libcusparse.cusparseDcsrmv.restype = int
libcusparse.cusparseDcsrmv.argtypes = [cusparseHandle_t,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_int,
                                       ctypes.c_void_p,
                                       cusparseMatDescr_t,
                                       ctypes.c_ulong,
                                       ctypes.c_ulong,
                                       ctypes.c_ulong,
                                       ctypes.c_ulong,
                                       ctypes.c_void_p,
                                       ctypes.c_ulong]


class cusparseOperation(IntEnum):
    CUSPARSE_OPERATION_NON_TRANSPOSE = 0
    CUSPARSE_OPERATION_TRANSPOSE = 1
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2


class cusparseError(Exception):
    """Error in call to cuSPARSE library."""
    error_message = {
        0: 'The operation completed successfully.',
        1: 'The cuSPARSE library was not initialized. This is usually caused by the lack of a prior call, an error in the CUDA Runtime API called by the cuSPARSE routine, or an error in the hardware setup.',
        2: 'Resource allocation failed inside the cuSPARSE library. This is usually caused by a cudaMalloc() failure.',
        3: 'An unsupported value or parameter was passed to the function (a negative vector size, for example).',
        4: 'The function requires a feature absent from the device architecture; usually caused by the lack of support for atomic operations or double precision.',
        5: 'An access to GPU memory space failed, which is usually caused by a failure to bind a texture.',
        6: 'The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.',
        7: 'An internal cuSPARSE operation failed. This error is usually caused by a cudaMemcpyAsync() failure.',
        8: 'The matrix type is not supported by this function. This is usually caused by passing an invalid matrix descriptor to the function.',
        9: 'CUSPARSE_STATUS_ZERO_PIVOT.'
    }

    def __init__(self, status):
        self.status = status

    def __str__(self) -> str:
        return self.error_message[self.status]


def cusparseCreate() -> cusparseHandle_t:
    handle = cusparseHandle_t()

    status = libcusparse.cusparseCreate(ctypes.byref(handle))
    if status != 0:
        raise cusparseError(status)

    return handle


def cusparseDestroy(handle: cusparseHandle_t) -> None:
    status = libcusparse.cusparseDestroy(handle)
    if status != 0:
        raise cusparseError(status)

    return handle


def cusparseGetVersion(handle: cusparseHandle_t) -> int:
    version = ctypes.c_int()

    status = libcusparse.cusparseGetVersion(handle, ctypes.byref(version))
    if status != 0:
        raise cusparseError(status)

    return version


def cusparseCreateMatDescr() -> ctypes.c_int:
    descr = cusparseMatDescr_t()

    status = libcusparse.cusparseCreateMatDescr(ctypes.byref(descr))
    if status != 0:
        raise cusparseError(status)

    return descr


def cusparseDestroyMatDescr(descr: cusparseMatDescr_t) -> None:
    status = libcusparse.cusparseDestroyMatDescr(descr)
    if status != 0:
        raise cusparseError(status)


def cusparseDcsrmv(handle: cusparseHandle_t, transA: cusparseOperation,
                   m: int, n: int, nnz: int, alpha: float,
                   descrA: cusparseMatDescr_t, csrValA: DeviceNDArray,
                   csrRowPtrA: DeviceNDArray, csrColIndA: DeviceNDArray,
                   x: DeviceNDArray, beta:
                   float, y: DeviceNDArray) -> None:
    alpha_ = ctypes.c_double(alpha)
    beta_ = ctypes.c_double(beta)
    status = libcusparse.cusparseDcsrmv(handle,
                                        transA,
                                        m,
                                        n,
                                        nnz,
                                        ctypes.byref(alpha_),
                                        descrA,
                                        csrValA.device_ctypes_pointer,
                                        csrRowPtrA.device_ctypes_pointer,
                                        csrColIndA.device_ctypes_pointer,
                                        x.device_ctypes_pointer,
                                        ctypes.byref(beta_),
                                        y.device_ctypes_pointer)
    if status != 0:
        raise cusparseError(status)
