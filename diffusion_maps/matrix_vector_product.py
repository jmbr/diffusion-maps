import numpy as np

import scipy.sparse

import pycuda.gpuarray as gpuarray

from . import cusparse as cs


class MatrixVectorProduct:
    """Perform GPU-based, sparse matrix-vector products."""
    def __init__(self, matrix: scipy.sparse.csr_matrix) -> None:
        self.m = matrix.shape[0]
        self.n = matrix.shape[1]
        self.nnz = matrix.nnz
        self.csrValA = gpuarray.to_gpu(matrix.data.astype(np.float64))
        self.csrRowPtrA = gpuarray.to_gpu(matrix.indptr)
        self.csrColIndA = gpuarray.to_gpu(matrix.indices)
        self.handle = cs.cusparseCreate()
        self.descr = cs.cusparseCreateMatDescr()

    def __del__(self) -> None:
        if self.descr is not None:
            cs.cusparseDestroyMatDescr(self.descr)
            self.descr = None
        if self.handle is not None:
            cs.cusparseDestroy(self.handle)
            self.handle = None

    def product(self, x: gpuarray.GPUArray) -> gpuarray.GPUArray:
        """Multiply sparse matrix by dense vector."""
        y = gpuarray.empty_like(x)
        op = cs.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE
        cs.cusparseDcsrmv(self.handle, op, self.m, self.n, self.nnz, 1.0,
                          self.descr, self.csrValA, self.csrRowPtrA,
                          self.csrColIndA, x, 0.0, y)
        return y
