import numpy as np

import scipy.sparse

import numba.cuda as cuda
DeviceNDArray = cuda.cudadrv.devicearray.DeviceNDArray


from . import cusparse as cs


class MatrixVectorProduct:
    """Perform GPU-based, sparse matrix-vector products."""
    def __init__(self, matrix: scipy.sparse.csr_matrix) -> None:
        self.m = matrix.shape[0]
        self.n = matrix.shape[1]
        self.nnz = matrix.nnz
        self.csrValA = cuda.to_device(matrix.data.astype(np.float64))
        self.csrRowPtrA = cuda.to_device(matrix.indptr)
        self.csrColIndA = cuda.to_device(matrix.indices)
        self.handle = cs.cusparseCreate()
        self.descr = cs.cusparseCreateMatDescr()

    def __del__(self) -> None:
        if hasattr(self, 'descr'):
            cs.cusparseDestroyMatDescr(self.descr)
            self.descr = None
        if hasattr(self, 'handle'):
            cs.cusparseDestroy(self.handle)
            self.handle = None

    def product(self, x: DeviceNDArray) -> DeviceNDArray:
        """Multiply sparse matrix by dense vector."""
        y = cuda.device_array(self.m)
        op = cs.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE
        cs.cusparseDcsrmv(self.handle,
                          op,
                          self.m,
                          self.n,
                          self.nnz,
                          1.0,
                          self.descr,
                          self.csrValA,
                          self.csrRowPtrA,
                          self.csrColIndA,
                          x,
                          0.0,
                          y)
        return y
