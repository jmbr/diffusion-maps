"""Miscellaneous utilities.

"""


__all__ = ['coo_tocsr']


import numpy as np
import scipy.sparse

from . import clock


@clock.log
def coo_tocsr(A: scipy.sparse.coo_matrix) -> scipy.sparse.csr_matrix:
    """Convert matrix to Compressed Sparse Row format, fast.

    This function is derived from the corresponding SciPy code but it avoids
    the sanity checks that slow `scipy.sparse.coo_matrix.to_csr down`. In
    particular, by not summing duplicates we can attain important speed-ups
    for large matrices.

    """
    from scipy.sparse import csr_matrix
    if A.nnz == 0:
        return csr_matrix(A.shape, dtype=A.dtype)

    m, n = A.shape
    # Using 32-bit integer indices allows for matrices of up to 2,147,483,647
    # non-zero entries.
    idx_dtype = np.int32
    row = A.row.astype(idx_dtype, copy=False)
    col = A.col.astype(idx_dtype, copy=False)

    indptr = np.empty(n+1, dtype=idx_dtype)
    indices = np.empty_like(row, dtype=idx_dtype)
    data = np.empty_like(A.data, dtype=A.dtype)

    scipy.sparse._sparsetools.coo_tocsr(m, n, A.nnz, row, col, A.data,
                                        indptr, indices, data)

    return csr_matrix((data, indices, indptr), shape=A.shape)
