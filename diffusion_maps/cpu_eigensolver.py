"""Compute dominant eigenvectors of a sparse matrix.

"""

__all__ = ['eigensolver']

from typing import Optional, Tuple

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from . import default


def eigensolver(matrix: scipy.sparse.csr_matrix,
                num_eigenpairs: int = default.num_eigenpairs,
                sigma: Optional[float] = None,
                initial_vector: Optional[np.array] = None)  \
        -> Tuple[np.array, np.array]:
    """Solve  eigenvalue problem for sparse matrix.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        A matrix in compressed sparse row format.
    num_eigenpairs : int, optional
        Number of eigenvalue/eigenvector pairs to obtain.
    sigma : float, optional
        Find eigenvalues close to the value of sigma.
    initial_vector : np.array, optional
        Initial vector to use in the Arnoldi iteration. If not set, a vector
        with all entries equal to one will be used.

    Returns
    -------
    ew : np.array
        Eigenvalues in descending order of magnitude.
    ev : np.array
        Eigenvectors corresponding to the eigenvalues in `ew`.

    """
    if initial_vector is None:
        initial_vector = np.ones(matrix.shape[0])
    ew, ev = scipy.sparse.linalg.eigs(matrix, k=num_eigenpairs, which='LM',
                                      sigma=sigma, v0=initial_vector)
    ii = np.argsort(np.abs(ew))[::-1]
    return ew[ii], ev[:, ii].T
