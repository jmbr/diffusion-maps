"""Arnoldi iteration eigensolver using ARPACK-NG.

See also: https://github.com/opencollab/arpack-ng

"""


from ctypes import (byref, cdll, create_string_buffer, c_int, c_char,
                    c_char_p, c_double, POINTER, sizeof)
import ctypes.util
import logging
from typing import Optional, Tuple
import sys

try:
    import pycuda.gpuarray as gpuarray
except ImportError:
    gpuarray = None

import numpy as np
import scipy.sparse

from . import matrix_vector_product as mvp
from . import clock


EPSILON = sys.float_info.epsilon

MAX_ITERATIONS = int(1e7)


arpack = cdll.LoadLibrary(ctypes.util.find_library('arpack'))

dnaupd = arpack.dnaupd_
dnaupd.argtypes = [POINTER(c_int), c_char_p, POINTER(c_int), c_char_p,
                   POINTER(c_int), POINTER(c_double), POINTER(c_double),
                   POINTER(c_int), POINTER(c_double), POINTER(c_int),
                   POINTER(c_int), POINTER(c_int), POINTER(c_double),
                   POINTER(c_double), POINTER(c_int), POINTER(c_int)]
dnaupd_messages = dict([
    (0, "Normal exit."),
    (1, "Maximum number of iterations taken. "
     "All possible eigenvalues of OP has been found. "
     "IPARAM(5) returns the number of wanted converged Ritz values."),
    (2, "No longer an informational error. "
     "Deprecated starting with release 2 of ARPACK."),
    (3, "No shifts could be applied during a cycle of the Implicitly "
     "restarted Arnoldi iteration. One possibility is to increase the"
     " size of NCV relative to NEV."),
    (-1, "N must be positive."),
    (-2, "NEV must be positive."),
    (-3, "NCV-NEV >= 2 and less than or equal to N."),
    (-4, "The maximum number of Arnoldi update iteration must be "
     "greater than zero."),
    (-5, "WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'"),
    (-6, "BMAT must be one of 'I' or 'G'."),
    (-7, "Length of private work array is not sufficient."),
    (-8, "Error return from LAPACK eigenvalue calculation;"),
    (-9, "Starting vector is zero."),
    (-10, "IPARAM(7) must be 1,2,3,4."),
    (-11, "IPARAM(7) = 1 and BMAT = 'G' are incompatible."),
    (-12, "IPARAM(1) must be equal to 0 or 1."),
    (-9999, "Could not build an Arnoldi factorization. "
     "IPARAM(5) returns the size of the current Arnoldi factorization.")])

dneupd = arpack.dneupd_
# dneupd.argtypes = [POINTER(c_int), c_char_p, POINTER(c_int),
#                    POINTER(c_double), POINTER(c_double), POINTER(c_double),
#                    POINTER(c_int), POINTER(c_double), POINTER(c_double),
#                    POINTER(c_double), c_char_p, POINTER(c_int), c_char_p,
#                    POINTER(c_int), POINTER(c_double), POINTER(c_double),
#                    POINTER(c_int), POINTER(c_double), POINTER(c_int),
#                    POINTER(c_int), POINTER(c_int), POINTER(c_double),
#                    POINTER(c_double), POINTER(c_int), POINTER(c_int)]
dneupd_messages = dict([
    (0, "Normal exit."),
    (1, "The Schur form computed by LAPACK routine dlahqr could not be "
     "reordered by LAPACK routine dtrsen. Re-enter subroutine dneupd with "
     "IPARAM(5)=NCV and increase the size of the arrays DR and DI to have "
     "dimension at least dimension NCV and allocate at least NCV columns "
     "for Z. NOTE, \"Not necessary if Z and V share the same space. "
     "Please notify the authors if this error occurs.\""),
    (-1, "N must be positive."),
    (-2, "NEV must be positive."),
    (-3, "NCV-NEV >= 2 and less than or equal to N."),
    (-5, "WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'"),
    (-6, "BMAT must be one of 'I' or 'G'."),
    (-7, "Length of private work WORKL array is not sufficient."),
    (-8, "Error return from calculation of a real Schur form. "
     "Informational error from LAPACK routine dlahqr."),
    (-9, "Error return from calculation of eigenvectors. "
     "Informational error from LAPACK routine dtrevc."),
    (-10, "IPARAM(7) must be 1,2,3,4."),
    (-11, "IPARAM(7) = 1 and BMAT = 'G' are incompatible."),
    (-12, "HOWMNY = 'S' not yet implemented."),
    (-13, "HOWMNY must be one of 'A' or 'P' if RVEC = .true."),
    (-14, "DNAUPD did not find any eigenvalues to sufficient accuracy."),
    (-15, "DNEUPD got a different count of the number of converged Ritz "
     "values than DNAUPD got. This indicates the user probably made an "
     "error in passing data from DNAUPD to DNEUPD or that the data was "
     "modified before entering DNEUPD.")])


class ArpackError(Exception):
    pass


def eigensolver(matrix: scipy.sparse.csr_matrix,
                num_eigenpairs: Optional[int] = 10,
                sigma: Optional[float] = None,
                initial_vector: Optional[np.array] = None) \
        -> Tuple[np.array, np.array]:
    """Solve eigenvalue problem for sparse matrix.

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        A matrix in compressed sparse row format.
    num_eigenpairs : int, optional
        Number of eigenpairs to compute. Default is 10.
    sigma : float, optional
        Find eigenvalues close to the value of sigma. The default value is
        near 1.0. Currently unsupported on the GPU.
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
    if sigma is not None:
        raise RuntimeError('Shift/invert mode not implemented on the GPU')

    N = matrix.shape[0]

    if initial_vector is None:
        initial_vector = np.ones(N)

    n = c_int(N)                # Dimension of the eigenproblem.
    maxn = n
    nev = c_int(num_eigenpairs + 1)  # Number of eigenvalues to compute.
    ncv = c_int(num_eigenpairs + 3)  # Number of columns of the matrix V.
    maxncv = ncv.value
    ldv = c_int(maxn.value)

    # assert 0 < nev.value < n.value - 1
    # assert 2 - nev.value <= ncv.value <= n.value

    tol = c_double(EPSILON)

    d = (c_double * (3 * maxncv))()

    resid = initial_vector.ctypes.data_as(POINTER(c_double))

    vnp = np.zeros((maxncv, ldv.value), dtype=np.float64)
    v = vnp.ctypes.data_as(POINTER(c_double))

    workdnp = np.zeros(3 * maxn.value, dtype=np.float64)
    workd = workdnp.ctypes.data_as(POINTER(c_double))
    workev = (c_double * (3 * maxncv))()
    workl = (c_double * (3 * maxncv * maxncv + 6 * maxncv))()

    ipntr = (c_int * 14)()
    select = (c_int * maxncv)()

    bmat = create_string_buffer(b'I')  # B = I, standard eigenvalue problem.
    which = create_string_buffer(b'LM')  # Eigenvalues of largest magnitude.
    ido = c_int(0)
    lworkl = c_int(len(workl))
    info = c_int(1)

    ishfts = c_int(1)   # Use exact shifts.
    maxitr = c_int(MAX_ITERATIONS)
    # mode = c_int(3)           # A x = lambda x (OP = inv(A - sigma I), B = I)
    mode = c_int(1)           # A x = lambda x (OP = A, B = I)
    iparam = (c_int * 11)(ishfts, 0, maxitr, 0, 0, 0, mode)

    ierr = c_int(0)
    rvec = c_int(1)
    howmny = c_char(b'A')
    if sigma is not None:
        sigmar = c_double(np.real(sigma))
        sigmai = c_double(np.imag(sigma))
    else:
        sigmar = c_double()
        sigmai = c_double()

    MVP = mvp.MatrixVectorProduct(matrix)

    # eye = scipy.sparse.spdiags(np.ones(N), 0, N, N)
    # A = (matrix - (sigma) * eye).tocsc()
    # solve = scipy.sparse.linalg.factorized(A)

    logging.debug('Running Arnoldi iteration with tolerance {:g}...'
                  .format(tol.value))

    clk = clock.Clock()
    clk.tic()
    for itr in range(maxitr.value):
        dnaupd(byref(ido), bmat, byref(n), which, byref(nev), byref(tol),
               resid, byref(ncv), v, byref(ldv), iparam, ipntr, workd, workl,
               byref(lworkl), byref(info))

        if info.value != 0:
            raise ArpackError(dnaupd_messages[info.value])

        if ido.value == 99:
            break
        elif abs(ido.value) != 1:
            logging.warning('DNAUPD repoted IDO = {}'.format(ido.value))
            break

        idx_rhs, idx_sol = ipntr[0] - 1, ipntr[1] - 1
        rhs = workdnp[idx_rhs:(idx_rhs+N)]

        # # sol = solve(rhs)
        # sol = matrix @ rhs
        # workdnp[idx_sol:idx_sol+N] = sol
        sol = MVP.product(gpuarray.to_gpu(rhs.astype(np.float64)))
        workdnp[idx_sol:idx_sol+N] = sol.get()
    clk.toc()

    logging.debug('Done with Arnoldi iteration after {} steps. '
                  'Elapsed time: {} seconds'.format(itr, clk))

    logging.debug('Running post-processing step...')

    clk.tic()
    d0 = byref(d, 0)
    d1 = byref(d, maxncv * sizeof(c_double))
    dneupd(byref(rvec), byref(howmny), select, d0, d1, v, byref(ldv),
           byref(sigmar), byref(sigmai), workev, byref(bmat), byref(n),
           which, byref(nev), byref(tol), resid, byref(ncv), v, byref(ldv),
           iparam, ipntr, workd, workl, byref(lworkl), byref(ierr))
    clk.toc()

    logging.debug('Done with postprocessing step after {} seconds. '
                  'Status: {}'.format(clk, ('OK' if ierr.value == 0
                                            else 'FAIL')))

    if ierr.value != 0:
        raise ArpackError(dneupd_messages[ierr.value])

    nconv = iparam[4] - 1
    logging.debug('Converged on {} eigenpairs.'.format(nconv))

    ew = np.array(d[:nconv])
    ii = np.argsort(np.abs(ew))[::-1]
    ev = vnp[ii, :]

    return ew[ii], ev
