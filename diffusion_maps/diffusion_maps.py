"""Diffusion maps module.

This module implements the diffusion maps method for dimensionality
reduction, as introduced in:

Coifman, R. R., & Lafon, S. (2006). Diffusion maps. Applied and Computational
Harmonic Analysis, 21(1), 5–30. DOI:10.1016/j.acha.2006.04.006

"""

__all__ = ['BaseDiffusionMaps', 'SparseDiffusionMaps', 'DenseDiffusionMaps',
           'DiffusionMaps', 'downsample']


import logging
import sys
from typing import Optional, Dict, Tuple

import numpy as np
import scipy
import scipy.sparse
import scipy.spatial

from . import default
from . import utils
from . import clock


def downsample(data: np.array, num_samples: int) -> np.array:
    """Randomly sample a subset of a data set while preserving order.

    The sampling is done without replacement.

    Parameters
    ----------
    data : np.array
        Array whose 0-th axis indexes the data points.
    num_samples : int
        Number of items to randomly (uniformly) sample from the data.  This
        is typically less than the total number of elements in the data set.

    Returns
    -------
    sampled_data : np.array
       A total of `num_samples` uniformly randomly sampled data points from
       `data`.

    """
    assert num_samples <= data.shape[0]
    indices = sorted(np.random.choice(range(data.shape[0]), num_samples,
                                      replace=False))
    return data[indices, :]


def DiffusionMaps(*args, **kwargs):
    """Convenience function to select the right diffusion map routine.

    Consult the documentation for BaseDiffusionMaps and its constructor for
    details.

    """
    cut_off = kwargs.get('cut_off')

    if cut_off is None or np.isinf(cut_off):
        kwargs['cut_off'] = None
        return DenseDiffusionMaps(*args, **kwargs)
    else:
        return SparseDiffusionMaps(*args, **kwargs)


class BaseDiffusionMaps:
    """Diffusion maps.

    Attributes
    ----------
    epsilon : float
        Bandwidth for kernel.
    kernel_matrix : scipy.sparse.spmatrix
        (Possibly stochastic) matrix obtained by evaluating a Gaussian kernel
        on the data points.
    renormalization : float or None
        Renormalization exponent (alpha in the diffusion maps literature).
    eigenvectors : np.array
        Right eigenvectors of `kernel_matrix`.
    eigenvalues : np.array
        Eigenvalues of `kernel_matrix`.

    """
    def __init__(self, points: np.array, epsilon: float,
                 cut_off: Optional[float] = None,
                 num_eigenpairs: Optional[int] = default.num_eigenpairs,
                 normalize_kernel: Optional[bool] = True,
                 renormalization: Optional[float] = default.renormalization,
                 kdtree_options: Optional[Dict] = None,
                 use_cuda: Optional[bool] = default.use_cuda) \
            -> None:
        """Compute diffusion maps.

        This function computes the eigendecomposition of the transition
        matrix associated to a random walk on the data using a bandwidth
        (time) equal to epsilon.

        Parameters
        ----------
        points : np.array
            Data set to analyze. Its 0-th axis must index each data point.
        epsilon : float
            Bandwidth to use for the kernel.
        cut_off : float, optional
            Cut-off for the distance matrix computation. It should be at
            least equal to `epsilon`.
        num_eigenpairs : int, optional
            Number of eigenpairs to compute. Default is
            `default.num_eigenpairs`.
        normalize_kernel : bool, optional
            Whether to convert the kernel into a stochastic matrix or
            not. Default is `True`.
        renormalization : float, optional
            Renormalization exponent to use if `normalize_kernel` is
            True. This is the parameter $\alpha$ in the diffusion maps
            literature. It must take a value between zero and one.
        kdtree_options : dict, optional
            A dictionary containing parameters to pass to the underlying
            cKDTree object.
        use_cuda : bool, optional
            Determine whether to use CUDA-enabled eigenvalue solver or not.

        """
        raise NotImplementedError

    @clock.log
    def normalize_kernel_matrix(self, matrix, alpha: Optional[float] = 1):
        """Compute normalized random walk Laplacian from similarity matrix.

        Parameters
        ----------
        matrix
            A similarity matrix obtained by evaluating a kernel function on a
            distance matrix.
        alpha : float, optional
            Renormalization parameter. The value of `alpha` must lie in the
            closed unit interval.

        Returns
        -------
        matrix
            A (suitably normalized) row-stochastic random walk Laplacian.

        """
        pass

    @staticmethod
    def make_stochastic_matrix(matrix: scipy.sparse.csr_matrix) \
            -> scipy.sparse.csr_matrix:
        """Convert a non-negative matrix into a row-stochastic matrix.

        Carries out the normalization (in the 1-norm) of each row of a
        non-negative matrix inplace.  The matrix should be in Compressed
        Sparse Row format. Note that this method overwrites the input matrix.

        Parameters
        ----------
        matrix
            A matrix with non-negative entries to be converted.

        Returns
        -------
        matrix
            The same matrix passed as input but normalized.

        """
        raise NotImplementedError

    @staticmethod
    @clock.log
    def solve_eigenproblem(kernel_matrix, num_eigenpairs: int,
                           use_cuda: bool) -> Tuple[np.array, np.array]:
        """Solve eigenvalue problem using CPU or GPU solver.

        """
        if use_cuda is True:
            from .gpu_eigensolver import eigensolver
        else:
            from .cpu_eigensolver import eigensolver

        return eigensolver(kernel_matrix, num_eigenpairs)


class SparseDiffusionMaps(BaseDiffusionMaps):
    """Diffusion maps.

    Attributes
    ----------
    epsilon : float
        Bandwidth for kernel.
    _cut_off : float
        Cut off for the computation of pairwise distances between points.
    _kdtree : cKDTree
        KD-tree for accelerating pairwise distance computation.
    kernel_matrix : scipy.sparse.spmatrix
        (Possibly stochastic) matrix obtained by evaluating a Gaussian kernel
        on the data points.
    renormalization : float or None
        Renormalization exponent (alpha in the diffusion maps literature).
    eigenvectors : np.array
        Right eigenvectors of `kernel_matrix`.
    eigenvalues : np.array
        Eigenvalues of `kernel_matrix`.

    """
    def __init__(self, points: np.array, epsilon: float,
                 cut_off: Optional[float] = None,
                 num_eigenpairs: Optional[int] = default.num_eigenpairs,
                 normalize_kernel: Optional[bool] = True,
                 renormalization: Optional[float] = default.renormalization,
                 kdtree_options: Optional[Dict] = None,
                 use_cuda: Optional[bool] = default.use_cuda) \
            -> None:
        """Compute diffusion maps.

        This function computes the eigendecomposition of the transition
        matrix associated to a random walk on the data using a bandwidth
        (time) equal to epsilon.

        Parameters
        ----------
        points : np.array
            Data set to analyze. Its 0-th axis must index each data point.
        epsilon : float
            Bandwidth to use for the kernel.
        cut_off : float, optional
            Cut-off for the distance matrix computation. It should be at
            least equal to `epsilon`.
        num_eigenpairs : int, optional
            Number of eigenpairs to compute. Default is
            `default.num_eigenpairs`.
        normalize_kernel : bool, optional
            Whether to convert the kernel into a stochastic matrix or
            not. Default is `True`.
        renormalization : float, optional
            Renormalization exponent to use if `normalize_kernel` is
            True. This is the parameter $\alpha$ in the diffusion maps
            literature. It must take a value between zero and one.
        kdtree_options : dict, optional
            A dictionary containing parameters to pass to the underlying
            cKDTree object.
        use_cuda : bool, optional
            Determine whether to use CUDA-enabled eigenvalue solver or not.

        """
        self.points = points
        self.epsilon = epsilon

        self._cut_off = cut_off if cut_off is not None else np.inf

        self._kdtree = self.compute_kdtree(points, kdtree_options)

        distance_matrix = utils.coo_tocsr(self.compute_distance_matrix())
        kernel_matrix = self.compute_kernel_matrix(distance_matrix)
        if normalize_kernel is True:
            kernel_matrix = self.normalize_kernel_matrix(kernel_matrix,
                                                         renormalization)
        self.kernel_matrix = kernel_matrix
        self.renormalization = renormalization if normalize_kernel else None

        ew, ev = self.solve_eigenproblem(self.kernel_matrix, num_eigenpairs,
                                         use_cuda)
        if np.linalg.norm(ew.imag > 1e2 * sys.float_info.epsilon, np.inf):
            raise ValueError('Eigenvalues have non-negligible imaginary part')
        self.eigenvalues = ew.real
        self.eigenvectors = ev.real

    @staticmethod
    @clock.log
    def compute_kdtree(points: np.array, kdtree_options: Optional[Dict]) \
            -> None:
        """Compute kd-tree from points.

        """
        if kdtree_options is None:
            kdtree_options = dict()

        return scipy.spatial.cKDTree(points, **kdtree_options)

    @clock.log
    def compute_distance_matrix(self) -> scipy.sparse.coo_matrix:
        """Compute sparse distance matrix in COO format.

        """
        distance_matrix \
            = self._kdtree.sparse_distance_matrix(self._kdtree,
                                                  self._cut_off,
                                                  output_type='coo_matrix')

        logging.debug('Distance matrix has {} nonzero entries ({:.4f}% dense).'
                      .format(distance_matrix.nnz, distance_matrix.nnz
                              / np.prod(distance_matrix.shape) * 100))

        return distance_matrix

    @clock.log
    def compute_kernel_matrix(self, distance_matrix: scipy.sparse.spmatrix) \
            -> scipy.sparse.spmatrix:
        """Compute kernel matrix.

        Returns the (unnormalized) Gaussian kernel matrix corresponding to
        the data set and choice of bandwidth `epsilon`.

        Parameters
        ----------
        distance_matrix : scipy.sparse.spmatrix
            A sparse matrix whose entries are the distances between data
            points.

        Returns
        -------
        kernel_matrix : scipy.sparse.spmatrix
            A similarity matrix (unnormalized kernel matrix) obtained by
            applying `kernel_function` to the entries in `distance_matrix`.

        """
        data = distance_matrix.data
        transformed_data = self.kernel_function(data)
        kernel_matrix = distance_matrix._with_data(transformed_data, copy=True)
        return kernel_matrix

    @clock.log
    def normalize_kernel_matrix(self, matrix: scipy.sparse.csr_matrix,
                                alpha: Optional[float] = 1) \
            -> scipy.sparse.csr_matrix:
        """Compute normalized random walk Laplacian from similarity matrix.

        Parameters
        ----------
        matrix : scipy.sparse.csr_matrix
            A similarity matrix obtained by evaluating a kernel function on a
            distance matrix.
        alpha : float, optional
            Renormalization parameter. The value of `alpha` must lie in the
            closed unit interval.

        Returns
        -------
        matrix : scipy.sparse.csr_matrix
            A (suitably normalized) row-stochastic random walk Laplacian.

        """
        assert 0 <= alpha <= 1, 'Invalid normalization exponent.'

        if alpha > 0:
            shape = matrix.shape

            row_sums = np.asarray(matrix.sum(axis=1)).squeeze()

            inv_diag = 1.0 / row_sums**alpha
            inv_diag[np.isnan(inv_diag)] = 0.0
            Dinv = scipy.sparse.spdiags(inv_diag, 0, shape[0], shape[1])

            Wtilde = Dinv @ matrix @ Dinv

            return self.make_stochastic_matrix(Wtilde)
        else:
            return self.make_stochastic_matrix(matrix)

    @staticmethod
    @clock.log
    def make_stochastic_matrix(matrix: scipy.sparse.csr_matrix) \
            -> scipy.sparse.csr_matrix:
        """Convert a sparse non-negative matrix into a row-stochastic matrix.

        Carries out the normalization (in the 1-norm) of each row of a
        non-negative matrix inplace.  The matrix should be in Compressed
        Sparse Row format. Note that this method overwrites the input matrix.

        Parameters
        ----------
        matrix : scipy.sparse.csr_matrix
            A matrix with non-negative entries to be converted.

        Returns
        -------
        matrix : scipy.sparse.csr_matrix
            The same matrix passed as input but normalized.

        """
        data = matrix.data
        indptr = matrix.indptr
        for i in range(matrix.shape[0]):
            a, b = indptr[i:i+2]
            norm1 = np.sum(data[a:b])
            data[a:b] /= norm1

        return matrix

    @clock.log
    def kernel_function(self, distances: np.array) -> np.array:
        """Evaluate kernel function.

        """
        return np.exp(-np.square(distances) / (2.0 * self.epsilon))


class DenseDiffusionMaps(BaseDiffusionMaps):
    def __init__(self, points: np.array, epsilon: float,
                 cut_off: Optional[float] = None,
                 num_eigenpairs: Optional[int] = default.num_eigenpairs,
                 normalize_kernel: Optional[bool] = True,
                 renormalization: Optional[float] = default.renormalization,
                 kdtree_options: Optional[Dict] = None,
                 use_cuda: Optional[bool] = default.use_cuda) -> None:
        self.points = points
        self.epsilon = epsilon

        if cut_off is not None:
            import warnings
            warnings.warn('A cut off was specified for dense diffusion maps.')

        distance_matrix_squared = scipy.spatial.distance.pdist(points, metric='sqeuclidean')  # noqa
        kernel_matrix = np.exp(-distance_matrix_squared / (2.0 * epsilon))
        kernel_matrix = scipy.spatial.distance.squareform(kernel_matrix)
        if normalize_kernel is True:
            kernel_matrix = self.normalize_kernel_matrix(kernel_matrix,
                                                         renormalization)
        self.kernel_matrix = kernel_matrix
        self.renormalization = renormalization if normalize_kernel else None

        if use_cuda is True:
            import warnings
            warnings.warn('Dense diffusion maps are not implemented on the '
                          'GPU. Using the CPU instead.')

        ew, ev = self.solve_eigenproblem(self.kernel_matrix, num_eigenpairs,
                                         use_cuda=False)
        if np.linalg.norm(ew.imag > 1e2 * sys.float_info.epsilon, np.inf):
            raise ValueError('Eigenvalues have non-negligible imaginary part')
        self.eigenvalues = ew.real
        self.eigenvectors = ev.real

    def normalize_kernel_matrix(self, matrix: np.array,
                                alpha: Optional[float] = 1) \
            -> np.array:
        assert 0 <= alpha <= 1, 'Invalid normalization exponent.'

        if alpha > 0:
            row_sums = np.asarray(matrix.sum(axis=1)).squeeze()

            inv_diag = 1.0 / row_sums**alpha
            inv_diag[np.isnan(inv_diag)] = 0.0

            Wtilde = np.multiply(inv_diag[:, np.newaxis], matrix)
            Wtilde = np.multiply(Wtilde, inv_diag)

            return self.make_stochastic_matrix(Wtilde)
        else:
            return self.make_stochastic_matrix(matrix)

    @staticmethod
    def make_stochastic_matrix(matrix: np.array) -> np.array:
        assert matrix.shape[0] == matrix.shape[1], 'Matrix must be square'
        Dinv = np.diag(1.0 / np.sum(matrix, axis=1))
        return Dinv @ matrix
