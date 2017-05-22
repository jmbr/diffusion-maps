"""Diffusion maps module.

This module implements the diffusion maps method for dimensionality
reduction, as introduced in:

Coifman, R. R., & Lafon, S. (2006). Diffusion maps. Applied and Computational
Harmonic Analysis, 21(1), 5–30. DOI:10.1016/j.acha.2006.04.006

"""

__all__ = ['DiffusionMaps', 'downsample']


import logging
from typing import Optional, Dict, Tuple

import numpy as np
import scipy
import scipy.sparse
from scipy.spatial import cKDTree

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


class DiffusionMaps:
    """Diffusion maps.

    Attributes
    ----------
    epsilon : float
        Bandwidth for kernel.
    _cut_off : float
        Cut off for the computation of pairwise distances between points.
    _kdtree : cKDTree
        KD-tree for accelerating pairwise distance computation.
    eigenvectors : np.array
        Right eigenvectors of `K`.
    eigenvalues : np.array
        Eigenvalues of `K`.
    kernel_matrix : scipy.sparse.spmatrix
        (Possibly stochastic) matrix obtained by evaluating a Gaussian kernel
        on the data points.

    """
    def __init__(self, points: np.array, epsilon: float,
                 cut_off: Optional[float] = None,
                 num_eigenpairs: Optional[int] = default.num_eigenpairs,
                 normalize_kernel: Optional[bool] = True,
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
        kdtree_options : dict, optional
            A dictionary containing parameters to pass to the underlying
            cKDTree object.
        use_cuda : bool, optional
            Determine whether to use CUDA-enabled eigenvalue solver or not.

        """
        self.epsilon = epsilon

        if cut_off is None:
            self._cut_off = self.__get_cut_off(self.epsilon)
        else:
            self._cut_off = cut_off

        self.compute_kdtree(points, kdtree_options)

        distance_matrix = self.compute_distance_matrix()
        distance_matrix = utils.coo_tocsr(distance_matrix)

        self.kernel_matrix = self.compute_kernel_matrix(distance_matrix)

        if normalize_kernel is True:
            self.normalize_kernel_matrix(self.kernel_matrix)

        ew, ev = self.solve_eigenproblem(num_eigenpairs, use_cuda)

        self.eigenvalues = ew
        self.eigenvectors = ev

    @staticmethod
    def __get_cut_off(epsilon: float) -> float:
        """Return a reasonable cut off value.

        """
        return 2.0 * epsilon  # XXX Validate this.

    @clock.log
    def compute_kdtree(self, points: np.array,
                       kdtree_options: Optional[Dict]) -> None:
        """Compute kd-tree from points.

        """
        if kdtree_options is None:
            kdtree_options = dict()
        self._kdtree = cKDTree(points, **kdtree_options)

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
                              / np.prod(distance_matrix.shape)))

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

        See also
        --------
        _compute_distance_matrix, make_stochastic_matrix

        """
        data = distance_matrix.data
        transformed_data = self.kernel_function(data)
        kernel_matrix = distance_matrix._with_data(transformed_data, copy=True)
        return kernel_matrix

    @staticmethod
    @clock.log
    def normalize_kernel_matrix(matrix: scipy.sparse.csr_matrix) -> None:
        """Normalize a sparse, non-negative matrix in CSR format.

        Normalizes (in the 1-norm) each row of a non-negative matrix inplace.

        Parameters
        ----------
        matrix : np.array
            A matrix with non-negative entries to be normalized.

        """
        data = matrix.data
        indptr = matrix.indptr
        for i in range(matrix.shape[0]):
            a, b = indptr[i:i+2]
            norm1 = np.sum(data[a:b])
            data[a:b] /= norm1

    @clock.log
    def kernel_function(self, distances: np.array) -> np.array:
        """Evaluate kernel function.

        """
        return np.exp(-np.square(distances) / (2.0 * self.epsilon))

    @clock.log
    def solve_eigenproblem(self, num_eigenpairs: int, use_cuda: bool) \
            -> Tuple[np.array, np.array]:
        """Solve eigenvalue problem using CPU or GPU solver.

        """
        if use_cuda is True:
            from .gpu_eigensolver import eigensolver
        else:
            from .cpu_eigensolver import eigensolver

        return eigensolver(self.kernel_matrix, num_eigenpairs)
