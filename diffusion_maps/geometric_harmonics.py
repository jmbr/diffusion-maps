"""Geometric harmonics module.

This module implements out-of-sample evaluation of functions using the
Geometric Harmonics method introduced in:

Coifman, R. R., & Lafon, S. (2006). Geometric harmonics: A novel tool for
multiscale out-of-sample extension of empirical functions. Applied and
Computational Harmonic Analysis, 21(1), 31–52. DOI:10.1016/j.acha.2005.07.005

"""

__all__ = ['GeometricHarmonicsInterpolator']

from typing import Optional, Dict

import numpy as np
import scipy.spatial
from scipy.interpolate.interpnd import (NDInterpolatorBase,
                                        _ndim_coords_from_arrays)

from diffusion_maps.diffusion_maps import DiffusionMaps


class GeometricHarmonicsInterpolator(NDInterpolatorBase):
    """Geometric Harmonics interpolator.

    """
    def __init__(self, points: np.array, values: np.array, epsilon: float,
                 diffusion_maps_options: Optional[Dict] = None) -> None:
        NDInterpolatorBase.__init__(self, points, values,
                                    need_contiguous=False, need_values=True)
        self.epsilon = epsilon
        if diffusion_maps_options is None:
            diffusion_maps_options = dict()
        diffusion_maps_options['normalize_kernel'] = False
        self.diffusion_maps = DiffusionMaps(self.points, epsilon,
                                            **diffusion_maps_options)

    def __call__(self, *args):
        """Evaluate interpolator at the given points.

        """
        xi = _ndim_coords_from_arrays(args, ndim=self.points.shape[1])
        xi = self._check_call_shape(xi)

        kdtree = scipy.spatial.cKDTree(xi)
        dmaps_kdtree = self.diffusion_maps._kdtree
        radius = self.diffusion_maps._cut_off

        ew = self.diffusion_maps.eigenvalues
        ev = self.diffusion_maps.eigenvectors
        aux = ev.T @ np.diag(1.0 / ew) @ ev @ self.values

        distance_matrix \
            = kdtree.sparse_distance_matrix(dmaps_kdtree, radius,
                                            output_type='coo_matrix')
        kernel_matrix \
            = self.diffusion_maps._compute_kernel_matrix(distance_matrix)

        return np.squeeze(kernel_matrix @ aux)
