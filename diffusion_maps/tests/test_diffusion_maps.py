"""Unit test for the diffusion_maps module.

"""

import logging
import unittest

import numpy as np
# import matplotlib.pyplot as plt

from diffusion_maps import DiffusionMaps, downsample

from .misc import make_strip


class DiffusionMapsTest(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        self.xmin = 0.0
        self.ymin = 0.0
        self.width = 1.0
        self.height = 1e-1
        self.num_samples = 50000
        self.data = make_strip(self.xmin, self.ymin,
                               self.width, self.height,
                               self.num_samples)

    @staticmethod
    def _compute_rayleigh_quotients(matrix, eigenvectors):
        """Compute Rayleigh quotients."""
        N = eigenvectors.shape[0]
        rayleigh_quotients = np.zeros(N)
        for n in range(N):
            v = eigenvectors[n, :]
            rayleigh_quotients[n] = np.dot(v, matrix @ v) / np.dot(v, v)
        rayleigh_quotients = np.sort(np.abs(rayleigh_quotients))
        return rayleigh_quotients[::-1]

    def test_accuracy(self):
        num_samples = 10000
        logging.debug('Computing diffusion maps on a matrix of size {}'
                      .format(num_samples))
        num_eigenpairs = 10
        epsilon = 5e-1
        downsampled_data = downsample(self.data, num_samples)

        dm = DiffusionMaps(downsampled_data, epsilon,
                           num_eigenpairs=num_eigenpairs)

        ew = dm.eigenvalues
        rq = self._compute_rayleigh_quotients(dm.kernel_matrix,
                                              dm.eigenvectors)

        logging.debug('Eigenvalues: {}'.format(ew))
        logging.debug('Rayleigh quotients: {}'.format(rq))

        self.assertTrue(np.allclose(np.abs(ew), np.abs(rq)))

        # import scipy.io
        # scipy.io.mmwrite('kernel_matrix.mtx', dm.kernel_matrix)

    def test_multiple_epsilon_values(self):
        num_samples = 5000
        num_maps = 10
        num_eigenpairs = 10
        epsilon_min, epsilon_max = 1e-1, 1e1
        epsilons = np.logspace(np.log10(epsilon_min),
                               np.log10(epsilon_max), num_maps)

        downsampled_data = downsample(self.data, num_samples)

        evs = np.zeros((num_maps, num_eigenpairs, downsampled_data.shape[0]))
        ews = np.zeros((num_maps, num_eigenpairs))

        logging.basicConfig(level=logging.WARNING)

        for i, epsilon in enumerate(reversed(epsilons)):
            dm = DiffusionMaps(downsampled_data, epsilon,
                               num_eigenpairs=num_eigenpairs)

            evs[i, :, :] = dm.eigenvectors
            ews[i, :] = dm.eigenvalues

            ew = dm.eigenvalues
            rq = self._compute_rayleigh_quotients(dm.kernel_matrix,
                                                  dm.eigenvectors)
            self.assertTrue(np.allclose(np.abs(ew), np.abs(rq)))

            # plt.title('$\\epsilon$ = {:.3f}'.format(epsilon))
            # for k in range(1, 10):
            #     plt.subplot(2, 5, k)
            #     plt.scatter(downsampled_data[:, 0], downsampled_data[:, 1],
            #                 c=evs[i, k, :])
            #     plt.xlim([self.xmin, self.xmin + self.width])
            #     plt.ylim([self.ymin, self.ymin + self.height])
            #     plt.tight_layout()
            #     plt.gca().set_title('$\\psi_{}$'.format(k))
            # plt.subplot(2, 5, 10)
            # plt.step(range(ews[i, :].shape[0]), np.abs(ews[i, :]))
            # plt.title('epsilon = {:.2f}'.format(epsilon))
            # plt.show()


if __name__ == '__main__':
    import os
    verbose = os.getenv('VERBOSE')
    if verbose is not None:
        logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    else:
        logging.basicConfig(level=logging.ERROR, format='%(message)s')
    unittest.main()
