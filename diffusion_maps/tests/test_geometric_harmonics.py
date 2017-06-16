"""Unit test for the Geometric Harmonics module.

"""

import logging
import unittest

import numpy as np

import matplotlib.pyplot as plt

from diffusion_maps import (GeometricHarmonicsInterpolator,
                            DiffusionMaps)

from diffusion_maps.clock import Clock


def make_points(num_points: int, x0: float, y0: float, x1: float, y1: float) \
        -> np.array:
    xx, yy = np.meshgrid(np.linspace(x0, x1, num_points),
                         np.linspace(y0, y1, num_points))
    return np.stack((xx.ravel(), yy.ravel())).T


def plot(points: np.array, values: np.array, **kwargs) -> None:
    title = kwargs.pop('title', None)
    if title:
        plt.title(title)
    plt.scatter(points[:, 0], points[:, 1], c=values,
                marker='o', rasterized=True, s=2.5, **kwargs)
    cb = plt.colorbar()
    cb.set_clim([np.min(values), np.max(values)])
    cb.set_ticks(np.linspace(np.min(values), np.max(values), 5))
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.gca().set_aspect('equal')


def f(points: np.array) -> np.array:
    """Function to interpolate.

    """
    # return np.ones(points.shape[0])
    # return np.arange(points.shape[0])
    return np.sin(np.linalg.norm(points, axis=-1))


def show_info(title: str, values: np.array) -> None:
    """Log relevant information.

    """
    logging.info('{}: mean = {:g}, std. = {:g}, max. abs. = {:g}'
                 .format(title, np.mean(values), np.std(values),
                         np.max(np.abs(values))))


class GeometricHarmonicsTest(unittest.TestCase):
    def setUp(self):
        # self.num_points = 1000
        # self.points = downsample(np.load('data.npy'), self.num_points)
        # self.values = np.ones(self.num_points)

        # np.save('actual-data.npy', self.points)

        # self.points = np.load('actual-data.npy')
        # self.num_points = self.points.shape[0]
        # self.values = np.ones(self.num_points)

        self.points = make_points(23, -4, -4, 4, 4)
        self.num_points = self.points.shape[0]
        self.values = f(self.points)

    def test_geometric_harmonics_interpolator(self):
        logging.basicConfig(level=logging.DEBUG)

        eps = 1e-1
        dmaps_opts = {'num_eigenpairs': self.num_points-3,
                      'cut_off': 1e1 * eps}
        ghi = GeometricHarmonicsInterpolator(self.points, self.values,
                                             eps, dmaps_opts)

        points = make_points(100, -4, -4, 4, 4)

        with Clock() as clock:
            values = ghi(points)
        logging.debug('Evaluation of geometric harmonics done ({} seconds).'.
                      format(clock))

        residual = values - f(points)
        self.assertLess(np.max(np.abs(residual)), 7.5e-2)

        show_info('Original function', f(points))
        show_info('Sampled points', self.values)
        show_info('Reconstructed function', values)
        show_info('Residual', residual)

        # plt.subplot(2, 2, 1)
        # plot(points, f(points), title='Original function')
        #
        # plt.subplot(2, 2, 2)
        # plot(self.points, self.values, title='Sampled function')
        #
        # plt.subplot(2, 2, 4)
        # plot(points, values, title='Reconstructed function')
        #
        # plt.subplot(2, 2, 3)
        # plot(points, residual, title='Residual', cmap='RdBu_r')
        #
        # plt.tight_layout()
        # plt.show()

    def test_eigenfunctions(self):
        logging.basicConfig(level=logging.DEBUG)

        eps = 1e1
        cut_off = 1e1 * eps
        num_eigenpairs = 3

        from .aux import make_strip
        points = make_strip(0, 0, 1, 1e-1, 3000)

        dm = DiffusionMaps(points, eps, cut_off=cut_off,
                           num_eigenpairs=num_eigenpairs)
        ev = dm.eigenvectors

        # plt.subplot(1, 2, 1)
        # plt.scatter(points[:, 0], points[:, 1], c=ev[1, :], cmap='RdBu_r')
        # plt.subplot(1, 2, 2)
        # plt.scatter(points[:, 0], points[:, 1], c=ev[2, :], cmap='RdBu_r')
        # plt.show()

        dmaps_opts = {'num_eigenpairs': num_eigenpairs, 'cut_off': cut_off}
        ev1 = GeometricHarmonicsInterpolator(points, ev[1, :], eps, dmaps_opts)
        ev2 = GeometricHarmonicsInterpolator(points, ev[2, :], eps, dmaps_opts)

        # new_points = make_points(50, 0, 0, 1, 1e-1)
        # ev1i = ev1(new_points)
        # ev2i = ev2(new_points)
        # plt.subplot(1, 2, 1)
        # plt.scatter(new_points[:, 0], new_points[:, 1], c=ev1i,
        #             cmap='RdBu_r')
        # plt.subplot(1, 2, 2)
        # plt.scatter(new_points[:, 0], new_points[:, 1], c=ev2i,
        #             cmap='RdBu_r')
        # plt.show()

        rel_err1 = (np.linalg.norm(ev[1, :] - ev1(points), np.inf) /
                    np.linalg.norm(ev[1, :], np.inf))
        self.assertAlmostEqual(rel_err1, 0, places=1)

        rel_err2 = (np.linalg.norm(ev[2, :] - ev2(points), np.inf) /
                    np.linalg.norm(ev[2, :], np.inf))
        self.assertAlmostEqual(rel_err2, 0, places=1)


if __name__ == '__main__':
    import os
    verbose = os.getenv('VERBOSE')
    if verbose is not None:
        logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    else:
        logging.basicConfig(level=logging.ERROR, format='%(message)s')
    unittest.main()
