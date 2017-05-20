"""Unit test for the Geometric Harmonics module.

"""

import logging
import unittest

import numpy as np

import matplotlib.pyplot as plt

from diffusion_maps import GeometricHarmonicsInterpolator

from diffusion_maps.clock import Clock


def make_points(num_points: int) -> np.array:
    xx, yy = np.meshgrid(np.linspace(-4, 4, num_points),
                         np.linspace(-4, 4, num_points))
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

        self.points = make_points(23)
        self.num_points = self.points.shape[0]
        self.values = f(self.points)

    def test_geometric_harmonics_interpolator(self):
        logging.basicConfig(level=logging.DEBUG)

        eps = 1e-1
        dmaps_opts = {'num_eigenpairs': self.num_points-3,
                      'cut_off': 1e1 * eps}
        ghi = GeometricHarmonicsInterpolator(self.points, self.values,
                                             eps, dmaps_opts)

        points = make_points(100)

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


if __name__ == '__main__':
    import os
    verbose = os.getenv('VERBOSE')
    if verbose is not None:
        logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    else:
        logging.basicConfig(level=logging.ERROR, format='%(message)s')
    unittest.main()
