"""Miscellaneous functions used by the unit tests.

"""


from typing import Optional

import numpy as np


def make_strip(xmin: float, ymin: float, width: float,
               height: float, num_samples: int) -> np.array:
    """Draw samples from a 2D strip with uniform distribution.

    """
    x = width * np.random.rand(num_samples) - xmin
    y = height * np.random.rand(num_samples) - ymin

    return np.stack((x, y), axis=-1)


def make_points(num_points: int, x0: float, y0: float, x1: float, y1: float) \
        -> np.array:
    xx, yy = np.meshgrid(np.linspace(x0, x1, num_points),
                         np.linspace(y0, y1, num_points))
    return np.stack((xx.ravel(), yy.ravel())).T


def swiss_roll(nt: int, ns: int, freq: Optional[float] = 2.0) -> np.array:
    """Draw samples from the swiss roll manifold.

    """
    tt = np.linspace(0.0, 2.0 * np.pi, nt)
    ss = np.linspace(-0.5, 0.5, ns)
    t, s = np.meshgrid(tt, ss)

    x = t * np.cos(freq * t)
    y = t * np.sin(freq * t)
    z = s

    return np.stack((x.ravel(), y.ravel(), z.ravel())).T
