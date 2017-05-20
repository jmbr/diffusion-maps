"""Module for plotting eigenvectors.

"""

__all__ = ['plot_eigenvectors']

from typing import Tuple

import matplotlib.pyplot as plt

import numpy as np

from . import default


def get_rows_and_columns(num_plots: int) -> Tuple[int, int]:
    """Get optimal number of rows and columns to display figures.

    Parameters
    ----------
    num_plots : int
        Number of subplots

    Returns
    -------
    rows : int
        Optimal number of rows.
    cols : int
        Optimal number of columns.

    """
    if num_plots <= 10:
        layouts = {
            1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3),
            6: (2, 3), 7: (2, 4), 8: (2, 4), 9: (3, 9), 10: (2, 5)
        }
        rows, cols = layouts[num_plots]
    else:
        rows = int(np.ceil(np.sqrt(num_plots)))
        cols = rows

    return rows, cols


def plot_eigenvectors(data: np.array, eigenvectors: np.array) -> None:
    """Plot eigenvectors onto 2D data.

    """
    x = data[:, 0]
    y = data[:, 1]

    num_eigenvectors = min(eigenvectors.shape[0]-1, default.num_eigenpairs-1)

    rows, cols = get_rows_and_columns(num_eigenvectors)

    for k in range(1, eigenvectors.shape[0]):
        plt.subplot(rows, cols, k)
        plt.scatter(x, y, c=eigenvectors[k, :], cmap='RdBu_r',
                    rasterized=True)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        cb = plt.colorbar()
        cb.set_label('Eigenvector value')
        plt.title('$\\psi_{{{}}}$'.format(k))

    # plt.tight_layout()
    plt.show()
