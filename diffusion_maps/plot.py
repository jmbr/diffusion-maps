"""Module for plotting diffusion maps and associated diagnostics.

"""

__all__ = ['plot_results']

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


def plot_results(data: np.array, eigenvalues: np.array,
                 eigenvectors: np.array) -> None:
    """Plot results.

    Plots three figures. The first one is shows the modulus of the spectrum
    of the kernel in the diffusion map calculation.  The second displays the
    original (2D) data colored by the value of each diffusion map.  The third
    figure displays the data, as trasnformed by the first two diffusion maps.

    Parameters
    ----------
    data : np.array
        Original (or downsampled) data set.
    eigenvalues : np.array
        Eigenvalues of the kernel matrix.
    eigenvectors : np.array
        Eigenvectors of the kernel matrix. The zeroth axis indexes each
        vector.

    """
    x = data[:, 0]
    y = data[:, 1]

    num_eigenvectors = min(eigenvectors.shape[0]-1, default.num_eigenpairs-1)

    plt.figure(1)
    plt.step(range(eigenvalues.shape[0]), np.abs(eigenvalues))
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Modulus (norm) of eigenvalue')
    plt.title('Eigenvalues')

    plt.figure(2)
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

    plt.figure(3)
    plt.scatter(eigenvectors[1, :], eigenvectors[2, :],
                color='black', alpha=0.5)
    plt.xlabel('$\\psi_1$')
    plt.ylabel('$\\psi_2$')
    plt.title('Data set in diffusion map space')

    # plt.tight_layout()
    plt.show()
