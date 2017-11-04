#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK                                -*- mode: python -*-

"""Command line interface for the computation of diffusion maps.

"""

import argparse
try:
    import argcomplete
except ImportError:
    argcomplete = None
import logging
import os
import sys
import warnings

import numpy as np

import diffusion_maps.default as default
import diffusion_maps.version as version
from diffusion_maps import downsample, SparseDiffusionMaps, DenseDiffusionMaps
from diffusion_maps.profiler import Profiler
from diffusion_maps.plot import plot_diffusion_maps


def output_eigenvalues(ew: np.array) -> None:
    """Output the table of eigenvalues.

    """
    logging.info('Index    Eigenvalue')
    fmt = '{:5d}   {:2.9f}'
    for i, eigenvalue in enumerate(ew):
        logging.info(fmt.format(i, eigenvalue))


def use_cuda(args: argparse.Namespace) -> bool:
    """Determine whether to use GPU-accelerated code or not.

    """
    try:
        import pycuda           # noqa
        use_cuda = True and not args.no_gpu
    except ImportError:
        use_cuda = False

    return use_cuda


def main():
    parser = argparse.ArgumentParser(description='computation of diffusion '
                                     'maps')
    parser.add_argument('data_file', metavar='FILE', type=str,
                        help='process %(metavar)s (admits NumPy, MATLAB, and '
                        'CSV formats)')
    parser.add_argument('epsilon', metavar='VALUE', type=float,
                        help='kernel bandwidth')
    parser.add_argument('-b', '--bounds', type=str, required=False,
                        metavar='FILE', help='Vector of upper bounds for '
                        'periodic boxes')
    parser.add_argument('-n', '--num-samples', type=float, metavar='NUM',
                        required=False, help='number of data points to use')
    parser.add_argument('-e', '--num-eigenpairs', type=int, metavar='NUM',
                        required=False, default=default.num_eigenpairs,
                        help='number of eigenvalue/eigenvector pairs to '
                        'compute')
    parser.add_argument('-c', '--cut-off', type=float, required=False,
                        metavar='DISTANCE', help='cut-off to use in order to '
                        'enforce sparsity')
    parser.add_argument('-r', '--renormalization', type=float, required=False,
                        default=default.renormalization, metavar='ALPHA',
                        help='renormalization parameter for kernel matrix')
    parser.add_argument('--no-gpu', action='store_true', required=False,
                        help='explicitly disable GPU eigensolver')
    parser.add_argument('--dense', action='store_true', required=False,
                        help='Use dense instead of sparse linear algebra '
                        'routines')
    parser.add_argument('-o', '--output-data', type=str, required=False,
                        default='actual-data.npy', metavar='FILE', help='save '
                        'actual data used in computation to %(metavar)s')
    parser.add_argument('-w', '--eigenvalues', type=str,
                        default='eigenvalues.dat', required=False,
                        metavar='FILE', help='save eigenvalues to '
                        '%(metavar)s')
    parser.add_argument('-v', '--eigenvectors', type=str,
                        default='eigenvectors.npy', required=False,
                        metavar='FILE', help='save eigenvectors to '
                        '%(metavar)s')
    parser.add_argument('-m', '--matrix', type=str, required=False,
                        metavar='FILE', help='save transition matrix to '
                        '%(metavar)s')
    parser.add_argument('-p', '--plot', action='store_true', default=False,
                        help='plot first two eigenvectors')
    parser.add_argument('--debug', action='store_true', required=False,
                        help='print debugging information')
    parser.add_argument('--profile', required=False, metavar='FILE',
                        type=argparse.FileType('w', encoding='utf-8'),
                        help='run under profiler and save report to '
                        '%(metavar)s')

    args = parser.parse_args(sys.argv[1:])

    if args.debug is True:
        logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    if args.cut_off is not None and args.cut_off < args.epsilon:
        logging.error('Error: cut off ({}) is smaller than bandwidth ({}).'
                      .format(args.cut_off, args.epsilon))
        sys.exit(-1)

    prog_name = os.path.basename(sys.argv[0])
    logging.info('{} {}'.format(prog_name, version.v_long))
    logging.info('')
    logging.info('Reading data from {!r}...'.format(args.data_file))

    try:
        import pathlib
        ext = pathlib.Path(args.data_file).suffix
        if ext.endswith('dat') or ext.endswith('csv'):
            orig_data = np.loadtxt(args.data_file)
        elif ext.endswith('mat'):
            import scipy.io
            mdict = scipy.io.matlab.loadmat(args.data_file)
            orig_data = mdict.popitem()[-1]
        else:
            orig_data = np.load(args.data_file)
    except FileNotFoundError as exc:
        logging.error('Error: {}'.format(exc))
        sys.exit(-1)

    if args.num_samples is not None:
        num_samples = int(args.num_samples)
        if num_samples <= orig_data.shape[0]:
            data = downsample(orig_data, num_samples)
        else:
            logging.warning('Data set contains {} points but {} points '
                            'were requested'.format(orig_data.shape[0],
                                                    num_samples))
            sys.exit(-1)
    else:
        data = orig_data

    logging.info('Computing {} diffusion maps with epsilon = {:g} '
                 'on {} data points...'
                 .format(args.num_eigenpairs-1, args.epsilon, data.shape[0]))

    if args.bounds is not None:
        try:
            bounds = np.load(args.bounds)
        except FileNotFoundError as exc:
            logging.error('Unable to find file {!r}.'.format(args.bounds))
            sys.exit(-1)
        except OSError:
            bounds = np.loadtxt(args.bounds)
        kdtree_options = {'boxsize': np.concatenate((bounds, bounds))}
    else:
        kdtree_options = {}

    if args.dense is True:
        # Sometimes sparse linear algebra may induce large memory overheads,
        # so we may want to try another approach.
        diff_maps = DenseDiffusionMaps
    else:
        diff_maps = SparseDiffusionMaps

    with Profiler(args.profile):
        dm = diff_maps(data, args.epsilon, cut_off=args.cut_off,
                       num_eigenpairs=args.num_eigenpairs,
                       normalize_kernel=True,
                       renormalization=args.renormalization,
                       kdtree_options=kdtree_options,
                       use_cuda=use_cuda(args))

    if args.profile:
        args.profile.close()

    if hasattr(np, 'ComplexWarning'):
        warnings.simplefilter('ignore', np.ComplexWarning)

    output_eigenvalues(dm.eigenvalues)

    threshold = 1e1 * sys.float_info.epsilon
    if np.linalg.norm(dm.eigenvectors.imag, np.inf) > threshold:
        logging.warning('Eigenvectors have a non-negligible imaginary part. '
                        'This may be fixed by increasing the value of '
                        '--cut-off.')

    if args.matrix:
        logging.info('Saving transition matrix to {!r}'
                     .format(args.matrix))
        import scipy.io
        scipy.io.mmwrite(args.matrix, dm.kernel_matrix)

    if args.eigenvalues:
        logging.info('Saving eigenvalues to {!r}'
                     .format(args.eigenvalues))
        np.savetxt(args.eigenvalues, dm.eigenvalues)

    if args.eigenvectors:
        logging.info('Saving eigenvectors to {!r}'
                     .format(args.eigenvectors))
        np.save(args.eigenvectors, dm.eigenvectors)

    if args.output_data and args.num_samples:
        logging.info('Saving downsampled data to {!r}'
                     .format(args.output_data))
        np.save(args.output_data, data)

    if args.plot is True:
        plot_diffusion_maps(data, dm)


if __name__ == '__main__':
    main()
