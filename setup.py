#!/usr/bin/env python

from setuptools import setup, find_packages

import diffusion_maps


setup(name='diffusion-maps',
      version=diffusion_maps.version.v_short,
      description='Diffusion maps',
      long_description='Library for computing diffusion maps',
      license='MIT License',
      author='Juan M. Bello-Rivas',
      author_email='jmbr@superadditive.com',
      packages=find_packages(),
      package_dir={'diffusion_maps': 'diffusion_maps'},
      package_data={'': ['LICENSE']},
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=['scipy', 'numpy'],
      extras_require={
          'plotting': ['matplotlib'],
          'cuda': ['numba']
      },
      entry_points={
          'console_scripts': 'diffusion-maps = diffusion_maps.command_line_interface:main'
      })
