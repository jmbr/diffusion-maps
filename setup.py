#!/usr/bin/env python

import importlib
import os

from setuptools import setup, find_packages

def read_version():
    """This reads the version from diffusion_maps/version.py without importing parts of
    the actual package (which would require some of the dependencies already
    installed)."""
    # code parts were taken from here https://stackoverflow.com/a/67692

    path2setup = os.path.dirname(__file__)
    version_file = os.path.abspath(os.path.join(path2setup, "diffusion_maps",
                                                "version.py"))

    spec = importlib.util.spec_from_file_location("version", version_file)
    version = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(version)
    return version.version.v_short

setup(name='diffusion-maps',
      version=read_version(),
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
