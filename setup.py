#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup tools file for distribution-analysis package

To install this package run the following command in the source directory
'''
pip install .
'''
"""
__version__ = 0.1
__author__ = "Christopher Körber"

import os
import setuptools

# Get long description
with open(os.path.join(os.getcwd(), "README.md"), "r", encoding='utf-8') as f:
  LONG_DESCRIPTION = f.read()

# Specify install
setuptools.setup(
  # See: https://packaging.python.org/specifications/core-metadata/
  name='distribution-analysis',
  version=str(__version__),
  description="Python package for visulaizing statistical distributions",
  long_description=LONG_DESCRIPTION,
  long_description_content_type='text/markdown',
  url='https://github.com/ckoerber/distribution-analysis',
  author=__author__,
  install_requires=[
    "numpy",
    "scipy",
    "pandas",
    "statsmodels",
    "matplotlib",
    "seaborn",
    "gvar",
  ],
  # See https://pypi.org/classifiers/
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
  keywords='Distribution Analysis Visualizations',
  packages=setuptools.find_packages(),
  test_suite='tests',
)
