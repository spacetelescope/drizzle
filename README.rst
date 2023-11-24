drizzle Documentation
=====================

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

.. image:: https://codecov.io/github/spacetelescope/drizzle/branch/master/graphs/badge.svg
    :target: https://codecov.io/gh/spacetelescope/drizzle
    :alt: Drizzle's Coverage Status

.. image:: https://github.com/spacetelescope/drizzle/workflows/CI/badge.svg
    :target: https://github.com/spacetelescope/drizzle/actions
    :alt: CI Status

.. image:: https://readthedocs.org/projects/spacetelescope-drizzle/badge/?version=latest
    :target: https://spacetelescope-drizzle.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/drizzle.svg
    :target: https://pypi.org/project/drizzle
    :alt: PyPI Status

The ``drizzle`` library is a Python package for combining dithered images into a
single image. This library is derived from code used in DrizzlePac. Like
DrizzlePac, most of the code is implemented in the C language. The biggest
change from DrizzlePac is that this code passes an array that maps the input to
output image into the C code, while the DrizzlePac code computes the mapping by
using a Python callback. Switching to using an array allowed the code to be
greatly simplified.

The DrizzlePac code is currently used in the Space Telescope processing
pipelines. This library is forward looking in that it can be used with
the new GWCS code.

Requirements
------------

- Python 3.9 or later

- Numpy

- Astropy

.. include:: docs/drizzle/algorithms.rst

.. include:: docs/drizzle/user.rst

