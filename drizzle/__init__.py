# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
A package for combining dithered images into a single image
"""

from __future__ import absolute_import, division, unicode_literals, print_function

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    __version__ = 'unknown'
