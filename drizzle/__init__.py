"""
A package for combining dithered images into a single image
"""
from importlib.metadata import PackageNotFoundError, version

from drizzle import cdrizzle, resample, utils  # noqa: F401

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = ''
