"""
A package for combining dithered images into a single image
"""
from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = 'unknown'
