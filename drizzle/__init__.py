"""
A package for combining dithered images into a single image
"""
from . import drizzle
from . import dodrizzle
from . import doblot
from . import calc_pixmap
from . import util
from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    __version__ = 'unknown'
