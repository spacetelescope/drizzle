import os.path

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
from astropy import wcs
from astropy.io import fits

from drizzle import calc_pixmap


TEST_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(TEST_DIR, 'data')


def test_map_rectangular():
    """
    Make sure the initial index array has correct values
    """
    naxis1 = 1000
    naxis2 = 10

    pixmap = np.indices((naxis1, naxis2), dtype='float32')
    pixmap = pixmap.transpose()

    assert_equal(pixmap[5,500], (500,5))


def test_map_to_self():
    """
    Map a pixel array to itself. Should return the same array.
    """
    input_file = os.path.join(DATA_DIR, 'input1.fits')
    input_hdu = fits.open(input_file)

    input_wcs = wcs.WCS(input_hdu[1].header)
    naxis1, naxis2 = input_wcs.pixel_shape
    input_hdu.close()

    ok_pixmap = np.indices((naxis1, naxis2), dtype='float32')
    ok_pixmap = ok_pixmap.transpose()

    pixmap = calc_pixmap.calc_pixmap(input_wcs, input_wcs)

    # Got x-y transpose right
    assert_equal(pixmap.shape, ok_pixmap.shape)
    # Mapping an array to itself
    assert_almost_equal(pixmap, ok_pixmap, decimal=5)


def test_translated_map():
    """
    Map a pixel array to  at translated array.
    """
    first_file = os.path.join(DATA_DIR, 'input1.fits')
    first_hdu = fits.open(first_file)
    first_header = first_hdu[1].header

    first_wcs = wcs.WCS(first_header)
    naxis1, naxis2 = first_wcs.pixel_shape
    first_hdu.close()

    second_file = os.path.join(DATA_DIR, 'input3.fits')
    second_hdu = fits.open(second_file)
    second_header = second_hdu[1].header

    second_wcs = wcs.WCS(second_header)
    second_hdu.close()

    ok_pixmap = np.indices((naxis1, naxis2), dtype='float32') - 2.0
    ok_pixmap = ok_pixmap.transpose()

    pixmap = calc_pixmap.calc_pixmap(first_wcs, second_wcs)

    # Got x-y transpose right
    assert_equal(pixmap.shape, ok_pixmap.shape)
    # Mapping an array to a translated array
    assert_almost_equal(pixmap, ok_pixmap, decimal=5)
