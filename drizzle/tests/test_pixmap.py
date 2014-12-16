#!/usr/bin/env python

import sys
import glob
import nose
import os.path
import numpy as np
import numpy.testing as npt

from astropy.io import fits
import stwcs

TEST_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(TEST_DIR, 'data')
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))

sys.path.append(TEST_DIR)
sys.path.append(PROJECT_DIR)

from drizzlepac.calc_pixmap import calc_pixmap

class TestPixmap(object):

    def __init__(self):
        """
        Initialize test environment
        """
        self.setup()

    def setup(self):
        """
        Remove fits test files
        """
        pattern = os.path.join(DATA_DIR, "output*.fits")
        for filename in glob.glob(pattern):
            if filename: os.remove(filename)

    def test_map_rectangular(self):
        """
        Make sure the initial index array has correct values
        """
        naxis1 = 1000
        naxis2 = 10
        
        pixmap = np.indices((naxis1, naxis2), dtype='float32')
        pixmap = pixmap.transpose()
        npt.assert_equal(pixmap[5,500], (500,5))

    def test_map_to_self(self):
        """
        Map a pixel array to itself. Should return the same array.
        """
        input_file = os.path.join(DATA_DIR, 'input1.fits')
        input_hdu = fits.open(input_file)
        input_header = input_hdu[1].header
        input_shape = input_hdu[1].data.shape
    
        input_wcs = stwcs.wcsutil.HSTWCS(input_hdu, 1)
        naxis1 = input_wcs._naxis1
        naxis2 = input_wcs._naxis2
        input_hdu.close()

        ok_pixmap = np.indices((naxis1, naxis2), dtype='float32')
        ok_pixmap = ok_pixmap.transpose()

        pixmap = calc_pixmap(input_wcs, input_wcs)
        npt.assert_equal(pixmap.shape, ok_pixmap.shape) # Got x-y transpose right
        npt.assert_almost_equal(pixmap, ok_pixmap, decimal=5) # Mapping an array to itself

    def test_translated_map(self):
        """
        Map a pixel array to  at translated array.
        """
        first_file = os.path.join(DATA_DIR, 'input1.fits')
        first_hdu = fits.open(first_file)
        first_header = first_hdu[1].header
        
        first_wcs = stwcs.wcsutil.HSTWCS(first_hdu, 1)
        naxis1 = first_wcs._naxis1
        naxis2 = first_wcs._naxis2
        first_hdu.close()

        second_file = os.path.join(DATA_DIR, 'input3.fits')
        second_hdu = fits.open(second_file)
        second_header = second_hdu[1].header
        
        second_wcs = stwcs.wcsutil.HSTWCS(second_hdu, 1)
        second_hdu.close()

        ok_pixmap = np.indices((naxis1, naxis2), dtype='float32') - 2.0
        ok_pixmap = ok_pixmap.transpose()

        pixmap = calc_pixmap(first_wcs, second_wcs)
        npt.assert_equal(pixmap.shape, ok_pixmap.shape) # Got x-y transpose right
        npt.assert_almost_equal(pixmap, ok_pixmap, decimal=5) # Mapping an array to a translated array

if __name__ == "__main__":
    go = TestPixmap()
    go.test_map_rectangular()
    go.test_map_to_self()
    go.test_translated_map()
    ##nose.run()
