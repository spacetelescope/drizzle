#!/usr/bin/env python

from __future__ import division, print_function, unicode_literals, absolute_import

import sys
import glob
import math
import os.path
import numpy as np
import numpy.ma as ma
import numpy.testing as npt

from astropy import wcs
from astropy.io import fits

TEST_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(TEST_DIR, 'data')
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))

sys.path.append(TEST_DIR)
sys.path.append(PROJECT_DIR)

##from .. import cdrizzle
import drizzle
import drizzle.util
import drizzle.drizzle

class TestFileIO(object):

    def __init__(self):
        """
        Initialize test environment
        """
        args = {}
        for flag in sys.argv[1:]:
            args[flag] = 1
        
        flags = ['ok']
        for flag in flags:
            self.__dict__[flag] = args.has_key(flag)

        self.setup()

    def setup(self):
        """
        Create python arrays used in testing
        """

    def read_header(self, filename):
        """
        Read the primary header from a fits file
        """
        fileroot, extn = drizzle.util.parse_filename(os.path.join(DATA_DIR, filename))
        hdu = fits.open(fileroot)

        header = hdu[0].header
        hdu.close()
        return header
    
    def read_image(self, filename):
        """
        Read the image from a fits file
        """
        fileroot, extn = drizzle.util.parse_filename(os.path.join(DATA_DIR, filename))
        hdu = fits.open(fileroot)

        image = hdu[1].data
        hdu.close()
        return image
    
    def read_wcs(self, filename):
        """
        Read the wcs of a fits file
        """
        fileroot, extn = drizzle.util.parse_filename(os.path.join(DATA_DIR, filename))
        hdu = fits.open(fileroot)

        the_wcs = wcs.WCS(hdu[1].header)
        hdu.close()
        return the_wcs

    def test_null_run(self):
        """
        Create an empty drizzle image
        """
        output_file = os.path.join(DATA_DIR, 'output_null_run.fits')
        output_template = os.path.join(DATA_DIR, 'reference_square_point.fits')

        output_wcs = self.read_wcs(output_template)
        driz = drizzle.drizzle.Drizzle(outwcs=output_wcs, wt_scl="expsq",
                                       pixfrac=0.5, kernel="turbo",
                                       fillval="999.")
        driz.write(output_file)
        
        assert(os.path.exists(output_file))
        handle = fits.open(output_file)

        assert(handle.index_of("SCI") == 1)
        assert(handle.index_of("WHT") == 2)
        assert(handle.index_of("CTX") == 3)

        pheader = handle[0].header
        assert(pheader['DRIZOUDA'] == 'SCI')
        assert(pheader['DRIZOUWE'] == 'WHT')
        assert(pheader['DRIZOUCO'] == 'CTX')
        assert(pheader['DRIZWTSC'] == 'expsq')
        assert(pheader['DRIZKERN'] == 'turbo')
        assert(pheader['DRIZPIXF'] == 0.5)
        assert(pheader['DRIZFVAL'] == '999.')
        assert(pheader['DRIZOUUN'] == 'cps')
        assert(pheader['EXPTIME'] == 0.0)
        assert(pheader['DRIZEXPT'] == 1.0)

    def test_file_init(self):
        """
        Initialize drizzle object from a file
        """
        input_file = os.path.join(DATA_DIR, 'output_null_run.fits')        
        output_file = os.path.join(DATA_DIR, 'output_null_run.fits')

        driz = drizzle.drizzle.Drizzle(infile=input_file)
        driz.write(output_file)

        assert(driz.outexptime == 1.0)
        assert(driz.wt_scl == 'expsq')
        assert(driz.kernel == 'turbo')
        assert(driz.pixfrac == 0.5)
        assert(driz.fillval == '999.')

    def test_add_header(self):
        """
        Add extra keywords read from the header
        """
        input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits')        
        output_file = os.path.join(DATA_DIR, 'output_add_header.fits')
        output_template = os.path.join(DATA_DIR, 'reference_square_point.fits')

        driz = drizzle.drizzle.Drizzle(infile=output_template)
        image = self.read_image(input_file)
        inwcs = self.read_wcs(input_file)
        driz.add_image(image, inwcs)

        hdu = fits.ImageHDU()
        hdu.header['ONEVAL'] = 1.0
        hdu.header['TWOVAL'] = 2.0
        driz.write(output_file, outheader=hdu.header)
        driz.write(output_file)
        
        header = self.read_header(output_file)
        ##assert(header['ONEVAL'] == 1.0)
        ##assert(header['TWOVAL'] == 2.0)
        ##assert(header['NDRIZIM'] == 1)

    def test_add_file(self):
        """
        Add an image read from a file
        """
        input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits[1]')        
        output_file = os.path.join(DATA_DIR, 'output_add_file.fits')
        test_file = os.path.join(DATA_DIR, 'output_add_header.fits')
        output_template = os.path.join(DATA_DIR, 'reference_square_point.fits')

        driz = drizzle.drizzle.Drizzle(infile=output_template)
        driz.add_fits_file(input_file)
        driz.write(output_file)

        output_image = self.read_image(output_file)
        test_image =  self.read_image(test_file)
        diff_image = np.absolute(output_image - test_image)
        assert(np.amax(diff_image) == 0.0)

    def test_blot_file(self):
        """
        Blot an image read from a file
        """
        input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits[1]')
        output_file = os.path.join(DATA_DIR, 'output_blot_file.fits')
        test_file = os.path.join(DATA_DIR, 'output_blot_image.fits')
        output_template = os.path.join(DATA_DIR, 'reference_blot_image.fits')

        blotwcs = self.read_wcs(input_file)

        driz = drizzle.drizzle.Drizzle(infile=output_template)
        driz.add_fits_file(input_file)
        driz.blot_image(blotwcs)
        driz.write(test_file)
        
        driz = drizzle.drizzle.Drizzle(infile=output_template)
        driz.add_fits_file(input_file)
        driz.blot_fits_file(input_file)
        driz.write(output_file)

        output_image = self.read_image(output_file)
        test_image =  self.read_image(test_file)
        diff_image = np.absolute(output_image - test_image)
        assert(np.amax(diff_image) == 0.0)

if __name__ == "__main__":
    io = TestFileIO()
    io.test_null_run()
    io.test_file_init()
    io.test_add_header()
    io.test_add_file()
    io.test_blot_file()
