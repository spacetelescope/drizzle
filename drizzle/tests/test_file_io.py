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

from .. import drizzle
from .. import util

def read_header(filename):
    """
    Read the primary header from a fits file
    """
    fileroot, extn = util.parse_filename(os.path.join(DATA_DIR, filename))
    hdu = fits.open(fileroot)

    header = hdu[0].header
    hdu.close()
    return header

def read_image(filename):
    """
    Read the image from a fits file
    """
    fileroot, extn = util.parse_filename(os.path.join(DATA_DIR, filename))
    hdu = fits.open(fileroot)

    image = hdu[1].data
    hdu.close()
    return image

def read_wcs(filename):
    """
    Read the wcs of a fits file
    """
    fileroot, extn = util.parse_filename(os.path.join(DATA_DIR, filename))
    hdu = fits.open(fileroot)

    the_wcs = wcs.WCS(hdu[1].header)
    hdu.close()
    return the_wcs

def test_null_run():
    """
    Create an empty drizzle image
    """
    output_file = os.path.join(DATA_DIR, 'output_null_run.fits')
    output_template = os.path.join(DATA_DIR, 'reference_square_point.fits')

    output_wcs = read_wcs(output_template)
    driz = drizzle.Drizzle(outwcs=output_wcs, wt_scl="expsq",
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

def test_file_init():
    """
    Initialize drizzle object from a file
    """
    input_file = os.path.join(DATA_DIR, 'output_null_run.fits')        
    output_file = os.path.join(DATA_DIR, 'output_null_run.fits')

    driz = drizzle.Drizzle(infile=input_file)
    driz.write(output_file)

    assert(driz.outexptime == 1.0)
    assert(driz.wt_scl == 'expsq')
    assert(driz.kernel == 'turbo')
    assert(driz.pixfrac == 0.5)
    assert(driz.fillval == '999.')

def test_add_header():
    """
    Add extra keywords read from the header
    """
    input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits')        
    output_file = os.path.join(DATA_DIR, 'output_add_header.fits')
    output_template = os.path.join(DATA_DIR, 'reference_square_point.fits')

    driz = drizzle.Drizzle(infile=output_template)
    image = read_image(input_file)
    inwcs = read_wcs(input_file)
    driz.add_image(image, inwcs)

    header = fits.header.Header()
    header['ONEVAL'] = (1.0, 'test value')
    header['TWOVAL'] = (2.0, 'test value')

    driz.write(output_file, outheader=header)
    
    header = read_header(output_file)
    assert(header['ONEVAL'] == 1.0)
    assert(header['TWOVAL'] == 2.0)
    assert(header['DRIZKERN'] == 'square')

def test_add_file():
    """
    Add an image read from a file
    """
    input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits[1]')        
    output_file = os.path.join(DATA_DIR, 'output_add_file.fits')
    test_file = os.path.join(DATA_DIR, 'output_add_header.fits')
    output_template = os.path.join(DATA_DIR, 'reference_square_point.fits')

    driz = drizzle.Drizzle(infile=output_template)
    driz.add_fits_file(input_file)
    driz.write(output_file)

    output_image = read_image(output_file)
    test_image =  read_image(test_file)
    diff_image = np.absolute(output_image - test_image)
    assert(np.amax(diff_image) == 0.0)

def test_blot_file():
    """
    Blot an image read from a file
    """
    input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits[1]')
    output_file = os.path.join(DATA_DIR, 'output_blot_file.fits')
    test_file = os.path.join(DATA_DIR, 'output_blot_image.fits')
    output_template = os.path.join(DATA_DIR, 'reference_blot_image.fits')

    blotwcs = read_wcs(input_file)

    driz = drizzle.Drizzle(infile=output_template)
    driz.add_fits_file(input_file)
    driz.blot_image(blotwcs)
    driz.write(test_file)
    
    driz = drizzle.Drizzle(infile=output_template)
    driz.add_fits_file(input_file)
    driz.blot_fits_file(input_file)
    driz.write(output_file)

    output_image = read_image(output_file)
    test_image =  read_image(test_file)
    diff_image = np.absolute(output_image - test_image)
    assert(np.amax(diff_image) == 0.0)
