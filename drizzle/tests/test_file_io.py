import os

import pytest
import numpy as np
from astropy import wcs
from astropy.io import fits

from drizzle import drizzle
from drizzle import util


TEST_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(TEST_DIR, 'data')


def read_header(filename):
    """
    Read the primary header from a fits file
    """
    fileroot, extn = util.parse_filename(os.path.join(DATA_DIR, filename))
    with fits.open(fileroot) as hdulist:
        header = hdulist[0].header
    return header


def read_image(filename):
    """
    Read the image from a fits file
    """
    fileroot, extn = util.parse_filename(os.path.join(DATA_DIR, filename))
    with fits.open(fileroot, memmap=False) as hdulist:
        data = hdulist[1].data.copy()
    return data


def read_wcs(filename):
    """
    Read the wcs of a fits file
    """
    fileroot, extn = util.parse_filename(os.path.join(DATA_DIR, filename))
    with fits.open(fileroot) as hdulist:
        the_wcs = wcs.WCS(hdulist[1].header)
    return the_wcs


@pytest.fixture
def run_drizzle_reference_square_points(tmpdir):
    """Create an empty drizzle image"""
    output_file = str(tmpdir.join('output_null_run.fits'))
    output_template = os.path.join(DATA_DIR, 'reference_square_point.fits')

    output_wcs = read_wcs(output_template)
    driz = drizzle.Drizzle(outwcs=output_wcs, wt_scl="expsq", pixfrac=0.5,
                           kernel="turbo", fillval="NaN")
    driz.write(output_file)

    return output_file


def test_null_run(run_drizzle_reference_square_points):
    output_file = run_drizzle_reference_square_points
    with fits.open(output_file) as hdulist:

        assert hdulist.index_of("SCI") == 1
        assert hdulist.index_of("WHT") == 2
        assert hdulist.index_of("CTX") == 3

        pheader = hdulist["PRIMARY"].header

    assert pheader['DRIZOUDA'] == 'SCI'
    assert pheader['DRIZOUWE'] == 'WHT'
    assert pheader['DRIZOUCO'] == 'CTX'
    assert pheader['DRIZWTSC'] == 'expsq'
    assert pheader['DRIZKERN'] == 'turbo'
    assert pheader['DRIZPIXF'] == 0.5
    assert pheader['DRIZFVAL'] == 'NaN'
    assert pheader['DRIZOUUN'] == 'cps'
    assert pheader['EXPTIME'] == 0.0
    assert pheader['DRIZEXPT'] == 1.0


def test_file_init(run_drizzle_reference_square_points):
    """
    Initialize drizzle object from a file
    """
    input_file = run_drizzle_reference_square_points

    driz = drizzle.Drizzle(infile=input_file)

    assert driz.outexptime == 1.0
    assert driz.wt_scl == 'expsq'
    assert driz.kernel == 'turbo'
    assert driz.pixfrac == 0.5
    assert driz.fillval == 'NaN'


@pytest.fixture
def add_header(tmpdir):
    """Add extra keywords read from the header"""
    output_file = str(tmpdir.join('output_add_header.fits'))

    input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits')
    output_template = os.path.join(DATA_DIR, 'reference_square_point.fits')

    driz = drizzle.Drizzle(infile=output_template)
    image = read_image(input_file)
    inwcs = read_wcs(input_file)
    driz.add_image(image, inwcs)

    header = fits.header.Header()
    header['ONEVAL'] = (1.0, 'test value')
    header['TWOVAL'] = (2.0, 'test value')

    driz.write(output_file, outheader=header)

    return output_file


def test_add_header(add_header):
    output_file = add_header
    header = read_header(output_file)
    assert header['ONEVAL'] == 1.0
    assert header['TWOVAL'] == 2.0
    assert header['DRIZKERN'] == 'square'


def test_add_file(add_header, tmpdir):
    """
    Add an image read from a file
    """
    test_file = add_header
    output_file = str(tmpdir.join('output_add_file.fits'))

    input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits[1]')
    output_template = os.path.join(DATA_DIR, 'reference_square_point.fits')

    driz = drizzle.Drizzle(infile=output_template)
    driz.add_fits_file(input_file)
    driz.write(output_file)

    output_image = read_image(output_file)
    test_image = read_image(test_file)
    diff_image = np.absolute(output_image - test_image)

    assert np.amax(diff_image) == 0.0


def test_blot_file(tmpdir):
    """
    Blot an image read from a file
    """
    output_file = str(tmpdir.join('output_blot_file.fits'))
    test_file = str(tmpdir.join('output_blot_image.fits'))

    input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits[1]')
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
    test_image = read_image(test_file)
    diff_image = np.absolute(output_image - test_image)

    assert np.amax(diff_image) == 0.0
