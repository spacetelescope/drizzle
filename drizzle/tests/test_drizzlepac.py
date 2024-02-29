import os
import pytest

import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy.nddata import bitfield_to_boolean_mask

from drizzle import cdrizzle

TEST_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(TEST_DIR, 'data')
ok = False

# Filename template for all references created by drizzlepac
REFNAME = 'reference_drizzlepac_kernel_drz_sci.fits'
# Generic name of input subarray image
INPUT_NAME = 'test_drizzlepac_flt.fits'
# pre-defined pixmap which replicates drizzlepac results for this test data
PIXMAP_NAME = 'test_drizzlepac_pixmap.fits'
#
# Basic kernels demonstrate accuracy of pixmap
TEST_KERNELS_BASIC = ['point']
#
# Extended kernels exercise effects of larger kernels
# especially with regard to edge effects which currently
# result in different results from drizzlepac.
# If these differences are understood and corrected to match
# drizzlepac, then they can be moved to TEST_KERNELS_BASIC and
# be expected to PASS.
TEST_KERNELS_EXTENDED = ['square','gaussian', 'lanczos3', 'turbo']


def read_inputs(kernel):
    # build the path to the reference file in the test directory
    refname = os.path.join(DATA_DIR, REFNAME.replace('kernel', kernel))

    # read in input image and create weight array
    indict = {}
    with fits.open(os.path.join(DATA_DIR, INPUT_NAME)) as fhdu:
        expin = fhdu[0].header.get('exptime')
        indict['insci'] = fhdu[('SCI', 1)].data
        indict['inwht'] = bitfield_to_boolean_mask(fhdu[('DQ', 1)].data,
                                                   784, good_mask_value=1)
        indict['expin'] = expin

    # read in reference image WCS to define output images
    outwcs = wcs.WCS(fits.open(refname)[0])

    # Read pixmap into memory
    indict['pixmap'] = fits.getdata(os.path.join(DATA_DIR, PIXMAP_NAME))

    # define output arrays here
    outshape = outwcs.array_shape
    indict['outsci'] = np.zeros(outshape, dtype=np.float32)
    indict['outwht'] = np.zeros(outshape, dtype=np.float32)
    indict['outcon'] = np.zeros(outshape, dtype=np.int32)

    return indict


def run_drizzle_test(kernel):
    """
    Test do_drizzle and compare results to drizzlepac generated reference
    """
    # Read in parameters and setup output arrays for each test
    # testdict = setup_pars()
    testdict = read_inputs(kernel)

    # resample:
    #
    # The parameter 'scale' needs to be hard-coded to 1.0 in order to match
    # the behavior of Drizzlepac, since Drizzlepac uses the IDCSCALE based on the
    # undistorted WCS plate-scale to define the output plate scale.  This is
    # required since the distortion model could be applying an arbitrary plate
    # scale change by default (as specified by the IDCSCALE keyword value).
    #
    _ = cdrizzle.tdriz(
        testdict['insci'].astype(np.float32),
        testdict['inwht'].astype(np.float32),
        testdict['pixmap'],
        testdict['outsci'],
        testdict['outwht'],
        testdict['outcon'],
        uniqid=1,
        xmin=0, xmax=0,
        ymin=0, ymax=0,
        scale=1.0,
        pixfrac=1.0,
        kernel=kernel,
        in_units='cps',
        expscale=1.0,
        wtscale=testdict['expin'],
        fillstr=str(0),
    )

    # convert reference from list to ndarray
    # refdata = np.array(testdict['references'][kernel], dtype=np.float32)
    # read in reference data from FITS file
    refdata = fits.getdata(os.path.join(DATA_DIR, REFNAME.replace('kernel', kernel)))

    # check that result is within floating point precision of Drizzlepac result
    assert np.allclose(testdict['outsci'], refdata, rtol=1e-5)


@pytest.mark.parametrize(
    'kernel', ['point'],
)
def test_vs_drizzlepac_basic(kernel):
    """
    Test do_drizzle and compare results to drizzlepac generated reference
    """
    run_drizzle_test(kernel)

@pytest.mark.xfail
@pytest.mark.parametrize(
    'kernel', ['turbo', 'square', 'lanczos3', 'gaussian'],
)
def test_vs_drizzlepac_extended(kernel):
    run_drizzle_test(kernel)

