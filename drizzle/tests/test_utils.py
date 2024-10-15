import os

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal

from astropy import wcs
from astropy.io import fits
from drizzle.utils import (
    _estimate_pixel_scale,
    calc_pixmap,
    decode_context,
    estimate_pixel_scale_ratio,
)

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

    assert_equal(pixmap[5, 500], (500, 5))


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

    pixmap = calc_pixmap(input_wcs, input_wcs)

    # Got x-y transpose right
    assert_equal(pixmap.shape, ok_pixmap.shape)

    # Mapping an array to itself
    assert_almost_equal(pixmap, ok_pixmap, decimal=5)

    # user-provided shape
    pixmap = calc_pixmap(input_wcs, input_wcs, (12, 34))
    assert_equal(pixmap.shape, (12, 34, 2))

    # Check that an exception is raised for WCS without pixel_shape or
    # bounding_box:
    input_wcs.pixel_shape = None
    with pytest.raises(ValueError):
        calc_pixmap(input_wcs, input_wcs)

    # user-provided shape when array_shape is not set:
    pixmap = calc_pixmap(input_wcs, input_wcs, (12, 34))
    assert_equal(pixmap.shape, (12, 34, 2))

    # from bounding box:
    input_wcs.bounding_box = ((5.3, 33.5), (2.8, 11.5))
    pixmap = calc_pixmap(input_wcs, input_wcs)
    assert_equal(pixmap.shape, (12, 34, 2))

    # from bounding box and pixel_shape (the later takes precedence):
    input_wcs.pixel_shape = (naxis1, naxis2)
    pixmap = calc_pixmap(input_wcs, input_wcs)
    assert_equal(pixmap.shape, ok_pixmap.shape)


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

    pixmap = calc_pixmap(first_wcs, second_wcs)

    # Got x-y transpose right
    assert_equal(pixmap.shape, ok_pixmap.shape)
    # Mapping an array to a translated array
    assert_almost_equal(pixmap, ok_pixmap, decimal=5)


def test_estimate_pixel_scale_ratio():
    input_file = os.path.join(DATA_DIR, 'j8bt06nyq_flt.fits')

    with fits.open(input_file) as h:
        hdr = h[1].header
        assert abs(hdr['CRPIX1'] / 512 - 1) < 1e-14
        assert abs(hdr['CRPIX2'] / 512 - 1) < 1e-14
        assert abs(hdr['CRVAL1'] / 6.027148333333000000000 - 1) < 1e-14
        assert abs(hdr['CRVAL2'] / 72.08351111111000000000 + 1) < 1e-14
        assert abs(hdr['CD1_1'] / 7.360500000000000000000E-06 + 1) < 1e-14
        assert abs(hdr['CD1_2'] / 1.845600000000000000000E-06 - 1) < 1e-14
        assert abs(hdr['CD2_1'] / 2.868580000000000000000E-06 - 1) < 1e-14
        assert abs(hdr['CD2_2'] / 6.649460000000000000000E-06 - 1) < 1e-14
        w = wcs.WCS(h[1].header)
        assert np.allclose(w.wcs.crval, [6.027148333333000000000, -72.08351111111000000000], rtol=1e-14)
        assert np.allclose(w.wcs.crpix, [512.0, 512.0], rtol=1e-14)
        assert np.allclose(
            w.wcs.cd,
            [
                [-7.360500000000000000000E-06, 1.845600000000000000000E-06],
                [2.868580000000000000000E-06, 6.649460000000000000000E-06]
            ],
            rtol=1e-14
        )

        refpix = np.array([0., 0.])
        l1, phi1 = w.pixel_to_world_values(*(refpix - 0.5))
        l2, phi2 = w.pixel_to_world_values(*(refpix + [-0.5, 0.5]))
        l3, phi3 = w.pixel_to_world_values(*(refpix + 0.5))
        l4, phi4 = w.pixel_to_world_values(*(refpix + [0.5, -0.5]))
        assert abs(l1 - 6.0363204188670885) < 1e-14
        assert abs(l2 - 6.036326416554582) < 1e-14
        assert abs(l3 - 6.036302482421591) < 1e-14
        assert abs(l4 - 6.036296484726444) < 1e-14
        assert abs(phi1 - -72.08837937371479) < 1e-14
        assert abs(phi2 - -72.08837272397372) < 1e-14
        assert abs(phi3 - -72.08836985651423) < 1e-14
        assert abs(phi4 - -72.08837650625458) < 1e-14
        #pscale1 = _estimate_pixel_scale(w, refpix)
        #assert abs(pscale1 - 1.285368350331663e-07) < 1.e-21

        refpix = np.array([512., 512.])
        l1, phi1 = w.pixel_to_world_values(*(refpix - 0.5))
        l2, phi2 = w.pixel_to_world_values(*(refpix + [-0.5, 0.5]))
        l3, phi3 = w.pixel_to_world_values(*(refpix + 0.5))
        l4, phi4 = w.pixel_to_world_values(*(refpix + [0.5, -0.5]))
        assert abs(l1 - 6.027139369821101) < 1e-14
        assert abs(l2 - 6.027145369226525) < 1e-14
        assert abs(l3 - 6.027121442811089) < 1e-14
        assert abs(l4 - 6.027115443398036) < 1e-14
        assert abs(phi1 - -72.08350635208977) < 1e-13
        assert abs(phi2 - -72.08349970262996) < 1e-13
        assert abs(phi3 - -72.08349683404813) < 1e-13
        assert abs(phi4 - -72.08350348350724) < 1e-13
        #pscale2 = _estimate_pixel_scale(w, refpix)
        #assert abs(pscale2 - 1.285368361004753e-07) < 1.e-21

    pscale = estimate_pixel_scale_ratio(w, w, w.wcs.crpix, (0, 0))

    #assert (pscale1 / pscale2 - 0.9999999916964737) < 1e-14

    # assert abs(pscale - 0.9999999916967218) < 1e-14  # if using numpy in estimate_pixel_scale_ratio
    assert abs(pscale - 0.9999999916964737) < 1e-14


def test_estimate_pixel_scale_no_refpix():
    # create a WCS without higher order (polynomial) distortions:
    fits_file = os.path.join(DATA_DIR, 'input1.fits')
    with fits.open(fits_file) as h:
        w = wcs.WCS(h[1].header, h)
        w.sip = None
        w.det2im1 = None
        w.det2im2 = None
        w.cpdis1 = None
        w.cpdis2 = None
        pixel_shape = w.pixel_shape[:]

    ref_pscale = _estimate_pixel_scale(w, w.wcs.crpix)

    if hasattr(w, 'bounding_box'):
        del w.bounding_box
    pscale1 = _estimate_pixel_scale(w, None)
    assert np.allclose(ref_pscale, pscale1, atol=0.0, rtol=1.0e-8)

    w.bounding_box = None
    w.pixel_shape = None
    pscale2 = _estimate_pixel_scale(w, None)
    assert np.allclose(pscale1, pscale2, atol=0.0, rtol=1.0e-8)

    w.pixel_shape = pixel_shape
    pscale3 = _estimate_pixel_scale(w, None)
    assert np.allclose(pscale1, pscale3, atol=0.0, rtol=1.0e-14)

    w.bounding_box = ((-0.5, pixel_shape[0] - 0.5), (-0.5, pixel_shape[1] - 0.5))
    pscale4 = _estimate_pixel_scale(w, None)
    assert np.allclose(pscale3, pscale4, atol=0.0, rtol=1.0e-8)


def test_decode_context():
    ctx = np.array(
        [[[0, 0, 0, 0, 0, 0],
          [0, 0, 0, 36196864, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 537920000, 0, 0, 0]],
         [[0, 0, 0, 0, 0, 0,],
          [0, 0, 0, 67125536, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 163856, 0, 0, 0]],
         [[0, 0, 0, 0, 0, 0],
          [0, 0, 0, 8203, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 32865, 0, 0, 0]]],
        dtype=np.int32
    )

    idx1, idx2 = decode_context(ctx, [3, 2], [1, 4])

    assert sorted(idx1) == [9, 12, 14, 19, 21, 25, 37, 40, 46, 58, 64, 65, 67, 77]
    assert sorted(idx2) == [9, 20, 29, 36, 47, 49, 64, 69, 70, 79]

    # context array must be 3D:
    with pytest.raises(ValueError):
        decode_context(ctx[0], [3, 2], [1, 4])

    # pixel coordinates must be integer:
    with pytest.raises(ValueError):
        decode_context(ctx, [3.0, 2], [1, 4])

    # coordinate lists must be equal in length:
    with pytest.raises(ValueError):
        decode_context(ctx, [3, 2], [1, 4, 5])

    # coordinate lists must be 1D:
    with pytest.raises(ValueError):
        decode_context(ctx, [[3, 2]], [[1, 4]])
