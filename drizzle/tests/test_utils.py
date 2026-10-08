import math
import warnings

import numpy as np
import pytest
from astropy import coordinates as coord
from astropy import units as u
from astropy.modeling import models
from astropy.wcs import FITSFixedWarning, WCS
from gwcs import coordinate_frames as cf
from gwcs.wcs import WCS as GWCS
from numpy.testing import assert_almost_equal, assert_equal

from drizzle.tests.helpers import wcs_from_file
from drizzle.utils import (
    _DEG2RAD,
    _estimate_pixel_scale,
    calc_pixmap,
    decode_context,
    estimate_pixel_scale_ratio,
)


def _minimal_tan_wcs(crval, crpix, cdelt=0.01):
    # Minimal hand-built TAN WCS. Note: no wcsset() has been run yet, so
    # world_axis_units will report '' for both axes (see test below).
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = list(crval)
    w.wcs.crpix = list(crpix)
    w.wcs.cdelt = [-cdelt, cdelt]
    w.wcs.set()
    return w


def test_map_rectangular():
    """
    Make sure the initial index array has correct values
    """
    naxis1 = 1000
    naxis2 = 10

    pixmap = np.indices((naxis1, naxis2), dtype="float32")
    pixmap = pixmap.transpose()

    assert_equal(pixmap[5, 500], (500, 5))


@pytest.mark.parametrize("wcs_type", ["fits", "gwcs"])
def test_map_to_self(wcs_type):
    """
    Map a pixel array to itself. Should return the same array.
    """
    input_wcs = wcs_from_file("j8bt06nyq_sip_flt.fits", ext=1, wcs_type=wcs_type)
    shape = input_wcs.array_shape

    ok_pixmap = np.indices(shape, dtype="float64")
    ok_pixmap = ok_pixmap.transpose()

    pixmap = calc_pixmap(input_wcs, input_wcs)

    # Got x-y transpose right
    assert_equal(pixmap.shape, ok_pixmap.shape)

    # Mapping an array to itself
    assert_almost_equal(pixmap, ok_pixmap, decimal=5)

    # user-provided shape
    pixmap = calc_pixmap(input_wcs, input_wcs, (12, 34))
    assert_equal(pixmap.shape, (12, 34, 2))

    # Check that an exception is raised for WCS without pixel_shape and
    # bounding_box:
    input_wcs.pixel_shape = None
    input_wcs.bounding_box = None
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
    input_wcs.array_shape = shape
    pixmap = calc_pixmap(input_wcs, input_wcs)
    assert_equal(pixmap.shape, ok_pixmap.shape)


@pytest.mark.parametrize("wcs_type", ["fits", "gwcs"])
def test_translated_map(wcs_type):
    """
    Map a pixel array to  at translated array.
    """
    first_wcs = wcs_from_file("j8bt06nyq_sip_flt.fits", ext=1, wcs_type=wcs_type)
    second_wcs = wcs_from_file(
        "j8bt06nyq_sip_flt.fits",
        ext=1,
        crpix_shift=(-2, -2),  # shift loaded WCS by adding this to CRPIX
        wcs_type=wcs_type,
    )

    ok_pixmap = np.indices(first_wcs.array_shape, dtype="float32") - 2.0
    ok_pixmap = ok_pixmap.transpose()

    pixmap = calc_pixmap(first_wcs, second_wcs)

    # Got x-y transpose right
    assert_equal(pixmap.shape, ok_pixmap.shape)
    # Mapping an array to a translated array
    assert_almost_equal(pixmap[2:, 2:], ok_pixmap[2:, 2:], decimal=5)


def test_disable_gwcs_bbox():
    """
    Map a pixel array to a translated version ofitself.
    """
    first_wcs = wcs_from_file("j8bt06nyq_sip_flt.fits", ext=1, wcs_type="gwcs")
    second_wcs = wcs_from_file(
        "j8bt06nyq_sip_flt.fits",
        ext=1,
        crpix_shift=(-2, -2),  # shift loaded WCS by adding this to CRPIX
        wcs_type="gwcs",
    )

    ok_pixmap = np.indices(first_wcs.array_shape, dtype="float64") - 2.0
    ok_pixmap = ok_pixmap.transpose()

    # Mapping an array to a translated array

    # disable both bounding boxes:
    pixmap = calc_pixmap(first_wcs, second_wcs, disable_bbox="both")
    assert_almost_equal(pixmap[2:, 2:], ok_pixmap[2:, 2:], decimal=5)
    assert np.all(np.isfinite(pixmap[:2, :2]))
    assert np.all(np.isfinite(pixmap[-2:, -2:]))
    # check bbox was restored
    assert first_wcs.bounding_box is not None
    assert second_wcs.bounding_box is not None

    # disable "from" bounding box:
    pixmap = calc_pixmap(second_wcs, first_wcs, disable_bbox="from")
    assert_almost_equal(pixmap[:-2, :-2], ok_pixmap[:-2, :-2] + 4.0, decimal=5)
    assert np.all(np.logical_not(np.isfinite(pixmap[-2:, -2:])))
    # check bbox was restored
    assert first_wcs.bounding_box is not None
    assert second_wcs.bounding_box is not None

    # disable "to" bounding boxes:
    pixmap = calc_pixmap(first_wcs, second_wcs, disable_bbox="to")
    assert_almost_equal(pixmap[2:, 2:], ok_pixmap[2:, 2:], decimal=5)
    assert np.all(np.isfinite(pixmap[:2, :2]))
    assert np.all(pixmap[:2, :2] < 0.0)
    assert np.all(np.isfinite(pixmap[-2:, -2:]))
    # check bbox was restored
    assert first_wcs.bounding_box is not None
    assert second_wcs.bounding_box is not None

    # enable all bounding boxes:
    pixmap = calc_pixmap(first_wcs, second_wcs, disable_bbox="none")
    assert_almost_equal(pixmap[2:, 2:], ok_pixmap[2:, 2:], decimal=5)
    assert np.all(np.logical_not(np.isfinite(pixmap[:2, :2])))
    # check bbox was restored
    assert first_wcs.bounding_box is not None
    assert second_wcs.bounding_box is not None


def test_estimate_pixel_scale_ratio():
    w = wcs_from_file("j8bt06nyq_flt.fits", ext=1)
    pscale = estimate_pixel_scale_ratio(w, w, w.wcs.crpix, (0, 0))
    assert abs(pscale - 0.9999999911644557) < 1.0e-9


def test_estimate_pixel_scale_no_refpix():
    # create a WCS without higher order (polynomial) distortions:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FITSFixedWarning)
        w = wcs_from_file("j8bt06nyq_sip_flt.fits", ext=1)
    w.sip = None
    w.det2im1 = None
    w.det2im2 = None
    w.cpdis1 = None
    w.cpdis2 = None
    pixel_shape = w.pixel_shape[:]

    ref_pscale = _estimate_pixel_scale(w, w.wcs.crpix)

    if hasattr(w, "bounding_box"):
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
        [
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 36196864, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 537920000, 0, 0, 0],
            ],
            [
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [0, 0, 0, 67125536, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 163856, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 8203, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 32865, 0, 0, 0],
            ],
        ],
        dtype=np.int32,
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


def test_estimate_pixel_scale_longitude_wrap():
    """
    Regression test: the reference pixel straddles the RA = 0/360 meridian,
    so its corners have longitudes ~359.99 and ~0.01 degrees. Without
    unwrapping, the pixel "area" spans ~360 degrees and the scale comes out
    off by orders of magnitude (~1.9 rad instead of ~1.7e-4 rad).
    """
    cdelt = 0.01
    # pixel (0, 0) center maps exactly to RA = 0
    w = _minimal_tan_wcs(crval=(0.0, 40.0), crpix=(1.0, 1.0), cdelt=cdelt)

    pscale = _estimate_pixel_scale(w, (0, 0))

    # at dec = 40 the pixel is cdelt x cdelt on the sky by construction (TAN
    # is conformal at the tangent point), so the scale must be cdelt in rad
    assert math.isclose(pscale, cdelt * _DEG2RAD, rel_tol=1e-6)

    # and it must agree with an identical pixel far from the meridian
    w_ref = _minimal_tan_wcs(crval=(180.0, 40.0), crpix=(1.0, 1.0), cdelt=cdelt)
    assert math.isclose(pscale, _estimate_pixel_scale(w_ref, (0, 0)), rel_tol=1e-9)

    # the ratio of the two must be 1, and the public function must agree too
    assert math.isclose(estimate_pixel_scale_ratio(w_ref, w, (0, 0), (0, 0)), 1.0, rel_tol=1e-9)


def test_estimate_pixel_scale_fits_wcs_empty_units():
    """
    A hand-built astropy.wcs.WCS reports world_axis_units == ['', ''] until
    wcsset() has run (e.g. on the first transform) even if CTYPE is set to
    celestial coordinates. The FITS standard defines
    celestial coordinates to be in degrees in that case, so an empty unit
    string must be accepted by the units check.
    """
    w = WCS(naxis=2)
    with pytest.raises(ValueError, match="degrees"):
        _estimate_pixel_scale(w, (0, 0))


def test_estimate_pixel_scale_rejects_non_degree_units():
    """
    A gwcs WCS whose celestial output frame is explicitly in arcseconds must
    be rejected rather than silently treated as degrees.
    """
    det = cf.Frame2D(name="detector", axes_order=(0, 1), unit=(u.pix, u.pix))
    sky = cf.CelestialFrame(
        name="sky",
        reference_frame=coord.ICRS(),
        unit=(u.arcsec, u.arcsec),
    )
    # a trivial but valid pixel -> sky transform (values are irrelevant here)
    transform = models.Scale(36.0) & models.Scale(36.0)
    w = GWCS([(det, transform), (sky, None)])

    assert list(w.world_axis_units) == ["arcsec", "arcsec"]

    with pytest.raises(ValueError, match="degrees"):
        _estimate_pixel_scale(w, (0, 0))


def test_estimate_pixel_scale_refpix_center_even_shape():
    """
    With no refpix and no bounding box, the reference pixel is the image
    center, which is a half-integer for even-sized axes.
    """
    w = _minimal_tan_wcs(crval=(30.0, 40.0), crpix=(50.5, 50.5))
    w = wcs_from_file("j8bt06nyq_sip_flt.fits", ext=1, wcs_type="gwcs")

    w.pixel_shape = (100, 98)
    w.bounding_box = None
    assert math.isclose(
        _estimate_pixel_scale(w, None),
        _estimate_pixel_scale(w, (49.5, 48.5)),
        rel_tol=1e-14,
    )

    w.pixel_shape = None
    w.bounding_box = ((0, 99), (0, 97))
    assert math.isclose(
        _estimate_pixel_scale(w, None),
        _estimate_pixel_scale(w, (49.5, 48.5)),
        rel_tol=1e-14,
    )

def test_estimate_pixel_scale_rejects_non_2d():
    w = WCS(naxis=3)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN", "WAVE"]
    w.pixel_shape = (10, 10, 5)

    with pytest.raises(ValueError, match="2D"):
        _estimate_pixel_scale(w, None)

    with pytest.raises(ValueError, match="2D"):
        _estimate_pixel_scale(w, (1, 2, 3))


def test_calc_pixmap_bbox_object_preserved():
    """
    calc_pixmap must not replace the WCS bounding_box object (e.g. a gwcs
    ModelBoundingBox) with a plain tuple when restoring it or leave it as None.
    """
    w = wcs_from_file("j8bt06nyq_sip_flt.fits", ext=1, wcs_type="gwcs")
    w_orig = wcs_from_file("j8bt06nyq_sip_flt.fits", ext=1, wcs_type="gwcs")

    calc_pixmap(w, w, disable_bbox="both")
    assert w.bounding_box == w_orig.bounding_box

    calc_pixmap(w, w, disable_bbox="none")
    assert w.bounding_box == w_orig.bounding_box
