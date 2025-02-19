import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal

from drizzle.tests.helpers import wcs_from_file
from drizzle.utils import (
    _estimate_pixel_scale,
    calc_pixmap,
    decode_context,
    estimate_pixel_scale_ratio,
)


def test_map_rectangular():
    """
    Make sure the initial index array has correct values
    """
    naxis1 = 1000
    naxis2 = 10

    pixmap = np.indices((naxis1, naxis2), dtype='float32')
    pixmap = pixmap.transpose()

    assert_equal(pixmap[5, 500], (500, 5))


@pytest.mark.parametrize(
    "wcs_type", ["fits", "gwcs"]
)
def test_map_to_self(wcs_type):
    """
    Map a pixel array to itself. Should return the same array.
    """
    input_wcs = wcs_from_file("j8bt06nyq_sip_flt.fits", ext=1, wcs_type=wcs_type)
    shape = input_wcs.array_shape

    ok_pixmap = np.indices(shape, dtype='float64')
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


@pytest.mark.parametrize(
    "wcs_type", ["fits", "gwcs"]
)
def test_translated_map(wcs_type):
    """
    Map a pixel array to  at translated array.
    """
    first_wcs = wcs_from_file(
        "j8bt06nyq_sip_flt.fits",
        ext=1,
        wcs_type=wcs_type
    )
    second_wcs = wcs_from_file(
        "j8bt06nyq_sip_flt.fits",
        ext=1,
        crpix_shift=(-2, -2),  # shift loaded WCS by adding this to CRPIX
        wcs_type=wcs_type
    )

    ok_pixmap = np.indices(first_wcs.array_shape, dtype='float32') - 2.0
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
    first_wcs = wcs_from_file(
        "j8bt06nyq_sip_flt.fits",
        ext=1,
        wcs_type="gwcs"
    )
    second_wcs = wcs_from_file(
        "j8bt06nyq_sip_flt.fits",
        ext=1,
        crpix_shift=(-2, -2),  # shift loaded WCS by adding this to CRPIX
        wcs_type="gwcs"
    )

    ok_pixmap = np.indices(first_wcs.array_shape, dtype='float64') - 2.0
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
    assert abs(pscale - 0.9999999916964737) < 1.0e-9


def test_estimate_pixel_scale_no_refpix():
    # create a WCS without higher order (polynomial) distortions:
    w = wcs_from_file("j8bt06nyq_sip_flt.fits", ext=1)
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
