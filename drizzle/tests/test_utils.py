import warnings

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal

from astropy.wcs import FITSFixedWarning, WCS
from astropy.coordinates import ICRS, SkyCoord
from astropy.wcs.utils import pixel_to_pixel

from drizzle.tests.helpers import wcs_from_file
from drizzle.utils import (
    FrameMismatchError,
    FrameMismatchWarning,
    _estimate_pixel_scale,
    calc_pixmap,
    decode_context,
    estimate_pixel_scale_ratio,
)


_SHAPE = (3, 4)


class _DuckWCS:
    """Low-level APE 14 only; no world_axis_object_classes."""

    array_shape = (3, 4)

    def pixel_to_world_values(self, x, y):
        return x, y

    def world_to_pixel_values(self, a, b):
        return a, b


class _FrameDuckWCS(_DuckWCS):
    """
    Low-level-only duck WCS that declares celestial frame(s) via
    ``world_axis_object_classes`` but has no high-level API.
    """

    def __init__(self, *frames, cls="SkyCoord", with_kwargs=True):
        classes = {}
        for k, frame in enumerate(frames):
            kwargs = {"frame": frame} if with_kwargs else {}
            classes[f"celestial{k}"] = (cls, (), kwargs)
        self.world_axis_object_classes = classes


def _identity_pixmap(shape=_SHAPE):
    y, x = np.indices(shape, dtype=float)
    return np.dstack([x, y])


def _celestial_wcs(ctype, crval, cdelt=(-1.0e-4, 1.0e-4), shape=(50, 60)):
    w = WCS(naxis=2)
    w.wcs.ctype = list(ctype)
    w.wcs.crval = list(crval)
    w.wcs.crpix = [shape[1] / 2 + 0.5, shape[0] / 2 + 0.5]
    w.wcs.cdelt = list(cdelt)
    w.wcs.cunit = ["deg", "deg"]
    if ctype[0].startswith(("RA", "DEC")):
        w.wcs.radesys = "ICRS"
    w.wcs.set()
    w.array_shape = shape
    return w


def _icrs_and_galactic_wcs():
    ra, dec = 150.0, 2.0
    gal = SkyCoord(ra, dec, unit="deg", frame="icrs").galactic
    w_icrs = _celestial_wcs(("RA---TAN", "DEC--TAN"), (ra, dec))
    # slightly offset so the mapping is not a trivial identity:
    w_gal = _celestial_wcs(
        ("GLON-TAN", "GLAT-TAN"),
        (gal.l.deg + 1.0e-3, gal.b.deg - 1.0e-3),
        cdelt=(-1.2e-4, 1.2e-4),
    )
    return w_icrs, w_gal


def _reference_pixmap(wcs_from, wcs_to):
    y, x = np.indices(wcs_from.array_shape, dtype=float)
    xo, yo = pixel_to_pixel(wcs_from, wcs_to, x, y)
    return np.dstack([xo, yo])


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


def test_frameless_wcs_strict_raises():
    with pytest.raises(FrameMismatchError, match="neither"):
        calc_pixmap(_DuckWCS(), _DuckWCS(), frame_check="strict")


@pytest.mark.parametrize("frame_check", ["warn", "ignore"])
def test_frameless_wcs_lenient_is_silent(frame_check):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pixmap = calc_pixmap(_DuckWCS(), _DuckWCS(), frame_check=frame_check)
    y, x = np.indices((3, 4), dtype=float)
    np.testing.assert_array_equal(pixmap, np.dstack([x, y]))


@pytest.mark.parametrize("frame_check", ["strict", "warn"])
def test_fits_frame_conversion(frame_check):
    w_icrs, w_gal = _icrs_and_galactic_wcs()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FrameMismatchWarning)
        pixmap = calc_pixmap(w_icrs, w_gal, frame_check=frame_check)
    assert_almost_equal(pixmap, _reference_pixmap(w_icrs, w_gal), decimal=6)


def test_fits_frame_ignore_skips_conversion():
    w_icrs, w_gal = _icrs_and_galactic_wcs()
    pixmap = calc_pixmap(w_icrs, w_gal, frame_check="ignore")

    # legacy behavior: numeric world values passed straight through
    y, x = np.indices(w_icrs.array_shape, dtype=float)
    xo, yo = w_gal.world_to_pixel_values(*w_icrs.pixel_to_world_values(x, y))
    assert_equal(pixmap, np.dstack([xo, yo]))

    # ... which is very different from the correct, converted result:
    assert not np.allclose(pixmap, _reference_pixmap(w_icrs, w_gal), atol=1.0)


def test_fits_swapped_axes_converts():
    ra, dec = 150.0, 2.0
    w_normal = _celestial_wcs(("RA---TAN", "DEC--TAN"), (ra, dec))
    w_swapped = _celestial_wcs(
        ("DEC--TAN", "RA---TAN"), (dec, ra), cdelt=(1.0e-4, -1.0e-4)
    )
    pixmap = calc_pixmap(w_normal, w_swapped, frame_check="strict")
    assert_almost_equal(pixmap, _reference_pixmap(w_normal, w_swapped), decimal=6)


def test_fits_same_frame_strict():
    w = wcs_from_file("j8bt06nyq_sip_flt.fits", ext=1, wcs_type="fits")
    pixmap = calc_pixmap(w, w, frame_check="strict")
    ok_pixmap = np.indices(w.array_shape, dtype="float64").transpose()
    assert_almost_equal(pixmap, ok_pixmap, decimal=5)


@pytest.mark.parametrize("swap", [False, True])
def test_frame_info_on_one_side_only(swap):
    wcs_pair = [_FrameDuckWCS(ICRS()), _DuckWCS()]
    if swap:
        wcs_pair.reverse()

    with pytest.raises(FrameMismatchError, match="does not provide"):
        calc_pixmap(*wcs_pair, frame_check="strict")

    with pytest.warns(FrameMismatchWarning, match="does not provide"):
        pixmap = calc_pixmap(*wcs_pair, frame_check="warn")
    assert_equal(pixmap, _identity_pixmap())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        calc_pixmap(*wcs_pair, frame_check="ignore")


def test_undefined_frame():
    w_from = _FrameDuckWCS(ICRS(), with_kwargs=False)
    w_to = _FrameDuckWCS(ICRS())
    with pytest.raises(FrameMismatchError, match="not defined"):
        calc_pixmap(w_from, w_to, frame_check="strict")
    with pytest.warns(FrameMismatchWarning, match="not defined"):
        calc_pixmap(w_from, w_to, frame_check="warn")


def test_uncomparable_frames():
    w_from = _FrameDuckWCS(object())
    w_to = _FrameDuckWCS(object())
    with pytest.raises(FrameMismatchError, match=r"(?i)unable to compare"):
        calc_pixmap(w_from, w_to, frame_check="strict")
    with pytest.warns(FrameMismatchWarning, match=r"(?i)unable to compare"):
        calc_pixmap(w_from, w_to, frame_check="warn")


def test_different_number_of_celestial_axes():
    w_from = _FrameDuckWCS()  # world_axis_object_classes == {}
    w_to = _FrameDuckWCS(ICRS())
    with pytest.raises(FrameMismatchError, match="different numbers"):
        calc_pixmap(w_from, w_to, frame_check="strict")


def test_string_frames_equivalent_case_insensitive():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pixmap = calc_pixmap(
            _FrameDuckWCS("icrs"), _FrameDuckWCS("ICRS"), frame_check="strict"
        )
    assert_equal(pixmap, _identity_pixmap())


def test_string_frames_differ_no_high_level_api():
    w_from = _FrameDuckWCS("icrs")
    w_to = _FrameDuckWCS("galactic")
    with pytest.raises(FrameMismatchError, match="high-level API"):
        calc_pixmap(w_from, w_to, frame_check="strict")
    with pytest.warns(FrameMismatchWarning, match="high-level API"):
        pixmap = calc_pixmap(w_from, w_to, frame_check="warn")
    assert_equal(pixmap, _identity_pixmap())


@pytest.mark.parametrize("cls", [SkyCoord, "SkyCoord", "astropy.coordinates.SkyCoord"])
def test_skycoord_class_given_as_type(cls):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        calc_pixmap(
            _FrameDuckWCS(ICRS(), cls=cls),
            _FrameDuckWCS(ICRS(), cls=cls),
            frame_check="strict",
        )


@pytest.mark.parametrize(
    ("from_type", "to_type"),
    [
        ("fits", "fits"),
        ("fits", "gwcs"),
        ("gwcs", "fits"),
        ("gwcs", "gwcs"),
    ],
)
@pytest.mark.parametrize("frame_check", ["strict", "warn"])
def test_same_frame_wcs_combinations(from_type, to_type, frame_check):
    filename = "j8bt06nyq_sip_flt.fits"
    w_from = wcs_from_file(filename, ext=1, wcs_type=from_type)
    w_to = wcs_from_file(
        filename,
        ext=1,
        crpix_shift=(-2, -2),
        wcs_type=to_type,
    )

    # Input grid starts at zero. The shifted output WCS can map border
    # pixels outside its bounding box.
    shape = (12, 34)
    y, x = np.indices(shape, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FrameMismatchWarning)
        expected_x, expected_y = pixel_to_pixel(w_from, w_to, x, y)
        actual = calc_pixmap(
            w_from,
            w_to,
            shape=shape,
            disable_bbox="none",
            frame_check=frame_check,
        )

    expected = np.dstack([expected_x, expected_y])
    assert actual.shape == (*shape, 2)

    # Match the reference's component-wise handling of out-of-bounds pixels.
    assert_equal(np.isnan(actual), np.isnan(expected))

    # Interior is always finite:
    assert np.all(np.isfinite(actual[2:, 2:]))

    # Compare all finite values as well as matching NaNs.
    np.testing.assert_allclose(
        actual, expected, rtol=0, atol=1e-5, equal_nan=True
    )


def test_non_celestial_world_objects_are_skipped():
    w_from = _FrameDuckWCS(ICRS())
    w_from.world_axis_object_classes["spectral"] = ("SpectralCoord", (), {})
    w_to = _FrameDuckWCS(ICRS())
    w_to.world_axis_object_classes["spectral"] = ("SpectralCoord", (), {})
    calc_pixmap(w_from, w_to, frame_check="strict")


@pytest.mark.parametrize("frame_check", ["raise", "STRICT", None, ""])
def test_invalid_frame_check(frame_check):
    with pytest.raises(ValueError, match="frame_check") as excinfo:
        calc_pixmap(_DuckWCS(), _DuckWCS(), frame_check=frame_check)
    assert not isinstance(excinfo.value, FrameMismatchError)


def test_frame_mismatch_error_is_value_error():
    assert issubclass(FrameMismatchError, ValueError)
    with pytest.raises(ValueError):
        calc_pixmap(_DuckWCS(), _DuckWCS(), frame_check="strict")


def test_warning_stacklevel():
    with pytest.warns(FrameMismatchWarning) as record:
        calc_pixmap(_FrameDuckWCS(ICRS()), _DuckWCS(), frame_check="warn")
    assert record[0].filename == __file__


def test_bbox_untouched_on_frame_error():
    w_from = _FrameDuckWCS(ICRS())
    w_from.bounding_box = ((-0.5, 3.5), (-0.5, 2.5))
    w_to = _DuckWCS()
    w_to.bounding_box = ((-0.5, 3.5), (-0.5, 2.5))

    with pytest.raises(FrameMismatchError):
        calc_pixmap(w_from, w_to, disable_bbox="both", frame_check="strict")

    assert w_from.bounding_box == ((-0.5, 3.5), (-0.5, 2.5))
    assert w_to.bounding_box == ((-0.5, 3.5), (-0.5, 2.5))


def test_estimate_pixel_scale_ratio():
    w = wcs_from_file("j8bt06nyq_flt.fits", ext=1)
    pscale = estimate_pixel_scale_ratio(w, w, w.wcs.crpix, (0, 0))
    assert abs(pscale - 0.9999999916964737) < 1.0e-9


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
