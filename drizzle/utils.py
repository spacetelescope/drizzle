"""
Utility functions for computing inputs to the drizzle algorithm.

The functions in this module, such as `calc_pixmap` and
`estimate_pixel_scale_ratio`, are convenience helpers. They cover common
cases, primarily 2D celestial images described by `astropy.wcs.WCS` or
`gwcs.WCS` objects that follow the APE 14 shared WCS interface, and they
also illustrate how drizzle's inputs can be derived from WCS information.

They are not intended to handle every possible WCS. Coordinate systems,
distortion models, projections, bounding-box conventions and frame
definitions vary widely, and a general-purpose solution for all of them
is outside the scope of this package. If these helpers do not support
your WCS objects, or produce results that are inaccurate for your data,
compute the required quantities yourself:

- a pixel map can be computed by any means and passed directly to
  `drizzle.resample.Drizzle.add_image`;
- a pixel scale ratio can likewise be computed independently and passed
  via the ``scale`` argument of `drizzle.resample.Drizzle.add_image`.

The core resampling code does not depend on this module.

"""
import math
import warnings

import numpy as np

__all__ = [
    "FrameMismatchError",
    "FrameMismatchWarning",
    "calc_pixmap",
    "decode_context",
    "estimate_pixel_scale_ratio"
]

_DEG2RAD = math.pi / 180.0
_FRAME_CHECK_MODES = ("strict", "warn", "ignore")


class FrameMismatchError(ValueError):
    """
    Raised by `calc_pixmap` when ``frame_check="strict"`` and frame
    compatibility cannot be established, or frames differ and
    cannot be converted.
    """


class FrameMismatchWarning(UserWarning):
    """
    Issued by `calc_pixmap` when ``frame_check="warn"`` under the same
    conditions in which `FrameMismatchError` would be raised with
    ``frame_check="strict"``, except when neither WCS provides frame
    information. In that case the WCS objects are assumed to use the
    same frame and no warning is issued.
    """


def _axes_compatible(wcs_from, wcs_to):
    """
    Check that `wcs_from` and `wcs_to` have the same world axis order
    (physical types) and units for the fast path.

    """
    try:
        return (
            tuple(wcs_from.world_axis_physical_types)
            == tuple(wcs_to.world_axis_physical_types)
            and tuple(map(str, wcs_from.world_axis_units))
            == tuple(map(str, wcs_to.world_axis_units))
        )
    except AttributeError:
        # If either WCS object does not have the required attributes,
        # assume they are compatible for the fast path.
        return True


def _process_frame_mismatch(msg, frame_check):
    if frame_check == "strict":
        raise FrameMismatchError(msg)
    elif frame_check == "warn":
        warnings.warn(
            f"{msg}. World coordinate values are passed without frame "
            "conversion; the pixel map may be inaccurate.",
            FrameMismatchWarning,
            stacklevel=3,
        )


def _get_celestial_frames(wcs):
    """
    Return a list of celestial frames declared via APE 14
    ``world_axis_object_classes`` (one per ``SkyCoord`` world object;
    `None` entries for celestial objects without a declared frame),
    or `None` if the WCS does not provide ``world_axis_object_classes``.
    """
    classes = getattr(wcs, "world_axis_object_classes", None)
    if classes is None:
        return None

    frames = []
    for entry in classes.values():
        cls = entry[0]
        kwargs = entry[2] if len(entry) > 2 else None
        name = cls if isinstance(cls, str) else getattr(cls, "__name__", "")
        if name.rsplit(".", 1)[-1] != "SkyCoord":
            continue  # non-celestial axes have no frame
        frames.append(kwargs.get("frame") if isinstance(kwargs, dict) else None)
    return frames


def _compare_frames(wcs_from, wcs_to, strict=False):
    """
    Determine whether a frame conversion is needed.

    Parameters
    ----------
    strict : bool, optional
        If `True`, the case when neither WCS provides frame information is
        reported as "cannot be determined" instead of being assumed
        compatible (legacy behavior).

    Returns
    -------
    convert : bool or None
        `False` - frames are known to be equivalent (or, when ``strict`` is
        `False`, neither WCS provides frame information), use the fast
        numeric path;
        `True` - frames (or world axis order/units) are known to differ,
        conversion required;
        `None` - compatibility cannot be determined.
    reason : str or None
        Explanation when ``convert`` is `None`.
    """
    f_from = _get_celestial_frames(wcs_from)
    f_to = _get_celestial_frames(wcs_to)

    if f_from is None and f_to is None:
        if strict:
            return None, (
                "neither 'wcs_from' nor 'wcs_to' provides "
                "'world_axis_object_classes', so their celestial frames "
                "cannot be verified to be equivalent"
            )
        # legacy: assume same frame
        return False, None

    if f_from is None or f_to is None:
        which = "wcs_from" if f_from is None else "wcs_to"
        return None, (
            f"'{which}' does not provide 'world_axis_object_classes', so its "
            "celestial frame cannot be compared with that of the other WCS"
        )

    if len(f_from) != len(f_to):
        return None, (
            "'wcs_from' and 'wcs_to' have different numbers of celestial "
            f"world axis objects ({len(f_from)} vs {len(f_to)})"
        )

    convert = False
    for a, b in zip(f_from, f_to):
        if a is None or b is None:
            return None, "celestial frame is not defined for 'wcs_from' and/or 'wcs_to'"
        if isinstance(a, str) and isinstance(b, str):
            equivalent = a.lower() == b.lower()
        else:
            try:
                equivalent = a.is_equivalent_frame(b)
            except (AttributeError, TypeError):
                return None, f"unable to compare celestial frames {a!r} and {b!r}"
        if not equivalent:
            convert = True

    if not convert and not _axes_compatible(wcs_from, wcs_to):
        # same frame, but axis order or units differ
        convert = True

    return convert, None


def _get_bbox(wcs):
    if (bbox := getattr(wcs, "bounding_box", None)) is not None:
        try:
            # to avoid dependency on astropy just to check whether
            # the bounding box is an instance of
            # modeling.bounding_box.ModelBoundingBox, we try to
            # directly use and bounding_box(order='F') and if it fails,
            # fall back to converting the bounding box to a tuple
            # (of intervals):
            bbox = bbox.bounding_box(order="F")
        except AttributeError:
            bbox = tuple(bbox)
    return bbox


def calc_pixmap(wcs_from, wcs_to, shape=None, disable_bbox="to", frame_check="warn"):
    """
    Calculate a discretized on a grid mapping between the pixels of two images
    using provided WCS of the original ("from") image and the destination ("to")
    image.

    Parameters
    ----------
    wcs_from : wcs
        A WCS object representing the coordinate system you are
        converting from. This object's ``array_shape`` (or ``pixel_shape``)
        property will be used to define the shape of the pixel map array.
        If ``shape`` parameter is provided, it will take precedence
        over this object's ``array_shape`` value.

    wcs_to : wcs
        A WCS object representing the coordinate system you are
        converting to.

    shape : tuple, None, optional
        A tuple of integers indicating the shape of the output array in the
        ``numpy.ndarray`` order. When provided, it takes precedence over the
        ``wcs_from.array_shape`` property.

    disable_bbox : {"to", "from", "both", "none"}, optional
        Indicates whether to use or not to use the bounding box of either
        (both) ``wcs_from`` or (and) ``wcs_to`` when computing pixel map. When
        ``disable_bbox`` is "none", pixel coordinates outside of the bounding
        box are set to `NaN` only if ``wcs_from`` or (and) ``wcs_to`` sets
        world coordinates to NaN when input pixel coordinates are outside of
        the bounding box.

    frame_check : {"strict", "warn", "ignore"}, optional
        Controls how celestial frame compatibility of ``wcs_from`` and
        ``wcs_to`` is handled:

        - ``"strict"``: frames must be verified equivalent or converted;
          otherwise raise `FrameMismatchError`.
        - ``"warn"``: convert when frames are known to differ; otherwise
          issue a `FrameMismatchWarning` when compatibility cannot be verified.
        - ``"ignore"``: no frame checks (legacy behavior).

        See Notes for the full behavior table.

    Returns
    -------
    pixmap : numpy.ndarray
        A three dimensional array representing the transformation between
        the two. The last dimension is of length two and contains the x and
        y coordinates of a pixel center, respectively. The other two
        coordinates correspond to the two coordinates of the image the
        first WCS is from.

    Raises
    ------
    ValueError
        If the pixel map shape cannot be determined, or if ``frame_check`` has
        an invalid value.

    FrameMismatchError
        If ``frame_check="strict"`` and frame compatibility cannot be established.

    Notes
    -----
    When ``shape`` is not provided and ``wcs_from.array_shape`` is not set
    (i.e., it is `None`), `calc_pixmap` will attempt to determine pixel map
    shape from the ``bounding_box`` property of the input ``wcs_from`` object.
    If ``bounding_box`` is not available, a `ValueError` will be raised.

    **Frame handling.** Frames are determined from the APE 14
    ``world_axis_object_classes`` property of the WCS objects. The
    "fast path" passes world coordinate *values* computed by ``wcs_from``
    directly to ``wcs_to`` (no frame conversion). "Convert" uses the
    APE 14 high-level ``pixel_to_world``/``world_to_pixel`` methods, so
    the WCS objects themselves perform the frame conversion.

    .. list-table::
       :header-rows: 1
       :widths: 45 20 20 15

       * - Situation
         - ``"strict"``
         - ``"warn"``
         - ``"ignore"``
       * - Frames are known and equivalent (or neither WCS has
           celestial axes), and world axis order and units match
         - Fast path
         - Fast path
         - Fast path
       * - Frames (or world axis order/units) are known to differ;
           high-level API available
         - Convert
         - Convert
         - Fast path (no conversion)
       * - Frames differ; high-level API missing
         - `FrameMismatchError`
         - `FrameMismatchWarning`, then fast path
         - Fast path
       * - Frame compatibility cannot be determined: only one WCS
           provides ``world_axis_object_classes``, a celestial axis
           has no frame, frames cannot be compared, or the numbers
           of celestial axes differ
         - `FrameMismatchError`
         - `FrameMismatchWarning`, then fast path
         - Fast path
       * - Neither WCS provides ``world_axis_object_classes``
           (e.g., duck-typed WCS objects)
         - `FrameMismatchError`
         - Fast path (no warning)
         - Fast path

    Whenever the fast path is used, both WCS must use the same celestial
    frame, world axis order, and units. A celestial world axis object
    without a declared frame is treated as "cannot be determined"; it is
    not assumed to be ICRS.

    """
    if frame_check not in _FRAME_CHECK_MODES:
        raise ValueError(
            f"'frame_check' must be one of {_FRAME_CHECK_MODES}; "
            f"got {frame_check!r}"
        )

    # Frame check is done *before* bounding boxes are modified so that an
    # exception cannot leave the WCS objects in an altered state.
    convert = False
    if frame_check != "ignore":
        convert, reason = _compare_frames(
            wcs_from, wcs_to, strict=(frame_check == "strict")
        )

        if convert is None:
            # frames unknown / cannot be compared
            _process_frame_mismatch(reason, frame_check)
            convert = False

        elif convert and not (
            hasattr(wcs_from, "pixel_to_world") and hasattr(wcs_to, "world_to_pixel")
        ):
            # frames known to differ but cannot be converted
            _process_frame_mismatch(
                "celestial frames (or world axis order/units) of 'wcs_from' "
                "and 'wcs_to' differ, but the WCS objects do not support the "
                "APE 14 high-level API needed to convert between them",
                frame_check,
            )
            convert = False

    bbox_from = _get_bbox(wcs_from)
    bbox_to = _get_bbox(wcs_to)

    if shape is None:
        shape = wcs_from.array_shape
        if shape is None and bbox_from is not None:
            if (ndim := np.ndim(bbox_from)) == 1:
                bbox_from = (bbox_from,)
            if ndim > 1:
                shape = tuple(math.ceil(lim[1] + 0.5) for lim in bbox_from[::-1])

    if shape is None:
        raise ValueError('The "from" WCS must have pixel_shape property set.')

    y, x = np.indices(shape, dtype=np.float64)

    # temporarily disable the bounding box for the "from" WCS:
    if disable_bbox in ["from", "both"] and bbox_from is not None:
        wcs_from.bounding_box = None
    if disable_bbox in ["to", "both"] and bbox_to is not None:
        wcs_to.bounding_box = None
    try:
        if convert:
            # high-level APE 14 path: frame conversion is done by the
            # WCS objects themselves (no astropy import in drizzle):
            world = wcs_from.pixel_to_world(x, y)
            if not isinstance(world, (list, tuple)):
                world = (world,)
            x, y = wcs_to.world_to_pixel(*world)
        else:
            x, y = wcs_to.world_to_pixel_values(*wcs_from.pixel_to_world_values(x, y))
    finally:
        if bbox_from is not None:
            wcs_from.bounding_box = bbox_from
        if bbox_to is not None:
            wcs_to.bounding_box = bbox_to

    pixmap = np.dstack([x, y])
    return pixmap


def estimate_pixel_scale_ratio(wcs_from, wcs_to, refpix_from=None, refpix_to=None):
    """
    Compute the ratio of the pixel scale of the "to" WCS at the ``refpix_to``
    position to the pixel scale of the "from" WCS at the ``refpix_from``
    position. Pixel scale ratio,
    when requested, is computed near the centers of the bounding box
    (a property of the WCS object) or near ``refpix_*`` coordinates
    if supplied.

    Pixel scale is estimated as the square root of pixel's area, i.e.,
    pixels are assumed to have a square shape at the reference
    pixel position. If input reference pixel position for a WCS is `None`,
    it will be taken as the center of the bounding box
    if ``wcs_*`` has a bounding box defined, or as the center of the box
    defined by the ``pixel_shape`` attribute of the input WCS if
    ``pixel_shape`` is defined (not `None`), or at pixel coordinates
    ``(0, 0)``.

    Parameters
    ----------
    wcs_from : wcs
        A WCS object representing the coordinate system you are
        converting from. This object *must* have ``pixel_shape`` property
        defined.

    wcs_to : wcs
        A WCS object representing the coordinate system you are
        converting to.

    refpix_from : numpy.ndarray, tuple, list
        Image coordinates of the reference pixel near which pixel scale should
        be computed in the "from" image. In FITS WCS this could be, for example,
        the value of CRPIX of the ``wcs_from`` WCS.

    refpix_to : numpy.ndarray, tuple, list
        Image coordinates of the reference pixel near which pixel scale should
        be computed in the "to" image. In FITS WCS this could be, for example,
        the value of CRPIX of the ``wcs_to`` WCS.

    Returns
    -------
    pixel_scale_ratio : float
        Estimate the ratio of "to" to "from" WCS pixel scales. This value is
        returned only when ``estimate_pixel_scale_ratio`` is `True`.

    """
    pscale_ratio = _estimate_pixel_scale(wcs_to, refpix_to) / _estimate_pixel_scale(
        wcs_from, refpix_from
    )
    return pscale_ratio


def _estimate_pixel_scale(wcs, refpix):
    # estimate pixel scale (in rad) using approximate algorithm
    # from https://trs.jpl.nasa.gov/handle/2014/40409
    if refpix is None:
        if hasattr(wcs, "bounding_box") and wcs.bounding_box is not None:
            refpix = np.mean(wcs.bounding_box, axis=-1)
        else:
            if wcs.pixel_shape:
                refpix = np.array([(i - 1) // 2 for i in wcs.pixel_shape])
            else:
                refpix = np.zeros(wcs.pixel_n_dim)

    else:
        refpix = np.asarray(refpix)

    l1, phi1 = wcs.pixel_to_world_values(*(refpix - 0.5))
    l2, phi2 = wcs.pixel_to_world_values(*(refpix + [-0.5, 0.5]))
    l3, phi3 = wcs.pixel_to_world_values(*(refpix + 0.5))
    l4, phi4 = wcs.pixel_to_world_values(*(refpix + [0.5, -0.5]))
    area = _DEG2RAD * abs(
        0.5
        * (
            (l4 - l2) * (math.sin(_DEG2RAD * phi1) - math.sin(_DEG2RAD * phi3))
            + (l1 - l3) * (math.sin(_DEG2RAD * phi2) - math.sin(_DEG2RAD * phi4))
        )
    )
    return math.sqrt(area)


def decode_context(context, x, y):
    """Get 0-based indices of input images that contributed to (resampled)
    output pixel with coordinates ``x`` and ``y``.

    Parameters
    ----------
    context: numpy.ndarray
        A 3D `~numpy.ndarray` of integral data type.

    x: int, list of integers, numpy.ndarray of integers
        X-coordinate of pixels to decode (3rd index into the ``context`` array)

    y: int, list of integers, numpy.ndarray of integers
        Y-coordinate of pixels to decode (2nd index into the ``context`` array)

    Returns
    -------
    A list of `numpy.ndarray` objects each containing indices of input images
    that have contributed to an output pixel with coordinates ``x`` and ``y``.
    The length of returned list is equal to the number of input coordinate
    arrays ``x`` and ``y``.

    Examples
    --------
    An example context array for an output image of array shape ``(5, 6)``
    obtained by resampling 80 input images.

    >>> import numpy as np
    >>> from drizzle.utils import decode_context
    >>> ctx = np.array(
    ...     [[[0, 0, 0, 0, 0, 0],
    ...       [0, 0, 0, 36196864, 0, 0],
    ...       [0, 0, 0, 0, 0, 0],
    ...       [0, 0, 0, 0, 0, 0],
    ...       [0, 0, 537920000, 0, 0, 0]],
    ...      [[0, 0, 0, 0, 0, 0,],
    ...       [0, 0, 0, 67125536, 0, 0],
    ...       [0, 0, 0, 0, 0, 0],
    ...       [0, 0, 0, 0, 0, 0],
    ...       [0, 0, 163856, 0, 0, 0]],
    ...      [[0, 0, 0, 0, 0, 0],
    ...       [0, 0, 0, 8203, 0, 0],
    ...       [0, 0, 0, 0, 0, 0],
    ...       [0, 0, 0, 0, 0, 0],
    ...       [0, 0, 32865, 0, 0, 0]]],
    ...     dtype=np.int32
    ... )
    >>> decode_context(ctx, [3, 2], [1, 4])
    [array([ 9, 12, 14, 19, 21, 25, 37, 40, 46, 58, 64, 65, 67, 77]),
     array([ 9, 20, 29, 36, 47, 49, 64, 69, 70, 79])]

    """
    if context.ndim != 3:
        raise ValueError("'context' must be a 3D array.")

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    if x.size != y.size:
        raise ValueError("Coordinate arrays must have equal length.")

    if x.ndim != 1:
        raise ValueError("Coordinates must be scalars or 1D arrays.")

    if not (np.issubdtype(x.dtype, np.integer) and np.issubdtype(y.dtype, np.integer)):
        raise ValueError("Pixel coordinates must be integer values")

    nbits = 8 * context.dtype.itemsize
    one = np.array(1, context.dtype)
    flags = np.array([one << i for i in range(nbits)])

    idx = []
    for xi, yi in zip(x, y):
        idx.append(np.flatnonzero(np.bitwise_and.outer(context[:, yi, xi], flags)))

    return idx
