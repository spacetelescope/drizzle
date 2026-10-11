import math

import numpy as np

__all__ = ["calc_pixmap", "decode_context", "estimate_pixel_scale_ratio"]

_DEG2RAD = math.pi / 180.0


def _get_bbox_intervals(wcs):
    """Return the bounding box of ``wcs`` as a tuple of (min, max) intervals
    in pixel-axis (Fortran) order, or None if there is no bounding box."""
    if (bbox := getattr(wcs, "bounding_box", None)) is None:
        return None

    # to avoid dependency on astropy just to check whether the bounding box
    # is an instance of modeling.bounding_box.ModelBoundingBox, we try to
    # directly use bounding_box(order='F') and if it fails, fall back to
    # converting the bounding box to a tuple (of intervals):
    try:
        bbox = bbox.bounding_box(order="F")
    except AttributeError:
        bbox = tuple(bbox)

    return bbox


def calc_pixmap(wcs_from, wcs_to, shape=None, disable_bbox="to"):
    """
    Calculate a discretized on a grid mapping between the pixels of two images
    using provided WCS of the original ("from") image and the destination ("to")
    image.

    .. note::
       This function assumes that output frames of ``wcs_from`` and ``wcs_to``
       WCS have the same units.

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

    Returns
    -------
    pixmap : numpy.ndarray
        A three dimensional array representing the transformation between
        the two. The last dimension is of length two and contains the x and
        y coordinates of a pixel center, respectively. The other two coordinates
        correspond to the two coordinates of the image the first WCS is from.

    Raises
    ------
    ValueError
        A `ValueError` is raised when output pixel map shape cannot be
        determined from provided inputs.

    Notes
    -----
    When ``shape`` is not provided and ``wcs_from.array_shape`` is not set
    (i.e., it is `None`), `calc_pixmap` will attempt to determine pixel map
    shape from the ``bounding_box`` property of the input ``wcs_from`` object.
    If ``bounding_box`` is not available, a `ValueError` will be raised.

    """
    orig_bbox_from = getattr(wcs_from, "bounding_box", None)
    orig_bbox_to = getattr(wcs_to, "bounding_box", None)
    bbox_from = _get_bbox_intervals(wcs_from)

    if shape is None:
        shape = wcs_from.array_shape
        if shape is None and bbox_from is not None and np.ndim(bbox_from) > 1:
            shape = tuple(math.ceil(lim[1] + 0.5) for lim in bbox_from[::-1])

    if shape is None:
        raise ValueError(
            "Cannot determine pixel map shape: pass 'shape' or use a 'from' "
            "WCS with 'array_shape' or 'bounding_box' set."
        )
    y, x = np.indices(shape, dtype=np.float64)

    # temporarily disable bounding boxes as requested:
    disable_from = disable_bbox in ("from", "both") and orig_bbox_from is not None
    disable_to = disable_bbox in ("to", "both") and orig_bbox_to is not None
    if disable_from:
        wcs_from.bounding_box = None
    if disable_to:
        wcs_to.bounding_box = None

    try:
        x, y = wcs_to.world_to_pixel_values(*wcs_from.pixel_to_world_values(x, y))
    finally:
        # restore original bounding boxes if they were temporarily disabled
        if disable_from:
            wcs_from.bounding_box = orig_bbox_from
        if disable_to:
            wcs_to.bounding_box = orig_bbox_to

    pixmap = np.dstack([x, y])
    return pixmap


def estimate_pixel_scale_ratio(wcs_from, wcs_to, refpix_from=None, refpix_to=None):
    """
    Compute the ratio of the pixel scale of the "to" WCS at the ``refpix_to``
    position to the pixel scale of the "from" WCS at the ``refpix_from``
    position. The pixel scale ratio is computed near the centers of the
    bounding box (a property of the WCS object) or near ``refpix_*``
    coordinates if supplied.

    Pixel scale is estimated as the square root of the pixel's area on the
    sky, i.e., pixels are assumed to have a square shape at the reference
    pixel position. If the reference pixel position for a WCS is `None`,
    it will be taken as the center of the bounding box if ``wcs_*`` has a
    bounding box defined, or as the center of the box defined by the
    ``pixel_shape`` attribute of the input WCS if ``pixel_shape`` is defined
    (not `None`), or at pixel coordinates ``(0, 0)``.

    Parameters
    ----------
    wcs_from : wcs
        A WCS object representing the coordinate system you are
        converting from. Must be a 2D celestial WCS whose
        ``pixel_to_world_values`` returns ``(longitude, latitude)`` in
        **degrees** (see Notes).

    wcs_to : wcs
        A WCS object representing the coordinate system you are
        converting to. Same requirements as ``wcs_from``.

    refpix_from : numpy.ndarray, tuple, list, None, optional
        Image coordinates of the reference pixel near which pixel scale should
        be computed in the "from" image. In FITS WCS this could be, for example,
        the value of CRPIX of the ``wcs_from`` WCS.

    refpix_to : numpy.ndarray, tuple, list, None, optional
        Image coordinates of the reference pixel near which pixel scale should
        be computed in the "to" image. In FITS WCS this could be, for example,
        the value of CRPIX of the ``wcs_to`` WCS.

    Returns
    -------
    pixel_scale_ratio : float
        Estimate of the ratio of "to" to "from" WCS pixel scales.

    Raises
    ------
    ValueError
        If either WCS is not two-dimensional, or if it reports world axis
        units other than degrees.

    Notes
    -----
    This function assumes that both WCS objects describe celestial
    coordinates and that ``pixel_to_world_values`` returns longitude and
    latitude **in degrees**. This is always the case for
    `astropy.wcs.WCS` celestial axes and for `gwcs.WCS` objects whose output
    frame is a `~gwcs.coordinate_frames.CelestialFrame` with default units
    (as produced, for example, by the JWST and Roman pipelines). If a WCS
    object exposes the ``world_axis_units`` attribute (APE 14) and reports
    units other than degrees, a `ValueError` is raised.

    For WCS objects that do not meet these requirements, compute the pixel
    scale ratio by other means and pass it directly to
    `drizzle.resample.Drizzle.add_image` via its ``scale`` argument.

    """
    pscale_ratio = _estimate_pixel_scale(wcs_to, refpix_to) / _estimate_pixel_scale(
        wcs_from, refpix_from
    )
    return pscale_ratio


def _estimate_pixel_scale(wcs, refpix):
    # estimate pixel scale (in rad) using planar projection
    if refpix is None:
        if (bbox := _get_bbox_intervals(wcs)) is not None:
            refpix = np.mean(bbox, axis=-1, dtype=float)
        elif getattr(wcs, "pixel_shape", None):
            refpix = np.array([0.5 * (i - 1) for i in wcs.pixel_shape], dtype=float)
        else:
            refpix = np.zeros(wcs.pixel_n_dim, dtype=float)

        if refpix.shape != (2,):
            raise ValueError("Input WCS must be a 2D WCS")

    else:
        refpix = np.asarray(refpix, dtype=float)
        if refpix.shape != (2,):
            raise ValueError("'refpix' must be of length 2 for 2D WCS")

    units = getattr(wcs, "world_axis_units", None)
    if (units is not None and
            any(str(unit).strip().lower() not in ("deg", "degree") for unit in units)):
        raise ValueError(
            f"World axis units {tuple(units)} are not supported; pixel scale "
            "estimation assumes celestial coordinates in degrees."
        )
    # else if units is None - assume they are in degrees

    # project to a tangent plane near the pixel and compute planar area:
    refx1 = refpix[0] - 0.5
    refx2 = refpix[0] + 0.5
    refy1 = refpix[1] - 0.5
    refy2 = refpix[1] + 0.5

    lon, lat = wcs.pixel_to_world_values(
        [refx1, refx1, refx2, refx2],
        [refy1, refy2, refy2, refy1],
    )
    lon = _DEG2RAD * np.asarray(lon, dtype=float)
    lat = _DEG2RAD * np.asarray(lat, dtype=float)

    # unit vectors of the four pixel corners on the celestial sphere.
    cs = np.cos(lat)
    x = cs * np.cos(lon)
    y = cs * np.sin(lon)
    z = np.sin(lat)
    v = np.stack([x, y, z], axis=1)

    # area of the (tiny) spherical quadrilateral ~ area of the planar
    # quadrilateral through its corners, from the cross product of its
    # diagonals: A = |d1 x d2| / 2. The plane is defined by the two
    # diagonals.
    area = 0.5 * np.linalg.norm(np.cross(v[2] - v[0], v[3] - v[1]))
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
