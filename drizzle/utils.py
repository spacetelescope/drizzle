import math

import numpy as np

__all__ = ["calc_pixmap", "decode_context", "estimate_pixel_scale_ratio"]

_DEG2RAD = math.pi / 180.0


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
        y coordinates of a pixel center, repectively. The other two coordinates
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
    if (bbox_from := getattr(wcs_from, "bounding_box", None)) is not None:
        try:
            # to avoid dependency on astropy just to check whether
            # the bounding box is an instance of
            # modeling.bounding_box.ModelBoundingBox, we try to
            # directly use and bounding_box(order='F') and if it fails,
            # fall back to converting the bounding box to a tuple
            # (of intervals):
            bbox_from = bbox_from.bounding_box(order='F')
        except AttributeError:
            bbox_from = tuple(bbox_from)

    if (bbox_to := getattr(wcs_to, "bounding_box", None)) is not None:
        try:
            # to avoid dependency on astropy just to check whether
            # the bounding box is an instance of
            # modeling.bounding_box.ModelBoundingBox, we try to
            # directly use and bounding_box(order='F') and if it fails,
            # fall back to converting the bounding box to a tuple
            # (of intervals):
            bbox_to = bbox_to.bounding_box(order='F')
        except AttributeError:
            bbox_to = tuple(bbox_to)

    if shape is None:
        shape = wcs_from.array_shape
        if shape is None and bbox_from is not None:
            if (nd := np.ndim(bbox_from)) == 1:
                bbox_from = (bbox_from, )
            if nd > 1:
                shape = tuple(
                    math.ceil(lim[1] + 0.5) for lim in bbox_from[::-1]
                )

    if shape is None:
        raise ValueError(
            'The "from" WCS must have pixel_shape property set.'
        )

    y, x = np.indices(shape, dtype=np.float64)

    # temporarily disable the bounding box for the "from" WCS:
    if disable_bbox in ["from", "both"] and bbox_from is not None:
        wcs_from.bounding_box = None
    if disable_bbox in ["to", "both"] and bbox_to is not None:
        wcs_to.bounding_box = None
    try:
        x, y = wcs_to.world_to_pixel_values(
            *wcs_from.pixel_to_world_values(x, y)
        )
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
    pscale_ratio = (_estimate_pixel_scale(wcs_to, refpix_to) /
                    _estimate_pixel_scale(wcs_from, refpix_from))
    return pscale_ratio


def _estimate_pixel_scale(wcs, refpix):
    # estimate pixel scale (in rad) using approximate algorithm
    # from https://trs.jpl.nasa.gov/handle/2014/40409
    if refpix is None:
        if hasattr(wcs, 'bounding_box') and wcs.bounding_box is not None:
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
        0.5 * (
            (l4 - l2) * (math.sin(_DEG2RAD * phi1) - math.sin(_DEG2RAD * phi3)) +
            (l1 - l3) * (math.sin(_DEG2RAD * phi2) - math.sin(_DEG2RAD * phi4))
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

    if not (np.issubdtype(x.dtype, np.integer) and
            np.issubdtype(y.dtype, np.integer)):
        raise ValueError('Pixel coordinates must be integer values')

    nbits = 8 * context.dtype.itemsize
    one = np.array(1, context.dtype)
    flags = np.array([one << i for i in range(nbits)])

    idx = []
    for xi, yi in zip(x, y):
        idx.append(
            np.flatnonzero(np.bitwise_and.outer(context[:, yi, xi], flags))
        )

    return idx
