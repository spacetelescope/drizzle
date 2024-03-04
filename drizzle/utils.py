import numpy as np

__all__ = ["calc_pixmap", "decode_context"]


def calc_pixmap(wcs_from, wcs_to, estimate_pixel_scale_ratio=False,
                refpix_from=None, refpix_to=None):
    """
    Calculate a mapping between the pixels of two images. Pixel scale ratio,
    when requested, is computed near the centers of the bounding box
    (a property of the WCS object) or near ``refpix_*`` coordinates
    if supplied.

    Parameters
    ----------
    wcs_from : wcs
        A WCS object representing the coordinate system you are
        converting from. This object *must* have ``pixel_shape`` property
        defined.

    wcs_to : wcs
        A WCS object representing the coordinate system you are
        converting to.

    estimate_pixel_scale_ratio : bool, optional
        Estimate the ratio of "to" to "from" WCS pixel scales.

        .. note::
            Pixel scale is estimated as the square root of pixel's area
            (i.e., pixels are assumed to have a square shape) at the reference
            pixel position which is taken as the center of the bounding box
            if ``wcs_*`` has a bounding box defined, or as the center of the box
            defined by the ``pixel_shape`` attribute of the input WCS if
            ``pixel_shape`` is defined (not `None`), or at pixel coordinates
            ``(0, 0)``.

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
    pixmap : numpy.ndarray
        A three dimensional array representing the transformation between
        the two. The last dimension is of length two and contains the x and
        y coordinates of a pixel center, repectively. The other two coordinates
        correspond to the two coordinates of the image the first WCS is from.

    pixel_scale_ratio : float
        Estimate the ratio of "to" to "from" WCS pixel scales. This value is
        returned only when ``estimate_pixel_scale_ratio`` is `True`.

    """
    # We add one to the pixel co-ordinates before the transformation and subtract
    # it afterwards because wcs co-ordinates are one based, while pixel co-ordinates
    # are zero based, The result is the final values in pixmap give a tranformation
    # between the pixel co-ordinates in the first image to pixel co-ordinates in the
    # co-ordinate system of the second.

    if wcs_from.pixel_shape is None:
        raise ValueError('The "from" WCS must have pixel_shape property set.')
    y, x = np.indices(wcs_from.pixel_shape, dtype=np.float64)
    x, y = wcs_to.world_to_pixel(wcs_from.pixel_to_world(x, y))

    pixmap = np.dstack([x, y])

    if estimate_pixel_scale_ratio:
        pscale_ratio = (_estimate_pixel_scale(wcs_to, refpix_to) /
                        _estimate_pixel_scale(wcs_from, refpix_from))
        return pixmap, pscale_ratio

    return pixmap


def _estimate_pixel_scale(wcs, refpix):
    # estimate pixel scale (in rad) using approximate algorithm
    # from https://trs.jpl.nasa.gov/handle/2014/40409
    if refpix is None:
        if not hasattr(wcs, 'bounding_box') or wcs.bounding_box is None:
            if wcs.pixel_shape:
                refpix = np.array([(i - 1) // 2 for i in wcs.pixel_shape])
            else:
                refpix = np.zeros(wcs.pixel_n_dim)
        else:
            refpix = np.mean(wcs.bounding_box, axis=-1)
    else:
        refpix = np.asarray(refpix)

    l1, phi1 = np.deg2rad(wcs.pixel_to_world_values(*(refpix - 0.5)))
    l2, phi2 = np.deg2rad(wcs.pixel_to_world_values(*(refpix + [-0.5, 0.5])))
    l3, phi3 = np.deg2rad(wcs.pixel_to_world_values(*(refpix + 0.5)))
    l4, phi4 = np.deg2rad(wcs.pixel_to_world_values(*(refpix + [0.5, -0.5])))
    area = np.abs(0.5 * ((l4 - l2) * (np.sin(phi1) - np.sin(phi3)) +
                         (l1 - l3) * (np.sin(phi2) - np.sin(phi4))))
    return np.sqrt(area)


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
