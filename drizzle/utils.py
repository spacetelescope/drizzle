import numpy as np


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
    x, y = wcs_to.invert(*wcs_from(x, y))
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
        if wcs.bounding_box is None:
            refpix = np.ones(wcs.pixel_n_dim)
        else:
            refpix = np.mean(wcs.bounding_box, axis=-1)
    else:
        refpix = np.asarray(refpix)

    l1, phi1 = np.deg2rad(wcs.__call__(*(refpix - 0.5)))
    l2, phi2 = np.deg2rad(wcs.__call__(*(refpix + [-0.5, 0.5])))
    l3, phi3 = np.deg2rad(wcs.__call__(*(refpix + 0.5)))
    l4, phi4 = np.deg2rad(wcs.__call__(*(refpix + [0.5, -0.5])))
    area = np.abs(0.5 * ((l4 - l2) * (np.sin(phi1) - np.sin(phi3)) +
                         (l1 - l3) * (np.sin(phi2) - np.sin(phi4))))
    return np.sqrt(area)
