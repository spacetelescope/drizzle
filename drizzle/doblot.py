"""
STScI Python compatable blot module
"""
import numpy as np

from . import calc_pixmap
from . import cdrizzle


def doblot(source, source_wcs, blot_wcs, exptime, coeffs=True,
           interp='poly5', sinscl=1.0, stepsize=10, wcsmap=None):
    """
    Low level routine for performing the 'blot' operation.

    Create a single blotted image from a single source image. The
    interface is compatible with STScI code. All distortion information
    is assumed to be included in the WCS specification of the 'output'
    blotted image given in 'blot_wcs'.

    Parameters
    ----------

    source : 2d array
        Input numpy array of the source image in units of 'cps'.

    source_wcs : wcs
        The source image WCS.

    blot_wcs : wcs
        The blotted image WCS. The WCS that the source image will be
        resampled to.

    exptime : float
        The exposure time of the input image.

    interp : str, optional
        The type of interpolation used in the blotting. The
        possible values are "nearest" (nearest neighbor interpolation),
        "linear" (bilinear interpolation), "poly3" (cubic polynomial
        interpolation), "poly5" (quintic polynomial interpolation),
        "sinc" (sinc interpolation), "lan3" (3rd order Lanczos
        interpolation), and "lan5" (5th order Lanczos interpolation).

    sincscl : float, optional
        The scaling factor for sinc interpolation.

    Returns
    -------

    A 2d numpy array with the blotted image

    Other Parameters
    ----------------

    coeffs : bool, optional
        Not used. Only kept for backwards compatibility.

    stepsize : float, optional
        Was used when input to output mapping was computed
        internally. Is no longer used and only here for backwards compatibility.

    wcsmap : function, optional
        Was used when input to output mapping was computed
        internally. Is no longer used and only here for backwards compatibility.
    """
    _outsci = np.zeros(blot_wcs.pixel_shape[::-1], dtype=np.float32)

    # compute the undistorted 'natural' plate scale
    blot_wcs.sip = None
    blot_wcs.cpdis1 = None
    blot_wcs.cpdis2 = None
    blot_wcs.det2im = None

    pixmap = calc_pixmap.calc_pixmap(blot_wcs, source_wcs)
    pix_ratio = source_wcs.pscale / blot_wcs.pscale

    cdrizzle.tblot(source, pixmap, _outsci, scale=pix_ratio, kscale=1.0,
                   interp=interp, exptime=exptime, misval=0.0, sinscl=sinscl)

    return _outsci
