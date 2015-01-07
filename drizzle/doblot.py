from __future__ import division, print_function, unicode_literals, absolute_import

# THIRD-PARTY
import numpy as np

# LOCAL
from . import util
from . import cdrizzle
from . import calc_pixmap

def doblot(source, source_wcs, blot_wcs, exptime, coeffs = True,
            interp='poly5', sinscl=1.0, stepsize=10, wcsmap=None):
    """
    Core functionality of performing the 'blot' operation to create a single
    blotted image from a single source image. Interface is compatible with
    STScI code. All distortion information is assumed to be included in the
    WCS specification of the 'output' blotted image given in 'blot_wcs'.

    Parameters
    ----------
    source: Input numpy array of the source image in units of 'cps'.
    
    source_wcs: The source image WCS.
    
    blot_wcs: The blotted image WCS.
    
    exptime: The exposure time of the input image.
    
    coeffs: Not used. Only kept for backwards compatibility.
    
    interp: The type of interpolation used in the blotting. The
        possible values are "nearest" (nearest neighbor interpolation),
        "linear" (bilinear interpolation), "poly3" (cubic polynomial
        interpolation), "poly5" (quintic polynomial interpolation),
        "sinc" (sinc interpolation), "lan3" (3rd order Lanczos
        interpolation), and "lan5" (5th order Lanczos interpolation).

    sincscl: The scaling factor for sinc interpolation.

    stepsize: Was used when input to output mapping was computed
    internally. Is no longer used and only here for backwards compatibility.

    wcsmap: Was used when input to output mapping was computed
    internally. Is no longer used and only here for backwards compatibility.
    
    Returns
    -------
    A 2d numpy array with the blotted image
    """
    _outsci = np.zeros((blot_wcs._naxis2,blot_wcs._naxis1),dtype=np.float32)

    # compute the undistorted 'natural' plate scale 
    wcslin = blot_wcs
    blot_wcs.sip = None
    blot_wcs.cpdis1 = None
    blot_wcs.cpdis2 = None
    blot_wcs.det2im = None

    pixmap = calc_pixmap.calc_pixmap(blot_wcs, source_wcs)
    pix_ratio = source_wcs.pscale/blot_wcs.pscale

    cdrizzle.tblot(source, pixmap, _outsci, scale=pix_ratio, kscale=1.0,
                   interp=interp, exptime=exptime, misval=0.0, sinscl=sinscl)

    return _outsci
