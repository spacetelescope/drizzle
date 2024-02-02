"""
STScI Python compatable drizzle module
"""
import numpy as np

from . import util
from . import calc_pixmap
from . import cdrizzle


def dodrizzle(insci, input_wcs, inwht,
              output_wcs, outsci, outwht, outcon,
              expin, in_units, wt_scl,
              wcslin_pscale=1.0, uniqid=1,
              xmin=0, xmax=0, ymin=0, ymax=0,
              pixfrac=1.0, kernel='square', fillval="INDEF"):
    """
    Low level routine for performing 'drizzle' operation.on one image.

    The interface is compatible with STScI code. All images are Python
    ndarrays, instead of filenames. File handling (input and output) is
    performed by the calling routine.

    Parameters
    ----------

    insci : 2d array
        A 2d numpy array containing the input image to be drizzled.
        it is an error to not supply an image.

    input_wcs : 2d array
        The world coordinate system of the input image.

    inwht : 2d array
        A 2d numpy array containing the pixel by pixel weighting.
        Must have the same dimensions as insci. If none is supplied,
        the weghting is set to one.

    output_wcs : wcs
        The world coordinate system of the output image.

    outsci : 2d array
        A 2d numpy array containing the output image produced by
        drizzling. On the first call it should be set to zero.
        Subsequent calls it will hold the intermediate results

    outwht : 2d array
        A 2d numpy array containing the output counts. On the first
        call it should be set to zero. On subsequent calls it will
        hold the intermediate results.

    outcon : 2d or 3d array, optional
        A 2d or 3d numpy array holding a bitmap of which image was an input
        for each output pixel. Should be integer zero on first call.
        Subsequent calls hold intermediate results.

    expin : float
        The exposure time of the input image, a positive number. The
        exposure time is used to scale the image if the units are counts.

    in_units : str
        The units of the input image. The units can either be "counts"
        or "cps" (counts per second.)

    wt_scl : float
        A scaling factor applied to the pixel by pixel weighting.

    wcslin_pscale : float, optional
        The pixel scale of the input image. Conceptually, this is the
        linear dimension of a side of a pixel in the input image, but it
        is not limited to this and can be set to change how the drizzling
        algorithm operates.

    uniqid : int, optional
        The id number of the input image. Should be one the first time
        this function is called and incremented by one on each subsequent
        call.

    xmin : float, optional
        This and the following three parameters set a bounding rectangle
        on the input image. Only pixels on the input image inside this
        rectangle will have their flux added to the output image. Xmin
        sets the minimum value of the x dimension. The x dimension is the
        dimension that varies quickest on the image. If the value is zero,
        no minimum will be set in the x dimension. All four parameters are
        zero based, counting starts at zero.

    xmax : float, optional
        Sets the maximum value of the x dimension on the bounding box
        of the input image. If the value is zero, no maximum will
        be set in the x dimension, the full x dimension of the output
        image is the bounding box.

    ymin : float, optional
        Sets the minimum value in the y dimension on the bounding box. The
        y dimension varies less rapidly than the x and represents the line
        index on the input image. If the value is zero, no minimum  will be
        set in the y dimension.

    ymax : float, optional
        Sets the maximum value in the y dimension. If the value is zero, no
        maximum will be set in the y dimension,  the full x dimension
        of the output image is the bounding box.

    pixfrac : float, optional
        The fraction of a pixel that the pixel flux is confined to. The
        default value of 1 has the pixel flux evenly spread across the image.
        A value of 0.5 confines it to half a pixel in the linear dimension,
        so the flux is confined to a quarter of the pixel area when the square
        kernel is used.

    kernel: str, optional
        The name of the kernel used to combine the input. The choice of
        kernel controls the distribution of flux over the kernel. The kernel
        names are: "square", "gaussian", "point", "tophat", "turbo", "lanczos2",
        and "lanczos3". The square kernel is the default.

    fillval: str, optional
        The value a pixel is set to in the output if the input image does
        not overlap it. The default value of INDEF does not set a value.

    Returns
    -------
    A tuple with three values: a version string, the number of pixels
    on the input image that do not overlap the output image, and the
    number of complete lines on the input image that do not overlap the
    output input image.

    """

    # Ensure that the fillval parameter gets properly interpreted
    # for use with tdriz
    if util.is_blank(fillval):
        fillval = 'INDEF'
    else:
        fillval = str(fillval)

    if in_units == 'cps':
        expscale = 1.0
    else:
        expscale = expin

    # Add input weight image if it was not passed in

    if (insci.dtype > np.float32):
        insci = insci.astype(np.float32)

    if inwht is None:
        inwht = np.ones_like(insci)

    # Compute what plane of the context image this input would
    # correspond to:
    planeid = int((uniqid - 1) / 32)

    # Check if the context image has this many planes
    if outcon.ndim == 3:
        nplanes = outcon.shape[0]
    elif outcon.ndim == 2:
        nplanes = 1
    else:
        nplanes = 0

    if nplanes <= planeid:
        raise IndexError("Not enough planes in drizzle context image")

    # Alias context image to the requested plane if 3d
    if outcon.ndim == 3:
        outcon = outcon[planeid]

    pix_ratio = output_wcs.pscale / wcslin_pscale

    # Compute the mapping between the input and output pixel coordinates
    pixmap = calc_pixmap.calc_pixmap(input_wcs, output_wcs)

    #
    # Call 'drizzle' to perform image combination
    # This call to 'cdriz.tdriz' uses the new C syntax
    #
    _vers, nmiss, nskip = cdrizzle.tdriz(
        insci, inwht, pixmap, outsci, outwht, outcon,
        uniqid=uniqid, xmin=xmin, xmax=xmax,
        ymin=ymin, ymax=ymax, scale=pix_ratio, pixfrac=pixfrac,
        kernel=kernel, in_units=in_units, expscale=expscale,
        wtscale=wt_scl, fillstr=fillval)

    return _vers, nmiss, nskip
