from __future__ import division, print_function, unicode_literals, absolute_import

# THIRD-PARTY
import numpy as np

# LOCAL
from . import util
from . import calc_pixmap
from . import cdrizzle

"""
STScI Python compatable drizzle module
"""

def dodrizzle(insci, input_wcs, inwht,
              output_wcs, outsci, outwht, outcon,
              expin, in_units, wt_scl,
              wcslin_pscale=1.0, uniqid=1,
              xmin=0, xmax=0, ymin=0, ymax=0,
              pixfrac=1.0, kernel='square', fillval="INDEF",
              stepsize=10, wcsmap=None):
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

    outcon : 2d array
        A 3d numpy array holding a bitmap of which image was an input
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
        on the output image. Only pixels on the output image inside this
        rectangle will have their flux updated. Xmin sets the minimum value
        of the x dimension. The x dimension is the dimension that varies
        quickest on the image. If the value is zero, no minimum will
        be set in the x dimension. All four parameters are zero based,
        counting starts at zero.
        
    xmax : float, optional
        Sets the maximum value of the x dimension on the bounding box
        of the ouput image. If the value is zero, no maximum will 
        be set in the x dimension, the full x dimension of the output
        image is the bounding box.

    ymin : float, optional
        Sets the minimum value in the y dimension on the bounding box. The
        y dimension varies less rapidly than the x and represents the line
        index on the output image. If the value is zero, no minimum 
        will be set in the y dimension.
        
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
    on the output image that were not covered by the input image, and the
    number of complete lines on the output image that were not covered by
    the input input image.
    
    
    Other Parameters
    ----------------
    
    stepsize
        Was used when input to output mapping was computed
        inside the cdrizzle code. Is no longer used and only here for
        backwards compatibility. It may be re-used in the future if we
        need to use a subsampled pixel map to improve speed.

    wcsmap
        Was used when input to output mapping was computed
        inside the cdrizzle code. Is no longer used and only here for
        backwards compatibility.
    """
    
    # Insure that the fillval parameter gets properly interpreted for use with tdriz
    if util.is_blank(fillval):
        fillval = 'INDEF'
    else:
        fillval = str(fillval)

    if in_units == 'cps':
        expscale = 1.0
    else:
        expscale = expin

    # Compute what plane of the context image this input would
    # correspond to:
    _planeid = int((uniqid-1) /32)

    # Compute how many planes will be needed for the context image.
    _nplanes = _planeid + 1

    if outcon is not None and (outcon.ndim < 3 or (outcon.ndim == 3 and
                                                   outcon.shape[0] < _nplanes)):
        
        # convert context image to 3-D array and pass along correct plane for drizzling
        if outcon.ndim == 3:
            nplanes = outcon.shape[0]+1
        else:
            nplanes = 1
            
        # We need to expand the context image here to accomodate the addition of
        # this new image
        newcon = np.zeros((nplanes,output_wcs._naxis2,output_wcs._naxis1),dtype=np.int32)

        # now copy original outcon arrays into new array
        if outcon.ndim == 3:
            for n in range(outcon.shape[0]):
                newcon[n] = outcon[n].copy()
        else:
            newcon[0] = outcon.copy()
    else:
        if outcon is None:
            outcon = np.zeros((1,output_wcs._naxis2,output_wcs._naxis1),dtype=np.int32)
            _planeid = 0
        newcon = outcon

    # At this point, newcon will always be a 3-D array, so only pass in
    # correct plane to drizzle code
    outctx = newcon[_planeid]

    pix_ratio = output_wcs.pscale/wcslin_pscale

    # Compute the mapping between the input and output pixel coordinates
    pixmap = calc_pixmap.calc_pixmap(input_wcs, output_wcs)

    #
    # Call 'drizzle' to perform image combination
    # This call to 'cdriz.tdriz' uses the new C syntax
    # 
    if (insci.dtype > np.float32):
        insci = insci.astype(np.float32)

    _vers, nmiss, nskip = cdrizzle.tdriz(
        insci, inwht, pixmap, outsci, outwht, outctx, 
        uniqid=uniqid, xmin=xmin, xmax=xmax,
        ymin=ymin, ymax=ymax, scale=pix_ratio, pixfrac=pixfrac,
        kernel=kernel, in_units=in_units, expscale=expscale, 
        wtscale=wt_scl, fillstr=fillval)

    return _vers, nmiss, nskip
