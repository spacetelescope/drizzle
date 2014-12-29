from __future__ import division, print_function, unicode_literals, absolute_import

# THIRD-PARTY
import numpy as np

# LOCAL
from . import util
from . import cdrizzle
from . import calc_pixmap

def dodrizzle(insci, input_wcs, inwht,
              output_wcs, outsci, outwht, outcon,
              expin, in_units, wt_scl,
              wcslin_pscale=1.0, uniqid=1,
              xmin=0, xmax=0, ymin=0, ymax=0,
              pixfrac=1.0, kernel='square', fillval="INDEF",
              stepsize=10, wcsmap=None):
    """
    Low level routine for performing 'drizzle' operation on a single input
    image. Interface is compatible with STScI code. All images are Python
    ndarrays, instead of filenames. File handling (input and output) is
    performed by the calling routine.
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
