from __future__ import division, print_function, unicode_literals, absolute_import

# SYSTEM
import os
import os.path

# THIRD-PARTY
import numpy as np
from astropy.io import fits

def is_blank(val):
    """
    Determines whether or not a value is considered 'blank'.
    """
    return val.strip() == ""

def parse_extn(extn=None):
    """
    Parse a string representing a qualified fits extension name as in the
    output of parse_filename and return a tuple (str(extname), int(extver)),
    which can be passed to pyfits functions using the 'ext' kw.
    Default return is the first extension in a fits file.

    Examples
    --------
        >>>parse_extn('sci,2')
        ('sci', 2)
        >>>parse_extn('2')
        ('', 2)
        >>>parse_extn('sci')
        ('sci', 1)
    """
    if not extn:
        return ('', 0)

    try:
        lext = extn.split(',')
    except:
        return ('', 1)

    if len(lext) == 1 and lext[0].isdigit():
        return ("", int(lext[0]))
    elif len(lext) == 2:
        return (lext[0], int(lext[1]))
    else:
        return (lext[0], 1)

def parse_filename(filename):
    """
    Parse out filename from any specified extensions.
    Returns rootname and string version of extension name.
    """
    # Parse out any extension specified in filename
    _indx = filename.find('[')
    if _indx > 0:
        # Read extension name provided
        _fname = filename[:_indx]
        _extn = filename[_indx+1:-1]
    else:
        _fname = filename
        _extn = None

    return _fname, _extn

def set_orient(the_wcs):
    """
    Computes ORIENTAT from the CD matrix
    """
    cd12 = the_wcs.wcs.cd[0][1]
    cd22 = the_wcs.wcs.cd[1][1]
    the_wcs.orientat = np.rad2deg(np.arctan2(cd12,cd22))

def set_pscale(the_wcs):
    """
    Calculates the plate scale from the CD matrix
    """
    cd11 = the_wcs.wcs.cd[0][0]
    cd21 = the_wcs.wcs.cd[1][0]
    the_wcs.pscale = np.sqrt(np.power(cd11,2)+np.power(cd21,2)) * 3600.
