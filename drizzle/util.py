import numpy as np


def find_keyword_extn(fimg, keyword, value=None):
    """
    This function will return the index of the extension in a multi-extension
    FITS file which contains the desired keyword with the given value.

    Parameters
    ----------

    fimg : hdulist
        A list of header data units

    keyword : str
        The keyword to search for

    value : str or number, optional
        If set, the value the keyword must have to match

    Returns
    -------

    The index of the extension
    """

    i = 0
    extnum = -1
    # Search through all the extensions in the FITS object
    for chip in fimg:
        hdr = chip.header
        # Check to make sure the extension has the given keyword
        if keyword in hdr:
            if value is not None:
                # If it does, then does the value match the desired value
                # MUST use 'str.strip' to match against any input string!
                if hdr[keyword].strip() == value:
                    extnum = i
                    break
            else:
                extnum = i
                break
        i += 1
    # Return the index of the extension which contained the
    # desired EXTNAME value.
    return extnum


def get_extn(fimg, extn=''):
    """
    Returns the FITS extension corresponding to extension specified in
    filename. Defaults to returning the first extension with data or the
    primary extension, if none have data.

    Parameters
    ----------

    fimg : hdulist
        A list of header data units

    extn : str
        The extension name and version to match

    Returns
    -------

    The matching header data unit
    """

    if extn:
        try:
            _extn = parse_extn(extn)
            if _extn[0]:
                _e = fimg.index_of(_extn)
            else:
                _e = _extn[1]

        except KeyError:
            _e = None

        if _e is None:
            _extn = None
        else:
            _extn = fimg[_e]

    else:
        # If no extension is provided, search for first extension
        # in FITS file with data associated with it.

        # Set up default to point to PRIMARY extension.
        _extn = fimg[0]
        # then look for first extension with data.
        for _e in fimg:
            if _e.data is not None:
                _extn = _e
                break

    return _extn


def get_keyword(fimg, keyword, default=None):
    """
    Return a keyword value from the header of an image,
    or the default if the keyword is not found.

    Parameters
    ----------

    fimg : hdulist
        A list of header data units

    keyword : hdulist
        The keyword value to search for

    default : str or number, optional
        The default value if not found

    Returns
    -------

    The value if found or default if not
    """

    value = None
    if keyword:
        _nextn = find_keyword_extn(fimg, keyword)
        try:
            value = fimg[_nextn].header[keyword]
        except KeyError:
            value = None

    if value is None and default is not None:
        value = default

    return value


def is_blank(value):
    """
    Determines whether or not a value is considered 'blank'.

    Parameters
    ----------

    value : str
        The value to check

    Returns
    -------

    True or False
    """
    return value.strip() == ""


def parse_extn(extn=''):
    """
    Parse a string representing a qualified fits extension name as in the
    output of parse_filename and return a tuple (str(extname), int(extver)),
    which can be passed to pyfits functions using the 'ext' kw.
    Default return is the first extension in a fits file.

    Examples
    --------
        >>> parse_extn('sci,2')
        ('sci', 2)
        >>> parse_extn('2')
        ('', 2)
        >>> parse_extn('sci')
        ('sci', 1)

    Parameters
    ----------
    extn : str
        The extension name

    Returns
    -------
    A tuple of the extension name and value
    """
    if not extn:
        return ('', 0)

    try:
        lext = extn.split(',')
    except Exception:
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

    Parameters
    ----------

    filename : str
        The filename to be parsed

    Returns
    -------

    A tuple with the filename root and extension
    """
    # Parse out any extension specified in filename
    _indx = filename.find('[')
    if _indx > 0:
        # Read extension name provided
        _fname = filename[:_indx]
        _extn = filename[_indx + 1:-1]
    else:
        _fname = filename
        _extn = ''

    return _fname, _extn


def set_pscale(the_wcs):
    """
    Calculates the plate scale from cdelt and the pc  matrix and adds
    it to the WCS. Plate scale is not part of the WCS standard, but is
    required by the drizzle code

    Parameters
    ----------

    the_wcs : wcs
        A WCS object
    """
    try:
        cdelt = the_wcs.wcs.get_cdelt()
        pc = the_wcs.wcs.get_pc()

    except Exception:
        try:
            # for non-celestial axes, get_cdelt doesnt work
            cdelt = the_wcs.wcs.cd * the_wcs.wcs.cdelt
        except AttributeError:
            cdelt = the_wcs.wcs.cdelt

        try:
            pc = the_wcs.wcs.pc
        except AttributeError:
            pc = 1

    pccd = np.array(cdelt * pc)
    scales = np.sqrt((pccd ** 2).sum(axis=0, dtype=float))
    the_wcs.pscale = scales[0]
