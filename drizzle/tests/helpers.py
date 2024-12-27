import os

import numpy as np

from astropy import wcs
from astropy.io import fits

__all__ = ["wcs_from_file"]

TEST_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(TEST_DIR, 'data')


def wcs_from_file(filename, ext=None, return_data=False, crpix_shift=None):
    """
    Read the WCS from a ".fits" file.
    """
    full_file_name = os.path.join(DATA_DIR, filename)
    path = os.path.join(DATA_DIR, full_file_name)
    with fits.open(path) as hdu:
        if ext is None:
            for k, u in enumerate(hdu):
                if "CTYPE1" in u.header:
                    ext = k
                    break

        hdr = hdu[ext].header
        naxis1 = hdr.get("WCSNAX1", hdr.get("NAXIS1"))
        naxis2 = hdr.get("WCSNAX2", hdr.get("NAXIS2"))
        if naxis1 is not None and naxis2 is not None:
            shape = (naxis2, naxis1)
            if hdu[ext].data is None:
                hdu[ext].data = np.zeros(shape, dtype=np.float32)
        else:
            shape = None

        if crpix_shift is not None and "CRPIX1" in hdr:
            hdr["CRPIX1"] += crpix_shift[0]
            hdr["CRPIX2"] += crpix_shift[1]

        result = wcs.WCS(hdr, hdu)
        result.array_shape = shape

        if return_data:
            result = (result, )
            if not isinstance(return_data, (list, tuple)):
                return_data = [ext]
            for ext in return_data:
                data = (hdu[ext].data, )
                result = result + data

    return result
