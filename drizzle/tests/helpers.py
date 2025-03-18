import os

import gwcs
import numpy as np
from gwcs.coordinate_frames import CelestialFrame, Frame2D

from astropy import coordinates as coord
from astropy import units
from astropy import wcs as fits_wcs
from astropy.io import fits
from astropy.modeling.models import (
    Mapping,
    Pix2Sky_TAN,
    Polynomial2D,
    RotateNative2Celestial,
    Shift,
)
from astropy.modeling.projections import AffineTransformation2D

__all__ = ["wcs_from_file"]

TEST_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(TEST_DIR, 'data')


def wcs_from_file(filename, ext=None, return_data=False, crpix_shift=None,
                  wcs_type="fits"):
    """
    Read the WCS from a ".fits" file.

    Parameters
    ----------
    filename : str
        Name of the file to load WCS from.

    ext : int, None, optional
        Extension number to load the WCS from. When `None`, the WCS will be
        loaded from the first extension containing a WCS.

    return_data : bool, optional
        When `True`, this function will return a tuple with first item
        being the WCS and the second item being the image data array.

    crpix_shift : tuple, None, optional
        A tuple of two values to be added to header CRPIX values before
        creating the WCS. This effectively introduces a constant shift
        in the image coordinate system.

    wcs_type : {"fits", "gwcs"}, optional
        Return either a FITS WCS or a gwcs.

    Returns
    -------
    WCS or tuple of WCS and image data

    """
    full_file_name = os.path.join(DATA_DIR, filename)
    path = os.path.join(DATA_DIR, full_file_name)

    def get_shape(hdr):
        naxis1 = hdr.get("WCSNAX1", hdr.get("NAXIS1"))
        naxis2 = hdr.get("WCSNAX2", hdr.get("NAXIS2"))
        if naxis1 is None or naxis2 is None:
            return None
        return (naxis2, naxis1)

    def data_from_hdr(hdr, data=None, shape=None):
        if data is not None:
            return data
        bitpix = hdr.get("BITPIX", -32)
        dtype = fits.hdu.BITPIX2DTYPE[bitpix]
        shape = get_shape(hdr) or shape
        if shape is None:
            return None
        return np.zeros(shape, dtype=dtype)

    if os.path.splitext(filename)[1] in [".hdr", ".txt"]:
        hdul = None
        hdr = fits.Header.fromfile(
            path,
            sep='\n',
            endcard=False,
            padding=False
        )

    else:
        with fits.open(path) as fits_hdul:
            hdul = fits.HDUList([hdu.copy() for hdu in fits_hdul])

        if ext is None and hdul is not None:
            for k, u in enumerate(hdul):
                if "CTYPE1" in u.header:
                    ext = k
                    break

        hdr = hdul[ext].header

    if crpix_shift is not None and "CRPIX1" in hdr:
        hdr["CRPIX1"] += crpix_shift[0]
        hdr["CRPIX2"] += crpix_shift[1]

    result = fits_wcs.WCS(hdr, hdul)
    shape = get_shape(hdr)
    result.array_shape = shape

    if wcs_type == "gwcs":
        result = _gwcs_from_hst_fits_wcs(result)

    if return_data:
        if hdul is None:
            data = data_from_hdr(hdr, data=None, shape=shape)
            return (result, data)

        result = (result, )
        if not isinstance(return_data, (list, tuple)):
            return_data = [ext]
        for ext in return_data:
            data = data_from_hdr(
                hdul[ext].header,
                data=hdul[ext].data,
                shape=shape
            )
            result = result + (data, )

    return result


def _gwcs_from_hst_fits_wcs(w):
    # NOTE: this function ignores table distortions
    def coeffs_to_poly(mat, degree):
        pol = Polynomial2D(degree=degree)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if 0 < i + j <= degree:
                    setattr(pol, f'c{i}_{j}', mat[i, j])
        return pol

    nx, ny = w.pixel_shape
    x0, y0 = w.wcs.crpix - 1

    cd = w.wcs.piximg_matrix

    if w.sip is None:
        # construct GWCS:
        det2sky = (
            (Shift(-x0) & Shift(-y0)) |
            Pix2Sky_TAN() | RotateNative2Celestial(*w.wcs.crval, 180)
        )
    else:
        cfx, cfy = np.dot(cd, [w.sip.a.ravel(), w.sip.b.ravel()])
        a = np.reshape(cfx, w.sip.a.shape)
        b = np.reshape(cfy, w.sip.b.shape)
        a[1, 0] = cd[0, 0]
        a[0, 1] = cd[0, 1]
        b[1, 0] = cd[1, 0]
        b[0, 1] = cd[1, 1]

        polx = coeffs_to_poly(a, w.sip.a_order)
        poly = coeffs_to_poly(b, w.sip.b_order)

        sip = Mapping((0, 1, 0, 1)) | (polx & poly)

        # construct GWCS:
        det2sky = (
            (Shift(-x0) & Shift(-y0)) | sip |
            Pix2Sky_TAN() | RotateNative2Celestial(*w.wcs.crval, 180)
        )

    detector_frame = Frame2D(
        name="detector",
        axes_names=("x", "y"),
        unit=(units.pix, units.pix)
    )
    sky_frame = CelestialFrame(
        reference_frame=getattr(coord, w.wcs.radesys).__call__(),
        name=w.wcs.radesys,
        unit=(units.deg, units.deg)
    )
    pipeline = [(detector_frame, det2sky), (sky_frame, None)]
    gw = gwcs.wcs.WCS(pipeline)
    gw.array_shape = w.array_shape
    gw.bounding_box = ((-0.5, nx - 0.5), (-0.5, ny - 0.5))

    if w.sip is not None:
        # compute inverse SIP and re-create output GWCS

        # compute inverse SIP:
        hdr = gw.to_fits_sip(
            max_inv_pix_error=1e-5,
            inv_degree=None,
            npoints=64,
            crpix=w.wcs.crpix,
            projection='TAN',
            verbose=False
        )
        winv = fits_wcs.WCS(hdr)
        ap = winv.sip.ap.copy()
        bp = winv.sip.bp.copy()
        ap[1, 0] += 1
        bp[0, 1] += 1
        polx_inv = coeffs_to_poly(ap, winv.sip.ap_order)
        poly_inv = coeffs_to_poly(bp, winv.sip.bp_order)
        af = AffineTransformation2D(
            matrix=np.linalg.inv(winv.wcs.piximg_matrix)
        )

        # set analytical inverses:
        sip.inverse = af | Mapping((0, 1, 0, 1)) | (polx_inv & poly_inv)

        # construct GWCS:
        det2sky = (
            (Shift(-x0) & Shift(-y0)) | sip |
            Pix2Sky_TAN() | RotateNative2Celestial(*w.wcs.crval, 180)
        )

        pipeline = [(detector_frame, det2sky), (sky_frame, None)]
        gw = gwcs.wcs.WCS(pipeline)
        gw.array_shape = w.array_shape
        gw.bounding_box = ((-0.5, nx - 0.5), (-0.5, ny - 0.5))

    return gw
