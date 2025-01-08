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

        result = fits_wcs.WCS(hdr, hdu)
        result.array_shape = shape

        if wcs_type == "gwcs":
            result = _gwcs_from_hst_fits_wcs(result)

        if return_data:
            result = (result, )
            if not isinstance(return_data, (list, tuple)):
                return_data = [ext]
            for ext in return_data:
                data = (hdu[ext].data, )
                result = result + data

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
