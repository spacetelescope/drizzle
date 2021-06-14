"""
The `drizzle` module defines the `Drizzle` class, for combining input
images into a single output image using the drizzle algorithm.
"""
import os
import os.path

import numpy as np
from astropy import wcs
from astropy.io import fits

from . import util
from . import doblot
from . import dodrizzle


class Drizzle(object):
    """
    Combine images using the drizzle algorithm
    """
    def __init__(self, infile="", outwcs=None,
                 wt_scl="exptime", pixfrac=1.0, kernel="square",
                 fillval="INDEF"):
        """
        Create a new Drizzle output object and set the drizzle parameters.

        All parameters are optional, but either infile or outwcs must be supplied.
        If infile initializes the object from a file written after a
        previous run of drizzle. Results from the previous run will be combined
        with new results. The value passed in outwcs will be ignored. If infile is
        not set, outwcs will be used to initilize a new run of drizzle.

        Parameters
        ----------

        infile : str, optional
            A fits file containing results from a previous run. The three
            extensions SCI, WHT, and CTX contain the combined image, total counts
            and image id bitmap, repectively. The WCS of the combined image is
            also read from the SCI extension.

        outwcs : wcs, optional
            The world coordinate system (WCS) of the combined image. This
            parameter must be present if no input file is given and is ignored if
            one is.

        wt_scl : str, optional
            How each input image should be scaled. The choices are `exptime`
            which scales each image by its exposure time, `expsq` which scales
            each image by the exposure time squared, or an empty string, which
            allows each input image to be scaled individually.

        pixfrac : float, optional
            The fraction of a pixel that the pixel flux is confined to. The
            default value of 1 has the pixel flux evenly spread across the image.
            A value of 0.5 confines it to half a pixel in the linear dimension,
            so the flux is confined to a quarter of the pixel area when the square
            kernel is used.

        kernel : str, optional
            The name of the kernel used to combine the inputs. The choice of
            kernel controls the distribution of flux over the kernel. The kernel
            names are: "square", "gaussian", "point", "tophat", "turbo", "lanczos2",
            and "lanczos3". The square kernel is the default.

        fillval : str, otional
            The value a pixel is set to in the output if the input image does
            not overlap it. The default value of INDEF does not set a value.
        """

        # Initialize the object fields

        self.outsci = None
        self.outwht = None
        self.outcon = None

        self.outexptime = 0.0
        self.uniqid = 0

        self.outwcs = outwcs
        self.wt_scl = wt_scl
        self.kernel = kernel
        self.fillval = fillval
        self.pixfrac = float(pixfrac)

        self.sciext = "SCI"
        self.whtext = "WHT"
        self.ctxext = "CTX"

        out_units = "cps"

        if not util.is_blank(infile):
            if os.path.exists(infile):
                handle = fits.open(infile)

                # Read parameters from image header
                self.outexptime = util.get_keyword(handle, "DRIZEXPT", default=0.0)
                self.uniqid = util.get_keyword(handle, "NDRIZIM", default=0)

                self.sciext = util.get_keyword(handle, "DRIZOUDA", default="SCI")
                self.whtext = util.get_keyword(handle, "DRIZOUWE", default="WHT")
                self.ctxext = util.get_keyword(handle, "DRIZOUCO", default="CTX")

                self.wt_scl = util.get_keyword(handle, "DRIZWTSC", default=wt_scl)
                self.kernel = util.get_keyword(handle, "DRIZKERN", default=kernel)
                self.fillval = util.get_keyword(handle, "DRIZFVAL", default=fillval)
                self.pixfrac = float(util.get_keyword(handle,
                                     "DRIZPIXF", default=pixfrac))

                out_units = util.get_keyword(handle, "DRIZOUUN", default="cps")

                try:
                    hdu = handle[self.sciext]
                    self.outsci = hdu.data.copy().astype(np.float32)
                    self.outwcs = wcs.WCS(hdu.header, fobj=handle)
                except KeyError:
                    pass

                try:
                    hdu = handle[self.whtext]
                    self.outwht = hdu.data.copy().astype(np.float32)
                except KeyError:
                    pass

                try:
                    hdu = handle[self.ctxext]
                    self.outcon = hdu.data.copy().astype(np.int32)
                    if self.outcon.ndim == 2:
                        self.outcon = np.reshape(self.outcon, (1,
                                                 self.outcon.shape[0],
                                                 self.outcon.shape[1]))

                    elif self.outcon.ndim == 3:
                        pass

                    else:
                        msg = ("Drizzle context image has wrong dimensions: " +
                               infile)
                        raise ValueError(msg)

                except KeyError:
                    pass

                handle.close()

        # Check field values

        if self.outwcs:
            util.set_pscale(self.outwcs)
        else:
            raise ValueError("Either an existing file or wcs must be supplied to Drizzle")

        if util.is_blank(self.wt_scl):
            self.wt_scl = ''
        elif self.wt_scl != "exptime" and self.wt_scl != "expsq":
            raise ValueError("Illegal value for wt_scl: %s" % out_units)

        if out_units == "counts":
            np.divide(self.outsci, self.outexptime, self.outsci)
        elif out_units != "cps":
            raise ValueError("Illegal value for wt_scl: %s" % out_units)

        # Initialize images if not read from a file
        outwcs_naxis1, outwcs_naxis2 = self.outwcs.pixel_shape
        if self.outsci is None:
            self.outsci = np.zeros(self.outwcs.pixel_shape[::-1],
                                   dtype=np.float32)

        if self.outwht is None:
            self.outwht = np.zeros(self.outwcs.pixel_shape[::-1],
                                   dtype=np.float32)
        if self.outcon is None:
            self.outcon = np.zeros((1, outwcs_naxis2, outwcs_naxis1),
                                   dtype=np.int32)

    def add_fits_file(self, infile, inweight="",
                      xmin=0, xmax=0, ymin=0, ymax=0,
                      unitkey="", expkey="", wt_scl=1.0):
        """
        Combine a fits file with the output drizzled image.

        Parameters
        ----------

        infile : str
            The name of the fits file, possibly including an extension.

        inweight : str, otional
            The name of a file containing a pixel by pixel weighting
            of the input data. If it is not set, an array will be generated
            where all values are set to one.

        xmin : float, otional
            This and the following three parameters set a bounding rectangle
            on the output image. Only pixels on the output image inside this
            rectangle will have their flux updated. Xmin sets the minimum value
            of the x dimension. The x dimension is the dimension that varies
            quickest on the image. If the value is zero or less, no minimum will
            be set in the x dimension. All four parameters are zero based,
            counting starts at zero.

        xmax : float, otional
            Sets the maximum value of the x dimension on the bounding box
            of the ouput image. If the value is zero or less, no maximum will
            be set in the x dimension.

        ymin : float, optional
            Sets the minimum value in the y dimension on the bounding box. The
            y dimension varies less rapidly than the x and represents the line
            index on the output image. If the value is zero or less, no minimum
            will be set in the y dimension.

        ymax : float, optional
            Sets the maximum value in the y dimension. If the value is zero or
            less, no maximum will be set in the y dimension.

        unitkey : string, optional
            The name of the header keyword containing the image units. The
            units can either be "counts" or "cps" (counts per second.) If it is
            left blank, the value is assumed to be "cps." If the value is counts,
            before using the input image it is scaled by dividing it by the
            exposure time.

        expkey : string, optional
            The name of the header keyword containing the exposure time. The
            exposure time is used to scale the image if the units are counts and
            to scale the image weighting if the drizzle was initialized with
            wt_scl equal to "exptime" or "expsq." If the value of this parameter
            is blank, the exposure time is set to one, implying no scaling.

        wt_scl : float, optional
            If drizzle was initialized with wt_scl left blank, this value will
            set a scaling factor for the pixel weighting. If drizzle was
            initialized with wt_scl set to "exptime" or "expsq", the exposure time
            will be used to set the weight scaling and the value of this parameter
            will be ignored.
        """

        insci = None
        inwht = None

        if not util.is_blank(infile):
            fileroot, extn = util.parse_filename(infile)

            if os.path.exists(fileroot):
                handle = fits.open(fileroot)
                hdu = util.get_extn(handle, extn=extn)

                if hdu is not None:
                    insci = hdu.data
                    inwcs = wcs.WCS(header=hdu.header)
                    insci = hdu.data.copy()
                handle.close()

        if insci is None:
            raise ValueError("Drizzle cannot find input file: %s" % infile)

        if not util.is_blank(inweight):
            fileroot, extn = util.parse_filename(inweight)

            if os.path.exists(fileroot):
                handle = fits.open(fileroot)
                hdu = util.get_extn(handle, extn=extn)

                if hdu is not None:
                    inwht = hdu.data.copy()
                handle.close()

        in_units = util.get_keyword(fileroot, unitkey, "cps")
        expin = util.get_keyword(fileroot, expkey, 1.0)

        self.add_image(insci, inwcs, inwht=inwht,
                       xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                       expin=expin, in_units=in_units, wt_scl=wt_scl)

    def add_image(self, insci, inwcs, inwht=None,
                  xmin=0, xmax=0, ymin=0, ymax=0,
                  expin=1.0, in_units="cps", wt_scl=1.0):
        """
        Combine an input image with the output drizzled image.

        Instead of reading the parameters from a fits file, you can set
        them by calling this lower level method. `Add_fits_file` calls
        this method after doing its setup.

        Parameters
        ----------

        insci : array
            A 2d numpy array containing the input image to be drizzled.
            it is an error to not supply an image.

        inwcs : wcs
            The world coordinate system of the input image. This is
            used to convert the pixels to the output coordinate system.

        inwht : array, optional
            A 2d numpy array containing the pixel by pixel weighting.
            Must have the same dimenstions as insci. If none is supplied,
            the weghting is set to one.

        xmin : float, optional
            This and the following three parameters set a bounding rectangle
            on the output image. Only pixels on the output image inside this
            rectangle will have their flux updated. Xmin sets the minimum value
            of the x dimension. The x dimension is the dimension that varies
            quickest on the image. If the value is zero or less, no minimum will
            be set in the x dimension. All four parameters are zero based,
            counting starts at zero.

        xmax : float, optional
            Sets the maximum value of the x dimension on the bounding box
            of the ouput image. If the value is zero or less, no maximum will
            be set in the x dimension.

        ymin : float, optional
            Sets the minimum value in the y dimension on the bounding box. The
            y dimension varies less rapidly than the x and represents the line
            index on the output image. If the value is zero or less, no minimum
            will be set in the y dimension.

        ymax : float, optional
            Sets the maximum value in the y dimension. If the value is zero or
            less, no maximum will be set in the y dimension.

        expin : float, optional
            The exposure time of the input image, a positive number. The
            exposure time is used to scale the image if the units are counts and
            to scale the image weighting if the drizzle was initialized with
            wt_scl equal to "exptime" or "expsq."

        in_units : str, optional
            The units of the input image. The units can either be "counts"
            or "cps" (counts per second.) If the value is counts, before using
            the input image it is scaled by dividing it by the exposure time.

        wt_scl : float, optional
            If drizzle was initialized with wt_scl left blank, this value will
            set a scaling factor for the pixel weighting. If drizzle was
            initialized with wt_scl set to "exptime" or "expsq", the exposure time
            will be used to set the weight scaling and the value of this parameter
            will be ignored.
        """

        insci = insci.astype(np.float32)
        util.set_pscale(inwcs)

        if inwht is None:
            inwht = np.ones(insci.shape, dtype=insci.dtype)
        else:
            inwht = inwht.astype(np.float32)

        if self.wt_scl == "exptime":
            wt_scl = expin
        elif self.wt_scl == "expsq":
            wt_scl = expin * expin

        self.increment_id()
        self.outexptime += expin

        dodrizzle.dodrizzle(insci, inwcs, inwht, self.outwcs,
                            self.outsci, self.outwht, self.outcon,
                            expin, in_units, wt_scl,
                            wcslin_pscale=inwcs.pscale, uniqid=self.uniqid,
                            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                            pixfrac=self.pixfrac, kernel=self.kernel,
                            fillval=self.fillval)

    def blot_fits_file(self, infile, interp='poly5', sinscl=1.0):
        """
        Resample the output using another image's world coordinate system.

        Parameters
        ----------

        infile : str
            The name of the fits file containing the world coordinate
            system that the output file will be resampled to. The name may
            possibly include an extension.

        interp : str, optional
            The type of interpolation used in the resampling. The
            possible values are "nearest" (nearest neighbor interpolation),
            "linear" (bilinear interpolation), "poly3" (cubic polynomial
            interpolation), "poly5" (quintic polynomial interpolation),
            "sinc" (sinc interpolation), "lan3" (3rd order Lanczos
            interpolation), and "lan5" (5th order Lanczos interpolation).

        sincscl : float, optional
            The scaling factor for sinc interpolation.
        """
        blotwcs = None

        fileroot, extn = util.parse_filename(infile)

        if os.path.exists(fileroot):
            handle = fits.open(fileroot)
            hdu = util.get_extn(handle, extn=extn)

            if hdu is not None:
                blotwcs = wcs.WCS(header=hdu.header)
            handle.close()

        if not blotwcs:
            raise ValueError("Drizzle did not get a blot reference image")

        self.blot_image(blotwcs, interp=interp, sinscl=sinscl)

    def blot_image(self, blotwcs, interp='poly5', sinscl=1.0):
        """
        Resample the output image using an input world coordinate system.

        Parameters
        ----------

        blotwcs : wcs
            The world coordinate system to resample on.

        interp : str, optional
            The type of interpolation used in the resampling. The
            possible values are "nearest" (nearest neighbor interpolation),
            "linear" (bilinear interpolation), "poly3" (cubic polynomial
            interpolation), "poly5" (quintic polynomial interpolation),
            "sinc" (sinc interpolation), "lan3" (3rd order Lanczos
            interpolation), and "lan5" (5th order Lanczos interpolation).

        sincscl : float, optional
            The scaling factor for sinc interpolation.
        """

        util.set_pscale(blotwcs)
        self.outsci = doblot.doblot(self.outsci, self.outwcs, blotwcs,
                                    1.0, interp=interp, sinscl=sinscl)

        self.outwcs = blotwcs

    def increment_id(self):
        """
        Increment the id count and add a plane to the context image if needed

        Drizzle tracks which input images contribute to the output image
        by setting a bit in the corresponding pixel in the context image.
        The uniqid indicates which bit. So it must be incremented each time
        a new image is added. Each plane in the context image can hold 32 bits,
        so after each 32 images, a new plane is added to the context.
        """

        # Compute what plane of the context image this input would
        # correspond to:
        planeid = int(self.uniqid / 32)

        # Add a new plane to the context image if planeid overflows

        if self.outcon.shape[0] == planeid:
            plane = np.zeros_like(self.outcon[0])
            self.outcon = np.append(self.outcon, [plane], axis=0)

        # Increment the id
        self.uniqid += 1

    def write(self, outfile, out_units="cps", outheader=None):
        """
        Write the output from a set of drizzled images to a file.

        The output file will contain three extensions. The "SCI" extension
        contains the resulting image. The "WHT" extension contains the
        combined weights. The "CTX" extension is a bit map. The nth bit
        is set to one if the nth input image contributed non-zero flux
        to the output image. The "CTX" image is three dimensionsional
        to account for the possibility that there are more than 32 input
        images.

        Parameters
        ----------

        outfile : str
            The name of the output file. If the file already exists,
            the old file is deleted after writing the new file.

        out_units : str, optional
            The units of the output image, either `counts` or `cps`
            (counts per second.) If the units are counts, the resulting
            image will be multiplied by the computed exposure time.

        outheader : header, optional
            A fits header containing cards to be added to the primary
            header of the output image.
        """

        if out_units != "counts" and out_units != "cps":
            raise ValueError("Illegal value for out_units: %s" % str(out_units))

        # Write the WCS to the output image

        handle = self.outwcs.to_fits()
        phdu = handle[0]

        # Write the class fields to the primary header
        phdu.header['DRIZOUDA'] = \
            (self.sciext, 'Drizzle, output data image')
        phdu.header['DRIZOUWE'] = \
            (self.whtext, 'Drizzle, output weighting image')
        phdu.header['DRIZOUCO'] = \
            (self.ctxext, 'Drizzle, output context image')
        phdu.header['DRIZWTSC'] = \
            (self.wt_scl, 'Drizzle, weighting factor for input image')
        phdu.header['DRIZKERN'] = \
            (self.kernel, 'Drizzle, form of weight distribution kernel')
        phdu.header['DRIZPIXF'] = \
            (self.pixfrac, 'Drizzle, linear size of drop')
        phdu.header['DRIZFVAL'] = \
            (self.fillval, 'Drizzle, fill value for zero weight output pix')
        phdu.header['DRIZOUUN'] = \
            (out_units, 'Drizzle, units of output image - counts or cps')

        # Update header keyword NDRIZIM to keep track of how many images have
        # been combined in this product so far
        phdu.header['NDRIZIM'] = (self.uniqid, 'Drizzle, number of images')

        # Update header of output image with exptime used to scale the output data
        # if out_units is not counts, this will simply be a value of 1.0
        # the keyword 'exptime' will always contain the total exposure time
        # of all input image regardless of the output units

        phdu.header['EXPTIME'] = \
            (self.outexptime, 'Drizzle, total exposure time')

        outexptime = 1.0
        if out_units == 'counts':
            np.multiply(self.outsci, self.outexptime, self.outsci)
            outexptime = self.outexptime
        phdu.header['DRIZEXPT'] = (outexptime, 'Drizzle, exposure time scaling factor')

        # Copy the optional header to the primary header

        if outheader:
            phdu.header.extend(outheader, unique=True)

        # Add three extensions containing, the drizzled output image,
        # the total counts, and the context bitmap, in that order

        extheader = self.outwcs.to_header()

        ehdu = fits.ImageHDU()
        ehdu.data = self.outsci
        ehdu.header['EXTNAME'] = (self.sciext, 'Extension name')
        ehdu.header['EXTVER'] = (1, 'Extension version')
        ehdu.header.extend(extheader, unique=True)
        handle.append(ehdu)

        whdu = fits.ImageHDU()
        whdu.data = self.outwht
        whdu.header['EXTNAME'] = (self.whtext, 'Extension name')
        whdu.header['EXTVER'] = (1, 'Extension version')
        whdu.header.extend(extheader, unique=True)
        handle.append(whdu)

        xhdu = fits.ImageHDU()
        xhdu.data = self.outcon
        xhdu.header['EXTNAME'] = (self.ctxext, 'Extension name')
        xhdu.header['EXTVER'] = (1, 'Extension version')
        xhdu.header.extend(extheader, unique=True)
        handle.append(xhdu)

        handle.writeto(outfile, overwrite=True)
        handle.close()
