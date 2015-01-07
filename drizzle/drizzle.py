# -*- coding: utf-8 -*-

# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
The `drizzle` module defines the `Drizzle` class, for combining input
images into a single output image.
"""

from __future__ import division, print_function, unicode_literals, absolute_import

# SYSTEM
import os
import os.path

# THIRD-PARTY

import numpy as np
from astropy import wcs
from astropy.io import fits

# LOCAL
from . import util
from . import doblot
from . import dodrizzle

class Drizzle(object):
    """
    The `Drizzle` class contains the structure and methods used for combining
    input images.
    """
    def __init__(self, infile="", outwcs=None,  
                 wt_scl="", pixfrac=1.0, kernel="square", 
                 fillval="INDEF"):
        r"""
        All parameters are optional, but either infile or outwcs must be supplied.
        If infile initializes the object from a file written after a
        previous run of drizzle. Results from the previous run will be combined
        with new results. The value passed in outwcs will be ignored. If infile is
        not set, outwcs will be used to initilize a new run of drizzle.

        Parameters
        ----------
        infile: A fits file containing results from a previous run. The three
            extensions SCI, WHT, and CTX contain the combined image, total counts
            and image id bitmap, repectively. The WCS of the combined image is also
            read from the SCI extension.

        outwcs: The world coordinate system (WCS) of the combined image. This
            parameter must be present if no input file is given and is ignored if
            one is.

        wt_scl: How each input image should be scaled. The choices are `exptime`
            which scales each image by its exposure time, `expsq` which scales
            each image by the exposure time squared, or an empty string, which
            allows each input image to be scaled individually.
            
        pixfrac: The fraction of a pixel that the pixel flux is confined to. The
            default value of 1 has the pixel flux evenly spread across the image.
            A value of 0.5 confines it to half a pixel in the linear dimension,
            so the flux is confined to a quarter of the pixel area when the square
            kernel is used. 
        
        kernel: The name of the kernel used to combine the inputs. The choice of
            kernel controls the distribution of flux over the kernel. The kernel
            names are: "square", "gaussian", "point", "tophat", "turbo", "lanczos2",
            and "lanczos3". The square kernel is the default.

        fillval: The value a pixel is set to in the output if the input image does
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
            fileroot, extn = util.parse_filename(infile)

            if os.path.exists(fileroot):
                handle = fits.open(fileroot)
                self.outwcs = wcs.WCS(handle[0].header)

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

                hdu = util.get_extn(handle, extn=self.sciext)        
                if hdu is not None:
                    self.outsci = hdu.data.copy().astype(np.float32)
                    del hdu

                hdu = util.get_extn(handle, extn=self.whtext)
                if hdu is not None:
                    self.outwht = hdu.data.copy().astype(np.float32)
                    del hdu

                hdu = util.get_extn(handle, extn=self.ctxext)
                if hdu is not None:
                    self.outcon = hdu.data.copy().astype(np.int32)
                    del hdu

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

        if self.outsci is None:
            self.outsci = np.zeros((self.outwcs._naxis2,
                                   self.outwcs._naxis1),
                                   dtype=np.float32)

        if self.outwht is None:
            self.outwht = np.zeros((self.outwcs._naxis2,
                                    self.outwcs._naxis1),
                                    dtype=np.float32)
        if self.outcon is None:
            self.outcon = np.zeros((self.outwcs._naxis2,
                                    self.outwcs._naxis1),
                                    dtype=np.int32)

    def add_fits_file(self, infile, inweight="",
                      xmin=0, xmax=0, ymin=0, ymax=0,
                      unitkey="", expkey="", wt_scl=1.0):
        """
        Combine a fits file with the output drizzled image. 
        
        Parameters
        ----------
        infile: The name of the fits file, possibly including an extension.
        
        inweight: The name of a file containing a pixel by pixel weighting
            of the input data. If it is not set, an array will be generated
            where all values are set to one.
            
        xmin: This and the following three parameters set a bounding rectangle
            on the output image. Only pixels on the output image inside this
            rectangle will have their flux updated. Xmin sets the minimum value
            of the x dimension. The x dimension is the dimension that varies
            quickest on the image. If the value is zero or less, no minimum will
            be set in the x dimension. All four parameters are zero based,
            counting starts at zero.
            
        xmax: Sets the maximum value of the x dimension on the bounding box
            of the ouput image. If the value is zero or less, no maximum will 
            be set in the x dimension.

        ymin: Sets the minimum value in the y dimension on the bounding box. The
            y dimension varies less rapidly than the x and represents the line
            index on the output image. If the value is zero or less, no minimum 
            will be set in the y dimension.
            
        ymax: Sets the maximum value in the y dimension. If the value is zero or
            less, no maximum will be set in the y dimension.
            
        unitkey: The name of the header keyword containing the image units. The 
            units can either be "counts" or "cps" (counts per second.) If it is 
            left blank, the value is assumed to be "cps." If the value is counts, 
            before using the input image it is scaled by dividing it by the
            exposure time.
            
        expkey: The name of the header keyword containing the exposure time. The
            exposure time is used to scale the image if the units are counts and
            to scale the image weighting if the drizzle was initialized with
            wt_scl equal to "exptime" or "expsq." If the value of this parameter
            is blank, the exposure time is set to one, implying no scaling.
            
        wt_scl: If drizzle was initialized with wt_scl left blank, this value will
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
                    del hdu
                handle.close()

        if not insci:
            raise ValueError("Drizzle cannot find input file: %s" % infile)

        if not util.is_blank(inweight):
            fileroot, extn = util.parse_filename(inweight)

            if os.path.exists(fileroot):
                handle = fits.open(fileroot)
                hdu = util.get_extn(handle, extn=extn)
    
                if hdu is not None:
                    inwht = hdu.data.copy()
                    del hdu
                handle.close()

        in_units = util.get_keyword(fileroot, unitkey, "cps")
        expin = util.get_keyword(fileroot, expkey, 1.0)
       
        self.add_image(self, insci=insci, inwht=inwht, inwcs=inwcs,
                       xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                       expin=expin, in_units=in_units, wt_scl=wtscl)

    def add_image(self, insci=None, inwht=None, inwcs=None,
                  xmin=0, xmax=0, ymin=0, ymax=0,
                  expin=1.0, in_units="cps", wt_scl=1.0):
        r"""
        Combine an input image with the output drizzled image. Instead
        of reading the parameters from a fits file, you can set them
        by calling this lower level method. `Add_fits_file` calls this
        method after doing its setup.
        
        Parameters
        ----------
        insci: A 2d numpy array containing the input image to be drizzled.
            it is an error to not supply an image.
        
        inwht: A 2d numpy array containing the pixel by pixel weighting.
            Must have the same dimenstions as insci. If none is supplied,
            the weghting is set to one.
            
        inwcs: The world coordinate system of the input image. This is
            used to convert the pixels to the output coordinate system.
            
        xmin: This and the following three parameters set a bounding rectangle
            on the output image. Only pixels on the output image inside this
            rectangle will have their flux updated. Xmin sets the minimum value
            of the x dimension. The x dimension is the dimension that varies
            quickest on the image. If the value is zero or less, no minimum will
            be set in the x dimension. All four parameters are zero based,
            counting starts at zero.
            
        xmax: Sets the maximum value of the x dimension on the bounding box
            of the ouput image. If the value is zero or less, no maximum will 
            be set in the x dimension.

        ymin: Sets the minimum value in the y dimension on the bounding box. The
            y dimension varies less rapidly than the x and represents the line
            index on the output image. If the value is zero or less, no minimum 
            will be set in the y dimension.
            
        ymax: Sets the maximum value in the y dimension. If the value is zero or
            less, no maximum will be set in the y dimension.
            
        expin: The exposure time of the input image, a positive number. The
            exposure time is used to scale the image if the units are counts and
            to scale the image weighting if the drizzle was initialized with
            wt_scl equal to "exptime" or "expsq."

        in_units: The units of the input image. The units can either be "counts" 
            or "cps" (counts per second.) If the value is counts, before using
            the input image it is scaled by dividing it by the exposure time.
            
        wt_scl: If drizzle was initialized with wt_scl left blank, this value will
            set a scaling factor for the pixel weighting. If drizzle was
            initialized with wt_scl set to "exptime" or "expsq", the exposure time
            will be used to set the weight scaling and the value of this parameter
            will be ignored.
        """

        if insci is None:
            raise ValueError("Drizzle did not get an input image")
        else:
            insci = insci.astype(np.float32)

        if inwht is None:
            inwht = np.ones(insci.shape, dtype=insci.dtype)
        else:
            inwht = inwht.astype(np.float32)
        
        if inwcs is None:
            raise ValueError(missing_data)
        else:
            util.set_pscale(inwcs)

        if self.wt_scl == "exptime":
            wt_scl = expin
        elif self.wt_scl == "expsq":
            wt_scl = expin * expin

        self.uniqid += 1
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
        Resample an output image using a world coordinate system read
        from an input file.
        
        Parameters
        ----------
        infile: The name of the fits file containing the world coordinate
            system that the output file will be resampled to. The name may
            possibly include an extension.
            
        interp: The type of interpolation used in the resampling. The
            possible values are "nearest" (nearest neighbor interpolation),
            "linear" (bilinear interpolation), "poly3" (cubic polynomial
            interpolation), "poly5" (quintic polynomial interpolation),
            "sinc" (sinc interpolation), "lan3" (3rd order Lanczos
            interpolation), and "lan5" (5th order Lanczos interpolation).
            
        sincscl: The scaling factor for sinc interpolation.
        """
        blotwcs = None

        fileroot, extn = util.parse_filename(infile)

        if os.path.exists(fileroot):
            handle = fits.open(fileroot)
            hdu = util.get_extn(handle, extn=extn)

            if hdu is not None:
                blotwcs = wcs.WCS(header=hdu.header)
                del hdu
            handle.close()

        if not blotwcs:
            raise ValueError("Drizzle did not get a blot reference image")

        self.blot_image(blotwcs, interp=interp, sinscl=sinscl)
    
    def blot_image(self, blotwcs, interp='poly5', sinscl=1.0):
        """
        Resample an output image using a world coordinate system.
        
        Parameters
        ----------
        blotwcs: The world coordinate system to resample on.

        interp: The type of interpolation used in the resampling. The
            possible values are "nearest" (nearest neighbor interpolation),
            "linear" (bilinear interpolation), "poly3" (cubic polynomial
            interpolation), "poly5" (quintic polynomial interpolation),
            "sinc" (sinc interpolation), "lan3" (3rd order Lanczos
            interpolation), and "lan5" (5th order Lanczos interpolation).
            
        sincscl: The scaling factor for sinc interpolation.
        """

        util.set_pscale(blotwcs)
        self.outsci = doblot.doblot(self.outsci, self.outwcs, blotwcs, 
                                    1.0, interp=interp, sinscl=sinscl)

        self.outwcs = blotwcs
        
    def write(self, outfile, out_units="cps", outheader=None):
        """
        Write the output from a set of drizzled images to a file. The
        output file will contain three extensions. The "SCI" extension
        contains the resulting image. The "WHT" extension contains the
        combined weights. The "CTX" extension is a bit map. The nth bit
        is set to one if the nth input image contributed non-zero flux
        to the output image. The "CTX" image is three dimensionsional
        to account for the possibility that there are more than 32 input
        images.
        
        Parameters
        ----------
        outfile: The name of the output file. If the file already exists,
            the old file is deleted after writing the new file.

        out_units: The units of the output image, either `counts` or `cps`
            (counts per second.) If the units are counts, the resulting
            image will be multiplied by the computed exposure time.
        
        outheader: Header keywords added to the primary header of the
            output image.
        """

        if out_units != "counts" and out_units != "cps":
            raise ValueError("Illegal value for out_units: %s" % str(out_units))

        # Write the WCS to the output image
        
        handle = self.outwcs.to_fits()
        phdu = handle[0]

        # Copy the otional header to the primary header
        
        if outheader:
            phdu.header.update(outheader)

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
        phdu.header['NDRIZIM'] = self.uniqid

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
        phdu.header['DRIZEXPT'] = \
        (outexptime, 'Drizzle, exposure time scaling factor')

        # Add three extensions containing, the drizzled output image,
        # the total counts, and the context bitmap, in that order
        
        ehdu = fits.ImageHDU()
        ehdu.header['EXTNAME'] = (self.sciext, 'Extension name')
        ehdu.header['EXTVER'] = (1, 'Extension version')
        ehdu.data = self.outsci
        handle.append(ehdu)
        
        whdu = fits.ImageHDU()
        whdu.header['EXTNAME'] = (self.whtext, 'Extension name')
        whdu.header['EXTVER'] = (1, 'Extension version')
        whdu.data = self.outwht
        handle.append(whdu)
            
        xhdu = fits.ImageHDU()
        xhdu.header['EXTNAME'] = (self.ctxext, 'Extension name')
        xhdu.header['EXTVER'] = (1, 'Extension version')
        xhdu.data = self.outcon
        handle.append(xhdu)

        handle.writeto(outfile, clobber=True)
        handle.close()
