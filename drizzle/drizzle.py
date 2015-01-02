# -*- coding: utf-8 -*-

# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
The drizzle class combines dithered input images into a single output image
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

##__all__ = ['Drizzle']

class Drizzle(object):

    def __init__(self, infile="", outwcs=None, out_units="cps", 
                 wt_scl="exptime", pixfrac=1.0, kernel="square", 
                 fillval="INDEF"):

        self.outsci = None
        self.outwht = None
        self.outcon = None
        self.outwcs = None

        if not util.is_blank(infile):
            fileroot, extn = util.parse_filename(infile)

            if os.path.exists(fileroot):
                handle = fits.open(fileroot)
                hdu = util.get_extn(handle, extn="SCI")
        
                if hdu is not None:
                    self.outsci = hdu.data.copy().astype(np.float32)
                    self.outwcs = wcs.WCS(hdu.header)
                    del hdu

                    hdu = util.get_extn(handle, extn="WHT")
                    if hdu is not None:
                        self.outwht = hdu.data.copy().astype(np.float32)
                        del hdu

                    hdu = util.get_extn(handle, extn="CTX")
                    if hdu is not None:
                        self.outcon = hdu.data.copy().astype(np.int32)
                        del hdu

                handle.close()

        if outwcs:
           self.outwcs = outwcs

        if not self.outwcs:
            raise ValueError("Either an existing file or wcs must be supplied to Drizzle")

        util.set_pscale(self.outwcs)

        if wt_scl == "exptime" or wt_scl == "expsq":
            self.wt_scl = wt_scl
        elif util.is_blank(wt_scl):
            self.wt_scl = ''
        else:
            raise ValueError("Illegal value for wt_scl: %s" % str(wt_scl))
        
        if out_units == "counts" or out_units == "cps":
            self.out_units = out_units
        else:
            raise ValueError("Illegal value for out_units: %s" % str(out_units))

        self.kernel = kernel
        self.fillval = fillval
        self.out_units = out_units
        self.pixfrac = float(pixfrac)

        if infile:
            self.outexptime = util.get_keyword(fileroot, "DRIZEXPT", default=0.0)
            self.uniqid = util.get_keyword(fileroot, "NDRIZIM", default=0)

            if (self.outsci is not None and
                self.outexptime > 0.0 and
                self.out_units == "counts"):
                np.divide(self.outsci, self.outexptime, self.outsci)
    
        else:
            self.outexptime = 0.0
            self.uniqid = 0

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
        Combine a fits file with the output drizzled image
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
        """
        Combine an input image with the output drizzled image
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
            util.set_orient(inwcs)

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

    def blot_fits_file(self, infile="",  interp='poly5', sinscl=1.0):
        """
        Resample an output image using corrdinates read from a file
        """
        blotwcs = None
        if not util.is_blank(infile):
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
        Resample the output inage onto a different grid
        """

        util.set_pscale(blotwcs)
        self.outsci = doblot.doblot(self.outsci, self.outwcs, blotwcs, 
                                    1.0, interp=interp, sinscl=sinscl)

        self.outwcs = blotwcs
        
    def write(self, outfile):
        """
        Write the drizzled image to a file
        """
        fileroot, extn = util.parse_filename(outfile)

        if os.path.exists(fileroot):
            handle = fits.open(fileroot, mode='update')
            ehdu = util.get_extn(handle, extn=extn)
            
        else:
            # We need to create the new file
            handle = fits.HDUList()
            phdu = fits.PrimaryHDU()
            handle.append(phdu)
            ehdu = None
 
        if ehdu is None:
            if extn:
                extname =util.parse_extn(extn)
            else:
                extname = ("SCI", 1)

            # Create a MEF file with the specified extname
            ehdu = fits.ImageHDU()
            ehdu.header['EXTNAME'] = extname[0]
            ehdu.header['EXTVER'] = extname[1]
            handle.append(ehdu)
 
        # Update header of output image with exptime used to scale the output data
        # if out_units is not counts, this will simply be a value of 1.0
        # the keyword 'exptime' will always contain the total exposure time
        # of all input image regardless of the output units
        ehdu.header['EXPTIME'] = self.outexptime
    
        # create CTYPE strings
        ctype1 = self.outwcs.wcs.ctype[0]
        ctype2 = self.outwcs.wcs.ctype[1]
        if ctype1.find('-SIP'): ctype1 = ctype1.replace('-SIP','')
        if ctype2.find('-SIP'): ctype2 = ctype2.replace('-SIP','')
    
        # Update header with WCS keywords
        util.set_orient(self.outwcs)
        ehdu.header['ORIENTAT'] = self.outwcs.orientat
        ehdu.header['CD1_1'] = self.outwcs.wcs.cd[0][0]
        ehdu.header['CD1_2'] = self.outwcs.wcs.cd[0][1]
        ehdu.header['CD2_1'] = self.outwcs.wcs.cd[1][0]
        ehdu.header['CD2_2'] = self.outwcs.wcs.cd[1][1]
        ehdu.header['CRVAL1'] = self.outwcs.wcs.crval[0]
        ehdu.header['CRVAL2'] = self.outwcs.wcs.crval[1]
        ehdu.header['CRPIX1'] = self.outwcs.wcs.crpix[0]
        ehdu.header['CRPIX2'] = self.outwcs.wcs.crpix[1]
        ehdu.header['CTYPE1'] = ctype1
        ehdu.header['CTYPE2'] = ctype2
        ehdu.header['VAFACTOR'] = 1.0
    
        if self.out_units == 'counts':
            np.multiply(self.outsci, self.outexptime, self.outsci)
            ehdu.header['DRIZEXPT'] = self.outexptime
        else:
            ehdu.header['DRIZEXPT'] = 1.0
    
        # Update header keyword NDRIZIM to keep track of how many images have
        # been combined in this product so far
        ehdu.header['NDRIZIM'] = self.uniqid

        # add output array to output file
        ehdu.data = self.outsci

        if not extn:
            whdu = util.get_extn(handle, "WHT")
            if whdu is None:
                whdu = fits.ImageHDU()
                whdu.header['EXTNAME'] = "WHT"
                whdu.header['EXTVER'] = 1
                handle.append(whdu)
                
            whdu.header = ehdu.header.copy()
            whdu.data = self.outwht
            
            xhdu = util.get_extn(handle, "CTX")
            if xhdu is None:
                xhdu = fits.ImageHDU()
                xhdu.header['EXTNAME'] = "CTX"
                xhdu.header['EXTVER'] = 1
                handle.append(xhdu)

            xhdu.data = self.outcon

        handle.writeto(fileroot, clobber=True)
        handle.close()
