Using Drizzle
==============

There are 2 ways to use this package:  use the ``Drizzle`` class in an object-oriented way
or use the core function ``drizzle.dodrizzle.dodrizzle()`` directly.  Which approach
should be taken depends entirely on whether or not the input images have WCS information
which are accessed using ``astropy.wcs.WCS`` or not.  Images whose WCS information can read
in using ``astropy.wcs.WCS`` can take advantage of the simpler object-oriented ``Drizzle``
class.  Data which includes WCS information written to the image header using some convention
other than the FITS conventions understood by ``astropy.wcs.WCS``, such as JWST data, will
need to use the core function directly.

The object-oriented interface simplifies calling this code by computing the many
specific inputs for the core function, ``drizzle.dodrizzle.dodrizzle()`` within the
class itself.  This implementation, though, assumes that the input and output images
are FITS images with WCS information written to the headers in a convention supported
by ``astropy.wcs.WCS``.  As long as the data is compatible with ``astropy.wcs.WCS``,
the ``Drizzle`` class should be able to successfully combine the data using the
drizzle algorithm.

Calling the core drizzle function ``drizzle.dodrizzle.dodrizzle()`` directly
allows the user the freedom to generate the inputs to the core function using whatever conventions
best support the input data.  This function relies on a parameter called a ``pixmap``
to specify the translation from the input array to the output array based on the
WCS information of the input and output data.  Computing the ``pixmap`` requires
reading and interpreting the WCS information from the input and output images,
something which the user can implement using whatever code supports the data
they are processing.

The following sections describe each interface and provide examples on how to
call the necessary code to drizzle the data.


The Drizzle Class
------------------
The Drizzle package contains an object-oriented interface that you can use
by first creating an object of the ``Drizzle`` class.
To create a new Drizzle output image, supply an Astropy
WCS object representing the coordinate system of the output image.
The other parameters are:
  * the linear pixel dimension described in the previous section
  * the drizzle kernel used
  * how each input image is scaled (by exposure time or time squared)
  * the pixel value set in the output image where the input images do not overlap

After creating a Drizzle object, you add one or more images by calling the
``add_fits_file`` method. The arguments are the name of the FITS file containing
the input image and optionally the name of a FITS file containing the pixel
weighting. Both file names can be followed by an extension name or number in
square brackets. Optionally you can pass the name of the header keywords
containing the exposure time and units. Two units are understood: counts and
cps (counts per second).

The following object-oriented demos require a small set of initial import
statements to load the packages used in the demos::

    import drizzle
    from astropy import wcs
    from astropy.io import fits


The following function is a demonstration of how you can create a new output
image::

    def drizzle_demo_one(reference, outfile, infiles):
        """
        First demonstration of drizzle

        Parameters
        ==========
        reference
            A file containing the wcs of the output image

        outfile
            The name of the output image

        infiles
            The names of the input images to be combined
        """
        # Get the WCS for the output image
        hdulist = fits.open(reference)
        reference_wcs = wcs.WCS(hdulist[1].header)

        # Initialize the output with the WCS
        driz = drizzle.drizzle.Drizzle(outwcs=reference_wcs)

        # Combine the input images into on drizzle image
        for infile in infiles:
            driz.add_fits_file(infile)

        # Write the drizzled image out
        driz.write(outfile)

Optionally you can supply the input and weight images as Numpy arrays by using
the ``add_image`` method. If you use this method, you must supply the extra
information that would otherwise be read from the FITS image: The WCS
of the input image, the exposure time, and image units.

Here is an example of how you would call ``add_image``::

    def drizzle_demo_two(reference, outfile, infiles):
        """
        Demonstration of drizzle with add image.

        Parameters
        ==========
        reference
            A file containing the wcs of the output image.

        outfile
            The name of the output image.

        infiles
            The names of the input images to be combined.
        """
        # Get the WCS for the output image
        reflist = fits.open(reference)
        reference_wcs = wcs.WCS(reflist[1].header)

        # Initialize the output with the WCS
        driz = drizzle.drizzle.Drizzle(outwcs=reference_wcs)

        # Combine the input images into on drizzle image
        for infile in infiles:
            # Open the file and read the image and wcs
            # This is a contrived example, we would not do this
            # unless the data came from another source
            # than a FITS file
            imlist = fits.open(reference)
            image = imlist[1].data
            image_wcs = wcs.WCS(imlist[1].header)
            driz.add_image(image, image_wcs)

        # Write the drizzled image out
        driz.write(outfile)

After combining all the input images, you write the output image into a FITS
file with the ``write`` method. You must pass the name of the output image and
optionally the units. You can also supply a set of header cards to be added
to the primary header of the output FITS file.

You can also add more images to an existing Drizzle output file by creating
a new Drizzle object and passing the existing output file name as the new
object is created. In that case the output WCS and all
other parameters are read from the file.

Here is a demonstration of adding additional input images to a drizzled image::

    def drizzle_demo_three(outfile, infiles):
        """
        Demonstration of drizzle and adding to an existing output.

        Parameters
        ==========
        outfile
            Name of output image that new files will be appended to.

        infiles
            The names of the input images to be added.
        """
        # Re-open the output file
        driz = drizzle.drizzle.Drizzle(infile=outfile)

        # Add the input images to the existing output image
        for infile in infiles:
            driz.add_fits_file(infile)

        # Write the modified drizzled image out
        driz.write(outfile)

You can use the methods ``blot_fits_file`` and ``blot_image`` to transform the drizzled
output image into another WCS. Most usually this is the
coordinates of one of the input images and is used to identify cosmic rays or
other defects. The two methods ``blot_fits_file`` and ``blot_image`` allow you to
retrieve the WCS from the FITS file header or input it directly.
The optional parameter ``interp`` allows you to selct the method used to resample
the pixels on the new grid, and ``sincscl`` is used to scale the sinc function if one
of the sinc interpolation methods is used. This function demonstrates how both
methods are called::

    def drizzle_demo_four(outfile, blotfile):
        """
        Demonstration of blot methods.

        Parameters
        ==========
        outfile
            Name of output image that will be converted.

        blotfile
            Name of image containing wcs to be transformed to.
        """
        # Open drizzle using the output file
        # Transform it to another coordinate system
        driz = drizzle.drizzle.Drizzle(infile=outfile)
        driz.blot_fits_file(blotfile)
        driz.write(outfile)

        # Read the WCS and transform using it instead
        # This is a contrived example
        blotlist = fits.open(blotfile)
        blot_wcs = wcs.WCS(blotlist[1].header)
        driz = drizzle.drizzle.Drizzle(infile=outfile)
        driz.blot_image(blot_wcs)
        driz.write(outfile)

HST Example
===========
This example uses an HST WFPC2 exposure to illustrate how to use the ``Drizzle``
object to combine 3 of the 4 chips of the WFPC2 image into a single mosaic.
This example includes code for robustly handling HST filenames as well as
FITS extensions, something which may not be necessary for all situations.  It
expands on the demo code specified above to show how to use that code from end-to-end
with multi-chip data containing a WCS supported by ``astropy.wcs``.

The initial import statements needed for the HST/WFPC2 example is not too different from
what is needed for the Drizzle class examples; namely::

    from astropy.io import fits
    from astropy.wcs import wcs
    import drizzle
    from drizzle import util as drizutil


The example starts by defining a function which performs the drizzling of all specified
input chips.::

    def drizzle_demo_astropy(reference, outfile, infiles):
        """
        Demonstration of drizzle with add image.
        Parameters
        ==========
        reference
            A file containing the wcs of the output image.  The filename can contain
            the extension of the reference SCI array if it is not in the PRIMARY header.
            For example, "u26kqr01t_drz.fits[1]" for the WCS in FITS extension 1.
        outfile
            The name of the output image.
        infiles
            The names of the input images to be combined.  Each filename can include
            the specification of the SCI extension to be drizzled.
            For example, "['u26kqr01t_c0m.fits[2]', 'u26kqr01t_c0m.fits[3]']".

        """
        # Get the WCS for the output image
        refname, refext = drizutil.parse_filename(reference)
        refext = drizutil.parse_extn(refext)

        # Open reference FITS file that defines the output frame
        # all the input images will be drizzled into an array the
        # the same shape as this reference image and will have the same WCS
        reflist = fits.open(refname)
        # read in reference WCS
        reference_wcs = wcs.WCS(reflist[refext].header)
        reflist.close()

        # Initialize the output with the WCS using all default parameter values
        # This would be when different parameters can be set to change how the
        # drizzle algorithm gets applied to the input images.
        driz = drizzle.Drizzle(infile=reference, outwcs=reference_wcs)

        # Combine the input images into on drizzle image
        for iname in infiles:
            # parse input filename into rootname and any specified science extension
            # input will have format of "u26kqr01t_c0m.fits[2]"
            infile, inext = drizutil.parse_filename(iname)
            inext = drizutil.parse_extn(inext)

            # Open the file and read the image and wcs from the specified extension
            # This is a contrived example, we would not do this
            # unless the data came from another source
            # than a FITS file
            imlist = fits.open(infile)
            # Get the array for the specific chip
            image = imlist[inext].data
            # Get the WCS for the same chip
            image_wcs = wcs.WCS(imlist[inext].header)
            # drizzle this chip onto the output image
            driz.add_image(image, image_wcs)

        # Write the drizzled image out
        driz.write(outfile)

This function can be called for the sample WFPC2 exposure using::

    # This demo code can be called using WFPC2 data from exposure u26kqr01t
    # The exposure can be retrieved from the HST MAST archive.
    # This output image was originally obtained from archive
    drz = 'u26kqr01t_drw.fits[sci,1]'
    # create a new filename for the new output
    newdrz = "combine_drz.fits"
    # create a list of all input SCI arrays for this 1 exposure with the 0.1"/pix plate scale
    infiles = [f"u26kqr01t_c0m.fits[{extnum}]" for extnum in range(2,5)]

    # completely optional, but allows for unambiguous output results
    # start by creating a copy of the pipeline drz after zeroing out the data
    output = fits.open(drz.split('[')[0])
    output[1].data *= 0.0  # zero out SCI array for new output
    output.writeto(newdrz, overwrite=True)

    # now the demo code can be called using:
    drizzle_demo_astropy(drz, newdrz, infiles)


JWST Example
*************
This JWST example illustrates how to use the code in the JWST pipeline to resample
using a JWST-based class called ``GWCSDrizzle``.  This class replicates the behavior
of the original ``drizzle.Drizzle`` class except that is handles the input file format
and JWST WCS information for JWST data correctly.

The example uses a subset of only 4 exposures from the NIRCAM association JW0163-0107_nircam
as retrieved from the HST MAST archive website.

The JWST example requires these import statements from the JWST pipeline::

    from jwst import datamodels
    from jwst.resample import resample_utils, gwcs_drizzle


This short example is based on code from ``jwst.resample.resample.py``::

    # This examples uses an asn that is a subset of the full ASN for this proposal
    # It only contains the first 4 "exposures" nrc[a|b][1-4] for a total of 32 FITS files.
    # The header of the ASN is:
    """
       "asn_type": "None",
        "asn_rule": "DMS_Level3_Base",
        "version_id": null,
        "code_version": "1.4.2.dev5+gd7f97405.d20220120",
        "degraded_status": "No known degraded exposures in association.",
        "program": "noprogram",
        "constraints": "No constraints",
        "asn_id": "a3001",
        "target": "none",
        "asn_pool": "none",
        "products": [
            {
                "name": "jw01063-o107_nircam_f150w2",
                "members": [
    """
    # The full ASN file from the archive was copied into a new file with the following name
    asn = 'jw01063-o107_nircam_small_image3_asn.json'
    # It was then edited to only contain the first 4 exposures, to allow this example
    # to run in a reasonable amount of time and resources.
    # This code now reads in the ASN as a jwst.DataModel
    input_models = datamodels.open(asn)  # This may take a while...

    # Define output WCS that will encompass the input exposures
    output_wcs = resample_utils.make_output_wcs(
                    self.input_models,
                    ref_wcs=None,
                    pscale_ratio=1.0,
                    pscale=None,
                    rotation=0.0,
                    shape=None if output_shape is None else output_shape[::-1]
                )

    # Create a blank ImageModel based on the output WCS
    blank_output = datamodels.ImageModel(tuple(output_wcs.array_shape))
    # Add the metadata from the first input exposure to define the output metadata
    blank_output.update(input_models[0])
    # Add the computed output WCS
    blank_output.meta.wcs = output_wcs
    # Create the GWCSDrizzle object
    # input parameters used for controlling the drizzling operation
    # are provided here as pixfrac, kernel and fillval parameters
    driz = gwcs_drizzle.GWCSDrizzle(output_model, pixfrac=1.0,
                                    kernel='square', fillval='INDEF')

    # Drizzle each input exposure onto the output frame
    log.info("Resampling science data")
    for img in input_models:
        # create the weight map for the input
        inwht = resample_utils.build_driz_weight(img,
                                                 weight_type='EXPTIME',
                                                 good_bits=0)
        # drizzle the input exposure here...
        driz.add_image(img.data.copy(), img.meta.wcs, inwht=inwht)
        del inwht

    # Write out the result to a FITS file.
    blank_output.to_fits("jwst_drizzle.fits")



The Drizzle Function
---------------------
Data which contains a WCS specified in a file format or convention which astropy
does not support can be drizzled by calling the core ``dodrizzle()`` function
directly.  It can also be called directly for cases where you may want control
over each chip such that only some chips in the input image get drizzled to the
output frame and not others.  Images taken by the HST WFPC2 camera contain one
chip with a plate scale on the sky of 0.045"/pixel while the other 3 chips have
a plate scale of roughly 0.1"/pixel on the sky.  Frequently only the chips with
the same plate scale are drizzled, and for the examples given here, WFPC2 data
will be used.  The HST/WFPC2 data are fully supported by ``astropy.wcs.WCS``, so
the part of the computation which needs to be revised for other types of data
will be indicated in the examples.  An example based on JWST data is also being
provided to illustrate the differences in computations due to the image file
format and WCS library.

The basic function has a signature of::

    def dodrizzle(insci, input_wcs, inwht,
                  output_wcs, outsci, outwht, outcon,
                  expin, in_units, wt_scl,
                  wcslin_pscale=1.0, uniqid=1,
                  xmin=0, xmax=0, ymin=0, ymax=0,
                  pixfrac=1.0, kernel='square', fillval="INDEF"):

This input parameters for this function are numpy arrays for the input and output images
and ``astropy.wcs.WCS`` instances for the ``input_wcs`` and ``output_wcs`` parameters.  All
the rest of the parameters starting with ``expin`` control how the drizzling gets applied and are
simple floats, ints or strings, making it very easy to call.  The computations requires
for calling this function are limited to defining the input weighting array ``inwht`` based on
the form of weighting desired by the user, to keeping track of the unique ID ``uniqid`` for each input,
and to passing in a valid value for ``wt_scl``.  The rest either are read in from the input, such
as the input exposure time ``expin``, or specified by the user, such as the kernel being used.  In
rare cases, only a subset of the output array will be used for drizzling as specified by the
``xmin``, ``xmax``, ``ymin``, and ``ymax`` parameters, which may require some additional computations
or checks in the calling code to set the correct values.

The number of required parameters makes calling this function a fairly long process which
requires a significant amount of code.


HST example
***********

::

        dodrizzle.dodrizzle(insci, inwcs, inwht, self.outwcs,
                            self.outsci, self.outwht, self.outcon,
                            expin, in_units, wt_scl,
                            wcslin_pscale=inwcs.pscale, uniqid=self.uniqid,
                            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                            pixfrac=self.pixfrac, kernel=self.kernel,
                            fillval=self.fillval)


Non-HST Example
****************
Calling this function to drizzle imaging data written out in a format not supported by
Astropy requires creating a copy of the function.  That copy would then need to be
modified to call a new function for computing the pixmap using code specific to the data.
The calling code would also need to handle the I/O of the input and output data using
libraries compatible with the data.

JWST data, for example, relies on the ASDF package for file I/O to extract the image
data as numpy arrays and on the GWCS package for the WCS interpretation.  The
JWST pipeline includes a revised copy of this ``dodrizzle()`` function based on
the GWCS package in the ``jwst.resample.gwcs_drizzle`` module.



