Using Drizzle
==============

There are 2 ways to use this package:

  #. use the ``Drizzle`` class in an object-oriented way, or
  #. use the core function ``drizzle.dodrizzle.apply_drizzle()`` directly

Which approach should be taken depends entirely on whether or not the input images have WCS information
which are based on ``astropy.wcs.WCS`` or not.  Images whose WCS information can be read
in using ``astropy.wcs.WCS`` can take advantage of the object-oriented ``Drizzle``
class.  Data which includes WCS information written to the image header using some convention
other than the FITS conventions understood by ``astropy.wcs.WCS``, such as JWST data, will
need to use the core function directly.

The object-oriented interface simplifies calling this code by computing the many
specific inputs for the core function, ``drizzle.dodrizzle.apply_drizzle()`` within the
class itself.  This implementation, though, assumes that the input and output images
are FITS images with WCS information written to the headers in a convention supported
by ``astropy.wcs.WCS``.  As long as the data is compatible with ``astropy.wcs.WCS``,
the ``Drizzle`` class should be able to successfully combine the data using the
drizzle algorithm.

Calling the core drizzle function ``drizzle.dodrizzle.apply_drizzle()`` directly
allows the user the freedom to generate the inputs to the core function using whatever
WCS conventions best support the input data.  This function relies on a parameter
called a ``pixmap`` to specify the translation from the input array to the output array
based on the WCS information of the input and output data.  Computing the ``pixmap``
requires reading and interpreting the WCS information from the input and output images,
something which the user can implement using whatever code supports the data
they are processing.

The following sections describe each interface and provide examples on how to
call the necessary code to drizzle the data.

Defining Output WCS
--------------------
The most important aspect of using this code is defining the WCS for the
output frame.  This defines where the pixels from the input frame will go
when they are drizzled.  This output WCS can be defined in any one of 3 ways:

  #. create new WCS from scratch
  #. use the all the input WCSs and define a WCS without distortion that
     encompasses all the input frames
  #. read in a previously defined WCS from an already existing, preferably
     undistorted, image file.  Obviously this will only work if one of the previous
     two methods were used to write out the previous image.

Create WCS from scratch
************************
This method leaves the entire definition of the WCS up to the end user and how
this gets done depends on the WCS library that needs to be used for the type of
output image file to be created.

For FITS images, a basic WCS can be created using ``astropy.wcs.WCS`` by defining
the output frame reference point on the sky, the plate scale and orientation.
Descriptions on how to define such a WCS using ``astropy.wcs`` can be found
at https://docs.astropy.org/en/stable/wcs/example_create_imaging.html.

For JWST images based on GWCS, the GWCS readthedocs pages contain an explanation
of how to define an imaging WCS from scratch at https://gwcs.readthedocs.io/en/latest/#a-step-by-step-example-of-constructing-an-imaging-gwcs-object.

Define new WCS based on input WCSs
***********************************
The default output WCS desired most of the time would be one which defines
a tangent plane that matches the undistorted plate scale and orientation of
the input frames and that spans the full field-of-view of all input frames.
Defining such a WCS requires understaning how the distortion model has been
included in the input image WCSs, so that a new WCS can be defined without those
terms.

This computation requires determining the full extent of the input frames on the sky
after applying the distortion models, then using that footprint to define
the tangent plane.

For FITS images, the ``STWCS`` package contains a function,
``stwcs.distortion.output_wcs()``, designed specifically
to perform this computation based on a list of input astropy-compatible WCS objects.
This function serves as an example of the steps needed to perform this computation.

For JWST images based on GWCS, the JWST pipeline contains a function,
``jwst.assign_wcs.util.wcs_from_footprints()``, which performs this computation.


The Drizzle Function
---------------------
Data which contains a WCS specified in a file format or convention which astropy
does not support can be drizzled by calling the core ``apply_drizzle()`` function
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

    def apply_drizzle(insci, inwht,
                      pixmap,
                      outsci, outwht, outcon,
                      expin, in_units, wt_scl,
                      pix_ratio=1.0, uniqid=1,
                      xmin=0, xmax=0, ymin=0, ymax=0,
                      pixfrac=1.0, kernel='square',
                      fillval="INDEF")


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

.. note:: The ``apply_drizzle()`` function replaces the original ``dodrizzle()`` function from the same module.  The ``dodrizzle()`` function only works with astropy-compatible WCS objects as it calls a hard-coded version of ``calc_pixmap()`` for FITS images only.  As such, the ``dodrizzle()`` function should be considered **DEPRECATED**.


HST example
***********
This simple example demonstrates how to use this code to drizzle a single HST/WFC3 image
onto an arbitrary (pre-defined) output frame. ::

    import numpy as np
    from astropy import wcs
    from astropy.io import fits
    from drizzle import dodrizzle, calc_pixmap
    import stwcs

    # open input science array and create a WCS object for it
    fhdu = fits.open('ib3y01c6q_flt.fits')
    # Use the 'fobj' parameter so that wcs.WCS can find all related distortion header keywords
    input_wcs = wcs.WCS(header=fhdu["sci",1].header, fobj=fhdu)
    # do the same for the output array, if array and WCS are not already in memory
    ohdu = fits.open('output_drz.fits', mode='update')
    # create the output WCS based on the input WCS object
    # this example will rely on the STWCS function for simplicity of this example
    #
    output_wcs = stwcs.distortion.output_wcs([input_wcs])

    # define the output arrays
    outsci = np.zeros(output_wcs.pixel_shape, dtype=fhdu["sci",1].data.dtype)
    outwht = np.zeros(output_wcs.pixel_shape, dtype=fhdu["sci",1].data.dtype)
    outcon = np.zeros(output_wcs.pixel_shape, dtype=np.uint32)

    # define drizzling parameters - typically, user inputs
    expin = fhdu[0].header['exptime']
    in_units = 'cps'
    pixfrac = 1.0
    kernel = 'square'
    fillval = 'INDEF'
    uniqid = 1

    # create the pixmap
    pixmap = calc_pixmap.calc_pixmap(input_wcs, output_wcs)

    # drizzle the input array onto the output frame
    _vers, nmiss, nskip = dodrizzle.apply_drizzle(insci, inwht, pixmap,
                                                  outsci, outwht, outcon,
                                                  expin, in_units, wt_scl=1.0,
                                                  pix_ratio=pix_ratio, uniqid=uniqid,
                                                  xmin=0, xmax=0, ymin=0, ymax=0,
                                                  pixfrac=pixfrac, kernel=kernel, fillval=fillval)
    # write out output arrays to a file now with basic header...
    fits.PrimaryHDU(data=outsci, header=output_wcs.to_header())


Defining the Output
*********************
Creating the output WCS can be done in any number of ways depending on what is desired for the output frame.
The only real requirement is that there is a defined method or function that can be used to transform sky
coordinates into the correct position in the output array.  A valid output WCS can be defined from scratch
by defining the fiducial or reference point or tangent point (depending on the type of WCS), a matrix
providing the plate scale and orientation of the pixels on the sky, and finally keywords defining the
number of pixels that make up the output array to emcompass the desired region on the sky.
The JWST pipeline performs this computation using the ``jwst.assign_wcs.util.wcs_from_footprints()`` function, which
as the name suggests, computes an ouptut WCS that (by default) fully encompasses all pixels from all input frames.
The HST pipeline performs essentially the same computation using the **STWCS** package's
``stwcs.distortion.utils.output_wcs()`` function.


JWST Example
****************
Calling the ``apply_drizzle()`` function to drizzle imaging data written out in a format not supported by
astropy requires the same steps based on the detector's file format and WCS specification.  For JWST,
the JWST package includes all the code necessary for file I/O and for defining the WCS objects.
JWST data relies on the ASDF package for file I/O to extract the image
data as numpy arrays and on the GWCS package for the WCS interpretation.  The
JWST pipeline includes a revised copy of this ``dodrizzle()`` function based on
the GWCS package in the ``jwst.resample.gwcs_drizzle`` module.

This example demonstrates the basic steps that can be used outside of the pipeline
to resample a set of input frames onto an output frame using the ``apply_drizzle()``
function.  ::

    from jwst.assign_wcs import utils
    from jwst.resample import resample_utils
    from stdatamodels.jwst import datamodels

    # read in the input frames based on a JWST association `input_asn`
    input_models = datamodels.open(input_asn)

    # create output WCS to fully encompass all input frames
    # for simplicity of this example, it calls a JWST function
    # which wraps the fundamental jwst.assign_wcs.wcs_from_footprints() function
    #
    output_wcs = resample_utils.make_output_wcs(input_models,
                                                ref_wcs=None,
                                                pscale_ratio=pscale_ratio,
                                                pscale=pscale,
                                                rotation=rotation,
                                                shape=None,
                                                crpix=None,
                                                crval=None)

    blank_output = datamodels.ImageModel(tuple(output_wcs.array_shape))
    # update meta data and wcs
    blank_output.update(input_models[0])
    blank_output.meta.wcs = output_wcs

    # define drizzling parameters - typically, user inputs
    in_units = 'cps'
    pixfrac = 1.0
    kernel = 'square'

    # drizzle each input onto the output frame
    for uniqid,input in enumerate(input_models):
            expin = input.meta.exposure.exposure_time
            inwht = resample_utils.build_driz_weight(input,
                                                     weight_type='exptime',
                                                     good_bits=0)
            # compute pixmap for input
            pixmap = resample_utils.calc_gwcs_pixmap(input.meta.wcs, output_wcs, input.data.shape)

            # drizzle the input array onto the output frame
            _vers, nmiss, nskip = dodrizzle.apply_drizzle(input.data, inwht, pixmap,
                                                          blank_output.data,
                                                          blank_output.wht,
                                                          blank_output.con,
                                                          expin, in_units, wt_scl='exptime',
                                                          pix_ratio=1.0, uniqid=uniqid,
                                                          xmin=0, xmax=0, ymin=0, ymax=0,
                                                          pixfrac=pixfrac, kernel=kernel,
                                                          fillval="INDEF")
    # write out blank_output to a file here...
    blank_output.to_asdf(f"{input_asn.meta.asn_table.products[0].name}_resampled.asdf")


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
***********
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

