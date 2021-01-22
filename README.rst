drizzle Documentation
=====================

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

.. image:: https://codecov.io/github/spacetelescope/drizzle/branch/master/graphs/badge.svg
    :target: https://codecov.io/gh/spacetelescope/drizzle
    :alt: Drizzle's Coverage Status

.. image:: https://github.com/spacetelescope/drizzle/workflows/CI/badge.svg
    :target: https://github.com/spacetelescope/drizzle/actions
    :alt: CI Status

The ``drizzle`` library is a Python package for combining dithered images into a
single image. This library is derived from code used in DrizzlePac. Like
DrizzlePac, most of the code is implemented in the C language. The biggest
change from DrizzlePac is that this code passes an array that maps the input to
output image into the C code, while the DrizzlePac code computes the mapping by
using a Python callback. Switching to using an array allowed the code to be
greatly simplified.

The DrizzlePac code is currently used in the Space Telescope processing
pipelines. This library is forward looking in that it can be used with
the new GWCS code.

Requirements
------------

- Python 3.6 or later

- Numpy 1.13 or later

- Astropy 3.0 or later

The Drizzle Algorithm
---------------------

This section has been extracted from Chapter 2 of
`The DrizzlePac Handbook <http://www.stsci.edu/hst/HST_overview/drizzlepac/documents/handbooks/drizzlepac.pdf>`_ [Driz2012]_

There are a family of linear reconstruction techniques that, at two opposite
extremes, are represented by the interlacing and shift-and-add techniques, with
the Drizzle algorithm representing a continuum between these two extremes.

If the dithers are particularly well-placed, one can simply interlace the pixels
from the images onto a finer grid. In the interlacing method, pixels from the
independent input images are placed in alternate pixels on the output image
according to the alignment of the pixel centers in the original images. However,
due to occasional small positioning errors by the telescope, and non-uniform
shifts in pixel space across the detector caused by geometric distortion of the
optics, true interlacing of images is generally not feasible.

Another standard simple linear technique for combining shifted images,
descriptively named “shift-and-add”, has been used for many years to combine
dithered infrared data onto finer grids. Each input pixel is block-replicated
onto a finer subsampled grid, shifted into place, and added to the output image.
Shift-and-add has the advantage of being able to easily handle arbitrary dither
positions. However, it convolves the image yet again with the original pixel,
thus adding to the blurring of the image and to the correlation of noise in the
image. Furthermore, it is difficult to use shift-and-add in the presence of
missing data (e.g., from cosmic rays) and geometric distortion.

In response to the limitations of the two techniques described above, an
improved method known formally as variable-pixel linear reconstruction, and more
commonly referred to as Drizzle, was developed by Andy Fruchter and Richard
Hook, initially for the purposes of combining dithered images of the Hubble Deep
Field North (HDF-N). This algorithm can be thought of as a continuous set of
linear functions that vary smoothly between the optimum linear combination
technique (interlacing) and shift-and-add. This often allows an improvement in
resolution and a reduction in correlated noise, compared with images produced by
only using shift-and-add.

The degree to which the algorithm departs from interlacing and moves towards
shift-and-add depends upon how well the PSF is subsampled by the shifts in the
input images. In practice, the behavior of the Drizzle algorithm is controlled
through the use of a parameter called pixfrac, which can be set to values
ranging from 0 to 1, that represents the amount by which input pixels are shrunk
before being mapped onto the output image plane.

A key to understanding the use of pixfrac is to realize that a CCD image can be
thought of as the true image convolved first by the optics, then by the pixel
response function (ideally a square the size of a pixel), and then sampled by a
delta-function at the center of each pixel. A CCD image is thus a set of point
samples of a continuous two-dimensional function. Hence the natural value of
pixfrac is 0, which corresponds to pure interlacing. Setting pixfrac to values
greater than 0 causes additional broadening of the output PSF by convolving the
original PSF with pixels of non-zero size. Thus, setting pixfrac to its maximum
value of 1 is equivalent to shift-and-add, the other extreme of linear
combination, in which the output image PSF has been smeared by a convolution
with the full size of the original input pixels.

The Drizzle algorithm is conceptually straightforward. Pixels in the original
input images are mapped into pixels in the subsampled output image, taking into
account shifts and rotations between images and the optical distortion of the
camera. However, in order to avoid convolving the image with the large pixel
“footprint” of the camera, Drizzle allows the user to shrink the pixel before it
is averaged into the output image through the pixfrac parameter.

The flux value of each input pixel is divided up into the output pixels with
weights proportional to the area of overlap between the “drop” and each output
pixel. If the drop size is too small, not all output pixels have data added to
them from each of the input images. One should therefore choose a drop size that
is small enough to avoid convolving the image with too large an input pixel
footprint, yet sufficiently large to ensure that there is not too much variation
in the number of input pixels contributing to each output pixel.

When images are combined using Drizzle, a weight map can be specified for each
input image. The weight image contains information about bad pixels in the image
(in that bad pixels result in lower weight values). When the final output
science image is generated, an output weight map which combines information from
all the input weight images, is also saved.

Drizzle has a number of advantages over standard linear reconstruction methods.
Since the pixel area can be scaled by the Jacobian of the geometric distortion,
it is preserved for surface and absolute photometry. Therefore, the flux in the
drizzled image, that was corrected for geometric distortion, can be measured
with an aperture size that's not dependent of its position on the image. Since
the Drizzle code anticipates that a given output pixel might not receive any
information from an input pixel, missing data does not cause a substantial
problem as long as the observer has taken enough dither samples to fill in the
missing information.

The blot methods perform the inverse operation of drizzle. That is, blotting
performs the inverse mapping to transform the dithered median image back into
the coordinate system of the original input image. Blotting is primarily used
for identifying cosmic rays in the original image. Like the original drizzle
task, blot requires the user to provide the world coordinate system (WCS)
transformations as inputs.

.. [Driz2012] Gonzaga, S., Hack, W., Fruchter, A., Mack, J., eds. 2012, The DrizzlePac Handbook. (Baltimore, STScI)


The Drizzle Library
-------------------

The Drizzle library is object-oriented and you use it by first creating an object of
the ``Drizzle`` class. To create a new Drizzle output image, supply an Astropy
WCS object representing the coordinate system of the output image.
The other parameters are the linear pixel dimension described in the previous
section, the drizzle kernel used, how each input image is scaled (by exposure
time or time squared), and the pixel value set in the output image where the
input images do not overlap.

After creating a Drizzle object, you add one or more images by calling the
``add_fits_file`` method. The arguments are the name of the FITS file containing
the input image and optionally the name of a FITS file containing the pixel
weighting. Both file names can be followed by an extension name or number in
square brackets. Optionally you can pass the name of the header keywords
containing the exposure time and units. Two units are understood: counts and
cps (counts per second).

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

The lower level function ``dodrizzle`` is present for backwards compatibility with
the existing STScI DrizzlePac code and should not be used unless you are also
concerned with this compatibility.
