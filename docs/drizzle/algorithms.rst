Drizzle Algorithms
===================
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


Pixmap Generation
------------------
A ``pixmap`` provides the position in the output frame of each input pixel as a lookup table.
This 3-D array contains 2 ``NxM`` arrays where the dimensions ``NxM`` are identical to the input
image array's and the first array specifies the output X position while the second array specifies
the output Y position.  Creating this array requires the WCS information for both the input
and output arrays.  Each input pixel position ``(x,y)`` gets translated by the input
WCS into sky coordinates ``(RA,Dec)`` using the ``.all_pix2world()`` method of the input astropy WCS.
These sky coordinates ``(RA,Dec)`` then get translated to pixel positions in the output array
using the ``.all_sky2world()`` method of the output astropy WCS.  These output pixel
positions ``(xout,yout)`` then get written out the pixmap array element corresponding to
the original input pixel value ``(x,y)``.  The ``drizzle.drizzle.calc_pixmap`` module includes
the function ``calc_pixmap()`` for computing the pixmap for astropy-compatible images.

For data which are not supported by ``astropy.wcs.WCS``, the pixmap array can be computed
using the methods for converting pixel positions to sky coordinates using the WCS code
which supports the instrument that took the data.  JWST data, for example, relies on the
``gwcs`` package to specify the WCS as read in from the ASDF files using the ``asdf``
package.  The computation of the JWST pixmap relies on using the ``.forward_transform()``
method to compute the sky coordinates for each input pixel position and the ``.backward_transform()``
to convert those sky positions into output pixel positions.  This computation can be seen in the
``jwst.resample.resample_utils`` module of the JWST pipeline.
