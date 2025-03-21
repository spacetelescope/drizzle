drizzle Documentation
=====================

.. image:: https://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: https://www.astropy.org
    :alt: Powered by Astropy Badge

.. image:: https://codecov.io/github/spacetelescope/drizzle/branch/main/graphs/badge.svg
    :target: https://codecov.io/gh/spacetelescope/drizzle
    :alt: Drizzle's Coverage Status

.. image:: https://github.com/spacetelescope/drizzle/workflows/CI/badge.svg
    :target: https://github.com/spacetelescope/drizzle/actions
    :alt: CI Status

.. image:: https://readthedocs.org/projects/spacetelescope-drizzle/badge/?version=latest
    :target: https://spacetelescope-drizzle.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/drizzle.svg
    :target: https://pypi.org/project/drizzle
    :alt: PyPI Status

.. image:: https://zenodo.org/badge/28100377.svg
    :target: https://doi.org/10.5281/zenodo.10672889
    :alt: Zenodo DOI

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

- Python 3.10 or later

- Numpy

- Astropy

The Drizzle Algorithm
---------------------

This section has been extracted from Chapter 3 of
`The DrizzlePac Handbook <https://hst-docs.stsci.edu/drizzpac>`_ [Driz2025]_

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
for identifying cosmic rays in the original image. Blot requires the user
to provide the world coordinate system (WCS)-based transformations in the
form of a pixel map array as input.

.. [Driz2025] Anand, G. S., Mack, J., et al., 2025, “The DrizzlePac Handbook”, Version 3.0, (Baltimore: STScI)
