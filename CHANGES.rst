.. _release_notes:

=============
Release Notes
=============

.. 1.14.4 (unreleased)
   ===================


1.14.3 (2023-10-02)
===================

- Disable logging to fix a segfault on some systems. [#119]


1.14.2 (2023-09-16)
===================

- Addressed test failures on big endian architectures. [#116]


1.14.1 (2023-09-16)
===================

- Duplicate re-release of 1.14.0.


1.14.0 (2023-09-15)
===================

- Fixed a bug in how drizzle would compute overlaps between input images and
  the output image. Due to this bug large parts of input image data may be
  missing in the resampled data when output image size was set by the
  caller to be smaller than size needed to hold *all* image data. [#104]

- Replace buggy polygon intersection algorithm with the Sutherland-Hodgman
  polygon-clipping algorithm. [#110]


1.13.7 (2023-02-09)
===================

- Fixed a bug in identification of lines in input images that should be skipped
  because they would not map to the output image. This bug may result in large
  chunks of input image incorrectly missing from the resampled image. [#89]


1.13.6 (2021-08-05)
===================

- Fixed a bug in how interpolation and pixel mapping was reporting invalid
  values. This bug may have resulted in resampled images containing all
  zeroes. [#85]


1.13.5 (2021-08-04)
===================

- Pin astropy min version to 5.0.4. [#81]

- Fix a bug in the interpolation algorithm used by the 'square' kernel that
  resulted in shifts of the resampled image typically by 0.5 pixels compared
  to the location indicated by the WCS. [#83]


1.13.4 (2021-12-23)
===================

- drizzle ignores the weight of input image pixels when the weight of the
  corresponding output pixel (onto which input pixel flux is to be dropped)
  is zero. [#79]


1.13.3 (2021-06-17)
===================

- Remove Cython as a runtime dependency [#72]


1.13.2 (2021-06-16)
===================

- Specify ``oldest-supported-numpy`` in pyproject.toml so that the built C
  code has the widest possible compatibility with runtime versions of
  ``numpy``. [#60]

- Fix a memory corruption issue in ``interpolate_bilinear()`` in
  ``cdrizzleblot.c`` which could result in segfault. [#66]

- Fix a context image bug when drizzling more than 32 images into a single
  output. [#69]
