.. _release_notes:

=============
Release Notes
=============

.. 1.13.7 (unreleased)
   ===================

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
