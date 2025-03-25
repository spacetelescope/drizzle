.. _release_notes:

=============
Release Notes
=============


2.0.2 (unreleased)
==================

- Fix a bug in ``resample.Drizzle`` due to which initially, before adding
  the first image, ``Drizzle.output_img`` is not filled with ``NaN`` when
  ``fillval`` is either ``INDEF``, ``NAN``, ``None`` *and* the ``Drizzle``
  object was initialized with ``out_img=None``. [#170]

- Fixes a crash when ``Drizzle`` is initialized with ``disable_ctx``
  set to ``True``. [#180]


2.0.1 (2025-01-28)
==================

- Update ``utils.calc_pixmap`` code to be ready for upcoming changes in GWCS
  due to which inverse WCS transformations will respect bounding box by
  allowing the caller of ``utils.calc_pixmap`` to disable the bounding box(es)
  on both or one of the input WCS when computing the pixel map. [#164]


2.0.0 (2024-10-23)
==================

- Backward incompatible major re-design of API to make the code I/O agnostic.
  Removed FITS-specific code. Backward compatibility was
  maintained with JWST and Roman pipelines only. [#134]

- Deprecated module ``util``. New software should not import from this
  module as it will be removed in a future release. [#134]

- Bug fix: exposure time was undefined when ``in_units`` were not "cps". [#134]

- BUG FIX: ``cdrizzle.tdriz`` signature was out of date. [#134]

- Removed support for the ``'tophat'`` kernel. [#134]


1.15.3 (2024-08-19)
===================

- Fixed return type of ``PyInit_cdrizzle``. [#150]

- Allow ``context`` to be ``None`` in ``tdriz``. [#151]


1.15.2 (2024-06-17)
===================

- build wheels with Numpy 2.0 release candidate [#149]


1.15.1 (2024-03-05)
===================

- Fixed the warning type for the "gaussian", "lanczos2", and "lanczos3" kernels
  (``DeprecationWarning`` to ``Warning``). [#141]


1.15.0 (2024-02-16)
===================

- Dropped Python 3.8. [#128]

- Fixed a bug in the pixmap coordinate inversion routine. [#137]

- Deprecated "tophat" kernel which will be remover in the next release. It is
  not working correctly and should not be used. [#140]

- Added warnings that "gaussian", "lanczos2", and "lanczos3" kernels are not
  flux conserving. [#140]


1.14.4 (2023-11-15)
===================

- N/A


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
