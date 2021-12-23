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
