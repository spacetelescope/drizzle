#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import sys
from setuptools import setup, Extension

import numpy


def get_extensions():
    srcdir = os.path.join(os.path.dirname(__file__), 'src')
    cdriz_sources = ['cdrizzleapi.c',
                     'cdrizzleblot.c',
                     'cdrizzlebox.c',
                     'cdrizzlemap.c',
                     'cdrizzleutil.c',
                     os.path.join('tests', 'utest_cdrizzle.c')]
    sources = [os.path.join(srcdir, x) for x in cdriz_sources]

    cfg = {
        'include_dirs': [],
        'libraries': [],
        'define_macros': []
    }
    cfg['include_dirs'].append(numpy.get_include())
    cfg['include_dirs'].append(srcdir)
    if sys.platform != 'win32':
        cfg['libraries'].append('m')
    if sys.platform == 'win32':
        cfg['define_macros'].append(('WIN32', None))
        cfg['define_macros'].append(('__STDC__', 1))
        cfg['define_macros'].append(('_CRT_SECURE_NO_WARNINGS', None))

    return [Extension(str('drizzle.cdrizzle'), sources, **cfg)]


setup(use_scm_version=True, ext_modules=get_extensions())
