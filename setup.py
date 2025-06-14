#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import sys

import numpy
from setuptools import Extension, setup


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
        'define_macros': [],
    }
    cfg['include_dirs'].append(numpy.get_include())
    cfg['include_dirs'].append(srcdir)

    if sys.platform == 'win32':
        cfg['define_macros'].extend(
            [
                ('WIN32', None),
                ('__STDC__', 1),
                ('_CRT_SECURE_NO_WARNINGS', None),
            ]
        )
    else:
        cfg['libraries'].append('m')
        cfg['extra_compile_args'] = [
            '-O3',
            '-Wall',
            '-Wextra',
            '-Wpedantic',
            '-Wno-unused-parameter',
            '-Wincompatible-pointer-types'
        ]
    # importing these extension modules is tested in `.github/workflows/build.yml`;
    # when adding new modules here, make sure to add them to the `test_command` entry there
    return [Extension(str('drizzle.cdrizzle'), sources, **cfg)]


setup(
    ext_modules=get_extensions(),
)
