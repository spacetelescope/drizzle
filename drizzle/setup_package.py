# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, unicode_literals, print_function

from distutils.core import Extension
import os
import sys

from astropy_helpers import setup_helpers

def requires_2to3():
    return False

def get_extensions():
    ROOT = os.path.relpath(os.path.dirname(__file__))


    test_source = os.path.join('tests', 'utest_cdrizzle.c')

    cdriz_sources = ['cdrizzleapi.c',
                     'cdrizzleblot.c',
                     'cdrizzlebox.c',
                     'cdrizzlemap.c',
                     'cdrizzleutil.c',
                     test_source]
    
    sources = [str(os.path.join(ROOT, 'src', x)) for x in cdriz_sources]
    
    cfg = setup_helpers.DistutilsExtensionArgs()
    
    cfg['libraries'].append('m')
    cfg['include_dirs'].append('numpy')
    cfg['include_dirs'].append(str(os.path.join(ROOT, 'src')))
                               
    return [Extension(str('drizzle.cdrizzle'), sources, **cfg)]

def get_external_libraries():
    return []
