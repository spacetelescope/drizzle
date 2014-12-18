#!/usr/bin/env python

from __future__ import print_function, absolute_import

import sys
import glob
import os.path
import numpy as np
import drizzle
import drizzle.cdrizzle

TEST_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(TEST_DIR, 'data')
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))

sys.path.append(TEST_DIR)
sys.path.append(PROJECT_DIR)

class TestDrizzle(object):

    def __init__(self, size):
        """
        Initialize test environment
        """
        self.setup(size)

    def setup(self, size):
        """
        Create python arrays used in testing
        """

        self.data = np.zeros((size,size), dtype='float32')
        self.weights = np.ones((size,size), dtype='float32')

        pixmap = np.indices((size,size), dtype='float64')
        pixmap = pixmap.transpose()
        self.pixmap = pixmap

        self.output_data = np.zeros((size,size), dtype='float32')
        self.output_counts = np.zeros((size,size), dtype='float32')
        self.output_context = np.zeros((size,size), dtype='int32')

    def test_cdrizzle(self):
        """
        Call C unit tests for cdrizzle, which are in the src/tests directory
        """
        drizzle.cdrizzle.test_cdrizzle(self.data, self.weights, self.pixmap,
                                       self.output_data, self.output_counts,
                                       self.output_context)

if __name__ == "__main__":
    go = TestDrizzle(100)
    go.test_cdrizzle()

