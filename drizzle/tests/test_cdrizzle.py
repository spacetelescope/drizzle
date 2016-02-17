from __future__ import print_function, absolute_import

import numpy as np

from .. import drizzle
from .. import cdrizzle

def test_cdrizzle():
    """
    Call C unit tests for cdrizzle, which are in the src/tests directory
    """

    size = 100
    data = np.zeros((size,size), dtype='float32')
    weights = np.ones((size,size), dtype='float32')

    pixmap = np.indices((size,size), dtype='float64')
    pixmap = pixmap.transpose()

    output_data = np.zeros((size,size), dtype='float32')
    output_counts = np.zeros((size,size), dtype='float32')
    output_context = np.zeros((size,size), dtype='int32')

    cdrizzle.test_cdrizzle(data, weights, pixmap,
                           output_data, output_counts,
                           output_context)

if __name__ == "__main__":
    test_cdrizzle()
