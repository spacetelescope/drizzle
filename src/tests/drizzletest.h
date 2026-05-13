/* Declarations for test functions */

#include <Python.h>
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_21_API_VERSION
#endif

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

#include <numpy/arrayobject.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

int
do_kernel_square(struct driz_param_t *p);

void
set_test_arrays(
    PyArrayObject *dat, PyArrayObject *wei, PyArrayObject *map, PyArrayObject *odat,
    PyArrayObject *ocnt, PyArrayObject *ocon);

int
utest_cdrizzle(int argc, char *argv[]);
