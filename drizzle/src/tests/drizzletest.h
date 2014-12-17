/* Declarations for test functions */

#include <Python.h>
#include <numpy/arrayobject.h>

int do_kernel_square(struct driz_param_t* p);

void set_test_arrays(PyArrayObject *dat, PyArrayObject *wei, PyArrayObject *map,
                     PyArrayObject *odat, PyArrayObject *ocnt, PyArrayObject *ocon);

int utest_cdrizzle(int argc, char* argv[]);
