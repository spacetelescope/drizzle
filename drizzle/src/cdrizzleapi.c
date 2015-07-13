#include <Python.h>

#define _USE_MATH_DEFINES       /* MS Windows needs to define M_PI */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>

#include <numpy/arrayobject.h>

#include "cdrizzleblot.h"
#include "cdrizzlebox.h"
#include "cdrizzlemap.h"
#include "cdrizzleutil.h"
#include "tests/drizzletest.h"

static PyObject *gl_Error;

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "cdrizzle",          /* m_name */
  "C Drizzle module",  /* m_doc */
  -1,                  /* m_size */
  cdrizzle_methods,    /* m_methods */
  NULL,                /* m_reload */
  NULL,                /* m_traverse */
  NULL,                /* m_clear */
  NULL,                /* m_free */
};
#endif

/** --------------------------------------------------------------------------------------------------
 * Multiply each pixel in an image by a scale factor
 */

static void
scale_image(PyArrayObject *image, double scale_factor) {
  long  i, size;
  float *imptr;
  
  assert(image);
  imptr = (float *) PyArray_DATA(image);

  size = PyArray_DIMS(image)[0] * PyArray_DIMS(image)[1];

  for (i = size; i > 0; --i) {
    *imptr++ *= scale_factor;
  }
  
  return;
}

/** --------------------------------------------------------------------------------------------------
 * Top level function for drizzling, interfaces with python code
 */

static PyObject *
tdriz(PyObject *obj UNUSED_PARAM, PyObject *args, PyObject *keywords)
{
  const char *kwlist[] = {"input", "weights", "pixmap",
                          "output", "counts", "context",
                          "uniqid", "xmin", "xmax", "ymin", "ymax",
                          "scale", "pixfrac", "kernel", "in_units", 
                          "expscale", "wtscale", "fillstr", NULL};

  /* Arguments in the order they appear */
  PyObject *oimg, *owei, *pixmap, *oout, *owht, *ocon;
  long uniqid = 1;
  long xmin = 0;
  long xmax = 0;
  long ymin = 0;
  long ymax = 0;
  double scale = 1.0;
  double pfract = 1.0;
  char *kernel_str = "square";
  char *inun_str = "cps";
  float expin = 1.0;
  float wtscl = 1.0;
  char *fillstr = "INDEF";

  /* Derived values */
  
  PyArrayObject *img = NULL, *wei = NULL, *out = NULL, *wht = NULL, *con = NULL, *map = NULL;
  enum e_kernel_t kernel;
  enum e_unit_t inun;
  char *fillstr_end;
  bool_t do_fill;
  float fill_value;
  float inv_exposure_time;
  struct driz_error_t error;
  struct driz_param_t p;
  integer_t isize[2];

  driz_error_init(&error);
  
  if (!PyArg_ParseTupleAndKeywords(args, keywords, "OOOOOO|lllllddssffs:tdriz", (char **)kwlist,
                        &oimg, &owei, &pixmap, &oout, &owht, &ocon, /* OOOOOO */
                        &uniqid, &xmin, &xmax, &ymin, &ymax,  /* lllll */
                        &scale, &pfract, &kernel_str, &inun_str, /* ddss */
                        &expin, &wtscl,  &fillstr) /* ffs */
                       ) {
    return NULL;
  }

  /* Get raw C-array data */
  img = (PyArrayObject *)PyArray_ContiguousFromAny(oimg, PyArray_FLOAT, 2, 2);
  if (!img) {
    driz_error_set_message(&error, "Invalid input array");
    goto _exit;
  }

  wei = (PyArrayObject *)PyArray_ContiguousFromAny(owei, PyArray_FLOAT, 2, 2);
  if (!wei) {
    driz_error_set_message(&error, "Invalid weights array");
    goto _exit;
  }

  map = (PyArrayObject *)PyArray_ContiguousFromAny(pixmap, PyArray_DOUBLE, 3, 3);
  if (!map) {
    driz_error_set_message(&error, "Invalid pixmap array");
    goto _exit;
  }

  out = (PyArrayObject *)PyArray_ContiguousFromAny(oout, PyArray_FLOAT, 2, 2);
  if (!out) {
    driz_error_set_message(&error, "Invalid output array");
    goto _exit;
  }

  wht = (PyArrayObject *)PyArray_ContiguousFromAny(owht, PyArray_FLOAT, 2, 2);
  if (!wht) {
    driz_error_set_message(&error, "Invalid counts array");
    goto _exit;
  }

  con = (PyArrayObject *)PyArray_ContiguousFromAny(ocon, PyArray_INT32, 2, 2);
  if (!con) {
    driz_error_set_message(&error, "Invalid context array");
    goto _exit;
  }

  /* Convert the fill value string */

  if (fillstr == NULL ||
      *fillstr == 0 ||
      strncmp(fillstr, "INDEF", 6) == 0 ||
      strncmp(fillstr, "indef", 6) == 0) {

    do_fill = 0;
    fill_value = 0.0;

  } else {
    do_fill = 1;
#ifdef _WIN32
    fill_value = atof(fillstr);
#else
    fill_value = strtof(fillstr, &fillstr_end);
    if (fillstr == fillstr_end || *fillstr_end != '\0') {
      driz_error_set_message(&error, "Illegal fill value");
      goto _exit;
    }
#endif
  }

  /* Set the area to be processed */
  
  get_dimensions(img, isize);
  if (xmax == 0) xmax = isize[0];
  if (ymax == 0) ymax = isize[1];
  
  /* Convert strings to enumerations */
  
  if (kernel_str2enum(kernel_str, &kernel, &error) ||
      unit_str2enum(inun_str, &inun, &error)) {
    goto _exit;
  }
  
  if (pfract <= 0.001){
    printf("kernel reset to POINT due to pfract being set to 0.0...\n");
    kernel_str2enum("point", &kernel, &error);
  }

  /* If the input image is not in CPS we need to divide by the exposure */
  if (inun != unit_cps) {
    inv_exposure_time = 1.0f / p.exposure_time;
    scale_image(img, inv_exposure_time);
  }

  /* Setup reasonable defaults for drizzling */
  driz_param_init(&p);

  p.data = img;
  p.weights = wei;
  p.pixmap = map;
  p.output_data = out;
  p.output_counts = wht;
  p.output_context = con;
  p.uuid = uniqid;
  p.xmin = xmin;
  p.ymin = ymin;
  p.xmax = xmax;
  p.ymax = ymax;
  p.scale = scale;
  p.pixel_fraction = pfract;
  p.kernel = kernel;
  p.in_units = inun;
  p.exposure_time = expin;
  p.weight_scale = wtscl;
  p.fill_value = fill_value;
  p.error = &error;

  if (driz_error_check(&error, "xmin must be >= 0", p.xmin >= 0)) goto _exit;
  if (driz_error_check(&error, "ymin must be >= 0", p.ymin >= 0)) goto _exit;
  if (driz_error_check(&error, "xmax must be > xmin", p.xmax > p.xmin)) goto _exit;
  if (driz_error_check(&error, "ymax must be > ymin", p.ymax > p.ymin)) goto _exit;
  if (driz_error_check(&error, "scale must be > 0", p.scale > 0.0)) goto _exit;
  if (driz_error_check(&error, "exposure time must be > 0", p.exposure_time)) goto _exit;
  if (driz_error_check(&error, "weight scale must be > 0", p.weight_scale > 0.0)) goto _exit;
  
  if (dobox(&p)) {
    goto _exit;
  }
  
  /* Put in the fill values (if defined) */
  if (do_fill) {
    put_fill(&p, fill_value);
  }

 _exit:
  Py_XDECREF(con);
  Py_XDECREF(img); 
  Py_XDECREF(wei); 
  Py_XDECREF(out);
  Py_XDECREF(wht);
  Py_XDECREF(map);

  if (driz_error_is_set(&error)) {
    PyErr_SetString(PyExc_ValueError, driz_error_get_message(&error));
    return NULL;
  } else {
    return Py_BuildValue("sii", "Callable C-based DRIZZLE Version 0.9 (10th July 2015)", p.nmiss, p.nskip);
  }
}

/** --------------------------------------------------------------------------------------------------
 * Top level function for blotting, interfaces with python code
 */

static PyObject *
tblot(PyObject *obj, PyObject *args, PyObject *keywords)
{
  const char *kwlist[] = {"source", "pixmap", "output",
                          "xmin", "xmax", "ymin", "ymax",
                          "scale", "kscale", "interp", "exptime",
                          "misval", "sinscl", NULL};

  /* Arguments in the order they appear */
  PyObject *oimg, *pixmap, *oout;
  long xmin = 0;
  long xmax = 0;
  long ymin = 0;
  long ymax = 0;
  double scale = 1.0;
  float kscale = 1.0;
  char *interp_str = "poly5";
  float ef = 1.0;
  float misval = 0.0;
  float sinscl = 1.0;

  PyArrayObject *img = NULL, *out = NULL, *map = NULL;
  enum e_interp_t interp;
  int istat = 0;
  struct driz_error_t error;
  struct driz_param_t p;
  integer_t osize[2];

  driz_error_init(&error);
  
  if (!PyArg_ParseTupleAndKeywords(args, keywords, "OOO|lllldfsfff:tblot", (char **)kwlist,
                        &oimg, &pixmap, &oout, /* OOO */
                        &xmin, &xmax, &ymin, &ymax, /* llll */
                        &scale, &kscale, &interp_str, &ef, /* dfsf */
                        &misval, &sinscl) /* ff */
                       ){
    return NULL;
  }
  
  img = (PyArrayObject *)PyArray_ContiguousFromAny(oimg, PyArray_FLOAT, 2, 2);
  if (!img) {
    driz_error_set_message(&error, "Invalid input array");
    goto _exit;
  }
  
  map = (PyArrayObject *)PyArray_ContiguousFromAny(pixmap, PyArray_DOUBLE, 3, 3);
  if (!map) {
    driz_error_set_message(&error, "Invalid pixmap array");
    goto _exit;
  }
  
  out = (PyArrayObject *)PyArray_ContiguousFromAny(oout, PyArray_FLOAT, 2, 2);
  if (!out) {
    driz_error_set_message(&error, "Invalid output array");
    goto _exit;
  }

  if (interp_str2enum(interp_str, &interp, &error)) {
    goto _exit;
  }

  get_dimensions(out, osize);
  if (xmax == 0) xmax = osize[0];
  if (ymax == 0) ymax = osize[1];

  driz_param_init(&p);
  
  p.data = img;
  p.output_data = out;
  p.xmin = xmin;
  p.xmax = xmax;
  p.ymin = ymin;
  p.ymax = ymax;
  p.scale = scale;
  p.kscale = kscale;
  p.in_units = unit_cps;
  p.interpolation = interp;
  p.ef = ef;
  p.misval = misval;
  p.sinscl = sinscl;
  p.pixmap = map;
  p.error = &error;
  
  if (driz_error_check(&error, "xmin must be >= 0", p.xmin >= 0)) goto _exit;
  if (driz_error_check(&error, "ymin must be >= 0", p.ymin >= 0)) goto _exit;
  if (driz_error_check(&error, "xmax must be > xmin", p.xmax > p.xmin)) goto _exit;
  if (driz_error_check(&error, "ymax must be > ymin", p.ymax > p.ymin)) goto _exit;
  if (driz_error_check(&error, "scale must be > 0", p.scale > 0.0)) goto _exit;
  if (driz_error_check(&error, "kscale must be > 0", p.kscale > 0.0)) goto _exit;
  if (driz_error_check(&error, "exposure time must be > 0", p.ef > 0.0)) goto _exit;

  if (doblot(&p)) goto _exit;

 _exit:
  Py_DECREF(img);
  Py_DECREF(out);
  Py_DECREF(map);

  if (driz_error_is_set(&error)) {
    if (strcmp(driz_error_get_message(&error), "<PYTHON>") != 0)
      PyErr_SetString(PyExc_Exception, driz_error_get_message(&error));
    return NULL;
  } else {
    return Py_BuildValue("i",istat);
  }
}


/** --------------------------------------------------------------------------------------------------
 * Top level of C unit tests, interfaces with python code
 */

static PyObject *
test_cdrizzle(PyObject *self, PyObject *args)
{
  PyObject *data, *weights, *pixmap, *output_data, *output_counts, *output_context;
  PyArrayObject *dat, *wei, *map, *odat, *ocnt, *ocon;

  int argc = 1;
  char *argv[] = {"utest_cdrizzle", NULL};
  
  if (!PyArg_ParseTuple(args,"OOOOOO:test_cdrizzle", &data, &weights, &pixmap,
                                          &output_data, &output_counts, &output_context)) {
    return NULL;
  }

  dat = (PyArrayObject *)PyArray_ContiguousFromAny(data, PyArray_FLOAT, 2, 2);
  if (! dat) {
    return PyErr_Format(gl_Error, "Invalid data array.");
  }

  wei = (PyArrayObject *)PyArray_ContiguousFromAny(weights, PyArray_FLOAT, 2, 2);
  if (! wei) {
    return PyErr_Format(gl_Error, "Invalid weghts array.");
  }

  map = (PyArrayObject *)PyArray_ContiguousFromAny(pixmap, PyArray_DOUBLE, 3, 3);
  if (! map) {
    return PyErr_Format(gl_Error, "Invalid pixmap.");
  }
  
  odat = (PyArrayObject *)PyArray_ContiguousFromAny(output_data, PyArray_FLOAT, 2, 2);
  if (! odat) {
    return PyErr_Format(gl_Error, "Invalid output data array.");
  }

  ocnt = (PyArrayObject *)PyArray_ContiguousFromAny(output_counts, PyArray_FLOAT, 2, 2);
  if (! ocnt) {
    return PyErr_Format(gl_Error, "Invalid output counts array.");
  }

  ocon = (PyArrayObject *)PyArray_ContiguousFromAny(output_context, PyArray_INT32, 2, 2);
  if (! ocon) {
    return PyErr_Format(gl_Error, "Invalid context array");
  }

  set_test_arrays(dat, wei, map, odat, ocnt, ocon);
  utest_cdrizzle(argc, argv);
  
  return Py_BuildValue("");
}

/** --------------------------------------------------------------------------------------------------
 * Table of functions callable from python
*/

static PyMethodDef cdrizzle_methods[] =
  {
    {"tdriz",  (PyCFunction)tdriz, METH_VARARGS|METH_KEYWORDS,
    "tdriz(image, weight, output, outweight, context, uniqid,  xmin, ymin, scale, pfract, kernel, inun, expin, wtscl, fill, nmiss, nskip, pixmap)"},
    {"tblot",  (PyCFunction)tblot, METH_VARARGS|METH_KEYWORDS,
    "tblot(image, output, xmin, xmax, ymin, ymax, scale, kscale, interp, ef, misval, sinscl, pixmap)"},
    {"test_cdrizzle", test_cdrizzle, METH_VARARGS,
    "test_cdrizzle(data, weights, pixmap, output_data, output_counts)"},
    {0, 0, 0, 0}                             /* sentinel */
  };

/** --------------------------------------------------------------------------------------------------
 */

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "cdrizzle",          /* m_name */
  "C Drizzle module",  /* m_doc */
  -1,                  /* m_size */
  cdrizzle_methods,    /* m_methods */
  NULL,                /* m_reload */
  NULL,                /* m_traverse */
  NULL,                /* m_clear */
  NULL,                /* m_free */
};
#endif

/** --------------------------------------------------------------------------------------------------
 */

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_cdrizzle(void)
#else
initcdrizzle(void)
#endif
{
  PyObject* m;
  import_array();

#if PY_MAJOR_VERSION >= 3
  m = PyModule_Create(&moduledef);
  return m;
#else
  m = Py_InitModule3("cdrizzle", cdrizzle_methods, "C Drizzle module");
  if (m == NULL)
    return;
#endif
}
