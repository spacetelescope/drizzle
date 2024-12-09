#include <Python.h>

#define _USE_MATH_DEFINES /* MS Windows needs to define M_PI */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <stdio.h>

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION
#endif
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

#include "cdrizzleblot.h"
#include "cdrizzlebox.h"
#include "cdrizzlemap.h"
#include "cdrizzleutil.h"
#include "tests/drizzletest.h"

static PyObject *gl_Error;
FILE *driz_log_handle = NULL;

/** ---------------------------------------------------------------------------
 * Multiply each pixel in an image by a scale factor
 */

static void
scale_image(PyArrayObject *image, double scale_factor) {
    long i, size;
    float *imptr;

    assert(image);
    imptr = (float *)PyArray_DATA(image);

    size = PyArray_DIMS(image)[0] * PyArray_DIMS(image)[1];

    for (i = size; i > 0; --i) {
        *imptr++ *= scale_factor;
    }

    return;
}

/** ---------------------------------------------------------------------------
 * Top level function for drizzling, interfaces with python code
 */

static int
process_array_list(PyObject *list, integer_t *nx, integer_t *ny,
                   const char *name, PyArrayObject ***arrays, int *nmax,
                   int allow_none, int *n_none, struct driz_error_t *error) {
    npy_intp *ndim;
    int inx, iny;
    PyObject *list_elem = NULL;
    PyArrayObject **arr_list = NULL;
    PyArrayObject *arr = NULL;
    int i, at_least_one;

    // *nx = -1;
    // *ny = -1;
    *arrays = NULL;
    *nmax = 0;
    if (n_none) {
        *n_none = 0;
    }

    if (list == NULL || list == Py_None) {
        driz_error_set(error, PyExc_ValueError,
                       "Array list '%s' is None or NULL.", name);
        return 1;
    }

    if (PyArray_CheckExact(list)) {
        if (!(arr_list = (PyArrayObject **)malloc(sizeof(PyArrayObject *)))) {
            driz_error_set(error, PyExc_MemoryError,
                           "Memory allocation failed.");
            return 1;
        }
        arr = (PyArrayObject *)PyArray_ContiguousFromAny(list, NPY_FLOAT, 2, 2);
        if (!arr) {
            driz_error_set(error, PyExc_ValueError, "Invalid '%s' array.",
                           name);
            free(arr_list);
            return 1;
        }
        *arr_list = arr;
        *nmax = 1;
        *n_none = 0;
        ndim = PyArray_DIMS(arr);
        *nx = (int)ndim[1];
        *ny = (int)ndim[0];
        *arrays = arr_list;
        return 0;

    } else if (!PyList_Check(list) && !PyTuple_Check(list)) {
        driz_error_set(
            error, PyExc_TypeError,
            "Argument '%s' is not a list or a tuple of numpy.ndarray or None.",
            name);
    }

    at_least_one = 0;

    if (!(*nmax = PySequence_Size(list))) {
        return 0;
    }

    arr_list = (PyArrayObject **)calloc(*nmax, sizeof(PyArrayObject *));
    if (!arr_list) {
        driz_error_set(error, PyExc_MemoryError, "Memory allocation failed.");
        return 1;
    }

    for (i = 0; i < *nmax; ++i) {
        if (!(list_elem = PySequence_GetItem(list, i))) {
            driz_error_set(error, PyExc_RuntimeError,
                           "Error retrieving array %d from the '%s' list.", i,
                           name);
            goto _exit_on_err;
        }

        if (list_elem == Py_None) {
            if (allow_none) {
                if (n_none) {
                    (*n_none)++;
                }
                arr = NULL;
                Py_XDECREF(list_elem);
                continue;
            } else {
                driz_error_set(
                    error, PyExc_ValueError,
                    "Element %d of '%s' list is None which is not allowed.", i,
                    name);
                goto _exit_on_err;
            }
        } else {
            arr = (PyArrayObject *)PyArray_ContiguousFromAny(list_elem,
                                                             NPY_FLOAT, 2, 2);
            if (!arr) {
                driz_error_set(error, PyExc_ValueError,
                               "Invalid array in '%s' at position %d.", name,
                               i);
                goto _exit_on_err;
            }
            at_least_one = 1;
        }
        Py_XDECREF(list_elem);

        arr_list[i] = arr;

        if (*nx < 0) {
            ndim = PyArray_DIMS(arr);
            *nx = (int)ndim[1];
            *ny = (int)ndim[0];
        } else {
            ndim = PyArray_DIMS(arr);
            inx = (int)ndim[1];
            iny = (int)ndim[0];
            if ((*nx != inx) || (*ny != iny)) {
                driz_error_set(
                    error, PyExc_ValueError,
                    "Inconsistent image shape in the '%s' image list.", name);
                goto _exit_on_err;
            }
        }
    }

    if (!at_least_one && !allow_none) {
        free(arr_list);
        arr_list = NULL;
        *nmax = 0;
    }

    *arrays = arr_list;
    return 0;

_exit_on_err:

    Py_XDECREF(list_elem);
    if (arr_list) {
        for (i = 0; i < *nmax; ++i) {
            Py_XDECREF(arr_list[i]);
        }
        free(arr_list);
    }
}

static PyObject *
tdriz(PyObject *obj UNUSED_PARAM, PyObject *args, PyObject *keywords) {
    const char *kwlist[] = {
        "input",   "weights", "pixmap",   "output", "counts",   "context",
        "input2",  "output2", "uniqid",   "xmin",   "xmax",     "ymin",
        "ymax",    "scale",   "pixfrac",  "kernel", "in_units", "expscale",
        "wtscale", "fillstr", "fillstr2", NULL};

    /* Arguments in the order they appear */
    PyObject *oimg, *owei, *pixmap, *oout, *owht, *ocon;
    PyObject *oimg2 = NULL, *oout2 = NULL;
    int i, n_none;
    int nsq_args, nsq_arr = 0, nsq_arr_out = 0;

    integer_t uniqid = 1;
    integer_t xmin = 0;
    integer_t xmax = 0;
    integer_t ymin = 0;
    integer_t ymax = 0;
    double scale = 1.0;
    double pfract = 1.0;
    char *kernel_str = "square";
    char *inun_str = "cps";
    float expin = 1.0;
    float wtscl = 1.0;
    char *fillstr = "INDEF";
    char *fillstr2 = "INDEF";

    /* Derived values */

    PyArrayObject *img = NULL, *wei = NULL, *out = NULL, *wht = NULL,
                  *con = NULL, *map = NULL;

    PyArrayObject **img2_list = NULL, **out2_list = NULL;

    enum e_kernel_t kernel;
    enum e_unit_t inun;
    char *fillstr_end;
    bool_t do_fill, do_fill2;
    float fill_value, fill_value2;
    float inv_exposure_time;
    struct driz_error_t error;
    struct driz_param_t p;
    integer_t size[2];
    integer_t nx, ny;   /* image dimensions */
    integer_t inx, iny; /* input image dimensions */
    integer_t onx, ony; /* output image dimensions */
    npy_intp *ndim;
    char warn_msg[128];

    driz_log_handle = driz_log_init(driz_log_handle);
    driz_log_message("starting tdriz");
    driz_error_init(&error);

    if (!PyArg_ParseTupleAndKeywords(
            args, keywords, "OOOOOO|OOiiiiiddssffss:tdriz", (char **)kwlist,
            &oimg, &owei, &pixmap, &oout, &owht, &ocon, &oimg2,
            &oout2,                                  /* OOOOOOOO */
            &uniqid, &xmin, &xmax, &ymin, &ymax,     /* iiiii */
            &scale, &pfract, &kernel_str, &inun_str, /* ddss */
            &expin, &wtscl, &fillstr, &fillstr2)     /* ffss */
    ) {
        return NULL;
    }

    /* Get raw C-array data */
    img = (PyArrayObject *)PyArray_ContiguousFromAny(oimg, NPY_FLOAT, 2, 2);
    if (!img) {
        driz_error_set_message(&error, "Invalid input array");
        goto _exit;
    }

    wei = (PyArrayObject *)PyArray_ContiguousFromAny(owei, NPY_FLOAT, 2, 2);
    if (!wei) {
        driz_error_set_message(&error, "Invalid weights array");
        goto _exit;
    }

    map = (PyArrayObject *)PyArray_ContiguousFromAny(pixmap, NPY_DOUBLE, 3, 3);
    if (!map) {
        driz_error_set_message(&error, "Invalid pixmap array");
        goto _exit;
    }

    out = (PyArrayObject *)PyArray_ContiguousFromAny(oout, NPY_FLOAT, 2, 2);
    if (!out) {
        driz_error_set_message(&error, "Invalid output array");
        goto _exit;
    }

    wht = (PyArrayObject *)PyArray_ContiguousFromAny(owht, NPY_FLOAT, 2, 2);
    if (!wht) {
        driz_error_set_message(&error, "Invalid counts array");
        goto _exit;
    }

    if (ocon == Py_None) {
        con = NULL;
    } else {
        con = (PyArrayObject *)PyArray_ContiguousFromAny(ocon, NPY_INT32, 2, 2);
        if (!con) {
            driz_error_set_message(&error, "Invalid context array");
            goto _exit;
        }
    }

    /* Convert the fill value string */
    if (fillstr == NULL || *fillstr == 0 || strncmp(fillstr, "INDEF", 6) == 0 ||
        strncmp(fillstr, "indef", 6) == 0) {
        do_fill = 0;
        fill_value = 0.0;

    } else if (strncmp(fillstr, "NaN", 4) == 0 ||
               strncmp(fillstr, "nan", 4) == 0) {
        do_fill = 1;
        fill_value = NPY_NANF;

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

    if (fillstr2 == NULL || *fillstr2 == 0 ||
        strncmp(fillstr2, "INDEF", 6) == 0 ||
        strncmp(fillstr2, "indef", 6) == 0) {
        do_fill2 = 0;
        fill_value2 = 0.0;

    } else if (strncmp(fillstr2, "NaN", 4) == 0 ||
               strncmp(fillstr2, "nan", 4) == 0) {
        do_fill2 = 1;
        fill_value2 = NPY_NANF;

    } else {
        do_fill2 = 1;
#ifdef _WIN32
        fill_value2 = atof(fillstr2);
#else
        fill_value2 = strtof(fillstr2, &fillstr_end);
        if (fillstr2 == fillstr_end || *fillstr_end != '\0') {
            driz_error_set_message(&error, "Illegal fill value");
            goto _exit;
        }
#endif
    }

    /* Check input array dimensions */
    ndim = PyArray_DIMS(img);
    inx = ndim[1];
    iny = ndim[0];

    ndim = PyArray_DIMS(map);
    size[0] = (integer_t)ndim[1];
    size[1] = (integer_t)ndim[0];

    if (size[0] != inx || size[1] != iny) {
        if (snprintf(warn_msg, 128,
                     "Pixel map dimensions (%d, %d) != input dimensions "
                     "(%d, %d).",
                     size[0], size[1], inx, iny) < 1) {
            strcpy(warn_msg, "Pixel map dimensions != input dimensions.");
        }
        driz_error_set_message(&error, warn_msg);
        goto _exit;
    }
    if (ndim[2] != 2) {
        driz_error_set_message(&error, "Pixel map depth (3rd dim) must be 2.");
        goto _exit;
    }

    if (wei) {
        get_dimensions(wei, size);
        if (size[0] != inx || size[1] != iny) {
            if (snprintf(warn_msg, 128,
                         "Weights array dimensions (%d, %d) != input "
                         "dimensions (%d, %d).",
                         size[0], size[1], inx, iny) < 1) {
                strcpy(warn_msg,
                       "Weights array dimensions != input dimensions.");
            }
            driz_error_set_message(&error, warn_msg);
            goto _exit;
        }
    }

    /* Check output array dimensions */
    ndim = PyArray_DIMS(out);
    onx = ndim[1];
    ony = ndim[0];

    get_dimensions(wht, size);
    if (size[0] != onx || size[1] != ony) {
        if (snprintf(warn_msg, 128,
                     "Output weight dimensions (%d, %d) != output dimensions "
                     "(%d, %d).",
                     size[0], size[1], onx, ony) < 1) {
            strcpy(warn_msg, "Output weight dimensions != output dimensions.");
        }

        driz_error_set_message(&error, warn_msg);
        goto _exit;
    }

    if (con) {
        get_dimensions(con, size);
        if (size[0] != onx || size[1] != ony) {
            if (snprintf(warn_msg, 128,
                         "Context dimensions (%d, %d) != output dimensions "
                         "(%d, %d).",
                         size[0], size[1], onx, ony) < 1) {
                strcpy(warn_msg, "Context dimensions != output dimensions.");
            }
            driz_error_set_message(&error, warn_msg);
            goto _exit;
        }
    }

    /* Handle optional arguments that supply arrays for resampling with
     * squared coefficients */
    nsq_args = ((int)(oimg2 != NULL && oimg2 != Py_None)) +
               ((int)(oout2 != NULL && oout2 != Py_None));
    if (nsq_args == 2) {
        nx = inx;
        ny = iny;
        if (process_array_list(oimg2, &nx, &ny, "input2", &img2_list, &nsq_arr,
                               1, &n_none, &error)) {
            goto _exit;
        }
        if (n_none == nsq_arr && img2_list) {
            free(img2_list);
            nsq_arr = 0;
        }

        if (nsq_arr) {
            if (nx != inx || ny != iny) {
                driz_error_set_message(&error,
                                       "'input2' arrays must have the same "
                                       "dimensions as the 'input' array.");
                goto _exit;
            }

            nx = onx;
            ny = ony;
            if (process_array_list(oout2, &nx, &ny, "output2", &out2_list,
                                   &nsq_arr_out, 0, NULL, &error)) {
                goto _exit;
            }
            if (nx != onx || ny != ony) {
                driz_error_set_message(&error,
                                       "'output2' arrays must have the same "
                                       "dimensions as the 'output' array.");
                goto _exit;
            }

            if (nsq_arr != nsq_arr_out) {
                driz_error_set_message(
                    &error,
                    "The number of 'output2' arrays must match "
                    "the number of 'input2' arrays.");
                goto _exit;
            }
        }
    } else if (nsq_args == 1) {
        driz_error_set_message(
            &error,
            "'input2' and 'output2' must both be either None, "
            "numpy.ndarray, or lists of numpy.ndarray of equal lengths.");
        goto _exit;
    }

    /* Set the area to be processed */
    if (xmax == 0 || xmax >= inx) {
        xmax = inx - 1;
    }
    if (ymax == 0 || ymax >= iny) {
        ymax = iny - 1;
    }

    if (shrink_image_section(map, &xmin, &xmax, &ymin, &ymax)) {
        driz_error_set_message(&error,
                               "No or too few valid pixels in the pixel map.");
        goto _exit;
    }

    /* Convert strings to enumerations */
    if (kernel_str2enum(kernel_str, &kernel, &error) ||
        unit_str2enum(inun_str, &inun, &error)) {
        goto _exit;
    }

    if (kernel == kernel_gaussian || kernel == kernel_lanczos2 ||
        kernel == kernel_lanczos3) {
        if (snprintf(warn_msg, 128,
                     "Kernel '%s' is not a flux-conserving kernel.",
                     kernel_str) < 1) {
            strcpy(warn_msg,
                   "Selected kernel is not a flux-conserving kernel.");
        }
        PyErr_WarnEx(PyExc_Warning, warn_msg, 1);
    }

    if (pfract <= 0.001) {
        printf("kernel reset to POINT due to pfract being set to 0.0...\n");
        kernel_str2enum("point", &kernel, &error);
    }

    /* If the input image is not in CPS we need to divide by the exposure */
    if (inun != unit_cps) {
        inv_exposure_time = 1.0f / expin;
        scale_image(img, inv_exposure_time);
        if (img2_list) {
            for (i = 0; i < nsq_arr; ++i) {
                if (img2_list[i] != NULL) {
                    scale_image(img2_list[i], pow(inv_exposure_time, 2.0));
                }
            }
        }
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
    p.fill_value2 = fill_value2;
    p.in_nx = inx;
    p.in_ny = iny;
    p.out_nx = onx;
    p.out_ny = ony;
    p.data2 = img2_list;
    p.output_data2 = out2_list;
    p.ndata2 = nsq_arr;
    p.error = &error;

    if (driz_error_check(&error, "xmin must be >= 0", p.xmin >= 0)) {
        goto _exit;
    }
    if (driz_error_check(&error, "ymin must be >= 0", p.ymin >= 0)) {
        goto _exit;
    }
    if (driz_error_check(&error, "xmax must be > xmin", p.xmax > p.xmin)) {
        goto _exit;
    }
    if (driz_error_check(&error, "ymax must be > ymin", p.ymax > p.ymin)) {
        goto _exit;
    }
    if (driz_error_check(&error, "scale must be > 0", p.scale > 0.0f)) {
        goto _exit;
    }
    if (driz_error_check(&error, "exposure time must be > 0",
                         p.exposure_time > 0.0f)) {
        goto _exit;
    }
    if (driz_error_check(&error, "weight scale must be > 0",
                         p.weight_scale > 0.0f)) {
        goto _exit;
    }

    get_dimensions(p.pixmap, psize);
    if (psize[0] != isize[0] || psize[1] != isize[1]) {
        if (snprintf(
                warn_msg, 128,
                "Pixel map dimensions (%d, %d) != input dimensions (%d, %d).",
                psize[0], psize[1], isize[0], isize[1]) < 1) {
            strcpy(warn_msg, "Pixel map dimensions != input dimensions.");
        }
        driz_error_set_message(&error, warn_msg);
        goto _exit;
    }

    if (dobox(&p)) {
        goto _exit;
    }

    /* Put in the fill values (if defined) */
    if (do_fill) {
        put_fill(&p);
    }

_exit:
    driz_log_message("ending tdriz");
    driz_log_close(driz_log_handle);
    Py_XDECREF(con);
    Py_XDECREF(img);
    Py_XDECREF(wei);
    Py_XDECREF(out);
    Py_XDECREF(wht);
    Py_XDECREF(map);

    if (nsq_arr > 0 && img2_list) {
        for (i = 0; i < nsq_arr; ++i) {
            Py_XDECREF(img2_list[i]);
        }
        free(img2_list);
    }

    if (nsq_arr_out > 0 && out2_list) {
        for (i = 0; i < nsq_arr_out; ++i) {
            Py_XDECREF(out2_list[i]);
        }
        free(out2_list);
    }

    if (driz_error_is_set(&error)) {
        if (error.type == NULL) {
            error.type = PyExc_ValueError; /* default error type */
        }
        PyErr_SetString(error.type, driz_error_get_message(&error));
        return NULL;
    } else {
        return Py_BuildValue("sii", "Callable C-based DRIZZLE Version 2.1.0",
                             p.nmiss, p.nskip);
    }
}

/** ---------------------------------------------------------------------------
 * Top level function for blotting, interfaces with python code
 */

static PyObject *
tblot(PyObject *obj, PyObject *args, PyObject *keywords) {
    const char *kwlist[] = {"source",  "pixmap", "output", "xmin",   "xmax",
                            "ymin",    "ymax",   "scale",  "kscale", "interp",
                            "exptime", "misval", "sinscl", NULL};

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
    integer_t psize[2], osize[2];
    char warn_msg[128];

    driz_log_handle = driz_log_init(driz_log_handle);
    driz_log_message("starting tblot");
    driz_error_init(&error);

    if (!PyArg_ParseTupleAndKeywords(
            args, keywords, "OOO|lllldfsfff:tblot", (char **)kwlist, &oimg,
            &pixmap, &oout,                    /* OOO */
            &xmin, &xmax, &ymin, &ymax,        /* llll */
            &scale, &kscale, &interp_str, &ef, /* dfsf */
            &misval, &sinscl)                  /* ff */
    ) {
        return NULL;
    }

    img = (PyArrayObject *)PyArray_ContiguousFromAny(oimg, NPY_FLOAT, 2, 2);
    if (!img) {
        driz_error_set_message(&error, "Invalid input array");
        goto _exit;
    }

    map = (PyArrayObject *)PyArray_ContiguousFromAny(pixmap, NPY_DOUBLE, 3, 3);
    if (!map) {
        driz_error_set_message(&error, "Invalid pixmap array");
        goto _exit;
    }

    out = (PyArrayObject *)PyArray_ContiguousFromAny(oout, NPY_FLOAT, 2, 2);
    if (!out) {
        driz_error_set_message(&error, "Invalid output array");
        goto _exit;
    }

    if (interp_str2enum(interp_str, &interp, &error)) {
        goto _exit;
    }

    get_dimensions(map, psize);
    get_dimensions(out, osize);

    if (psize[0] != osize[0] || psize[1] != osize[1]) {
        if (snprintf(warn_msg, 128,
                     "Pixel map dimensions (%d, %d) != output dimensions "
                     "(%d, %d).",
                     psize[0], psize[1], osize[0], osize[1]) < 1) {
            strcpy(warn_msg, "Pixel map dimensions != output dimensions.");
        }
        driz_error_set_message(&error, warn_msg);
        goto _exit;
    }

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
    if (driz_error_check(&error, "xmax must be > xmin", p.xmax > p.xmin))
        goto _exit;
    if (driz_error_check(&error, "ymax must be > ymin", p.ymax > p.ymin))
        goto _exit;
    if (driz_error_check(&error, "scale must be > 0", p.scale > 0.0))
        goto _exit;
    if (driz_error_check(&error, "kscale must be > 0", p.kscale > 0.0))
        goto _exit;
    if (driz_error_check(&error, "exposure time must be > 0", p.ef > 0.0))
        goto _exit;

    if (doblot(&p)) goto _exit;

_exit:
    driz_log_message("ending tblot");
    driz_log_close(driz_log_handle);
    Py_DECREF(img);
    Py_DECREF(out);
    Py_DECREF(map);

    if (driz_error_is_set(&error)) {
        if (strcmp(driz_error_get_message(&error), "<PYTHON>") != 0)
            PyErr_SetString(PyExc_Exception, driz_error_get_message(&error));
        return NULL;
    } else {
        return Py_BuildValue("i", istat);
    }
}

/** ---------------------------------------------------------------------------
 * Top level of C unit tests, interfaces with python code
 */

static PyObject *
test_cdrizzle(PyObject *self, PyObject *args) {
    PyObject *data, *weights, *pixmap, *output_data, *output_counts,
        *output_context;
    PyArrayObject *dat, *wei, *map, *odat, *ocnt, *ocon;

    int argc = 1;
    char *argv[] = {"utest_cdrizzle", NULL};

    if (!PyArg_ParseTuple(args, "OOOOOO:test_cdrizzle", &data, &weights,
                          &pixmap, &output_data, &output_counts,
                          &output_context)) {
        return NULL;
    }

    dat = (PyArrayObject *)PyArray_ContiguousFromAny(data, NPY_FLOAT, 2, 2);
    if (!dat) {
        return PyErr_Format(gl_Error, "Invalid data array.");
    }

    wei = (PyArrayObject *)PyArray_ContiguousFromAny(weights, NPY_FLOAT, 2, 2);
    if (!wei) {
        return PyErr_Format(gl_Error, "Invalid weghts array.");
    }

    map = (PyArrayObject *)PyArray_ContiguousFromAny(pixmap, NPY_DOUBLE, 2, 4);
    if (!map) {
        return PyErr_Format(gl_Error, "Invalid pixmap.");
    }

    odat = (PyArrayObject *)PyArray_ContiguousFromAny(output_data, NPY_FLOAT, 2,
                                                      2);
    if (!odat) {
        return PyErr_Format(gl_Error, "Invalid output data array.");
    }

    ocnt = (PyArrayObject *)PyArray_ContiguousFromAny(output_counts, NPY_FLOAT,
                                                      2, 2);
    if (!ocnt) {
        return PyErr_Format(gl_Error, "Invalid output counts array.");
    }

    ocon = (PyArrayObject *)PyArray_ContiguousFromAny(output_context, NPY_INT32,
                                                      2, 2);
    if (!ocon) {
        return PyErr_Format(gl_Error, "Invalid context array");
    }

    set_test_arrays(dat, wei, map, odat, ocnt, ocon);
    utest_cdrizzle(argc, argv);

    return Py_BuildValue("");
}

static PyObject *
invert_pixmap_wrap(PyObject *self, PyObject *args) {
    PyObject *pixmap, *xyout, *bbox;
    PyArrayObject *xyout_arr, *pixmap_arr, *bbox_arr;
    struct driz_param_t par;
    double *xy, *xyin;
    npy_intp *ndim, xyin_dim = 2;
    const double half = 0.5 - DBL_EPSILON;

    xyin = (double *)malloc(2 * sizeof(double));

    if (!PyArg_ParseTuple(args, "OOO:invpixmap", &pixmap, &xyout, &bbox)) {
        return NULL;
    }

    xyout_arr =
        (PyArrayObject *)PyArray_ContiguousFromAny(xyout, NPY_DOUBLE, 1, 1);
    if (!xyout_arr) {
        return PyErr_Format(gl_Error, "Invalid xyout array.");
    }

    pixmap_arr =
        (PyArrayObject *)PyArray_ContiguousFromAny(pixmap, NPY_DOUBLE, 3, 3);
    if (!pixmap_arr) {
        return PyErr_Format(gl_Error, "Invalid pixmap.");
    }

    par.pixmap = pixmap_arr;
    ndim = PyArray_DIMS(pixmap_arr);

    if (bbox == Py_None) {
        par.xmin = 0;
        par.xmax = ndim[1] - 1;
        par.ymin = 0;
        par.ymax = ndim[0] - 1;
    } else {
        bbox_arr =
            (PyArrayObject *)PyArray_ContiguousFromAny(bbox, NPY_DOUBLE, 2, 2);
        if (!bbox_arr) {
            return PyErr_Format(gl_Error, "Invalid input bounding box.");
        }
        par.xmin =
            (integer_t)(*(double *)PyArray_GETPTR2(bbox_arr, 0, 0) - half);
        par.xmax =
            (integer_t)(*(double *)PyArray_GETPTR2(bbox_arr, 0, 1) + half);
        par.ymin =
            (integer_t)(*(double *)PyArray_GETPTR2(bbox_arr, 1, 0) - half);
        par.ymax =
            (integer_t)(*(double *)PyArray_GETPTR2(bbox_arr, 1, 1) + half);
    }

    xy = (double *)PyArray_DATA(xyout_arr);

    if (invert_pixmap(&par, xy[0], xy[1], &xyin[0], &xyin[1])) {
        return Py_BuildValue("");
    }

    PyArrayObject *arr = (PyArrayObject *)PyArray_SimpleNewFromData(
        1, &xyin_dim, NPY_DOUBLE, xyin);

    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_OWNDATA);

    return Py_BuildValue("N", arr);
}

static PyObject *
clip_polygon_wrap(PyObject *self, PyObject *args) {
    int k;
    PyObject *pin, *qin;
    PyArrayObject *pin_arr, *qin_arr;
    struct polygon p, q, pq;
    PyObject *list, *tuple;

    if (!PyArg_ParseTuple(args, "OO:clip_polygon", &pin, &qin)) {
        return NULL;
    }

    pin_arr = (PyArrayObject *)PyArray_ContiguousFromAny(pin, NPY_DOUBLE, 2, 2);
    if (!pin_arr) {
        return PyErr_Format(gl_Error, "Invalid P.");
    }

    qin_arr = (PyArrayObject *)PyArray_ContiguousFromAny(qin, NPY_DOUBLE, 2, 2);
    if (!qin_arr) {
        return PyErr_Format(gl_Error, "Invalid Q.");
    }

    p.npv = PyArray_SHAPE(pin_arr)[0];
    for (k = 0; k < p.npv; ++k) {
        p.v[k].x = *((double *)PyArray_GETPTR2(pin_arr, k, 0));
        p.v[k].y = *((double *)PyArray_GETPTR2(pin_arr, k, 1));
    }

    q.npv = PyArray_SHAPE(qin_arr)[0];
    for (k = 0; k < q.npv; ++k) {
        q.v[k].x = *((double *)PyArray_GETPTR2(qin_arr, k, 0));
        q.v[k].y = *((double *)PyArray_GETPTR2(qin_arr, k, 1));
    }

    clip_polygon_to_window(&p, &q, &pq);

    list = PyList_New(pq.npv);

    for (k = 0; k < pq.npv; ++k) {
        tuple = PyTuple_New(2);
        PyTuple_SetItem(tuple, 0, PyFloat_FromDouble(pq.v[k].x));
        PyTuple_SetItem(tuple, 1, PyFloat_FromDouble(pq.v[k].y));
        PyList_SetItem(list, k, tuple);
    }

    return Py_BuildValue("N", list);
}

/** ---------------------------------------------------------------------------
 * Table of functions callable from python
 */
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-function-type-mismatch"
#endif
static struct PyMethodDef cdrizzle_methods[] = {
    {"tdriz", (PyCFunction)tdriz, METH_VARARGS | METH_KEYWORDS,
     "tdriz(image, weights, pixmap, output, counts, context, image2, "
     "counts2, "
     "output2, uniqid, xmin, xmax, ymin, ymax, scale, pixfrac, kernel, "
     "in_units, expscale, wtscale, fillstr, fillstr2)"},
    {"tblot", (PyCFunction)tblot, METH_VARARGS | METH_KEYWORDS,
     "tblot(image, pixmap, output, xmin, xmax, ymin, ymax, scale, kscale, "
     "interp, exptime, misval, sinscl)"},
    {"test_cdrizzle", test_cdrizzle, METH_VARARGS,
     "test_cdrizzle(data, weights, pixmap, output_data, output_counts)"},
    {"invert_pixmap", invert_pixmap_wrap, METH_VARARGS,
     "invert_pixmap(pixmap, xyout, bbox)"},
    {"clip_polygon", clip_polygon_wrap, METH_VARARGS, "clip_polygon(p, q)"},
    {NULL, NULL} /* sentinel */
};
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif

/** ---------------------------------------------------------------------------
 */

#if PY_MAJOR_VERSION < 3
PyMODINIT_FUNC
initcdrizzle(void) {
    /* Create the module and add the functions */
    (void)Py_InitModule("cdrizzle", cdrizzle_methods);

    /* Check for errors */
    if (PyErr_Occurred()) Py_FatalError("can't initialize module cdrizzle");

    import_array();
}

#else
static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT,
                                       "cdrizzle",
                                       NULL,
                                       -1,
                                       cdrizzle_methods,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL};

PyMODINIT_FUNC
PyInit_cdrizzle(void) {
    PyObject *m;
    m = PyModule_Create(&moduledef);

    /* Check for errors */
    if (PyErr_Occurred()) Py_FatalError("can't initialize module cdrizzle");

    import_array();
    return m;
}

#endif
