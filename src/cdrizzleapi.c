#include <Python.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <stdio.h>

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_21_API_VERSION
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/npy_math.h>
#pragma GCC diagnostic pop

#include "cdrizzleblot.h"
#include "cdrizzlebox.h"
#include "cdrizzlemap.h"
#include "cdrizzleutil.h"
#include "tests/drizzletest.h"

static PyObject *gl_Error;
FILE *driz_log_handle = NULL;

static PyArrayObject *
ensure_array(PyObject *obj, int npy_type, int min_depth, int max_depth, int *is_copy)
{
    if (PyArray_CheckExact(obj) && PyArray_IS_C_CONTIGUOUS((PyArrayObject *) obj) &&
        PyArray_TYPE((PyArrayObject *) obj) == npy_type) {
        *is_copy = 0;
        return (PyArrayObject *) obj;
    } else {
        *is_copy = 1;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
        PyArray_Descr *dtype_descr =
            (PyArray_Descr *) ((void *) PyArray_DescrFromType((int) npy_type));
#pragma GCC diagnostic pop

        if (dtype_descr == NULL) {
            PyErr_SetString(PyExc_TypeError, "Invalid numpy type for array conversion.");
            *is_copy = 0;
            return NULL;
        }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
        return (PyArrayObject *) PyArray_FromAny(
            obj, dtype_descr, min_depth, max_depth, NPY_ARRAY_DEFAULT | NPY_ARRAY_ENSUREARRAY,
            NULL);
#pragma GCC diagnostic pop
    }
}

static int
process_array_list(
    PyObject *list, integer_t *nx, integer_t *ny, const char *name, PyArrayObject ***arrays,
    int *nmax, int allow_none, int *n_none, int **free_arrays, struct driz_error_t *error)
{
    npy_intp *ndim;
    int inx, iny;
    PyObject *list_elem = NULL;
    PyArrayObject **arr_list = NULL;
    PyArrayObject *arr = NULL;
    int i, at_least_one;
    int cpy;

    *arrays = NULL;
    *free_arrays = NULL;
    *nmax = 0;
    if (n_none) {
        *n_none = 0;
    }

    if (list == NULL || list == Py_None) {
        driz_error_set(error, PyExc_ValueError, "Array list '%s' is None or NULL.", name);
        return 1;
    }

    if (PyArray_CheckExact(list)) {
        if (!(arr_list = (PyArrayObject **) malloc(sizeof(PyArrayObject *)))) {
            driz_error_set(error, PyExc_MemoryError, "Memory allocation failed.");
            return 1;
        }
        if (!(*free_arrays = (int *) malloc(sizeof(int)))) {
            driz_error_set(error, PyExc_MemoryError, "Memory allocation failed.");
            return 1;
        }
        arr = ensure_array(list, NPY_FLOAT, 2, 2, &cpy);
        (*free_arrays)[0] = cpy;
        if (!arr) {
            driz_error_set(error, PyExc_ValueError, "Invalid '%s' array.", name);
            free(arr_list);
            return 1;
        }
        *arr_list = arr;
        *nmax = 1;
        *n_none = 0;
        ndim = PyArray_DIMS(arr);
        *nx = (int) ndim[1];
        *ny = (int) ndim[0];
        *arrays = arr_list;
        return 0;

    } else if (!PyList_Check(list) && !PyTuple_Check(list)) {
        driz_error_set(
            error, PyExc_TypeError,
            "Argument '%s' is not a list or a tuple of numpy.ndarray or None.", name);
    }

    at_least_one = 0;

    if (!(*nmax = PySequence_Size(list))) {
        return 0;
    }

    arr_list = (PyArrayObject **) calloc(*nmax, sizeof(PyArrayObject *));
    if (!arr_list) {
        driz_error_set(error, PyExc_MemoryError, "Memory allocation failed.");
        return 1;
    }
    if (!(*free_arrays = (int *) calloc(*nmax, sizeof(int)))) {
        driz_error_set(error, PyExc_MemoryError, "Memory allocation failed.");
        return 1;
    }

    for (i = 0; i < *nmax; ++i) {
        if (!(list_elem = PySequence_GetItem(list, i))) {
            driz_error_set(
                error, PyExc_RuntimeError, "Error retrieving array %d from the '%s' list.", i,
                name);
            goto _exit_on_err;
        }

        if (list_elem == Py_None) {
            if (allow_none) {
                if (n_none) {
                    (*n_none)++;
                }
                arr_list[i] = NULL;
                (*free_arrays)[i] = 0;
                Py_XDECREF(list_elem);
                continue;
            } else {
                *n_none = 1;
                driz_error_set(
                    error, PyExc_ValueError,
                    "Element %d of '%s' list is None which is not allowed.", i, name);
                goto _exit_on_err;
            }
        } else {
            arr = ensure_array(list_elem, NPY_FLOAT, 2, 2, &cpy);
            if (!arr) {
                driz_error_set(
                    error, PyExc_ValueError, "Invalid array in '%s' at position %d.", name, i);
                goto _exit_on_err;
            }
            (*free_arrays)[i] = cpy;
            at_least_one = 1;
        }
        Py_XDECREF(list_elem);

        arr_list[i] = arr;

        if (*nx < 0) {
            ndim = PyArray_DIMS(arr);
            *nx = (int) ndim[1];
            *ny = (int) ndim[0];
        } else {
            ndim = PyArray_DIMS(arr);
            inx = (int) ndim[1];
            iny = (int) ndim[0];
            if ((*nx != inx) || (*ny != iny)) {
                driz_error_set(
                    error, PyExc_ValueError, "Inconsistent image shape in the '%s' image list.",
                    name);
                goto _exit_on_err;
            }
        }
    }

    if (!at_least_one && !allow_none) {
        free(arr_list);
        free(*free_arrays);
        arr_list = NULL;
        *nmax = 0;
    }

    *arrays = arr_list;
    return 0;

_exit_on_err:

    Py_XDECREF(list_elem);
    if (arr_list) {
        for (i = 0; i < *nmax; ++i) {
            if (free_arrays && (*free_arrays)[i]) {
                Py_XDECREF(arr_list[i]);
            }
        }
        free(arr_list);
    }
    if (*free_arrays) {
        free(*free_arrays);
    }
    return 1;
}

/** ---------------------------------------------------------------------------
 * Top level function for drizzling, interfaces with python code
 */
static PyObject *
tdriz(PyObject *self, PyObject *args, PyObject *keywords)
{
    (void) self;

    const char *kwlist[] = {
        "input",    "weights", "pixmap",       "output",   "counts",  "context", "input2",
        "output2",  "dq",      "outdq",        "uniqid",   "xmin",    "xmax",    "ymin",
        "ymax",     "iscale",  "pscale_ratio", "scale",    "pixfrac", "kernel",  "in_units",
        "expscale", "wtscale", "fillstr",      "fillstr2", NULL};

    /* Arguments in the order they appear */
    PyObject *oimg, *owei, *pixmap, *oout, *owht, *ocon, *odq = NULL;
    PyObject *oimg2 = NULL, *oout2 = NULL, *ooutdq = NULL, *opscale_ratio = NULL, *oscale = NULL;
    int i, n_none;
    int nsq_args, nsq_arr = 0, nsq_arr_out = 0;
    int *free_arrays2 = NULL, *free_out_arrays2 = NULL;

    integer_t uniqid = 1;
    integer_t xmin = 0;
    integer_t xmax = 0;
    integer_t ymin = 0;
    integer_t ymax = 0;
    float iscale = 1.0f;
    float pscale_ratio = NPY_NANF;
    float scale = NPY_NANF;
    double pfract = 1.0;
    char *kernel_str = "square";
    char *inun_str = "cps";
    float expin = 1.0;
    float wtscl = 1.0;
    char *fillstr = "INDEF";
    char *fillstr2 = "INDEF";

    /* Derived values */

    PyArrayObject *img = NULL, *wei = NULL, *out = NULL, *wht = NULL, *con = NULL, *map = NULL,
                  *dq = NULL, *outdq = NULL;

    int free_img = 0, free_wei = 0, free_out = 0, free_wht = 0;
    int free_con = 0, free_map = 0, free_dq = 0, free_outdq = 0;

    PyArrayObject **img2_list = NULL, **out2_list = NULL;

    enum e_kernel_t kernel;
    enum e_unit_t inun;
    char *fillstr_end;
    bool_t do_fill, do_fill2;
    float fill_value, fill_value2;
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
            args, keywords, "OOOOOO|OOOOiiiiifOOdssffss:tdriz", (char **) kwlist, &oimg, &owei,
            &pixmap, &oout, &owht, &ocon,        /* OOOOOO */
            &oimg2, &oout2, &odq, &ooutdq,       /* OOOO */
            &uniqid, &xmin, &xmax, &ymin, &ymax, /* iiiii */
            &iscale, &opscale_ratio, &oscale,    /* fOO */
            &pfract, &kernel_str, &inun_str,     /* dss */
            &expin, &wtscl, &fillstr, &fillstr2) /* ffss */
    ) {
        return NULL;
    }

    if (oscale != NULL && oscale != Py_None) {
        scale = (float) PyFloat_AsDouble(oscale);
        if (PyErr_Occurred()) {
            driz_error_set_message(&error, "Argument 'scale' is not a number.");
            goto _exit;
        }
        if (scale <= 0.0f || !isfinite(scale)) {
            driz_error_set_message(&error, "Argument 'scale' must be positive and finite.");
            goto _exit;
        }
        iscale = scale * scale;
        pscale_ratio = scale;
        if (py_warning(
                PyExc_DeprecationWarning,
                "Argument 'scale' has been deprecated since version 3.0 "
                "and it will be removed in a future release. "
                "Use 'iscale' and 'pscale_ratio' instead and set "
                "iscale=pscale_ratio**2 to achieve the same effect as with "
                "'scale'.") != 0) {
            goto _exit;
        }

    } else {
        if (iscale <= 0.0f || !isfinite(iscale)) {
            driz_error_set_message(&error, "Argument 'iscale' must be positive and finite.");
            goto _exit;
        }

        if (opscale_ratio != NULL && opscale_ratio != Py_None) {
            pscale_ratio = (float) PyFloat_AsDouble(opscale_ratio);
            if (PyErr_Occurred()) {
                driz_error_set_message(&error, "Argument 'pscale_ratio' is not a number.");
                goto _exit;
            }
            if (pscale_ratio <= 0.0f || !isfinite(pscale_ratio)) {
                driz_error_set_message(
                    &error, "Argument 'pscale_ratio' must be positive and finite.");
                goto _exit;
            }
        }
    }

    /* Get raw C-array data */
    img = ensure_array(oimg, NPY_FLOAT, 2, 2, &free_img);
    if (!img) {
        driz_error_set_message(&error, "Invalid input array");
        goto _exit;
    }

    wei = ensure_array(owei, NPY_FLOAT, 2, 2, &free_wei);
    if (!wei) {
        driz_error_set_message(&error, "Invalid weights array");
        goto _exit;
    }

    map = ensure_array(pixmap, NPY_DOUBLE, 3, 3, &free_map);
    if (!map) {
        driz_error_set_message(&error, "Invalid pixmap array");
        goto _exit;
    }

    out = ensure_array(oout, NPY_FLOAT, 2, 2, &free_out);
    if (!out) {
        driz_error_set_message(&error, "Invalid output array");
        goto _exit;
    }

    wht = ensure_array(owht, NPY_FLOAT, 2, 2, &free_wht);
    if (!wht) {
        driz_error_set_message(&error, "Invalid counts array");
        goto _exit;
    }

    if (ocon == Py_None) {
        con = NULL;
    } else {
        con = ensure_array(ocon, NPY_INT32, 2, 2, &free_con);
        if (!con) {
            driz_error_set_message(&error, "Invalid context array");
            goto _exit;
        }
    }

    if (odq == Py_None || odq == NULL) {
        dq = NULL;
    } else {
        dq = ensure_array(odq, NPY_UINT32, 2, 2, &free_dq);
        if (!dq) {
            driz_error_set_message(&error, "Invalid input DQ array");
            goto _exit;
        }
    }

    if (ooutdq == Py_None || ooutdq == NULL) {
        if (dq != NULL) {
            driz_error_set_message(&error, "When 'dq' is provided, 'outdq' must also be provided.");
            goto _exit;
        }
        outdq = NULL;
    } else {
        outdq = ensure_array(ooutdq, NPY_UINT32, 2, 2, &free_outdq);
        if (!outdq) {
            driz_error_set_message(&error, "Invalid output DQ array");
            goto _exit;
        }
    }

    /* Convert the fill value string */
    if (fillstr == NULL || *fillstr == 0 || strncmp(fillstr, "INDEF", 6) == 0 ||
        strncmp(fillstr, "indef", 6) == 0) {
        do_fill = 0;
        fill_value = 0.0;

    } else if (strncmp(fillstr, "NaN", 4) == 0 || strncmp(fillstr, "nan", 4) == 0) {
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

    if (fillstr2 == NULL || *fillstr2 == 0 || strncmp(fillstr2, "INDEF", 6) == 0 ||
        strncmp(fillstr2, "indef", 6) == 0) {
        do_fill2 = 0;
        fill_value2 = 0.0;

    } else if (strncmp(fillstr2, "NaN", 4) == 0 || strncmp(fillstr2, "nan", 4) == 0) {
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
    size[0] = (integer_t) ndim[1];
    size[1] = (integer_t) ndim[0];

    if (size[0] != inx || size[1] != iny) {
        if (snprintf(
                warn_msg, 128,
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
            if (snprintf(
                    warn_msg, 128,
                    "Weights array dimensions (%d, %d) != input "
                    "dimensions (%d, %d).",
                    size[0], size[1], inx, iny) < 1) {
                strcpy(warn_msg, "Weights array dimensions != input dimensions.");
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
        if (snprintf(
                warn_msg, 128,
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
            if (snprintf(
                    warn_msg, 128,
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
    nsq_args =
        ((int) (oimg2 != NULL && oimg2 != Py_None)) + ((int) (oout2 != NULL && oout2 != Py_None));
    if (nsq_args == 2) {
        nx = inx;
        ny = iny;
        if (process_array_list(
                oimg2, &nx, &ny, "input2", &img2_list, &nsq_arr, 1, &n_none, &free_arrays2,
                &error)) {
            goto _exit;
        }
        if (n_none == nsq_arr && img2_list) {
            free(img2_list);
            nsq_arr = 0;
        }

        if (nsq_arr) {
            if (nx != inx || ny != iny) {
                driz_error_set_message(
                    &error, "'input2' arrays must have the same "
                            "dimensions as the 'input' array.");
                goto _exit;
            }

            nx = onx;
            ny = ony;
            if (process_array_list(
                    oout2, &nx, &ny, "output2", &out2_list, &nsq_arr_out, 0, NULL,
                    &free_out_arrays2, &error)) {
                goto _exit;
            }
            if (nx != onx || ny != ony) {
                driz_error_set_message(
                    &error, "'output2' arrays must have the same "
                            "dimensions as the 'output' array.");
                goto _exit;
            }

            if (nsq_arr != nsq_arr_out) {
                driz_error_set_message(
                    &error, "The number of 'output2' arrays must match "
                            "the number of 'input2' arrays.");
                goto _exit;
            }
        }
    } else if (nsq_args == 1) {
        driz_error_set_message(
            &error, "'input2' and 'output2' must both be either None, "
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
        driz_error_set_message(&error, "No or too few valid pixels in the pixel map.");
        goto _exit;
    }

    /* Convert strings to enumerations */
    if (kernel_str2enum(kernel_str, &kernel, &error) || unit_str2enum(inun_str, &inun, &error)) {
        goto _exit;
    }

    if (kernel == kernel_gaussian || kernel == kernel_lanczos2 || kernel == kernel_lanczos3) {
        if (snprintf(warn_msg, 128, "Kernel '%s' is not a flux-conserving kernel.", kernel_str) <
            1) {
            strcpy(warn_msg, "Selected kernel is not a flux-conserving kernel.");
        }
        PyErr_WarnEx(PyExc_Warning, warn_msg, 1);
    }

    if (pfract <= 0.001) {
        if (snprintf(
                warn_msg, 128,
                "Kernel reset to 'point' due to input 'pixfrac' "
                " being too small.") < 1) {
            strcpy(
                warn_msg, "Kernel reset to 'point' due to input 'pixfrac' "
                          " being too small.");
        }
        PyErr_WarnEx(PyExc_Warning, warn_msg, 1);
        kernel_str2enum("point", &kernel, &error);
    }

    /* If the input image is not in CPS we need to divide by the exposure */
    if (inun != unit_cps) {
        iscale /= expin;
    }

    /* Setup reasonable defaults for drizzling */
    driz_param_init(&p);

    p.data = img;
    p.dq = dq;
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
    p.iscale = iscale;
    p.pscale_ratio = pscale_ratio;
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
    p.output_dq = outdq;
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
    if (driz_error_check(&error, "exposure time must be > 0", p.exposure_time > 0.0f)) {
        goto _exit;
    }
    if (driz_error_check(&error, "weight scale must be > 0", p.weight_scale > 0.0f)) {
        goto _exit;
    }

    if (dobox(&p)) {
        goto _exit;
    }

    /* Put in the fill values (if defined) */
    if (do_fill) {
        put_fill(&p, do_fill, do_fill2);
    }

_exit:
    driz_log_message("ending tdriz");
    driz_log_close(driz_log_handle);
    if (free_con) {
        Py_XDECREF(con);
    }
    if (free_img) {
        Py_XDECREF(img);
    }
    if (free_wei) {
        Py_XDECREF(wei);
    }
    if (free_out) {
        Py_XDECREF(out);
    }
    if (free_wht) {
        Py_XDECREF(wht);
    }
    if (free_map) {
        Py_XDECREF(map);
    }
    if (free_dq) {
        Py_XDECREF(dq);
    }
    if (free_outdq) {
        Py_XDECREF(outdq);
    }

    if (nsq_arr > 0 && img2_list) {
        for (i = 0; i < nsq_arr; ++i) {
            if (free_arrays2[i]) {
                Py_XDECREF(img2_list[i]);
            }
        }
        free(img2_list);
        free(free_arrays2);
    }

    if (nsq_arr_out > 0 && out2_list) {
        for (i = 0; i < nsq_arr_out; ++i) {
            if (free_out_arrays2[i]) {
                Py_XDECREF(out2_list[i]);
            }
        }
        free(out2_list);
        free(free_out_arrays2);
    }

    if (driz_error_is_set(&error)) {
        if (error.type == NULL) {
            error.type = PyExc_ValueError; /* default error type */
        }
        PyErr_SetString(error.type, driz_error_get_message(&error));
        return NULL;
    } else {
        return Py_BuildValue("sii", "Callable C-based DRIZZLE Version 2.1.0", p.nmiss, p.nskip);
    }
}

/** ---------------------------------------------------------------------------
 * Top level function for blotting, interfaces with python code
 */

static PyObject *
tblot(PyObject *self, PyObject *args, PyObject *keywords)
{
    (void) self;

    const char *kwlist[] = {"source",  "pixmap", "output", "xmin",  "xmax",   "ymin",
                            "ymax",    "iscale", "kscale", "scale", "interp", "exptime",
                            "fillval", "misval", "sinscl", NULL};

    /* Arguments in the order they appear */
    PyObject *oimg, *pixmap, *oout;
    PyObject *oscale = NULL, *okscale = NULL, *oiscale = NULL;
    PyObject *oef = NULL, *omisval = NULL, *ofillval = NULL;
    long xmin = 0;
    long xmax = 0;
    long ymin = 0;
    long ymax = 0;
    float scale = 1.0f;
    float iscale = 1.0f;
    char *interp_str = "poly5";
    float ef = 1.0f;
    float fillval;
    float sinscl = 1.0f;

    PyArrayObject *img = NULL, *out = NULL, *map = NULL;
    enum e_interp_t interp;
    int istat = 0;
    struct driz_error_t error;
    struct driz_param_t p;
    integer_t psize[2], osize[2];
    char warn_msg[128];
    int free_img = 0, free_out = 0, free_map = 0;

    driz_log_handle = driz_log_init(driz_log_handle);
    driz_log_message("starting tblot");
    driz_error_init(&error);

    if (!PyArg_ParseTupleAndKeywords(
            args, keywords, "OOO|llllOOOsOOOf:tblot", (char **) kwlist, /* */
            &oimg, &pixmap, &oout,                                      /* OOO */
            &xmin, &xmax, &ymin, &ymax,                                 /* llll */
            &oiscale, &okscale, &oscale, &interp_str, &oef,             /* fOOsO */
            &ofillval, &omisval, &sinscl)                               /* OOf */
    ) {
        return NULL;
    }

    if (oscale != NULL && !Py_IsNone(oscale)) {
        scale = (float) PyFloat_AsDouble(oscale);

        if (PyErr_Occurred()) {
            driz_error_set_message(&error, "Argument 'scale' is not a number.");
            goto _exit;
        }

        if (scale <= 0.0f || !isfinite(scale)) {
            driz_error_set_message(&error, "Argument 'scale' must be positive and finite.");
            goto _exit;
        }
        iscale = 1.0 / (scale * scale);

        if (py_warning(
                PyExc_DeprecationWarning, "Argument 'scale' is deprecated, use 'iscale' "
                                          "instead and set it to 1.0 / (scale*scale).") != 0) {
            goto _exit;
        }

    } else {
        if (oiscale == NULL || Py_IsNone(oiscale)) {
            iscale = 1.0f;
        } else {
            iscale = (float) PyFloat_AsDouble(oiscale);

            if (PyErr_Occurred()) {
                driz_error_set_message(&error, "Argument 'iscale' is not a number.");
                goto _exit;
            }
        }
        if (iscale <= 0.0f || !isfinite(iscale)) {
            driz_error_set_message(&error, "Argument 'iscale' must be positive and finite.");
            goto _exit;
        }
    }

    if (okscale != NULL && !Py_IsNone(okscale)) {
        if (py_warning(
                PyExc_DeprecationWarning,
                "Argument 'kscale' has been deprecated and it will be "
                "removed in a future version. It is no longer used by "
                "the blotting algorithm and can be safely ignored.") != 0) {
            goto _exit;
        }
    }

    if (omisval != NULL && !Py_IsNone(omisval)) {
        if (py_warning(
                PyExc_DeprecationWarning,
                "Argument 'misval' has been deprecated and has been "
                "replaced by 'fillval' to achieve the same effect.") != 0) {
            goto _exit;
        }

        fillval = (float) PyFloat_AsDouble(omisval);

        if (ofillval != NULL && !Py_IsNone(ofillval)) {
            driz_error_set_message(
                &error, "Argument 'fillval' should not be set when 'misval' is set.");
            goto _exit;
        }
    } else {
        if (ofillval != NULL && !Py_IsNone(ofillval)) {
            fillval = (float) PyFloat_AsDouble(ofillval);

            if (PyErr_Occurred()) {
                driz_error_set_message(&error, "Argument 'fillval' is not a number.");
                goto _exit;
            }
        } else {
            fillval = 0.0f;
        }
    }

    if (oef != NULL && !Py_IsNone(oef)) {
        if (py_warning(
                PyExc_DeprecationWarning, "Argument 'exptime' has been deprecated and it will be "
                                          "removed in a future version. Use 'iscale' to achieve "
                                          "the same.") != 0) {
            goto _exit;
        }

        ef = (float) PyFloat_AsDouble(oef);

        if (PyErr_Occurred()) {
            driz_error_set_message(&error, "Argument 'exptime' is not a number.");
            goto _exit;
        }
        if (ef <= 0.0f || !isfinite(ef)) {
            driz_error_set_message(&error, "Argument 'exptime' must be positive and finite.");
            goto _exit;
        }
    } else {
        ef = 1.0f;
    }

    img = ensure_array(oimg, NPY_FLOAT, 2, 2, &free_img);
    if (!img) {
        driz_error_set_message(&error, "Invalid input array");
        goto _exit;
    }

    map = ensure_array(pixmap, NPY_DOUBLE, 3, 3, &free_map);
    if (!map) {
        driz_error_set_message(&error, "Invalid pixmap array");
        goto _exit;
    }

    out = ensure_array(oout, NPY_FLOAT, 2, 2, &free_out);
    if (!out) {
        driz_error_set_message(&error, "Invalid output array");
        goto _exit;
    }

    if (interp_str2enum(interp_str, &interp, &error)) {
        goto _exit;
    }
    if (strncmp(interp_str, "sinc", 4) == 0) {
        if (py_warning(
                PyExc_DeprecationWarning, "The \"sinc\" interpolation is currently investigated for"
                                          "possible issues and its use is not recommended.") != 0) {
            goto _exit;
        }
    }

    get_dimensions(map, psize);
    get_dimensions(out, osize);

    if (psize[0] != osize[0] || psize[1] != osize[1]) {
        if (snprintf(
                warn_msg, 128,
                "Pixel map dimensions (%d, %d) != output dimensions "
                "(%d, %d).",
                psize[0], psize[1], osize[0], osize[1]) < 1) {
            strcpy(warn_msg, "Pixel map dimensions != output dimensions.");
        }
        driz_error_set_message(&error, warn_msg);
        goto _exit;
    }

    if (xmax == 0) {
        xmax = osize[0];
    }
    if (ymax == 0) {
        ymax = osize[1];
    }

    driz_param_init(&p);

    p.data = img;
    p.output_data = out;
    p.xmin = xmin;
    p.xmax = xmax;
    p.ymin = ymin;
    p.ymax = ymax;
    p.iscale = iscale;
    p.in_units = unit_cps;
    p.interpolation = interp;
    p.ef = ef;
    p.fill_value = fillval;
    p.sinscl = sinscl;
    p.pixmap = map;
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
    if (driz_error_check(&error, "exposure time must be > 0", p.ef > 0.0)) {
        goto _exit;
    }

    if (doblot(&p)) {
        goto _exit;
    }

_exit:
    driz_log_message("ending tblot");
    driz_log_close(driz_log_handle);
    if (free_img) {
        Py_XDECREF(img);
    }
    if (free_out) {
        Py_XDECREF(out);
    }
    if (free_map) {
        Py_XDECREF(map);
    }

    if (driz_error_is_set(&error)) {
        if (strcmp(driz_error_get_message(&error), "<PYTHON>") != 0) {
            PyErr_SetString(PyExc_Exception, driz_error_get_message(&error));
        }
        return NULL;
    } else {
        return Py_BuildValue("i", istat);
    }
}

/** ---------------------------------------------------------------------------
 * Top level of C unit tests, interfaces with python code
 */

static PyObject *
test_cdrizzle(PyObject *self, PyObject *args)
{
    (void) self;

    PyObject *data, *weights, *pixmap, *output_data, *output_counts, *output_context;
    PyArrayObject *dat, *wei, *map, *odat, *ocnt, *ocon;
    int argc = 1;
    char *argv[] = {"utest_cdrizzle", NULL};
    int free_data = 0, free_wei = 0, free_map = 0;
    int free_odat = 0, free_ocnt = 0, free_ocon = 0;

    if (!PyArg_ParseTuple(
            args, "OOOOOO:test_cdrizzle", &data, &weights, &pixmap, &output_data, &output_counts,
            &output_context)) {
        return NULL;
    }

    dat = ensure_array(data, NPY_FLOAT, 2, 2, &free_data);
    if (!dat) {
        return PyErr_Format(gl_Error, "Invalid data array.");
    }

    wei = ensure_array(weights, NPY_FLOAT, 2, 2, &free_wei);
    if (!wei) {
        return PyErr_Format(gl_Error, "Invalid weghts array.");
    }

    map = ensure_array(pixmap, NPY_DOUBLE, 2, 4, &free_map);
    if (!map) {
        return PyErr_Format(gl_Error, "Invalid pixmap.");
    }

    odat = ensure_array(output_data, NPY_FLOAT, 2, 2, &free_odat);
    if (!odat) {
        return PyErr_Format(gl_Error, "Invalid output data array.");
    }

    ocnt = ensure_array(output_counts, NPY_FLOAT, 2, 2, &free_ocnt);
    if (!ocnt) {
        return PyErr_Format(gl_Error, "Invalid output counts array.");
    }

    ocon = ensure_array(output_context, NPY_INT32, 2, 2, &free_ocon);
    if (!ocon) {
        return PyErr_Format(gl_Error, "Invalid context array");
    }

    set_test_arrays(dat, wei, map, odat, ocnt, ocon);
    utest_cdrizzle(argc, argv);

    if (free_data) {
        Py_XDECREF(dat);
    }
    if (free_wei) {
        Py_XDECREF(wei);
    }
    if (free_map) {
        Py_XDECREF(map);
    }
    if (free_odat) {
        Py_XDECREF(odat);
    }
    if (free_ocnt) {
        Py_XDECREF(ocnt);
    }
    if (free_ocon) {
        Py_XDECREF(ocon);
    }

    return Py_BuildValue("");
}

static PyObject *
invert_pixmap_wrap(PyObject *self, PyObject *args)
{
    (void) self;

    PyObject *pixmap, *xyout, *bbox;
    PyArrayObject *xyout_arr, *pixmap_arr, *bbox_arr;
    struct driz_param_t par;
    double *xy, *xyin;
    npy_intp *ndim, xyin_dim = 2;
    const double half = 0.5 - DBL_EPSILON;
    int free_xyout = 0, free_pixmap = 0, free_bbox = 0;

    xyin = (double *) malloc(2 * sizeof(double));

    if (!PyArg_ParseTuple(args, "OOO:invpixmap", &pixmap, &xyout, &bbox)) {
        return NULL;
    }

    xyout_arr = ensure_array(xyout, NPY_DOUBLE, 1, 1, &free_xyout);
    if (!xyout_arr) {
        return PyErr_Format(gl_Error, "Invalid xyout array.");
    }

    pixmap_arr = ensure_array(pixmap, NPY_DOUBLE, 3, 3, &free_pixmap);
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
        bbox_arr = ensure_array(bbox, NPY_DOUBLE, 2, 2, &free_bbox);
        if (!bbox_arr) {
            return PyErr_Format(gl_Error, "Invalid input bounding box.");
        }
        par.xmin = (integer_t) (*(double *) PyArray_GETPTR2(bbox_arr, 0, 0) - half);
        par.xmax = (integer_t) (*(double *) PyArray_GETPTR2(bbox_arr, 0, 1) + half);
        par.ymin = (integer_t) (*(double *) PyArray_GETPTR2(bbox_arr, 1, 0) - half);
        par.ymax = (integer_t) (*(double *) PyArray_GETPTR2(bbox_arr, 1, 1) + half);
    }

    xy = (double *) PyArray_DATA(xyout_arr);

    if (invert_pixmap(&par, xy[0], xy[1], &xyin[0], &xyin[1])) {
        return Py_BuildValue("");
    }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
    PyArrayObject *arr =
        (PyArrayObject *) PyArray_SimpleNewFromData(1, &xyin_dim, NPY_DOUBLE, xyin);
#pragma GCC diagnostic pop

    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_OWNDATA);
    if (free_xyout) {
        Py_XDECREF(xyout_arr);
    }
    if (free_pixmap) {
        Py_XDECREF(pixmap_arr);
    }
    if (free_bbox) {
        Py_XDECREF(bbox_arr);
    }

    return Py_BuildValue("N", arr);
}

static PyObject *
clip_polygon_wrap(PyObject *self, PyObject *args)
{
    (void) self;

    int k;
    PyObject *pin, *qin;
    PyArrayObject *pin_arr, *qin_arr;
    struct polygon p, q, pq;
    PyObject *list, *tuple;
    int free_pin = 0, free_qin = 0;

    if (!PyArg_ParseTuple(args, "OO:clip_polygon", &pin, &qin)) {
        return NULL;
    }

    pin_arr = ensure_array(pin, NPY_DOUBLE, 2, 2, &free_pin);
    if (!pin_arr) {
        return PyErr_Format(gl_Error, "Invalid P.");
    }

    qin_arr = ensure_array(qin, NPY_DOUBLE, 2, 2, &free_qin);
    if (!qin_arr) {
        return PyErr_Format(gl_Error, "Invalid Q.");
    }

    p.npv = PyArray_SHAPE(pin_arr)[0];
    for (k = 0; k < p.npv; ++k) {
        p.v[k].x = *((double *) PyArray_GETPTR2(pin_arr, k, 0));
        p.v[k].y = *((double *) PyArray_GETPTR2(pin_arr, k, 1));
    }

    q.npv = PyArray_SHAPE(qin_arr)[0];
    for (k = 0; k < q.npv; ++k) {
        q.v[k].x = *((double *) PyArray_GETPTR2(qin_arr, k, 0));
        q.v[k].y = *((double *) PyArray_GETPTR2(qin_arr, k, 1));
    }

    clip_polygon_to_window(&p, &q, &pq);

    list = PyList_New(pq.npv);

    for (k = 0; k < pq.npv; ++k) {
        tuple = PyTuple_New(2);
        PyTuple_SetItem(tuple, 0, PyFloat_FromDouble(pq.v[k].x));
        PyTuple_SetItem(tuple, 1, PyFloat_FromDouble(pq.v[k].y));
        PyList_SetItem(list, k, tuple);
    }
    if (free_pin) {
        Py_XDECREF(pin_arr);
    }
    if (free_qin) {
        Py_XDECREF(qin_arr);
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
    {"tdriz", (PyCFunction) (void (*)(void)) tdriz, METH_VARARGS | METH_KEYWORDS,
     "tdriz(image, weights, pixmap, output, counts, context, image2, "
     "output2, dq, outdq, uniqid, xmin, xmax, ymin, ymax, iscale, "
     "pscale_ratio, pixfrac, kernel, in_units, expscale, wtscale, fillstr, "
     "fillstr2)"},
    {"tblot", (PyCFunction) (void (*)(void))(PyCFunctionWithKeywords) tblot,
     METH_VARARGS | METH_KEYWORDS,
     "tblot(image, pixmap, output, xmin, xmax, ymin, ymax, iscale, "
     "interp, exptime, fillval, misval, sinscl)"},
    {"test_cdrizzle", (PyCFunction) test_cdrizzle, METH_VARARGS,
     "test_cdrizzle(data, weights, pixmap, output_data, output_counts)"},
    {"invert_pixmap", (PyCFunction) invert_pixmap_wrap, METH_VARARGS,
     "invert_pixmap(pixmap, xyout, bbox)"},
    {"clip_polygon", (PyCFunction) clip_polygon_wrap, METH_VARARGS, "clip_polygon(p, q)"},
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
initcdrizzle(void)
{
    /* Create the module and add the functions */
    (void) Py_InitModule("cdrizzle", cdrizzle_methods);

    /* Check for errors */
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module cdrizzle");
    }

    import_array();
}

#else
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "cdrizzle", NULL, -1, cdrizzle_methods, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC
PyInit_cdrizzle(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);

    /* Check for errors */
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module cdrizzle");
    }

    import_array();
    return m;
}

#endif
