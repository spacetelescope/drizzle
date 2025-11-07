#include <assert.h>
#define _USE_MATH_DEFINES /* needed for MS Windows to define M_PI */
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_21_API_VERSION
#endif
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cdrizzle_util_api

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <numpy/npy_math.h>
#include <numpy/arrayobject.h>
#pragma GCC diagnostic pop

#include "cdrizzlemap.h"
#include "cdrizzleutil.h"

/*****************************************************************
 ERROR HANDLING
*/

void
driz_error_init(struct driz_error_t *error) {
    assert(error);
    error->type = NULL;
    error->last_message[0] = 0;
}

int
driz_error_check(struct driz_error_t *error, const char *message, int test) {
    if (!test) {
        driz_error_set_message(error, message);
        return 1;
    }

    return 0;
}

void
driz_error_set_message(struct driz_error_t *error, const char *message) {
    assert(error);
    assert(message);

    strncpy(error->last_message, message, MAX_DRIZ_ERROR_LEN - 1);
    error->last_message[MAX_DRIZ_ERROR_LEN - 1] = '\0';
}

void
driz_error_format_message(struct driz_error_t *error, const char *format, ...) {
    /* See http://c-faq.com/varargs/vprintf.html
       for an explanation of how all this variable length argument list stuff
       works. */
    va_list argp;

    assert(error);
    assert(format);

    va_start(argp, format);
    (void)vsnprintf(error->last_message, MAX_DRIZ_ERROR_LEN - 1, format, argp);
    va_end(argp);
    error->last_message[MAX_DRIZ_ERROR_LEN - 1] = '\0';
}

void
driz_error_set(struct driz_error_t *error, PyObject *type, const char *format,
               ...) {
    /* See http://c-faq.com/varargs/vprintf.html
       for an explanation of how all this variable length argument list stuff
       works. */
    va_list argp;

    assert(error);
    assert(format);

    va_start(argp, format);
    (void)vsnprintf(error->last_message, MAX_DRIZ_ERROR_LEN - 1, format, argp);
    va_end(argp);

    error->last_message[MAX_DRIZ_ERROR_LEN - 1] = '\0';
    error->type = type;
}

const char *
driz_error_get_message(struct driz_error_t *error) {
    assert(error);

    return error->last_message;
}

int
driz_error_is_set(struct driz_error_t *error) {
    assert(error);

    return error->last_message[0] != 0;
}

void
driz_error_unset(struct driz_error_t *error) {
    assert(error);

    driz_error_init(error);
}

/**
 * py_warning - Issue a Python warning with a formatted message.
 *
 * Constructs a warning message from a printf-style format string and
 * variable arguments, then issues it to the Python warning system via
 * PyErr_WarnEx(..., stacklevel=1).
 *
 * Parameters:
 *   warning_type  - A Python warning exception/type object (e.g.
 * PyExc_Warning). If NULL, PyExc_Warning is used. format        - printf-style
 * format string describing the warning text.
 *   ...           - Arguments matching the format string.
 *
 * Behavior and constraints:
 *   - The formatted message is written into a fixed-size buffer of length
 *     MAX_DRIZ_ERROR_LEN. The implementation writes at most
 *     MAX_DRIZ_ERROR_LEN - 1 bytes using vsnprintf and ensures a terminating
 *     NUL at warn_msg[MAX_DRIZ_ERROR_LEN - 1]. As a result, long messages may
 *     be truncated.
 *   - If vsnprintf reports a formatting error (return value < 1), a fallback
 *     message "Warning message formatting error." is used.
 *   - The function calls PyErr_WarnEx(warning_type, message, 1), so the
 *     warning's stacklevel is 1.
 *   - The caller must hold the Python GIL when calling this function because
 *     it uses Python C-API functions.
 *
 * Return value:
 *   - Returns 0 on success.
 *   - Returns -1 on failure (a Python error/warning handling error), with a
 *     corresponding Python error indicator set.
 *
 * Notes:
 *   - This function is intended for reporting non-fatal warnings to Python
 *     code; it does not raise an exception in the usual sense but routes the
 *     message through Python's warning machinery.
 *   - Because the function uses a local fixed-size buffer, it is safe from
 *     buffer-overflow as implemented, but the resulting message may be
 *     truncated for very long formatted strings.
 */
int
py_warning(PyObject *warning_type, const char *format, ...) {
    char warn_msg[MAX_DRIZ_ERROR_LEN];
    va_list argp;
    va_start(argp, format);
    if (vsnprintf(warn_msg, MAX_DRIZ_ERROR_LEN - 1, format, argp) < 1) {
        strncpy(warn_msg, "Warning message formatting error.",
                MAX_DRIZ_ERROR_LEN - 1);
    }
    va_end(argp);

    if (!warning_type) {
        warning_type = PyExc_Warning;
    }

    warn_msg[MAX_DRIZ_ERROR_LEN - 1] = '\0';

    return PyErr_WarnEx(warning_type, warn_msg, 1);
}

/*****************************************************************
 DATA TYPES
*/
void
driz_param_dump(struct driz_param_t *p) {
    assert(p);

    printf("DRIZZLING PARAMETERS:\n");
    printf("  kernel:          %s\n", kernel_enum2str(p->kernel));
    printf("  pixel_fraction:  %f\n", p->pixel_fraction);
    printf("  exposure_time:   %f\n", p->exposure_time);
    printf("  weight_scale:    %f\n", p->weight_scale);
    printf("  fill_value:      %f\n", p->fill_value);
    printf("  fill_value2:     %f\n", p->fill_value2);
    printf("  do_fill:         %s\n", bool2str(p->do_fill));
    printf("  do_fill2:        %s\n", bool2str(p->do_fill2));
    printf("  in_units:        %s\n", unit_enum2str(p->in_units));
    printf("  out_units:       %s\n", unit_enum2str(p->out_units));
    printf("  iscale:          %f\n", p->iscale);
    printf("  pscale_ratio:    %f\n", p->pscale_ratio);
}

void
driz_param_init(struct driz_param_t *p) {
    assert(p);

    /* Kernel shape and size */
    p->kernel = kernel_square;
    p->pixel_fraction = 1.0;

    /* Exposure time */
    p->exposure_time = 1.0f;

    /* Weight scale */
    p->weight_scale = 1.0f;

    /* Filling */
    p->fill_value = 0.0f;
    p->fill_value2 = 0.0f;
    p->do_fill = 0;
    p->do_fill2 = 0;

    /* CPS / Counts */
    p->in_units = unit_counts;
    p->out_units = unit_counts;

    p->iscale = 1.0f;
    p->pscale_ratio = NPY_NANF; /* Scaling for kernel size. NaN */

    /* Input data */
    p->data = NULL;
    p->data2 = NULL;
    p->dq = NULL;
    p->weights = NULL;
    p->pixmap = NULL;
    p->ndata2 = 0;

    /* Input image dimensions */
    p->in_nx = -1;
    p->in_ny = -1;
    p->out_nx = -1;
    p->out_ny = -1;

    /* Output data */
    p->output_data = NULL;
    p->output_data2 = NULL;
    p->output_counts = NULL;
    p->output_context = NULL;
    p->output_dq = NULL;

    p->nmiss = 0;
    p->nskip = 0;
    p->error = NULL;
}

/*****************************************************************
 STRING TO ENUMERATION CONVERSIONS
*/
static const char *kernel_string_table[] = {
    "square", "gaussian", "point", "turbo", "lanczos2", "lanczos3", NULL};

static const char *unit_string_table[] = {"counts", "cps", NULL};

static const char *interp_string_table[] = {
    "nearest", "linear", "poly3", "poly5", "spline3",
    "sinc",    "lsinc",  "lan3",  "lan5",  NULL};

static const char *bool_string_table[] = {"FALSE", "TRUE", NULL};

static int
str2enum(const char *s, const char *table[], int *result,
         struct driz_error_t *error) {
    const char **it = table;

    assert(s);
    assert(table);
    assert(result);
    assert(error);

    while (*it != NULL) {
        if (strncmp(s, *it, 32) == 0) {
            *result = it - table;
            return 0;
        }
        ++it;
    }

    return 1;
}

int
kernel_str2enum(const char *s, enum e_kernel_t *result,
                struct driz_error_t *error) {
    if (str2enum(s, kernel_string_table, (int *)result, error)) {
        driz_error_format_message(error, "Unknown kernel type '%s'", s);
        return 1;
    }

    return 0;
}

int
unit_str2enum(const char *s, enum e_unit_t *result,
              struct driz_error_t *error) {
    if (str2enum(s, unit_string_table, (int *)result, error)) {
        driz_error_format_message(error, "Unknown unit type '%s'", s);
        return 1;
    }

    return 0;
}

int
interp_str2enum(const char *s, enum e_interp_t *result,
                struct driz_error_t *error) {
    if (str2enum(s, interp_string_table, (int *)result, error)) {
        driz_error_format_message(error, "Unknown interp type '%s'", s);
        return 1;
    }

    return 0;
}

const char *
kernel_enum2str(enum e_kernel_t value) {
    assert(value >= 0 && value < kernel_LAST);

    return kernel_string_table[value];
}

const char *
unit_enum2str(enum e_unit_t value) {
    assert(value >= 0 && value < 2);

    return unit_string_table[value];
}

const char *
interp_enum2str(enum e_interp_t value) {
    assert(value >= 0 && value < interp_LAST);

    return interp_string_table[value];
}

const char *
bool2str(bool_t value) {
    return bool_string_table[value ? 1 : 0];
}

/*****************************************************************
 NUMERICAL UTILITIES
*/
void
create_lanczos_lut(const int kernel_order, const size_t npix, const double del,
                   double *lanczos_lut) {
    double order = (double)kernel_order;
    double poff, x, r;

    assert(lanczos_lut);

    /* Set the first value to avoid arithmetic problems */
    lanczos_lut[0] = 1.0;

    for (size_t i = 1; i < npix; ++i) {
        x = (double)i * del;
        if (x <= order) {
            poff = M_PI * x;
            r = poff / order;
            lanczos_lut[i] = sin(poff) * sin(r) / (poff * r);
        } else {
            lanczos_lut[i] = 0.0;
        }
    }
}

void
put_fill(struct driz_param_t *p, int fill, int fill2) {
    integer_t i, j, k, osize[2], osize2[2], csize[2];

    assert(p);
    if (!p->output_data2) {
        fill2 = 0;
    }
    if (!fill && !fill2) {
        return;
    }

    get_dimensions(p->output_data, osize);
    get_dimensions(p->output_counts, csize);
    if (osize[0] != csize[0] || osize[1] != csize[1]) {
        driz_error_set(p->error, PyExc_ValueError,
                       "Mismatch between output_data and output_counts "
                       "array size.");
        return;
    }

    if (fill2) {
        for (k = 0; k < p->ndata2; ++k) {
            get_dimensions(p->output_data2[k], osize2);
            if ((osize2[0] != osize[0]) || (osize2[1] != osize[1])) {
                driz_error_set(p->error, PyExc_ValueError,
                               "Mismatch between output_data and output_data2 "
                               "array size.");
                return;
            }
        }
    }

    if (fill && !fill2) {
        for (j = 0; j < osize[1]; ++j) {
            for (i = 0; i < osize[0]; ++i) {
                if (get_pixel(p->output_counts, i, j) == 0.0) {
                    set_pixel(p->output_data, i, j, p->fill_value);
                }
            }
        }
    } else if (!fill && fill2) {
        for (j = 0; j < osize[1]; ++j) {
            for (i = 0; i < osize[0]; ++i) {
                if (get_pixel(p->output_counts, i, j) == 0.0) {
                    for (k = 0; k < p->ndata2; ++k) {
                        set_pixel(p->output_data2[k], i, j, p->fill_value2);
                    }
                }
            }
        }
    } else { /* fill && fill2 */
        for (j = 0; j < osize[1]; ++j) {
            for (i = 0; i < osize[0]; ++i) {
                if (get_pixel(p->output_counts, i, j) == 0.0) {
                    set_pixel(p->output_data, i, j, p->fill_value);
                    if (p->output_data2) {
                        for (k = 0; k < p->ndata2; ++k) {
                            set_pixel(p->output_data2[k], i, j, p->fill_value2);
                        }
                    }
                }
            }
        }
    }
}

double
mgf2(double lambda) {
    double sig, sig2;

    sig = 1.0e7 / lambda;
    sig2 = sig * sig;

    return sqrt(1.0 + 2.590355e10 / (5.312993e10 - sig2) +
                4.4543708e9 / (11.17083e9 - sig2) +
                4.0838897e5 / (1.766361e5 - sig2));
}
