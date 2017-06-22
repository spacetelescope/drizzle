#ifndef CDRIZZLEUTIL_H
#define CDRIZZLEUTIL_H
#include "driz_portability.h"

#include <Python.h>
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif
#include <numpy/arrayobject.h>

#include <assert.h>
#include <errno.h>
#define _USE_MATH_DEFINES       /* needed for MS Windows to define M_PI */ 
#include <math.h>
#if __STDC_VERSION__ >= 199901L
#include <stdint.h>
#endif
#include <stdlib.h>

/*****************************************************************
 ERROR HANDLING
*/
#define MAX_DRIZ_ERROR_LEN 512

struct driz_error_t {
  char last_message[MAX_DRIZ_ERROR_LEN];
};

void driz_error_init(struct driz_error_t* error);
int driz_error_check(struct driz_error_t* error, const char* message, int test);
void driz_error_set_message(struct driz_error_t* error, const char* message);
void driz_error_format_message(struct driz_error_t* error, const char* format, ...);
const char* driz_error_get_message(struct driz_error_t* error);
int driz_error_is_set(struct driz_error_t* error);
void driz_error_unset(struct driz_error_t* error);

/*****************************************************************
 CONVENIENCE MACROS
*/
#if !defined(MIN)
  #define MIN(a, b)  (((a) < (b)) ? (a) : (b))
#endif

#if !defined(MAX)
  #define MAX(a, b)  (((a) > (b)) ? (a) : (b))
#endif

#define CLAMP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
#define CLAMP_ABOVE(x, low)  (((x) < low) ? (low) : (x))
#define CLAMP_BELOW(x, high)  (((x) > high) ? (high) : (x))

#ifdef __GNUC__
#define UNUSED_PARAM __attribute__((unused))
#else
#define UNUSED_PARAM
#endif

#define MAX_DOUBLE 1.7976931348623158e+308
#define MIN_DOUBLE 2.2250738585072014e-308

#define MAX_COEFFS 128
#define COEFF_OFFSET 100

#undef TRUE
#define TRUE 1

#undef FALSE
#define FALSE 0

/*****************************************************************
 DATA TYPES
*/
typedef int integer_t;
#if __STDC_VERSION__ >= 199901L
typedef int_fast8_t bool_t;
#else
typedef unsigned char bool_t;
#endif

enum e_kernel_t {
  kernel_square,
  kernel_gaussian,
  kernel_point,
  kernel_tophat,
  kernel_turbo,
  kernel_lanczos2,
  kernel_lanczos3,
  kernel_LAST
};

enum e_unit_t {
  unit_counts,
  unit_cps
};

enum e_interp_t {
  interp_nearest,
  interp_bilinear,
  interp_poly3,
  interp_poly5,
  interp_spline3,
  interp_sinc,
  interp_lsinc,
  interp_lanczos3,
  interp_lanczos5,
  interp_LAST
};

/* Lanczos values */
struct lanczos_param_t {
  size_t nlut;
  float* lut;
  double sdp;
  integer_t nbox;
  float space;
  float misval;
};

struct driz_param_t {
  /* Options */
  enum e_kernel_t kernel; /* Kernel shape and size */
  double          pixel_fraction; /* was: PIXFRAC */
  float           exposure_time; /* Exposure time was: EXPIN */
  float           weight_scale; /* Weight scale was: WTSCL */
  float           fill_value; /* Filling was: FILVAL */
  bool_t          do_fill; /* was: FILL */
  enum e_unit_t   in_units; /* CPS / counts was: INCPS, either counts or CPS */
  enum e_unit_t   out_units; /* CPS / counts was: INCPS, either counts or CPS */
  integer_t       uuid; /* was: UNIQID */

  /* Scaling */
  double scale;

  /* Image subset */
  integer_t xmin;
  integer_t xmax;
  integer_t ymin;
  integer_t ymax;

  /* Blotting-specific parameters */
  enum e_interp_t interpolation; /* was INTERP */
  float ef; 
  float misval;
  float sinscl;
  float kscale;

  /* Input images */
  PyArrayObject *data; 
  PyArrayObject *weights;
  PyArrayObject *pixmap;

  /* Output images */
  PyArrayObject *output_data; 
  PyArrayObject *output_counts;  /* was: COU */
  PyArrayObject *output_context; /* was: CONTIM */

  /* Other output */
  integer_t nmiss;
  integer_t nskip;
  struct driz_error_t* error;

};

/**
Initialize all of the members of the drizzle_param_t to sane default
values, mostly zeroes.  Note, these are not *meaningful* values, just
ones that help with memory management etc.  It is up to users of the
struct, e.g. cdrizzle_, to fill the struct with valid parameters.
*/
void
driz_param_init(struct driz_param_t* p);

void
driz_param_dump(struct driz_param_t* p);


/****************************************************************************/
/* LOGGING */

#define LOGGING 0

#if LOGGING
extern FILE *driz_log_handle;

static inline_macro FILE*
driz_log_init(FILE *handle) {
    handle = fopen("/tmp/drizzle.log", "a");
    setbuf(handle, 0);
    return handle;
}

static inline_macro int
driz_log_close(FILE *handle) {
    return fclose(driz_log_handle);
}

static inline_macro int
driz_log_message(const char* message) {
    if (! driz_log_handle)
        driz_log_handle = driz_log_init(driz_log_handle);

    fputs(message, driz_log_handle);
    fputs("\n", driz_log_handle);
    return 0;
}

#else
static inline_macro void *
driz_log_idem(void *ptr) {
    return ptr;
}

#define driz_log_init(handle) driz_log_idem(handle)
#define driz_log_close(handle) driz_log_idem(handle)
#define driz_log_message(message) driz_log_idem(message)
#endif

/****************************************************************************/
/* ARRAY ACCESSORS */

/* New numpy based accessors */

static inline_macro void
get_dimensions(PyArrayObject *image, integer_t size[2]) {

  npy_intp *ndim = PyArray_DIMS(image);
  
  /* Put dimensions in xy order */  
  size[0] = ndim[1];
  size[1] = ndim[0];

  return;
}

static inline_macro double*
get_pixmap(PyArrayObject *pixmap, integer_t xpix, integer_t ypix) {
  return (double*) PyArray_GETPTR3(pixmap, ypix, xpix, 0);
}

#if LOGGING

static inline_macro int
oob_pixel(PyArrayObject *image, integer_t xpix, integer_t ypix) {
  char buffer[64];
  int flag = 0;

  npy_intp *ndim = PyArray_DIMS(image);
  if (xpix < 0 || xpix >= ndim[1]) flag = 1;
  if (ypix < 0 || ypix >= ndim[0]) flag = 1;

  if (flag) {
    sprintf(buffer, "Point [%d,%d] is outside of [%d, %d]",
            xpix, ypix, (int) ndim[1], (int) ndim[0]);
    driz_log_message(buffer);
  }
  
  return flag;
}

#else
#define oob_pixel(image, xpix, ypix)   0
#endif

static inline_macro float
get_pixel(PyArrayObject *image, integer_t xpix, integer_t ypix) {
  return *(float*) PyArray_GETPTR2(image, ypix, xpix);
}

static inline_macro float
get_pixel_at_pos(PyArrayObject *image, integer_t pos) {
  float *imptr;
  imptr = (float *) PyArray_DATA(image);
  return imptr[pos];
}

static inline_macro void
set_pixel(PyArrayObject *image, integer_t xpix, integer_t ypix, double value) {  
  *(float*) PyArray_GETPTR2(image, ypix, xpix) = value;
  return;
}

static inline_macro int
get_bit(PyArrayObject *image, integer_t xpix, integer_t ypix, integer_t bitval) {
  integer_t value;
  value = *(integer_t*) PyArray_GETPTR2(image, ypix, xpix) & bitval;
  return value? 1 : 0;
}

static inline_macro void
set_bit(PyArrayObject *image, integer_t xpix, integer_t ypix, integer_t bitval) {  
  *(integer_t*) PyArray_GETPTR2(image, ypix, xpix) |= bitval;
  return;
}

static inline_macro void
unset_bit(PyArrayObject *image, integer_t xpix, integer_t ypix) {
  *(integer_t*) PyArray_GETPTR2(image, ypix, xpix) = 0;
  return;
}

/*****************************************************************
 STRING TO ENUMERATION CONVERSIONS
*/
int
kernel_str2enum(const char* s, enum e_kernel_t* result, struct driz_error_t* error);

int
unit_str2enum(const char* s, enum e_unit_t* result, struct driz_error_t* error);

int
interp_str2enum(const char* s, enum e_interp_t* result, struct driz_error_t* error);

const char*
kernel_enum2str(enum e_kernel_t value);

const char*
unit_enum2str(enum e_unit_t value);

const char*
interp_enum2str(enum e_interp_t value);

const char*
bool2str(bool_t value);

/*****************************************************************
 NUMERICAL UTILITIES
*/
/**
Fill up a look-up-table of Lanczos interpolation kernel values for
rapid weighting determination for kernel == kernel_lanczos.

@param kernel_order the order of the kernel.
@param npix the size of the lookup table
@param del the spacings of the sampling of the function
@param lanczos_lut 1d array of lookup values.  This is a single-sided Lanczos
   function with lanczos_lut[0] being the central value.

Note that no checking is done to see whether the values are sensible.

was: FILALU
*/
void
create_lanczos_lut(const int kernel_order, const size_t npix,
                   const float del, float* lanczos_lut);

void
put_fill(struct driz_param_t* p, const float fill_value);

/**
 Calculate the refractive index of MgF2 for a given C wavelength (in
 nm) using the formula given by Trauger (1995)
*/
double
mgf2(double lambda);

/**
Weighted sum of 2 real vectors.

was: WSUMR
*/
static inline_macro void
weighted_sum_vectors(const integer_t npix,
                     const float* a /*[npix]*/, const float w1,
                     const float* b /*[npix]*/, const float w2,
                     /* Output arguments */
                     float* c /*[npix]*/) {
  float* c_end = c + npix;

  assert(a);
  assert(b);
  assert(c);

  while(c != c_end)
    *(c++) = *(a++) * w1 + *(b++) * w2;
}

/**
 Round to nearest integer in a way that mimics fortrans NINT
*/
static inline_macro integer_t
fortran_round(const double x) {
  return (x >= 0) ? (integer_t)floor(x + .5) : (integer_t)-floor(.5 - x);
}

static inline_macro double
min_doubles(const double* a, const integer_t size) {
  const double* end = a + size;
  double value = MAX_DOUBLE;
  for ( ; a != end; ++a)
    if (*a < value)
      value = *a;
  return value;
}

static inline_macro double
max_doubles(const double* a, const integer_t size) {
  const double* end = a + size;
  double value = MIN_DOUBLE;
  for ( ; a != end; ++a)
    if (*a > value)
      value = *a;
  return value;
}

/**
Evaluate a 3rd order radial geometric distortion in 2d
X version. Note that there is no zero order coefficient
as this is physically meaningless.

@param x The x coordinate

@param y The y coordinate

@param co An array of length 4 of coefficients

@param[out] xo The distorted x coordinate

@param[out] yo The distorted y coordinate
*/
static inline_macro void
rad3(const double x, const double y, const double* co,
     /* Output parameters */
     double* xo, double* yo) {
  double r, f;

  assert(co);
  assert(xo);
  assert(yo);

  r = sqrt(x*x + y*y);

  f = 1.0 + co[0] + co[1]*r + co[2]*r*r;
  *xo = f*x;
  *yo = f*y;
}

#endif /* CDRIZZLEUTIL_H */
