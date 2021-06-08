#define NO_IMPORT_ARRAY
#define NO_IMPORT_ASTROPY_WCS_API

#include "driz_portability.h"
#include "cdrizzlemap.h"
#include "cdrizzleblot.h"
#include "cdrizzleutil.h"

#include <assert.h>
#define _USE_MATH_DEFINES       /* needed for MS Windows to define M_PI */
#include <math.h>
#include <stdlib.h>
#include <numpy/npy_math.h>
#include <numpy/arrayobject.h>

/** --------------------------------------------------------------------------------------------------
 * Signature for functions that perform blotting interpolation.
 */

typedef int (interp_function)(const void*,
                              PyArrayObject*,
                              const float, const float,
                              /* Output parameters */
                              float*,
                              struct driz_error_t*);

/** --------------------------------------------------------------------------------------------------
 * A standard set of asserts for all of the interpolation functions
 */

#define INTERPOLATION_ASSERTS \
  assert(data); \
  assert(isize[0] > 0); \
  assert(isize[1] > 0); \
  assert(x >= 0.0f && x < (float)isize[0]);      \
  assert(y >= 0.0f && y < (float)isize[1]);      \
  assert(value); \
  assert(error); \

/** --------------------------------------------------------------------------------------------------
 * A structure to hold parameters for sinc interpolation.
 */

struct sinc_param_t {
  /*The scaling factor for sinc interpolation */
  float sinscl;
};

/** --------------------------------------------------------------------------------------------------
 * Procedure to evaluate the bicubic polynomial interpolant.  The array coeff contains the coefficients
 * of the 2D interpolant.  The procedure assumes that 1 <= x <= isize[0] and 1 <= y <= isize[1] and
 * that coeff[1+first_point] = datain[1,1]. The interpolant is evaluated using Everett's central
 * difference formula. (Was: IIBIP3)
 *
 * coeff:     Array of shape \a [len_coeff][len_coeff] contains the coefficients of the 2D interpolant.
 * len_coeff: The dimension (each side of the square) of the coefficient array.
 * firstt:    Offset of the first data point.  (In practice, this is always zero.)
 * npts:      The number of points to calculate.
 * x:         An array of length \a npts of x values.
 * y:         An array of length \a npts of y values.
 * zfit:      An array of length \a npts of interpolated values. (output)
 */

static inline_macro void
ii_bipoly3(const float* coeff /* [len_coeff][len_coeff] */,
           const integer_t len_coeff, const integer_t firstt,
           const integer_t npts,
           const float* x /* [npts] */, const float* y /* [npts] */,
           /* Output parameters */
           float* zfit /* [npts] */) {
  float sx, tx, sx2m1, tx2m1, sy, ty;
  float cd20[4], cd21[4], ztemp[4];
  float cd20y, cd21y;
  integer_t nxold, nyold;
  integer_t nx, ny;
  integer_t firstw, index;
  integer_t i, j;

  nxold = nyold = -1;
  for (i = 0; i < npts; ++i) {
    nx = (integer_t)x[i];
    assert(nx >= 0);

    sx = x[i] - (float)nx;
    tx = 1.0f - sx;
    sx2m1 = sx*sx - 1.0f;
    tx2m1 = tx*tx - 1.0f;

    ny = (integer_t)y[i];
    assert(ny >= 0);

    sy = y[i] - (float)ny;
    ty = 1.0f - sy;

    /* Calculate pointer to data[nx, ny-1] */
    firstw = firstt + (ny - 2) * len_coeff + nx - 1;

    /* loop over the 4 surrounding rows of data calculate the central
       differences at each value of y

       If new data point calculate the central differences in x for
       each y */
    if (nx != nxold || ny != nyold) {
      for (j = 0, index = firstw; j < 4; ++j, index += len_coeff) {
        assert(index > 0 && index < (len_coeff*len_coeff) - 2);

        cd20[j] = 1.0f/6.0f * (coeff[index+1] -
                               2.0f * coeff[index] +
                               coeff[index-1]);
        cd21[j] = 1.0f/6.0f * (coeff[index+2] -
                               2.0f * coeff[index+1] +
                               coeff[index]);
      }
    }

    /* Interpolate in x at each value of y */
    for (j = 0, index = firstw; j < 4; ++j, index += len_coeff) {
      assert(index >= 0 && index < (len_coeff*len_coeff) - 1);

      ztemp[j] = sx * (coeff[index+1] + sx2m1 * cd21[j]) +
                 tx * (coeff[index] + tx2m1 * cd20[j]);
    }

    /* Calculate y central differences */
    cd20y = 1.0f/6.0f * (ztemp[2] - 2.0f * ztemp[1] + ztemp[0]);
    cd21y = 1.0f/6.0f * (ztemp[3] - 2.0f * ztemp[2] + ztemp[1]);

    /* Interpolate in y */
    zfit[i] = sy * (ztemp[2] + (sy * sy - 1.0f) * cd21y) +
              ty * (ztemp[1] + (ty * ty - 1.0f) * cd20y);

    nxold = nx;
    nyold = ny;
  }
}

/** --------------------------------------------------------------------------------------------------
 * Procedure to evaluate a biquintic polynomial.  The array coeff contains the coefficents of the
 * 2D interpolant.  The routine assumes that 0 <= x < isize[0] and 0 <= y < isize[1]. The interpolant
 * is evaluated using Everett's central difference formula. (Was: IIBIP5)
 *
 * coeff:     Array of shape \a [len_coeff][len_coeff] contains the coefficients of the 2D interpolant.
 * len_coeff: The dimension (each side of the square) of the coefficient array.
 * firstt:    Offset of the first data point.  (In practice, this is always zero.)
 * npts:      The number of points to calculate.
 * x:         An array of length \a npts of x values.
 * y:         An array of length \a npts of y values.
 * zfit:      An array of length \a npts of interpolated values. (output)
 */

static inline_macro void
ii_bipoly5(const float* coeff /* [len_coeff][len_coeff] */,
           const integer_t len_coeff, const integer_t firstt,
           const integer_t npts,
           const float* x /* [npts] */, const float* y /* [npts] */,
           /* Output parameters */
           float* zfit /* [npts] */) {
  integer_t nxold, nyold;
  integer_t nx, ny;
  float sx, sx2, tx, tx2, sy, sy2, ty, ty2;
  float sx2m1, sx2m4, tx2m1, tx2m4;
  float cd20[6], cd21[6], cd40[6], cd41[6];
  float cd20y, cd21y, cd40y, cd41y;
  float ztemp[6];
  integer_t firstw, index;
  integer_t i, j;

  assert(coeff);
  assert(len_coeff > 0);
  assert(npts > 0);
  assert(x);
  assert(y);
  assert(zfit);

  nxold = nyold = -1;
  for (i = 0; i < npts; ++i) {
    nx = (integer_t)x[i];
    ny = (integer_t)y[i];
    assert(nx >= 0);
    assert(ny >= 0);

    sx = x[i] - (float)nx;
    sx2 = sx * sx;
    sx2m1 = sx2 - 1.0f;
    sx2m4 = sx2 - 4.0f;
    tx = 1.0f - sx;
    tx2 = tx * tx;
    tx2m1 = tx2 - 1.0f;
    tx2m4 = tx2 - 4.0f;

    sy = y[i] - (float)ny;
    sy2 = sy * sy;
    ty = 1.0f - sy;
    ty2 = ty * ty;

    /* Calculate value of pointer to data[nx,ny-2] */
    firstw = firstt + (ny - 3)*len_coeff + nx - 1;

    /* Calculate the central differences in x at each value of y */
    if (nx != nxold || ny != nyold) {
      for (j = 0, index = firstw; j < 6; ++j, index += len_coeff) {
        assert(index >= 1 && index < len_coeff*len_coeff - 3);

        cd20[j] = 1.0f/6.0f * (coeff[index+1] -
                               2.0f * coeff[index] +
                               coeff[index-1]);
        cd21[j] = 1.0f/6.0f * (coeff[index+2] -
                               2.0f * coeff[index+1] +
                               coeff[index]);
        cd40[j] = 1.0f/120.0f * (coeff[index-2] -
                                 4.0f * coeff[index-1] +
                                 6.0f * coeff[index] -
                                 4.0f * coeff[index+1] +
                                 coeff[index+2]);
        cd41[j] = 1.0f/120.0f * (coeff[index-1] -
                                 4.0f * coeff[index] +
                                 6.0f * coeff[index+1] -
                                 4.0f * coeff[index+2] +
                                 coeff[index+3]);
      }
    }

    /* Interpolate in x at each value of y */
    for (j = 0, index = firstw; j < 6; ++j, index += len_coeff) {
      assert(index >= 0 && index < len_coeff*len_coeff - 1);

      ztemp[j] = sx * (coeff[index+1] + sx2m1 * (cd21[j] + sx2m4 * cd41[j])) +
        tx * (coeff[index]   + tx2m1 * (cd20[j] + tx2m4 * cd40[j]));
    }

    /* Central differences in y */
    cd20y = 1.0f/6.0f * (ztemp[3] - 2.0f * ztemp[2] + ztemp[1]);
    cd21y = 1.0f/6.0f * (ztemp[4] - 2.0f * ztemp[3] + ztemp[2]);
    cd40y = 1.0f/120.0f * (ztemp[0] -
                           4.0f * ztemp[1] +
                           6.0f * ztemp[2] -
                           4.0f * ztemp[3] +
                           ztemp[4]);
    cd41y = 1.0f/120.0f * (ztemp[1] -
                           4.0f * ztemp[2] +
                           6.0f * ztemp[3] -
                           4.0f * ztemp[4] +
                           ztemp[5]);

    /* Interpolate in y */
    zfit[i] = sy * (ztemp[3] + (sy2 - 1.0f) * (cd21y + (sy2 - 4.0f) * cd41y)) +
      ty * (ztemp[2] + (ty2 - 1.0f) * (cd20y + (ty2 - 4.0f) * cd40y));

    nxold = nx;
    nyold = ny;
  }
}

/** --------------------------------------------------------------------------------------------------
 * Perform nearest neighbor interpolation.
 *
 * state: A pointer to any constant values specific to this interpolation type. (NULL).
 * data:  A 2D data array
 * x:     The fractional x coordinate
 * y:     The fractional y coordinate
 * value: The resulting value at x, y after interpolating the data (output)
 * error: The error structure (output)
 */

static int
interpolate_nearest_neighbor(const void* state UNUSED_PARAM,
                             PyArrayObject* data,
                             const float x, const float y,
                             /* Output parameters */
                             float* value,
                             struct driz_error_t* error UNUSED_PARAM) {

  integer_t   isize[2];
  get_dimensions(data, isize);

  assert(state == NULL);
  INTERPOLATION_ASSERTS;

  *value = get_pixel(data, (integer_t)(x + 0.5), (integer_t)(y + 0.5));
  return 0;
}

/** --------------------------------------------------------------------------------------------------
 * Perform basic bilinear interpolation.
 *
 * state: A pointer to any constant values specific to this interpolation type. (NULL).
 * data:  A 2D data array
 * x:     The fractional x coordinate
 * y:     The fractional y coordinate
 * value: The resulting value at x, y after interpolating the data (output)
 * error: The error structure (output)
 */

static int
interpolate_bilinear(const void* state UNUSED_PARAM,
                     PyArrayObject* data,
                     const float x, const float y,
                     /* Output parameters */
                     float* value,
                     struct driz_error_t* error UNUSED_PARAM) {
  integer_t nx, ny;
  float sx, tx, sy, ty, f00;
  integer_t isize[2];

  get_dimensions(data, isize);

  assert(state == NULL);
  INTERPOLATION_ASSERTS;

  nx = (integer_t) x;
  ny = (integer_t) y;

  if (nx < 0 || ny < 0 || nx >= isize[0] || ny >= isize[1]) {
      driz_error_set_message(error,
          "Bilinear interpolation: point outside of the image.");
      return 1;
  }

  f00 = get_pixel(data, nx, ny);

  if (nx == (isize[0] - 1)) {
    if (ny == (isize[1] - 1)) {
      /* This is the last pixel (in x and y). Assign constant value of this pixel. */
      *value = f00;
      return 0;
    }
    /* Interpolate along Y-direction only */
    sy = y - (float)ny;
    *value = (1.0f - sy) * f00 + sy * get_pixel(data, nx, ny + 1);
  } else if (ny == (isize[1] - 1)) {
    /* Interpolate along X-direction only */
    sx = x - (float)nx;
    *value = (1.0f - sx) * f00 + sx * get_pixel(data, nx + 1, ny);
  } else {
    /* Bilinear - interpolation */
    sx = x - (float)nx;
    tx = 1.0f - sx;
    sy = y - (float)ny;
    ty = 1.0f - sy;

    *value = tx * ty * f00 +
             sx * ty * get_pixel(data, nx + 1, ny) +
             sy * tx * get_pixel(data, nx, ny + 1) +
             sx * sy * get_pixel(data, nx + 1, ny + 1);
  }

  return 0;
}

/** --------------------------------------------------------------------------------------------------
 * Perform cubic polynomial interpolation.
 *
 * state: A pointer to any constant values specific to this interpolation type. (NULL).
 * data:  A 2D data array
 * x:     The fractional x coordinate
 * y:     The fractional y coordinate
 * value: The resulting value at x, y after interpolating the data (output)
 * error: The error structure (output)
 */

static int
interpolate_poly3(const void* state UNUSED_PARAM,
                  PyArrayObject* data,
                  const float x, const float y,
                  /* Output parameters */
                  float* value,
                  struct driz_error_t* error UNUSED_PARAM) {
  integer_t nx, ny;
  const integer_t rowleh = 4;
  const integer_t nterms = 4;
  float coeff[4][4];
  integer_t i, j;
  integer_t firstw, lastrw;
  float xval, yval;
  float* ci;
  integer_t   isize[2];
  get_dimensions(data, isize);

  assert(state == NULL);
  INTERPOLATION_ASSERTS;;

  nx = (integer_t)x;
  ny = (integer_t)y;

  ci = &coeff[0][0];
  for (j = ny - 1; j <= ny + 2; ++j) {
    if (j >= 0 && j < isize[1]) {
      for (i = nx - 1; i <= nx + 2; ++i, ++ci) {
        if (i < 0) {
          *ci = 2.0f * get_pixel(data, 0, j) - get_pixel(data, -i, j);
        } else if (i >= isize[0]) {
          *ci = 2.0f * get_pixel(data, isize[0]-1, j) - get_pixel(data, 2*isize[0]-2-i, j);
        } else {
          *ci = get_pixel(data, i, j);
        }
      }
    } else if (j == ny + 2) {
      for (i = nx - 1; i <= nx + 2; ++i, ++ci) {
        if (i < 0) {
          *ci = 2.0f * get_pixel(data, 0, isize[1]-3) - get_pixel(data, -i, isize[1]-3);
        } else if (i >= isize[0]) {
          *ci = 2.0f * get_pixel(data, isize[0]-1, isize[1]-3) - get_pixel(data, 2*isize[0]-2-i, isize[1]-3);
        } else {
          *ci = get_pixel(data, i, isize[1]-3);
        }
      }
    } else {
      ci += 4;
    }
  }

  firstw = MAX(0, 1 - ny);
  if (firstw > 0) {
    assert(firstw < nterms);

    for (j = 0; j < firstw; ++j) {
      assert(2*firstw-j >= 0 && 2*firstw-j < nterms);

      weighted_sum_vectors(nterms,
                           &coeff[firstw][0], 2.0,
                           &coeff[2*firstw-j][0], -1.0,
                           &coeff[j][0]);
    }
  }

  lastrw = MIN(nterms - 1, isize[1] - ny);
  if (lastrw < nterms - 1) {
    assert(lastrw >= 0 && lastrw < nterms);

    for (j = lastrw + 1; j <= nterms - 1; ++j) {
      assert(2*lastrw-j >= 0 && 2*lastrw-j < nterms);
      assert(j >= 0 && j < 4);

      weighted_sum_vectors(nterms,
                           &coeff[lastrw][0], 2.0,
                           &coeff[2*lastrw-j][0], -1.0,
                           &coeff[j][0]);
    }
  } else if (lastrw == 1) {
    assert(lastrw >= 0 && lastrw < nterms);

    weighted_sum_vectors(nterms,
                         &coeff[lastrw][0], 2.0,
                         &coeff[3][0], -1.0,
                         &coeff[3][0]);
  } else {
    assert(lastrw >= 0 && lastrw < nterms);
    assert(2*lastrw-3 >= 0 && 2*lastrw-3 < nterms);

    weighted_sum_vectors(nterms,
                         &coeff[lastrw][0], 2.0,
                         &coeff[2*lastrw-3][0], -1.0,
                         &coeff[3][0]);
  }

  xval = 2.0f + (x - (float)nx);
  yval = 2.0f + (y - (float)ny);

  ii_bipoly3(&coeff[0][0], rowleh, 0, 1, &xval, &yval, value);

  return 0;
}

/** --------------------------------------------------------------------------------------------------
 * Perform quintic polynomial interpolation.
 *
 * state: A pointer to any constant values specific to this interpolation type. (NULL).
 * data:  A 2D data array
 * x:     The fractional x coordinate
 * y:     The fractional y coordinate
 * value: The resulting value at x, y after interpolating the data (output)
 * error: The error structure (output)
 */

static int
interpolate_poly5(const void* state UNUSED_PARAM,
                  PyArrayObject* data,
                  const float x, const float y,
                  /* Output parameters */
                  float* value,
                  struct driz_error_t* error UNUSED_PARAM) {
  integer_t nx, ny;
  const integer_t rowleh = 6;
  const integer_t nterms = 6;
  float coeff[6][6];
  integer_t i, j;
  integer_t firstw, lastrw;
  float xval, yval;
  float* ci;
  integer_t   isize[2];
  get_dimensions(data, isize);

  assert(state == NULL);
  INTERPOLATION_ASSERTS;

  nx = (integer_t)x;
  ny = (integer_t)y;

  ci = &coeff[0][0];
  for (j = ny - 2; j <= ny + 3; ++j) {
    if (j >= 0 && j < isize[1]) {
      for (i = nx - 2; i <= nx + 3; ++i, ++ci) {
        if (i < 0) {
          *ci = 2.0f * get_pixel(data, 0, j) - get_pixel(data, -i, j);
        } else if (i >= isize[0]) {
          *ci = 2.0f * get_pixel(data, isize[0]-1, j) - get_pixel(data, 2*isize[0]-2-i, j);
        } else {
          *ci = get_pixel(data, i, j);
        }
      }
    } else if (j == (ny + 3)) {
      for (i = nx - 2; i <= nx + 3; ++i, ++ci) {
        if (i < 0) {
          *ci = 2.0f * get_pixel(data, 0, isize[1]-4) - get_pixel(data, -i, isize[1]-4);
        } else if (i >= isize[0]) {
          *ci = 2.0f * get_pixel(data, isize[0]-1, isize[1]-4) - get_pixel(data, 2 * isize[0]-2-i, isize[1]-4);
        } else {
          *ci = get_pixel(data, i, isize[1]-4);
        }
      }
    } else {
      ci += 6;
    }
  }

  firstw = MAX(0, 2 - ny);
  assert(firstw >= 0 && firstw < nterms);

  if (firstw > 0) {
    for (j = 0; j <= firstw; ++j) {
      assert(2*firstw-j >= 0 && 2*firstw-j < nterms);

      weighted_sum_vectors(nterms,
                           &coeff[firstw][0], 2.0,
                           &coeff[2*firstw-j][0], -1.0,
                           &coeff[j][0]);
    }
  }

  lastrw = MIN(nterms - 1, isize[1] - ny + 1);
  assert(lastrw < nterms);

  if (lastrw < nterms - 1) {
    for (j = lastrw + 1; j <= nterms - 2; ++j) {
      assert(2*lastrw-j >= 0 && 2*lastrw-j < nterms);

      weighted_sum_vectors(nterms,
                           &coeff[lastrw][0], 2.0,
                           &coeff[2*lastrw-j][0], -1.0,
                           &coeff[j][0]);
    }
  } else if (lastrw == 2) {
    weighted_sum_vectors(nterms,
                         &coeff[2][0], 2.0,
                         &coeff[5][0], -1.0,
                         &coeff[5][0]);
  } else {
    assert(2*lastrw - 5 >= 0 && 2*lastrw-5 < nterms);

    weighted_sum_vectors(nterms,
                         &coeff[lastrw][0], 2.0,
                         &coeff[2*lastrw-5][0], -1.0,
                         &coeff[5][0]);
  }

  xval = 3.0f + (x - (float)nx);
  yval = 3.0f + (y - (float)ny);

  ii_bipoly5(&coeff[0][0], rowleh, 0, 1, &xval, &yval, value);

  return 0;
}

/** --------------------------------------------------------------------------------------------------
 * (Was: iinisc)
 */

#define INTERPOLATE_SINC_NCONV 15

static inline_macro int
interpolate_sinc_(PyArrayObject* data,
                  const integer_t firstt, const integer_t npts,
                  const float* x /*[npts]*/, const float* y /*[npts]*/,
                  const float mindx, const float mindy,
                  const float sinscl,
                  /* Output parameters */
                  float* value,
                  struct driz_error_t* error UNUSED_PARAM) {
  const integer_t nconv = INTERPOLATE_SINC_NCONV;
  const integer_t nsinc = (nconv - 1) / 2;
  /* TODO: This is to match Fortan, but is probably technically less precise */
  const float halfpi = 1.5707963267948966192f; /* M_PI / 2.0; */
  const float sconst = powf((halfpi / (float)nsinc), 2.0f);
  const float a2 = -0.49670f;
  const float a4 = 0.03705f;
  float taper[INTERPOLATE_SINC_NCONV];
  float ac[INTERPOLATE_SINC_NCONV], ar[INTERPOLATE_SINC_NCONV];
  float sdx, dx, dy, dxn, dyn, dx2;
  float ax, ay, px, py;
  float sum, sumx, sumy;
  float tmp;
  integer_t minj, maxj, offj;
  integer_t mink, maxk, offk;
  integer_t nx, ny;
  integer_t i, j, k, m, index;
  integer_t indices[3][4];
  integer_t   isize[2];
  get_dimensions(data, isize);

  assert(x);
  assert(y);
  assert(value);
  assert(error);

  if ((nsinc % 2) == 0) {
    sdx = 1.0;
    for (j = -nsinc; j <= nsinc; ++j) {
      assert(j + nsinc >= 0 && j + nsinc < INTERPOLATE_SINC_NCONV);

      taper[j + nsinc] = 1.0;
    }
  } else {
    sdx = -1.0;
    errno = 0;
    for (j = -nsinc; j <= nsinc; ++j) {
      assert(j + nsinc >= 0 && j + nsinc < INTERPOLATE_SINC_NCONV);

      dx2 = sconst * (float)j * (float)j;
      tmp = powf(1.0f + a2*dx2 + a4*dx2*dx2, 2.0);
      if (errno != 0) {
        driz_error_set_message(error, "pow failed");
        return 1;
      }
      taper[j + nsinc] = sdx * tmp;

      sdx = -sdx;
    }
  }

  for (i = 0; i < npts; ++i) {
    nx = fortran_round(x[i]);
    ny = fortran_round(y[i]);
    if (nx < 0 || nx >= isize[0] || ny < 0 || ny >= isize[1]) {
      value[i] = 0.0;
      continue;
    }

    dx = (x[i] - (float)nx) * sinscl;
    dy = (y[i] - (float)ny) * sinscl;

    if (fabsf(dx) < mindx && fabsf(dy) < mindy) {
      index = firstt + (ny - 1) * isize[0] + nx - 1; /* TODO: Base check */
      value[i] = get_pixel_at_pos(data, index);
      continue;
    }

    dxn = 1.0f + (float)nsinc + dx;
    dyn = 1.0f + (float)nsinc + dy;
    sumx = 0.0f;
    sumy = 0.0f;
    for (j = 0; j < nconv; ++j) {
      /* TODO: These out of range indices also seem to be in Fortran... */
      ax = dxn - (float)j - 1; /* TODO: Base check */
      ay = dyn - (float)j - 1; /* TODO: Base check */
      assert(ax != 0.0);
      assert(ay != 0.0);

      if (ax == 0.0) {
        px = 1.0;
      } else if (dx == 0.0) {
        px = 0.0;
      } else {
        px = taper[j - 1] / ax;
      }

      if (ay == 0.0) {
        py = 1.0;
      } else if (dy == 0.0) {
        py = 0.0;
      } else {
        py = taper[j - 1] / ay;
      }

      /* TODO: These out of range indices also seem to be in Fortran... */
      ac[j - 1] = px;
      ar[j - 1] = py;
      sumx += px;
      sumy += py;
    }

    /* Compute the limits of the convolution */
    minj = MAX(0, ny - nsinc - 1); /* TODO: Bases check */
    maxj = MIN(isize[1], ny + nsinc); /* TODO: Bases check */
    offj = nsinc - ny; /* TODO: Bases check */

    mink = MAX(0, nx - nsinc - 1); /* TODO: Bases check */
    maxk = MIN(isize[0], nx + nsinc); /* TODO: Bases check */
    offk = nsinc - nx; /* TODO: Bases check */

    value[i] = 0.0;

    indices[0][0] = ny - nsinc;
    indices[0][1] = minj - 1;
    indices[0][2] = firstt;
    indices[0][3] = 0;

    indices[1][0] = minj;
    indices[1][1] = maxj;
    indices[1][2] = firstt;
    indices[1][3] = isize[0];

    indices[2][0] = maxj + 1;
    indices[2][1] = ny + nsinc;
    indices[2][2] = firstt + (isize[1] - 1) * isize[0];
    indices[2][3] = 0;

    for (m = 0; m < 3; ++m) {
      for (j = indices[m][0]; j <= indices[m][1]; ++j) {
        sum = 0.0;
        index = indices[m][2] + j * indices[m][3];
        assert(index >= 0 && index < isize[0]*isize[1] - 1);
        assert(index+isize[0] >= 0 && index+isize[0] < isize[0]*isize[1]);

        for (k = nx - nsinc; k < mink - 1; ++k) { /* TODO: Bases check */
          assert(k+offk >= 0 && k+offk < INTERPOLATE_SINC_NCONV);

          sum += ac[k+offk] * get_pixel_at_pos(data, index+1);
        }

        for (k = mink; k <= maxk; ++k) { /* TODO: Bases check */
          assert(k+offk >= 0 && k+offk < INTERPOLATE_SINC_NCONV);
          assert(index+k >= 0 && index+k < isize[0]*isize[1]);

          sum += ac[k+offk] * get_pixel_at_pos(data, index+k);
        }

        for (k = maxk + 1; k <= nx + nsinc; ++k) {
          assert(k+offk >= 0 && k+offk < INTERPOLATE_SINC_NCONV);

          sum += ac[k+offk] * get_pixel_at_pos(data, index+isize[0]);
        }

        assert(j + offj >= 0 && j + offj < INTERPOLATE_SINC_NCONV);

        value[i] += ar[j + offj] * sum;
      }
    }

    assert(sumx != 0.0);
    assert(sumy != 0.0);

    value[i] = value[i] / sumx / sumy;
  }

  return 0;
}

/** --------------------------------------------------------------------------------------------------
 * Perform sinc interpolation.
 *
 * state: A pointer to a \a sinc_param_t object
 * data:  A 2D data array
 * x:     The fractional x coordinate
 * y:     The fractional y coordinate
 * value: The resulting value at x, y after interpolating the data (output)
 * error: The error structure (output)
 */

static int
interpolate_sinc(const void* state,
                 PyArrayObject* data,
                 const float x, const float y,
                 /* Output parameters */
                 float* value,
                 struct driz_error_t* error) {
  const struct sinc_param_t* param = (const struct sinc_param_t*)state;
  integer_t   isize[2];
  get_dimensions(data, isize);

  assert(state);
  INTERPOLATION_ASSERTS;

  return interpolate_sinc_(data, 0, 1, &x, &y, 0.001f, 0.001f,
                           param->sinscl, value, error);
}

/** --------------------------------------------------------------------------------------------------
 * Perform Lanczos interpolation.
 *
 * state: A pointer to a \a lanczos_param_t object
 * data:  A 2D data array
 * x:     The fractional x coordinate
 * y:     The fractional y coordinate
 * value: The resulting value at x, y after interpolating the data (output)
 * error: The error structure (output)
 */

static int
interpolate_lanczos(const void* state,
                    PyArrayObject* data,
                    const float x, const float y,
                    /* Output parameters */
                    float* value,
                    struct driz_error_t* error UNUSED_PARAM) {
  integer_t ixs, iys, ixe, iye;
  integer_t xoff, yoff;
  float luty, sum;
  integer_t nbox;
  integer_t i, j;
  const struct lanczos_param_t* lanczos = (const struct lanczos_param_t*)state;
  integer_t   isize[2];
  get_dimensions(data, isize);

  assert(state);
  INTERPOLATION_ASSERTS;

  nbox = lanczos->nbox;

  /* First check for being close to the edge and, if so, return the
     missing value */
  ixs = (integer_t)(x) - nbox;
  ixe = (integer_t)(x) + nbox;
  iys = (integer_t)(y) - nbox;
  iye = (integer_t)(y) + nbox;
  if (ixs < 0 || ixe >= isize[0] ||
      iys < 0 || iye >= isize[1]) {
    *value = lanczos->misval;
    return 0;
  }

  /* Don't divide-by-zero errors */
  assert(lanczos->space != 0.0);

  /* Loop over the box, which is assumed to be scaled appropriately */
  sum = 0.0;
  for (j = iys; j <= iye; ++j) {
    yoff = (integer_t)(fabs((y - (float)j) / lanczos->space));
    assert(yoff >= 0 && yoff < lanczos->nlut);

    luty = lanczos->lut[yoff];
    for (i = ixs; i <= ixe; ++i) {
      xoff = (integer_t)(fabs((x - (float)i) / lanczos->space));
      assert(xoff >= 0 && xoff < lanczos->nlut);

      sum += get_pixel(data, i, j) * lanczos->lut[xoff] * luty;
    }
  }

  *value = sum;
  return 0;
}

/** --------------------------------------------------------------------------------------------------
 * A mapping from e_interp_t enumeration values to function pointers that actually
 * perform the interpolation.  NULL elements will raise an "unimplemented" error.
 */

interp_function* interp_function_map[interp_LAST] = {
  &interpolate_nearest_neighbor,
  &interpolate_bilinear,
  &interpolate_poly3,
  &interpolate_poly5,
  NULL,
  &interpolate_sinc,
  &interpolate_sinc,
  &interpolate_lanczos,
  &interpolate_lanczos
};

/** --------------------------------------------------------------------------------------------------
 * Interpolate grid of pixels onto new grid of different size.
 *
 * p:   structure containing options, input, and output
 */

int
doblot(struct driz_param_t* p) {

  const size_t nlut = 2048;
  const float space = 0.01;
  integer_t isize[2], osize[2];
  float scale2, xo, yo, v;
  integer_t i, j;
  interp_function* interpolate;
  struct sinc_param_t sinc;
  struct lanczos_param_t lanczos;
  void* state = NULL;

  driz_log_message("starting doblot");
  get_dimensions(p->data, isize);
  get_dimensions(p->output_data, osize);

  /* Select interpolation function */
  assert(p->interpolation >= 0 && p->interpolation < interp_LAST);
  interpolate = interp_function_map[p->interpolation];
  if (interpolate == NULL) {
    driz_error_set_message(p->error, "Requested interpolation type not implemented.");
    goto doblot_exit_;
  }

  lanczos.lut = NULL;

  /* Some interpolation functions need some pre-calculated state */
  if (p->interpolation == interp_lanczos3 || p->interpolation == interp_lanczos5) {

    if ((lanczos.lut = (float*)malloc(nlut * sizeof(float))) == NULL) {
      driz_error_set_message(p->error, "Out of memory");
      goto doblot_exit_;
    }

    create_lanczos_lut(p->interpolation == interp_lanczos3 ? 3 : 5,
                       nlut, space, lanczos.lut);

    lanczos.nbox = (integer_t)(3.0 / p->kscale);
    lanczos.nlut = nlut;
    lanczos.space = space;
    lanczos.misval = p->misval;

    state = &lanczos;

  } else if (p->interpolation == interp_sinc || p->interpolation == interp_lsinc) {
    sinc.sinscl = p->sinscl;
    state = &sinc;

  } /* Otherwise state is NULL */

  /* In the WCS case, we can't use the scale to calculate the Jacobian,
     so we need to do it.

     Note that we use the center of the image, rather than the reference pixel
     as the reference here.

     This is taken from dobox, except for the inversion of the image order.

     This section applies in WBLOT mode and now contains the addition
     correction to separate the distortion-induced scale change.
  */

  /* Recalculate the area scaling factor */
  scale2 = p->scale * p->scale;
  v = 1.0;

  for (j = 0; j < osize[1]; ++j) {

    /* Loop through the output positions and do the interpolation */
    for (i = 0; i < osize[0]; ++i) {
      if (oob_pixel(p->pixmap, i, j)) {
          driz_error_format_message(p->error, "OOB in pixmap[%d,%d]", i, j);
          return 1;
      } else {
        xo = get_pixmap(p->pixmap, i, j)[0];
        yo = get_pixmap(p->pixmap, i, j)[1];
      }

      if (npy_isnan(xo) || npy_isnan(yo)) {
          driz_error_format_message(p->error, "NaN in pixmap[%d,%d]", i, j);
          return 1;
      }

      /* Check it is on the input image */
      if (xo >= 0.0 && xo < (float)isize[0] &&
          yo >= 0.0 && yo < (float)isize[1]) {

        double value;

        /* Check for look-up-table interpolation */
        if (interpolate(state, p->data, xo, yo, &v, p->error)) {
          goto doblot_exit_;
        }

        value = v * p->ef / scale2;
        if (oob_pixel(p->output_data, i, j)) {
          driz_error_format_message(p->error, "OOB in output_data[%d,%d]", i, j);
          return 1;
        } else {
          set_pixel(p->output_data, i, j, value);
        }

      } else {
        /* If there is nothing for us then set the output to missing C
           value flag */
        if (oob_pixel(p->output_data, i, j)) {
          driz_error_format_message(p->error, "OOB in output_data[%d,%d]", i, j);
          return 1;
        } else {
          set_pixel(p->output_data, i, j, p->misval);
          p->nmiss++;
        }
      }
    }
  }

 doblot_exit_:
  driz_log_message("ending doblot");
  if (lanczos.lut) free(lanczos.lut);

  return driz_error_is_set(p->error);
}
