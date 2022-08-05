#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <Python.h>
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_10_API_VERSION
#endif
#include <numpy/npy_math.h>
#include <numpy/arrayobject.h>

#include "driz_portability.h"
#include "cdrizzlemap.h"
#include "cdrizzleutil.h"

/** --------------------------------------------------------------------------------------------------
 * Initialize a segment structure to contain the two points (x1, y1) and (x2, y2)
 * the invalid flag is initially set to 0 (valid)
 */

void
initialize_segment(struct segment *self, integer_t x1, integer_t y1, integer_t x2, integer_t y2) {
  self->point[0][0] = x1;
  self->point[0][1] = y1;
  self->point[1][0] = x2;
  self->point[1][1] = y2;
  self->invalid = 0;

  return;
}

/** --------------------------------------------------------------------------------------------------
 * Generate a string representation of a segment for debugging
 *
 * self: the segment
 * str:  the string representation, at least 64 chars (output)
 */

void
show_segment(struct segment *self, char *str) {
  sprintf(str, "(%10f,%10f) - (%10f,%10f) [%2d]",
          self->point[0][0], self->point[0][1],
          self->point[1][0], self->point[1][1],
          self->invalid);

  return;
}

/** --------------------------------------------------------------------------------------------------
 * Test if a pixmap vaue is bad (NaN)
 *
 * pixmap: the pixel mapping between input and output images
 * i:      the index of a pixel within a line
 * j:      the index of a line within an image
 */

int
bad_pixel(PyArrayObject *pixmap, int i, int j) {
  int k;
  for (k = 0; k < 2; ++k) {
    oob_pixel(pixmap, i, j);
    if (npy_isnan(get_pixmap(pixmap, i, j)[k])) {
      return 1;
    }
  }

  return 0;
}

/** --------------------------------------------------------------------------------------------------
 * Test if weight value is bad (zero)
 *
 * weights the weight to be given each pixel in an image
 * i:      the index of a pixel within a line
 * j:      the index of a line within an image
 */

int
bad_weight(PyArrayObject *weights, int i, int j) {

  if (weights) {
    oob_pixel(weights, i, j);
    if (get_pixel(weights, i, j) == 0.0) {
      return 1;
    } else {
      return 0;
    }
  }

  return 0;
}

/** --------------------------------------------------------------------------------------------------
 * Set the bounds of a segment to the range containing valid data
 *
 * self:    the segment
 * p:       the stucture containing the image pointers
 * pixmap:  array mapping input pixel coordinates to output pixel coordinates
 * weights: array of weights to apply when summing pixels
 */

void
shrink_segment(struct segment *self,
               PyArrayObject *array,
               int (*is_bad_value)(PyArrayObject *, int, int)) {

  int i, j, imin, imax, jmin, jmax;

  imin = self->point[1][0];
  jmin = self->point[1][1];

  for (j = self->point[0][1]; j < self->point[1][1]; ++j) {
    for (i = self->point[0][0]; i < self->point[1][0]; ++ i) {
      if (! is_bad_value(array, i, j)) {
        if (i < imin) {
          imin = i;
        }
        if (j < jmin) {
          jmin = j;
        }
        break;
      }
    }
  }

  imax = self->point[0][0];
  jmax = self->point[0][1];

  for (j = self->point[1][1]; j > self->point[0][1]; --j) {
    for (i = self->point[1][0]; i > self->point[0][0]; -- i) {
      if (! is_bad_value(array, i-1, j-1)) {
        if (i > imax) {
          imax = i;
        }
        if (j > jmax) {
          jmax = j;
        }
        break;
      }
    }
  }

  initialize_segment(self, imin, jmin, imax, jmax);
  self->invalid = imin >= imax || jmin >= jmax;
  return;
}

/** --------------------------------------------------------------------------------------------------
 * Sort points in increasing order on jdim coordinate
 *
 * self: the segment
 * jdim: the dimension to sort on, x (0) or y (1)
 */

void
sort_segment(struct segment *self, int jdim) {
  int idim;

  if (self->invalid == 0) {
    if (self->point[0][jdim] > self->point[1][jdim]) {
      double t;

      for (idim = 0; idim < 2; ++idim) {
        t = self->point[0][idim];
        self->point[0][idim] = self->point[1][idim];
        self->point[1][idim] = t;
      }
    }
  }

  return;
}

/** --------------------------------------------------------------------------------------------------
 * Take the the union of several line segments along a dimension.
 * That is, the result is the combined range of all the segments along a dimension
 *
 * npoint:   number of line segments to combine
 * jdim:     dimension to take union along, x (0) or y (1)
 * sybounds: the array of line segments
 * bounds:   union of the segment range (output)
 */

void
union_of_segments(int npoint, int jdim, struct segment xybounds[], integer_t bounds[2]) {
  int ipoint;
  int none = 1;

  for (ipoint = 0; ipoint < npoint; ++ipoint) {
    sort_segment(&xybounds[ipoint], jdim);

    if (xybounds[ipoint].invalid == 0) {
      integer_t lo = floor(xybounds[ipoint].point[0][jdim]);
      integer_t hi = ceil(xybounds[ipoint].point[1][jdim]);

      if (none == 0) {
        if (lo < bounds[0]) bounds[0] = lo;
        if (hi > bounds[1]) bounds[1] = hi;

      } else {
        none = 0;
        bounds[0] = lo;
        bounds[1] = hi;
      }
    }
  }

  if (none) {
    bounds[0] = 0;
    bounds[1] = 0;
  }

  return;
}

/** --------------------------------------------------------------------------------------------------
 * Find the points that bound the linear interpolation
 *
 * pixmap:   The mapping of the pixel centers from input to output image
 * xyin:     An (x,y) point on the input image
 * idim:     The dimension to search across
 * xybounds: The bounds for the linear interpolation (output)
 */

int interpolation_starting_point(
  PyArrayObject *pixmap,
  const double  xyin[2],
  int           *xypix
  ) {

  int kdim;
  int xydim[2];

  get_dimensions(pixmap, xydim);

  /* Make sure starting point is inside image */
  for (kdim = 0; kdim < 2; ++kdim) {
    /* Starting point rounds down input pixel position
       to integer value
    */
    xypix[kdim] = (int)xyin[kdim];

    if (xypix[kdim] < 0) {
      xypix[kdim] = 0;
    } else if (xypix[kdim] > xydim[kdim] - 2) {
      xypix[kdim] = xydim[kdim] - 2;
    }
  }
  return 0;
}

/** --------------------------------------------------------------------------------------------------
 * Map a point on the input image to the output image using
 * a mapping of the pixel centers between the two by interpolating
 * between the centers in the mapping
 *
 * pixmap: The mapping of the pixel centers from input to output image
 * xyin:   An (x,y) point on the input image
 * xyout:  The same (x, y) point on the output image (output)
 */

int interpolate_point(PyArrayObject *pixmap, const double xyin[2], double xyout[2]) {
  int xypix[2];
  int ipix, jpix, npix, idim;
  int i0, j0;
  double x, y, x1, y1, f00, f01, f10, f11, g00, g01, g10, g11;
  double *p;

  interpolation_starting_point(pixmap, xyin, (int *)xypix);

  /* Bilinear interpolation from
     https://en.wikipedia.org/wiki/Bilinear_interpolation#On_the_unit_square
  */
  i0 = xypix[0];
  j0 = xypix[1];
  x = xyin[0] - i0;
  y = xyin[1] - j0;
  x1 = 1.0 - x;
  y1 = 1.0 - y;

  p = get_pixmap(pixmap, i0, j0);
  f00 = p[0];
  g00 = p[1];

  p = get_pixmap(pixmap, i0 + 1, j0);
  f10 = p[0];
  g10 = p[1];

  p = get_pixmap(pixmap, i0, j0 + 1);
  f01 = p[0];
  g01 = p[1];

  p = get_pixmap(pixmap, i0 + 1, j0 + 1);
  f11 = p[0];
  g11 = p[1];

  xyout[0] = f00 * x1 * y1 + f10 * x * y1 + f01 * x1 * y + f11 * x * y;
  xyout[1] = g00 * x1 * y1 + g10 * x * y1 + g01 * x1 * y + g11 * x * y;

  if (npy_isnan(xyout[0]) || npy_isnan(xyout[1])) return 1;

  return 0;
}

/** --------------------------------------------------------------------------------------------------
 * Map an integer pixel position from the input to the output image.
 * Fall back on interpolation if the value at the point is undefined
 *
 * pixmap: The mapping of the pixel centers from input to output image
 * i        The index of the x coordinate
 * j        The index of the y coordinate
 * xyout:  The (x, y) point on the output image (output)
 */

int
map_pixel(
  PyArrayObject *pixmap,
  int           i,
  int           j,
  double        xyout[2]
  ) {

  int k;

  oob_pixel(pixmap, i, j);
  for (k = 0; k < 2; ++k) {
    xyout[k] = get_pixmap(pixmap, i, j)[k];

    if (npy_isnan(xyout[k])) return 1;
  }

  return 0;
}

/** --------------------------------------------------------------------------------------------------
 * Map a point on the input image to the output image either by interpolation
 * or direct array acces if the input position is integral.
 *
 * pixmap: The mapping of the pixel centers from input to output image
 * xyin:   An (x,y) point on the input image
 * xyout:  The same (x, y) point on the output image (output)
 */

int
map_point(
  PyArrayObject *pixmap,
  const double  xyin[2],
  double        xyout[2]
  ) {

  int i, j, status, mapsize[2];

  i = xyin[0];
  j = xyin[1];
  get_dimensions(pixmap, mapsize);

  if ((double) i == xyin[0] && i >= 0 && i < mapsize[0] &&
      (double) j == xyin[1] && j >= 0 && j < mapsize[1]) {
    status = map_pixel(pixmap, i, j, xyout);

  } else {
    status = interpolate_point(pixmap, xyin, xyout);
  }

  return status;
}

/** --------------------------------------------------------------------------------------------------
 * Clip a line segment from an input image to the limits of an output image along one dimension
 *
 * pixmap:   the mapping between input and output images
 * outlimit:  the limits  of the output image
 * xybounds: the clipped line segment (output)
 *
 */

int
clip_bounds(PyArrayObject *pixmap, struct segment *outlimit,
            struct segment *xybounds) {
  int ipoint, idim, jdim;

  if (xybounds->invalid) {
    return 0;
  }

  for (idim = 0; idim < 2; ++idim) {
    for (ipoint = 0; ipoint < 2; ++ipoint) {
      int m = 21;         /* maximum iterations */
      int side = 0;       /* flag indicating which side moved last */

      double xyin[2], xyout[2];
      double a, b, c, fa, fb, fc;

      /* starting values at endpoints of interval */

      for (jdim = 0; jdim < 2; ++jdim) {
        xyin[jdim] = xybounds->point[0][jdim];
      }

      if (map_point(pixmap, xyin, xyout)) {
        /* Cannot find bound */
        return 0;
      }

      a = xybounds->point[0][idim];
      fa = xyout[idim] - outlimit->point[ipoint][idim];

      for (jdim = 0; jdim < 2; ++jdim) {
        xyin[jdim] = xybounds->point[1][jdim];
      }

      if (map_point(pixmap, xyin, xyout)) {
        /* Cannot find bound */
        return 0;
      }

      c = xybounds->point[1][idim];
      fc = xyout[idim] - outlimit->point[ipoint][idim];

      /* Solution via the method of false position (regula falsi) */

      if (fa * fc < 0.0) {
        int n; /* for loop limit is just for safety's sake */

        xybounds->invalid = 0;
        for (n = 0; n < m; n++) {
          b = (fa * c - fc * a) / (fa - fc);

          /* Solution is exact if within a pixel because linear interpolation */
          if (floor(a) == floor(c)) break;

          xyin[idim] = b;
          if (map_point(pixmap, xyin, xyout)) {
            /*  cannot map point, so end interpolation */
            break;
          }
          fb = xyout[idim] - outlimit->point[ipoint][idim];

          /* Maintain the bound by copying b to the variable
           * with the same sign as b
           */

          if (fb * fc > 0.0) {
            c = b;
            fc = fb;
            if (side == -1) {
                fa *= 0.5;
            }
            side = -1;

          } else if (fa * fb > 0.0) {
            a = b;
            fa = fb;
            if (side == +1) {
                fc *= 0.5;
            }
            side = +1;

          } else {
            /* if the product is zero, we have converged */
            break;
          }
        }

        if (n > m) {
          return 1;
        }

        xybounds->point[ipoint][idim] = b;

      } else {
        /* No bracket, so track which side the bound lies on */
        if (xybounds->invalid == 0) {
            xybounds->invalid = 1;
        }
        xybounds->invalid *= fa > 0.0 ? +1 : -1;
      }
    }

    if (xybounds->invalid > 0) {
      /* Positive means both bounds are outside the image */
      xybounds->point[1][idim] = xybounds->point[0][idim];
      break;

    } else {
      /* Negative means both bounds are inside, which is not a problem */
      xybounds->invalid = 0;
    }
  }

  return 0;
}

/** --------------------------------------------------------------------------------------------------
 * Determine the range of pixels in a specified line of an input image
 * which are inside the output image. Range is one-sided, that is, the second
 * value returned is one greater than the last pixel that is on the image.
 *
 * p:       the stucture containing the image pointers
 * margin:  a margin in pixels added to the limits
 * j:       the index of the line in the input image whose range is computed
 * xbounds: the input pixels bounding the overlap (output)
 */

int
check_line_overlap(struct driz_param_t* p, int margin, integer_t j, integer_t *xbounds) {
  struct segment outlimit, xybounds;
  integer_t isize[2], osize[2];

  get_dimensions(p->output_data, osize);
  initialize_segment(&outlimit, - margin, - margin,
                     osize[0] + margin, osize[1] + margin);

  initialize_segment(&xybounds, p->xmin, j, p->xmax, j+1);
  shrink_segment(&xybounds, p->pixmap, &bad_pixel);

  if (clip_bounds(p->pixmap, &outlimit, &xybounds)) {
    driz_error_set_message(p->error, "cannot compute xbounds");
    return 1;
  }

  sort_segment(&xybounds, 0);
  shrink_segment(&xybounds, p->weights, &bad_weight);

  xbounds[0] = floor(xybounds.point[0][0]);
  xbounds[1] = ceil(xybounds.point[1][0]);

  get_dimensions(p->data, isize);
  if (driz_error_check(p->error, "xbounds must be inside input image",
                       xbounds[0] >= 0 && xbounds[1] <= isize[0])) {
    return 1;

  } else {
    return 0;
  }
}

/** --------------------------------------------------------------------------------------------------
 * Determine the range of lines in the input image that overlap the output image
 * Range is one-sided, that is, the second value returned is one greater than the
 * last line that is on the image.
 *
 * p:       the stucture containing the image pointers
 * margin:  a margin in pixels added to the limits
 * ybounds: the input lines bounding the overlap (output)
 */

int
check_image_overlap(struct driz_param_t* p, const int margin, integer_t *ybounds) {

  struct segment inlimit, outlimit, xybounds[2];
  integer_t isize[2], osize[2];
  int ipoint;

  get_dimensions(p->output_data, osize);
  initialize_segment(&outlimit, - margin, - margin,
                     osize[0] + margin, osize[1] + margin);

  initialize_segment(&inlimit, p->xmin, p->ymin, p->xmax, p->ymax);
  shrink_segment(&inlimit, p->pixmap, &bad_pixel);

  if (inlimit.invalid == 1) {
      driz_error_set_message(p->error, "no valid pixels on input image");
      return 1;
    }

  for (ipoint = 0; ipoint < 2; ++ipoint) {
    initialize_segment(&xybounds[ipoint],
                       inlimit.point[ipoint][0], inlimit.point[0][1],
                       inlimit.point[ipoint][0], inlimit.point[1][1]);

    if (clip_bounds(p->pixmap, &outlimit, &xybounds[ipoint])) {
      driz_error_set_message(p->error, "cannot compute ybounds");
      return 1;
    }
  }

  union_of_segments(2, 1, xybounds, ybounds);

  get_dimensions(p->pixmap, isize);
  if (driz_error_check(p->error, "ybounds must be inside input image",
                       ybounds[0] >= 0 && ybounds[1] <= isize[1])) {
    return 1;

  } else {
    return 0;
  }
}

