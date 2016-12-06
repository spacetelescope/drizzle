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
 * Shrink the bounds to the range containing valid numbere (! is_nan)
 *
 * self: the segment
 * jdim: the dimension to shrink, x (0) or y (1)
 */

void
shrink_segment(struct segment *self, PyArrayObject *pixmap, int jdim) {
  int iside;
  int idim = (jdim + 1) % 2;
  
  for (iside = 0; iside < 2; ++iside) {
    int delta;
    integer_t pix[2];
    int jside = (iside + 1) % 2;
    
    pix[idim] = self->point[iside][idim];
    pix[jdim] = self->point[iside][jdim];
    
    if (self->point[iside][jdim] < self->point[jside][jdim]) {
      delta = 1;
    } else {
      delta = -1;
      /* Asymetric limits */
      pix[jdim] += delta;
    }
    
    while (pix[jdim] != self->point[jside][jdim]) {
      int kdim;
      int isnan = 0;

      for (kdim = 0; kdim < 2; ++kdim) {
        double pixval = get_pixmap(pixmap, pix[0], pix[1])[kdim];

        if (npy_isnan(pixval)) {
          isnan = 1;
        }
      }

      if (! isnan) {
        if (self->point[iside][jdim] < self->point[jside][jdim]) {
          self->point[iside][jdim] = pix[jdim];
        } else {
          /* Asymetric limits */
          self->point[iside][jdim] = pix[jdim] + 1;
        }
        break;
      }
    
      pix[jdim] += delta;
    }
  }

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
    bounds[1] = bounds[0];
  }

  return;
}

/** --------------------------------------------------------------------------------------------------
 * Find the points that bound the linear interpolation
 *
 * pixmap:   The mapping of the pixel centers from input to output image
 * xyin:     An (x,y) point on the input image
 * xybounds: The bounds for the linear interpolation (output)
 */

void
map_bounds(
  PyArrayObject *pixmap,
  const double  xyin[2],
  int           *xypix
  ) {

  int xy[2];
  int xydim[2];
  int xystart[2];

  int n;
  int ipix = 0;
  int d[2] = {0, 0};
  int v[2] = {0, -1};
  int *xyptr = xypix;
  
  /* Starting point rounds down input pixel position
   * to integer value
   */
  for (int idim = 0; idim < 2; ++idim) {
    xystart[idim] = floor(xyin[idim]);
  }

  /* Make sure starting point is inside image */
  get_dimensions(pixmap, xydim);
  n = 4 * xydim[0] * xydim[1];

  for (int idim = 0; idim < 2; ++idim) {
    if (xystart[idim] < 0) {
      xystart[idim] = 0;
    } else if (xystart[idim] >= xydim[idim]) {
      xystart[idim] = xydim[idim] - 1;
    }
  }

  /* Spiral around the starting point until
   * we find four valid points on the line
   */
  while (--n > 0 && ipix < 4) {
    
    /* Get next point to check */
    for (int idim = 0; idim < 2; ++idim) {
      xy[idim] = xystart[idim] + d[idim];
    }
    
    /* If we are on the image */
    if (xy[0] >= 0 && xy[0] < xydim[0] && xy[1] >= 0 && xy[1] < xydim[1]) {
      int isnan = 0;

      /* Check if the pixel value is NaN */ 
      for (int idim = 0; idim < 2; ++idim) {
        double pixval = get_pixmap(pixmap, xy[0], xy[1])[idim];
        isnan |= npy_isnan(pixval);
      }
    
      /* If not, copy it to output as a good point */
      if (! isnan) {
        for (int idim = 0; idim < 2; ++idim) {
          *xyptr++ = xy[idim];
        }
        ++ ipix;
      }
    }

    /* Change directions on spiral */
    if ((d[0] == d[1]) ||
        (d[0] < 0  && d[0] == -d[1]) ||
        (d[0] > 0  && d[0] == 1 - d[1])) {
      int t = v[0];
      v[0] = - v[1];
      v[1] = t;
    }
    
    /* Move to the next point on the spiral */
    for (int idim = 0; idim < 2; ++idim) {
      d[idim] += v[idim];
    }
  }
  
  assert(ipix == 4);
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

void
map_point(
  PyArrayObject *pixmap, 
  const double  xyin[2], 
  double        xyout[2] 
  ) {

  int xypix[4][2];
  double partial[4][2];
    
  /* Find the four points that bound the linear interpolation */
  map_bounds(pixmap, xyin, (int *)xypix);
    
  for (int ipix = 0; ipix < 4; ++ ipix) {
    /* Evaluate pixmap at these points */
    for (int idim = 0; idim < 2; ++idim) {
      partial[ipix][idim] = get_pixmap(pixmap,
                                       xypix[ipix][0],
                                       xypix[ipix][1])[idim];
    }
  }

  /* Do linear interpolation between each set of points */
  for (int npix = 4; npix > 1; npix /= 2) {
    int ipix, jpix;

    for (ipix = jpix = 0; ipix < npix; ipix += 2, jpix += 1) {
      for (int idim = 0; idim < 2; ++idim) {
        double frac = (xyin[idim] - xypix[ipix][idim]) /
                      (xypix[ipix+1][idim] - xypix[ipix][idim]);
                        
        partial[jpix][idim] = (1.0 - frac) * partial[ipix][idim] +
                              frac * partial[ipix+1][idim];
      }
    }
  }

  for (int idim = 0; idim < 2; ++idim) {
    xyout[idim] = partial[0][idim];
  }
}

/** --------------------------------------------------------------------------------------------------
 * Clip a line segment from an input image to the limits of an output image along one dimension
 *
 * pixmap:   the mapping between input and output images
 * xylimit:  the limits  of the output image
 * xybounds: the clipped line segment (output)
 * 
 */

int
clip_bounds(PyArrayObject *pixmap, struct segment *xylimit, struct segment *xybounds) {
  int ipoint, idim, jdim;
  
  xybounds->invalid = 1; /* Track if bounds are both outside the image */

  for (idim = 0; idim < 2; ++idim) {
    shrink_segment(xybounds, pixmap, idim);

    for (ipoint = 0; ipoint < 2; ++ipoint) {
      int m = 21;         /* maximum iterations */
      int side = 0;       /* flag indicating which side moved last */
  
      double xyin[2], xyout[2];
      double a, b, c, fa, fb, fc;
      
      /* starting values at endpoints of interval */
  
      for (jdim = 0; jdim < 2; ++jdim) {
        xyin[jdim] = xybounds->point[0][jdim];
      }
      
      map_point(pixmap, xyin, xyout);
      a = xybounds->point[0][idim];
      fa = xyout[idim] - xylimit->point[ipoint][idim];
      
      for (jdim = 0; jdim < 2; ++jdim) {
        xyin[jdim] = xybounds->point[1][jdim];
      }
  
      map_point(pixmap, xyin, xyout);
      c = xybounds->point[1][idim];
      fc = xyout[idim] - xylimit->point[ipoint][idim];
  
      /* Solution via the method of false position (regula falsi) */
  
      if (fa * fc < 0.0) {
        int n; /* for loop limit is just for safety's sake */
        
        for (n = 0; n < m; n++) {
          b = (fa * c - fc * a) / (fa - fc);
          
          /* Solution is exact if within a pixel because linear interpolation */
          if (floor(a) == floor(c)) break;
          
          xyin[idim] = b;
          map_point(pixmap, xyin, xyout);
          fb = xyout[idim] - xylimit->point[ipoint][idim];
  
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
          xybounds->invalid = 1;
          return 1;
        }

        xybounds->invalid = 0;
        xybounds->point[ipoint][idim] = b;
  
      } else {
        /* No bracket, so track which side the bound lies on */
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
  struct segment xylimit, xybounds;
  integer_t isize[2], osize[2];
  
  get_dimensions(p->output_data, osize);  
  initialize_segment(&xylimit, - margin, - margin,
                     osize[0] + margin, osize[1] + margin);

  initialize_segment(&xybounds, p->xmin, j, p->xmax, j);

  if (clip_bounds(p->pixmap, &xylimit, &xybounds)) {
    driz_error_set_message(p->error, "cannot compute xbounds");
    return 1;
  }
 
  sort_segment(&xybounds, 0);
  
  xbounds[0] = floor(xybounds.point[0][0]);
  xbounds[1] = ceil(xybounds.point[1][0]);

  get_dimensions(p->pixmap, isize);
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

  struct segment xylimit, xybounds[2];
  integer_t isize[2], osize[2];
  int ipoint;
  
  ybounds[0] = p->xmin;
  ybounds[1] = p->xmax;
  
  get_dimensions(p->output_data, osize);  
  initialize_segment(&xylimit, - margin, - margin,
                     osize[0] + margin, osize[1] + margin);

  for (ipoint = 0; ipoint < 2; ++ipoint) {
    initialize_segment(&xybounds[ipoint], ybounds[ipoint], p->ymin,
                                          ybounds[ipoint], p->ymax);
    
    if (clip_bounds(p->pixmap, &xylimit, &xybounds[ipoint])) {
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

