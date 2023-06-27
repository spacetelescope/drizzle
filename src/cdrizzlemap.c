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

#include <float.h>

static const double VERTEX_ATOL = 1.0e-12;
static const double APPROX_ZERO = 1.0e3 * DBL_MIN;
static const double MAX_INV_ERR = 0.03;


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


static int
eval_inversion(PyArrayObject *pixmap, double x, double y,
               double xyref[2], double *dist2) {
    double xy[2], xyi[2], dx, dy;

    xy[0] = x;
    xy[1] = y;

    if (interpolate_point(pixmap, xy, xyi)) {
        return 1;
    }
    dx = xyi[0] - xyref[0];
    dy = xyi[1] - xyref[1];
    *dist2 = dx * dx + dy * dy;  // sqrt would be slower

    return 0;
}


int
invert_pixmap(PyArrayObject *pixmap, const double xyout[2], double xyin[2]) {
    // invert input 'xyout' (output image) coordinates iteratively to the input
    // image coordinates 'xyin' - output of this function.

    const double gr = 0.6180339887498948482;  // Golden Ratio: (sqrt(5)-1)/2
    const int nmax_iter = 50;
    int nx, ny, niter;
    double xmin, xmax, ymin, ymax, dx, dy, x1, x2, y1, y2;
    double d11, d12, d21, d22;
    npy_intp *ndim;

    ndim = PyArray_DIMS(pixmap);
    nx = ndim[1];
    ny = ndim[0];

    xmin = -0.5;
    xmax = (double) nx - 0.5;
    ymin = -0.5;
    ymax = (double) ny - 0.5;
    dx = xmax;
    dy = ymax;

    niter = 0;

    while ((dx > MAX_INV_ERR || dy > MAX_INV_ERR) && niter < nmax_iter) {
        niter+=1;

        x1 = xmax - gr * dx;
        x2 = xmin + gr * dx;
        y1 = ymax - gr * dy;
        y2 = ymin + gr * dy;

        if (eval_inversion(pixmap, x1, y1, xyout, &d11)) return 1;
        if (eval_inversion(pixmap, x1, y2, xyout, &d12)) return 1;
        if (eval_inversion(pixmap, x2, y1, xyout, &d21)) return 1;
        if (eval_inversion(pixmap, x2, y2, xyout, &d22)) return 1;

        if (d11 < d12 && d11 < d21 && d11 < d22) {
            xmax = x2;
            ymax = y2;
        } else if (d12 < d11 && d12 < d21 && d12 < d22) {
            xmax = x2;
            ymin = y1;
        } else if (d21 < d11 && d21 < d12 && d21 < d22) {
            xmin = x1;
            ymax = y2;
        } else {
            xmin = x1;
            ymin = y1;
        }

        dx = xmax - xmin;
        dy = ymax - ymin;
    }

    xyin[0] = 0.5 * (xmin + xmax);
    xyin[1] = 0.5 * (ymin + ymax);

    if (niter == nmax_iter) return 1;

    return 0;
}


// computes modulus of a % b  (with b > 0) similar to Python. A more robust
// approach would be to do this: (((a % b) + b) % b). However the polygon
// intersection code will never have a < -1 and so a simplified and faster
// version was implemented that works for a >= -b.
inline int
mod(int a, int b) {
    return ((a + b) % b);
    // return (((a % b) + b) % b);
}

// test whether two vertices (points) are equal to within a specified
// absolute tolerance
static inline int
equal_vertices(struct vertex a, struct vertex b, double atol) {
    return (fabs(a.x - b.x) < atol && fabs(a.y - b.y) < atol);
}

// Z-axis/k-component of the cross product a x b
static inline double
area(struct vertex a, struct vertex b) {
    return (a.x * b.y - a.y * b.x);
}


// tests whether a point is in a half-plane of the vector going from
// vertex v_ to vertex v (including the case of the point lying on the
// vector (v_, v)). Specifically, it tests (v - v_) x (pt - v_) >= 0:
static inline int
is_point_in_hp(struct vertex pt, struct vertex v_, struct vertex v) {
    // (v - v_) x (pt - v_) = v x pt - v x v_ - v_ x pt + v_ x v_ =
    // = v x pt - v x v_ - v_ x pt
    return ((area(v, pt) - area(v_, pt) - area(v, v_)) >= -APPROX_ZERO);
}


// same as is_point_in_hp but tests strict inequality (point not on the vector)
static inline int
is_point_strictly_in_hp(const struct vertex pt, const struct vertex v_,
                        const struct vertex v) {
    return ( (area(v, pt) - area(v_, pt) - area(v, v_)) > APPROX_ZERO );
}


// returns 1 if all vertices from polygon p are inside polygon q or 0 if
// at least one vertex of p is outside of q.
static inline int
is_poly_contained(const struct polygon *p, const struct polygon *q) {
    int i, j;
    struct vertex *v_, *v;

    v_ = q->v + (q->npv - 1);
    v = q->v;

    for (i = 0; i < q->npv; i++) {
        for (j = 0; j < p->npv; j++) {
            if (!is_point_in_hp(p->v[j], *v_, *v)) {
                return 0;
            }
        }
        v_ = v;
        v++;
    }

    return 1;
}


// Append a vertex to the polygon's list of vertices and increment
// vertex count.
// return 1 if storage capacity is exceeded or 0 on success
static int
append_vertex(struct polygon *p, struct vertex v) {
    if ((p->npv > 0) && equal_vertices(p->v[p->npv - 1], v, VERTEX_ATOL)) {
        return 0;
    }
    if ((p->npv > 0) && equal_vertices(p->v[0], v, VERTEX_ATOL)) {
        return 1;
    }
    if (p->npv >= 2 * IMAGE_OUTLINE_NPTS - 1) {
        return 1;
    }
    p->v[p->npv++] = v;
    return 0;
}


// remove midpoints (if any) - vertices that lie on a line connecting
// other two vertices
static void
simplify_polygon(struct polygon *p) {
    struct polygon pqhull;
    struct vertex dp, dq, *pv, *pv_, *pvnxt;
    int k;

    if (p->npv < 3) return;

    pqhull.npv = 0;

    pv_ = (struct vertex *)(p->v) + (p->npv - 1);
    pv = (struct vertex *)p->v;
    pvnxt = ((struct vertex *)p->v) + 1;

    for (k = 0; k < p->npv; k++) {
        dp.x = pvnxt->x - pv_->x;
        dp.y = pvnxt->y - pv_->y;
        dq.x = pv->x - pv_->x;
        dq.y = pv->y - pv_->y;

        if (fabs(area(dp, dq)) > APPROX_ZERO &&
            sqrt(dp.x * dp.x + dp.y * dp.y) > VERTEX_ATOL) {
            pqhull.v[pqhull.npv++] = *pv;
        }
        pv_ = pv;
        pv = pvnxt;
        pvnxt = ((struct vertex *)p->v) + (mod(2 + k, p->npv));
    }

    p->npv = pqhull.npv;
    for (k = 0; k < p->npv; k++) {
        p->v[k] = pqhull.v[k];
    }
}


int
intersect_convex_polygons(const struct polygon *p, const struct polygon *q,
                          struct polygon *pq) {

    int ip=0, iq=0, first_k, k;
    int inside=0;  // 0 - not set, 1 - "P", -1 - "Q"
    int pv_in_hpdq, qv_in_hpdp;
    struct vertex *pv, *pv_, *qv, *qv_, dp, dq, vi, first_intersect;
    double t, u, d, dot, signed_area;

    if ((p->npv < 3) || (q->npv < 3)) {
        return 1;
    }

    if (is_poly_contained(p, q)) {
        *pq = *p;
        simplify_polygon(pq);
        return 0;
    } else if (is_poly_contained(q, p)) {
        *pq = *q;
        simplify_polygon(pq);
        return 0;
    }

    pv_ = (struct vertex *)(p->v + (p->npv - 1));
    pv = (struct vertex *)p->v;
    qv_ = (struct vertex *)(q->v + (q->npv - 1));
    qv = (struct vertex *)q->v;

    first_k = -2;
    pq->npv = 0;

    for (k = 0; k <= 2 * (p->npv + q->npv); k++) {
        dp.x = pv->x - pv_->x;
        dp.y = pv->y - pv_->y;
        dq.x = qv->x - qv_->x;
        dq.y = qv->y - qv_->y;

        // https://en.wikipedia.org/wiki/Line–line_intersection
        t = (pv_->y - qv_->y) * dq.x - (pv_->x - qv_->x) * dq.y;
        u = (pv_->y - qv_->y) * dp.x - (pv_->x - qv_->x) * dp.y;
        signed_area = area(dp, dq);
        if (signed_area >= 0.0) {
            d = signed_area;
        } else {
            t = -t;
            u = -u;
            d = -signed_area;
        }

        pv_in_hpdq = is_point_strictly_in_hp(*qv_, *qv, *pv);
        qv_in_hpdp = is_point_strictly_in_hp(*pv_, *pv, *qv);

        if ((0.0 <= t) && (t <= d) && (0.0 <= u) && (u <= d) &&
            (d > APPROX_ZERO)) {
            t = t / d;
            u = u / d;
            vi.x = pv_->x + (pv->x - pv_->x) * t;
            vi.y = pv_->y + (pv->y - pv_->y) * t;

            if (first_k < 0) {
                first_intersect = vi;
                first_k = k;
                if (append_vertex(pq, vi)) break;
            } else if (equal_vertices(first_intersect, vi, VERTEX_ATOL)) {
                if (k > (first_k + 1)) {
                    break;
                }
                first_k = k;
            } else {
                if (append_vertex(pq, vi)) break;
            }

            if (pv_in_hpdq) {
                inside = 1;
            } else if (qv_in_hpdp) {
                inside = -1;
            }
        }

        // advance:
        if (d < 1.0e-12 && !pv_in_hpdq && !qv_in_hpdp) {
            if (inside == 1) {
                iq += 1;
                qv_ = qv;
                qv = q->v + mod(iq, q->npv);
            } else {
                ip += 1;
                pv_ = pv;
                pv = p->v + mod(ip, p->npv);
            }

        } else if (signed_area >= 0.0) {
            if (qv_in_hpdp) {
                if (inside == 1) {
                    if (append_vertex(pq, *pv)) break;
                }
                ip += 1;
                pv_ = pv;
                pv = p->v + mod(ip, p->npv);
            } else {
                if (inside == -1) {
                    if (append_vertex(pq, *qv)) break;
                }
                iq += 1;
                qv_ = qv;
                qv = q->v + mod(iq, q->npv);
            }

        } else {
            if (pv_in_hpdq) {
                if (inside == -1) {
                    if (append_vertex(pq, *qv)) break;
                }
                iq += 1;
                qv_ = qv;
                qv = q->v + mod(iq, q->npv);
            } else {
                if (inside == 1) {
                    if (append_vertex(pq, *pv)) break;
                }
                ip += 1;
                pv_ = pv;
                pv = p->v + mod(ip, q->npv);
            }
        }
    }

    simplify_polygon(pq);

    return 0;
}


static void
init_edge(struct edge *e, struct vertex v1, struct vertex v2, int position) {
    e->v1 = v1;
    e->v2 = v2;
    e->p = position;  // -1 for left-side edge and +1 for right-side edge
    e->m = (v2.x - v1.x) / (v2.y - v1.y);
    e->b = (v1.x * v2.y - v1.y * v2.x) / (v2.y - v1.y);
    e->c = e->b - copysign(0.5 + 0.5 * fabs(e->m), (double) position);
};


/*
bbox = [[xmin, xmax], [ymin, ymax]]
*/
int
init_scanner(struct polygon *p, struct scanner *s,
             int image_width, int image_height) {
    int k, i1, i2;
    int min_right, min_left, max_right, max_left;
    double min_y, max_y;

    s->left = NULL;
    s->right = NULL;
    s->nleft = 0;
    s->nright = 0;

    if (p->npv < 3) {
        // not a polygon
        return 1;
    }

    // find minimum/minima:
    min_y = p->v[0].y;
    min_left = 0;
    for (k = 1; k < p->npv; k++) {
        if (p->v[k].y < min_y) {
            min_left = k;
            min_y = p->v[k].y;
        }
    }

    i1 = mod(min_left - 1, p->npv);
    i2 = mod(min_left + 1, p->npv);
    min_right = ( p->v[i1].y < p->v[i2].y ) ? i1 : i2;
    if (p->v[min_right].y <= min_y * (1.0 + copysign(VERTEX_ATOL, min_y))) {
        if (p->v[min_left].x > p->v[min_right].x) {
            k = min_left;
            min_left = min_right;
            min_right = k;
        }
    } else {
        min_right = min_left;
    }

    // find maximum/maxima:
    max_y = p->v[0].y;
    max_right = 0;
    for (k = 1; k < p->npv; k++) {
        if (p->v[k].y > max_y) {
            max_right = k;
            max_y = p->v[k].y;
        }
    }

    i1 = mod(max_right - 1, p->npv);
    i2 = mod(max_right + 1, p->npv);
    max_left = ( p->v[i1].y > p->v[i2].y ) ? i1 : i2;
    if (p->v[max_left].y >= max_y * (1.0 - copysign(VERTEX_ATOL, max_y))) {
        if (p->v[max_left].x > p->v[max_right].x) {
            k = max_left;
            max_left = max_right;
            max_right = k;
        }
    } else {
        max_left = max_right;
    }

    // Left: start with minimum and move counter-clockwise:
    if (max_left > min_left) {
        min_left += p->npv;
    }
    s->nleft = min_left - max_left;

    for (k = 0; k < s->nleft; k++) {
        i1 = mod(min_left - k, p->npv);
        i2 = mod(i1 - 1, p->npv);
        init_edge(s->left_edges + k, p->v[i1], p->v[i2], -1);
    }

    // Right: start with minimum and move clockwise:
    if (max_right < min_right) {
        max_right += p->npv;
    }
    s->nright = max_right - min_right;

    for (k = 0; k < s->nright; k++) {
        i1 = mod(min_right + k, p->npv);
        i2 = mod(i1 + 1, p->npv);
        init_edge(s->right_edges + k, p->v[i1], p->v[i2], 1);
    }

    s->left = (struct edge *) s->left_edges;
    s->right = (struct edge *) s->right_edges;
    s->ymin = min_y;
    s->ymax = max_y;
    s->width = image_width;
    s->height = image_height;

    return 0;
}

/*
get_scanline_limits returns x-limits for an image row that fits between edges
(of a polygon) specified by the scanner structure.

This function is intended to be called successively with input 'y' *increasing*
from s->ymin to s->ymax.

Return code:
    0 - no errors
    1 - scan ended (y reached the top vertex/edge)
    2 - pixel centered on y is outside of scanner's limits or image [0, height - 1]
    3 - limits (x1, x2) are equal (line with is 0)

*/
int
get_scanline_limits(struct scanner *s, int y, int *x1, int *x2) {
    double pyb, pyt;  // pixel top and bottom limits
    double xlb, xlt, xrb, xrt, edge_ymax;
    struct edge *el_max, *er_max;

    el_max = ((struct edge *) s->left_edges) + (s->nleft - 1);
    er_max = ((struct edge *) s->right_edges) + (s->nright - 1);

    if (s->height >= 0 && (y < 0 || y >= s->height)) {
        return 2;
    }

    pyb = (double)y - 0.5;
    pyt = (double)y + 0.5;

    if (pyt <= s->ymin || pyb >= s->ymax + 1) {
        return 2;
    }

    if (s->left == NULL || s->right == NULL) {
        return 1;
    }

    while (pyb > s->left->v2.y) {
        if (s->left == el_max) {
            s->left = NULL;
            s->right = NULL;
            return 1;
        }
        ++s->left;
    };

    while (pyb > s->right->v2.y) {
        if (s->right == er_max) {
            s->left = NULL;
            s->right = NULL;
            return 1;
        }
        ++s->right;
    };

    /* For double precision:
    xlb = s->left->m * y + s->left->c;
    xrb = s->right->m * y + s->right->c;
    */
    xlb = (int)round(s->left->m * y + s->left->c + 0.0);
    xrb = (int)round(s->right->m * y + s->right->c + 0.0);
    xlb = s->left->m * y + s->left->c - MAX_INV_ERR;
    xrb = s->right->m * y + s->right->c + MAX_INV_ERR;

    edge_ymax = s->left->v2.y + 0.5 + MAX_INV_ERR;
    while (pyt > edge_ymax) {
        if (s->left == el_max) {
            s->left = NULL;
            s->right = NULL;
            return 1;
        }
        ++s->left;
        edge_ymax = s->left->v2.y + 0.5 + MAX_INV_ERR;
    };

    edge_ymax = s->right->v2.y + 0.5 + MAX_INV_ERR;
    while (pyt > edge_ymax) {
        if (s->right == er_max) {
            s->left = NULL;
            s->right = NULL;
            return 1;
        }
        ++s->right;
        edge_ymax = s->right->v2.y + 0.5 + MAX_INV_ERR;
    };

    /* For double precision:
    xlt = s->left->m * y + s->left->c;
    xrt = s->right->m * y + s->right->c;
    */
    xlt = (int)(s->left->m * y + s->left->c + 0.5 + MAX_INV_ERR);
    xrt = (int)(s->right->m * y + s->right->c + 0.5 + MAX_INV_ERR);
    xlt = s->left->m * y + s->left->c - MAX_INV_ERR;
    xrt = s->right->m * y + s->right->c + MAX_INV_ERR;

    if (s->width >= 0) {
        if (xlb < -0.5) {
            xlb = -0.5;
        }
        if (xlt < -0.5) {
            xlt = -0.5;
        }
        if (xrb >= s->width) {
            xrb = s->width - 0.5;
        }
        if (xrt >= s->width) {
            xrt = s->width - 0.5;
        }
    }

    if (xlt >= xrt) {
        *x1 = (int)round(xlb);
        *x2 = (int)round(xrb);
        if (xlb >= xrb) {
            return 3;
        }
    } else if (xlb >= xrb) {
        *x1 = (int)round(xlt);
        *x2 = (int)round(xrt);
    } else {
        *x1 = (int)round((xlb > xlt) ? xlb : xlt);
        *x2 = (int)round((xrb < xrt) ? xrb : xrt);
    }

    return 0;
}


static int
map_to_output_vertex(struct driz_param_t* par, double x, double y, struct vertex *v) {
    double xyin[2], xyout[2];
    // convert coordinates to the output frame
    xyin[0] = x;
    xyin[1] = y;
    if (map_point(par->pixmap, xyin, xyout)) {
        driz_error_set_message(par->error,
            "error computing input image bounding box");
        return 1;
    }
    v->x = xyout[0];
    v->y = xyout[1];
    return 0;
}


static int
map_to_input_vertex(struct driz_param_t* par, double x, double y, struct vertex *v) {
    double xyin[2], xyout[2];
    char buf[MAX_DRIZ_ERROR_LEN];
    int n;
    // convert coordinates to the input frame
    xyout[0] = x;
    xyout[1] = y;
    if (invert_pixmap(par->pixmap, xyout, xyin)) {
        n = sprintf(buf,
            "failed to invert pixel map at position (%.2f, %.2f)", x, y);
        if (n < 0) {
            strcpy(buf, "failed to invert pixel map");
        }
        driz_error_set_message(par->error, buf);
        return 1;
    }
    v->x = xyin[0];
    v->y = xyin[1];
    return 0;
}


int
init_image_scanner(struct driz_param_t* par, struct scanner *s,
                   int *ymin, int *ymax) {
    struct polygon p, q, pq, inpq;
    double  xyin[2], xyout[2];
    integer_t isize[2], osize[2];
    int ipoint;
    int k, n;
    npy_intp *ndim;
    int ixmin, ixmax, iymin, iymax, in_width, in_height;

    // find smallest bounding box for the input image:
    ndim = PyArray_DIMS(par->data);
    in_width = ndim[1];
    in_height = ndim[0];
    ixmin = par->xmin > 0 ? (int)par->xmin : 0;
    ixmax = ndim[1] - 1;
    if (ixmax > (int)par->xmax) ixmax = (int)par->xmax;

    iymin = par->ymin > 0 ? (int)par->ymin : 0;
    iymax = ndim[0] - 1;
    if (iymax > (int)par->ymax) iymax = (int)par->ymax;

    // convert coordinates to the output frame and define a polygon
    // bounding the input image in the output frame:
    if (map_to_output_vertex(par, ixmin - 0.5, iymin - 0.5, p.v)) return 1;
    if (map_to_output_vertex(par, ixmax + 0.5, iymin - 0.5, p.v + 1)) return 1;
    if (map_to_output_vertex(par, ixmax + 0.5, iymax + 0.5, p.v + 2)) return 1;
    if (map_to_output_vertex(par, ixmin - 0.5, iymax + 0.5, p.v + 3)) return 1;
    p.npv = 4;

    // define a polygon bounding output image:
    ndim = PyArray_DIMS(par->output_data);
    q.npv = 4;
    q.v[0].x = -0.5;
    q.v[0].y = -0.5;
    q.v[1].x = ndim[1] - 0.5;
    q.v[1].y = -0.5;
    q.v[2].x = ndim[1] - 0.5;
    q.v[2].y = ndim[0] - 0.5;
    q.v[3].x = -0.5;
    q.v[3].y = ndim[0] - 0.5;

    // compute intersection of P and Q (in output frame):
    if (intersect_convex_polygons(&p, &q, &pq)) {
        driz_error_set_message(par->error,
            "failed to compute polygon intersection");
        return 1;
    }

    // convert coordinates of vertices of the intersection polygon
    // back to input image coordinate system:
    for (k = 0; k < pq.npv; k++) {
        if (map_to_input_vertex(par, pq.v[k].x, pq.v[k].y, &inpq.v[k])) {
            return 1;
        }
    }
    inpq.npv = pq.npv;

    // initialize polygon scanner:
    n = init_scanner(&inpq, s, in_width, in_height);
    *ymin = MAX(0, (int)(s->ymin + 0.5 + 2.0 * MAX_INV_ERR));
    *ymax = MIN(in_height - 1, (int)(s->ymax + 2.0 * MAX_INV_ERR));
    return n;
}
