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

/** ---------------------------------------------------------------------------
 * Find the tighest bounding box around valid (finite) pixmap values.
 *
 * This function takes as input a pixel map array and four values indicating
 * some given bounding box defined by: xmin, xmax, ymin, ymax. Starting with
 * these values, this function checks values of pixel map on the border and
 * if there are no valid values along one or more edges, it will adjust the
 * values of xmin, xmax, ymin, ymax to find the tightest box that has
 * at least one valid pixel on every edge of the bounding box.
 *
 * @param[in] PyArrayObject *pixmap - pixel map of shape (N, M, 2).
 * @param[in,out] int xmin - position of the left edge of the bounding box.
 * @param[in,out] int xmax - position of the right edge of the bounding box.
 * @param[in,out] int ymin - position of the bottom edge of the bounding box.
 * @param[in,out] int ymax - position of the top edge of the bounding box.
 * @return 0 if successul and 1 if there is only one or no valid pixel map
 * values.
 *
 */
int
shrink_image_section(PyArrayObject *pixmap, int *xmin, int *xmax, int *ymin,
                     int *ymax) {
    int i, j, imin, imax, jmin, jmax, i1, i2, j1, j2;
    double *pv;

    j1 = *ymin;
    j2 = *ymax;
    i1 = *xmin;
    i2 = *xmax;

    imin = i2;
    jmin = j2;

    for (j = j1; j <= j2; ++j) {
        for (i = i1; i <= i2; ++i) {
            pv = (double *)PyArray_GETPTR3(pixmap, j, i, 0);
            if (!(npy_isnan(pv[0]) || npy_isnan(pv[1]))) {
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

    imax = imin;
    jmax = jmin;

    for (j = j2; j >= j1; --j) {
        for (i = i2; i >= i1; --i) {
            pv = (double *)PyArray_GETPTR3(pixmap, j, i, 0);
            if (!(npy_isnan(pv[0]) || npy_isnan(pv[1]))) {
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

    *xmin = imin;
    *xmax = imax;
    *ymin = jmin;
    *ymax = jmax;

    return (imin >= imax || jmin >= jmax);
}

/** ---------------------------------------------------------------------------
 * Map a point on the input image to the output image using
 * a mapping of the pixel centers between the two by interpolating
 * between the centers in the mapping
 *
 * pixmap: The mapping of the pixel centers from input to output image
 * xyin:   An (x,y) point on the input image
 * xyout:  The same (x, y) point on the output image (output)
 */
int
interpolate_point(struct driz_param_t *par, double xin, double yin,
                  double *xout, double *yout) {
    int ipix, jpix, npix, idim;
    int i0, j0, nx2, ny2;
    npy_intp *ndim;
    double x, y, x1, y1, f00, f01, f10, f11, g00, g01, g10, g11;
    double *p;
    PyArrayObject *pixmap;

    pixmap = par->pixmap;

    /* Bilinear interpolation from
       https://en.wikipedia.org/wiki/Bilinear_interpolation#On_the_unit_square
    */
    i0 = (int)xin;
    j0 = (int)yin;

    ndim = PyArray_DIMS(pixmap);
    nx2 = (int)ndim[1] - 2;
    ny2 = (int)ndim[0] - 2;

    // point is outside the interpolation range. adjust limits to extrapolate.
    if (i0 < 0) {
        i0 = 0;
    } else if (i0 > nx2) {
        i0 = nx2;
    }
    if (j0 < 0) {
        j0 = 0;
    } else if (j0 > ny2) {
        j0 = ny2;
    }

    x = xin - i0;
    y = yin - j0;
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

    *xout = f00 * x1 * y1 + f10 * x * y1 + f01 * x1 * y + f11 * x * y;
    *yout = g00 * x1 * y1 + g10 * x * y1 + g01 * x1 * y + g11 * x * y;

    if (npy_isnan(*xout) || npy_isnan(*yout)) return 1;

    return 0;
}

/** ---------------------------------------------------------------------------
 * Map an integer pixel position from the input to the output image.
 * Fall back on interpolation if the value at the point is undefined
 *
 * pixmap: The mapping of the pixel centers from input to output image
 * i - The index of the x coordinate
 * j - The index of the y coordinate
 * x - X-coordinate of the point on the output image (output)
 * y - Y-coordinate of the point on the output image (output)
 */

int
map_pixel(PyArrayObject *pixmap, int i, int j, double *x, double *y) {
    double *pv = (double *)PyArray_GETPTR3(pixmap, j, i, 0);
    *x = *pv;
    *y = *(pv + 1);
    return ((npy_isnan(*x) || npy_isnan(*y)) ? 1 : 0);
}

/** ---------------------------------------------------------------------------
 * Map a point on the input image to the output image either by interpolation
 * or direct array access if the input position is integral.
 *
 * pixmap: The mapping of the pixel centers from input to output image
 * xin:   X-coordinate of a point on the input image
 * yin:   Y-coordinate of a point on the input image
 * xout:  X-coordinate of the same point on the output image (output)
 * yout:  Y-coordinate of the same point on the output image (output)
 *
 */
int
map_point(struct driz_param_t *par, double xin, double yin, double *xout,
          double *yout) {
    int i, j, status;

    i = (int)xin;
    j = (int)yin;

    if ((double)i == xin && (double)j == yin) {
        if (i >= par->xmin && i <= par->xmax && j >= par->ymin &&
            j <= par->ymax) {
            status = map_pixel(par->pixmap, i, j, xout, yout);
        } else {
            return 1;
        }
    } else {
        return interpolate_point(par, xin, yin, xout, yout);
    }
}

/** ---------------------------------------------------------------------------
 * Evaluate quality of coordinate inversion.
 *
 * Given a pair of coordinates (x, y) in the input frame obtained from
 * a pair of coordinates (xref, yref) in the output frame, this function uses
 * par->pixmap to perform forward transformation (input to output frame) of
 * (x, y) back to the output frame (x', y') and then compares interpolated
 * values with (xref, yref) pair.
 *
 * @param[in] struct driz_param_t - drizzle parameters.
 * @param[in] double x - x-coordinate of the inverted point in the input frame.
 * @param[in] double y - y-coordinate of the inverted point in the input frame.
 * @param[in] double xref - x-coordinate of the initial point in the output
 *                   frame.
 * @param[in] double yref - y-coordinate of the initial point in the output
 *                   frame.
 * @param[out] double *dist2 - |(x', y') - (xref, yref)|**2.
 * @return 0 if successul and 1 if the forward interpolation fails.
 *
 */
static int
eval_inversion(struct driz_param_t *par, double x, double y, double xref,
               double yref, double *dist2) {
    double xout, yout, dx, dy;

    if (interpolate_point(par, x, y, &xout, &yout)) {
        return 1;
    }
    dx = xout - xref;
    dy = yout - yref;
    *dist2 = dx * dx + dy * dy;  // sqrt would be slower

    return 0;
}

/** ---------------------------------------------------------------------------
 * Inverse mapping of coordinates from the output frame to input frame.
 *
 * Inverts input (xout, yout) (output image frame) coordinates iteratively
 * to the input image image frame (xin, yin) - the output of this function.
 * Ths function uses the method of Golden-section search - see
 * https://en.wikipedia.org/wiki/Golden-section_search for the 1D case -
 * generalized to support planar data.
 *
 * @param[in] struct driz_param_t - drizzle parameters.
 * @param[in] double xout - x-coordinate of a point in the output frame.
 * @param[in] double yout - y-coordinate of a point in the output frame.
 * @param[out] double *xin - x-coordinate of the point in the input frame.
 * @param[out] double *yin - y-coordinate of the point in the input frame.
 * @return 0 if successul and 1 iterative process fails.
 *
 */
int
invert_pixmap(struct driz_param_t *par, double xout, double yout, double *xin,
              double *yin) {
    const double gr = 0.6180339887498948482;  // Golden Ratio: (sqrt(5)-1)/2
    const int nmax_iter = 50;
    int niter;
    double xmin, xmax, ymin, ymax, dx, dy, x1, x2, y1, y2;
    double d11, d12, d21, d22;

    xmin = ((double)par->xmin) - 0.5;
    xmax = ((double)par->xmax) + 0.5;
    ymin = ((double)par->ymin) - 0.5;
    ymax = ((double)par->ymax) + 0.5;
    dx = xmax;
    dy = ymax;

    niter = 0;

    while ((dx > MAX_INV_ERR || dy > MAX_INV_ERR) && niter < nmax_iter) {
        niter += 1;

        x1 = xmax - gr * dx;
        x2 = xmin + gr * dx;
        y1 = ymax - gr * dy;
        y2 = ymin + gr * dy;

        if (eval_inversion(par, x1, y1, xout, yout, &d11)) return 1;
        if (eval_inversion(par, x1, y2, xout, yout, &d12)) return 1;
        if (eval_inversion(par, x2, y1, xout, yout, &d21)) return 1;
        if (eval_inversion(par, x2, y2, xout, yout, &d22)) return 1;

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

    *xin = 0.5 * (xmin + xmax);
    *yin = 0.5 * (ymin + ymax);

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
// vector (v_, v)). Specifically, it tests (v - v_) x (pt - v_) > 0:
static inline int
is_point_strictly_in_hp(const struct vertex pt, const struct vertex v_,
                        const struct vertex v) {
    return ((area(v, pt) - area(v_, pt) - area(v, v_)) > 0.0);
}

/**
 * Append a vertex to a polygon.
 *
 * Append a vertex to the polygon's list of vertices and increment
 * vertex count.
 *
 * @param[in,out] p struct polygon* to which the vertex is to be added.
 * @param[in] v struct vertex to be added to polygon p.
 * @return 0 on success and 1 if no more storage for vertices is available.
 */
static int
append_vertex(struct polygon *p, struct vertex v) {
    if ((p->npv > 0) && equal_vertices(p->v[p->npv - 1], v, VERTEX_ATOL)) {
        return 0;
    }
    if ((p->npv > 0) && equal_vertices(p->v[0], v, VERTEX_ATOL)) {
        return 0;
    }
    if (p->npv >= 2 * IMAGE_OUTLINE_NPTS) {
        return 1;
    }
    p->v[p->npv++] = v;
    return 0;
}

/**
 * Simplify polygon.
 *
 * Removes midpoints (if any), i.e., vertices that lie on a line connecting
 * two adjacent vertices (on each side of the mid-vertex).
 *
 * @param[in,out] struct polygon type whose vertices may be re-arranged upon
 *                return.
 */
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

/**
 * Orient a polygon counter-clockwise.
 *
 * Reverse the order of the polygon p's vertices in such a way that
 * polygon p is oriented counter-clockwise.
 *
 * @param[in,out] struct polygon type whose vertices may be re-arranged upon
 *                return.
 */
static void
orient_ccw(struct polygon *p) {
    // re-arrange (reverse the order of the) polygon p (input and output)
    // vertices in such a way that polygon p is oriented counter-clockwise.
    int k, m;
    struct vertex v1, v2, cm = {0, 0};

    if (p->npv < 3) return;

    // center of mass:
    for (k = 0; k < p->npv; ++k) {
        cm.x += p->v[k].x;
        cm.y += p->v[k].y;
    }
    cm.x /= p->npv;
    cm.y /= p->npv;

    // pick first two polygon vertices and subtract center:
    v1 = p->v[0];
    v2 = p->v[1];
    v1.x -= cm.x;
    v1.y -= cm.y;
    v2.x -= cm.x;
    v2.y -= cm.y;

    if (area(v1, v2) >= 0.0) {
        return;
    } else {
        for (k = 0; k < (p->npv / 2); ++k) {
            v1 = p->v[k];
            m = p->npv - 1 - k;
            p->v[k] = p->v[m];
            p->v[m] = v1;
        }
    }
}

/**
 * Clip a polygon to a window.
 *
 * This function implements the Sutherland-Hodgman polygon-clipping algorithm
 * as described in
 * https://www.cs.drexel.edu/~david/Classes/CS430/Lectures/L-05_Polygons.6.pdf
 *
 * @param[in] struct polygon p - polygon (last vertex != first)
 * @param[in] struct polygon wnd - clipping window - must be a a convex polygon
 *                                 (last vertex != first)
 * @param[out] struct polygon *cp - intersection polygon
 *
 * @returns 0 - success, 1 - failure (input polygons have less than 3 vertices)
 */
int
clip_polygon_to_window(const struct polygon *p, const struct polygon *wnd,
                       struct polygon *cp) {
    int k, j;
    int v1_inside, v2_inside;
    struct polygon p1, p2, *ppin, *ppout, *tpp;
    struct vertex *pv, *pv_, *wv, *wv_, dp, dw, vi;
    double t, u, d, signed_area, app_, aww_;

    if ((p->npv < 3) || (wnd->npv < 3)) {
        return 1;
    }

    orient_ccw(p);
    orient_ccw(wnd);

    p1 = *p;

    ppin = &p2;
    ppout = &p1;

    wv_ = (struct vertex *)(wnd->v + (wnd->npv - 1));
    wv = (struct vertex *)wnd->v;

    for (k = 0; k < wnd->npv; k++) {
        dw.x = wv->x - wv_->x;
        dw.y = wv->y - wv_->y;

        // use output from previous iteration as input for the current
        tpp = ppin;
        ppin = ppout;
        ppout = tpp;
        ppout->npv = 0;

        pv_ = (struct vertex *)(ppin->v + (ppin->npv - 1));
        pv = (struct vertex *)ppin->v;

        for (j = 0; j < ppin->npv; j++) {
            dp.x = pv->x - pv_->x;
            dp.y = pv->y - pv_->y;

            v1_inside = is_point_strictly_in_hp(*wv_, *wv, *pv_);
            v2_inside = is_point_strictly_in_hp(*wv_, *wv, *pv);

            if (v2_inside != v1_inside) {
                // compute intersection point:
                // https://en.wikipedia.org/wiki/Line–line_intersection
                d = area(dp, dw);  // d != 0 because (v2_inside != v1_inside)
                app_ = area(*pv, *pv_);
                aww_ = area(*wv, *wv_);
                vi.x = (app_ * dw.x - aww_ * dp.x) / d;
                vi.y = (app_ * dw.y - aww_ * dp.y) / d;

                append_vertex(ppout, vi);
                if (v2_inside) {
                    // outside to inside:
                    append_vertex(ppout, *pv);
                }
            } else if (v1_inside) {
                // both edge vertices are inside
                append_vertex(ppout, *pv);
            }
            // nothing to do when both edge vertices are outside

            // advance polygon edge:
            pv_ = pv;
            pv = pv + 1;
        }

        // advance window edge:
        wv_ = wv;
        wv = wv + 1;
    }

    orient_ccw(ppout);
    simplify_polygon(ppout);
    *cp = *ppout;

    return 0;
}

/**
 * Initializes an edge structure from two end vertices.
 *
 * Initialization includes edge vertices and computing the slope and
 * the intersect of the line connecting the vertices. Alternative intersect
 * "c" is computed in such a way that pixels in their entirety fit either
 * to the left of the right edges or to the right of left edges.
 *
 * NOTE: Left edges are the ones to the left (small X) of the line that
 *       connects top and bottom polygon's vertices and the right edges are
 *       to the right of this lighn (high X).
 *
 * @param[in] struct edge *e - pointer to the edge structure to be initialized
 * @param[in] struct vertex v1 - first vertex of the edge
 * @param[in] struct vertex v2 - second vertex of the edge
 * @param[in] position: +1 for right edge of the polygon, -1 for left edge
 *
 */
static void
init_edge(struct edge *e, struct vertex v1, struct vertex v2, int position) {
    e->v1 = v1;
    e->v2 = v2;
    e->p = position;  // -1 for left-side edge and +1 for right-side edge
    e->m = (v2.x - v1.x) / (v2.y - v1.y);
    e->b = (v1.x * v2.y - v1.y * v2.x) / (v2.y - v1.y);
    e->c = e->b - copysign(0.5 + 0.5 * fabs(e->m), (double)position);
};

/**
 * Set-up scanner structure for a polygon.
 *
 * This function finds minimum and maximum y-coordinates of the polygon
 * vertices and splits all edges int left and right edges based on their
 * horizontal position relative to the line connecting the top and bottom
 * vertices of the polygon. It also copies the bounding box parameters
 * xmin, xmax, ymin, ymax from the driz_param_t structure.
 *
 * @param[in] struct polygon *p.
 * @param[in] struct driz_param_t - drizzle parameters (bounding box is used).
 * @param[out] struct scanner *s - scanner structure to be initialized.
 * @return 0 if successful and 1 if input polygon has only 2 vertices or less.
 *
 */
int
init_scanner(struct polygon *p, struct driz_param_t *par, struct scanner *s) {
    int k, i1, i2;
    int min_right, min_left, max_right, max_left;
    double min_y, max_y;

    s->left = NULL;
    s->right = NULL;
    s->nleft = 0;
    s->nright = 0;

    if (p->npv < 3) {
        // not a polygon
        s->overlap_valid = 0;
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
    min_right = (p->v[i1].y < p->v[i2].y) ? i1 : i2;
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
    max_left = (p->v[i1].y > p->v[i2].y) ? i1 : i2;
    if (p->v[max_left].y >= max_y * (1.0 - copysign(VERTEX_ATOL, max_y))) {
        if (p->v[max_left].x > p->v[max_right].x) {
            k = max_left;
            max_left = max_right;
            max_right = k;
        }
    } else {
        max_left = max_right;
    }

    // Left: start with minimum and move clockwise:
    if (max_left > min_left) {
        min_left += p->npv;
    }
    s->nleft = min_left - max_left;

    for (k = 0; k < s->nleft; k++) {
        i1 = mod(min_left - k, p->npv);  // -k for CW traverse direction
        i2 = mod(i1 - 1, p->npv);        // -1 for CW traverse direction
        init_edge(s->left_edges + k, p->v[i1], p->v[i2], -1);
    }

    // Right: start with minimum and move counter-clockwise:
    if (max_right < min_right) {
        max_right += p->npv;
    }
    s->nright = max_right - min_right;

    for (k = 0; k < s->nright; k++) {
        i1 = mod(min_right + k, p->npv);  // +k for CW traverse direction
        i2 = mod(i1 + 1, p->npv);         // +1 for CW traverse direction
        init_edge(s->right_edges + k, p->v[i1], p->v[i2], 1);
    }

    s->left = (struct edge *)s->left_edges;
    s->right = (struct edge *)s->right_edges;
    s->min_y = min_y;
    s->max_y = max_y;
    s->xmin = par->xmin;
    s->xmax = par->xmax;
    s->ymin = par->ymin;
    s->ymax = par->ymax;

    return 0;
}

/**
 * Get x-range of pixels in a row that are within the bounds of a polygon.
 *
 * get_scanline_limits returns x-limits (integer pixel locations) for an image
 * row that fit between the edges (of a polygon) specified by the scanner
 * structure. The limits are computed in such a way that the edge pixels are
 * entirely inside the polygon.
 *
 * This function is intended to be called successively with input
 * 'y' *increasing* from s->min_y to s->max_y.
 *
 * @param[in] struct scanner *s - scanner structure
 * @param[in] y - integer position of the row along the vertical direction
 * @param[out] int *x1 - horizontal position of the leftmost pixel within
 *             the bounding polygon
 * @param[out] int *x2 - horizontal position of the rightmost pixel within
 *             the bounding polygon
 * @return 0 no errors;
 *         1 scan ended (y reached the top vertex/edge);
 *         2 pixel centered on y is outside of scanner's limits or image
 *           [0, height - 1];
 *         3 limits (x1, x2) are equal (line with is 0).
 *
 */
int
get_scanline_limits(struct scanner *s, int y, int *x1, int *x2) {
    double pyb, pyt;  // pixel top and bottom limits
    double xlb, xlt, xrb, xrt, edge_ymax, xmin, xmax;
    struct edge *el_max, *er_max;

    el_max = ((struct edge *)s->left_edges) + (s->nleft - 1);
    er_max = ((struct edge *)s->right_edges) + (s->nright - 1);

    if (s->ymax >= s->ymin && (y < 0 || y > s->ymax)) {
        return 2;
    }

    pyb = (double)y - 0.5;
    pyt = (double)y + 0.5;

    if (pyt <= s->min_y || pyb >= s->max_y + 1) {
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

    xlt = s->left->m * y + s->left->c - MAX_INV_ERR;
    xrt = s->right->m * y + s->right->c + MAX_INV_ERR;

    xmin = s->xmin;
    xmax = s->xmax;
    if (s->xmax >= s->xmin) {
        if (xlb < xmin) {
            xlb = xmin;
        }
        if (xlt < xmin) {
            xlt = xmin;
        }
        if (xrb > xmax) {
            xrb = xmax;
        }
        if (xrt > xmax) {
            xrt = xmax;
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

/**
 * Map a vertex' coordinates from the input frame to the output frame.
 *
 * @param[in] struct driz_param_t - drizzle parameters (bounding box is used)
 * @param[in] struct vertex vin - vertex' coordinates in the input frame
 * @param[out] struct vertex *vout - vertex' coordinates in the output frame
 * @return 0 no errors and 1 on errors.
 *
 */
static int
map_vertex_to_output(struct driz_param_t *par, struct vertex vin,
                     struct vertex *vout) {
    // convert coordinates to the output frame
    return map_point(par, vin.x, vin.y, &vout->x, &vout->y);
}

/**
 * Map a vertex' coordinates from the output frame to the input frame.
 *
 * @param[in] struct driz_param_t - drizzle parameters (bounding box is used)
 * @param[in] struct vertex vout - vertex' coordinates in the output frame
 * @param[out] struct vertex *vin - vertex' coordinates in the input frame
 * @return 0 no errors and 1 on errors.
 *
 */
static int
map_vertex_to_input(struct driz_param_t *par, struct vertex vout,
                    struct vertex *vin) {
    double xin, yin;
    char buf[MAX_DRIZ_ERROR_LEN];
    int n;
    // convert coordinates to the input frame
    if (invert_pixmap(par, vout.x, vout.y, &xin, &yin)) {
        n = sprintf(buf, "failed to invert pixel map at position (%.2f, %.2f)",
                    vout.x, vout.y);
        if (n < 0) {
            strcpy(buf, "failed to invert pixel map");
        }
        driz_error_set_message(par->error, buf);
        return 1;
    }
    vin->x = xin;
    vin->y = yin;
    return 0;
}

/**
 * Set-up image scanner.
 *
 * This is a the main part of the computation of the bounding polygon in the
 * input frame. This function computes the bounding box of the input image,
 * maps it the ouput frame, intersects mapped input bounding box with the
 * bounding box of the output image. It then maps this intersection polygon
 * back to the input frame and then sets up the scanner structure to be used
 * by the resampling kernel functions to determine the horizontal scan limits
 * for a given input image row.
 *
 * @param[in] struct driz_param_t - drizzle parameters (bounding box is used).
 * @param[out] struct scanner *s - computed from the intersection of polygons.
 * @param[out] int *ymin - minimum y of a row in input image with pixels inside
 *                 the intersection polygon
 * @param[out] int *ymax - maximum y of a row in input image with pixels inside
 *                 the intersection polygon
 * @return see init_scanner for return values.
 *
 */
int
init_image_scanner(struct driz_param_t *par, struct scanner *s, int *ymin,
                   int *ymax) {
    struct polygon p, q, pq, inpq;
    double xyin[2], xyout[2];
    integer_t isize[2], osize[2];
    int ipoint;
    int k, n;
    npy_intp *ndim;

    // define a polygon bounding the input image:
    inpq.npv = 4;
    inpq.v[0].x = par->xmin - 0.5;
    inpq.v[0].y = par->ymin - 0.5;
    inpq.v[1].x = par->xmax + 0.5;
    inpq.v[1].y = inpq.v[0].y;
    inpq.v[2].x = inpq.v[1].x;
    inpq.v[2].y = par->ymax + 0.5;
    inpq.v[3].x = inpq.v[0].x;
    inpq.v[3].y = inpq.v[2].y;

    // convert coordinates of the above polygon to the output frame and
    // define a polygon bounding the input image in the output frame:
    // inpq will be updated/overwritten later if coordinate mapping, inversion,
    // and polygon intersection is successful.
    for (k = 0; k < inpq.npv; ++k) {
        if (map_vertex_to_output(par, inpq.v[k], p.v + k)) {
            s->overlap_valid = 0;
            driz_error_set_message(par->error,
                                   "error computing input image bounding box");
            goto _setup_scanner;
        }
    }
    p.npv = inpq.npv;

    // define a polygon bounding the output image:
    ndim = PyArray_DIMS(par->output_data);
    q.npv = 4;
    q.v[0].x = -0.5;
    q.v[0].y = -0.5;
    q.v[1].x = (double)ndim[1] - 0.5;
    q.v[1].y = -0.5;
    q.v[2].x = (double)ndim[1] - 0.5;
    q.v[2].y = (double)ndim[0] - 0.5;
    q.v[3].x = -0.5;
    q.v[3].y = (double)ndim[0] - 0.5;

    // compute intersection of P and Q (in the output frame):
    if (clip_polygon_to_window(&p, &q, &pq)) {
        s->overlap_valid = 0;
        goto _setup_scanner;
    }

    // convert coordinates of vertices of the intersection polygon
    // back to input image coordinate system:
    for (k = 0; k < pq.npv; k++) {
        if (map_vertex_to_input(par, pq.v[k], &inpq.v[k])) {
            s->overlap_valid = 0;
            goto _setup_scanner;
        }
    }
    inpq.npv = pq.npv;

    s->overlap_valid = 1;
    orient_ccw(&inpq);

_setup_scanner:

    // initialize polygon scanner:
    driz_error_unset(par->error);
    n = init_scanner(&inpq, par, s);
    *ymin = MAX(0, (int)(s->min_y + 0.5 + 2.0 * MAX_INV_ERR));
    *ymax = MIN(s->ymax, (int)(s->max_y + 2.0 * MAX_INV_ERR));
    return n;
}
