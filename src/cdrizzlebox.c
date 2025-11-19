#define NO_IMPORT_ARRAY

#include <assert.h>
#define _USE_MATH_DEFINES /* needed for MS Windows to define M_PI */
#include <math.h>
#include <stdlib.h>

#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_21_API_VERSION
#endif

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cdrizzle_box_api

#include "driz_portability.h"
#include "cdrizzlemap.h"
#include "cdrizzlebox.h"
#include "cdrizzleutil.h"

static const double lut_delta = 0.003; /* spacing of Lanczos LUT */

/** ---------------------------------------------------------------------------
 * Update the flux and counts in the output image using a weighted average
 *
 * p:   structure containing options, input, and output
 * ii:  x coordinate in output images
 * jj:  y coordinate in output images
 * d:   new contribution to weighted flux
 * dow: new contribution to weighted counts
 */

inline_macro static int
update_data(
    struct driz_param_t *p, const integer_t ii, const integer_t jj, const float d, const float dow)
{
    double vc_plus_dow;
    float vc;

    if (dow == 0.0f) {
        return 0;
    }

    // get previous output image weight:
    vc = get_pixel(p->output_counts, ii, jj);

    vc_plus_dow = vc + dow;

    if (vc == 0.0f) {
        if (oob_pixel(p->output_data, ii, jj)) {
            driz_error_format_message(p->error, "OOB in output_data[%d,%d]", ii, jj);
            return 1;
        } else {
            set_pixel(p->output_data, ii, jj, d);
        }

    } else {
        if (oob_pixel(p->output_data, ii, jj)) {
            driz_error_format_message(p->error, "OOB in output_data[%d,%d]", ii, jj);
            return 1;
        } else {
            double value;
            value = (get_pixel(p->output_data, ii, jj) * vc + dow * d) / (vc_plus_dow);
            set_pixel(p->output_data, ii, jj, value);
        }
    }

    if (oob_pixel(p->output_counts, ii, jj)) {
        driz_error_format_message(p->error, "OOB in output_counts[%d,%d]", ii, jj);
        return 1;
    } else {
        set_pixel(p->output_counts, ii, jj, vc_plus_dow);
    }

    return 0;
}

/** ---------------------------------------------------------------------------
 * Update the flux, variance, and counts in the output image using a weighted
 * average.
 *
 * p:   structure containing options, input, and output
 * ii:  x coordinate in output images
 * jj:  y coordinate in output images
 * d:   new contribution to weighted flux
 * vc:  previous value of counts
 * dow: new contribution to weighted counts
 * d2:  array of data2 values that need to be co-added using squared weights
 *      (i.e., variance arrays)
 */
inline_macro static int
update_data_var(
    struct driz_param_t *p, const integer_t ii, const integer_t jj, const float d, const float dow,
    float *d2, unsigned int dq)
{
    double vc_plus_dow, vc_plus_dow2;
    double v, vc2, dow2;
    float vc;
    int i;
    int output_dq_value;
    PyArrayObject **arr2;

    if (oob_output_pixel(p, ii, jj)) {
        driz_error_format_message(p->error, "OOB in accessing output data [%d,%d]", ii, jj);
        return 1;
    }

    if (dow == 0.0f) {
        return 0;
    }

    // get previous output image weight:
    vc = get_pixel(p->output_counts, ii, jj);

    // new output image weight:
    vc_plus_dow = vc + dow;

    if (vc == 0.0f) {
        set_pixel(p->output_data, ii, jj, d);
        if (d2 && (arr2 = p->output_data2)) {
            for (i = 0; i < p->ndata2; ++i) {
                set_pixel(arr2[i], ii, jj, d2[i]);
            }
        }
        if (p->dq) {
            set_uint_pixel(p->output_dq, ii, jj, dq);
        }
    } else {
        v = (get_pixel(p->output_data, ii, jj) * vc + dow * d) / vc_plus_dow;
        set_pixel(p->output_data, ii, jj, v);
        if (d2 && (arr2 = p->output_data2)) {
            dow2 = dow * dow;
            vc2 = vc * vc;
            vc_plus_dow2 = vc_plus_dow * vc_plus_dow;
            for (i = 0; i < p->ndata2; ++i) {
                v = (get_pixel(arr2[i], ii, jj) * vc2 + dow2 * d2[i]) / vc_plus_dow2;
                set_pixel(arr2[i], ii, jj, v);
            }
        }
        if (p->dq) {
            output_dq_value = get_uint_pixel(p->output_dq, ii, jj);
            set_uint_pixel(p->output_dq, ii, jj, output_dq_value | dq);
        }
    }
    set_pixel(p->output_counts, ii, jj, vc_plus_dow);

    return 0;
}

/** ---------------------------------------------------------------------------
 * The bit value, trimmed to the appropriate range
 *
 * uuid: the id of the input image
 */

integer_t
compute_bit_value(integer_t uuid)
{
    integer_t bv;
    int np, bit_no;

    np = (uuid - 1) / 32 + 1;
    bit_no = (uuid - 1 - (32 * (np - 1)));
    bv = (integer_t) (1 << bit_no);

    return bv;
}

/*
To calculate area under a line segment within unit square at origin.
This is used by BOXER.
*/

static inline_macro double
sgarea(const double x1, const double y1, const double x2, const double y2)
{
    double xlo, xhi, ylo, yhi, xtop;
    double dx, dy, det, sgn_dx;

    dx = x2 - x1;
    dy = y2 - y1;

    /* Trap vertical line */
    if (dx == 0) {
        return 0.0;
    }

    if (dx < 0) {
        sgn_dx = -1.0;
        xlo = x2;
        xhi = x1;
    } else {
        sgn_dx = 1.0;
        xlo = x1;
        xhi = x2;
    }

    /* And determine the bounds ignoring y for now */
    if (xlo >= 1.0 || xhi <= 0.0) {
        return 0.0;
    }

    xlo = MAX(xlo, 0.0);
    xhi = MIN(xhi, 1.0);

    /* Now look at y */
    double slope = dy / dx;
    ylo = y1 + slope * (xlo - x1);
    yhi = y1 + slope * (xhi - x1);

    /* Alternative code that may be more stable under certain circumstances */
    /*
        if (xlo < 0.0) {
            xlo = 0.0;
            ylo = y1 + (dy / dx) * (xlo - x1);
        } else {
            ylo = (sgn_dx > 0.0) ? y1 : y2;
        }

        if (xhi > 1.0) {
            xhi = 1.0;
            yhi = y1 + (dy / dx) * (xhi - x1);
        } else {
            yhi = (sgn_dx > 0.0) ? y2 : y1;
        }
    */

    /* Trap segment entirely below axis */
    if (ylo <= 0.0 && yhi <= 0.0) {
        return 0.0;
    }

    /* There are four possibilities: both y below 1, both y above 1 and
       one of each. */
    if (ylo >= 1.0 && yhi >= 1.0) {
        /* Line segment is entirely above square */
        return sgn_dx * (xhi - xlo);
    }

    det = x1 * y2 - y1 * x2;

    /* Adjust bounds if segment crosses axis (to exclude anything below
       axis) */
    if (ylo < 0.0) {
        ylo = 0.0;
        xlo = det / dy;
    }

    if (yhi < 0.0) {
        yhi = 0.0;
        xhi = det / dy;
    }

    if (ylo <= 1.0) {
        if (yhi <= 1.0) {
            /* Segment is entirely within the square.
               The case of zero slope will end up here. */
            return sgn_dx * 0.5 * (xhi - xlo) * (yhi + ylo);
        }

        /* Otherwise, it must cross the top of the square */
        xtop = (dx + det) / dy;
        return sgn_dx * (0.5 * (xtop - xlo) * (1.0 + ylo) + xhi - xtop);
    }

    xtop = (dx + det) / dy;
    return sgn_dx * (0.5 * (xhi - xtop) * (1.0 + yhi) + xtop - xlo);
}

/**
 compute area of box overlap

 Calculate the area common to input clockwise polygon x(n), y(n) with
 square (is, js) to (is+1, js+1).
 This version is for a quadrilateral.

 Used by do_square_kernel.
*/

double
boxer(double is, double js, const double x[4], const double y[4])
{
    integer_t i;
    double sum;
    double px[4], py[4];

    assert(x);
    assert(y);

    is -= 0.5;
    js -= 0.5;
    /* Set up coords relative to unit square at origin Note that the
       +0.5s were added when this code was included in DRIZZLE */

    for (i = 0; i < 4; ++i) {
        px[i] = x[i] - is;
        py[i] = y[i] - js;
    }

    /* For each line in the polygon (or at this stage, input
       quadrilateral) calculate the area common to the unit square
       (allow negative area for subsequent `vector' addition of
       subareas). */
    sum = 0.0;
    for (i = 0; i < 4; ++i) {
        sum += sgarea(px[i], py[i], px[(i + 1) & 0x3], py[(i + 1) & 0x3]);
    }

    return sum;
}

/** ---------------------------------------------------------------------------
 * Compute area of box overlap. Calculate the area common to input clockwise
 * polygon x(n), y(n) with square (is, js) to (is+1, js+1). This version is for
 * a quadrilateral. Used by do_square_kernel.
 *
 * is: x coordinate of a pixel on the output image
 * js: y coordinate of a pixel on the output image
 * x:  x coordinates of endpoints of quadrilateral containing flux of input
 * pixel y:  y coordinates of endpoints of quadrilateral containing flux of
 * input pixel
 */

double
compute_area(double is, double js, const double x[4], const double y[4])
{
    int ipoint, jpoint, idim, jdim, iside, outside, count;
    int positive[2];
    double area, width;
    double midpoint[2], delta[2];
    double border[2][2], segment[2][2];

    /* The area for a qadrilateral clipped to a square of unit length whose
     * sides are aligned with the axes. The area is computed by computing the
     * area under each line segment clipped to the boundary of three sides of
     * the sqaure. Since the computed width is positive for two of the sides and
     * negative for the other two, we subtract the area outside the
     * quadrilateral without any extra code.
     */
    area = 0.0;

    border[0][0] = is - 0.5;
    border[0][1] = js - 0.5;
    border[1][0] = is + 0.5;
    border[1][1] = js + 0.5;

    for (ipoint = 0; ipoint < 4; ++ipoint) {
        jpoint = (ipoint + 1) & 03; /* Next point in cyclical order */

        segment[0][0] = x[ipoint];
        segment[0][1] = y[ipoint];
        segment[1][0] = x[jpoint];
        segment[1][1] = y[jpoint];

        /* Compute the endpoints of the line segment that
         * lie inside the border (possibly the whole segment)
         */

        for (idim = 0, count = 3; idim < 2; ++idim) {
            for (iside = 0; iside < 2; ++iside, --count) {
                delta[0] = segment[0][idim] - border[iside][idim];
                delta[1] = segment[1][idim] - border[iside][idim];

                positive[0] = delta[0] > 0.0;
                positive[1] = delta[1] > 0.0;

                /* If both deltas have the same signe there is no baundary
                 * crossing
                 */
                if (positive[0] == positive[1]) {
                    /* A diagram will convince that you decide a point is
                     * inside or outside the boundary by the following test
                     */
                    if (positive[0] == iside) {
                        /* Segment is entirely outside the boundary */
                        if (count == 0) {
                            /* Implicitly multiplied by 1.0, the square height
                             */
                            width = segment[1][0] - segment[0][0];
                            area += width;
                        } else {
                            goto _nextsegment;
                        }

                    } else {
                        /* Segment entirely within the boundary */
                        if (count == 0) {
                            /* Use the trapezoid formula to compute the area
                             * under the segment. Delta is the distance to the
                             * top of the square and is negative or zero for the
                             * segment inside the square
                             */
                            width = segment[1][0] - segment[0][0];
                            area += 0.5 * width * (2.0 + delta[0] + delta[1]);
                        }
                    }

                } else {
                    /* If ends of the line segment are on opposite sides of the
                     * boundary, calculate midpoint, the point of intersection
                     */
                    outside = positive[iside];
                    jdim = (idim + 1) & 01; /* the other dimension */

                    midpoint[idim] = border[iside][idim];

                    midpoint[jdim] = (delta[1] * segment[0][jdim] - delta[0] * segment[1][jdim]) /
                                     (delta[1] - delta[0]);

                    if (count == 0) {
                        /* If a segment cross the boundary the formula for its
                         * area is a combination of the formulas for segments
                         * entirely inside and outside.
                         */
                        if (outside == 0) {
                            width = midpoint[0] - segment[0][0];
                            area += width;
                            width = segment[1][0] - midpoint[0];
                            /* Delta[0] is at the crossing point and thus zero
                             */
                            area += 0.5 * width * (2.0 + delta[1]);
                        } else {
                            width = segment[1][0] - midpoint[0];
                            area += width;
                            width = midpoint[0] - segment[0][0];
                            /* Delta[1] is at the crossing point and thus zero
                             */
                            area += 0.5 * width * (2.0 + delta[0]);
                        }

                    } else {
                        /* Clip segment against each boundary except the last */
                        segment[outside][0] = midpoint[0];
                        segment[outside][1] = midpoint[1];
                    }
                }
            }
        }

    _nextsegment:
        continue;
    }

    return fabs(area);
}

/** ---------------------------------------------------------------------------
 * Calculate overlap between an arbitrary rectangle, aligned with the axes, and
 * a pixel. This is a simplified version of the compute_area, only valid if axes
 * are nearly aligned. Used by do_kernel_turbo.
 *
 * i:    the x coordinate of a pixel on the output image
 * j:    the y coordinate of a pixel on the output image
 * xmin: the x coordinate of the lower edge of rectangle containing flux of
 * input pixel xmax: the x coordinate of the upper edge of rectangle containing
 * flux of input pixel ymin: the y coordinate of the lower edge of rectangle
 * containing flux of input pixel ymax: the y coordinate of the upper edge of
 * rectangle containing flux of input pixel
 */

static inline_macro double
over(
    const integer_t i, const integer_t j, const double xmin, const double xmax, const double ymin,
    const double ymax)
{
    double dx, dy;

    assert(xmin <= xmax);
    assert(ymin <= ymax);

    dx = MIN(xmax, (double) (i) + 0.5) - MAX(xmin, (double) (i) -0.5);
    dy = MIN(ymax, (double) (j) + 0.5) - MAX(ymin, (double) (j) -0.5);

    if (dx > 0.0 && dy > 0.0) {
        return dx * dy;
    }

    return 0.0;
}

/**
 * @brief Computes the pixel scale ratio (pscale_ratio) at the centroid of a
 * bounding polygon.
 *
 * This function estimates the pixel scale ratio (pscale_ratio) for a given
 * region defined by a bounding polygon within a pixmap. The scale is computed
 * at the centroid of the polygon by mapping the centroid to pixel coordinates,
 * then calculating the local transformation matrix determinant using
 * neighboring pixels.
 *
 * TODO: in the future, we could estimate pscale_ratio for each pixel - a
 * varying pscale_ratio across the input image that is more accurate for
 * distorted input images although this will have significant cost penalty.
 *
 * @param p Pointer to the driz_param_t structure containing pixmap and error
 * information.
 * @param bounding_polygon Pointer to the polygon structure representing the
 * region of interest.
 * @param pscale_ratio Pointer to a float where the computed pixel scale ratio
 * will be stored.
 *
 * @return 0 on success, 1 on failure. On failure, an error message is set in
 * p->error.
 */
int
compute_pscale_ratio(struct driz_param_t *p, struct polygon *bounding_polygon, float *pscale_ratio)
{
    integer_t i, j, nx, ny, mapsize[2];
    double cx, cy;
    double ox, oy, ox1, oy1, ox2, oy2;
    double cd11, cd12, cd21, cd22;

    if (polygon_centroid(bounding_polygon, &cx, &cy)) {
        goto _error;
    }
    // estimate pixel scale ratio:
    i = (integer_t) cx;
    j = (integer_t) cy;

    get_dimensions(p->pixmap, mapsize);
    nx = mapsize[0];
    ny = mapsize[1];

    if (nx < 2 || ny < 2) {
        goto _error;
    }
    i = MAX(MIN(i, nx - 2), 0);
    j = MAX(MIN(j, ny - 2), 0);

    if (map_pixel(p->pixmap, i, j, &ox, &oy) || map_pixel(p->pixmap, i + 1, j, &ox1, &oy1) ||
        map_pixel(p->pixmap, i, j + 1, &ox2, &oy2)) {
        goto _error;
    }

    cd11 = ox1 - ox;
    cd12 = oy1 - oy;
    cd21 = ox2 - ox;
    cd22 = oy2 - oy;
    *pscale_ratio = (float) sqrt(fabs(cd11 * cd22 - cd12 * cd21));

    return 0;

_error:
    driz_error_set_message(p->error, "Unable to estimate pscale_ratio");
    return 1;
}

/** ---------------------------------------------------------------------------
 * The kernel assumes all the flux in an input pixel is at the center
 *
 * p: structure containing options, input, and output
 */

static int
do_kernel_point_var(struct driz_param_t *p)
{
    struct scanner s;
    integer_t i, j, ii, jj, k;
    integer_t osize[2];
    float d, dow, *d2 = NULL, iscale2 = 1.0f;
    integer_t bv;
    int xmin, xmax, ymin, ymax, n, ndata2;
    unsigned int dqval = 0;

    ndata2 = p->ndata2;

    bv = compute_bit_value(p->uuid);

    if (init_image_scanner(p, &s, &ymin, &ymax)) {
        return 1;
    }

    p->nskip = (p->ymax - p->ymin) - (ymax - ymin);
    p->nmiss = p->nskip * (p->xmax - p->xmin);

    /* This is the outer loop over all the lines in the input image */
    get_dimensions(p->output_data, osize);

    if (ndata2 > 0) {
        iscale2 = p->iscale * p->iscale;

        if (!(d2 = (float *) malloc(p->ndata2 * sizeof(float)))) {
            driz_error_set(p->error, PyExc_MemoryError, "Memory allocation failed.");
            return 1;
        }
        if (!p->output_data2) {
            driz_error_set(
                p->error, PyExc_RuntimeError,
                "'output_data2' must be a valid pointer when "
                "'data2' is valid.");
            free(d2);
            return 1;
        }
        for (i = 0; i < ndata2; ++i) {
            if (!p->output_data2[i]) {
                driz_error_set(
                    p->error, PyExc_RuntimeError,
                    "Some arrays in 'output_data2' have invalid pointers.");
                free(d2);
                return 1;
            }
        }
    }

    for (j = ymin; j <= ymax; ++j) {
        /* Check the overlap with the output */
        n = get_scanline_limits(&s, j, &xmin, &xmax);
        if (n == 1) {
            // scan ended (y reached the top vertex/edge)
            p->nskip += (ymax + 1 - j);
            p->nmiss += (ymax + 1 - j) * (p->xmax - p->xmin);
            break;
        } else if (n == 2 || n == 3) {
            // pixel centered on y is outside of scanner's limits or image
            // [0, height - 1] OR: limits (x1, x2) are equal (line width is
            // 0)
            p->nmiss += (p->xmax - p->xmin);
            ++p->nskip;
            continue;
        } else {
            // limits (x1, x2) are equal (line width is 0)
            p->nmiss += (p->xmax - p->xmin) - (xmax + 1 - xmin);
        }

        for (i = xmin; i <= xmax; ++i) {
            double ox, oy;

            if (map_pixel(p->pixmap, i, j, &ox, &oy)) {
                ++p->nmiss;

            } else {
                ii = nintd(ox);
                jj = nintd(oy);

                /* Check it is on the output image */
                if (ii < 0 || ii >= osize[0] || jj < 0 || jj >= osize[1]) {
                    ++p->nmiss;

                } else {
                    /* Allow for stretching because of scale change */
                    d = get_pixel(p->data, i, j) * p->iscale;
                    for (k = 0; k < ndata2; ++k) {
                        if (p->data2[k]) {
                            d2[k] = get_pixel(p->data2[k], i, j) * iscale2;
                        } else {
                            d2[k] = 0.0f;
                        }
                    }

                    /* Scale the weighting mask by the scale factor.  Note
                       that we DON'T scale by the Jacobian as it hasn't been
                       calculated */
                    if (p->weights) {
                        dow = get_pixel(p->weights, i, j) * p->weight_scale;
                    } else {
                        dow = 1.0f;
                    }

                    if (p->dq) {
                        dqval = (int) get_uint_pixel(p->dq, i, j);
                    }

                    /* If we are creating or modifying the context image,
                       we do so here. */
                    if (p->output_context && dow > 0.0f) {
                        set_bit(p->output_context, ii, jj, bv);
                    }

                    if (update_data_var(p, ii, jj, d, dow, d2, dqval)) {
                        free(d2);
                        return 1;
                    }
                }
            }
        }
    }

    free(d2);
    return 0;
}

/** ---------------------------------------------------------------------------
 * This kernel assumes the flux is distributed acrass a gaussian around the
 * center of an input pixel
 *
 * p: structure containing options, input, and output
 */

static int
do_kernel_gaussian_var(struct driz_param_t *p)
{
    struct scanner s;
    integer_t bv, i, j, ii, jj, k, nxi, nxa, nyi, nya, nhit;
    integer_t osize[2];
    float d, dow, *d2 = NULL, iscale2 = 1.0f;
    double gaussian_efac, gaussian_es;
    double pfo, ac, w, ddx, ddy, r2, dover, kscale2;
    const double nsig = 2.5;
    int xmin, xmax, ymin, ymax, n, ndata2;
    unsigned int dqval = 0;

    ndata2 = p->ndata2;

    /* Added in V2.9 - make sure pfo doesn't get less than 1.2
       divided by the scale so that there are never holes in the
       output */

    if (init_image_scanner(p, &s, &ymin, &ymax)) {
        return 1;
    }

    /* if pscale_ratio is not defined, estimate it near the center of the
       intersection polygon.
     */
    if (!isfinite(p->pscale_ratio) &&
        compute_pscale_ratio(p, &s.bounding_polygon, &p->pscale_ratio)) {
        return 1;
    }

    kscale2 = (double) p->pscale_ratio * (double) p->pscale_ratio;

    pfo = nsig * p->pixel_fraction / 2.3548 / (double) p->pscale_ratio;
    pfo = CLAMP_ABOVE(pfo, 1.2 / (double) p->pscale_ratio);

    ac = 1.0 / (p->pixel_fraction * p->pixel_fraction);
    bv = compute_bit_value(p->uuid);

    gaussian_efac = (2.3548 * 2.3548) * kscale2 * ac / 2.0;
    gaussian_es = gaussian_efac / M_PI;

    p->nskip = (p->ymax - p->ymin) - (ymax - ymin);
    p->nmiss = p->nskip * (p->xmax - p->xmin);

    /* This is the outer loop over all the lines in the input image */

    get_dimensions(p->output_data, osize);

    if (ndata2 > 0) {
        iscale2 = p->iscale * p->iscale;
        if (!(d2 = (float *) malloc(p->ndata2 * sizeof(float)))) {
            driz_error_set(p->error, PyExc_MemoryError, "Memory allocation failed.");
            return 1;
        }
        if (!p->output_data2) {
            driz_error_set(
                p->error, PyExc_RuntimeError,
                "'output_data2' must be a valid pointer when "
                "'data2' is valid.");
            free(d2);
            return 1;
        }
        for (i = 0; i < ndata2; ++i) {
            if (!p->output_data2[i]) {
                driz_error_set(
                    p->error, PyExc_RuntimeError,
                    "Some arrays in 'output_data2' have invalid pointers.");
                free(d2);
                return 1;
            }
        }
    }

    for (j = ymin; j <= ymax; ++j) {
        /* Check the overlap with the output */
        n = get_scanline_limits(&s, j, &xmin, &xmax);
        if (n == 1) {
            // scan ended (y reached the top vertex/edge)
            p->nskip += (ymax + 1 - j);
            p->nmiss += (ymax + 1 - j) * (p->xmax - p->xmin);
            break;
        } else if (n == 2 || n == 3) {
            // pixel centered on y is outside of scanner's limits or image
            // [0, height - 1] OR: limits (x1, x2) are equal (line width is
            // 0)
            p->nmiss += (p->xmax - p->xmin);
            ++p->nskip;
            continue;
        } else {
            // limits (x1, x2) are equal (line width is 0)
            p->nmiss += (p->xmax - p->xmin) - (xmax + 1 - xmin);
        }

        for (i = xmin; i <= xmax; ++i) {
            double ox, oy;

            if (map_pixel(p->pixmap, i, j, &ox, &oy)) {
                nhit = 0;

            } else {
                /* Offset within the subset */
                nxi = MAX(nintd(ox - pfo), 0);
                nxa = MIN(nintd(ox + pfo), osize[0] - 1);
                nyi = MAX(nintd(oy - pfo), 0);
                nya = MIN(nintd(oy + pfo), osize[1] - 1);

                nhit = 0;

                /* Allow for stretching because of scale change */
                d = get_pixel(p->data, i, j) * p->iscale;
                for (k = 0; k < ndata2; ++k) {
                    if (p->data2[k]) {
                        d2[k] = get_pixel(p->data2[k], i, j) * iscale2;
                    } else {
                        d2[k] = 0.0f;
                    }
                }

                /* Scale the weighting mask by the scale factor and
                   inversely by the Jacobian to ensure conservation of
                   weight in the output
                 */
                if (p->weights) {
                    w = get_pixel(p->weights, i, j) * p->weight_scale;
                } else {
                    w = 1.0;
                }

                /* Loop over output pixels which could be affected */
                for (jj = nyi; jj <= nya; ++jj) {
                    ddy = oy - (double) jj;
                    for (ii = nxi; ii <= nxa; ++ii) {
                        ddx = ox - (double) ii;
                        /* Radial distance */
                        r2 = ddx * ddx + ddy * ddy;

                        /* Weight is a scaled Gaussian function of radial
                           distance */
                        dover = gaussian_es * exp(-r2 * gaussian_efac);

                        /* Count the hits */
                        ++nhit;

                        dow = (float) dover * w;

                        /* If we are creating or modifying the context
                           image, we do so here. */
                        if (p->output_context && dow > 0.0f) {
                            set_bit(p->output_context, ii, jj, bv);
                        }

                        if (p->dq) {
                            dqval = get_uint_pixel(p->dq, i, j);
                        }

                        if (update_data_var(p, ii, jj, d, dow, d2, dqval)) {
                            free(d2);
                            return 1;
                        }
                    }
                }
            }

            /* Count cases where the pixel is off the output image */
            if (nhit == 0) {
                ++p->nmiss;
            }
        }
    }

    free(d2);
    return 0;
}

/** ---------------------------------------------------------------------------
 * This kernel assumes flux of input pixel is distributed according to
 * lanczos function
 *
 * p: structure containing options, input, and output
 */

static int
do_kernel_lanczos_var(struct driz_param_t *p)
{
    struct scanner s;
    integer_t bv, i, j, ii, jj, k, nxi, nxa, nyi, nya, nhit;
    integer_t osize[2];
    float d, dow, *d2 = NULL, iscale2 = 1.0f;
    double pfo, xx, yy, w, dover;
    int kernel_order;
    size_t nlut;
    size_t ix, iy;
    double sdp;
    double *lut = NULL;
    int xmin, xmax, ymin, ymax, n, ndata2;
    unsigned int dqval = 0;

    if (fabs(p->pixel_fraction - 1.0) > 1.0e-5) {
        py_warning(
            NULL, "In lanczos kernel, pixel_fraction is ignored and "
                  "assumed to be 1.0");
    }

    ndata2 = p->ndata2;

    kernel_order = (p->kernel == kernel_lanczos2) ? 2 : 3;

    bv = compute_bit_value(p->uuid);

    /* Set up a look-up-table for Lanczos-style interpolation
       kernels */
    nlut = (size_t) ceil(kernel_order / lut_delta) + 1;
    if ((lut = malloc(nlut * sizeof(double))) == NULL) {
        driz_error_set_message(p->error, "Out of memory");
        return driz_error_is_set(p->error);
    }
    create_lanczos_lut(kernel_order, nlut, lut_delta, lut);

    if (init_image_scanner(p, &s, &ymin, &ymax)) {
        return 1;
    }

    /* if pscale_ratio is not defined, estimate it near the center of the
       intersection polygon.
     */
    if (!isfinite(p->pscale_ratio) &&
        compute_pscale_ratio(p, &s.bounding_polygon, &p->pscale_ratio)) {
        return 1;
    }

    pfo = (double) kernel_order / (double) p->pscale_ratio;
    sdp = p->pscale_ratio / lut_delta;

    p->nskip = (p->ymax - p->ymin) - (ymax - ymin);
    p->nmiss = p->nskip * (p->xmax - p->xmin);

    /* This is the outer loop over all the lines in the input image */

    get_dimensions(p->output_data, osize);

    if (ndata2 > 0) {
        iscale2 = p->iscale * p->iscale;

        if (!(d2 = (float *) malloc(p->ndata2 * sizeof(float)))) {
            driz_error_set(p->error, PyExc_MemoryError, "Memory allocation failed.");
            free(lut);
            return 1;
        }
        if (!p->output_data2) {
            driz_error_set(
                p->error, PyExc_RuntimeError,
                "'output_data2' must be a valid pointer when "
                "'data2' is valid.");
            free(lut);
            free(d2);
            return 1;
        }
        for (i = 0; i < ndata2; ++i) {
            if (!p->output_data2[i]) {
                driz_error_set(
                    p->error, PyExc_RuntimeError,
                    "Some arrays in 'output_data2' have invalid pointers.");
                free(lut);
                free(d2);
                return 1;
            }
        }
    }

    for (j = ymin; j <= ymax; ++j) {
        /* Check the overlap with the output */
        n = get_scanline_limits(&s, j, &xmin, &xmax);
        if (n == 1) {
            // scan ended (y reached the top vertex/edge)
            p->nskip += (ymax + 1 - j);
            p->nmiss += (ymax + 1 - j) * (p->xmax - p->xmin);
            break;
        } else if (n == 2 || n == 3) {
            // pixel centered on y is outside of scanner's limits or image
            // [0, height - 1] OR: limits (x1, x2) are equal (line width is
            // 0)
            p->nmiss += (p->xmax - p->xmin);
            ++p->nskip;
            continue;
        } else {
            // limits (x1, x2) are equal (line width is 0)
            p->nmiss += (p->xmax - p->xmin) - (xmax + 1 - xmin);
        }

        for (i = xmin; i <= xmax; ++i) {
            if (map_pixel(p->pixmap, i, j, &xx, &yy)) {
                nhit = 0;

            } else {
                nxi = MAX((integer_t) floor(xx - pfo) + 1, 0);
                nxa = MIN((integer_t) floor(xx + pfo), osize[0] - 1);
                nyi = MAX((integer_t) floor(yy - pfo) + 1, 0);
                nya = MIN((integer_t) floor(yy + pfo), osize[1] - 1);

                nhit = 0;

                /* Allow for stretching because of scale change */
                d = get_pixel(p->data, i, j) * p->iscale;
                for (k = 0; k < ndata2; ++k) {
                    if (p->data2[k]) {
                        d2[k] = get_pixel(p->data2[k], i, j) * iscale2;
                    } else {
                        d2[k] = 0.0f;
                    }
                }

                /* Scale the weighting mask by the scale factor and
                   inversely by the Jacobian to ensure conservation of
                   weight in the output
                 */
                if (p->weights) {
                    w = get_pixel(p->weights, i, j) * p->weight_scale;
                } else {
                    w = 1.0;
                }

                /* Loop over output pixels which could be affected */
                for (jj = nyi; jj <= nya; ++jj) {
                    for (ii = nxi; ii <= nxa; ++ii) {
                        /* X and Y offsets */
                        ix = nintd(fabs((xx - (double) ii) * sdp));
                        iy = nintd(fabs((yy - (double) jj) * sdp));
                        if (ix >= nlut || iy >= nlut) {
                            continue;
                        }

                        /* Count the hits */
                        ++nhit;

                        /* Weight is product of Lanczos function values in X
                         * and
                         * Y */
                        dover = lut[ix] * lut[iy];

                        dow = (float) (dover * w);

                        /* If we are creating or modifying the context
                           image, we do so here. */
                        if (p->output_context && dow > 0.0f) {
                            set_bit(p->output_context, ii, jj, bv);
                        }

                        if (p->dq) {
                            dqval = get_uint_pixel(p->dq, i, j);
                        }

                        if (update_data_var(p, ii, jj, d, dow, d2, dqval)) {
                            free(d2);
                            free(lut);
                            return 1;
                        }
                    }
                }
            }

            /* Count cases where the pixel is off the output image */
            if (nhit == 0) {
                ++p->nmiss;
            }
        }
    }

    free(d2);
    free(lut);

    return 0;
}

/** ---------------------------------------------------------------------------
 * This kernel assumes the input flux is evenly distributed over a rectangle
 * whose sides are aligned with the ouput pixel. Called turbo because it is
 * fast, but approximate.
 *
 * p: structure containing options, input, and output
 */

static int
do_kernel_turbo_var(struct driz_param_t *p)
{
    struct scanner s;
    integer_t bv, i, j, ii, jj, k, nhit, iis, iie, jjs, jje;
    integer_t osize[2];
    float d, dow, *d2 = NULL, iscale2 = 1.0f;
    double pfo, dover_scale, ac;
    double xxi, xxa, yyi, yya, w, dover;
    int xmin, xmax, ymin, ymax, n, ndata2;
    unsigned int dqval = 0;

    ndata2 = p->ndata2;

    driz_log_message("starting do_kernel_turbo");
    bv = compute_bit_value(p->uuid);
    ac = 1.0 / (p->pixel_fraction * p->pixel_fraction);

    if (init_image_scanner(p, &s, &ymin, &ymax)) {
        return 1;
    }

    /* if pscale_ratio is not defined, estimate it near the center of the
       intersection polygon.
     */
    if (!isfinite(p->pscale_ratio) &&
        compute_pscale_ratio(p, &s.bounding_polygon, &p->pscale_ratio)) {
        return 1;
    }

    pfo = p->pixel_fraction / p->pscale_ratio / 2.0;
    dover_scale = ac * (double) p->pscale_ratio * (double) p->pscale_ratio;

    p->nskip = (p->ymax - p->ymin) - (ymax - ymin);
    p->nmiss = p->nskip * (p->xmax - p->xmin);

    /* This is the outer loop over all the lines in the input image */

    get_dimensions(p->output_data, osize);

    if (ndata2 > 0) {
        iscale2 = p->iscale * p->iscale;

        if (!(d2 = (float *) malloc(ndata2 * sizeof(float)))) {
            driz_error_set(p->error, PyExc_MemoryError, "Memory allocation failed.");
            return 1;
        }
        if (!p->output_data2) {
            driz_error_set(
                p->error, PyExc_RuntimeError,
                "'output_data2' must be a valid pointer when "
                "'data2' is valid.");
            return 1;
        }
        for (i = 0; i < ndata2; ++i) {
            if (!p->output_data2[i]) {
                driz_error_set(
                    p->error, PyExc_RuntimeError,
                    "Some arrays in 'output_data2' have invalid pointers.");
                return 1;
            }
        }
    }

    for (j = ymin; j <= ymax; ++j) {
        /* Check the overlap with the output */
        n = get_scanline_limits(&s, j, &xmin, &xmax);

        if (n == 1) {
            // scan ended (y reached the top vertex/edge)
            p->nskip += (ymax + 1 - j);
            p->nmiss += (ymax + 1 - j) * (p->xmax - p->xmin);
            break;
        } else if (n == 2 || n == 3) {
            // pixel centered on y is outside of scanner's limits or image
            // [0, height - 1] OR: limits (x1, x2) are equal (line width is
            // 0)
            p->nmiss += (p->xmax - p->xmin);
            ++p->nskip;
            continue;
        } else {
            // limits (x1, x2) are equal (line width is 0)
            p->nmiss += (p->xmax - p->xmin) - (xmax + 1 - xmin);
        }

        for (i = xmin; i <= xmax; ++i) {
            double ox, oy;

            if (map_pixel(p->pixmap, i, j, &ox, &oy)) {
                nhit = 0;

            } else {
                /* Offset within the subset */
                xxi = ox - pfo;
                xxa = ox + pfo;
                yyi = oy - pfo;
                yya = oy + pfo;

                iis = MAX(nintd(xxi), 0);
                iie = MIN(nintd(xxa), osize[0] - 1);
                jjs = MAX(nintd(yyi), 0);
                jje = MIN(nintd(yya), osize[1] - 1);

                nhit = 0;

                /* Allow for stretching because of scale change */
                d = get_pixel(p->data, i, j) * p->iscale;
                for (k = 0; k < ndata2; ++k) {
                    if (p->data2[k]) {
                        d2[k] = get_pixel(p->data2[k], i, j) * iscale2;
                    } else {
                        d2[k] = 0.0f;
                    }
                }

                /* Scale the weighting mask by the scale factor and
                   inversely by the Jacobian to ensure conservation of
                   weight in the output.
                 */
                if (p->weights) {
                    w = get_pixel(p->weights, i, j) * p->weight_scale;
                } else {
                    w = 1.0;
                }

                /* Loop over the output pixels which could be affected */
                for (jj = jjs; jj <= jje; ++jj) {
                    for (ii = iis; ii <= iie; ++ii) {
                        /* Calculate the overlap using the simpler "aligned"
                           box routine */
                        dover = over(ii, jj, xxi, xxa, yyi, yya);

                        if (dover > 0.0) {
                            /* Correct for the pixfrac area factor */
                            dover *= dover_scale;

                            /* Count the hits */
                            ++nhit;

                            dow = (float) (dover * w);

                            /* If we are creating or modifying the context
                               image, we do so here. */
                            if (p->output_context && dow > 0.0f) {
                                set_bit(p->output_context, ii, jj, bv);
                            }

                            if (p->dq) {
                                dqval = get_uint_pixel(p->dq, i, j);
                            }

                            if (update_data_var(p, ii, jj, d, dow, d2, dqval)) {
                                return 1;
                            }
                        }
                    }
                }
            }

            /* Count cases where the pixel is off the output image */
            if (nhit == 0) {
                ++p->nmiss;
            }
        }
    }

    driz_log_message("ending do_kernel_turbo");
    return 0;
}

/** ---------------------------------------------------------------------------
 * This module does the actual mapping of input flux to output images. It
 * works by calculating the positions of the four corners of a quadrilateral
 * on the output grid corresponding to the corners of the input pixel and
 * then working out exactly how much of each pixel in the output is covered,
 * or not.
 *
 * p: structure containing options, input, and output
 */

int
do_kernel_square_var(struct driz_param_t *p)
{
    integer_t bv, i, j, ii, jj, k, min_ii, max_ii, min_jj, max_jj, nhit;
    integer_t osize[2], mapsize[2];
    float d, dow, *d2 = NULL, iscale2 = 1.0f;
    double dh, jaco, dover, w;

    double xin[4], yin[4], xout[4], yout[4];
    int ndata2;
    unsigned int dqval = 0;

    ndata2 = p->ndata2;

    struct scanner s;
    int xmin, xmax, ymin, ymax, n;

    driz_log_message("starting do_kernel_square");
    dh = 0.5 * p->pixel_fraction;
    bv = compute_bit_value(p->uuid);

    /* Next the "classic" drizzle square kernel...  this is different
       because we have to transform all four corners of the shrunken
       pixel */
    if (init_image_scanner(p, &s, &ymin, &ymax)) {
        return 1;
    }

    p->nskip = (p->ymax - p->ymin) - (ymax - ymin);
    p->nmiss = p->nskip * (p->xmax - p->xmin);

    /* This is the outer loop over all the lines in the input image */
    get_dimensions(p->output_data, osize);
    get_dimensions(p->pixmap, mapsize);

    if (ndata2 > 0) {
        iscale2 = p->iscale * p->iscale;

        if (!(d2 = (float *) malloc(p->ndata2 * sizeof(float)))) {
            driz_error_set(p->error, PyExc_MemoryError, "Memory allocation failed.");
            return 1;
        }
        if (!p->output_data2) {
            driz_error_set(
                p->error, PyExc_RuntimeError,
                "'output_data2' must be a valid pointer when "
                "'data2' is valid.");
            free(d2);
            return 1;
        }
        for (i = 0; i < ndata2; ++i) {
            if (!p->output_data2[i]) {
                driz_error_set(
                    p->error, PyExc_RuntimeError,
                    "Some arrays in 'output_data2' have invalid pointers.");
                free(d2);
                return 1;
            }
        }
    }

    for (j = ymin; j <= ymax; ++j) {
        /* Check the overlap with the output */
        n = get_scanline_limits(&s, j, &xmin, &xmax);
        if (n == 1) {
            // scan ended (y reached the top vertex/edge)
            p->nskip += (ymax + 1 - j);
            p->nmiss += (ymax + 1 - j) * (p->xmax - p->xmin);
            break;
        } else if (n == 2 || n == 3) {
            // pixel centered on y is outside of scanner's limits or image
            // [0, height - 1] OR: limits (x1, x2) are equal (line width is
            // 0)
            p->nmiss += (p->xmax - p->xmin);
            ++p->nskip;
            continue;
        } else {
            // limits (x1, x2) are equal (line width is 0)
            p->nmiss += (p->xmax - p->xmin) - (xmax + 1 - xmin);
        }

        /* Set the input corner positions */
        yin[1] = yin[0] = (double) j + dh;
        yin[3] = yin[2] = (double) j - dh;

        for (i = xmin; i <= xmax; ++i) {
            nhit = 0;

            // xin[3] = xin[0] = (double)i - dh;
            // xin[2] = xin[1] = (double)i + dh;

            // for (ii = 0; ii < 4; ++ii) {
            //     if (interpolate_point(p, xin[ii], yin[ii], xout + ii,
            //                           yout + ii)) {
            //         goto _miss;
            //     }
            // }

            /* Assuming we don't need to extrapolate, call a more
             * efficient interpolator that takes advantage of the fact
             * that pixfrac<1 and that we are using a square grid.
             */
            if (i > 0 && i < mapsize[0] - 2 && j > 0 && j < mapsize[1] - 2) {
                if (interpolate_four_points(
                        p, i, j, dh, xout, xout + 1, xout + 2, xout + 3, yout, yout + 1, yout + 2,
                        yout + 3)) {
                    goto _miss;
                }
            } else {
                xin[3] = xin[0] = (double) i - dh;
                xin[2] = xin[1] = (double) i + dh;
                for (ii = 0; ii < 4; ++ii) {
                    if (interpolate_point(p, xin[ii], yin[ii], xout + ii, yout + ii)) {
                        goto _miss;
                    }
                }
            }

            /* Work out the area of the quadrilateral on the output
             * grid.  If the points are in clockwise order we get a
             * postive area.  If they are in anticlockwise order, jaco
             * will be negative, but so will the areas computed by
             * boxer, so it doesn't actually matter once we divide it
             * out.
             */

            jaco = 0.5f * ((xout[1] - xout[3]) * (yout[0] - yout[2]) -
                           (xout[0] - xout[2]) * (yout[1] - yout[3]));

            /* Allow for stretching because of scale change */
            d = get_pixel(p->data, i, j) * p->iscale;
            for (k = 0; k < ndata2; ++k) {
                if (p->data2[k]) {
                    d2[k] = get_pixel(p->data2[k], i, j) * iscale2;
                } else {
                    d2[k] = 0.0f;
                }
            }

            /* Scale the weighting mask by the scale factor and inversely by
               the Jacobian to ensure conservation of weight in the output
             */
            if (p->weights) {
                w = get_pixel(p->weights, i, j) * p->weight_scale / jaco;
            } else {
                w = 1.0 / jaco;
            }

            /* Loop over output pixels which could be affected */
            min_jj = MAX(nintd(min_doubles(yout, 4)), 0);
            max_jj = MIN(nintd(max_doubles(yout, 4)), osize[1] - 1);
            min_ii = MAX(nintd(min_doubles(xout, 4)), 0);
            max_ii = MIN(nintd(max_doubles(xout, 4)), osize[0] - 1);

            for (jj = min_jj; jj <= max_jj; ++jj) {
                for (ii = min_ii; ii <= max_ii; ++ii) {
                    /* Call boxer to calculate overlap */
                    // dover = compute_area((double)ii, (double)jj, xout,
                    // yout);
                    dover = boxer((double) ii, (double) jj, xout, yout);

                    /* Could be positive or negative, depending on the sign
                     * of jaco */
                    if (dover != 0.0) {
                        dow = (float) (dover * w);

                        /* Count the hits */
                        ++nhit;

                        /* If we are creating or modifying the context image
                           we do so here */
                        if (p->output_context && dow > 0.0f) {
                            set_bit(p->output_context, ii, jj, bv);
                        }

                        if (p->dq) {
                            dqval = get_uint_pixel(p->dq, i, j);
                        }

                        if (update_data_var(p, ii, jj, d, dow, d2, dqval)) {
                            free(d2);
                            return 1;
                        }
                    }
                }
            }

        /* Count cases where the pixel is off the output image */
        _miss:
            if (nhit == 0) {
                ++p->nmiss;
            }
        }
    }

    driz_log_message("ending do_kernel_square");
    free(d2);
    return 0;
}

/** ---------------------------------------------------------------------------
 * The kernel assumes all the flux in an input pixel is at the center
 *
 * p: structure containing options, input, and output
 */

static int
do_kernel_point(struct driz_param_t *p)
{
    struct scanner s;
    integer_t i, j, ii, jj;
    integer_t osize[2];
    float d, dow;
    integer_t bv;
    int xmin, xmax, ymin, ymax, n;

    bv = compute_bit_value(p->uuid);

    if (init_image_scanner(p, &s, &ymin, &ymax)) {
        return 1;
    }

    p->nskip = (p->ymax - p->ymin) - (ymax - ymin);
    p->nmiss = p->nskip * (p->xmax - p->xmin);

    /* This is the outer loop over all the lines in the input image */
    get_dimensions(p->output_data, osize);
    for (j = ymin; j <= ymax; ++j) {
        /* Check the overlap with the output */
        n = get_scanline_limits(&s, j, &xmin, &xmax);
        if (n == 1) {
            // scan ended (y reached the top vertex/edge)
            p->nskip += (ymax + 1 - j);
            p->nmiss += (ymax + 1 - j) * (p->xmax - p->xmin);
            break;
        } else if (n == 2 || n == 3) {
            // pixel centered on y is outside of scanner's limits or image
            // [0, height - 1] OR: limits (x1, x2) are equal (line width is
            // 0)
            p->nmiss += (p->xmax - p->xmin);
            ++p->nskip;
            continue;
        } else {
            // limits (x1, x2) are equal (line width is 0)
            p->nmiss += (p->xmax - p->xmin) - (xmax + 1 - xmin);
        }

        for (i = xmin; i <= xmax; ++i) {
            double ox, oy;

            if (map_pixel(p->pixmap, i, j, &ox, &oy)) {
                ++p->nmiss;

            } else {
                ii = nintd(ox);
                jj = nintd(oy);

                /* Check it is on the output image */
                if (ii < 0 || ii >= osize[0] || jj < 0 || jj >= osize[1]) {
                    ++p->nmiss;

                } else {
                    /* Allow for stretching because of scale change */
                    d = get_pixel(p->data, i, j) * p->iscale;

                    /* Scale the weighting mask by the scale factor.  Note
                       that we DON'T scale by the Jacobian as it hasn't been
                       calculated */
                    if (p->weights) {
                        dow = get_pixel(p->weights, i, j) * p->weight_scale;
                    } else {
                        dow = 1.0f;
                    }

                    /* If we are creating or modifying the context image,
                       we do so here. */
                    if (p->output_context && dow > 0.0f) {
                        set_bit(p->output_context, ii, jj, bv);
                    }

                    if (update_data(p, ii, jj, d, dow)) {
                        return 1;
                    }
                }
            }
        }
    }

    return 0;
}

/** ---------------------------------------------------------------------------
 * This kernel assumes the flux is distributed acrass a gaussian around the
 * center of an input pixel
 *
 * p: structure containing options, input, and output
 */

static int
do_kernel_gaussian(struct driz_param_t *p)
{
    struct scanner s;
    integer_t bv, i, j, ii, jj, nxi, nxa, nyi, nya, nhit;
    integer_t osize[2];
    float d, dow;
    double gaussian_efac, gaussian_es;
    double pfo, ac, kscale2, w, ddx, ddy, r2, dover;
    const double nsig = 2.5;
    int xmin, xmax, ymin, ymax, n;

    if (init_image_scanner(p, &s, &ymin, &ymax)) {
        return 1;
    }

    /* if pscale_ratio is not defined, estimate it near the center of the
       intersection polygon.
     */
    if (!isfinite(p->pscale_ratio) &&
        compute_pscale_ratio(p, &s.bounding_polygon, &p->pscale_ratio)) {
        return 1;
    }

    /* Added in V2.9 - make sure pfo doesn't get less than 1.2
       divided by the scale so that there are never holes in the
       output */

    pfo = nsig * p->pixel_fraction / 2.3548 / (double) p->pscale_ratio;
    pfo = CLAMP_ABOVE(pfo, 1.2 / (double) p->pscale_ratio);

    ac = 1.0 / (p->pixel_fraction * p->pixel_fraction);
    kscale2 = (double) p->pscale_ratio * (double) p->pscale_ratio;
    bv = compute_bit_value(p->uuid);

    gaussian_efac = (2.3548 * 2.3548) * kscale2 * ac / 2.0;
    gaussian_es = gaussian_efac / M_PI;

    p->nskip = (p->ymax - p->ymin) - (ymax - ymin);
    p->nmiss = p->nskip * (p->xmax - p->xmin);

    /* This is the outer loop over all the lines in the input image */

    get_dimensions(p->output_data, osize);
    for (j = ymin; j <= ymax; ++j) {
        /* Check the overlap with the output */
        n = get_scanline_limits(&s, j, &xmin, &xmax);
        if (n == 1) {
            // scan ended (y reached the top vertex/edge)
            p->nskip += (ymax + 1 - j);
            p->nmiss += (ymax + 1 - j) * (p->xmax - p->xmin);
            break;
        } else if (n == 2 || n == 3) {
            // pixel centered on y is outside of scanner's limits or image
            // [0, height - 1] OR: limits (x1, x2) are equal (line width is
            // 0)
            p->nmiss += (p->xmax - p->xmin);
            ++p->nskip;
            continue;
        } else {
            // limits (x1, x2) are equal (line width is 0)
            p->nmiss += (p->xmax - p->xmin) - (xmax + 1 - xmin);
        }

        for (i = xmin; i <= xmax; ++i) {
            double ox, oy;

            if (map_pixel(p->pixmap, i, j, &ox, &oy)) {
                nhit = 0;

            } else {
                /* Offset within the subset */
                nxi = MAX(nintd(ox - pfo), 0);
                nxa = MIN(nintd(ox + pfo), osize[0] - 1);
                nyi = MAX(nintd(oy - pfo), 0);
                nya = MIN(nintd(oy + pfo), osize[1] - 1);

                nhit = 0;

                /* Allow for stretching because of scale change */
                d = get_pixel(p->data, i, j) * p->iscale;

                /* Scale the weighting mask by the scale factor and
                   inversely by the Jacobian to ensure conservation of
                   weight in the output
                 */
                if (p->weights) {
                    w = get_pixel(p->weights, i, j) * p->weight_scale;
                } else {
                    w = 1.0;
                }

                /* Loop over output pixels which could be affected */
                for (jj = nyi; jj <= nya; ++jj) {
                    ddy = oy - (double) jj;
                    for (ii = nxi; ii <= nxa; ++ii) {
                        ddx = ox - (double) ii;
                        /* Radial distance */
                        r2 = ddx * ddx + ddy * ddy;

                        /* Weight is a scaled Gaussian function of radial
                           distance */
                        dover = gaussian_es * exp(-r2 * gaussian_efac);

                        /* Count the hits */
                        ++nhit;

                        dow = (float) dover * w;

                        /* If we are creating or modifying the context
                           image, we do so here. */
                        if (p->output_context && dow > 0.0f) {
                            set_bit(p->output_context, ii, jj, bv);
                        }

                        if (update_data(p, ii, jj, d, dow)) {
                            return 1;
                        }
                    }
                }
            }

            /* Count cases where the pixel is off the output image */
            if (nhit == 0) {
                ++p->nmiss;
            }
        }
    }

    return 0;
}

/** ---------------------------------------------------------------------------
 * This kernel assumes flux of input pixel is distributed according to
 * lanczos function
 *
 * p: structure containing options, input, and output
 */

static int
do_kernel_lanczos(struct driz_param_t *p)
{
    struct scanner s;
    integer_t bv, i, j, ii, jj, nxi, nxa, nyi, nya, nhit;
    integer_t osize[2];
    float d, dow;
    double pfo, xx, yy, w, dover;
    int kernel_order;
    size_t nlut;
    size_t ix, iy;
    double sdp;
    double *lut = NULL;
    int xmin, xmax, ymin, ymax, n;

    if (fabs(p->pixel_fraction - 1.0) > 1.0e-5) {
        py_warning(
            NULL, "In lanczos kernel, pixel_fraction is ignored and "
                  "assumed to be 1.0");
    }

    kernel_order = (p->kernel == kernel_lanczos2) ? 2 : 3;

    bv = compute_bit_value(p->uuid);

    /* Set up a look-up-table for Lanczos-style interpolation
       kernels */
    nlut = (size_t) ceil(kernel_order / lut_delta) + 1;
    if ((lut = malloc(nlut * sizeof(double))) == NULL) {
        driz_error_set_message(p->error, "Out of memory");
        return driz_error_is_set(p->error);
    }
    create_lanczos_lut(kernel_order, nlut, lut_delta, lut);

    if (init_image_scanner(p, &s, &ymin, &ymax)) {
        return 1;
    }

    /* if pscale_ratio is not defined, estimate it near the center of the
       intersection polygon.
     */
    if (!isfinite(p->pscale_ratio) &&
        compute_pscale_ratio(p, &s.bounding_polygon, &p->pscale_ratio)) {
        return 1;
    }

    pfo = (double) kernel_order / (double) p->pscale_ratio;
    sdp = (double) p->pscale_ratio / lut_delta;

    p->nskip = (p->ymax - p->ymin) - (ymax - ymin);
    p->nmiss = p->nskip * (p->xmax - p->xmin);

    /* This is the outer loop over all the lines in the input image */

    get_dimensions(p->output_data, osize);
    for (j = ymin; j <= ymax; ++j) {
        /* Check the overlap with the output */
        n = get_scanline_limits(&s, j, &xmin, &xmax);
        if (n == 1) {
            // scan ended (y reached the top vertex/edge)
            p->nskip += (ymax + 1 - j);
            p->nmiss += (ymax + 1 - j) * (p->xmax - p->xmin);
            break;
        } else if (n == 2 || n == 3) {
            // pixel centered on y is outside of scanner's limits or image
            // [0, height - 1] OR: limits (x1, x2) are equal (line width is
            // 0)
            p->nmiss += (p->xmax - p->xmin);
            ++p->nskip;
            continue;
        } else {
            // limits (x1, x2) are equal (line width is 0)
            p->nmiss += (p->xmax - p->xmin) - (xmax + 1 - xmin);
        }

        for (i = xmin; i <= xmax; ++i) {
            if (map_pixel(p->pixmap, i, j, &xx, &yy)) {
                nhit = 0;

            } else {
                nxi = MAX((integer_t) floor(xx - pfo) + 1, 0);
                nxa = MIN((integer_t) floor(xx + pfo), osize[0] - 1);
                nyi = MAX((integer_t) floor(yy - pfo) + 1, 0);
                nya = MIN((integer_t) floor(yy + pfo), osize[1] - 1);

                nhit = 0;

                /* Allow for stretching because of scale change */
                d = get_pixel(p->data, i, j) * p->iscale;

                /* Scale the weighting mask by the scale factor and
                   inversely by the Jacobian to ensure conservation of
                   weight in the output
                 */
                if (p->weights) {
                    w = get_pixel(p->weights, i, j) * p->weight_scale;
                } else {
                    w = 1.0;
                }

                /* Loop over output pixels which could be affected */
                for (jj = nyi; jj <= nya; ++jj) {
                    for (ii = nxi; ii <= nxa; ++ii) {
                        /* X and Y offsets */
                        ix = nintd(fabs((xx - (double) ii) * sdp));
                        iy = nintd(fabs((yy - (double) jj) * sdp));
                        if (ix >= nlut || iy >= nlut) {
                            continue;
                        }

                        /* Count the hits */
                        ++nhit;

                        /* Weight is product of Lanczos function values in X
                         * and
                         * Y */
                        dover = lut[ix] * lut[iy];

                        dow = (float) (dover * w);

                        /* If we are creating or modifying the context
                           image, we do so here. */
                        if (p->output_context && dow > 0.0f) {
                            set_bit(p->output_context, ii, jj, bv);
                        }

                        if (update_data(p, ii, jj, d, dow)) {
                            free(lut);
                            return 1;
                        }
                    }
                }
            }

            /* Count cases where the pixel is off the output image */
            if (nhit == 0) {
                ++p->nmiss;
            }
        }
    }

    free(lut);
    return 0;
}

/** ---------------------------------------------------------------------------
 * This kernel assumes the input flux is evenly distributed over a rectangle
 * whose sides are aligned with the ouput pixel. Called turbo because it is
 * fast, but approximate.
 *
 * p: structure containing options, input, and output
 */

static int
do_kernel_turbo(struct driz_param_t *p)
{
    struct scanner s;
    integer_t bv, i, j, ii, jj, nhit, iis, iie, jjs, jje;
    integer_t osize[2];
    float d, dow;
    double pfo, ac;
    double xxi, xxa, yyi, yya, w, dover, dover_scale;
    int xmin, xmax, ymin, ymax, n;

    driz_log_message("starting do_kernel_turbo");
    bv = compute_bit_value(p->uuid);
    ac = 1.0 / (p->pixel_fraction * p->pixel_fraction);

    if (init_image_scanner(p, &s, &ymin, &ymax)) {
        return 1;
    }

    /* if pscale_ratio is not defined, estimate it near the center of the
       intersection polygon.
     */
    if (!isfinite(p->pscale_ratio) &&
        compute_pscale_ratio(p, &s.bounding_polygon, &p->pscale_ratio)) {
        return 1;
    }

    pfo = p->pixel_fraction / p->pscale_ratio / 2.0;
    dover_scale = ac * (double) p->pscale_ratio * (double) p->pscale_ratio;

    p->nskip = (p->ymax - p->ymin) - (ymax - ymin);
    p->nmiss = p->nskip * (p->xmax - p->xmin);

    /* This is the outer loop over all the lines in the input image */

    get_dimensions(p->output_data, osize);
    for (j = ymin; j <= ymax; ++j) {
        /* Check the overlap with the output */
        n = get_scanline_limits(&s, j, &xmin, &xmax);

        if (n == 1) {
            // scan ended (y reached the top vertex/edge)
            p->nskip += (ymax + 1 - j);
            p->nmiss += (ymax + 1 - j) * (p->xmax - p->xmin);
            break;
        } else if (n == 2 || n == 3) {
            // pixel centered on y is outside of scanner's limits or image
            // [0, height - 1] OR: limits (x1, x2) are equal (line width is
            // 0)
            p->nmiss += (p->xmax - p->xmin);
            ++p->nskip;
            continue;
        } else {
            // limits (x1, x2) are equal (line width is 0)
            p->nmiss += (p->xmax - p->xmin) - (xmax + 1 - xmin);
        }

        for (i = xmin; i <= xmax; ++i) {
            double ox, oy;

            if (map_pixel(p->pixmap, i, j, &ox, &oy)) {
                nhit = 0;

            } else {
                /* Offset within the subset */
                xxi = ox - pfo;
                xxa = ox + pfo;
                yyi = oy - pfo;
                yya = oy + pfo;

                iis = MAX(nintd(xxi), 0);
                iie = MIN(nintd(xxa), osize[0] - 1);
                jjs = MAX(nintd(yyi), 0);
                jje = MIN(nintd(yya), osize[1] - 1);

                nhit = 0;

                /* Allow for stretching because of scale change */
                d = get_pixel(p->data, i, j) * p->iscale;

                /* Scale the weighting mask by the scale factor and
                   inversely by the Jacobian to ensure conservation of
                   weight in the output.
                 */
                if (p->weights) {
                    w = get_pixel(p->weights, i, j) * p->weight_scale;
                } else {
                    w = 1.0;
                }

                /* Loop over the output pixels which could be affected */
                for (jj = jjs; jj <= jje; ++jj) {
                    for (ii = iis; ii <= iie; ++ii) {
                        /* Calculate the overlap using the simpler "aligned"
                           box routine */
                        dover = over(ii, jj, xxi, xxa, yyi, yya);

                        if (dover > 0.0) {
                            /* Correct for the pixfrac area factor */
                            dover *= dover_scale;

                            /* Count the hits */
                            ++nhit;

                            dow = (float) (dover * w);

                            /* If we are creating or modifying the context
                               image, we do so here. */
                            if (p->output_context && dow > 0.0f) {
                                set_bit(p->output_context, ii, jj, bv);
                            }

                            if (update_data(p, ii, jj, d, dow)) {
                                return 1;
                            }
                        }
                    }
                }
            }

            /* Count cases where the pixel is off the output image */
            if (nhit == 0) {
                ++p->nmiss;
            }
        }
    }

    driz_log_message("ending do_kernel_turbo");
    return 0;
}

/** ---------------------------------------------------------------------------
 * This module does the actual mapping of input flux to output images. It
 * works by calculating the positions of the four corners of a quadrilateral
 * on the output grid corresponding to the corners of the input pixel and
 * then working out exactly how much of each pixel in the output is covered,
 * or not.
 *
 * p: structure containing options, input, and output
 */

int
do_kernel_square(struct driz_param_t *p)
{
    integer_t bv, i, j, ii, jj, min_ii, max_ii, min_jj, max_jj, nhit;
    integer_t osize[2], mapsize[2];
    float d, dow;
    double dh, jaco, dover, w;
    double xin[4], yin[4], xout[4], yout[4];

    struct scanner s;
    int xmin, xmax, ymin, ymax, n;

    driz_log_message("starting do_kernel_square");
    dh = 0.5 * p->pixel_fraction;
    bv = compute_bit_value(p->uuid);

    /* Next the "classic" drizzle square kernel...  this is different
       because we have to transform all four corners of the shrunken
       pixel */
    if (init_image_scanner(p, &s, &ymin, &ymax)) {
        return 1;
    }

    p->nskip = (p->ymax - p->ymin) - (ymax - ymin);
    p->nmiss = p->nskip * (p->xmax - p->xmin);

    /* This is the outer loop over all the lines in the input image */
    get_dimensions(p->output_data, osize);
    get_dimensions(p->pixmap, mapsize);

    for (j = ymin; j <= ymax; ++j) {
        /* Check the overlap with the output */
        n = get_scanline_limits(&s, j, &xmin, &xmax);
        if (n == 1) {
            // scan ended (y reached the top vertex/edge)
            p->nskip += (ymax + 1 - j);
            p->nmiss += (ymax + 1 - j) * (p->xmax - p->xmin);
            break;
        } else if (n == 2 || n == 3) {
            // pixel centered on y is outside of scanner's limits or image
            // [0, height - 1] OR: limits (x1, x2) are equal (line width is
            // 0)
            p->nmiss += (p->xmax - p->xmin);
            ++p->nskip;
            continue;
        } else {
            // limits (x1, x2) are equal (line width is 0)
            p->nmiss += (p->xmax - p->xmin) - (xmax + 1 - xmin);
        }

        /* Set the input corner positions */
        yin[1] = yin[0] = (double) j + dh;
        yin[3] = yin[2] = (double) j - dh;

        for (i = xmin; i <= xmax; ++i) {
            nhit = 0;

            // xin[3] = xin[0] = (double)i - dh;
            // xin[2] = xin[1] = (double)i + dh;

            // for (ii = 0; ii < 4; ++ii) {
            //     if (interpolate_point(p, xin[ii], yin[ii], xout + ii,
            //                           yout + ii)) {
            //         goto _miss;
            //     }
            // }

            /* Assuming we don't need to extrapolate, call a more
             * efficient interpolator that takes advantage of the fact
             * that pixfrac<1 and that we are using a square grid.
             */
            if (i > 0 && i < mapsize[0] - 2 && j > 0 && j < mapsize[1] - 2) {
                if (interpolate_four_points(
                        p, i, j, dh, xout, xout + 1, xout + 2, xout + 3, yout, yout + 1, yout + 2,
                        yout + 3)) {
                    goto _miss;
                }
            } else {
                xin[3] = xin[0] = (double) i - dh;
                xin[2] = xin[1] = (double) i + dh;
                for (ii = 0; ii < 4; ++ii) {
                    if (interpolate_point(p, xin[ii], yin[ii], xout + ii, yout + ii)) {
                        goto _miss;
                    }
                }
            }

            /* Work out the area of the quadrilateral on the output
             * grid.  If the points are in clockwise order we get a
             * postive area.  If they are in anticlockwise order, jaco
             * will be negative, but so will the areas computed by
             * boxer, so it doesn't actually matter once we divide it
             * out.
             */

            jaco = 0.5f * ((xout[1] - xout[3]) * (yout[0] - yout[2]) -
                           (xout[0] - xout[2]) * (yout[1] - yout[3]));

            /* Allow for stretching because of scale change */
            d = get_pixel(p->data, i, j) * p->iscale;

            /* Scale the weighting mask by the scale factor and inversely by
               the Jacobian to ensure conservation of weight in the output
             */
            if (p->weights) {
                w = get_pixel(p->weights, i, j) * p->weight_scale / jaco;
            } else {
                w = 1.0 / jaco;
            }

            /* Loop over output pixels which could be affected */
            min_jj = MAX(nintd(min_doubles(yout, 4)), 0);
            max_jj = MIN(nintd(max_doubles(yout, 4)), osize[1] - 1);
            min_ii = MAX(nintd(min_doubles(xout, 4)), 0);
            max_ii = MIN(nintd(max_doubles(xout, 4)), osize[0] - 1);

            for (jj = min_jj; jj <= max_jj; ++jj) {
                for (ii = min_ii; ii <= max_ii; ++ii) {
                    /* Call boxer to calculate overlap */
                    // dover = compute_area((double)ii, (double)jj, xout,
                    // yout);
                    dover = boxer((double) ii, (double) jj, xout, yout);

                    /* Could be positive or negative, depending on the sign
                     * of jaco */
                    if (dover != 0.0) {
                        dow = (float) (dover * w);

                        /* Count the hits */
                        ++nhit;

                        /* If we are creating or modifying the context image
                           we do so here */
                        if (p->output_context && dow > 0.0f) {
                            set_bit(p->output_context, ii, jj, bv);
                        }

                        if (update_data(p, ii, jj, d, dow)) {
                            return 1;
                        }
                    }
                }
            }

        /* Count cases where the pixel is off the output image */
        _miss:
            if (nhit == 0) {
                ++p->nmiss;
            }
        }
    }

    driz_log_message("ending do_kernel_square");
    return 0;
}

/** ---------------------------------------------------------------------------
 * The user selects a kernel to use for drizzling from a function in the
 * following tables The kernels differ in how the flux inside a single pixel
 * is allocated: evenly spread across the pixel, concentrated at the central
 * point, or by some other function.
 */

static kernel_handler_t kernel_handler_map[] = {do_kernel_square,  do_kernel_gaussian,
                                                do_kernel_point,   do_kernel_turbo,
                                                do_kernel_lanczos, do_kernel_lanczos};

static kernel_handler_t kernel_var_handler_map[] = {do_kernel_square_var,  do_kernel_gaussian_var,
                                                    do_kernel_point_var,   do_kernel_turbo_var,
                                                    do_kernel_lanczos_var, do_kernel_lanczos_var};

/** ---------------------------------------------------------------------------
 * The executive function which calls the kernel which does the actual
 * drizzling
 *
 * p: structure containing options, input, and output
 */

int
dobox(struct driz_param_t *p)
{
    kernel_handler_t kernel_handler = NULL;
    driz_log_message("starting dobox");

    /* Set up a function pointer to handle the appropriate kernel */
    if (p->kernel < kernel_LAST) {
        if (p->ndata2 > 0 || p->dq != NULL) {
            kernel_handler = kernel_var_handler_map[p->kernel];
        } else {
            kernel_handler = kernel_handler_map[p->kernel];
        }

        if (kernel_handler != NULL) {
            kernel_handler(p);
        }
    }

    if (kernel_handler == NULL) {
        driz_error_set_message(p->error, "Invalid kernel type");
    }

    driz_log_message("ending dobox");
    return driz_error_is_set(p->error);
}
