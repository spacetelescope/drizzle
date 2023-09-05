#ifndef CDRIZZLEMAP_H
#define CDRIZZLEMAP_H

#include "driz_portability.h"
#include "cdrizzleutil.h"

/* Line segment structure, used for computing overlap
 * The first index on line is the endpoint
 * The second index is {x, y) coordinate of the point
 * The valid flag is non-zero if it does not intersect the image
 */

struct segment {
    double  point[2][2];
    int     invalid;
};

// IMAGE_OUTLINE_NPTS - maximum number of vertices in the bounding polygon
// for input and resampled images
#define IMAGE_OUTLINE_NPTS 4

/** vertex structure.
 *
 *  This structure holds coordinates of a polygon vertex.
 *
 */
struct vertex {
    double x; /**< x-coordinate */
    double y; /**< y-coordinate */
};

/** polygon structure.
 *
 *  This structure holds information about polygon vertices. The maximum number
 *  of vertices that this structure can hold is determined by the constant
 *  IMAGE_OUTLINE_NPTS and it is double of IMAGE_OUTLINE_NPTS value.
 *
 *  NOTE: polygons must not be closed (that is last vertex != first vertex).
 *
 */
struct polygon {
    struct vertex v[2 * IMAGE_OUTLINE_NPTS];  /**< polygon vertices */
    int    npv;  /**< actual number of polygon vertices */
};

/** edge structure.
 *
 *  This structure holds invormation about vertices of the edge, edge position
 *  in the polygon (left or right of the line going through the top and bottom
 *  vertices), slope, and interceipts.
 *
 */
struct edge {
    struct vertex v1; /**< first vertex */
    struct vertex v2; /**< second vertex */
    double m; /**< edge's slope */
    double b; /**< edge's interceipt */
    double c; /**< modified interceipt */
    int p;  /**< edge's position: -1 for left-side edge and +1 for right-side edge */
};

/** scanner structure.
 *
 *  This structure holds information needed to "scan" or rasterize the
 *  intersection polygon formed by intersecting the bounding box of the output
 *  frame with the bounding box of the input frame in the input frame's
 *  coordinate system.
 *
 */
struct scanner {
    struct edge left_edges[2 * IMAGE_OUTLINE_NPTS]; /**< left edges */
    struct edge right_edges[2 * IMAGE_OUTLINE_NPTS]; /**< right edges */
    struct edge *left; /**< pointer to the current left edge; NULL when top polygon vertex was reached */
    struct edge *right; /**< pointer to the current right edge; NULL when top polygon vertex was reached */
    int nleft; /**< number of left edges */
    int nright; /**< number of right edges */
    double min_y; /**< minimum y-coordinate of all polygon vertices */
    double max_y; /**< maximum y-coordinate of all polygon vertices */
    int xmin; /**< min valid pixels' x-coord in pixmap (from bounding box carried over from driz_param_t) rounded to int */
    int xmax; /**< max valid pixels' x-coord in pixmap (from bounding box carried over from driz_param_t) rounded to int */
    int ymin; /**< min valid pixels' y-coord in pixmap (from bounding box carried over from driz_param_t) rounded to int */
    int ymax; /**< max valid pixels' y-coord in pixmap (from bounding box carried over from driz_param_t) rounded to int */
    // overlap_valid: 1 if polygon intersection and coord inversion worked;
    //                0 if computation of xmin, xmax, ymin, ymax has
    //                  failed in which case they are carried over from driz_param_t.
    int overlap_valid; /**< 1 if x/y min/max updated from polygon intersection and 0 if carried over from driz_param_t */
};

int
interpolate_point(struct driz_param_t *par, double xin, double yin,
                  double *xout, double *yout);

int
map_point(struct driz_param_t *par, double xin, double yin,
          double *xout, double *yout);

int
map_pixel(PyArrayObject *pixmap, int i, int j, double *x, double *y);

int
shrink_image_section(PyArrayObject *pixmap, int *xmin, int *xmax,
                     int *ymin, int *ymax);

int
invert_pixmap(struct driz_param_t* par, double xout, double yout,
              double *xin, double *yin);

int
clip_polygon_to_window(const struct polygon *p, const struct polygon *wnd,
                       struct polygon *cp);

int
init_scanner(struct polygon *p, struct driz_param_t* par, struct scanner *s);

int
get_scanline_limits(struct scanner *s, int y, int *x1, int *x2);

int
init_image_scanner(struct driz_param_t* par, struct scanner *s,
                   int *ymin, int *ymax);

#endif /* CDRIZZLEMAP_H */
