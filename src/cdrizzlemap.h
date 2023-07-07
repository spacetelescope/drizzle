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

struct vertex {
    double x;
    double y;
};

struct polygon {
    // holds information about polygon vertices
    // NOTE: polygons are not closed (that is last vertex != first vertex)
    struct vertex v[2 * IMAGE_OUTLINE_NPTS];  // polygon vertices
    int    npv;  // actual number of polygon vertices <= 2 * IMAGE_OUTLINE_NPTS
};

struct edge {
    struct vertex v1, v2;
    double m, b, c;
    int p;  // -1 for left-side edge and +1 for right-side edge
};

struct scanner {
    struct edge left_edges[2 * IMAGE_OUTLINE_NPTS];
    struct edge right_edges[2 * IMAGE_OUTLINE_NPTS];
    struct edge *left, *right;  // when set to NULL => done scanning
    int nleft, nright;
    double min_y, max_y;  // bottom and top vertices
    int xmin, xmax, ymin, ymax;  // min/max x/y of valid pixels in pixmap;
                                 // the bounding box (rounded to int)
};

int
bad_pixel(PyArrayObject *pixmap,
          int i,
          int j
          );

int
bad_weight(PyArrayObject *weights,
           int i,
           int j
           );

int
map_point(struct driz_param_t* par,
          const double xyin[2],
          double xyout[2]
         );

int
map_pixel(PyArrayObject *pixmap,
          int    i,
          int    j,
          double xyout[2]
         );

int
shrink_image_section(PyArrayObject *pixmap, int *xmin, int *xmax,
                     int *ymin, int *ymax);

int
invert_pixmap(struct driz_param_t* par,
              const double xyout[2], double xyin[2]);

int
intersect_convex_polygons(const struct polygon *p, const struct polygon *q,
                          struct polygon *pq);

int
init_scanner(struct polygon *p, struct scanner *s, struct driz_param_t* par);

int
get_scanline_limits(struct scanner *s, int y, int *x1, int *x2);

int
init_image_scanner(struct driz_param_t* par, struct scanner *s,
                   int *ymin, int *ymax);

#endif /* CDRIZZLEMAP_H */
