#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "pandokia_fct.h"
#include "cdrizzlebox.h"
#include "cdrizzleblot.h"
#include "cdrizzlemap.h"
#include "cdrizzleutil.h"
#include "drizzletest.h"

static integer_t image_size[2];
static PyArrayObject *test_data;
static PyArrayObject *test_weights;
static PyArrayObject *test_pixmap;
static PyArrayObject *test_output_data;
static PyArrayObject *test_output_counts;
static PyArrayObject *test_context;

void
set_test_arrays(PyArrayObject *dat,
                PyArrayObject *wei,
                PyArrayObject *map,
                PyArrayObject *odat,
                PyArrayObject *ocnt,
                PyArrayObject *ocon) {
    
    test_data = dat;
    test_weights = wei;
    test_pixmap = map;
    test_output_data = odat;
    test_output_counts = ocnt;
    test_context = ocon;
    
    get_dimensions(test_data, image_size);
    return;
}

void
init_pixmap(struct driz_param_t *p) {
    int i, j;
    double xpix, ypix;
    
    ypix = 0.0;
    for (j = 0; j < image_size[1]; j++) {
       xpix = 0.0;
       for (i= 0; i < image_size[0]; i++) {
            get_pixmap(p->pixmap, i, j)[0] = xpix;
            get_pixmap(p->pixmap, i, j)[1] = ypix;
            xpix += 1.0;
       }
       ypix += 1.0;
    }

    return;
}

void
stretch_pixmap(struct driz_param_t *p, double stretch) {
    
    int i, j;
    double xpix, ypix;

    ypix = 0.0;
    for (j = 0; j < image_size[1]; j++) {
       xpix = 0.0;
       for (i= 0; i < image_size[0]; i++) {
            get_pixmap(p->pixmap, i, j)[0] = xpix;
            get_pixmap(p->pixmap, i, j)[1] = stretch * ypix;
            xpix += 1.0;
       }
       ypix += 1.0;
    }

    return;
}

void
offset_pixmap(struct driz_param_t *p, double x_offset, double y_offset) {
    
    int i, j;
    double xpix, ypix;

    ypix = 0.0;
    for (j = 0; j < image_size[1]; j++) {
       xpix = 0.0;
       for (i = 0; i < image_size[0]; i++) {
            get_pixmap(p->pixmap, i, j)[0] = xpix + x_offset;
            get_pixmap(p->pixmap, i, j)[1] = ypix + y_offset;
            xpix += 1.0;
       }
       ypix += 1.0;
    }

    return;
}

void
fill_image(PyArrayObject *image, double value) {
    npy_intp   *ndim = PyArray_DIMS(image);
    int ypix, xpix;

    for (ypix = 0; ypix < ndim[0]; ++ypix) {
        for (xpix = 0; xpix < ndim[1]; ++xpix) {
            set_pixel(image, xpix, ypix, value);
        }
    }
    
    return;
}

void
unset_context(PyArrayObject *context) {
    npy_intp   *ndim = PyArray_DIMS(context);
    int ypix, xpix;

    for (ypix = 0; ypix < ndim[0]; ++ypix) {
        for (xpix = 0; xpix < ndim[1]; ++xpix) {
            unset_bit(context, xpix, ypix);
        }
    }
    
    return;
}

void
print_image(char *title, PyArrayObject* image, int lo, int hi) {
    int j, i;
    
    printf("\n%s\n", title);
    for (j = lo; j < hi; ++j) {
        for (i = lo; i < hi; ++i) {
            printf("%10.2f", get_pixel(image, i, j));
        }
        printf("\n");
    }
    
    return;
}

void
print_context(char *title, struct driz_param_t *p, int lo, int hi) {
    int j, i;
    integer_t bv;
    
    bv = 1;
    printf("\n%s\n", title);

    for (j = lo; j < hi; ++j) {
        for (i = lo; i < hi; ++i) {
            printf("%4d", get_bit(p->output_context, i, j, bv));
        }
        printf("\n");
    }
    
    return;
}

void
print_pixmap(char *title, struct driz_param_t *p, int lo, int hi) {
    int     i, j, k;
    char *axis[2] = {"x", "y"};

    for (k = 0; k < 2; k ++) {
        double *imptr = (double *) PyArray_DATA(p->pixmap);
        printf("\n%s %s axis\n", title, axis[k]);

        for (j = 0; j < image_size[1]; j++ ) {
            for (i = 0; i < image_size[0]; i++) {
                if (i >= lo && i < hi && j >= lo && j < hi) {
                    printf("%10.2f", get_pixmap(p->pixmap, i, j)[k]);
                }
            }

            if (j >= lo && j < hi) printf("\n");
        }
    }
    
    return;
}

struct driz_param_t *
setup_parameters() {
    npy_intp *dims;
    struct driz_error_t *error;

    /* Initialize the parameter struct with vanilla defaults */
    
    struct driz_param_t *p;
    p = (struct driz_param_t *) malloc(sizeof(struct driz_param_t));

    driz_param_init(p);

    p->uuid = 1;
    p->xmin = 0;
    p->xmax = image_size[0];
    p->ymin = 0;
    p->ymax = image_size[1];
    p->scale = 1.0;
    p->pixel_fraction = 1.0;
    p->exposure_time = 1.0;
    p->ef = p->exposure_time;
    p->kernel = kernel_square;
    p->interpolation = interp_poly5;
    p->weight_scale = 1.0;

    p->data = test_data;
    p->weights = test_weights;
    p->pixmap = test_pixmap;
    p->output_data = test_output_data;
    p->output_counts = test_output_counts;
    p->output_context = test_context;
    p->nmiss = 0;
    p->nskip = 0;

    error = (struct driz_error_t *) malloc(sizeof(struct driz_error_t));
    driz_error_init(error);
    p->error = error;
    init_pixmap(p);
    
    fill_image(p->data, 0.0);
    fill_image(p->weights, 1.0);
    fill_image(p->output_data, 0.0);
    fill_image(p->output_counts, 0.0);
    unset_context(p->output_context);

    return p;
}

void
teardown_parameters(struct driz_param_t *p) {
    
    free(p->error);
    free(p);
}

FCT_BGN_FN(utest_cdrizzle)
{
    FCT_FIXTURE_SUITE_BGN("unit tests for drizzle")
    {       
        FCT_SETUP_BGN()
        {
        }
        FCT_SETUP_END();

        FCT_TEARDOWN_BGN()
        {
        }
        FCT_TEARDOWN_END();

        FCT_TEST_BGN(utest_map_lookup_01)
        {
            struct driz_param_t *p;
            p = setup_parameters();
            stretch_pixmap(p, 1000.0);
            
            double x = get_pixmap(p->pixmap, 1, 1)[0];
            double y = get_pixmap(p->pixmap, 1, 1)[1];
            
            fct_chk_eq_dbl(x, 1.0);
            fct_chk_eq_dbl(y, 1000.0);
            
            teardown_parameters(p);
        }
        FCT_TEST_END();
  
        FCT_TEST_BGN(utest_map_lookup_02)
        {
            struct driz_param_t *p;
            p = setup_parameters();
            stretch_pixmap(p, 1000.0);
            
            double x = get_pixmap(p->pixmap, 3, 0)[0];
            double y = get_pixmap(p->pixmap, 3, 0)[1];

            fct_chk_eq_dbl(x, 3.0);
            fct_chk_eq_dbl(y, 0.0);
            
            teardown_parameters(p);
        }
        FCT_TEST_END();

        FCT_TEST_BGN(utest_map_lookup_03)
        {
            struct driz_param_t *p;
            p = setup_parameters();
            stretch_pixmap(p, 1000.0);
            
            double x = get_pixmap(p->pixmap, 0, 1)[0];
            double y = get_pixmap(p->pixmap, 0, 1)[1];
            fct_chk_eq_dbl(x, 0.0);
            fct_chk_eq_dbl(y, 1000.0);
            
            teardown_parameters(p);
        }
        FCT_TEST_END();

        FCT_TEST_BGN(utest_map_lookup_04)
        {
            struct driz_param_t *p;
            p = setup_parameters();
            stretch_pixmap(p, 1000.0);
            
            double x = get_pixmap(p->pixmap, 3, 1)[0];
            double y = get_pixmap(p->pixmap, 3, 1)[1];
            fct_chk_eq_dbl(x, 3.0);
            fct_chk_eq_dbl(y, 1000.0);
            
            teardown_parameters(p);
        }
        FCT_TEST_END();

        FCT_TEST_BGN(utest_map_point_01)
        {
            double xyin[2], xyout[2];
            struct driz_param_t *p;

            p = setup_parameters();
            stretch_pixmap(p, 1000.0);
            
            xyin[0] = 2.5;
            xyin[1] = 1.5;
            
            map_point(p->pixmap, xyin, xyout);
    
            fct_chk_eq_dbl(xyout[0], 2.5);
            fct_chk_eq_dbl(xyout[1], 1500.0);
            
            teardown_parameters(p);
        }
        FCT_TEST_END();
  
        FCT_TEST_BGN(utest_map_point_02)
        {
            double xyin[2], xyout[2];
            struct driz_param_t *p;

            p = setup_parameters();
            stretch_pixmap(p, 1000.0);
            
            xyin[0] = -1.0;
            xyin[1] = 0.5;
            
            map_point(p->pixmap, xyin, xyout);
    
            fct_chk_eq_dbl(xyout[0], -1.0);
            fct_chk_eq_dbl(xyout[1], 500.0);
            
            teardown_parameters(p);
        }
        FCT_TEST_END();
  
        FCT_TEST_BGN(utest_check_line_overlap_01)
        {
            /* Test for complete overlap */
            
            const integer_t j = 0;      /* which image line to check ? */
            const int margin = 2;       /* extra margin around edge of image */
            integer_t xbounds[2];       /* start of in-bounds */
            struct driz_param_t *p;     /* parameter structure */
            
            p = setup_parameters();
            offset_pixmap(p, 0.0, 0.0);
            
            check_line_overlap(p, margin, j, xbounds);

            fct_chk_eq_int(xbounds[0], 0);
            fct_chk_eq_int(xbounds[1], image_size[0]);
            
            teardown_parameters(p);
        }
        FCT_TEST_END();

        FCT_TEST_BGN(utest_check_line_overlap_02)
        {
            /* Test for half overlap */
            
            const integer_t j = 0;      /* which image line to check ? */
            const integer_t margin = 2; /* extra margin around edge of image */
            integer_t xbounds[2];       /* start of in-bounds */
            struct driz_param_t *p;     /* parameter structure */
            
            p = setup_parameters();
            offset_pixmap(p, 70.0, 0.0);

            check_line_overlap(p, margin, j, xbounds);
            
            fct_chk_eq_int(xbounds[0], 0);
            fct_chk_eq_int(xbounds[1], 32);
            
            teardown_parameters(p);
        }
        FCT_TEST_END();

        FCT_TEST_BGN(utest_check_line_overlap_03)
        {
            /* Test for negative half overlap */
            
            const integer_t j = 0;      /* which image line to check ? */
            const integer_t margin = 2; /* extra margin around edge of image */
            integer_t xbounds[2];       /* start of in-bounds */
            struct driz_param_t *p;     /* parameter structure */
            
            p = setup_parameters();
            offset_pixmap(p, -70.0, 0.0);

            check_line_overlap(p, margin, j, xbounds);
            
            fct_chk_eq_int(xbounds[0], 68);
            fct_chk_eq_int(xbounds[1], 100);
            
            teardown_parameters(p);
        }
        FCT_TEST_END();

        FCT_TEST_BGN(utest_check_image_overlap_01)
        {
            /* Test for complete overlap */
            
            const int margin = 2;       /* extra margin around edge of image */
            integer_t ybounds[2];       /* start of in-bounds */
            struct driz_param_t *p;     /* parameter structure */
            
            p = setup_parameters();
            offset_pixmap(p, 0.0, 0.0);
            
            check_image_overlap(p, margin, ybounds);

            fct_chk_eq_int(ybounds[0], 0);
            fct_chk_eq_int(ybounds[1], image_size[1]);
            
            teardown_parameters(p);
        }
        FCT_TEST_END();

        FCT_TEST_BGN(utest_check_image_overlap_02)
        {
            /* Test for half overlap */
            
            const integer_t margin = 2; /* extra margin around edge of image */
            integer_t ybounds[2];       /* start of in-bounds */
            struct driz_param_t *p;     /* parameter structure */
            
            p = setup_parameters();
            offset_pixmap(p, 0.0, 70.0);

            check_image_overlap(p, margin, ybounds);
            
            fct_chk_eq_int(ybounds[0], 0);
            fct_chk_eq_int(ybounds[1], 32);
            
            teardown_parameters(p);
        }
        FCT_TEST_END();

        FCT_TEST_BGN(utest_check_image_overlap_03)
        {
            /* Test for negative half overlap */
            
            const integer_t margin = 2; /* extra margin around edge of image */
            integer_t ybounds[2];       /* start of in-bounds */
            struct driz_param_t *p;     /* parameter structure */
            
            p = setup_parameters();
            offset_pixmap(p, 0.0, -70.0);

            check_image_overlap(p, margin, ybounds);
            
            fct_chk_eq_int(ybounds[0], 68);
            fct_chk_eq_int(ybounds[1], 100);
            
            teardown_parameters(p);
        }
        FCT_TEST_END();

        FCT_TEST_BGN(utest_do_kernel_square_01)
        {
            /* Simplest case */
            
            integer_t i, x1;            /* start of in-bounds */
            integer_t j, x2;            /* end of in-bounds */
            struct driz_param_t *p;     /* parameter structure */
            int n, status;
            
            n = 100;
            p = setup_parameters();
            offset_pixmap(p, 0.0, 0.0);
            fill_image(p->data, 5.0);
            
            status = do_kernel_square(p);

            j = 3;
            x1 = 0;
            x2 = n;

            fct_chk_eq_int(status, 0);
            fct_chk_eq_dbl(get_pixel(p->output_data, x1, j), get_pixel(p->data, x1, j));
            fct_chk_eq_dbl(get_pixel(p->output_data, x2-1, j), get_pixel(p->data, x2-1, j));

            teardown_parameters(p);
        }
        FCT_TEST_END();

         FCT_TEST_BGN(utest_do_kernel_square_02)
        {
            /* Offset image */
            
            struct driz_param_t *p;     /* parameter structure */
            integer_t j;
            int k, n, status;
            double offset;
            
            n = 100;
            offset = 2.0;
            p = setup_parameters();
            offset_pixmap(p, offset, 0.0);
            fill_image(p->data, 5.0);
            
            j = 3;
            status = do_kernel_square(p);

            fct_chk_eq_int(status, 0);
            for (k = 1; k < n-2; k++) {
                fct_chk_eq_dbl(get_pixel(p->output_data, (k+1), j), get_pixel(p->data, k, j));
            }

            teardown_parameters(p);
        }
        FCT_TEST_END();

        FCT_TEST_BGN(utest_do_kernel_square_03)
        {
            /* Single pixel set */
            
            struct driz_param_t *p;     /* parameter structure */
            integer_t j;
            int k, n;
            double offset, value;
            
            n = 100;
            offset = 2.0;
            value = 5.0;
            p = setup_parameters();
            offset_pixmap(p, offset, 0.0);

            for (j = 0; j < n; ++j) {
                set_pixel(p->data, j, j, value);
            }
            
            k = 3;
            do_kernel_square(p);
            
            fct_chk_eq_dbl(get_pixel(p->output_data, (k+2), k), get_pixel(p->data, k, k));

            teardown_parameters(p);
        }
        FCT_TEST_END();

        FCT_TEST_BGN(utest_do_kernel_square_04)
        {
            /* Single pixel, fractional pixel offset */
            
            struct driz_param_t *p;     /* parameter structure */
            integer_t i, j, k;
            double offset, value;
            
            offset = 2.5;
            value = 4.0;
            p = setup_parameters();
            offset_pixmap(p, offset, offset);

            k = 1;
            set_pixel(p->data, k, k, value);
            
            do_kernel_square(p);

            for (i = 2; i <= 3; ++i) {
                for (j = 2; j <= 3; ++j) {
                    fct_chk_eq_dbl(get_pixel(p->output_data, (i+k), (j+k)), value/4.0);
                }
            }

            teardown_parameters(p);
        }
        FCT_TEST_END();

        FCT_TEST_BGN(utest_do_kernel_square_05)
        {
            /* Diagonal line, fractional pixel offset */
            
            struct driz_param_t *p;     /* parameter structure */
            integer_t i,j;
            int k, n;
            double offset, value;
            
            n = 100;
            offset = 2.5;
            value = 4.0;
            p = setup_parameters();
            offset_pixmap(p, offset, offset);

            for (j = 1; j < n-1; ++j) {
                set_pixel(p->data, j, j, value);
            }
            
            do_kernel_square(p);

            for (i = 4; i < n; ++i) {
                fct_chk_eq_dbl(get_pixel(p->output_data, i, i), value/2.0);
                fct_chk_eq_dbl(get_pixel(p->output_data, i-1, i), value/4.0);
                fct_chk_eq_dbl(get_pixel(p->output_data, i, i-1), value/4.0);
            }

            teardown_parameters(p);
        }
        FCT_TEST_END();

        FCT_TEST_BGN(utest_do_kernel_square_06)
        {
            /* Block of pixels, whole number offset */
            
            struct driz_param_t *p;     /* parameter structure */
            integer_t i,j;
            int k;
            double offset, value;
            
            k = 2;
            offset = 2.0;
            value = 4.0;
            p = setup_parameters();
            offset_pixmap(p, offset, offset);

            for (i = 0; i < k; ++i) {
                for (j = 0; j < k; ++j) {
                    set_pixel(p->data, i, j, value);
                }
            }
            
            do_kernel_square(p);

            for (i = 0; i < k; ++i) {
                for (j = 0; j < k; ++j) {
                    fct_chk_eq_dbl(get_pixel(p->output_data, (i+2), (j+2)), value);
                }
            }

            teardown_parameters(p);
        }
        FCT_TEST_END();

        FCT_TEST_BGN(utest_do_kernel_square_07)
        {
            /* Block of pixels, fractional offset */
            
            struct driz_param_t *p;     /* parameter structure */
            integer_t i, j;
            int k;
            double offset, value;
            
            k = 2;
            offset = 2.5;
            value = 4.0;
            p = setup_parameters();
            offset_pixmap(p, offset, offset);

            for (i = 1; i < k+1; ++i) {
                for (j = 1; j < k+1; ++j) {
                    set_pixel(p->data, i, j, value);
                }
            }
            
            do_kernel_square(p);

            fct_chk_eq_dbl(get_pixel(p->output_data, 3, 3), 1.0);
            fct_chk_eq_dbl(get_pixel(p->output_data, 3, 5), 1.0);
            fct_chk_eq_dbl(get_pixel(p->output_data, 5, 3), 1.0);
            fct_chk_eq_dbl(get_pixel(p->output_data, 5, 5), 1.0);
            fct_chk_eq_dbl(get_pixel(p->output_data, 3, 4), 2.0);
            fct_chk_eq_dbl(get_pixel(p->output_data, 4, 3), 2.0);
            fct_chk_eq_dbl(get_pixel(p->output_data, 4, 5), 2.0);
            fct_chk_eq_dbl(get_pixel(p->output_data, 5, 4), 2.0);
            fct_chk_eq_dbl(get_pixel(p->output_data, 4, 4), 4.0);

            teardown_parameters(p);
        }
        FCT_TEST_END();

        FCT_TEST_BGN(utest_dobox_01)
        {
            /* Single pixel set, whole number offset */
            
            struct driz_param_t *p;     /* parameter structure */
            int k;
            double offset, value;
            
            k = 2;
            offset = 2.0;
            value = 44.0;

            p = setup_parameters();
            offset_pixmap(p, offset, offset);
            p->kernel = kernel_turbo;

            set_pixel(p->data, k, k, value);
            dobox(p);
 
            fct_chk_eq_dbl(get_pixel(p->output_data, (k+2), (k+2)), get_pixel(p->data, k, k));
            fct_chk_eq_int(p->nskip, 0);
        }
        FCT_TEST_END();

        FCT_TEST_BGN(utest_dobox_02)
        {
            /* Single pixel, fractional pixel offset */

            struct driz_param_t *p;     /* parameter structure */
            int i, j, k;
            double offset, value;
            
            k = 2;
            offset = 2.5;
            value = 44.0;

            p = setup_parameters();
            offset_pixmap(p, offset, offset);
            p->kernel = kernel_turbo;

            set_pixel(p->data, k, k, value);
            dobox(p);

            for (i = 2; i <= 3; ++i) {
                for (j = 2; j <= 3; ++j) {
                    fct_chk_eq_dbl(get_pixel(p->output_data, (i+k), (j+k)), value/4.0);
                }
            }
        }
        FCT_TEST_END();

        FCT_TEST_BGN(utest_dobox_03)
        {
            /* Turbo mode kernel, diagonal line of pixels set */
            
            struct driz_param_t *p;     /* parameter structure */
            int i, j, n;
            double offset, value;
            
            n = 100;
            offset = 2.5;
            value = 4.0;

            p = setup_parameters();
            offset_pixmap(p, offset, offset);
            p->kernel = kernel_turbo;

            for (j = 1; j < n-1; ++j) {
                set_pixel(p->data, j, j, value);
            }

            dobox(p);
            
            for (i = 4; i < n; ++i) {
                fct_chk_eq_dbl(get_pixel(p->output_data, i, i), value/2.0);
                fct_chk_eq_dbl(get_pixel(p->output_data, i-1, i), value/4.0);
                fct_chk_eq_dbl(get_pixel(p->output_data, i, i-1), value/4.0);
            }

            teardown_parameters(p);
        }
        FCT_TEST_END();

        FCT_TEST_BGN(utest_dobox_04)
        {
            /* Check that context map is set for the affected pixels */
            
            struct driz_param_t *p;     /* parameter structure */
            int i, j, n;
            double offset, value;
            integer_t bv;
            
            n = 100;
            offset = 2.5;
            value = 4.0;

            p = setup_parameters();
            offset_pixmap(p, offset, offset);
            p->kernel = kernel_turbo;

            for (j = 1; j < n-1; ++j) {
                set_pixel(p->data, j, j, value);
            }

            dobox(p);

            bv = compute_bit_value(p->uuid);
            for (i = 4; i < 100; ++i) {
                fct_chk_eq_int(get_bit(p->output_context, i, i, bv), 1);
            }

            teardown_parameters(p);
        }
        FCT_TEST_END();

        FCT_TEST_BGN(utest_doblot_01)
        {
            /* Single pixel set blinear interpolation */
            
            struct driz_param_t *p;     /* parameter structure */
            int k;
            double offset, value;
            
            k = 2;
            offset = -2.0;
            value = 5.0;

            p = setup_parameters();
            p->interpolation = interp_bilinear;

            offset_pixmap(p, offset, offset);

            set_pixel(p->data, k, k, value);
            doblot(p);

           fct_chk_eq_dbl(get_pixel(p->output_data, (k+2), (k+2)), get_pixel(p->data, k, k));
        }
        FCT_TEST_END();

        FCT_TEST_BGN(utest_doblot_02)
        {
            /* Single pixel set quintic interpolation*/
            
            struct driz_param_t *p;     /* parameter structure */
            int k;
            double offset, value;
            
            k = 2;
            offset = -2.0;
            value = 5.0;

            p = setup_parameters();
            p->interpolation = interp_bilinear;

            offset_pixmap(p, offset, offset);

            set_pixel(p->data, k, k, value);
            doblot(p);

            fct_chk_eq_dbl(get_pixel(p->output_data, (k+2), (k+2)), get_pixel(p->data, k, k));
        }
        FCT_TEST_END();

   }
    FCT_FIXTURE_SUITE_END();
}
FCT_END_FN();
