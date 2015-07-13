#define NO_IMPORT_ARRAY
#define NO_IMPORT_ASTROPY_WCS_API
#include "cdrizzlemap.h"
#include "cdrizzleutil.h"

#include <assert.h>
#define _USE_MATH_DEFINES       /* needed for MS Windows to define M_PI */
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*****************************************************************
 ERROR HANDLING
*/
void
driz_error_init(struct driz_error_t* error) {
  assert(error);

  error->last_message[0] = 0;
}

int
driz_error_check(struct driz_error_t* error, const char* message, int test) {
  if (! test) {
    driz_error_set_message(error, message);
    return 1;
  }

  return 0;
}

void
driz_error_set_message(struct driz_error_t* error, const char* message) {
  assert(error);
  assert(message);

  strncpy(error->last_message, message, MAX_DRIZ_ERROR_LEN);
}

void
driz_error_format_message(struct driz_error_t* error, const char* format, ...) {
  /* See http://c-faq.com/varargs/vprintf.html
     for an explanation of how all this variable length argument list stuff
     works. */
  va_list argp;

  assert(error);
  assert(format);

  va_start(argp, format);
  (void)vsnprintf(error->last_message, MAX_DRIZ_ERROR_LEN, format, argp);
  va_end(argp);
}

const char*
driz_error_get_message(struct driz_error_t* error) {
  assert(error);

  return error->last_message;
}

int
driz_error_is_set(struct driz_error_t* error) {
  assert(error);

  return error->last_message[0] != 0;
}

void
driz_error_unset(struct driz_error_t* error) {
  assert(error);

  driz_error_init(error);
}

/*****************************************************************
 DATA TYPES
*/
void
driz_param_dump(struct driz_param_t* p) {
  assert(p);

  printf("DRIZZLING PARAMETERS:\n"
         "kernel:               %s\n"
         "pixel_fraction:       %f\n"
         "exposure_time:        %f\n"
         "weight_scale:         %f\n"
         "fill_value:           %f\n"
         "do_fill:              %s\n"
         "in_units:             %s\n"
         "out_units:            %s\n"
         "scale:                %f\n",
         kernel_enum2str(p->kernel),
         p->pixel_fraction,
         p->exposure_time,
         p->weight_scale,
         p->fill_value,
         bool2str(p->do_fill),
         unit_enum2str(p->in_units),
         unit_enum2str(p->out_units),
         p->scale
         );
}

void
driz_param_init(struct driz_param_t* p) {
  assert(p);

  /* Kernel shape and size */
  p->kernel = kernel_square;
  p->pixel_fraction = 1.0;

  /* Exposure time */
  p->exposure_time = 1.0;

  /* Weight scale */
  p->weight_scale = 1.0;

  /* Filling */
  p->fill_value = 0.0;
  p->do_fill = 0;

  /* CPS / Counts */
  p->in_units = unit_counts;
  p->out_units = unit_counts;

  p->scale = 1.0;

  /* Input data */
  p->data = NULL;
  p->weights = NULL;
  p->pixmap = NULL;

  /* Output data */
  p->output_data = NULL;
  p->output_counts = NULL;
  p->output_context = NULL;

  p->nmiss = 0;
  p->nskip = 0;
  p->error = NULL;
}

/*****************************************************************
 STRING TO ENUMERATION CONVERSIONS
*/
static const char* kernel_string_table[] = {
  "square",
  "gaussian",
  "point",
  "tophat",
  "turbo",
  "lanczos2",
  "lanczos3",
  NULL
};

static const char* unit_string_table[] = {
  "counts",
  "cps",
  NULL
};

static const char* interp_string_table[] = {
  "nearest",
  "linear",
  "poly3",
  "poly5",
  "spline3",
  "sinc",
  "lsinc",
  "lan3",
  "lan5",
  NULL
};

static const char* bool_string_table[] = {
  "FALSE",
  "TRUE",
  NULL
};

static int
str2enum(const char* s, const char* table[], int* result, struct driz_error_t* error) {
  const char** it = table;

  assert(s);
  assert(table);
  assert(result);
  assert(error);

  while (*it != NULL) {
    if (strncmp(s, *it, 32) == 0) {
      *result = it - table;
      return 0;
    }
    ++it;
  }

  return 1;
}

int
kernel_str2enum(const char* s, enum e_kernel_t* result, struct driz_error_t* error) {
  if (str2enum(s, kernel_string_table, (int *)result, error)) {
    driz_error_format_message(error, "Unknown kernel type '%s'", s);
    return 1;
  }

  return 0;
}

int
unit_str2enum(const char* s, enum e_unit_t* result, struct driz_error_t* error) {
  if (str2enum(s, unit_string_table, (int *)result, error)) {
    driz_error_format_message(error, "Unknown unit type '%s'", s);
    return 1;
  }

  return 0;
}

int
interp_str2enum(const char* s, enum e_interp_t* result, struct driz_error_t* error) {
  if (str2enum(s, interp_string_table, (int *)result, error)) {
    driz_error_format_message(error, "Unknown interp type '%s'", s);
    return 1;
  }

  return 0;
}

const char*
kernel_enum2str(enum e_kernel_t value) {
  assert(value >= 0 && value < kernel_LAST);

  return kernel_string_table[value];
}

const char*
unit_enum2str(enum e_unit_t value) {
  assert(value >= 0 && value < 2);

  return unit_string_table[value];
}

const char*
interp_enum2str(enum e_interp_t value) {
  assert(value >= 0 && value < interp_LAST);

  return interp_string_table[value];
}

const char*
bool2str(bool_t value) {
  return bool_string_table[value ? 1 : 0];
}

/*****************************************************************
 NUMERICAL UTILITIES
*/
void
create_lanczos_lut(const int kernel_order, const size_t npix,
                   const float del, float* lanczos_lut) {
  integer_t i;
  const float forder = (float)kernel_order;
  float poff;

  assert(lanczos_lut);
  assert(kernel_order < 6);

  /* Set the first value to avoid arithmetic problems */
  lanczos_lut[0] = 1.0;

  for (i = 1; i < npix; ++i) {
    poff = M_PI * (float)i * del;
    if (poff < M_PI * forder) {
      lanczos_lut[i] = sin(poff) / poff * sin(poff / forder) / (poff / forder);
    } else {
      lanczos_lut[i] = 0.0;
    }
  }
}

void
put_fill(struct driz_param_t* p, const float fill_value) {
  integer_t i, j, osize[2];

  assert(p);
  get_dimensions(p->output_data, osize);
  for (j = 0; j < osize[1]; ++j) {
    for (i = 0; i < osize[0]; ++i) {
      if (oob_pixel(p->output_counts, i, j)) {
        driz_error_format_message(p->error, "OOB in output_counts[%d,%d]", i, j);
        return;

      } else if (oob_pixel(p->output_data, i, j)) {
        driz_error_format_message(p->error, "OOB in output_data[%d,%d]", i, j);
        return;

      } else if (get_pixel(p->output_counts, i, j) == 0.0) {
        set_pixel(p->output_data, i, j, fill_value);
      }
    }
  }
}

double
mgf2(double lambda) {
  double sig, sig2;

  sig = 1.0e7 / lambda;
  sig2 = sig*sig;

  return sqrt(1.0 + 2.590355e10/(5.312993e10-sig2) +
              4.4543708e9/(11.17083e9-sig2) +
              4.0838897e5/(1.766361e5-sig2));
}
