#ifndef CDRIZZLEBLOT_H
#define CDRIZZLEBLOT_H

#include "cdrizzleutil.h"

/**
 * This routine does the interpolation of the input array.
 *
 * @param[in,out] p A set of blotting parameters.
 * @return Non-zero if an error occurred.
 */

int
doblot(struct driz_param_t* p);

#endif /* CDRIZZLEBLOT_H */
