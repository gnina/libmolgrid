/** \file libmolgrid.h
 *  \brief Single header to include for all libmolgrid functionality.
 *  libmolgrid provides GPU accelerated routines for converting molecular
 *  data into dense grids for use in deep learning workflows.
 */
 
#ifndef LIBMOLGRID_H_
#define LIBMOLGRID_H_

#include <random>

namespace libmolgrid {
    ///random engine used in libmolgrid
    extern std::default_random_engine random_engine;

}

#endif
