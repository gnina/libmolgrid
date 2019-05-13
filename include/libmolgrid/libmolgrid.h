/** \file libmolgrid.h
 *  \brief Single header to include for all libmolgrid functionality.
 *  libmolgrid provides GPU accelerated routines for converting molecular
 *  data into dense grids for use in deep learning workflows.
 */
 
#ifndef LIBMOLGRID_H_
#define LIBMOLGRID_H_

#include <random>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include <cuda_runtime.h>

// dimensionalities up to but not including LIBMOLGRID_MAX_GRID_DIM are pre-instantiated
#define LIBMOLGRID_MAX_GRID_DIM 9
namespace libmolgrid {
    ///random engine used in libmolgrid
    extern std::default_random_engine random_engine;

    using cuda_float3 = ::float3; //in case "someone" has redefined float3
    enum LogLevel { INFO, WARNING, ERROR, DEBUG};

    inline std::ostream& log(LogLevel level = INFO) {
      if(level > INFO)
        return std::cerr;
      else
        return std::cout; //todo, implement verbosity levels
    }

    inline std::string ftoa(float v) { return boost::lexical_cast<std::string>(v); }
    inline std::string itoa(int v) { return boost::lexical_cast<std::string>(v); }
}

#endif
