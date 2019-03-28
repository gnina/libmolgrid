/*
 * bindings.h
 *
 * Declarations for python bindings.
 *  Created on: Mar 27, 2019
 *      Author: dkoes
 */

#ifndef BINDINGS_H_
#define BINDINGS_H_

#include <vector>
#include <type_traits>
#include <utility>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/make_constructor.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/preprocessor/repetition.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "grid.h"
#include "managed_grid.h"

#define TYPEARG(Z, N, T) BOOST_PP_COMMA_IF(N) T
#define NTYPES(N, T) BOOST_PP_REPEAT(N, TYPEARG, T)

extern bool python_gpu_enabled;

//wrapper for float* since it isn't a native python type
template <typename T>
struct Pointer {
    T *ptr;

    Pointer(T *p): ptr(p) {}

    operator T*() const { return ptr; }
};

//register definition for specified grid type
template<class GridType, typename ... Types>
void define_grid(const char* name, bool numpysupport);

template<class GridType, typename ... Types>
void define_mgrid(const char* name);

//Grid bindings - these are actually instantiated and defined in bindings_grids.cpp
#define EINSTANTIATE_GRID_TN(N, TYPE, NAME) extern template void define_grid<libmolgrid::TYPE,NTYPES(N,unsigned)>(const char*, bool);
#define EINSTANTIATE_GRID(N, CUDA, T) \
    EINSTANTIATE_GRID_TN(N,Grid##N##T##CUDA, "Grid" #N #T #CUDA)

// MGrid bindings
#define EINSTANTIATE_MGRID_TN(N, TYPE, NAME) extern template void define_mgrid<libmolgrid::TYPE,NTYPES(N,unsigned)>(const char *);
#define EINSTANTIATE_MGRID(N, T) \
    EINSTANTIATE_MGRID_TN(N,MGrid##N##T,"MGrid" #N #T)

//instantiate all dimensions up to and including six
#define EINSTANTIATE_GRIDS(Z, N, _) \
    EINSTANTIATE_GRID(N,CUDA,f) \
    EINSTANTIATE_GRID(N,CUDA,d) \
    EINSTANTIATE_GRID(N, ,f) \
    EINSTANTIATE_GRID(N, ,d) \
    EINSTANTIATE_MGRID(N,f) \
    EINSTANTIATE_MGRID(N,d)

BOOST_PP_REPEAT_FROM_TO(1,LIBMOLGRID_MAX_GRID_DIM, EINSTANTIATE_GRIDS, 0);


#endif /* BINDINGS_H_ */
