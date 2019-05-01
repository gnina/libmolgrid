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

#include "libmolgrid/grid.h"
#include "libmolgrid/managed_grid.h"

#define TYPEARG(Z, N, T) BOOST_PP_COMMA_IF(N) T
#define NTYPES(N, T) BOOST_PP_REPEAT(N, TYPEARG, T)

extern bool python_gpu_enabled;
bool init_numpy();

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
#define MAKE_GRID_TN(N, TYPE, NAME) extern template void define_grid<libmolgrid::TYPE,NTYPES(N,unsigned)>(const char*, bool);
#define MAKE_GRID(N, CUDA, T) \
    MAKE_GRID_TN(N,Grid##N##T##CUDA, "Grid" #N #T #CUDA)

// MGrid bindings
#define MAKE_MGRID_TN(N, TYPE, NAME) extern template void define_mgrid<libmolgrid::TYPE,NTYPES(N,unsigned)>(const char *);
#define MAKE_MGRID(N, T) \
    MAKE_MGRID_TN(N,MGrid##N##T,"MGrid" #N #T)

//instantiate all dimensions up to and including six
#define MAKE_GRIDS(Z, N, _) \
    MAKE_GRID(N,CUDA,f) \
    MAKE_GRID(N,CUDA,d) \
    MAKE_GRID(N, ,f) \
    MAKE_GRID(N, ,d) \
    MAKE_MGRID(N,f) \
    MAKE_MGRID(N,d)

#define MAKE_ALL_GRIDS() BOOST_PP_REPEAT_FROM_TO(1,LIBMOLGRID_MAX_GRID_DIM, MAKE_GRIDS, 0);
MAKE_ALL_GRIDS()


#endif /* BINDINGS_H_ */
