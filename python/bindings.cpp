/*
 * bindings.cpp
 *
 *  Python bindings for libmolgrid
 */

#include <boost/python.hpp>
#include <boost/preprocessor/repetition.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include "grid.h"
#include "managed_grid.h"
#include "quaternion.h"
#include "transform.h"


BOOST_PYTHON_MODULE(molgrid)
{
    Py_Initialize();

    using namespace boost::python;
    using namespace libmolgrid;
    // Grids

#define TYPEARG(Z, N, T) BOOST_PP_COMMA_IF(N) T
#define NTYPES(N, T) BOOST_PP_REPEAT(N, TYPEARG, T)

//Grid bindings
#define DEFINE_GRID_TN(N, TYP, TYPE, NAME) \
    class_<TYPE>(NAME, init<TYP*, NTYPES(N,unsigned)>()) \
      .def("size",&TYPE::size) \
      .def("dimension",&TYPE::dimension) \
      .def("offset",&TYPE::offset);

#define DEFINE_GRID(N, CUDA, T,TYP) \
  DEFINE_GRID_TN(N,TYP,Grid##N##T##CUDA, "Grid" #N #T #CUDA)


// MGrid bindings
#define DEFINE_MGRID_TN(N, TYPE, NAME) \
    class_<M##TYPE, bases< TYPE##CUDA > >(NAME, init<NTYPES(N,unsigned)>());


#define DEFINE_MGRID(N, T) \
    DEFINE_MGRID_TN(N,Grid##N##T,"MGrid" #N #T)



    //instantiate all dimensions up to and including six
#define DEFINE_GRIDS(Z, N, _) \
    DEFINE_GRID(N,CUDA,f,float) \
    DEFINE_GRID(N,CUDA,d,double) \
    DEFINE_GRID(N, ,f,float) \
    DEFINE_GRID(N, ,d,double) \
    DEFINE_MGRID(N,f) \
    DEFINE_MGRID(N,d)

    BOOST_PP_REPEAT_FROM_TO(1,7, DEFINE_GRIDS, 0);


}



