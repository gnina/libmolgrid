/*
 * bindings.cpp
 *
 *  Python bindings for libmolgrid
 */

#include <vector>
#include <type_traits>
#include <utility>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/tuple.hpp>
#include <boost/preprocessor/repetition.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include "grid.h"
#include "managed_grid.h"
#include "quaternion.h"
#include "transform.h"

using namespace boost::python;
using namespace libmolgrid;

#define TYPEARG(Z, N, T) BOOST_PP_COMMA_IF(N) T
#define NTYPES(N, T) BOOST_PP_REPEAT(N, TYPEARG, T)

/// indicate how MGrid shoudl be automatically converted in python bindings
static bool python_gpu_enabled = true;

//given a python object, convert to appropriate grid type if possible
template<class Grid_t>
struct Grid_from_python {

    Grid_from_python() {
      //register on construction
      converter::registry::push_back(
          &Grid_from_python::convertible,
          &Grid_from_python::construct,
          type_id<Grid_t>());
    }

    static void* convertible(PyObject *obj_ptr) {
      extract<typename Grid_t::managed_t> mgrid(obj_ptr);
      if (mgrid.check() && Grid_t::GPU == python_gpu_enabled) {
        return obj_ptr;
      }
      return nullptr;
    }

    static void construct(PyObject* obj_ptr,
        boost::python::converter::rvalue_from_python_stage1_data* data) {

      extract<typename Grid_t::managed_t> mgrid(obj_ptr);
      if (mgrid.check()) {
        // Obtain a handle to the memory block that the converter has allocated
        // for the C++ type.
        Grid_t g = mgrid();
        typedef converter::rvalue_from_python_storage<Grid_t> storage_type;
        void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;
        data->convertible = new (storage) Grid_t(g);
      }
    }
};


template<typename GridType,
    typename std::enable_if<
        std::is_same<typename GridType::type, typename GridType::subgrid_t>::value,
        int>::type = 0>
void add_one_dim(class_<GridType>& C) {
  C.def("__setitem__",
      +[](GridType& g, size_t i, typename GridType::type val) {g[i] = val;});
}

template<typename GridType,
    typename std::enable_if<
        !std::is_same<typename GridType::type, typename GridType::subgrid_t>::value,
        int>::type = 0>
void add_one_dim(class_<GridType>& C) {
  //not one-dimensional grid type, do nothing
}

template<typename GridType, std::size_t ... I>
typename GridType::type& grid_get(GridType& g, tuple t,
    std::index_sequence<I...>) {
  return g(static_cast<size_t>(extract<size_t>(t[I]))...);
}

//add common grid methods
template<typename GridType>
void add_grid_members(class_<GridType>& C) {
  C.def("size", &GridType::size)
      .def("dimension", &GridType::dimension)
      .add_property("shape",
          make_function(
          +[](const GridType& g)->tuple {
            return tuple(std::vector<size_t>(g.dimensions(),g.dimensions()+GridType::N)); //hopefully tuple gets move constructed
          }))
      .def("__len__",
          +[](const GridType& g)->size_t {return g.dimension(0);}) //length of first dimension only
      .def("__getitem__",
          +[](const GridType& g, size_t i)-> typename GridType::subgrid_t {return g[i];})
      .def("__getitem__",
          +[](GridType& g, tuple t) -> typename GridType::type {return grid_get(g, t, std::make_index_sequence<GridType::N>());})
      .def("__setitem__",
          +[](GridType& g, tuple t, typename GridType::type val) {grid_get(g, t, std::make_index_sequence<GridType::N>()) = val;});
}

//register definition for specified grid type
template<class GridType, typename ... Types>
void define_grid(const char* name) {

  class_<GridType> C(name, init<typename GridType::type*, Types...>());
  add_grid_members(C);
  //setters only for one dimension grids
  add_one_dim(C); //SFINAE!

  Grid_from_python<GridType> convert; //register
}

template<class GridType, typename ... Types>
void define_mgrid(const char* name) {

  class_<GridType> C(name, init<Types...>());
  add_grid_members(C);
  //setters only for one dimension grids
  add_one_dim(C); //SFINAE!
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Transform_forward_overloads, Transform::forward, 2, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(Transform_backward_overloads, Transform::backward, 2, 3)


BOOST_PYTHON_MODULE(molgrid)
{
  Py_Initialize();

  def("set_random_seed", +[](long s) {random_engine.seed(s);}); //set random seed
  def("get_gpu_enabled", +[]()->bool {return python_gpu_enabled;},
      "Get if generated grids are on GPU by default.");
  def("set_gpu_enabled", +[](bool val) {python_gpu_enabled = val;},
      "Set if generated grids should be on GPU by default.");

// Grids

//Grid bindings
#define DEFINE_GRID_TN(N, TYPE, NAME) define_grid<TYPE,NTYPES(N,unsigned)>(NAME);
#define DEFINE_GRID(N, CUDA, T) \
DEFINE_GRID_TN(N,Grid##N##T##CUDA, "Grid" #N #T #CUDA)

// MGrid bindings
#define DEFINE_MGRID_TN(N, TYPE, NAME) define_mgrid<TYPE,NTYPES(N,unsigned)>(NAME);
#define DEFINE_MGRID(N, T) \
DEFINE_MGRID_TN(N,MGrid##N##T,"MGrid" #N #T)

//instantiate all dimensions up to and including six
#define DEFINE_GRIDS(Z, N, _) \
DEFINE_GRID(N,CUDA,f) \
DEFINE_GRID(N,CUDA,d) \
DEFINE_GRID(N, ,f) \
DEFINE_GRID(N, ,d) \
DEFINE_MGRID(N,f) \
DEFINE_MGRID(N,d)

  BOOST_PP_REPEAT_FROM_TO(1,7, DEFINE_GRIDS, 0);

//vector utility types
  class_<std::vector<size_t> >("SizeVec")
      .def(vector_indexing_suite<std::vector<size_t> >());

  class_<float3>("float3", no_init)
      .def("__init__",
      make_constructor(
          +[](float x, float y, float z) {return std::make_shared<float3>(make_float3(x,y,z));}))
      .def_readwrite("x", &float3::x)
      .def_readwrite("y", &float3::y)
      .def_readwrite("z", &float3::z);

// Quaternion - I lament the need for a custom quaternion class, yet here we are
  class_<Quaternion>("Quaternion")
      .def(init<float, float, float, float>())
      .def("R_component_1", &Quaternion::R_component_1)
      .def("R_component_2", &Quaternion::R_component_2)
      .def("R_component_3", &Quaternion::R_component_3)
      .def("R_component_4", &Quaternion::R_component_4)
      .def("real", &Quaternion::real)
      .def("conj", &Quaternion::conj)
      .def("norm", &Quaternion::norm)
      .def("rotate", &Quaternion::rotate)
      .def("transform", &Quaternion::transform)
      .def("inverse", &Quaternion::inverse)
      .def(self * self)
      .def(self *= self)
      .def(self / self)
      .def(self /= self);

// Transform

  class_<Transform>("Transform")
      .def(init<Quaternion>())
      .def(init<Quaternion, float3>())
      .def(init<Quaternion, float3, float3>())
      .def(init<float3>()) //center
  .def(init<float3, float>()) //center, translate
  .def(init<float3, float, bool>()) //center, translate, rotate
  .def("quaternion", &Transform::quaternion, return_value_policy<copy_const_reference>())
  .def("rotation_center", &Transform::rotation_center)
  .def("translation", &Transform::translation)
  //non-const references need to be passed by value, so wrap
  .def("forward", +[](Transform& self, const Grid2f& in, Grid2f out, bool dotranslate) {self.forward(in,out,dotranslate);},
      Transform_forward_overloads((arg("in"), arg("out"), arg("dotranslate")=true)))
  .def("forward",  +[](Transform& self, const Grid2fCUDA& in, Grid2fCUDA out, bool dotranslate) {self.forward(in,out,dotranslate);},
      Transform_forward_overloads((arg("in"), arg("out"), arg("dotranslate")=true)))
  .def("backward", +[](Transform& self, const Grid2f& in, Grid2f out, bool dotranslate) {self.backward(in,out,dotranslate);},
      Transform_backward_overloads((arg("in"), arg("out"), arg("dotranslate")=true)))
  .def("backward",+[](Transform& self, const Grid2fCUDA& in, Grid2fCUDA out, bool dotranslate) {self.backward(in,out,dotranslate);},
       Transform_backward_overloads((arg("in"), arg("out"), arg("dotranslate")=true)));

    }

